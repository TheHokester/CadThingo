using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.RenderGraph;

/// <summary>
/// Workflow<br/>
/// 1. AddResource() - declare all images the graph will use <br/>
/// 2. AddPass() - declare all passes and there read/write sets<br/>
/// 3. Compile() - topological sort + allocate gpu resources<br/>
/// 4. Execute() - record all passes into the command buffer in order<br/>
/// 5. Dispose() - free all gpu resources <br/>
/// </summary>
public unsafe class RenderGraph : IDisposable
{
    private readonly Vk _vk;
    private readonly Device _device;
    private readonly PhysicalDevice _physicalDevice;
    private bool _disposed;
    private bool _compiled;

    private readonly Dictionary<string, ImageResource> _imageResources = new();
    private List<Pass> passes;
    private List<int> executionOrder;

    // Tracks each resource's current Vulkan layout across passes within a frame and
    // across frames. Source layout for every transition barrier comes from this dict
    // so blends / post-process passes that consume previous-pass output read the
    // correct contents rather than racing an Undefined→X discard.
    private readonly Dictionary<string, ImageLayout> _currentLayout = new();

    public RenderGraph(Vk vk, Device device, PhysicalDevice physicalDevice)
    {
        _vk = vk;
        _device = device;
        _physicalDevice = physicalDevice;
        passes = new();
        executionOrder = new();
    }


    /// <summary>
    /// Creates a new image resource and adds it to the render graph
    /// </summary>
    /// <param name="name"></param>
    /// <param name="format"></param>
    /// <param name="extent"></param>
    /// <param name="usage"></param>
    /// <param name="initialLayout"></param>
    /// <param name="finalLayout"></param>
    public void AddResource(string name, Format format, Extent2D extent, ImageUsageFlags usage,
        ImageLayout initialLayout = ImageLayout.Undefined,
        ImageLayout finalLayout = ImageLayout.ShaderReadOnlyOptimal)
    {
        ImageResource resource = new(_vk, _device, name, format, extent, usage, initialLayout, finalLayout);
        AddResource(resource);
    }

    /// <summary>
    /// Add a resource to the render graph
    /// </summary>
    /// <param name="resource"></param>
    public void AddResource(ImageResource resource)
    {
        _imageResources[resource._name] = resource;
    }

    public ImageResource GetResource(string name)
    {
        if (!_imageResources.TryGetValue(name, out var resource))
        {
            throw new Exception($"Resource {name} not found");
        }

        return resource;
    }

    /// <summary>
    /// Add a pass to the render graph
    /// </summary>
    /// <param name="pass"></param>
    public void AddPass(Pass pass) => passes.Add(pass);

    /// <summary>
    /// Creates a new pass and adds it to the render graph
    /// </summary>
    /// <param name="name"></param>
    /// <param name="inputs"></param>
    /// <param name="outputs"></param>
    /// <param name="executeFunc"></param>
    public void AddPass(string name, List<string> inputs, List<string> outputs, Action<CommandBuffer, Renderer.FrameContext> executeFunc)
    {
        Pass pass = new(name, inputs, outputs)
        {
            ExecuteFunc = executeFunc
        };
        AddPass(pass);
    }

    /// <summary>
    /// Rendergraph compilation - Transforms declarative descriptions into executable pipelines<br/>
    /// This method performs dependency analysis, resource allocation, and execution planning<br/>
    /// </summary>
    public void Compile()
    {
        if (_compiled)
        {
            throw new InvalidOperationException("RenderGraph already compiled");
        }

        int n = passes.Count;

        // adjacency[i] = list of pass indices that depend on pass i
        var adjacency = new List<int>[n];
        var inDegree = new int[n];
        for (int i = 0; i < n; i++) adjacency[i] = new List<int>();

        // Build edges: for each resource, find (writer → reader) pairs.
        // Restricted to j > i: a pass only reads from writers declared before it.
        // Without this restriction, a later-declared writer of the same resource
        // (e.g. TransparentPass writing Depth that LightingPass also reads) creates
        // a back-edge into an earlier pass, and combined with writer→writer edges
        // for other shared resources produces a cycle that strands the topo sort.
        for (var i = 0; i < n; i++)
        {
            foreach (var output in passes[i]._outputs)
            {
                for (int j = i + 1; j < n; j++)
                {
                    if (passes[j]._inputs.Contains(output))
                    {
                        adjacency[i].Add(j);
                        inDegree[j]++;
                    }
                }
            }
        }

        // Writer → writer edges: when two passes both write the same resource,
        // serialize them in declaration order (earlier-declared runs first).
        // Without this the topo sort would put them in arbitrary order and
        // blend / accumulation chains (LightingPass → TransparentPass into
        // HDRColor) would break non-deterministically.
        for (int i = 0; i < n; i++)
        {
            foreach (var output in passes[i]._outputs)
            {
                for (int j = i + 1; j < n; j++)
                {
                    if (passes[j]._outputs.Contains(output) && !adjacency[i].Contains(j))
                    {
                        adjacency[i].Add(j);
                        inDegree[j]++;
                    }
                }
            }
        }

        //Topological sort for optimal execution order
        //this is done by Kahn's algorithm
        //BFS from all zero-in-degree nodes
        var queue = new Queue<int>();
        for (var i = 0; i < n; i++)
            if (inDegree[i] == 0)
                queue.Enqueue(i);

        while (queue.Count > 0)
        {
            int node = queue.Dequeue();
            executionOrder.Add(node);
            foreach (var adj in adjacency[node])
            {
                if (--inDegree[adj] == 0)
                    queue.Enqueue(adj);
            }
        }

        if (executionOrder.Count == 0)
        {
            throw new Exception("RenderGraph contains a cycle, check pass dependencies");
        }

        // Single command buffer / single queue: pipeline barriers between passes (in Execute)
        // handle ordering — no semaphores needed inside the graph.

        foreach (var resource in _imageResources.Values)
            resource.Allocate(_physicalDevice);

        // Seed the tracker. Resources start in their declared _initialLayout
        // (typically Undefined for fresh allocations); subsequent transitions
        // update the tracker so each pass sources from the actual prior layout.
        foreach (var kv in _imageResources)
        {
            _currentLayout[kv.Key] = kv.Value._initialLayout;
            
        }
        _compiled = true;
    }

    /// <summary>
    /// Records all passes (input barriers → pass callback → output barriers → final-layout barriers)
    /// into the provided command buffer. Caller owns Begin/End/Submit and the
    /// imageAvailable/renderFinished semaphores.
    /// </summary>
    public void Execute(CommandBuffer cmd, Renderer.FrameContext frameContext)
    {
        if (!_compiled)
            throw new InvalidOperationException("Call Compile() before Execute().");

        foreach (var passIndex in executionOrder)
        {
            var pass = passes[passIndex];
            
            // ----- Barrier transition inputs -> ShaderReadOnlyOptimal
            foreach (var inputName in pass._inputs)
            {
                ImageResource resource = _imageResources[inputName];
                bool isDepth =
                    resource._format is Format.D32Sfloat or Format.D24UnormS8Uint or Format.D16UnormS8Uint;
                if (_imageResources[inputName].IsAllocated)
                {
                    var barrier = new ImageMemoryBarrier()
                    {
                        SType = StructureType.ImageMemoryBarrier,
                        OldLayout = _currentLayout[inputName],
                        NewLayout = ImageLayout.ShaderReadOnlyOptimal,
                        SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
                        DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
                        Image = resource.Image,
                        SrcAccessMask = AccessFlags.MemoryReadBit,
                        DstAccessMask = AccessFlags.ShaderReadBit,
                        SubresourceRange = new ImageSubresourceRange()
                        {
                            AspectMask = isDepth ? ImageAspectFlags.DepthBit : ImageAspectFlags.ColorBit,
                            BaseMipLevel = 0,
                            LevelCount = 1,
                            BaseArrayLayer = 0,
                            LayerCount = 1
                        }
                    };

                    _vk.CmdPipelineBarrier(cmd,
                        PipelineStageFlags.AllCommandsBit,
                        PipelineStageFlags.FragmentShaderBit,
                        DependencyFlags.ByRegionBit,
                        0, null,
                        0, null,
                        1, ref barrier);

                    _currentLayout[inputName] = ImageLayout.ShaderReadOnlyOptimal;
                }
            }

            // ----- Barrier transition outputs -> ColorAttachmentOptimal / DepthStencilAttachmentOptimal
            foreach (var outputName in pass._outputs)
            {
                if (_imageResources[outputName].IsAllocated)
                {
                    ImageResource resource = _imageResources[outputName];
                    bool isDepth =
                        resource._format is Format.D32Sfloat or Format.D24UnormS8Uint or Format.D16UnormS8Uint;
                    var targetAttachmentLayout = isDepth
                        ? ImageLayout.DepthStencilAttachmentOptimal
                        : ImageLayout.ColorAttachmentOptimal;
                    var barrier = new ImageMemoryBarrier()
                    {
                        SType = StructureType.ImageMemoryBarrier,
                        OldLayout = _currentLayout[outputName],
                        NewLayout = targetAttachmentLayout,
                        SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
                        DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
                        Image = resource.Image,
                        SrcAccessMask = AccessFlags.MemoryReadBit,
                        DstAccessMask = isDepth
                            ? AccessFlags.DepthStencilAttachmentWriteBit
                            : AccessFlags.ColorAttachmentWriteBit,
                        SubresourceRange = new ImageSubresourceRange()
                        {
                            AspectMask = isDepth ? ImageAspectFlags.DepthBit : ImageAspectFlags.ColorBit,
                            BaseMipLevel = 0,
                            LevelCount = 1,
                            BaseArrayLayer = 0,
                            LayerCount = 1
                        }
                    };

                    _vk.CmdPipelineBarrier(cmd,
                        PipelineStageFlags.AllCommandsBit,
                        isDepth
                            ? PipelineStageFlags.EarlyFragmentTestsBit
                            : PipelineStageFlags.ColorAttachmentOutputBit,
                        DependencyFlags.ByRegionBit,
                        0, null,
                        0, null,
                        1, ref barrier);

                    _currentLayout[outputName] = targetAttachmentLayout;
                }
            }

            //----- Execute pass-------------------
            pass.ExecuteFunc(cmd, frameContext);

            //----- Barrier: transition outputs -> resource._finalLayout
            foreach (var outputName in pass._outputs)
            {
                if (!_imageResources[outputName].IsAllocated) continue;
                ImageResource resource = _imageResources[outputName];
                bool isDepth =
                    resource._format is Format.D32Sfloat or Format.D24UnormS8Uint or Format.D16UnormS8Uint;
                var barrier = new ImageMemoryBarrier
                {
                    SType = StructureType.ImageMemoryBarrier,
                    OldLayout = _currentLayout[outputName],
                    NewLayout = resource._finalLayout,
                    SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
                    DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
                    Image = resource.Image,
                    SrcAccessMask = isDepth
                        ? AccessFlags.DepthStencilAttachmentWriteBit
                        : AccessFlags.ColorAttachmentWriteBit,
                    DstAccessMask = AccessFlags.MemoryReadBit,
                    SubresourceRange = new ImageSubresourceRange()
                    {
                        AspectMask = isDepth ? ImageAspectFlags.DepthBit : ImageAspectFlags.ColorBit,
                        BaseMipLevel = 0,
                        LevelCount = 1,
                        BaseArrayLayer = 0,
                        LayerCount = 1
                    }
                };
                
                _vk.CmdPipelineBarrier(
                    cmd,
                    isDepth
                        ? PipelineStageFlags.LateFragmentTestsBit
                        : PipelineStageFlags.ColorAttachmentOutputBit,
                    PipelineStageFlags.AllCommandsBit, // before any subsequent work
                    DependencyFlags.ByRegionBit,
                    0, null, 0, null, 1, ref barrier);

                _currentLayout[outputName] = resource._finalLayout;
            }
        }
    }


//IDisposable ------------------
    public void Dispose()
    {
        if (_disposed) return;

        foreach (var resource in _imageResources.Values)
            resource.Dispose();
        _imageResources.Clear();

        _disposed = true;
        GC.SuppressFinalize(this);
    }
}