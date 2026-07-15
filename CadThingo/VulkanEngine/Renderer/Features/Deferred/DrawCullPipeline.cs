using System.Numerics;
using System.Runtime.InteropServices;
using CadThingo.VulkanEngine.Renderer.FrameGraph;
using CadThingo.VulkanEngine.Renderer.Pipelines;
using CadThingo.VulkanEngine.Renderer.Shaders;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Features.Deferred;
//  Draw-cull compute pass — frustum-tests scene renderables and emits
//  VkDrawIndexedIndirectCommand[] consumed by the geometry pass.
public sealed unsafe class DrawCullPipeline : ComputePipeline
{
    // Push constants pushed at every dispatch. 100 bytes — within the
    // 128B Vulkan minimum (maxPushConstantsSize).
    [StructLayout(LayoutKind.Sequential)]
    private struct CullPushConstants
    {
        public Vector4 PlaneL;
        public Vector4 PlaneR;
        public Vector4 PlaneB;
        public Vector4 PlaneT;
        public Vector4 PlaneN;
        public Vector4 PlaneF;
        public uint    RenderableCount;
        public uint    _pad0;
        public uint    _pad1;
        public uint    _pad2;
    }

    protected override string ShaderPath { get; } = ShaderPaths.Kernel("Deferred", "CullDraws");

    // Per-frame buffers owned by this pipeline. .
    private UboBuffer[] IndirectCmdBuffers     = new UboBuffer[Renderer.MAX_CONCURRENT_FRAMES];
    private UboBuffer[] IndirectCountBuffers   = new UboBuffer[Renderer.MAX_CONCURRENT_FRAMES];

    public Buffer GetIndirectCmdBuffer  (uint frame) => IndirectCmdBuffers[frame].buffer;
    public Buffer GetIndirectCountBuffer(uint frame) => IndirectCountBuffers[frame].buffer;
    public Buffer GetRenderablesBuffer  (uint frame) => Renderer.gpuScene.GetRenderablesBuffer(frame);

    // Pass-set contract for the deferred FrameGraph (descriptor-system.md phase C). The four
    // storage buffers this compute shader binds are all graph resources -- the input renderable
    // list plus the three post-cull outputs the graph imports -- so the graph allocates + writes
    // the set; the pipeline just owns the layout (its VkPipelineLayout borrows it at set 0) and
    // binds whatever set the graph hands Record. Names match the CullPass Read/Write binds, not
    // the .slang identifiers (bindings map by (set,binding) number).
    private static readonly BindingDesc[] _passBindings =
    {
        new("renderables",   0, 0, DescriptorType.StorageBuffer, 1, ShaderStageFlags.ComputeBit),
        new("indirectCmd",   0, 1, DescriptorType.StorageBuffer, 1, ShaderStageFlags.ComputeBit),
        new("instanceData",  0, 2, DescriptorType.StorageBuffer, 1, ShaderStageFlags.ComputeBit),
        new("indirectCount", 0, 3, DescriptorType.StorageBuffer, 1, ShaderStageFlags.ComputeBit),
    };

    public PassSetSpec PassSet => new(SetIndex: 0, DescriptorSetLayouts[0], _passBindings);

    /// <summary>Renderables packed in the most recent Record() call — drives
    /// maxDrawCount on vkCmdDrawIndexedIndirectCount.</summary>
    public uint LastRenderableCount { get; private set; }

    // BLEND-mode entities partitioned out of the cull input during Record.
    // Sorted back-to-front by view-space depth so the transparent pass renders
    // far-first.
    private readonly List<TransparentDraw> _transparentDraws = new();

    /// <summary>BLEND-mode draws captured this frame, sorted back-to-front by view-space Z.
    /// Consumed by the TransparentPass; empty when no scene material is BLEND-mode.</summary>
    public IReadOnlyList<TransparentDraw> LastTransparentDraws => _transparentDraws;

    public DrawCullPipeline(GpuContext gpu, Renderer renderer) : base(gpu, renderer)
    {
        PushConstantRanges = new[]
        {
            new PushConstantRange
            {
                StageFlags = ShaderStageFlags.ComputeBit,
                Offset     = 0,
                Size       = (uint)sizeof(CullPushConstants),
            }
        };
    }

    public override void Dispose()
    {
        foreach (var b in IndirectCmdBuffers)     Gfx.DestroyBuffer(b.buffer, b.alloc);
        foreach (var b in IndirectCountBuffers)   Gfx.DestroyBuffer(b.buffer, b.alloc);
        base.Dispose();
    }

    protected override void CreateDescriptorSetLayouts()
    {
        // The pass-set layout (set 0): 4 storage buffers, borrowed by the deferred FrameGraph
        // to allocate + write the descriptor set. The pipeline owns the layout; the graph owns
        // the sets. Exposed to the graph via PassSet.
        var bindings = stackalloc DescriptorSetLayoutBinding[4];
        for (uint b = 0; b < 4; b++)
        {
            bindings[b] = new DescriptorSetLayoutBinding
            {
                Binding         = b,
                DescriptorType  = DescriptorType.StorageBuffer,
                DescriptorCount = 1,
                StageFlags      = ShaderStageFlags.ComputeBit,
                PImmutableSamplers = null,
            };
        }
        DescriptorSetLayoutCreateInfo info = new()
        {
            SType        = StructureType.DescriptorSetLayoutCreateInfo,
            BindingCount = 4,
            PBindings    = bindings,
        };
        if (Vk.CreateDescriptorSetLayout(Device, &info, null, out var layout) != Result.Success)
            throw new Exception("Failed to create cull descriptor set layout");
        DescriptorSetLayouts = new[] { layout };
        OwnedDescriptorSetLayoutIndices = new[] { 0 };
    }

    protected override void CreateResources()
    {
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            // Indirect-command buffer also needs IndirectBuffer usage so the
            // vkCmdDraw...IndirectCount call can read it without validation errors.
            Gfx.CreateMappedStorageBuffer(
                (ulong)(Renderer.MAX_INSTANCES * (uint)sizeof(DrawIndexedIndirectCommandGpu)),
                ref IndirectCmdBuffers[i],
                BufferUsageFlags.IndirectBufferBit);

            // Count buffer is one uint. Needs IndirectBuffer for the count read and
            // TransferDst so vkCmdFillBuffer can reset it to 0 every frame.
            Gfx.CreateMappedStorageBuffer(
                sizeof(uint),
                ref IndirectCountBuffers[i],
                BufferUsageFlags.IndirectBufferBit | BufferUsageFlags.TransferDstBit);
        }
    }

    // Descriptor sets are graph-owned now: the deferred FrameGraph allocates them from this
    // pipeline's pass-set layout and writes the four storage buffers (renderables in +
    // cmds/instances/count out) by name each Compile. No CreateDescriptorSets / WriteDescriptors
    // here -- the pipeline binds whatever set Record is handed.

    /// <summary>CPU side of the cull pass.
    ///apply the view-dependent back-to-front sort
    /// to the BLEND candidates, read the opaque count, and record the frustum-cull
    /// dispatch + barriers.</summary> 
    /// <returns>The opaque count as a uint</returns>
    public uint Record(CommandBuffer cmd, uint frameIndex, Camera cam, DescriptorSet passSet)
    {
        // View-dependent transparent sort
        _transparentDraws.Clear();
        Matrix4x4 viewMat = cam != null ? cam.GetViewMatrix() : Matrix4x4.Identity;
        foreach (var c in Renderer.gpuScene.TransparentCandidates)
        {
            var worldOrigin = new Vector4(c.Model.M41, c.Model.M42, c.Model.M43, 1f);
            float viewZ = Vector4.Transform(worldOrigin, viewMat).Z;
            var d = c;
            d.ViewDepth = viewZ;
            _transparentDraws.Add(d);
        }
        _transparentDraws.Sort((a, b) => a.ViewDepth.CompareTo(b.ViewDepth));

        uint count = Renderer.gpuScene.ExtractedRenderableCount;
        LastRenderableCount = count;
        if (count == 0) return 0;

        // 1. Reset the count buffer to 0 via vkCmdFillBuffer
        Vk.CmdFillBuffer(cmd, IndirectCountBuffers[frameIndex].buffer, 0, sizeof(uint), 0);

        //  2. Barrier: transfer write -> compute shader access on count buffer 
        var fillBarrier = new BufferMemoryBarrier
        {
            SType         = StructureType.BufferMemoryBarrier,
            SrcAccessMask = AccessFlags.TransferWriteBit,
            DstAccessMask = AccessFlags.ShaderReadBit | AccessFlags.ShaderWriteBit,
            SrcQueueFamilyIndex = Silk.NET.Vulkan.Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Silk.NET.Vulkan.Vk.QueueFamilyIgnored,
            Buffer = IndirectCountBuffers[frameIndex].buffer,
            Offset = 0,
            Size   = sizeof(uint),
        };
        Vk.CmdPipelineBarrier(cmd,
            PipelineStageFlags.TransferBit,
            PipelineStageFlags.ComputeShaderBit,
            0, 0, null, 1, &fillBarrier, 0, null);

        Vk.CmdBindPipeline(cmd, PipelineBindPoint.Compute, PipelineHandle);
        // Set 0 is the graph-baked pass set (the four cull storage buffers).
        Vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute,
            PipelineLayoutHandle, 0, 1, &passSet, 0, null);

        // Build frustum from the camera's view*proj. Deliberately use a non-Y-flipped
        // projection here: the visible volume is the same in both conventions, and
        // Frustum.FromViewProjection assumes standard row-major Vulkan NDC.
        Matrix4x4 view = cam.GetViewMatrix();
        Matrix4x4 proj = cam.GetProjectionMatrix(
            (float)Renderer.renderExtent.Width / Renderer.renderExtent.Height, 0.1f, 100.0f);
        Matrix4x4 vp   = view * proj;
        var frustum    = Frustum.FromViewProjection(vp, vulkanNDC: true);

        var push = new CullPushConstants
        {
            PlaneL = frustum.PlaneLeft.Data,
            PlaneR = frustum.PlaneRight.Data,
            PlaneB = frustum.PlaneBottom.Data,
            PlaneT = frustum.PlaneTop.Data,
            PlaneN = frustum.PlaneNear.Data,
            PlaneF = frustum.PlaneFar.Data,
            RenderableCount = count,
        };
        Vk.CmdPushConstants(cmd, PipelineLayoutHandle, ShaderStageFlags.ComputeBit,
            0, (uint)sizeof(CullPushConstants), &push);

        // 64 threads per group; ceil-divide so the last group covers the tail.
        uint groups = (count + 63u) / 64u;
        Vk.CmdDispatch(cmd, groups, 1, 1);

        
        return count;
    }
}