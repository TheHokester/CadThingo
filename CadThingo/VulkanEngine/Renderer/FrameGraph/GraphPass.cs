using CadThingo.VulkanEngine.Renderer.Shaders;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.FrameGraph;

public enum PassType { Graphics, Compute, RayTrace, Transfer}

public delegate void PassSetup(GraphBuilder b);
public delegate void PassExecute(CommandBuffer cmd, PassResources resources, in Renderer.FrameContext frame);

/// <summary>
/// What a pipeline hands the graph so its pass-local descriptor set can be graph-baked.
///The graph allocates one set per frame-in-flight from
/// <paramref name="Layout"/> and fills each binding whose <see cref="BindingDesc.Name"/>
/// a Read/Write named. The layout is reflection-built and resize-stable, so the pipeline
/// owns it (and its VkPipelineLayout borrows it at <paramref name="SetIndex"/>); the graph
/// only borrows it to allocate + write the sets.
/// </summary>
public readonly record struct PassSetSpec(
    uint SetIndex,
    DescriptorSetLayout Layout,
    IReadOnlyList<BindingDesc> Bindings);

internal struct ResourceAccess
{
    public int ResourceId;
    public int Version;
    public ResourceUsage Usage;
    public bool IsWrite;
    public bool IsImage;
    // Set-N parameter name this resource fills in the pass's graph-baked descriptor set,
    // or null for a sync-only access (the pass binds it itself, or it is an attachment /
    // indirect-arg / scene-set resource). Matched against PassSet.Bindings by name.
    public string? Bind;
}
    
internal sealed class GraphPass
{
    public string Name;
    // Module scope prefix this pass was authored under ("" at top level, else e.g.
    // "Deferred" or "Deferred/Sub"). Set by AddPass from the GraphScope; used by ToDot to
    // emit nested module clusters and to flag cross-module edges. Name is the fully
    // qualified "scope/leaf"; Scope is just the prefix.
    public string Scope = "";
    public PassType Type;
    public QueueClass Queue; // phase 1 always graphics
    public bool PreferAsync; //phase 3 hint
    public bool HasSideEffects;// keep through dead-code cull even with no consumer
    public PassExecute Execute;

    public List<ResourceAccess> Reads = new();
    public List<ResourceAccess> Writes = new();

    public List<int> ColorTargets = new();
    public int DepthTarget = -1;

    // Baked at Compile (steps 6/7), replayed verbatim by Execute. One CmdPipelineBarrier2
    // per pass covering all of its input/output transitions — empty when nothing needs
    // syncing before this pass.
    public ImageMemoryBarrier2[]  ImageBarriers  = [];
    public BufferMemoryBarrier2[] BufferBarriers = [];
    // Parallel to BufferBarriers: the resource id each barrier targets, so Execute can
    // patch the .Buffer field per frame for double-buffered (per-frame) imports.
    public int[] BufferBarrierRes = [];

    // Graph-baked pass descriptor set. PassSet is the pipeline-supplied spec (null = the
    // pass binds its own sets, legacy behavior); PassSets holds the graph-allocated set per
    // frame-in-flight, written at Compile and handed to Execute via PassResources.
    public PassSetSpec?    PassSet;
    public DescriptorSet[] PassSets = [];
}

public readonly struct PassResources
{
    private readonly FrameGraph _g;
    private readonly DescriptorSet _passSet;
    internal PassResources(FrameGraph g, DescriptorSet passSet) { _g = g; _passSet = passSet; }
    public ImageView View(GraphImage h) => _g.ResolveView(h);
    public Image Image(GraphImage h) => _g.ResolveImage(h);
    public Buffer Buffer(GraphBuffer h) => _g.ResolveBuffer(h);

    /// <summary>This pass's graph-baked descriptor set for the frame being recorded, or the
    /// default handle if the pass did not call <see cref="GraphBuilder.UsePassSet"/>. The pass
    /// body binds it via its pipeline layout at the spec's set index.</summary>
    public DescriptorSet PassSet => _passSet;
}