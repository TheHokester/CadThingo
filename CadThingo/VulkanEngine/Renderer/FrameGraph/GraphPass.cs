using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.FrameGraph;

public enum PassType { Graphics, Compute, RayTrace, Transfer}

public delegate void PassSetup(GraphBuilder b);
public delegate void PassExecute(CommandBuffer cmd, PassResources resources, in Renderer.FrameContext frame);

internal struct ResourceAccess
{
    public int ResourceId;
    public int Version;
    public ResourceUsage Usage;
    public bool IsWrite;
    public bool IsImage;
}    
    
internal sealed class GraphPass
{
    public string Name;
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
}

public readonly struct PassResources
{
    private readonly FrameGraph _g;
    internal PassResources(FrameGraph g) => _g = g;
    public ImageView View(GraphImage h) => _g.ResolveView(h);
    public Image Image(GraphImage h) => _g.ResolveImage(h);
    public Buffer Buffer(GraphBuffer h) => _g.ResolveBuffer(h);
}