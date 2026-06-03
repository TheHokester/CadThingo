using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.FrameGraph;

public readonly record struct GraphImage(int resourceId, int Version);
public readonly record struct GraphBuffer(int resourceId, int Version);

public struct ImageDesc
{
    public Format Format;
    public Extent2D Extent;
    public uint Mips, Layers;
    public ImageUsageFlags Usage;
    public SampleCountFlags Samples;//default Count1Bit
}

public struct BufferDesc
{
    public ulong size;
    public BufferUsageFlags Usage;
}

public enum ResidencyKind
{
    Transient,
    Imported
}

internal sealed class GraphResource
{
    public int Id;
    public string Name;
    public bool IsImage;
    public ResidencyKind Residency;
    
    public ImageDesc? ImageDesc;    //valid when IsImage
    public BufferDesc? BufferDesc;  //valid when IsBuffer\
    
    //versioning: producers[v] = pass index that produced version v (or -1 = initial)
    public List<int> Producers = new() { -1 };
    public int CurrentVersion => Producers.Count - 1;
    
    //Filled at compile, Imported resources adopt an externally owned handle.
    public Image? PhysImage;
    public ImageView? PhysView;
    public Buffer? PhysBuffer;
    public SubAlloc? Alloc; // transients only
    public ImageLayout InitialLayout; // imported: layout the resource arrives in each frame
    public ImageLayout FinalLayout;   // imported images: layout the graph hands back on exit

    // Transient lifetime, in schedule-order indices (Compile step 4). FirstUse ==
    // int.MaxValue / LastUse == -1 means "never touched by a live pass". Consumed by
    // aliasing in Phase 4; computed in Phase 1 so the schedule is the one ordering source.
    public int FirstUse = int.MaxValue;
    public int LastUse = -1;
}

