using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.FrameGraph;

public interface IGraphBuilder
{
    GraphImage CreateImage(in ImageDesc desc, string name);
    GraphBuffer CreateBuffer(in BufferDesc desc, string name);

    GraphImage  Read(GraphImage h,   ResourceUsage usage, string? bind = null);
    GraphBuffer Read(GraphBuffer h,  ResourceUsage usage, string? bind = null);
    GraphImage  Write(GraphImage h,  ResourceUsage usage, string? bind = null);
    GraphBuffer Write(GraphBuffer h, ResourceUsage usage, string? bind = null);

    // Opt this pass into a graph-baked descriptor set: the graph fills each binding a named
    // Read/Write references. Legacy passes never call it and bind their own sets (unchanged).
    void UsePassSet(in PassSetSpec spec);
    string Scope { get; }
}

public sealed class GraphBuilder : IGraphBuilder
{
    private readonly FrameGraph _g;
    private readonly GraphPass _pass;
    public string Scope { get; }
    
    internal GraphBuilder(FrameGraph g, GraphPass pass)
    {
        _g = g;
        _pass = pass;
        Scope = pass.Name;
    }

    
    // Resource declaration forwards to the graph (the registry owner); the returned
    // handle is version 0 (producer = -1 sentinel) until a Write bumps it.
    public GraphImage CreateImage(in ImageDesc desc, string name)   => _g.CreateImage(in desc, name);
    public GraphBuffer CreateBuffer(in BufferDesc desc, string name) => _g.CreateBuffer(in desc, name);

    /// <summary>Opts this pass into a graph-baked descriptor set (see <see cref="PassSetSpec"/>).
    /// Bindings are filled from the resources a named Read/Write references.</summary>
    public void UsePassSet(in PassSetSpec spec) => _pass.PassSet = spec;

    public GraphImage Read(GraphImage h, ResourceUsage usage, string? bind = null)
    {
        _pass.Reads.Add(new ResourceAccess
        {
            ResourceId = h.resourceId, Version = h.Version, Usage = usage, IsWrite = false, IsImage = true, Bind = bind
        });
        return h;
    }


    public GraphImage Write(GraphImage h, ResourceUsage usage, string? bind = null)
    {
        var r = _g.GetResource(h.resourceId);

        r.Producers.Add(_g.CurrentPassIndex);     // new version, this pass produces it
        var nv = new GraphImage(h.resourceId, r.CurrentVersion);
        _pass.Writes.Add(new ResourceAccess {
            ResourceId = h.resourceId, Version = nv.Version,
            Usage = usage, IsWrite = true, IsImage = true, Bind = bind });
        if (usage is ResourceUsage.ColorAttachment) _pass.ColorTargets.Add(h.resourceId);
        if (usage is ResourceUsage.DepthAttachment) _pass.DepthTarget = h.resourceId;
        return nv;
    }


    public GraphBuffer Read(GraphBuffer h, ResourceUsage usage, string? bind = null)
    {
        _pass.Reads.Add(new ResourceAccess
        {
            ResourceId = h.resourceId, Version = h.Version, Usage = usage, IsWrite = false, IsImage = false, Bind = bind
        });
        return h;
    }

    public GraphBuffer Write(GraphBuffer h, ResourceUsage usage, string? bind = null)
    {
        var r = _g.GetResource(h.resourceId)!;
        r.Producers.Add(_g.CurrentPassIndex);     // new version, this pass produces it
        var nv = new GraphBuffer(h.resourceId, r.CurrentVersion);
        _pass.Writes.Add(new ResourceAccess
        {
            ResourceId = h.resourceId, Version = nv.Version, Usage = usage, IsWrite = true, IsImage = false, Bind = bind
        });
        return nv;
    }
}