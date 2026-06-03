using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.FrameGraph;

public interface IGraphBuilder
{
    GraphImage CreateImage(in ImageDesc desc, string name);
    GraphBuffer CreateBuffer(in BufferDesc desc, string name);

    GraphImage  Read(GraphImage h,   ResourceUsage usage);
    GraphBuffer Read(GraphBuffer h,  ResourceUsage usage);
    GraphImage  Write(GraphImage h,  ResourceUsage usage);
    GraphBuffer Write(GraphBuffer h, ResourceUsage usage);
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
    public GraphImage Read(GraphImage h, ResourceUsage usage)
    {
        _pass.Reads.Add(new ResourceAccess
        {
            ResourceId = h.resourceId, Version = h.Version, Usage = usage, IsWrite = false, IsImage = true
        });
        return h;
    }


    public GraphImage Write(GraphImage h, ResourceUsage usage)
    {
        var r = _g.GetResource(h.resourceId);

        r.Producers.Add(_g.CurrentPassIndex);     // new version, this pass produces it
        var nv = new GraphImage(h.resourceId, r.CurrentVersion);
        _pass.Writes.Add(new ResourceAccess {
            ResourceId = h.resourceId, Version = nv.Version,
            Usage = usage, IsWrite = true, IsImage = true });
        if (usage is ResourceUsage.ColorAttachment) _pass.ColorTargets.Add(h.resourceId);
        if (usage is ResourceUsage.DepthAttachment) _pass.DepthTarget = h.resourceId;
        return nv;
    }
    
    
    public GraphBuffer Read(GraphBuffer h, ResourceUsage usage)
    {
        _pass.Reads.Add(new ResourceAccess
        {
            ResourceId = h.resourceId, Version = h.Version, Usage = usage, IsWrite = false, IsImage = false
        });
        return h;
    }

    public GraphBuffer Write(GraphBuffer h, ResourceUsage usage)
    {
        var r = _g.GetResource(h.resourceId)!;
        r.Producers.Add(_g.CurrentPassIndex);     // new version, this pass produces it
        var nv = new GraphBuffer(h.resourceId, r.CurrentVersion);
        _pass.Writes.Add(new ResourceAccess
        {
            ResourceId = h.resourceId, Version = nv.Version, Usage = usage, IsWrite = true, IsImage = false
        });
        return nv;
    }
}