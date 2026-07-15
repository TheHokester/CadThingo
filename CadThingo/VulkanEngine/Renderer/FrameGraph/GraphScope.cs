using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.FrameGraph;

/// <summary>
/// The graph-scope authoring surface handed to an <see cref="IGraphModule{TInputs,TOutputs}"/>.
/// Wraps a <see cref="FrameGraph"/> plus a name prefix and forwards resource creation +
/// AddPass to it, applying the prefix so a module's resources/passes are namespaced
/// (e.g. "Deferred/HDRColor"). This is the *graph*-scope builder (CreateImage/AddPass); the
/// per-pass Read/Write surface is <see cref="GraphBuilder"/>.
///
/// Flatten, not nest: a scope appends straight into the one FrameGraph registry, so the
/// compiler keeps a global view for culling/sync/aliasing (render-graph.md sec 9 -- logical
/// subgraph, physical flatten). Modules compose by nesting scopes via <see cref="Child"/>;
/// the top level uses an empty-prefix root scope (<see cref="FrameGraph.RootScope"/>).
/// </summary>
public readonly struct GraphScope
{
    private readonly FrameGraph _g;

    /// <summary>The accumulated name prefix ("" at the root). Resource and pass names are
    /// emitted as <c>Prefix/name</c>.</summary>
    public string Prefix { get; }

    /// <summary>True when a pass declared <see cref="QueueClass.AsyncCompute"/> will really
    /// run on the device's dedicated compute queue. Modules branch on this to pick between
    /// an async layout and a single-queue fallback (e.g. the wavefront PT's Connect overlap).</summary>
    public bool AsyncComputeAvailable => _g.AsyncComputeAvailable;

    internal GraphScope(FrameGraph g, string prefix)
    {
        _g = g;
        Prefix = prefix;
    }

    /// <summary>Nests a child scope under <paramref name="name"/> (prefixes concatenate,
    /// e.g. "svgf" -> "svgf/atrous"). Two instances of one module are given distinct names
    /// (e.g. "tonemap#0") so their resources never collide.</summary>
    public GraphScope Child(string name) => new(_g, Combine(Prefix, name));

    private static string Combine(string a, string b) => string.IsNullOrEmpty(a) ? b : $"{a}/{b}";
    private string Qualify(string name) => Combine(Prefix, name);

    // ---- Resource declaration (forwarded with the scope prefix applied to names) ----------
    public GraphImage  CreateImage(in ImageDesc desc, string name)   => _g.CreateImage(in desc, Qualify(name));
    public GraphBuffer CreateBuffer(in BufferDesc desc, string name) => _g.CreateBuffer(in desc, Qualify(name));

    public GraphImage ImportImage(Image image, ImageView view, in ImageDesc desc,
        ImageLayout currentLayout, string name, ImageLayout? finalLayout = null) =>
        _g.ImportImage(image, view, in desc, currentLayout, Qualify(name), finalLayout);

    public GraphBuffer ImportBuffer(Buffer buffer, in BufferDesc desc, string name) =>
        _g.ImportBuffer(buffer, in desc, Qualify(name));

    public GraphBuffer ImportBufferPerFrame(Buffer[] perFrame, in BufferDesc desc, string name) =>
        _g.ImportBufferPerFrame(perFrame, in desc, Qualify(name));

    // ---- Graph-owned shared set -----------------------------------------------------------
    /// <summary>Opts the graph into owning one shared descriptor set for the whole module (see
    /// <see cref="GraphSharedSetSpec"/>). The pipeline supplies the layout + current buffer handles;
    /// the graph allocates + writes + owns the set and exposes it via <see cref="FrameGraph.GraphSharedSet"/>.</summary>
    public void UseGraphSharedSet(in GraphSharedSetSpec spec) => _g.UseGraphSharedSet(spec);

    // ---- Pass declaration -----------------------------------------------------------------
    public void AddPass(string name, PassType type, QueueClass queue,
        PassSetup setup, PassExecute execute,
        bool preferAsync = false, bool hasSideEffects = false) =>
        _g.AddPass(Qualify(name), type, queue, setup, execute, preferAsync, hasSideEffects, Prefix);

    // ---- Wire-time port validation --------------------------------------------------------
    /// <summary>Validates an input image port: the bound resource must be an image of the
    /// expected <paramref name="format"/> whose declared usage includes <paramref name="usage"/>.
    /// Throws at build time on mismatch (a wiring error, not runtime corruption). Returns the
    /// handle so calls chain. Extent is not checked -- imported targets may omit it.</summary>
    public GraphImage ExpectImage(GraphImage h, Format format, ImageUsageFlags usage)
    {
        var r = _g.GetResource(h.resourceId)
            ?? throw new InvalidOperationException($"GraphScope.ExpectImage: handle {h.resourceId} is not registered.");
        if (!r.IsImage)
            throw new InvalidOperationException($"GraphScope.ExpectImage: '{r.Name}' is a buffer; an image was expected.");
        var d = r.ImageDesc
            ?? throw new InvalidOperationException($"GraphScope.ExpectImage: '{r.Name}' has no ImageDesc.");
        if (d.Format != format)
            throw new InvalidOperationException($"GraphScope port mismatch on '{r.Name}': format is {d.Format}, expected {format}.");
        if ((d.Usage & usage) != usage)
            throw new InvalidOperationException($"GraphScope port mismatch on '{r.Name}': usage {d.Usage} lacks {usage}.");
        return h;
    }

    /// <summary>Validates an input buffer port: the bound resource must be a buffer whose
    /// declared usage includes <paramref name="usage"/>. Imported buffers (whose BufferDesc may
    /// be default) skip the usage check. Throws at build time on a type mismatch.</summary>
    public GraphBuffer ExpectBuffer(GraphBuffer h, BufferUsageFlags usage = 0)
    {
        var r = _g.GetResource(h.resourceId)
            ?? throw new InvalidOperationException($"GraphScope.ExpectBuffer: handle {h.resourceId} is not registered.");
        if (r.IsImage)
            throw new InvalidOperationException($"GraphScope.ExpectBuffer: '{r.Name}' is an image; a buffer was expected.");
        if (usage != 0 && r.BufferDesc is { } d && (d.Usage & usage) != usage)
            throw new InvalidOperationException($"GraphScope port mismatch on '{r.Name}': usage {d.Usage} lacks {usage}.");
        return h;
    }
}