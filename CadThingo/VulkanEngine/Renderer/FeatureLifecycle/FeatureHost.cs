using System.Text;
using CadThingo.VulkanEngine.Renderer.FeatureLifecycle.RenderFeatureInterfaces;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.FeatureLifecycle;

/// <summary>
/// Owns every feature's lifetime and every phase pump, so <c>Renderer.Initialize</c>,
/// <c>DrawFrame</c> and <c>Cleanup</c> stop naming individual features. Adding a feature is a new
/// file with a descriptor; nothing here changes.
///
/// Features are sorted into a bucket per phase interface at build time, so a pump is a walk of a
/// short pre-filtered list rather than a type test per feature per frame.
/// </summary>
internal sealed class FeatureHost : IDisposable
{
    private readonly GpuContext _gpu;
    private readonly Renderer   _host;

    // Every built feature, in Order. Dispose walks this backwards; the phase lists below are
    // subsets of it, each already in the same Order.
    private readonly List<IRenderFeature>   _all      = [];
    private readonly List<IPreDrawFeature>  _preDraw  = [];
    private readonly List<IPostDrawFeature> _postDraw = [];
    private readonly List<IResizeFeature>   _resize   = [];
    private readonly List<IBakeFeature>     _bake     = [];
    private readonly List<IRenderCore>      _cores    = [];

    // How many descriptors the gates rejected on this device. Only the manifest reads it - a
    // gated-out feature was never constructed, so there is nothing else left of it to report.
    private int _gatedOut;

    private bool _built;

    /// <summary>The render cores, in descriptor Order. Index into this is the mode-combo index.</summary>
    public IReadOnlyList<IRenderCore> Cores => _cores;

    /// <summary>Every built feature, in Order. For whole-set inspection only - the boot-time
    /// registry cross-check walks it to find the pipelines features own. Resolve a specific
    /// feature with <see cref="Get{T}"/>, not by scanning this.</summary>
    public IReadOnlyList<IRenderFeature> All => _all;

    public FeatureHost(GpuContext gpu, Renderer host)
    {
        _gpu  = gpu;
        _host = host;
    }

    /// <summary>
    /// Constructs, wires and initializes every feature the device can run, in three separate passes
    /// over the whole set.
    ///
    /// The passes are separate because each one needs the previous to have finished for EVERY
    /// feature, not just the ones ahead of it in Order. Wiring cannot start until all features
    /// exist, or a dependency would resolve to null purely because its Order was higher; Initialize
    /// cannot start until all wiring is done, or a feature would build against a half-wired
    /// collaborator. Doing it in one pass is exactly the ordering trap this shape removes.
    ///
    /// A fourth pass primes the resize path at the boot extent, so a feature's extent-sized state
    /// has exactly one construction site instead of an Initialize copy beside a Resize copy.
    /// </summary>
    public void BuildAll(in HostTargets targets)
    {
        if (_built) throw new InvalidOperationException("FeatureHost.BuildAll ran twice");
        _built = true;

        // 1. Construct, gated. A rejected feature is never built, so it allocates nothing.
        foreach (var desc in FeatureCatalog.Descriptors.OrderBy(d => d.Order))
        {
            if (!desc.Gate(_gpu)) { _gatedOut++; continue; }
            Add(desc.Make());
        }

        // 2. Wire. Fields that were null since construction get their values here.
        foreach (var f in _all) Wire(f);

        // 3. Initialize, in Order, so a feature can rely on any lower-Order collaborator being
        //    fully built by the time it runs.
        foreach (var f in _all) f.Initialize();

        // 4. Build the extent-sized state through the same path a viewport resize takes.
        Resize(targets);
    }

    /// <summary>Resolves a built feature by contract or concrete type. Null when the device gated
    /// it out - callers that cannot cope with absence should be gated the same way.</summary>
    public T? Get<T>() where T : class => _all.OfType<T>().FirstOrDefault();

    // ---- Phase pumps -------------------------------------------------------

    /// <summary>Out-of-band work before the frame's command buffer is recorded.</summary>
    public void PreDraw(in RenderView view)
    {
        foreach (var f in _preDraw) f.PreDraw(view);
    }

    /// <summary>Recording that lands after the active core has produced FinalColor.</summary>
    public void PostDraw(CommandBuffer cmd, in RenderView view)
    {
        foreach (var f in _postDraw) f.PostDraw(cmd, view);
    }

    /// <summary>Builds or rebuilds size-dependent state at the snapshotted targets. The caller has
    /// already idled the device and reallocated them. Must follow that realloc immediately - until
    /// this returns, every feature still holds a snapshot of the images the realloc disposed.</summary>
    public void Resize(in HostTargets targets)
    {
        foreach (var f in _resize) f.Resize(targets);
    }

    /// <summary>Services pending bakes, in Order. Request-driven: a feature with nothing stale
    /// costs one bool test. Ordered so a feature can bake against an already-baked collaborator.</summary>
    public void ServiceBakes()
    {
        foreach (var f in _bake)
            if (f.BakePending) f.Bake();
    }

    // ---- Boot manifest -----------------------------------------------------

    /// <summary>
    /// What actually got built on this device, in Order, with each feature's phase roles. This is
    /// the replacement for the boot-order list that scattering Order across feature files costs
    /// us - and it is strictly better, because it reflects the gates rather than the source.
    /// </summary>
    public string Dump()
    {
        var sb = new StringBuilder();
        sb.AppendLine($"[features] {_all.Count} built, {_gatedOut} gated out by device caps");
        foreach (var f in _all)
        {
            var roles = new List<string>();
            if (f is IRenderCore c)      roles.Add($"core:{c.Mode}");
            if (f is IBakeFeature)       roles.Add("bake");
            if (f is IPreDrawFeature)    roles.Add("preDraw");
            if (f is IPostDrawFeature)   roles.Add("postDraw");
            if (f is IResizeFeature and not IRenderCore) roles.Add("resize");
            if (f is INeedsHost)         roles.Add("NEEDS-HOST");
            sb.AppendLine($"  {f.Name,-36} {string.Join(' ', roles)}");
        }
        return sb.ToString().TrimEnd();
    }

    // ---- Internals ---------------------------------------------------------

    private void Add(IRenderFeature feature)
    {
        _all.Add(feature);

        if (feature is IPreDrawFeature p)  _preDraw.Add(p);
        if (feature is IPostDrawFeature q) _postDraw.Add(q);
        if (feature is IResizeFeature r)   _resize.Add(r);
        if (feature is IBakeFeature b)     _bake.Add(b);
        if (feature is IRenderCore core)   _cores.Add(core);
    }

    /// <summary>Fills a feature's declared dependencies. Infrastructure is a direct type test;
    /// collaborators need reflection because the contract type is a generic argument, which is only
    /// knowable from the feature's implemented interface list.</summary>
    private void Wire(IRenderFeature feature)
    {
        if (feature is INeedsGpu g)  g.Gpu  = _gpu;
        if (feature is INeedsHost h) h.Host = _host;   // transitional; see INeedsHost

        foreach (var iface in feature.GetType().GetInterfaces())
        {
            if (!iface.IsGenericType || iface.GetGenericTypeDefinition() != typeof(INeedsFeature<>))
                continue;

            var contract = iface.GetGenericArguments()[0];
            var provider = _all.FirstOrDefault(contract.IsInstanceOfType)
                ?? throw new InvalidOperationException(
                    $"{feature.Name} needs {contract.Name}, but no built feature provides it. " +
                    "Either its provider is gated out on this device (gate the consumer the same " +
                    "way) or the provider is not registered in the catalog.");

            iface.GetProperty("Dependency")!.SetValue(feature, provider);
        }
    }

    /// <summary>Reverse-Order dispose: a feature is torn down before anything it was allowed to
    /// depend on, which is the exact inverse of the Initialize guarantee.</summary>
    public void Dispose()
    {
        for (int i = _all.Count - 1; i >= 0; i--) _all[i].Dispose();

        _all.Clear();
        _preDraw.Clear();
        _postDraw.Clear();
        _resize.Clear();
        _bake.Clear();
        _cores.Clear();
    }
}
