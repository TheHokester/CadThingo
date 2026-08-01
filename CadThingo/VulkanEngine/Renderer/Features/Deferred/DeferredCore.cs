using System.Runtime.CompilerServices;
using CadThingo.VulkanEngine.Renderer.FeatureLifecycle;
using CadThingo.VulkanEngine.Renderer.FeatureLifecycle.RenderFeatureInterfaces;
using CadThingo.VulkanEngine.Renderer.Features.Forward;
using CadThingo.VulkanEngine.Renderer.Features.IBL;
using CadThingo.VulkanEngine.Renderer.Features.Tonemapping;
using CadThingo.VulkanEngine.Renderer.Features.Shared;
using CadThingo.VulkanEngine.Renderer.FrameGraph;
using Silk.NET.Vulkan;

// The graph type shares its name with its namespace; alias it so bare `FrameGraph`
// never has to disambiguate type-vs-namespace.
using DeferredGraph = CadThingo.VulkanEngine.Renderer.FrameGraph.FrameGraph;

namespace CadThingo.VulkanEngine.Renderer.Features.Deferred;

/// <summary>
/// The deferred-shading technique as an <see cref="IRenderCore"/>. <see cref="Render"/> does the
/// per-frame CPU packing (probes, light-cull dims, skybox) then executes the graph, which records
/// cull -> light-cull -> geometry -> lighting -> skybox -> transparent -> tonemap and leaves
/// FinalColor in <c>ShaderReadOnlyOptimal</c> for the host. This core owns the five pipelines the
/// chain is built from; tonemap and skybox are shared, so they come in from the feature host.
/// </summary>
internal sealed class DeferredCore : IRenderCore, IGraphCore, IBakeFeature,
                                     ISelfRegisteringFeature<DeferredCore>, INeedsGpu, INeedsHost,
                                     INeedsFeature<ISharedPipelines>, INeedsFeature<IIblProvider>, INeedsFeature<IReflectionProbeProvider>
{
    // Order 10 = first core built, so index 0 in the mode combo: the boot default and fallback.
    // Ungated - the deferred path is the one technique every device can run.
    public static FeatureDesc Desc => new(Order: 10, Gate: _ => true, Make: () => new DeferredCore());

    [ModuleInitializer]
    internal static void _Reg() => FeatureCatalog.Register<DeferredCore>();

    private GpuContext _gpu;
    
    GpuContext INeedsGpu.Gpu { set => _gpu = value; }

    //feature injections
    
    private ISharedPipelines _shared = null!;
    private IIblProvider _ibl = null!;
    private IReflectionProbeProvider _reflProbeProvider = null!;
    ISharedPipelines INeedsFeature<ISharedPipelines>.Dependency { set => _shared = value; }
    IIblProvider INeedsFeature<IIblProvider>.Dependency { set => _ibl = value; }
    IReflectionProbeProvider INeedsFeature<IReflectionProbeProvider>.Dependency { set => _reflProbeProvider = value; }
    

    // Host handle the technique pipelines still take at construction.
    private Renderer _host = null!;
    Renderer INeedsHost.Host { set => _host = value; }

    // Last snapshot the host handed over. Bake rebuilds the graph out of band, with no resize to
    // supply an extent, so it reads FinalColor and the extent from here.
    private HostTargets _targets;

    // Technique-private pipelines. Each owns its own VkPipeline, layouts, descriptor sets and
    // per-pipeline buffers; this core owns their lifetime.
    private GeometryPipeline    _geometry    = null!;
    private DrawCullPipeline    _drawCull    = null!;
    private LightCullPipeline   _lightCull   = null!;
    private PbrDeferredPipeline _pbr         = null!;
    private TransparentPipeline _transparent = null!;

    // The frame graph driving the deferred chain
    private DeferredGraph? _graph;

    // Specialization-constant gate for soft (PCSS-style) ray-queried shadows, plus the pending
    // rebuild it implies. The settings panel publishes the intent and never writes renderer state.
    private bool _softShadows = true;
    private bool _rebuildPending;
    private IDisposable? _softShadowSub;

    // Per-frame light-cull dispatch dims, written by Render (from PbrDeferredPipeline.UpdatePerFrame)
    // before Execute and pulled by the module's LightCullPass delegate at Execute time.
    private uint _frameLightCount;
    private uint _frameTileCountX;
    private uint _frameTileCountY;

    public string Name => "Deferred";
    public Renderer.RenderMode Mode => Renderer.RenderMode.Deferred;

    /// <summary>Builds this core's pipelines. Runs here and not in the constructor because every
    /// pipeline needs the injected GpuContext, which the wiring pass supplies after construction.
    /// The graph is sized to the render extent, so <see cref="Resize"/> builds it instead.</summary>
    public void Initialize()
    {
        _softShadowSub = Engine.EventBus.Subscribe<PbrSoftShadowingChangedEvent>(e =>
        {
            _softShadows    = e.GetEnabled;
            _rebuildPending = true;
        });

        // The g-buffers / depth / HDR are graph transients and PBR's g-buffer set is baked when the
        // graph compiles, so the order these initialize in does not matter.
        _geometry = new GeometryPipeline(_gpu, _host);
        _geometry.Initialize();

        _drawCull = new DrawCullPipeline(_gpu, _host);
        _drawCull.Initialize();

        _pbr = new PbrDeferredPipeline(_gpu, _host) { SoftShadowsEnabled = _softShadows };
        _pbr.Initialize();

        _lightCull = new LightCullPipeline(_gpu, _host);
        _lightCull.Initialize();

        // Forward+ BLEND pass between lighting and tonemap. Lights / TLAS / bindless come from the
        // scene set; the tile cull buffers are wired from LightCullPipeline inside the graph.
        _transparent = new TransparentPipeline(_gpu, _host) { SoftShadowsEnabled = _softShadows };
        _transparent.Initialize();
    }

    /// <summary>Soft-shadow state the settings panel checkbox reads back after publishing a
    /// change.</summary>
    internal bool SoftShadowsEnabled => _softShadows;

    // ---- Bake phase: soft-shadow spec-constant rebuild ----------------------

    public bool BakePending => _rebuildPending;

    /// <summary>
    /// Rebuilds the two PBR pipelines in place (stable object identity, fresh GPU handles) so the
    /// new soft-shadow spec constant takes effect. Rebuild recreates the set-1 layout handle, which
    /// orphans the graph's baked g-buffer set, so the graph rebuild that follows re-bakes it against
    /// the new layout. Scene-set bindings live on the registry and need no rewire.
    /// </summary>
    public void Bake()
    {
        _rebuildPending = false;
        _gpu.Gfx.Vk!.DeviceWaitIdle(_gpu.Gfx.Device);

        _pbr.SoftShadowsEnabled = _softShadows;
        _pbr.Rebuild();

        _transparent.SoftShadowsEnabled = _softShadows;
        _transparent.Rebuild();

        BuildGraph();
    }

    // (Re)builds the deferred chain as a DeferredGraph. The graph owns the g-buffers / depth / HDR
    // as transients allocated in Compile at the snapshot's extent, and imports FinalColor, which the
    // host owns and consumes outside the graph. Runs on every resize (fresh extent, fresh
    // transients) and on a soft-shadow bake.
    private void BuildGraph()
    {
        _graph?.Dispose();
        var fg = new DeferredGraph(_gpu.Gfx);

        // Tonemap and skybox are shared pipelines, composed in from the featurehost rather than owned here.
        var tonemap = new TonemapModule(_shared.Tonemap);
        var module = new DeferredModule(
            _drawCull, _lightCull, _geometry,
            _pbr, _shared.Skybox, _transparent, tonemap,
            () => (_frameLightCount, _frameTileCountX, _frameTileCountY));

        module.Build(fg.RootScope().Child("Deferred"),
            new DeferredModule.Inputs(_targets.FinalColor, _targets.Extent), out var o);

        fg.MarkOutput(o.Final);
        fg.Compile();
        _graph = fg;

        // Compile() bakes every graph-resident descriptor set (cull, light-cull, g-buffer and
        // tonemap's HDR input) from the fresh transients, so no manual rebind is needed here.
    }

    /// <summary>Nothing to rebind on activation: tonemap's HDR input is graph-baked from this
    /// core's own HDR transient, so switching to this core binds its graph's set.</summary>
    public void Activate() { }

    public void Resize(in HostTargets targets)
    {
        _targets = targets;
        BuildGraph();
    }

    public void Render(in RenderFrame frame)
    {
        var cmd    = frame.Cmd;
        var view   = frame.View;
        var camera = view.Camera;
        var scene  = view.Scene;
        uint currentFrame = view.FrameIndex;
        uint lightCount   = view.LightCount;   // packed by the draw loop's extraction

        // Reflection-probe scheduler bookkeeping.
        _reflProbeProvider.Tick(view.FrameCounter, scene);

        // Geometry needs no update here: its view+proj rides the scene set's (0,0) arena slot and
        // is pushed inside its Record call.
        var (tileCountX, tileCountY) = _pbr.UpdatePerFrame(view);
        _transparent.UpdatePerFrame(view, tileCountX, tileCountY);
        // Stash for the module's LightCullPass body, which runs inside the graph below.
        _frameLightCount = lightCount; _frameTileCountX = tileCountX; _frameTileCountY = tileCountY;
        _shared.Skybox.UpdatePerFrame(view, ImGui.EditorState.SkyboxIntensity);

        // Reflection-probe cluster cull. Tile-only today (zSlices=1).
        _reflProbeProvider.BuildClusters(view, 0.1f, 100f, tileCountX, tileCountY);
        // Refresh the per-probe SSBO read by the PBR lighting shader.
        _reflProbeProvider.WriteProbeRecords(currentFrame);

        // Reflection-probe capture for the next dirty probe (if any), bounded to one probe/frame.
        _reflProbeProvider.RecordCapture(cmd, view, scene);

        // Record the deferred chain: cull -> light-cull -> geometry -> lighting -> skybox ->
        // transparent -> tonemap. Cull and light-cull run as compute passes, and every barrier comes
        // from the usage table. The cull pass reads the renderables the draw loop extracted into the
        // view.
        _graph!.Execute(cmd, view);
    }

    // ---- IGraphCore surface (Stats panel + gfx-chunk submission) ---------------
    public GraphStats? GraphStats => _graph?.Stats;
    public string ToDot() => _graph?.ToDot() ?? "(no deferred frame graph)";
    public bool CollectPipelineStats
    {
        get => _graph?.CollectPipelineStats ?? false;
        set { if (_graph != null) _graph.CollectPipelineStats = value; }
    }
    // Deferred graph has no async-compute passes; gfx chunks are never deferred.
    public bool HasPendingGfxChunks => false;

    /// <exception cref="InvalidOperationException">Always, since this core never pends a chunk.</exception>
    public void SubmitGfxChunks(Queue gfxQueue, SemaphoreSubmitInfo imgAvailWait,
        SemaphoreSubmitInfo renderDoneSignal, CommandBuffer hostCmd, Fence fence)
        => throw new InvalidOperationException("DeferredCore: no pending gfx chunks");

    public void Dispose()
    {
        _softShadowSub?.Dispose();

        // Graph first: its passes hold references to the pipelines they record with.
        _graph?.Dispose();
        _graph = null;

        _transparent?.Dispose();
        _pbr?.Dispose();
        _lightCull?.Dispose();
        _drawCull?.Dispose();
        _geometry?.Dispose();
    }
}