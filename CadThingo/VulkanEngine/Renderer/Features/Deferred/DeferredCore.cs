using System.Runtime.CompilerServices;
using CadThingo.VulkanEngine.Renderer.FeatureLifecycle;
using CadThingo.VulkanEngine.Renderer.FeatureLifecycle.RenderFeatureInterfaces;
using CadThingo.VulkanEngine.Renderer.Features.Forward;
using CadThingo.VulkanEngine.Renderer.Features.Tonemapping;
using CadThingo.VulkanEngine.Renderer.Features.Shared;
using CadThingo.VulkanEngine.Renderer.FrameGraph;
using Silk.NET.Vulkan;

// The graph type shares its name with its namespace; alias it so bare `FrameGraph`
// never has to disambiguate type-vs-namespace.
using DeferredGraph = CadThingo.VulkanEngine.Renderer.FrameGraph.FrameGraph;

namespace CadThingo.VulkanEngine.Renderer.Features.Deferred;

/// <summary>
/// The deferred-shading technique as an <see cref="IRenderCore"/>. 
/// 
/// owns every piece of deferred-technique-local state that previously
/// lived on the <see cref="Renderer"/>: the graph's HDR / g-buffer transient views (cached for the
/// host-side descriptor rebinds) and the per-frame light-cull dimensions the LightCullPass delegate
/// reads.
///
/// <see cref="Render"/>  per-frame CPU packing
/// (materials / lights / probes / extract) then <c>graph.Execute</c>, which records
/// cull -> light-cull -> geometry -> lighting -> skybox -> transparent -> tonemap and leaves
/// FinalColor in <c>ShaderReadOnlyOptimal</c> for the host.
///
/// It also owns the five technique-private pipelines the chain is built from (cull, light-cull,
/// geometry, PBR lighting, transparent). Nothing outside this core records with them, so nothing
/// outside it needs to know they exist. The two genuinely shared ones - tonemap and skybox - stay
/// with the host and are composed in, not owned.
///
/// Soft shadows are a fragment-stage specialization constant, so toggling them is a pipeline
/// rebuild rather than a uniform write. That makes it a bake: the event handler stores the value
/// and flags, and the bake pump rebuilds at a point where no command buffer is open.
/// </summary>
internal sealed class DeferredCore : IRenderCore, IGraphCore, IBakeFeature,
                                     ISelfRegisteringFeature<DeferredCore>, INeedsGpu, INeedsHost,
                                     INeedsFeature<ISharedPipelines>
{
    // Order 10 = first core built, so index 0 in the mode combo: the boot default and fallback.
    // Ungated - the deferred path is the one technique every device can run.
    public static FeatureDesc Desc => new(Order: 10, Gate: _ => true, Make: () => new DeferredCore());

    [ModuleInitializer]
    internal static void _Reg() => FeatureCatalog.Register<DeferredCore>();

    private GpuContext _gpu;
    GpuContext INeedsGpu.Gpu { set => _gpu = value; }

    // The tonemap + skybox this chain composes. One instance each, shared with every other core -
    // which is why they arrive as a collaborator rather than being constructed here.
    private ISharedPipelines _shared = null!;
    ISharedPipelines INeedsFeature<ISharedPipelines>.Dependency { set => _shared = value; }

    // Transitional: render targets and the probe system still come through the host.
    private Renderer _host = null!;
    Renderer INeedsHost.Host { set => _host = value; }

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
    // rebuild it implies. Single owner: the settings panel publishes the intent and never writes
    // renderer state.
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

    /// <summary>Builds this core's pipelines and then the graph that composes them. Runs at
    /// Initialize rather than in the constructor because everything here needs the injected
    /// GpuContext, which does not exist until the wiring pass has run.</summary>
    public void Initialize()
    {
        _softShadowSub = Engine.EventBus.Subscribe<PbrSoftShadowingChangedEvent>(e =>
        {
            _softShadows    = e.GetEnabled;
            _rebuildPending = true;
        });

        // The g-buffers / depth / HDR are FrameGraph transients allocated when the graph compiles
        // below, and PBR's g-buffer set is (re)baked there, so these can initialize in any order.
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

        BuildGraph();
    }

    /// <summary>Current soft-shadow state, for the settings panel's checkbox. The single source -
    /// the panel stages a change and publishes it, and reads back from here.</summary>
    internal bool SoftShadowsEnabled => _softShadows;

    // ---- Bake phase: soft-shadow spec-constant rebuild ----------------------

    public bool BakePending => _rebuildPending;

    /// <summary>
    /// Rebuilds the two PBR pipelines in place (stable object identity, fresh GPU handles) so the
    /// new soft-shadow spec constant takes effect. Rebuild recreates the set-1 layout handle, so
    /// the graph's baked g-buffer set - allocated from the old layout - has to be re-baked against
    /// the new one, which is what the graph rebuild below does. Scene-set bindings live on the
    /// registry and need no rewire.
    /// </summary>
    public void Bake()
    {
        _rebuildPending = false;
        _host.gfx.Vk!.DeviceWaitIdle(_host.gfx.Device);

        _pbr.SoftShadowsEnabled = _softShadows;
        _pbr.Rebuild();

        _transparent.SoftShadowsEnabled = _softShadows;
        _transparent.Rebuild();

        BuildGraph();
    }

    /// <summary>
    /// (Re)builds the deferred chain as the <see cref="DeferredGraph"/>. The graph OWNS the
    /// g-buffers / depth / HDR as transients (allocated in Compile) and imports only FinalColor
    /// (RenderTargets-owned, consumed outside the graph). Called from the ctor and on every resize
    /// (fresh extent -> fresh transients). Pass bodies are closures over the host's pipelines --
    /// they only need to exist by the time Execute runs, which they do (built in Initialize before
    /// this core is constructed).
    /// </summary>
    private void BuildGraph()
    {
        _graph?.Dispose();
        var fg = new DeferredGraph(_host.gfx);

        // The deferred technique taking an imported tonemap module. Tonemap and skybox are the two
        // shared pipelines - composed in from the host, not owned here.
        var tonemap = new TonemapModule(_shared.Tonemap);
        var module = new DeferredModule(
            _drawCull, _lightCull, _geometry,
            _pbr, _shared.Skybox, _transparent, tonemap,
            () => (_frameLightCount, _frameTileCountX, _frameTileCountY));

        module.Build(fg.RootScope().Child("Deferred"),
            new DeferredModule.Inputs(_host.renderTargets.FinalColor, _host.renderExtent), out var o);

        fg.MarkOutput(o.Final);
        fg.Compile();
        _graph = fg;

        // Compile() baked every graph-resident descriptor set (cull, light-cull, g-buffer, and
        // tonemap's HDR input) from the fresh transients, so there is no manual descriptor rebind
        // left to do here.
    }

    /// <summary>Nothing to rebind on activation: tonemap's HDR input is graph-baked from this
    /// core's own HDR transient, so switching to this core just binds its graph's set.</summary>
    public void Activate() { }

    public void Resize(Extent2D extent) => BuildGraph();

    public void Render(in RenderFrame frame)
    {
        var cmd    = frame.Cmd;
        var view   = frame.View;
        var camera = view.Camera;
        var scene  = view.Scene;
        uint currentFrame = view.FrameIndex;
        uint lightCount   = view.LightCount;   // packed by the draw loop's extraction

        // Reflection-probe scheduler bookkeeping.
        _host.reflectionProbeSystem.Tick(_host.frameCounter, scene);

        // Geometry's view+proj now rides the scene set's (0,0) arena slot, pushed inside
        // its Record call - no per-frame UBO update needed here anymore.
        var (tileCountX, tileCountY) =
            _pbr.UpdatePerFrame(currentFrame, camera, lightCount);
        _transparent.UpdatePerFrame(currentFrame, camera, lightCount, tileCountX, tileCountY);
        // Stash for the module's LightCullPass body (it runs inside the graph, below).
        _frameLightCount = lightCount; _frameTileCountX = tileCountX; _frameTileCountY = tileCountY;
        // Skybox always updates
        _shared.Skybox.UpdatePerFrame(currentFrame, camera, ImGui.EditorState.SkyboxIntensity);

        // Reflection-probe cluster cull. Tile-only today (zSlices=1). Cheap CPU work.
        float aspect = (float)view.RenderExtent.Width / view.RenderExtent.Height;
        _host.reflectionProbeSystem.BuildClusters(currentFrame, camera, aspect, 0.1f, 100f,
            tileCountX, tileCountY);
        // Refresh the per-probe SSBO read by the PBR lighting shader.
        _host.reflectionProbeSystem.WriteProbeRecords(currentFrame);

        // Reflection-probe capture for the next dirty probe (if any), bounded to one probe/frame.
        _host.reflectionProbeSystem.RecordCapture(cmd, currentFrame, _host.frameCounter, scene);

        // Record the deferred chain via the graph: cull -> light-cull -> geometry -> lighting ->
        // skybox -> transparent -> tonemap. Cull/light-cull run as compute passes and every barrier
        // is derived from the usage table. The cull pass reads the renderables the draw loop
        // already extracted into the view.
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