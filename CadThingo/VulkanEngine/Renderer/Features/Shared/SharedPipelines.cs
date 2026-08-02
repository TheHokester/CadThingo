using System.Runtime.CompilerServices;
using CadThingo.Graphics.Rendering;
using CadThingo.VulkanEngine.Renderer.FeatureLifecycle;
using CadThingo.VulkanEngine.Renderer.FeatureLifecycle.RenderFeatureInterfaces;
using CadThingo.VulkanEngine.Renderer.Features.IBL;
using CadThingo.VulkanEngine.Renderer.Features.Tonemapping;

namespace CadThingo.VulkanEngine.Renderer.Features.Shared;

/// <summary>
/// The pipelines more than one technique composes. A consumer takes this as
/// <c>INeedsFeature&lt;ISharedPipelines&gt;</c> and builds a per-core graph module around the single
/// instance - it must not construct its own, which is the whole point of them being shared.
/// </summary>
public interface ISharedPipelines
{
    /// <summary>HDRColor -> LDR FinalColor. Every core ends its graph with a module wrapping this
    /// one pipeline; the HDR input is graph-baked per core, so they can share it without conflict.</summary>
    TonemapPipeline Tonemap { get; }

    /// <summary>Background environment-cube draw, between lighting and transparent. Only the
    /// deferred chain composes it today, but it belongs to the environment rather than to any one
    /// technique, so it lives here rather than inside a core.</summary>
    SkyboxPipeline Skybox { get; }
}

/// <summary>
/// Owns the two shared pipelines. Order 5 - above the AS, below every core - so both exist,
/// initialized, by the time a core's Initialize builds the graph that composes them.
///
/// The tone-map curve is a fragment-stage specialization constant, so switching it is a pipeline
/// rebuild rather than a uniform write. That makes it a bake, in the same shape as DeferredCore's
/// soft-shadow toggle: the event handler stores the value and flags, and the pump rebuilds at a
/// point where no command buffer is open.
/// </summary>
internal sealed class SharedPipelines
    : ISharedPipelines, IBakeFeature, ISelfRegisteringFeature<SharedPipelines>, INeedsGpu, INeedsHost
{
    public static FeatureDesc Desc =>
        new(Order: 5, Gate: _ => true, Make: () => new SharedPipelines());

    [ModuleInitializer]
    internal static void _Reg() => FeatureCatalog.Register<SharedPipelines>();

    public string Name => "Shared pipelines (tonemap + skybox)";

    private GpuContext _gpu;
    private Renderer   _host = null!;
    GpuContext INeedsGpu.Gpu   { set => _gpu  = value; }
    Renderer   INeedsHost.Host { set => _host = value; }

    private TonemapPipeline _tonemap = null!;
    private SkyboxPipeline  _skybox  = null!;

    public TonemapPipeline Tonemap => _tonemap;
    public SkyboxPipeline  Skybox  => _skybox;

    // Selected tone-map curve + the rebuild it implies. Single owner: the settings panel publishes
    // the intent and reads the current value back through the host bridge.
    private TonemapOperator _operator = TonemapOperator.Filmic;
    private bool            _rebuildPending;
    private IDisposable?    _sub;

    /// <summary>Current curve, for the settings panel's combo.</summary>
    internal TonemapOperator Operator => _operator;

    public void Initialize()
    {
        _sub = Engine.EventBus.Subscribe<TonemapFilterChangedEvent>(e =>
        {
            _operator       = e.GetOperator;
            _rebuildPending = true;
        });

        // Tone-map / post pass - reads whichever HDR image the active core's graph produced, writes
        // the LDR FinalColor the swapchain blit sources. Its HDR-input descriptor is graph-baked
        // when each core compiles, so there is no descriptor write here.
        _tonemap = new TonemapPipeline(_gpu, _host) { Operator = _operator };
        _tonemap.Initialize();

        // EditorState.SkyboxEnabled gates the draw without re-recording the graph.
        _skybox = new SkyboxPipeline(_gpu, _host);
        _skybox.Initialize();
    }

    // ---- Bake phase: tone-map spec-constant rebuild -------------------------

    public bool BakePending => _rebuildPending;

    public void Bake()
    {
        _rebuildPending = false;
        _gpu.Gfx.Vk!.DeviceWaitIdle(_gpu.Gfx.Device);

        // In-place rebuild (stable object identity, fresh GPU handles) so each core module's
        // tonemap reference stays valid. Operator is a spec constant, read by Rebuild's Initialize.
        _tonemap.Operator = _operator;
        _tonemap.Rebuild();

        // Every core's graph-baked HDR set survives the in-place rebuild - a set outlives its
        // source layout and binds by pipeline-layout compatibility (identical definition), and the
        // HDR view is unchanged - so no graph rebuild is needed. Re-activating the core is only to
        // restart the PT cores' progressive accumulation against the new curve.
        _host.ReactivateCore();
    }

    public void Dispose()
    {
        _sub?.Dispose();
        _skybox?.Dispose();
        _tonemap?.Dispose();
    }
}
