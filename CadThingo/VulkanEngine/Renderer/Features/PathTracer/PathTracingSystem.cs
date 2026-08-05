using System.Runtime.CompilerServices;
using CadThingo.VulkanEngine.Renderer.FeatureLifecycle;
using CadThingo.VulkanEngine.Renderer.FeatureLifecycle.RenderFeatureInterfaces;
using CadThingo.VulkanEngine.Renderer.Features.SceneAcceleration;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Features.PathTracer;

/// <summary>
/// Owns what every tracer shares: the accumulator / out-color pair, their registry bindings, the
/// restart flag, and the acceleration-structure scalars forwarded off <see cref="SceneAS"/>. Four
/// cores (compute, RT, wavefront, ReSTIR) integrate into the same two images and hand off to each
/// other on a mode switch, so one feature holds them and the cores take
/// <see cref="IPathTracingProvider"/>.
///
/// Order 4: after the AS, before every core, so the images exist and are registered by the time a
/// core's first resize imports them into its graph.
/// </summary>
internal sealed class PathTracingSystem
    : IPathTracingProvider, IResizeFeature, ISelfRegisteringFeature<PathTracingSystem>,
      INeedsGpu, INeedsOptionalFeature<SceneAS>
{
    public static FeatureDesc Desc =>
        new(Order: 4, Gate: _ => true, Make: () => new PathTracingSystem());

    [ModuleInitializer]
    internal static void _Reg() => FeatureCatalog.Register<PathTracingSystem>();

    public string Name => "Path tracing (shared IO + AS scalars)";

    private GpuContext _gpu;
    GpuContext INeedsGpu.Gpu { set => _gpu = value; }

    // Null on a device that gated the AS out. The tracers still run there; the scalars below report
    // what they would for an empty scene.
    private SceneAS? _as;
    SceneAS? INeedsOptionalFeature<SceneAS>.Dependency { set => _as = value; }

    private GraphicsDevice Gfx => _gpu.Gfx;

    private ImageResource _accumulator = null!;
    private ImageResource _outColor    = null!;

    public ImageResource Accumulator => _accumulator;
    public ImageResource OutColor    => _outColor;

    private bool _accumulatorDirty;
    public  bool AccumulatorDirty      => _accumulatorDirty;
    public  void MarkAccumulatorDirty()  => _accumulatorDirty = true;
    public  void ClearAccumulatorDirty() => _accumulatorDirty = false;

    public bool  RayInfraReady         => _as?.Ready                 ?? false;
    public uint  EmissiveTriangleCount => _as?.EmissiveTriangleCount ?? 0u;
    public float TotalEmissivePower    => _as?.TotalEmissivePower    ?? 0f;

    // All three scene-invalidation signals restart integration: this is the one subscriber that
    // cares about every one of them, since any of the three changes the image being converged on.
    private readonly List<IDisposable> _subs = [];

    /// <summary>Subscribes the accumulator to the invalidation signals. Nothing extent-independent
    /// to build - the image pair is sized to the render target, so <see cref="Resize"/> allocates it
    /// (primed once at boot).</summary>
    public void Initialize()
    {
        var bus = Engine.EventBus;
        _subs.Add(bus.Subscribe<SceneDirtyEvent>(_ => MarkAccumulatorDirty()));
        _subs.Add(bus.Subscribe<SceneDataDirtyEvent>(_ => MarkAccumulatorDirty()));
        _subs.Add(bus.Subscribe<PathTracingAccumulatorInvalidatedEvent>(_ => MarkAccumulatorDirty()));
    }

    /// <summary>Reallocates the pair at the new extent, lays both out in General, and re-registers
    /// the FeaturePTIO bindings. The cores rebuild their graphs off the same pump straight after, so
    /// the handle a graph imports and the descriptor a shader binds move together.</summary>
    public void Resize(in HostTargets targets)
    {
        DisposeImages();

        var vk = Gfx.Vk;
        var extent = targets.Extent;

        // outColor needs TransferSrc so a core can blit it into FinalColor.
        _accumulator = new ImageResource(vk, Gfx.Device, "accumulator", Format.R32G32B32A32Sfloat,
            extent,
            ImageUsageFlags.StorageBit | ImageUsageFlags.SampledBit,
            ImageLayout.Undefined, ImageLayout.General);

        _outColor = new ImageResource(vk, Gfx.Device, "outColor", Format.R32G32B32A32Sfloat,
            extent,
            ImageUsageFlags.StorageBit | ImageUsageFlags.SampledBit | ImageUsageFlags.TransferSrcBit,
            ImageLayout.Undefined, ImageLayout.General);

        // These live outside any graph, so allocate explicitly. High residency priority: they are
        // the tracers' hot working set, touched every frame, and should stay resident ahead of cold
        // resources when the process is over its WDDM budget.
        _accumulator.Allocate(Gfx.PhysicalDevice, GpuMemoryAllocator.PriorityHigh);
        _outColor.Allocate(Gfx.PhysicalDevice, GpuMemoryAllocator.PriorityHigh);

        // Into General up front so the first dispatch can imageStore with no first-use branch.
        var cmd = Gfx.BeginSingleTimeCommands();
        Gfx.TransitionImageLayout(cmd, _accumulator.Image, _accumulator._format,
            ImageLayout.Undefined, ImageLayout.General);
        Gfx.TransitionImageLayout(cmd, _outColor.Image, _outColor._format,
            ImageLayout.Undefined, ImageLayout.General);
        Gfx.EndSingleTimeCommands(cmd);

        // The progressive-accumulation IO pair (FeaturePTIO, set 5), shared by all tracers.
        _gpu.Registry.RegisterImage("accumulator", _accumulator.ImageView, ImageLayout.General);
        _gpu.Registry.RegisterImage("outColor",    _outColor.ImageView,    ImageLayout.General);

        // Fresh images hold garbage, so any in-progress integration is invalid.
        _accumulatorDirty = true;
    }

    private void DisposeImages()
    {
        _accumulator?.Dispose();
        _outColor?.Dispose();
    }

    public void Dispose()
    {
        foreach (var s in _subs) s.Dispose();
        _subs.Clear();
        DisposeImages();
    }
}
