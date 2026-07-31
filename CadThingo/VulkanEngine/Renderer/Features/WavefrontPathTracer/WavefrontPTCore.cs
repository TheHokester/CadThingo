using System.Numerics;
using System.Runtime.CompilerServices;
using CadThingo.VulkanEngine.Renderer.FeatureLifecycle;
using CadThingo.VulkanEngine.Renderer.FeatureLifecycle.RenderFeatureInterfaces;
using CadThingo.VulkanEngine.Renderer.Features.Shared;
using CadThingo.VulkanEngine.Renderer.FrameGraph;
using Silk.NET.Vulkan;

// The graph type shares its name with its namespace; alias so bare `FrameGraph`
// never has to disambiguate type-vs-namespace.
using WavefrontGraph = CadThingo.VulkanEngine.Renderer.FrameGraph.FrameGraph;

namespace CadThingo.VulkanEngine.Renderer.Features.WavefrontPathTracer;

/// <summary>
/// The wavefront path tracer as an <see cref="IRenderCore"/> (RenderMode.RayWavefront).
/// Unlike the megakernel cores (PathTraceCoreBase, which hand-roll their barriers + tonemap),
/// this core is graph-resident like <c>DeferredCore</c>: it owns a <see cref="WavefrontGraph"/>
/// built from <see cref="WavefrontPTModule"/> (Generate -> bounces -> Finalize -> Tonemap), so
/// every barrier is derived from the module's usage declarations and FinalColor leaves the graph
/// in ShaderReadOnly for the host post-stack.
///
/// The graph is rebuilt in the ctor + on resize. The pipeline owns the set-4 SoA buffers (resized
/// before the rebuild so the graph imports the fresh handles); this core owns only the graph + the
/// camera-snapshot / accumulator-reset bookkeeping. <see cref="Render"/> mirrors the PT skeleton:
/// camera-motion restart, per-frame light/material refresh, UpdatePerFrame, then graph.Execute.
/// </summary>
internal sealed class WavefrontPTCore : IRenderCore, IGraphCore,
                                        ISelfRegisteringFeature<WavefrontPTCore>, INeedsGpu, INeedsHost,
                                        INeedsFeature<ISharedPipelines>
{
    public static FeatureDesc Desc =>
        new(Order: 50, Gate: _ => true, Make: () => new WavefrontPTCore());

    [ModuleInitializer]
    internal static void _Reg() => FeatureCatalog.Register<WavefrontPTCore>();

    private GpuContext _gpu;
    GpuContext INeedsGpu.Gpu { set => _gpu = value; }

    // The tonemap this core's graph ends with - one shared instance, injected not constructed.
    private ISharedPipelines _shared = null!;
    ISharedPipelines INeedsFeature<ISharedPipelines>.Dependency { set => _shared = value; }

    // Transitional: the PT render targets are still renderer-owned.
    private Renderer _host = null!;
    Renderer INeedsHost.Host { set => _host = value; }

    private WavefrontPTPipeline _pipe = null!;
    private WavefrontGraph?     _graph;

    // Previous-frame camera snapshot -- any change restarts progressive integration (same scheme
    // as PathTraceCoreBase). Identity/zero defaults force a restart on first activation.
    private Matrix4x4 _lastCamView = Matrix4x4.Identity;
    private Vector3   _lastCamPos  = Vector3.Zero;
    private float     _lastCamFov;

    public string Name => "PathTrace (Wavefront)";
    public Renderer.RenderMode Mode => Renderer.RenderMode.RayWavefront;

    /// <summary>Exposed only for the settings panel + the stats readback, which read its counters.</summary>
    internal WavefrontPTPipeline Pipeline => _pipe;

    public void Initialize()
    {
        // Shares the accumulator / out-color images with the megakernel via FeaturePTIO and the
        // scene buffers via the scene set; the SoA working set is pipeline-owned. Technique-private,
        // so this core builds it rather than the host.
        _pipe = new WavefrontPTPipeline(_gpu, _host);
        _pipe.Initialize();
        BuildGraph();
    }

    /// <summary>(Re)build the wavefront chain as the <see cref="WavefrontGraph"/>. Imports the
    /// pipeline-owned set-4 buffers + the host accumulator / out-color / FinalColor (re-read here
    /// so a resize picks up the fresh handles). Called from the ctor + on every resize.</summary>
    private void BuildGraph()
    {
        _graph?.Dispose();
        var fg = new WavefrontGraph(_host.gfx);

        var module = new WavefrontPTModule(_pipe, _shared.Tonemap);
        module.Build(fg.RootScope().Child("Wavefront"),
            new WavefrontPTModule.Inputs(
                _host.renderTargets.PtAccumulator, _host.renderTargets.PtOutColor,
                _host.renderTargets.FinalColor),
            out var o);

        fg.MarkOutput(o.Final);
        fg.Compile();
        _graph = fg;

        // Hand the graph-owned SoA working set (set 2, baked in Compile from the pipeline's current
        // buffer handles) back to the pipeline for its record-time binds.
        _pipe.SetGraphSharedSet(fg.GraphSharedSet);

        // accumulator + out-color are registry-owned (FeaturePTIO), re-registered by the host as soon
        // as RebuildRenderTargets reallocates them - which is what keeps the bound descriptor and the
        // handle the graph imported in lockstep. Getting that wrong previously caused the bars /
        // background / firefly corruption after a render-extent change. All that is left here is
        // dropping the sample count, since the fresh images invalidate any accumulation.
        _pipe.MarkAccumulatorDirty();
    }

    /// <summary>On activation marks accumulator dirty to reset any potentially high sample count
    ///work done by other pathtraced cores.</summary>
    public void Activate() => _pipe.MarkAccumulatorDirty();

    /// <summary>Resize: reallocate the SoA working set to the new extent (which rewrites set 4 +
    /// marks the accumulator dirty), then rebuild the graph so it imports the fresh buffers + the
    /// freshly-reallocated PT/Final targets.</summary>
    public void Resize(Extent2D extent)
    {
        _pipe.ReallocSet4(extent);
        BuildGraph();
    }

    public void Render(in RenderFrame frame)
    {
        var cmd    = frame.Cmd;
        var view   = frame.View;
        var camera = view.Camera;
        uint currentFrame = view.FrameIndex;

        // Camera motion -> restart accumulation (structural compare against last frame's snapshot).
        if (camera != null)
        {
            var camView = camera.GetViewMatrix();
            var pos  = camera.GetPosition();
            var fov  = camera.Fov;
            if (camView != _lastCamView || pos != _lastCamPos || fov != _lastCamFov)
            {
                _host.MarkAccumulatorDirty();
                _lastCamView = camView; _lastCamPos = pos; _lastCamFov = fov;
            }
        }

        // Lights + materials were packed by the draw loop's extraction (resolveHit reads materials
        // and textures every bounce, so inspector edits land through that); the count only feeds
        // the pipeline's frame UBO.
        uint lightCount = view.LightCount;

        // Bridge the host's dirty signal into the pipeline's accumulator reset.
        if (_host.AccumulatorDirty)
        {
            _pipe.MarkAccumulatorDirty();
            _host.ClearAccumulatorDirty();
        }

        _pipe.UpdatePerFrame(currentFrame, camera!, lightCount, view.RenderExtent);

        // Record Generate -> (PrepExtend->Extend->PrepShade->Shade->PrepConnect->Connect) x8 ->
        // Finalize -> Tonemap; every barrier derived from the usage table.
        _graph!.Execute(cmd, view);

        // TEMP/debug: copy the per-bounce indirect args to the readback staging so the Stats panel
        // can show compaction in-app. Remove with the pipeline's _argsReadback feature.
        _pipe.RecordArgsReadback(cmd);
    }

    // ---- IGraphCore surface (Stats panel + gfx-chunk submission) ------------
    public GraphStats? GraphStats => _graph?.Stats;
    public string ToDot() => _graph?.ToDot() ?? "(no wavefront frame graph)";
    public bool CollectPipelineStats
    {
        get => _graph?.CollectPipelineStats ?? false;
        set { if (_graph != null) _graph.CollectPipelineStats = value; }
    }
    public bool HasPendingGfxChunks => _graph?.HasPendingGfxChunks ?? false;
    public void SubmitGfxChunks(Queue gfxQueue, SemaphoreSubmitInfo imgAvailWait,
        SemaphoreSubmitInfo renderDoneSignal, CommandBuffer hostCmd, Fence fence)
        => _graph!.SubmitGfxChunks(gfxQueue, imgAvailWait, renderDoneSignal, hostCmd, fence);

    public void Dispose()
    {
        // Graph first: its passes hold a reference to the pipeline they record with.
        _graph?.Dispose();
        _graph = null;
        _pipe?.Dispose();
    }
}