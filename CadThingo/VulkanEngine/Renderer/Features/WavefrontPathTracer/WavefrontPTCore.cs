using System.Numerics;
using CadThingo.VulkanEngine.Renderer.FrameGraph;
using CadThingo.VulkanEngine.Renderer.RenderCores;
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
internal sealed class WavefrontPTCore : IRenderCore, IGraphCore
{
    private readonly Renderer            _host;
    private readonly WavefrontPTPipeline _pipe;
    private WavefrontGraph?              _graph;

    // Previous-frame camera snapshot -- any change restarts progressive integration (same scheme
    // as PathTraceCoreBase). Identity/zero defaults force a restart on first activation.
    private Matrix4x4 _lastCamView = Matrix4x4.Identity;
    private Vector3   _lastCamPos  = Vector3.Zero;
    private float     _lastCamFov;

    public string Name => "PathTrace (Wavefront)";
    public Renderer.RenderMode Mode => Renderer.RenderMode.RayWavefront;

    public WavefrontPTCore(Renderer host)
    {
        _host = host;
        _pipe = host.wavefrontPipeline;
        BuildGraph();
    }

    /// <summary>(Re)build the wavefront chain as the <see cref="WavefrontGraph"/>. Imports the
    /// pipeline-owned set-4 buffers + the host accumulator / out-color / FinalColor (re-read here
    /// so a resize picks up the fresh handles). Called from the ctor + on every resize.</summary>
    private void BuildGraph()
    {
        _graph?.Dispose();
        var fg = new WavefrontGraph(_host.gfx);

        var module = new WavefrontPTModule(_pipe, _host.tonemapPipeline);
        module.Build(fg.RootScope().Child("Wavefront"),
            new WavefrontPTModule.Inputs(
                _host.renderTargets.PtAccumulator, _host.renderTargets.PtOutColor,
                _host.renderTargets.FinalColor),
            out var o);

        fg.MarkOutput(o.Final);
        fg.Compile();
        _graph = fg;

        // Set 4 is bound through stable pipeline-owned handles (already written at init / resize);
        // re-bind from those handles after compile to match the imports the graph just adopted.
        _pipe.WriteSet4Descriptors();

        // CRITICAL: rebind set 0 bindings 4/5 (accumulator + out-color storage images) to the SAME
        // host-owned handles the graph just imported. On a resize RebuildRenderTargets reallocates
        // these images (fresh VkImageViews) before calling Resize -> BuildGraph; without this rebind
        // the kernels would keep writing through the stale (freed) descriptors while the graph's
        // barriers + tonemap target the new images -- the source of the bars / background / firefly
        // corruption after a render-extent change. Tying the rebind to BuildGraph keeps the bound
        // descriptor and the imported handle in lockstep. (Also marks the accumulator dirty.)
        _pipe.WriteStorageImageDescriptors(
            _host.renderTargets.PtAccumulator.ImageView, _host.renderTargets.PtOutColor.ImageView);
    }

    /// <summary>Re-point tonemap's shared HDR-input descriptor at PtOutColor (the wavefront
    /// scene-colour image the graph's Tonemap pass reads), and restart progressive integration.
    /// Called by the host on mode switch / after a tonemap rebuild. The accumulator is SHARED with
    /// the megakernel PT cores, so on a switch into this core it still holds the previous core's
    /// (possibly high-sample-count) result; resetting here -- deterministically, after the host's
    /// DeviceWaitIdle -- guarantees a fresh start instead of `+=`-ing onto a stale image, rather
    /// than relying on the host accumulator-dirty flag arriving a frame later.</summary>
    public void Activate()
    {
        _host.tonemapPipeline.WriteHdrInputDescriptor(
            _host.renderTargets.PtOutColor.ImageView, _host.gBufferSampler);
        _pipe.MarkAccumulatorDirty();
    }

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
        var ctx    = frame.Frame;
        var camera = _host.Camera;
        var scene  = _host.Scene;
        uint currentFrame = ctx.FrameIndex;

        // Camera motion -> restart accumulation (structural compare against last frame's snapshot).
        if (camera != null)
        {
            var view = camera.GetViewMatrix();
            var pos  = camera.GetPosition();
            var fov  = camera.Fov;
            if (view != _lastCamView || pos != _lastCamPos || fov != _lastCamFov)
            {
                _host.MarkAccumulatorDirty();
                _lastCamView = view; _lastCamPos = pos; _lastCamFov = fov;
            }
        }

        // Per-frame SSBO refresh -- lights (for the UBO count + future NEE) AND materials (so
        // inspector edits land; resolveHit reads materials/textures every bounce).
        uint lightCount = _host.UpdateLights(currentFrame, scene);
        _host.UpdateMaterials(currentFrame, scene);

        // Bridge the host's dirty signal into the pipeline's accumulator reset.
        if (_host.AccumulatorDirty)
        {
            _pipe.MarkAccumulatorDirty();
            _host.ClearAccumulatorDirty();
        }

        _pipe.UpdatePerFrame(currentFrame, camera!, lightCount, ctx.RenderExtent);

        // Record Generate -> (PrepExtend->Extend->PrepShade->Shade->PrepConnect->Connect) x8 ->
        // Finalize -> Tonemap; every barrier derived from the usage table.
        _graph!.Execute(cmd, ctx);

        // TEMP/debug: copy the per-bounce indirect args to the readback staging so the Stats panel
        // can show compaction in-app. Remove with the pipeline's _argsReadback feature.
        _pipe.RecordArgsReadback(cmd);
    }

    // ---- Debug surface (read by the Stats panel) -----------------------------
    public GraphStats? GraphStats => _graph?.Stats;
    public string ToDot() => _graph?.ToDot() ?? "(no wavefront frame graph)";
    public bool CollectPipelineStats
    {
        get => _graph?.CollectPipelineStats ?? false;
        set { if (_graph != null) _graph.CollectPipelineStats = value; }
    }

    public void Dispose()
    {
        _graph?.Dispose();
        _graph = null;
    }
}