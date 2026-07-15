using System.Numerics;
using CadThingo.VulkanEngine.Renderer.FrameGraph;
using CadThingo.VulkanEngine.Renderer.RenderCores;
using Silk.NET.Vulkan;

// The graph type shares its name with its namespace; alias so bare references never have to
// disambiguate type-vs-namespace (same trick as WavefrontPTCore).
using ReStirGraph = CadThingo.VulkanEngine.Renderer.FrameGraph.FrameGraph;

namespace CadThingo.VulkanEngine.Renderer.Features.ReSTIR;

/// <summary>
/// ReSTIR DI core (RenderMode.ReStirDI). GRAPH-RESIDENT (like DeferredCore / WavefrontPTCore, not
/// the hand-rolled PathTraceCoreBase): it owns a <see cref="ReStirGraph"/> built from
/// <see cref="ReStirDIModule"/> (Trace -> Tonemap), so the RT-pipeline dispatch is sequenced by the
/// graph's derived Sync2 barriers and FinalColor leaves the graph in ShaderReadOnly for the host
/// post-stack. This is what render-graph RT-pass support buys ReSTIR: the reuse phases each drop in
/// as additional graph nodes.
///
/// The graph is rebuilt in the ctor + on resize. The pipeline owns the descriptor sets (storage
/// images rebound here so the graph imports the fresh handles after a resize); this core owns only
/// the graph + the camera-snapshot / accumulator-reset bookkeeping. <see cref="Render"/> mirrors
/// the PT skeleton: camera-motion restart, per-frame light/material refresh, UpdatePerFrame, then
/// graph.Execute. Only constructed when the device exposes the RT pipeline (reStirPipeline != null).
/// </summary>
internal sealed class ReStirDICore : IRenderCore, IGraphCore
{
    private readonly Renderer         _host;
    private readonly ReStirDIPipeline _pipe;
    private ReStirGraph?              _graph;

    // Previous-frame camera snapshot -- any change restarts progressive integration (same scheme as
    // WavefrontPTCore). Identity/zero defaults force a restart on first activation.
    private Matrix4x4 _lastCamView = Matrix4x4.Identity;
    private Vector3   _lastCamPos  = Vector3.Zero;
    private float     _lastCamFov;

    public string Name => "ReSTIR DI (RT pipeline)";
    public Renderer.RenderMode Mode => Renderer.RenderMode.ReStirDI;

    public ReStirDICore(Renderer host)
    {
        _host = host;
        host.RegisterCore(this);   // cores add themselves to the host's render-core registry
        _pipe = host.reStirPipeline!;
        BuildGraph();
    }

    /// <summary>(Re)build the ReSTIR chain as the <see cref="ReStirGraph"/>. Imports the host
    /// accumulator / out-color / FinalColor (re-read here so a resize picks up the fresh handles).
    /// Called from the ctor + on every resize.</summary>
    private void BuildGraph()
    {
        _graph?.Dispose();
        var fg = new ReStirGraph(_host.gfx);

        var module = new ReStirDIModule(_pipe, _host.tonemapPipeline);
        module.Build(fg.RootScope().Child("ReStirDI"),
            new ReStirDIModule.Inputs(
                _host.renderTargets.PtAccumulator, _host.renderTargets.PtOutColor,
                _host.renderTargets.FinalColor),
            out var o);

        fg.MarkOutput(o.Final);
        fg.Compile();
        _graph = fg;

        // Hand the graph-owned working set (set 2, baked in Compile) to the pipeline for its binds.
        _pipe.SetGraphSharedSet(fg.GraphSharedSet);

        // CRITICAL (same reasoning as WavefrontPTCore): rebind the pipeline's storage-image
        // descriptors (set 0 bindings 4/5 = accumulator + out-color) to the SAME host handles the
        // graph just imported. On a resize RebuildRenderTargets reallocates these images before
        // Resize -> BuildGraph; without this the tracer would write through stale descriptors while
        // the graph's barriers + tonemap target the new images. (Also marks the accumulator dirty.)
        _pipe.WriteStorageImageDescriptors(
            _host.renderTargets.PtAccumulator.ImageView, _host.renderTargets.PtOutColor.ImageView);
    }

    /// <summary>Restart progressive integration on activation. Ensures clean start when swapping between
    /// different pathtraced modes</summary>
    public void Activate() => _pipe.MarkAccumulatorDirty();

    /// <summary>Resize: reallocate the per-pixel reservoir + G-buffer to the new extent, then
    /// rebuild the graph so it imports the freshly-reallocated PT/Final targets (BuildGraph also
    /// rebinds the pipeline's storage-image descriptors + marks the accumulator dirty).</summary>
    public void Resize(Extent2D extent)
    {
        _pipe.ReallocReStirBuffers(extent);
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

        // Per-frame SSBO refresh -- lights (UBO count + NEE) AND materials (so inspector edits land).
        uint lightCount = _host.UpdateLights(currentFrame, scene);
        _host.UpdateMaterials(currentFrame, scene);

        // Bridge the host's dirty signal into the pipeline's accumulator reset.
        if (_host.AccumulatorDirty)
        {
            _pipe.MarkAccumulatorDirty();
            _host.ClearAccumulatorDirty();
        }

        _pipe.UpdatePerFrame(currentFrame, camera!, lightCount, ctx.RenderExtent);

        // Record Trace (CmdTraceRays) -> Tonemap; every barrier derived from the usage table.
        _graph!.Execute(cmd, ctx);
    }

    // ---- IGraphCore surface (Stats panel + gfx-chunk submission) ------------
    public GraphStats? GraphStats => _graph?.Stats;
    public string ToDot() => _graph?.ToDot() ?? "(no ReSTIR frame graph)";
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
        _graph?.Dispose();
        _graph = null;
    }
}