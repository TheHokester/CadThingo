using System.Numerics;
using CadThingo.VulkanEngine.Renderer.FrameGraph;
using CadThingo.VulkanEngine.Renderer.RenderCores;
using Silk.NET.Vulkan;

// The graph type shares its name with its namespace; alias so bare references never have to
// disambiguate type-vs-namespace (same trick as WavefrontPTCore).
using PTGraph = CadThingo.VulkanEngine.Renderer.FrameGraph.FrameGraph;

namespace CadThingo.VulkanEngine.Renderer.Features.PathTracer;

/// <summary>
/// Shared base for the two progressive megakernel path-trace cores: the compute (ray-query) path
/// and the RT-pipeline path. GRAPH-RESIDENT like WavefrontPTCore / ReStirDICore: owns a
/// <see cref="PTGraph"/> built from <see cref="PathTraceModule"/> (Trace -> Tonemap), so the
/// barriers and the tonemap HDR-input set are graph-derived / graph-baked instead of hand-rolled.
/// Subclasses bind the concrete pipeline via the protected hooks and call <see cref="BuildGraph"/>
/// at the end of their ctor (after their pipeline field is assigned).
/// </summary>
internal abstract class PathTraceCoreBase : IRenderCore, IGraphCore
{
    protected readonly Renderer _host;
    private PTGraph? _graph;

    // Previous-frame camera snapshot. Any change restarts progressive integration. Identity/zero
    // defaults mean the first frame after switching to this core always restarts (intended).
    private Matrix4x4 _lastCamView = Matrix4x4.Identity;
    private Vector3   _lastCamPos  = Vector3.Zero;
    // FOV doesn't move the view matrix, so it needs its own snapshot.
    private float     _lastCamFov;

    protected PathTraceCoreBase(Renderer host)
    {
        _host = host;
        host.RegisterCore(this);   // cores add themselves to the host's render-core registry
    }

    public abstract string Name { get; }
    public abstract Renderer.RenderMode Mode { get; }

    // ---- Pipeline-specific hooks (forwarded to PTComputePipeline / RTPipeline) ------------------
    /// <summary>Compute (ray query) vs RayTrace (CmdTraceRays) -- selects the Trace pass type and
    /// the stage the module's derived barriers target.</summary>
    protected abstract PassType TracePassType { get; }
    protected abstract void PipelineMarkAccumulatorDirty();
    protected abstract bool PipelineUpdatePerFrame(uint frameIndex, Camera camera, uint lightCount, Extent2D renderExtent);
    protected abstract void PipelineRecord(CommandBuffer cmd, in Renderer.FrameContext ctx);

    /// <summary>(Re)build the Trace -> Tonemap graph. Imports the host accumulator / out-color /
    /// FinalColor (re-read here so a resize picks up the fresh handles), then rebinds the
    /// pipeline's storage-image descriptors to the SAME handles the graph just imported (which
    /// also marks the accumulator dirty). Called from the subclass ctor + on every resize.</summary>
    protected void BuildGraph()
    {
        _graph?.Dispose();
        var fg = new PTGraph(_host.gfx);

        var module = new PathTraceModule(_host.tonemapPipeline, TracePassType,
            (CommandBuffer cmd, PassResources res, in Renderer.FrameContext f) => PipelineRecord(cmd, f));
        module.Build(fg.RootScope().Child("PathTrace"),
            new PathTraceModule.Inputs(
                _host.renderTargets.PtAccumulator, _host.renderTargets.PtOutColor,
                _host.renderTargets.FinalColor),
            out var o);

        fg.MarkOutput(o.Final);
        fg.Compile();
        _graph = fg;

        // The storage-image descriptors themselves are registry-owned (FeaturePTIO) and re-registered
        // by the host on resize; all that is left here is dropping the sample count, since fresh
        // images make any in-progress accumulation invalid by construction.
        PipelineMarkAccumulatorDirty();
    }

    /// <summary>Restart progressive integration on activation. Tonemap's HDR input is graph-baked
    /// by this core's composed TonemapModule from the imported PtOutColor, so no descriptor rebind
    /// is needed. The accumulator is SHARED with the other PT cores, so resetting here --
    /// deterministically, after the host's DeviceWaitIdle -- guarantees a fresh start instead of
    /// `+=`-ing onto a stale image.</summary>
    public void Activate() => PipelineMarkAccumulatorDirty();

    /// <summary>Resize: rebuild the graph so it imports the freshly-reallocated PT/Final targets
    /// (BuildGraph also rebinds the pipeline's storage-image descriptors + marks the accumulator
    /// dirty).</summary>
    public void Resize(Extent2D extent) => BuildGraph();

    public void Render(in RenderFrame frame)
    {
        var cmd    = frame.Cmd;
        var ctx    = frame.Frame;
        var camera = _host.Camera;
        var scene  = _host.Scene;
        uint currentFrame = ctx.FrameIndex;

        // Camera motion -> restart accumulation. Cheap structural-equality check against last
        // frame's snapshot. If Camera ever grows a built-in dirty flag, swap this out for that.
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

        // Refresh per-frame SSBOs -- lights AND materials (so inspector edits land in PT mode).
        uint lightCount = _host.UpdateLights(currentFrame, scene);
        _host.UpdateMaterials(currentFrame, scene);

        // Bridge the host's dirty signal into the pipeline's accumulator reset.
        if (_host.AccumulatorDirty)
        {
            PipelineMarkAccumulatorDirty();
            _host.ClearAccumulatorDirty();
        }

        PipelineUpdatePerFrame(currentFrame, camera!, lightCount, ctx.RenderExtent);

        // Record Trace -> Tonemap; every barrier derived from the usage table.
        _graph!.Execute(cmd, ctx);
    }

    // ---- IGraphCore surface (Stats panel + gfx-chunk submission) ------------
    public GraphStats? GraphStats => _graph?.Stats;
    public string ToDot() => _graph?.ToDot() ?? "(no pathtrace frame graph)";
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