using System.Numerics;
using CadThingo.VulkanEngine.Renderer.FeatureLifecycle.RenderFeatureInterfaces;
using CadThingo.VulkanEngine.Renderer.Features.Shared;
using CadThingo.VulkanEngine.Renderer.FrameGraph;
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
/// Subclasses bind the concrete pipeline via the protected hooks; the base resolves that pipeline
/// and builds the graph in <see cref="Initialize"/>, once the host is wired.
/// </summary>
internal abstract class PathTraceCoreBase
    : IRenderCore, IGraphCore, INeedsGpu, INeedsHost, INeedsFeature<ISharedPipelines>
{
    // Each PT core builds and owns its own tracer pipeline through this.
    protected GpuContext _gpu;
    GpuContext INeedsGpu.Gpu { set => _gpu = value; }

    // The tonemap this core's graph ends with. One instance, shared with every other core - which
    // is exactly why it arrives as a collaborator rather than being constructed here.
    protected ISharedPipelines _shared = null!;
    ISharedPipelines INeedsFeature<ISharedPipelines>.Dependency { set => _shared = value; }

    // Transitional: the accumulator / out-color pair is still renderer-owned.
    protected Renderer _host = null!;
    Renderer INeedsHost.Host { set => _host = value; }

    // Last snapshot the host handed over, re-read on every Resize.
    protected HostTargets _targets;

    private PTGraph? _graph;

    // Previous-frame camera snapshot. Any change restarts progressive integration. Identity/zero
    // defaults mean the first frame after switching to this core always restarts (intended).
    private Matrix4x4 _lastCamView = Matrix4x4.Identity;
    private Vector3   _lastCamPos  = Vector3.Zero;
    // FOV doesn't move the view matrix, so it needs its own snapshot.
    private float     _lastCamFov;

    public abstract string Name { get; }
    public abstract Renderer.RenderMode Mode { get; }

    /// <summary>Builds the subclass's tracer pipeline. Split off the ctor so the pipeline field is
    /// assigned exactly once, after wiring. The graph imports the extent-sized targets, so it is
    /// built in <see cref="Resize"/>, which the host primes once at boot.</summary>
    public void Initialize()
    {
        CreatePipeline();
    }

    // ---- Pipeline-specific hooks (forwarded to PTComputePipeline / RTPipeline) ------------------
    /// <summary>Constructs and initializes this core's tracer pipeline. It is technique-private -
    /// nothing outside this core records with it - so the core owns it outright rather than the
    /// host building it and handing it over.</summary>
    protected abstract void CreatePipeline();
    /// <summary>Frees whatever <see cref="CreatePipeline"/> built. Called from the base Dispose so
    /// every subclass tears down in the same order relative to its graph.</summary>
    protected abstract void DestroyPipeline();
    /// <summary>Compute (ray query) vs RayTrace (CmdTraceRays) -- selects the Trace pass type and
    /// the stage the module's derived barriers target.</summary>
    protected abstract PassType TracePassType { get; }
    protected abstract void PipelineMarkAccumulatorDirty();
    protected abstract bool PipelineUpdatePerFrame(uint frameIndex, Camera camera, uint lightCount, Extent2D renderExtent);
    protected abstract void PipelineRecord(CommandBuffer cmd, in RenderView ctx);

    /// <summary>(Re)build the Trace -> Tonemap graph. Imports the host accumulator / out-color /
    /// FinalColor (re-read here so a resize picks up the fresh handles), then rebinds the
    /// pipeline's storage-image descriptors to the SAME handles the graph just imported (which
    /// also marks the accumulator dirty). Called on every resize.</summary>
    protected void BuildGraph()
    {
        _graph?.Dispose();
        var fg = new PTGraph(_host.gfx);

        var module = new PathTraceModule(_shared.Tonemap, TracePassType,
            (CommandBuffer cmd, PassResources res, in RenderView f) => PipelineRecord(cmd, f));
        module.Build(fg.RootScope().Child("PathTrace"),
            new PathTraceModule.Inputs(
                _host.renderTargets.PtAccumulator, _host.renderTargets.PtOutColor,
                _targets.FinalColor),
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
        uint currentFrame = view.FrameIndex;

        // Camera motion -> restart accumulation. Cheap structural-equality check against last
        // frame's snapshot. If Camera ever grows a built-in dirty flag, swap this out for that.
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

        // Lights + materials were packed by the draw loop's extraction (which is what makes
        // inspector edits land in PT mode); the count only feeds the pipeline's frame UBO.
        uint lightCount = view.LightCount;

        // Bridge the host's dirty signal into the pipeline's accumulator reset.
        if (_host.AccumulatorDirty)
        {
            PipelineMarkAccumulatorDirty();
            _host.ClearAccumulatorDirty();
        }

        PipelineUpdatePerFrame(currentFrame, camera!, lightCount, view.RenderExtent);

        // Record Trace -> Tonemap; every barrier derived from the usage table.
        _graph!.Execute(cmd, view);
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
        // Graph first: its passes hold references to the pipeline they record with.
        _graph?.Dispose();
        _graph = null;
        DestroyPipeline();
    }
}