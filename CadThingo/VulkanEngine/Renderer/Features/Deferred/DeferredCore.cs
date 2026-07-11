using CadThingo.VulkanEngine.Renderer.Features.Tonemapping;
using CadThingo.VulkanEngine.Renderer.FrameGraph;
using CadThingo.VulkanEngine.Renderer.RenderCores;
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
/// FinalColor in <c>ShaderReadOnlyOptimal</c> for the host. The pipelines are
/// renderer-owned. 
/// </summary>
internal sealed class DeferredCore : IRenderCore, IGraphCore
{
    private readonly Renderer _host;

    // The frame graph driving the deferred chain
    private DeferredGraph? _graph;

    // Per-frame light-cull dispatch dims, written by Render (from PbrDeferredPipeline.UpdatePerFrame)
    // before Execute and pulled by the module's LightCullPass delegate at Execute time.
    private uint _frameLightCount;
    private uint _frameTileCountX;
    private uint _frameTileCountY;

    public string Name => "Deferred";
    public Renderer.RenderMode Mode => Renderer.RenderMode.Deferred;

    public DeferredCore(Renderer host)
    {
        _host = host;
        host.RegisterCore(this);   // cores add themselves to the host's render-core registry
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

        // The deferred technique taking an imported tonemap module, 
        var tonemap = new TonemapModule(_host.tonemapPipeline);
        var module = new DeferredModule(
            _host.drawCullPipeline, _host.lightCullPipeline, _host.geometryPipeline,
            _host.PbrDeferredPipeline, _host.skyboxPipeline, _host.transparentPipeline, tonemap,
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

    /// <summary>Rebuilds the deferred graph after an in-place PBR pipeline rebuild. Rebuild()
    /// recreates the pipeline's set-1 layout (new handle), so the graph's baked g-buffer set -
    /// allocated from the old layout - must be re-baked against the new one. The host calls this
    /// inside RebuildPbrPipelines, which has already idled the device.</summary>
    public void OnPbrPipelineRebuilt() => BuildGraph();

    public void Resize(Extent2D extent) => BuildGraph();

    public void Render(in RenderFrame frame)
    {
        var cmd   = frame.Cmd;
        var ctx   = frame.Frame;
        var camera = _host.Camera;
        var scene  = _host.Scene;
        uint currentFrame = ctx.FrameIndex;

        // Reflection-probe scheduler bookkeeping.
        _host.reflectionProbeSystem.Tick(_host.frameCounter, scene);

        // Per-frame material SSBO snapshot
        _host.UpdateMaterials(currentFrame, scene);

        // Geometry's view+proj now rides the scene set's (0,0) arena slot, pushed inside
        // its Record call - no per-frame UBO update needed here anymore.
        var (lightCount, tileCountX, tileCountY) =
            _host.PbrDeferredPipeline.UpdatePerFrame(currentFrame, camera, scene);
        _host.transparentPipeline.UpdatePerFrame(currentFrame, camera, lightCount, tileCountX, tileCountY);
        // Stash for the module's LightCullPass body (it runs inside the graph, below).
        _frameLightCount = lightCount; _frameTileCountX = tileCountX; _frameTileCountY = tileCountY;
        // Skybox always updates
        _host.skyboxPipeline.UpdatePerFrame(currentFrame, camera, ImGui.EditorState.SkyboxIntensity);

        // Reflection-probe cluster cull. Tile-only today (zSlices=1). Cheap CPU work.
        float aspect = (float)ctx.RenderExtent.Width / ctx.RenderExtent.Height;
        _host.reflectionProbeSystem.BuildClusters(currentFrame, camera, aspect, 0.1f, 100f,
            tileCountX, tileCountY);
        // Refresh the per-probe SSBO read by the PBR lighting shader.
        _host.reflectionProbeSystem.WriteProbeRecords(currentFrame);

        // Reflection-probe capture for the next dirty probe (if any), bounded to one probe/frame.
        _host.reflectionProbeSystem.RecordCapture(cmd, currentFrame, _host.frameCounter, scene);

        // Scene extraction
        _host.gpuScene.ExtractRenderables(currentFrame, scene);

        // Record the deferred chain via the graph: cull -> light-cull -> geometry -> lighting ->
        // skybox -> transparent -> tonemap. Cull/light-cull run as compute passes and every barrier
        // is derived from the usage table.
        _graph!.Execute(cmd, ctx);
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
        _graph?.Dispose();
        _graph = null;
    }
}