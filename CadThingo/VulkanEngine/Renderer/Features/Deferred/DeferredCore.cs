using CadThingo.VulkanEngine.Renderer.Features.Tonemapping;
using CadThingo.VulkanEngine.Renderer.FrameGraph;
using CadThingo.VulkanEngine.Renderer.RenderCores;
using Silk.NET.Vulkan;

// The graph type shares its name with its namespace; alias it so bare `FrameGraph`
// never has to disambiguate type-vs-namespace.
using DeferredGraph = CadThingo.VulkanEngine.Renderer.FrameGraph.FrameGraph;

namespace CadThingo.VulkanEngine.Renderer.Features.Deferred;

/// <summary>
/// The deferred-shading technique as an <see cref="IRenderCore"/> (renderer-refactor.md L3).
/// Owns the deferred <see cref="DeferredGraph"/> (built from <see cref="DeferredModule"/> + a nested
/// <see cref="TonemapModule"/>) plus every piece of deferred-technique-local state that previously
/// lived on the <see cref="Renderer"/>: the graph's HDR / g-buffer transient views (cached for the
/// host-side descriptor rebinds) and the per-frame light-cull dimensions the LightCullPass delegate
/// reads.
///
/// <see cref="Render"/> is the former <c>DrawDeferred</c> body: per-frame CPU packing
/// (materials / lights / probes / extract) then <c>graph.Execute</c>, which records
/// cull -> light-cull -> geometry -> lighting -> skybox -> transparent -> tonemap and leaves
/// FinalColor in <c>ShaderReadOnlyOptimal</c> for the host post-stack. The technique pipelines stay
/// renderer-owned and are reached through <see cref="_host"/>; this core only owns the graph + the
/// cached handles + the build/rebind orchestration.
/// </summary>
internal sealed class DeferredCore : IRenderCore, IGraphCore
{
    private readonly Renderer _host;

    // The frame graph driving the deferred chain (render-graph.md Phase 1 / sec 1.8). It OWNS the
    // g-buffers / depth / HDR as transients (allocated in Compile, freed in Dispose) and imports
    // only FinalColor (RenderTargets-owned, consumed by the swapchain blit + ImGui viewport). All
    // sync is derived from the usage table. Rebuilt on resize.
    private DeferredGraph? _graph;

    // Cached view of the graph's HDR transient, refreshed on every BuildGraph (the transient is
    // reallocated on every Compile). Tonemap's HDR-input descriptor is bound to it in Activate;
    // the host's tonemap-operator rebuild reads it through Activate too.
    private ImageView _hdrView;

    // Cached views of the graph's 5 g-buffer transients (pos/normal/albedo/material/emissive),
    // refreshed on every BuildGraph. The lighting set is bound to them in BuildGraph;
    // OnPbrPipelineRebuilt rebinds from these onto the rebuilt pipeline's fresh descriptor set.
    private readonly ImageView[] _gBufferViews = new ImageView[5];

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

        // The deferred technique as a core module (render-graph.md sec 9) producing the HDR handle,
        // with tonemap as a NESTED submodule inside it (Features/Tonemapping) that consumes the HDR
        // and writes FinalColor -- so "Deferred" -> "Deferred/Tonemap" shows as nested clusters in
        // ToDot. Pipelines stay renderer-owned and are injected (the tonemap submodule wraps the
        // renderer's TonemapPipeline); the per-frame light-cull dims are pulled at Execute via the
        // delegate. Recreated each call so a resize picks up fresh refs + the new FinalColor handle.
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

        // Compile() just (re)allocated the g-buffer + HDR transients. (Re)bind the only graph
        // resources read through a PRE-BAKED descriptor set rather than a per-pass res.View: the
        // lighting pass's 5 g-buffer samplers and tonemap's HDR input. These cached views also feed
        // OnPbrPipelineRebuilt and Activate. (Geometry / skybox / transparent / tonemap reach their
        // attachments via res.View, which resolves the fresh handle automatically.)
        _gBufferViews[0] = fg.ResolveView(o.Position);
        _gBufferViews[1] = fg.ResolveView(o.Normal);
        _gBufferViews[2] = fg.ResolveView(o.Albedo);
        _gBufferViews[3] = fg.ResolveView(o.Material);
        _gBufferViews[4] = fg.ResolveView(o.Emissive);
        _host.PbrDeferredPipeline.WriteGBufferDescriptors(
            _gBufferViews[0], _gBufferViews[1], _gBufferViews[2], _gBufferViews[3], _gBufferViews[4]);
        _hdrView = fg.ResolveView(o.Hdr);
        _host.tonemapPipeline.WriteHdrInputDescriptor(_hdrView, _host.gBufferSampler);
    }

    /// <summary>Re-point tonemap's shared HDR-input descriptor at this core's HDR transient. Called
    /// by the host on mode switch / after a tonemap rebuild -- replaces the per-frame _lastRenderMode
    /// rebind.</summary>
    public void Activate() =>
        _host.tonemapPipeline.WriteHdrInputDescriptor(_hdrView, _host.gBufferSampler);

    /// <summary>Re-issues the lighting pass's g-buffer descriptor writes from the cached transient
    /// views onto the rebuilt PBR pipeline's fresh descriptor set. The host calls this inside
    /// RebuildPbrPipelines (the only deferred-specific part of that cross-pipeline rewire).</summary>
    public void OnPbrPipelineRebuilt() =>
        _host.PbrDeferredPipeline.WriteGBufferDescriptors(
            _gBufferViews[0], _gBufferViews[1], _gBufferViews[2], _gBufferViews[3], _gBufferViews[4]);

    public void Resize(Extent2D extent) => BuildGraph();

    public void Render(in RenderFrame frame)
    {
        var cmd   = frame.Cmd;
        var ctx   = frame.Frame;
        var camera = _host.Camera;
        var scene  = _host.Scene;
        uint currentFrame = ctx.FrameIndex;

        // Reflection-probe scheduler bookkeeping. Cheap CPU-only walk; runs before the geometry
        // pass so a captured cube is visible to downstream shaders within the same frame once
        // it's hooked up.
        _host.reflectionProbeSystem.Tick(_host.frameCounter, scene);

        // Per-frame material SSBO snapshot — needs to land before the geometry pass reads it.
        _host.UpdateMaterials(currentFrame, scene);

        _host.geometryPipeline.UpdateUbo(currentFrame, camera);
        var (lightCount, tileCountX, tileCountY) =
            _host.PbrDeferredPipeline.UpdatePerFrame(currentFrame, camera, scene);
        _host.transparentPipeline.UpdatePerFrame(currentFrame, camera, lightCount, tileCountX, tileCountY);
        // Stash for the module's LightCullPass body (it runs inside the graph, below).
        _frameLightCount = lightCount; _frameTileCountX = tileCountX; _frameTileCountY = tileCountY;
        // Skybox always updates — pass body is conditional on EditorState.SkyboxEnabled so we can
        // flip the toggle without re-recording the graph.
        _host.skyboxPipeline.UpdatePerFrame(currentFrame, camera, ImGui.EditorState.SkyboxIntensity);

        // Reflection-probe cluster cull. Tile-only today (zSlices=1). Cheap CPU work.
        float aspect = (float)ctx.RenderExtent.Width / ctx.RenderExtent.Height;
        _host.reflectionProbeSystem.BuildClusters(currentFrame, camera, aspect, 0.1f, 100f,
            tileCountX, tileCountY);
        // Refresh the per-probe SSBO read by the PBR lighting shader.
        _host.reflectionProbeSystem.WriteProbeRecords(currentFrame);

        // Reflection-probe capture for the next dirty probe (if any), bounded to one probe/frame.
        _host.reflectionProbeSystem.RecordCapture(cmd, currentFrame, _host.frameCounter, scene);

        // Scene extraction (L2 step 4) — the single ECS walk that packs the cull-input SSBO +
        // classifies BLEND candidates. Must precede the cull pass. (The host-coherent write is
        // visible to the cull dispatch on submit.)
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