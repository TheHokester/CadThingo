using CadThingo.VulkanEngine.Renderer.FrameGraph;
using CadThingo.VulkanEngine.Renderer.Features.Tonemapping;
using CadThingo.VulkanEngine.Renderer.Pipelines;
using Silk.NET.Vulkan;
using HostRenderer = CadThingo.VulkanEngine.Renderer.Renderer;
using FrameContext = CadThingo.VulkanEngine.Renderer.Renderer.FrameContext;

namespace CadThingo.VulkanEngine.Renderer.Features.Deferred;

/// <summary>
/// The deferred technique as a composable graph module (render-graph.md sec 9; impl sec 1.8).
/// Builds the chain Cull -> LightCull -> Geometry -> Lighting -> Skybox -> Transparent into the
/// supplied <see cref="GraphScope"/>, owning the g-buffers / depth / HDR as transients, then
/// invokes a nested <see cref="TonemapModule"/> submodule (under the "Tonemap" child scope) that
/// consumes the HDR handle and writes FinalColor. Module-within-module: the submodule flattens
/// into the same graph, so the HDR read across the boundary is a derived cross-module dependency
/// (a blue edge in ToDot, inside a nested cluster).
///
/// The pipelines + kernels are NOT migrated yet: they remain renderer-owned. The tonemap
/// submodule is injected (it wraps the renderer's TonemapPipeline). The module is a pure graph
/// builder -- it records no descriptor writes and mutates no renderer state. The host calls
/// <see cref="Build"/>, then MarkOutput(<see cref="Outputs.Final"/>) + Compile, then does the
/// post-compile descriptor rebind from the returned handles (those views are cached on the
/// renderer and also feed the PT&lt;-&gt;deferred flip + tonemap-operator rebind, host-side).
/// </summary>
public sealed class DeferredModule : IGraphModule<DeferredModule.Inputs, DeferredModule.Outputs>
{
    /// <summary>External wiring: the host-owned FinalColor target (forwarded to the tonemap
    /// submodule) and the render extent the g-buffer / depth / HDR transients are sized to.</summary>
    public readonly record struct Inputs(ImageResource FinalColor, Extent2D Extent);

    /// <summary>Handles the host resolves after Compile: the five g-buffers + HDR feed the
    /// pre-baked descriptor sets (lighting samplers, tonemap HDR input); Final is the graph
    /// output to MarkOutput.</summary>
    public readonly record struct Outputs(
        GraphImage Position, GraphImage Normal, GraphImage Albedo, GraphImage Material,
        GraphImage Emissive, GraphImage Hdr, GraphImage Final);

    private readonly DrawCullPipeline     _cull;
    private readonly LightCullPipeline    _lightCull;
    private readonly GeometryPipeline     _geometry;
    private readonly PbrDeferredPipeline  _pbrDeferred;
    private readonly SkyboxPipeline       _skybox;
    private readonly TransparentPipeline  _transparent;
    private readonly TonemapModule        _tonemap;   // nested submodule (wraps TonemapPipeline)

    // Per-frame light-cull dispatch dims, read at Execute time. They are computed by the host
    // each frame (PbrDeferredPipeline.UpdatePerFrame, before the graph executes), so the pass
    // body pulls them through this delegate rather than capturing a stale value at Build.
    private readonly Func<(uint lightCount, uint tileCountX, uint tileCountY)> _lightCullParams;

    public DeferredModule(
        DrawCullPipeline cull, LightCullPipeline lightCull, GeometryPipeline geometry,
        PbrDeferredPipeline pbrDeferred, SkyboxPipeline skybox, TransparentPipeline transparent,
        TonemapModule tonemap,
        Func<(uint lightCount, uint tileCountX, uint tileCountY)> lightCullParams)
    {
        _cull = cull;
        _lightCull = lightCull;
        _geometry = geometry;
        _pbrDeferred = pbrDeferred;
        _skybox = skybox;
        _transparent = transparent;
        _tonemap = tonemap;
        _lightCullParams = lightCullParams;
    }

    public void Build(GraphScope scope, in Inputs inputs, out Outputs outputs)
    {
        var ext = inputs.Extent;

        // Deferred intermediates the graph OWNS as transients: it allocates their VkImages in
        // Compile() and frees them in Dispose(). Produced and consumed entirely within the
        // chain (geometry writes; lighting/skybox/transparent/tonemap read). Seeded Undefined
        // each frame (a legal discard -- every producing pass fully regenerates its target).
        GraphImage Color(string name, Format fmt) => scope.CreateImage(new ImageDesc
        {
            Format = fmt, Extent = ext, Mips = 1, Layers = 1,
            Usage = ImageUsageFlags.ColorAttachmentBit | ImageUsageFlags.SampledBit,
        }, name);

        var pos      = Color("GBuffer_Position", Format.R32G32B32A32Sfloat);
        var normal   = Color("GBuffer_Normal",   Format.R32G32B32A32Sfloat);
        var albedo   = Color("GBuffer_Albedo",   Format.R8G8B8A8Unorm);
        var material = Color("GBuffer_Material",  Format.R8G8B8A8Unorm);
        var emissive = Color("GBuffer_Emissive", Format.R16G16B16A16Sfloat);
        var hdr      = Color("HDRColor",          Format.R16G16B16A16Sfloat);
        var depth    = scope.CreateImage(new ImageDesc
        {
            Format = Format.D32Sfloat, Extent = ext, Mips = 1, Layers = 1,
            Usage = ImageUsageFlags.DepthStencilAttachmentBit,
        }, "Depth");

        // Per-frame compute-output buffers, imported so the graph derives the cull->geometry
        // and light-cull->lighting barriers + ordering. (The compute INPUTS -- renderables and
        // lights -- are host-coherent mapped writes, visible on submit, so they need no graph
        // barrier and aren't declared.) BufferDesc is unused for imports.
        var indirectCmdF   = new Buffer[HostRenderer.MAX_CONCURRENT_FRAMES];
        var indirectCountF = new Buffer[HostRenderer.MAX_CONCURRENT_FRAMES];
        var instanceF      = new Buffer[HostRenderer.MAX_CONCURRENT_FRAMES];
        var tileCountF     = new Buffer[HostRenderer.MAX_CONCURRENT_FRAMES];
        var tileIndicesF   = new Buffer[HostRenderer.MAX_CONCURRENT_FRAMES];
        for (uint i = 0; i < HostRenderer.MAX_CONCURRENT_FRAMES; i++)
        {
            indirectCmdF[i]   = _cull.GetIndirectCmdBuffer(i);
            indirectCountF[i] = _cull.GetIndirectCountBuffer(i);
            instanceF[i]      = Engine.ResourceManager.GetInstanceBuffer(i);
            tileCountF[i]     = _lightCull.GetTileLightCountBuffer(i);
            tileIndicesF[i]   = _lightCull.GetTileLightIndicesBuffer(i);
        }
        var indirectCmd   = scope.ImportBufferPerFrame(indirectCmdF,   default, "IndirectCmd");
        var indirectCount = scope.ImportBufferPerFrame(indirectCountF, default, "IndirectCount");
        var instance      = scope.ImportBufferPerFrame(instanceF,      default, "InstanceData");
        var tileCount     = scope.ImportBufferPerFrame(tileCountF,     default, "TileLightCount");
        var tileIndices   = scope.ImportBufferPerFrame(tileIndicesF,   default, "TileLightIndices");

        // Cull (compute): frustum-tests renderables -> post-cull indirect draw commands +
        // instance data. Its declared writes both keep it alive (writes imported buffers) and
        // order it before geometry (RAW on the indirect + instance buffers). Record() also runs
        // the view-dependent transparent sort consumed by TransparentPass.
        scope.AddPass("CullPass", PassType.Compute, QueueClass.Graphics,
            b =>
            {
                indirectCmd   = b.Write(indirectCmd,   ResourceUsage.StorageWriteCompute);
                indirectCount = b.Write(indirectCount, ResourceUsage.StorageWriteCompute);
                instance      = b.Write(instance,      ResourceUsage.StorageWriteCompute);
            },
            (CommandBuffer cmd, PassResources res, in FrameContext f) =>
                _cull.Record(cmd, f.FrameIndex, f.Camera));

        // Light-cull (compute): bins lights into the per-tile lists the lighting FS reads.
        // Tile/light counts are computed in DrawDeferred and read back via _lightCullParams.
        scope.AddPass("LightCullPass", PassType.Compute, QueueClass.Graphics,
            b =>
            {
                tileCount   = b.Write(tileCount,   ResourceUsage.StorageWriteCompute);
                tileIndices = b.Write(tileIndices, ResourceUsage.StorageWriteCompute);
            },
            (CommandBuffer cmd, PassResources res, in FrameContext f) =>
            {
                var (lightCount, tileCountX, tileCountY) = _lightCullParams();
                _lightCull.Record(cmd, f.FrameIndex, f.Camera, lightCount, tileCountX, tileCountY);
            });

        // Geometry -> g-buffers + depth. Reads the post-cull indirect buffers (IndirectArg) +
        // instance data (vertex storage read) -> RAW edges order it after CullPass.
        scope.AddPass("GeometryPass", PassType.Graphics, QueueClass.Graphics,
            b =>
            {
                b.Read(indirectCmd,   ResourceUsage.IndirectArg);
                b.Read(indirectCount, ResourceUsage.IndirectArg);
                b.Read(instance,      ResourceUsage.StorageReadVertex);
                pos      = b.Write(pos,      ResourceUsage.ColorAttachment);
                normal   = b.Write(normal,   ResourceUsage.ColorAttachment);
                albedo   = b.Write(albedo,   ResourceUsage.ColorAttachment);
                material = b.Write(material, ResourceUsage.ColorAttachment);
                emissive = b.Write(emissive, ResourceUsage.ColorAttachment);
                depth    = b.Write(depth,    ResourceUsage.DepthAttachment);
            },
            (CommandBuffer cmd, PassResources res, in FrameContext f) =>
            {
                var indirectCmdBuf   = _cull.GetIndirectCmdBuffer(f.FrameIndex);
                var indirectCountBuf = _cull.GetIndirectCountBuffer(f.FrameIndex);
                var drawCount        = _cull.LastRenderableCount;
                var attachments = new GeometryPipeline.Attachments(
                    res.View(pos), res.View(normal), res.View(albedo),
                    res.View(material), res.View(emissive), res.View(depth));
                _geometry.Record(cmd, f, indirectCmdBuf, indirectCountBuf, drawCount, attachments);
            });

        // Lighting samples the five g-buffers (ShaderReadOnly) and writes HDRColor@v1. It does
        // NOT bind depth -- the deferred lighting pass reconstructs position from the g-buffer,
        // so depth is left in DepthStencilAttachmentOptimal for the skybox/transparent depth
        // tests. Per-tile light lists from LightCullPass are read as StorageReadFragment -> RAW
        // edge orders lighting after light-cull.
        scope.AddPass("LightingPass", PassType.Graphics, QueueClass.Graphics,
            b =>
            {
                b.Read(pos,      ResourceUsage.SampledFragment);
                b.Read(normal,   ResourceUsage.SampledFragment);
                b.Read(albedo,   ResourceUsage.SampledFragment);
                b.Read(material, ResourceUsage.SampledFragment);
                b.Read(emissive, ResourceUsage.SampledFragment);
                b.Read(tileCount,   ResourceUsage.StorageReadFragment);
                b.Read(tileIndices, ResourceUsage.StorageReadFragment);
                hdr = b.Write(hdr, ResourceUsage.ColorAttachment);
            },
            (CommandBuffer cmd, PassResources res, in FrameContext f) =>
                _pbrDeferred.Record(cmd, f, res.View(hdr)));

        // Skybox / Transparent both LOAD HDRColor and depth-test (DepthWriteEnable=false)
        // against the geometry depth. Their CmdBeginRendering binds depth as a
        // DepthStencilAttachmentOptimal attachment, so depth is declared DepthAttachment here
        // (matching that layout). The depth WAW chain geometry->skybox->transparent and the
        // HDRColor version chain Lighting->Skybox->Transparent both fall out of the ledger.
        scope.AddPass("SkyboxPass", PassType.Graphics, QueueClass.Graphics,
            b =>
            {
                hdr   = b.Write(hdr,   ResourceUsage.ColorAttachment);
                depth = b.Write(depth, ResourceUsage.DepthAttachment);
            },
            (CommandBuffer cmd, PassResources res, in FrameContext f) =>
                _skybox.Record(cmd, f, new SkyboxPipeline.Attachments(res.View(hdr), res.View(depth))));

        scope.AddPass("TransparentPass", PassType.Graphics, QueueClass.Graphics,
            b =>
            {
                hdr   = b.Write(hdr,   ResourceUsage.ColorAttachment);
                depth = b.Write(depth, ResourceUsage.DepthAttachment);
            },
            (CommandBuffer cmd, PassResources res, in FrameContext f) =>
                _transparent.Record(cmd, f, _cull.LastTransparentDraws,
                    new TransparentPipeline.Attachments(res.View(hdr), res.View(depth))));

        // Tonemap is a nested submodule: it imports FinalColor and reads HDRColor@v3 (the
        // Transparent write), writing the final image. Built under the "Tonemap" child scope so
        // its pass nests inside this module's cluster and the HDR read is a cross-module edge.
        _tonemap.Build(scope.Child("Tonemap"),
            new TonemapModule.Input(hdr, inputs.FinalColor), out var tm);

        outputs = new Outputs(pos, normal, albedo, material, emissive, hdr, tm.FinalColor);
    }
}