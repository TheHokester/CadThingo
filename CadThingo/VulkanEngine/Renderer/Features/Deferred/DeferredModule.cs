using CadThingo.VulkanEngine.Renderer.FrameGraph;
using CadThingo.VulkanEngine.Renderer.Features.Tonemapping;
using CadThingo.VulkanEngine.Renderer.Features.Forward;
using CadThingo.VulkanEngine.Renderer.Features.IBL;
using CadThingo.VulkanEngine.Renderer.Pipelines;
using Silk.NET.Vulkan;
using HostRenderer = CadThingo.VulkanEngine.Renderer.Renderer;

namespace CadThingo.VulkanEngine.Renderer.Features.Deferred;

/// <summary>
/// The deferred render graph module.
/// Builds the chain Cull -> LightCull -> Geometry -> Lighting -> Skybox -> Transparent into the
/// supplied <see cref="GraphScope"/>,
/// invokes a <see cref="TonemapModule"/> submodule that
/// consumes the HDR handle and writes FinalColor.
///

///Tonemap module is injected. host calls
/// <see cref="Build"/>, then MarkOutput(<see cref="Outputs.Final"/>) + Compile, then does the
/// post-compile descriptor rebind from the returned handles (those views are cached on the
/// renderer and also feed the PT&lt;-&gt;deferred flip + tonemap-operator rebind, host-side).
/// </summary>
public sealed class DeferredModule : IGraphModule<DeferredModule.Inputs, DeferredModule.Outputs>
{
    
    public readonly record struct Inputs(ImageResource FinalColor, Extent2D Extent);

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

    // Per-frame light-cull dispatch dims, read at Execute time. 
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

        // Deferred intermediates the graph OWNS as transients
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
        // and light-cull->lighting barriers + ordering. Renderables is the cull INPUT, imported
        // too so it can fill the cull pass set's binding 0 (matched by name below).
        var renderablesF   = new Buffer[RenderConfig.MAX_CONCURRENT_FRAMES];
        var indirectCmdF   = new Buffer[RenderConfig.MAX_CONCURRENT_FRAMES];
        var indirectCountF = new Buffer[RenderConfig.MAX_CONCURRENT_FRAMES];
        var instanceF      = new Buffer[RenderConfig.MAX_CONCURRENT_FRAMES];
        var tileCountF     = new Buffer[RenderConfig.MAX_CONCURRENT_FRAMES];
        var tileIndicesF   = new Buffer[RenderConfig.MAX_CONCURRENT_FRAMES];
        for (uint i = 0; i < RenderConfig.MAX_CONCURRENT_FRAMES; i++)
        {
            renderablesF[i]   = _cull.GetRenderablesBuffer(i);
            indirectCmdF[i]   = _cull.GetIndirectCmdBuffer(i);
            indirectCountF[i] = _cull.GetIndirectCountBuffer(i);
            instanceF[i]      = Engine.ResourceManager.GetInstanceBuffer(i);
            tileCountF[i]     = _lightCull.GetTileLightCountBuffer(i);
            tileIndicesF[i]   = _lightCull.GetTileLightIndicesBuffer(i);
        }
        var renderables   = scope.ImportBufferPerFrame(renderablesF,   default, "Renderables");
        var indirectCmd   = scope.ImportBufferPerFrame(indirectCmdF,   default, "IndirectCmd");
        var indirectCount = scope.ImportBufferPerFrame(indirectCountF, default, "IndirectCount");
        var instance      = scope.ImportBufferPerFrame(instanceF,      default, "InstanceData");
        var tileCount     = scope.ImportBufferPerFrame(tileCountF,     default, "TileLightCount");
        var tileIndices   = scope.ImportBufferPerFrame(tileIndicesF,   default, "TileLightIndices");

       
        // Cull is the first graph-baked pass set (descriptor-system.md phase C): its four storage
        // buffers are all graph resources, so the graph fills the set by name and the pipeline
        // owns only the layout. Names match DrawCullPipeline.PassSet.
        scope.AddPass("CullPass", PassType.Compute, QueueClass.Graphics,
            b =>
            {
                b.UsePassSet(_cull.PassSet);
                b.Read(renderables, ResourceUsage.StorageReadCompute, "renderables");
                indirectCmd   = b.Write(indirectCmd,   ResourceUsage.StorageWriteCompute, "indirectCmd");
                instance      = b.Write(instance,      ResourceUsage.StorageWriteCompute, "instanceData");
                indirectCount = b.Write(indirectCount, ResourceUsage.StorageWriteCompute, "indirectCount");
            },
            (CommandBuffer cmd, PassResources res, in RenderView f) =>
                _cull.Record(cmd, f, res.PassSet));

        // Light-cull (compute): bins lights into the per-tile lists the lighting FS reads.
        // Tile/light counts are computed in DeferredCore.Render and read back via _lightCullParams.
        scope.AddPass("LightCullPass", PassType.Compute, QueueClass.Graphics,
            b =>
            {
                // Tile outputs (set 1) are graph-baked; names match LightCullPipeline.PassSet.
                b.UsePassSet(_lightCull.PassSet);
                tileCount   = b.Write(tileCount,   ResourceUsage.StorageWriteCompute, "tileLightCount");
                tileIndices = b.Write(tileIndices, ResourceUsage.StorageWriteCompute, "tileLightIndices");
            },
            (CommandBuffer cmd, PassResources res, in RenderView f) =>
            {
                var (lightCount, tileCountX, tileCountY) = _lightCullParams();
                _lightCull.Record(cmd, f.FrameIndex, f.Camera, lightCount, tileCountX, tileCountY, res.PassSet);
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
            (CommandBuffer cmd, PassResources res, in RenderView f) =>
            {
                var indirectCmdBuf   = _cull.GetIndirectCmdBuffer(f.FrameIndex);
                var indirectCountBuf = _cull.GetIndirectCountBuffer(f.FrameIndex);
                var drawCount        = _cull.LastRenderableCount;
                var attachments = new GeometryPipeline.Attachments(
                    res.View(pos), res.View(normal), res.View(albedo),
                    res.View(material), res.View(emissive), res.View(depth));
                _geometry.Record(cmd, f, indirectCmdBuf, indirectCountBuf, drawCount, attachments);
            });

        // Lighting samples the five g-buffers and writes HDRColor@v1.
        // The deferred lighting pass reconstructs position from the g-buffer,
        // so depth is left in DepthStencilAttachmentOptimal for the skybox/transparent depth
        // tests. Per-tile light lists from LightCullPass are read as StorageReadFragment -> RAW
        // edge orders lighting after light-cull.
        scope.AddPass("LightingPass", PassType.Graphics, QueueClass.Graphics,
            b =>
            {
                // G-buffer set (set 1) is graph-baked; names match PbrDeferredPipeline.PassSet.
                b.UsePassSet(_pbrDeferred.PassSet);
                b.Read(pos,      ResourceUsage.SampledFragment, "gPosition");
                b.Read(normal,   ResourceUsage.SampledFragment, "gNormal");
                b.Read(albedo,   ResourceUsage.SampledFragment, "gAlbedo");
                b.Read(material, ResourceUsage.SampledFragment, "gMaterial");
                b.Read(emissive, ResourceUsage.SampledFragment, "gEmissive");
                // Tile-cull outputs now ride the same pass set as the g-buffer (bindings 5/6).
                b.Read(tileCount,   ResourceUsage.StorageReadFragment, "tileLightCount");
                b.Read(tileIndices, ResourceUsage.StorageReadFragment, "tileLightIndices");
                hdr = b.Write(hdr, ResourceUsage.ColorAttachment);
            },
            (CommandBuffer cmd, PassResources res, in RenderView f) =>
                _pbrDeferred.Record(cmd, f, res.View(hdr), res.PassSet));

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
            (CommandBuffer cmd, PassResources res, in RenderView f) =>
                _skybox.Record(cmd, f, new SkyboxPipeline.Attachments(res.View(hdr), res.View(depth))));

        scope.AddPass("TransparentPass", PassType.Graphics, QueueClass.Graphics,
            b =>
            {
                // Tile-cull outputs ride the graph-baked pass set (set 1); IBL/probes come from
                // the registry's FeatureIBL set.
                b.UsePassSet(_transparent.PassSet);
                b.Read(tileCount,   ResourceUsage.StorageReadFragment, "tileLightCount");
                b.Read(tileIndices, ResourceUsage.StorageReadFragment, "tileLightIndices");
                hdr   = b.Write(hdr,   ResourceUsage.ColorAttachment);
                depth = b.Write(depth, ResourceUsage.DepthAttachment);
            },
            (CommandBuffer cmd, PassResources res, in RenderView f) =>
                _transparent.Record(cmd, f, _cull.LastTransparentDraws,
                    new TransparentPipeline.Attachments(res.View(hdr), res.View(depth)), res.PassSet));

        // Tonemap is a nested submodule: it imports FinalColor and reads HDRColor@v3 and writes the final image.
        _tonemap.Build(scope.Child("Tonemap"),
            new TonemapModule.Input(hdr, inputs.FinalColor), out var tm);

        outputs = new Outputs(pos, normal, albedo, material, emissive, hdr, tm.FinalColor);
    }
}