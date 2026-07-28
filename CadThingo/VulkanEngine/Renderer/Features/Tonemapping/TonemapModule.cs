using CadThingo.VulkanEngine.Renderer.FrameGraph;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Features.Tonemapping;

/// <summary>
/// Host post-stack tonemap as a standalone graph module (render-graph.md sec 9.5): reads a
/// scene-colour HDR handle produced upstream and writes the host-owned FinalColor. It is reused
/// by every core (deferred / PT / RT), which is why it lives at the host level -- appended after
/// whichever core ran -- not inside a core module. Flatten composition means the core->tonemap
/// seam (the HDR read) gets its barrier derived automatically, no hand-stitching.
///
/// HDR arrives as a GraphImage HANDLE, not an ImageResource: it is a transient shared with the
/// upstream core, so it must be the SAME graph resource (one create/import, versioned) -- a
/// second ImportImage of the same physical image would mint a separate id and break the
/// read-after-write dependency on the core's last HDR write. FinalColor is touched only here, so
/// the module imports it directly from the host-owned <see cref="ImageResource"/>.
///
/// The pipeline is injected. The HDR-input descriptor (set 0) is graph-baked: this module names
/// the HDR read, so the graph allocates + writes tonemap's set from the upstream resource and no
/// host-side rebind is needed. Every core is graph-resident and composes this module, so this is
/// the only binding path.
/// </summary>
public sealed class TonemapModule : IGraphModule<TonemapModule.Input, TonemapModule.Output>
{
    /// <summary><see cref="SceneColorHdr"/> is the upstream HDR handle (read); <see cref="FinalColor"/>
    /// is the host-owned target this module imports and writes.</summary>
    public readonly record struct Input(GraphImage SceneColorHdr, ImageResource FinalColor);

    /// <summary>The written FinalColor handle, for the host to MarkOutput.</summary>
    public readonly record struct Output(GraphImage FinalColor);

    private readonly TonemapPipeline _tonemap;
    // The expected upstream HDR format for the wire-time port check. Defaults to the deferred
    // chain's R16F scene-colour transient; the PT graph cores pass PtOutColor's R32F. The shader
    // just samples a float4, so any float colour format tonemaps correctly.
    private readonly Format _hdrFormat;

    public TonemapModule(TonemapPipeline tonemap, Format hdrFormat = Format.R16G16B16A16Sfloat)
    {
        _tonemap = tonemap;
        _hdrFormat = hdrFormat;
    }

    public void Build(GraphScope scope, in Input inputs, out Output outputs)
    {
        // Wire-time port check: the upstream HDR must be a sampleable image of the declared
        // format. A format / usage mismatch is a build-time throw, not a runtime corruption.
        var hdr = scope.ExpectImage(inputs.SceneColorHdr, _hdrFormat, ImageUsageFlags.SampledBit);

        // FinalColor: imported (host/RenderTargets-owned; the swapchain blit + ImGui viewport
        // sample it), handed back in ShaderReadOnly. Undefined incoming == discard (tonemap
        // rewrites every pixel).
        var fc = inputs.FinalColor;
        var finalDesc = new ImageDesc { Format = fc._format, Mips = 1, Layers = 1 };
        var final = scope.ImportImage(fc.Image, fc.ImageView, in finalDesc,
            ImageLayout.Undefined, "FinalColor", ImageLayout.ShaderReadOnlyOptimal);

        // Tonemap reads HDR, writes FinalColor. The HDR read crosses the core->host module
        // boundary, so it surfaces as a cross-module edge in ToDot.
        scope.AddPass("TonemapPass", PassType.Graphics, QueueClass.Graphics,
            b =>
            {
                // HDR-input set (set 0) is graph-baked; name matches TonemapPipeline.PassSet.
                b.UsePassSet(_tonemap.PassSet);
                b.Read(hdr, ResourceUsage.SampledFragment, "hdrInput");
                final = b.Write(final, ResourceUsage.ColorAttachment);
            },
            (CommandBuffer cmd, PassResources res, in RenderView f) =>
                _tonemap.Record(cmd, f, res.View(final), res.PassSet));

        outputs = new Output(final);
    }
}