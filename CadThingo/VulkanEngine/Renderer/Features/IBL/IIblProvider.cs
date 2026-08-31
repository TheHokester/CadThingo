using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Features.IBL;

/// <summary>
/// What the IBL feature offers to other features and to the lighting pipelines. Deliberately thin:
/// the split-sum images themselves are <i>bindable</i>, so they travel through
/// <see cref="Descriptors.DescriptorRegistry"/> by name and never appear here. Only what a consumer
/// cannot bind belongs on this contract - a scalar the shader needs as a uniform, and the two
/// handles the reflection probes reuse rather than duplicate.
///
/// Consumers take it as <c>INeedsFeature&lt;IIblProvider&gt;</c> and read through it at record /
/// bake time, never caching, so a rebake can never leave a stale copy behind.
/// </summary>
public interface IIblProvider
{
    /// <summary>Mip count of the prefiltered specular cube. The PBR and path-trace shaders need it
    /// as a uniform to map roughness onto a mip level, and it is a property of the image rather
    /// than of any one pipeline.</summary>
    uint PrefilteredCubeMipLevels { get; }

    /// <summary>The shared cubemap sampler (linear + mip-linear, ClampToEdge so face seams do not
    /// bleed at low mips). Reflection-probe prefiltering samples its capture cube with the exact
    /// same filtering as the global prefilter, so it borrows this rather than defining a second.</summary>
    Sampler CubeSampler { get; }

    /// <summary>The GGX prefilter bake pipeline. Reflection probes run the identical convolution
    /// over a per-probe capture cube, so they drive this pipeline instead of building a byte-identical
    /// second one - sharing the compiled kernel is the point of exposing it.</summary>
    IblBakePipeline PrefilterPipeline { get; }
}
