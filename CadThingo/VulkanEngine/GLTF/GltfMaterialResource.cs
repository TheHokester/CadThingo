using System.Numerics;
using CadThingo.VulkanEngine.Renderer;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.GLTF;

/// <summary>
/// Static helper that turns a SharpGLTF <see cref="SharpGLTF.Schema2.Material"/> into a
/// shader-ready <see cref="PbrMaterial"/> referencing bindless texture indices. Per-channel
/// textures missing on the glTF material fall back to the default indices in <see cref="GltfDefaults"/>.
/// </summary>
public static class GltfMaterialResource
{
    /// <summary>
    /// Core glTF channel names matching the texture-index field order on PbrMaterial:
    /// BaseColor, MetallicRoughness, Normal, Occlusion, Emissive. Extension channels
    /// (clearcoat / transmission) are loaded separately further down so missing
    /// extensions don't fail the lookup path.
    /// </summary>
    public static readonly string[] Channels = { "BaseColor", "MetallicRoughness", "Normal", "Occlusion", "Emissive" };

    /// <summary>
    /// Build a PbrMaterial for one glTF material — registers any missing textures into the
    /// bindless table and stores their indices on the material. Also pulls KHR_materials_*
    /// extension data (transmission, IOR, clearcoat) into the matching PbrMaterial fields;
    /// materials that don't carry the extension fall back to glTF defaults.
    /// </summary>
    public static PbrMaterial BuildAndRegister(
        string idPrefix,
        int matIdx,
        SharpGLTF.Schema2.Material gltfMat,
        ResourceManager rm,
        Renderer.Renderer renderer)
    {
        Span<int> texIdx = stackalloc int[5];

        for (int i = 0; i < Channels.Length; i++)
        {
            string channelName = Channels[i];
            var format = ChannelFormat(channelName);
            var channel = gltfMat.FindChannel(channelName);
            var image = channel?.Texture?.PrimaryImage;

            if (image != null)
            {
                // Key by image identity + format — multiple materials referencing
                // the same glTF image share one TextureResource and (via the
                // RegisterBindless dedup) one bindless slot.
                string texId = $"{idPrefix}:img:{image.LogicalIndex}:fmt{(int)format}";
                var tex = GltfTextureResource.BuildAndRegister(
                    texId, rm, renderer,
                    image.Content.Content,
                    image.Content.MimeType,
                    format);
                texIdx[i] = rm.RegisterBindless(tex);
            }
            else
            {
                texIdx[i] = ChannelDefaultIndex(channelName);
            }
        }

        var baseColorChannel = gltfMat.FindChannel("BaseColor");
        var mrChannel        = gltfMat.FindChannel("MetallicRoughness");
        var emChannel        = gltfMat.FindChannel("Emissive");

        // KHR_materials_transmission + KHR_materials_ior
        // Missing channels return null → factor stays at the glTF default
        // (transmission 0, IOR 1.5) and the texture slot points at the
        // white fallback (1.0 multiplier, so the factor controls the result).
        var transmissionChannel = gltfMat.FindChannel("Transmission");
        float transmissionFactor = (float)SafeGetFactor(transmissionChannel, "TransmissionFactor", 0.0);
        int transmissionTex = LoadExtensionTexture(idPrefix, matIdx, "Transmission", transmissionChannel,
                                                   Format.R8G8B8A8Unorm, GltfDefaults.OcclusionIndex,
                                                   rm, renderer);

        // SharpGLTF surfaces KHR_materials_ior as a direct property on Material.
        // Default is 1.5 (dielectric / glass).
        float ior = gltfMat.IndexOfRefraction;

        // KHR_materials_clearcoat
        var clearcoatChannel          = gltfMat.FindChannel("ClearCoat");
        var clearcoatRoughnessChannel = gltfMat.FindChannel("ClearCoatRoughness");
        var clearcoatNormalChannel    = gltfMat.FindChannel("ClearCoatNormal");

        // Factor names differ between SharpGLTF versions and per-channel — fall
        // back through a few plausible spellings and finally default if none
        // match. Cheaper than reflecting on Parameters; immune to API churn.
        float clearcoatFactor          = (float)SafeGetFactor(clearcoatChannel, "ClearCoatFactor",
                                            (float)SafeGetFactor(clearcoatChannel, "ClearcoatFactor", 0.0));
        float clearcoatRoughnessFactor = (float)SafeGetFactor(clearcoatRoughnessChannel, "ClearCoatRoughnessFactor",
                                            (float)SafeGetFactor(clearcoatRoughnessChannel, "ClearcoatRoughnessFactor",
                                            (float)SafeGetFactor(clearcoatRoughnessChannel, "RoughnessFactor", 0.0)));

        int clearcoatTex          = LoadExtensionTexture(idPrefix, matIdx, "ClearCoat", clearcoatChannel,
                                                          Format.R8G8B8A8Unorm, GltfDefaults.OcclusionIndex,
                                                          rm, renderer);
        int clearcoatRoughnessTex = LoadExtensionTexture(idPrefix, matIdx, "ClearCoatRoughness", clearcoatRoughnessChannel,
                                                          Format.R8G8B8A8Unorm, GltfDefaults.OcclusionIndex,
                                                          rm, renderer);
        int clearcoatNormalTex    = LoadExtensionTexture(idPrefix, matIdx, "ClearCoatNormal", clearcoatNormalChannel,
                                                          Format.R8G8B8A8Unorm, GltfDefaults.NormalIndex,
                                                          rm, renderer);

        // KHR_materials_volume — absorption only (thickness / scattering ignored).
        // FindChannel is null-safe: if the extension is absent (or the channel
        // name doesn't match this SharpGLTF build) we fall back to white /
        // sentinel distance, i.e. no absorption. White attenuationColor yields
        // σ_a = 0 regardless of distance, so the default is always inert.
        var volumeChannel = gltfMat.FindChannel("VolumeAttenuation");
        Vector3 attenuationColor = volumeChannel?.Color.AsVector3() ?? Vector3.One;
        float attenuationDistance = (float)SafeGetFactor(volumeChannel, "AttenuationDistance",
                                                         PbrMaterialVolume.NoAbsorptionDistance);

        return new PbrMaterial
        {
            BaseColorFactor          = baseColorChannel?.Color ?? new Vector4(1, 1, 1, 1),
            EmissiveFactor           = emChannel?.Color.AsVector3() ?? Vector3.Zero,
            AlphaCutoff              = gltfMat.AlphaCutoff,
            MetallicFactor           = (float)(mrChannel?.GetFactor("MetallicFactor")  ?? 1.0),
            RoughnessFactor          = (float)(mrChannel?.GetFactor("RoughnessFactor") ?? 1.0),
            Flags                    = (gltfMat.Alpha == SharpGLTF.Schema2.AlphaMode.MASK  ? 1u : 0u)
                                     | (gltfMat.Alpha == SharpGLTF.Schema2.AlphaMode.BLEND ? 4u : 0u),
            BaseColorTex             = texIdx[0],
            PhysicalDescriptorTex    = texIdx[1],
            NormalTex                = texIdx[2],
            OcclusionTex             = texIdx[3],
            EmissiveTex              = texIdx[4],
            TransmissionFactor       = transmissionFactor,
            Ior                      = ior,
            ClearcoatFactor          = clearcoatFactor,
            ClearcoatRoughnessFactor = clearcoatRoughnessFactor,
            TransmissionTex          = transmissionTex,
            ClearcoatTex             = clearcoatTex,
            ClearcoatRoughnessTex    = clearcoatRoughnessTex,
            ClearcoatNormalTex       = clearcoatNormalTex,
            AttenuationColor         = attenuationColor,
            AttenuationDistance      = attenuationDistance,
        };
    }

    /// <summary>
    /// SharpGLTF's MaterialChannel.GetFactor throws (LINQ Single) when the key is
    /// missing from the channel's parameters — defeats the standard `?? default`
    /// pattern. Wrap so missing-key channels fall back to <paramref name="fallback"/>
    /// without surfacing the exception. Null channel also returns fallback.
    /// </summary>
    private static double SafeGetFactor(SharpGLTF.Schema2.MaterialChannel? channel, string key, double fallback)
    {
        if (channel == null) return fallback;
        try
        {
            object v = channel.Value.GetFactor(key);
            return Convert.ToDouble(v);
        }
        catch
        {
            return fallback;
        }
    }

    /// <summary>
    /// Same idea as the inline loop above but for extension channels that may not
    /// be present at all. Returns the channel texture's bindless index when set,
    /// otherwise <paramref name="fallbackBindlessIndex"/> (typically a white/flat
    /// 1×1 default so the corresponding factor alone controls contribution).
    /// </summary>
    private static int LoadExtensionTexture(
        string idPrefix,
        int matIdx,
        string channelName,
        SharpGLTF.Schema2.MaterialChannel? channel,
        Format format,
        int fallbackBindlessIndex,
        ResourceManager rm,
        Renderer.Renderer renderer)
    {
        var image = channel?.Texture?.PrimaryImage;
        if (image == null) return fallbackBindlessIndex;

        // Image-identity keying — see comment in the core-channel loop above.
        // Critical for clearcoat assets where the same ORM-style atlas is
        // commonly read from three channels in one material.
        string texId = $"{idPrefix}:img:{image.LogicalIndex}:fmt{(int)format}";
        var tex = GltfTextureResource.BuildAndRegister(
            texId, rm, renderer,
            image.Content.Content,
            image.Content.MimeType,
            format);
        return rm.RegisterBindless(tex);
    }

    /// <summary>Vulkan format the channel's texture should be sampled in. BC-compressed to cut
    /// texture-cache fill: BC3 baseColor (keeps the MASK/BLEND alpha), BC1 emissive (no alpha), and
    /// BC5 normals (two channels; shaders reconstruct z via decodeTangentNormal). The loader falls
    /// back to RGBA8 when the device lacks textureCompressionBC. MetallicRoughness/Occlusion stay
    /// uncompressed for now (BC quantisation of packed data channels needs a closer quality look).</summary>
    private static Format ChannelFormat(string channelName) => channelName switch
    {
        "BaseColor" => Format.BC3SrgbBlock,
        "Emissive"  => Format.BC1RgbSrgbBlock,
        "Normal"    => Format.BC5UnormBlock,
        _           => Format.R8G8B8A8Unorm, // MetallicRoughness, Occlusion are linear data textures
    };

    private static int ChannelDefaultIndex(string channelName) => channelName switch
    {
        "BaseColor"         => GltfDefaults.BaseColorIndex,
        "MetallicRoughness" => GltfDefaults.MetallicRoughnessIndex,
        "Normal"            => GltfDefaults.NormalIndex,
        "Occlusion"         => GltfDefaults.OcclusionIndex,
        "Emissive"          => GltfDefaults.EmissiveIndex,
        _ => throw new ArgumentOutOfRangeException(nameof(channelName), channelName, "Unknown PBR channel"),
    };
}