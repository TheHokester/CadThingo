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
    /// Five glTF channel names matching the texture-index field order on PbrMaterial:
    /// BaseColor, MetallicRoughness, Normal, Occlusion, Emissive.
    /// </summary>
    public static readonly string[] Channels = { "BaseColor", "MetallicRoughness", "Normal", "Occlusion", "Emissive" };

    /// <summary>
    /// Build a PbrMaterial for one glTF material — registers any missing textures into the
    /// bindless table and stores their indices on the material.
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
                string texId = $"{idPrefix}:mat:{matIdx}:{channelName}";
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

        return new PbrMaterial
        {
            BaseColorFactor       = baseColorChannel?.Color ?? new Vector4(1, 1, 1, 1),
            MetallicFactor        = (float)(mrChannel?.GetFactor("MetallicFactor")  ?? 1.0),
            RoughnessFactor       = (float)(mrChannel?.GetFactor("RoughnessFactor") ?? 1.0),
            AlphaCutoff           = gltfMat.AlphaCutoff,
            Flags                 = gltfMat.Alpha == SharpGLTF.Schema2.AlphaMode.MASK ? 1u : 0u,
            BaseColorTex          = texIdx[0],
            PhysicalDescriptorTex = texIdx[1],
            NormalTex             = texIdx[2],
            OcclusionTex          = texIdx[3],
            EmissiveTex           = texIdx[4],
        };
    }

    /// <summary>Vulkan format the channel's texture should be sampled in (sRGB vs linear).</summary>
    private static Format ChannelFormat(string channelName) => channelName switch
    {
        "BaseColor" => Format.R8G8B8A8Srgb,
        "Emissive"  => Format.R8G8B8A8Srgb,
        _           => Format.R8G8B8A8Unorm, // MetallicRoughness, Normal, Occlusion are linear data textures
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