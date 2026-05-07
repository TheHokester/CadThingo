using CadThingo.VulkanEngine.Renderer;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.GLTF;

/// <summary>
/// 1×1 fallback textures for glTF material slots that don't ship a real texture.
/// Registered into the ResourceManager + bindless texture table once on first call so
/// subsequent loads share them. Bindless indices are exposed so material builds can
/// store them directly in PbrMaterial slot fields.
/// </summary>
public static unsafe class GltfDefaults
{
    public const string BaseColorId         = "__gltf_default_BaseColor";
    public const string MetallicRoughnessId = "__gltf_default_MetallicRoughness";
    public const string NormalId            = "__gltf_default_Normal";
    public const string OcclusionId         = "__gltf_default_Occlusion";
    public const string EmissiveId          = "__gltf_default_Emissive";

    private static bool _registered;
    private static Texture? _baseColor, _metallicRoughness, _normal, _occlusion, _emissive;

    public static int BaseColorIndex         { get; private set; } = -1;
    public static int MetallicRoughnessIndex { get; private set; } = -1;
    public static int NormalIndex            { get; private set; } = -1;
    public static int OcclusionIndex         { get; private set; } = -1;
    public static int EmissiveIndex          { get; private set; } = -1;

    public static Texture BaseColor         => _baseColor         ?? throw new InvalidOperationException("GltfDefaults not registered");
    public static Texture MetallicRoughness => _metallicRoughness ?? throw new InvalidOperationException("GltfDefaults not registered");
    public static Texture Normal            => _normal            ?? throw new InvalidOperationException("GltfDefaults not registered");
    public static Texture Occlusion         => _occlusion         ?? throw new InvalidOperationException("GltfDefaults not registered");
    public static Texture Emissive          => _emissive          ?? throw new InvalidOperationException("GltfDefaults not registered");

    public static void EnsureRegistered(ResourceManager rm, Renderer.Renderer renderer)
    {
        if (_registered) return;

        // BaseColor: white sRGB so the baseColorFactor multiplier comes through unmodified.
        _baseColor         = MakeOnePixel(renderer, 255, 255, 255, 255, Format.R8G8B8A8Srgb);
        // MetallicRoughness: Geometry.slang samples .bg → green=roughness, blue=metallic.
        // (0,255,0,255) → roughness=1, metallic=0 (matches glTF default factors).
        _metallicRoughness = MakeOnePixel(renderer,   0, 255,   0, 255, Format.R8G8B8A8Unorm);
        // Normal: flat normal pointing +Z in tangent space, encoded (128,128,255).
        _normal            = MakeOnePixel(renderer, 128, 128, 255, 255, Format.R8G8B8A8Unorm);
        // Occlusion: 1.0 = no occlusion.
        _occlusion         = MakeOnePixel(renderer, 255, 255, 255, 255, Format.R8G8B8A8Unorm);
        // Emissive: black sRGB so missing emissive contributes nothing additive.
        _emissive          = MakeOnePixel(renderer,   0,   0,   0, 255, Format.R8G8B8A8Srgb);

        rm.Load<TextureResource>(BaseColorId,         _ => new TextureResource(BaseColorId,         _baseColor));
        rm.Load<TextureResource>(MetallicRoughnessId, _ => new TextureResource(MetallicRoughnessId, _metallicRoughness));
        rm.Load<TextureResource>(NormalId,            _ => new TextureResource(NormalId,            _normal));
        rm.Load<TextureResource>(OcclusionId,         _ => new TextureResource(OcclusionId,         _occlusion));
        rm.Load<TextureResource>(EmissiveId,          _ => new TextureResource(EmissiveId,          _emissive));

        // Reserve stable bindless indices. Materials missing a real texture for a slot point here.
        BaseColorIndex         = rm.RegisterBindless(_baseColor);
        MetallicRoughnessIndex = rm.RegisterBindless(_metallicRoughness);
        NormalIndex            = rm.RegisterBindless(_normal);
        OcclusionIndex         = rm.RegisterBindless(_occlusion);
        EmissiveIndex          = rm.RegisterBindless(_emissive);

        _registered = true;
    }

    private static Texture MakeOnePixel(Renderer.Renderer renderer, byte r, byte g, byte b, byte a, Format format)
    {
        Span<byte> rgba = stackalloc byte[4] { r, g, b, a };
        fixed (byte* p = rgba)
        {
            return Texture.CreateTextureFromMemory(renderer, p, 1, 1, format, new Extent3D(1, 1, 1));
        }
    }
}