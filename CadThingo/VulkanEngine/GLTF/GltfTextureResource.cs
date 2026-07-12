using CadThingo.VulkanEngine.Renderer;
using Silk.NET.Vulkan;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace CadThingo.VulkanEngine.GLTF;

/// <summary>
/// Static helpers to decode glTF-embedded image bytes (PNG/JPG/WebP) into a Vulkan Texture
/// and register it as a TextureResource in the ResourceManager. The returned Texture
/// is what gets bound into a material descriptor set; the TextureResource exists for
/// lifetime/disposal tracking.
///
/// glTF can carry four image MIME types in total:
///   • image/jpeg       — supported
///   • image/png        — supported
///   • image/webp       — supported (EXT_texture_webp, decoded via ImageSharp's built-in WebP decoder)
///   • image/ktx2       — NOT supported yet (KHR_texture_basisu, supercompressed)
/// Anything outside the supported set throws so unsupported assets fail loudly
/// instead of producing garbled output.
/// </summary>
public static unsafe class GltfTextureResource
{
    /// <summary>Optional channel repack applied to decoded pixels before upload/compression.
    /// MetallicRoughnessRg moves glTF's roughness(G)/metallic(B) into R,G so BC5 (which keeps
    /// only the R,G channels) preserves both; consumers then sample .rg. None leaves pixels as-is.</summary>
    public enum PrePack { None, MetallicRoughnessRg }

    /// <summary>
    /// Decode <paramref name="encodedBytes"/> (PNG/JPG/WebP), upload as a Vulkan texture
    /// in the requested <paramref name="format"/> (sRGB for color, Unorm for data),
    /// and register a TextureResource keyed by <paramref name="id"/>. If a texture
    /// with this id is already registered, returns its underlying Texture.
    /// </summary>
    public static Texture BuildAndRegister(
        string id,
        ResourceManager rm,
        GraphicsDevice gfx,
        ReadOnlyMemory<byte> encodedBytes,
        string mimeType,
        Format format,
        PrePack prePack = PrePack.None)
    {
        // Idempotent: if already loaded, surface the existing Texture so multiple materials
        // sharing the same image don't re-decode/upload.
        if (rm.HasResource<TextureResource>(id))
        {
            return rm.GetResource<TextureResource>(id)!.Texture;
        }

        // Reject unsupported image formats up-front. KTX2 (KHR_texture_basisu) still
        // needs a dedicated decode path we haven't built yet; PNG/JPEG/WebP all funnel
        // through ImageSharp's auto-detecting Load<Rgba32>.
        switch (mimeType)
        {
            case "image/png":
            case "image/jpeg":
            case "image/webp":
                break;
            case "image/ktx2":
                throw new NotSupportedException(
                    $"glTF image '{id}' is image/ktx2 (KHR_texture_basisu). KTX2 decoding is not implemented yet.");
            default:
                throw new NotSupportedException(
                    $"glTF image '{id}' has unrecognised MIME type '{mimeType}'. Only image/png, image/jpeg, and image/webp are supported.");
        }

        using var img = ImageSharp.Image.Load<Rgba32>(encodedBytes.Span);
        var pixels = new Rgba32[img.Width * img.Height];
        img.CopyPixelDataTo(pixels);

        // Repack metallic(B)/roughness(G) into R,G so BC5 (R,G only) keeps both.
        if (prePack == PrePack.MetallicRoughnessRg)
            for (int i = 0; i < pixels.Length; i++)
                pixels[i] = new Rgba32(pixels[i].B, pixels[i].G, (byte)0, (byte)255);

        // BC formats are GPU-compressed at load to cut texture-cache (L1Tex->L2) fill; if the
        // device lacks textureCompressionBC, fall back to the equivalent uncompressed RGBA8.
        bool   wantBc      = Texture.IsBcFormat(format);
        bool   canBc       = gfx.TextureCompressionBcEnabled;
        Format uploadFmt   = wantBc && !canBc ? Texture.BcFallbackFormat(format) : format;

        Texture tex;
        fixed (Rgba32* p = pixels)
        {
            tex = wantBc && canBc
                ? Texture.CreateCompressedTexture(gfx, (byte*)p, (uint)img.Width, (uint)img.Height, format)
                : Texture.CreateTextureFromMemory(gfx, (byte*)p, (uint)img.Width, (uint)img.Height, uploadFmt,
                    new Extent3D((uint)img.Width, (uint)img.Height, 1));
        }

        rm.Load<TextureResource>(id, _ => new TextureResource(id, tex));
        return tex;
    }
}