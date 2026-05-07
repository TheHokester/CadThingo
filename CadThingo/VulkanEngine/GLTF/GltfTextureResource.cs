using CadThingo.VulkanEngine.Renderer;
using Silk.NET.Vulkan;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace CadThingo.VulkanEngine.GLTF;

/// <summary>
/// Static helpers to decode glTF-embedded image bytes (PNG/JPG) into a Vulkan Texture
/// and register it as a TextureResource in the ResourceManager. The returned Texture
/// is what gets bound into a material descriptor set; the TextureResource exists for
/// lifetime/disposal tracking.
///
/// glTF can carry four image MIME types in total:
///   • image/jpeg       — supported
///   • image/png        — supported
///   • image/ktx2       — NOT supported yet (KHR_texture_basisu, supercompressed)
///   • image/webp       — NOT supported yet (EXT_texture_webp)
/// Anything outside the supported set throws so unsupported assets fail loudly
/// instead of producing garbled output.
/// </summary>
public static unsafe class GltfTextureResource
{
    /// <summary>
    /// Decode <paramref name="encodedBytes"/> (PNG/JPG only), upload as a Vulkan texture
    /// in the requested <paramref name="format"/> (sRGB for color, Unorm for data),
    /// and register a TextureResource keyed by <paramref name="id"/>. If a texture
    /// with this id is already registered, returns its underlying Texture.
    /// </summary>
    public static Texture BuildAndRegister(
        string id,
        ResourceManager rm,
        Renderer.Renderer renderer,
        ReadOnlyMemory<byte> encodedBytes,
        string mimeType,
        Format format)
    {
        // Idempotent: if already loaded, surface the existing Texture so multiple materials
        // sharing the same image don't re-decode/upload.
        if (rm.HasResource<TextureResource>(id))
        {
            return rm.GetResource<TextureResource>(id)->Texture;
        }

        // Reject unsupported image formats up-front. KTX2 (KHR_texture_basisu) and WebP
        // (EXT_texture_webp) need dedicated decode paths we haven't built yet.
        switch (mimeType)
        {
            case "image/png":
            case "image/jpeg":
                break;
            case "image/ktx2":
                throw new NotSupportedException(
                    $"glTF image '{id}' is image/ktx2 (KHR_texture_basisu). KTX2 decoding is not implemented yet.");
            case "image/webp":
                throw new NotSupportedException(
                    $"glTF image '{id}' is image/webp (EXT_texture_webp). WebP decoding is not implemented yet.");
            default:
                throw new NotSupportedException(
                    $"glTF image '{id}' has unrecognised MIME type '{mimeType}'. Only image/png and image/jpeg are supported.");
        }

        using var img = ImageSharp.Image.Load<Rgba32>(encodedBytes.Span);
        var pixels = new Rgba32[img.Width * img.Height];
        img.CopyPixelDataTo(pixels);

        Texture tex;
        fixed (Rgba32* p = pixels)
        {
            tex = Texture.CreateTextureFromMemory(
                renderer,
                (byte*)p,
                (uint)img.Width,
                (uint)img.Height,
                format,
                new Extent3D((uint)img.Width, (uint)img.Height, 1));
        }

        rm.Load<TextureResource>(id, _ => new TextureResource(id, tex));
        return tex;
    }
}