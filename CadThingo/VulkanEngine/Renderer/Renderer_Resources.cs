using System.Numerics;
using System.Runtime.InteropServices;
using ImGuiNET;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer;
/* How the shadows with ray query work 
 * Three parts. The flow is: deferred shading hands the lighting pass a world-space position per pixel → ray query asks "is anything between that position and the light?" → AS makes that question fast.                       
                                                                                                                                                                                                                             
  1. How "this pixel" becomes "this point in the world"                                                                                                                                                                        
   
  Your geometry pass doesn't draw colors — it draws a g-buffer. For every pixel the screen covers, the geometry shader writes the world-space position of whatever surface is there into GBuffer_Position, plus the            
  normal/albedo/material into the others. That image is just a 2D grid of world coordinates indexed by screen pixel.

  The lighting pass then runs a fullscreen triangle. There's one fragment shader invocation per pixel of the framebuffer. Inside that shader, gBufferPositionSampler.Sample(input.UV) reads back the world position that the
  geometry pass wrote for that exact pixel. So WorldPos in PSMain literally means "the 3D point that this screen pixel is showing."

  That's the bridge. Once you have WorldPos per pixel, "shadow ray from this pixel to the light" makes sense — you cast from WorldPos toward lightPos, and if anything is in the way, the pixel's lit by something other than
  that light. No connection to triangles or meshes anymore — just two world points and a yes/no question.

  2. What an acceleration structure actually is

  Naive shadow trace: for each pixel, walk every triangle in the scene and test ray-triangle intersection. Viking room is ~3800 triangles, screen is ~1M pixels, one light → ~3.8 billion intersection tests per frame. Dead.

  An acceleration structure is a spatial index. It's a tree built over your geometry where each interior node holds a bounding box and each leaf holds a triangle. To check if a ray hits anything, you start at the root: does
   the ray intersect this node's box? If no, skip the entire subtree. If yes, recurse. You touch O(log N) triangles instead of O(N).

  The hardware on your 4070 Ti has dedicated silicon for this traversal — that's what RT cores are. RayQuery.TraceRayInline doesn't loop in software; it dispatches to those cores. That's why ~1M shadow rays per frame is
  cheap.

  Tree shape, build algorithm, node packing — all driver/hardware specifics behind vkCmdBuildAccelerationStructures. You hand it triangles + a flag like PreferFastTrace, and it picks the structure for you.

  3. Why two levels (BLAS + TLAS)

  Imagine 100 instances of the viking room scattered around. If everything were one big AS, you'd:
  - Pay to retesselate the same mesh into the structure 100 times (huge build cost),
  - Have to rebuild the entire thing every time any instance moves.

  Two levels split that:

  BLAS (Bottom-Level) — built per unique mesh, in mesh-local space. Holds the actual triangles. Expensive to build (it's the spatial tree over real geometry), but you only build it once per mesh and cache it. Your blasCache
   is keyed by Mesh* for exactly this reason.

  TLAS (Top-Level) — built per scene. Doesn't hold triangles at all. Each entry is a tiny record: a 3×4 transform + a pointer to a BLAS. So 100 viking rooms = one BLAS + 100 small TLAS records. When a viking room moves, you
   only rebuild the TLAS (cheap - it's effectively a scene graph in spatial form). The BLAS is untouched.
 */
public unsafe partial class Renderer
{
    // Device-service helpers moved to GraphicsDevice (L1). Forwarders keep the
    // former Renderer.* / Engine.renderer.* call sites (pipelines, ImageResource,
    // Texture, ResourceManager) compiling unchanged.
    public CommandBuffer BeginSingleTimeCommands() => gfx.BeginSingleTimeCommands();

    public void EndSingleTimeCommands(CommandBuffer cmd) => gfx.EndSingleTimeCommands(cmd);

    public void CreateBuffer(ulong size, BufferUsageFlags usage, MemoryPropertyFlags memProps,
        out Buffer buffer, out SubAlloc alloc, float priority = GpuMemoryAllocator.PriorityDefault,
        bool preferDeviceLocal = false)
        => gfx.CreateBuffer(size, usage, memProps, out buffer, out alloc, priority, preferDeviceLocal);

    public void CopyBuffer(Buffer src, Buffer dst, ulong size) => gfx.CopyBuffer(src, dst, size);

    public void UploadBufferData(Buffer dst, long dstOffset, void* srcData, ulong size)
        => gfx.UploadBufferData(dst, dstOffset, srcData, size);

    public void DestroyBuffer(Buffer buffer, SubAlloc alloc) => gfx.DestroyBuffer(buffer, alloc);

    internal void CreateMappedUniformBuffer(int sizeBytes, ref UboBuffer ubo)
        => gfx.CreateMappedUniformBuffer(sizeBytes, ref ubo);

    public void TransitionImageLayout(CommandBuffer cmd, Image image, Format format, ImageLayout oldLayout,
        ImageLayout newLayout, uint mipLevels = 1)
        => gfx.TransitionImageLayout(cmd, image, format, oldLayout, newLayout, mipLevels);

    internal void GenerateMipMaps(CommandBuffer cmds, Image image, Format format, uint width, uint height, uint mipLevels)
        => gfx.GenerateMipMaps(cmds, image, format, width, height, mipLevels);
    

    private void CreateDescriptorPool()
    {
        // Sizing budget per frame-in-flight where relevant:
        //   - UniformBuffer:   GeometryFrameUBO + LightingUBO  →  2 × MAX_FRAMES + headroom
        //   - StorageBuffer:   PbrMaterial[]   + InstanceData[] →  2 × MAX_FRAMES + headroom
        //   - SampledImage:    bindless texture array          →  MAX_BINDLESS_TEXTURES × MAX_FRAMES
        //   - Sampler:         bindless samplers (default 0)   →  8 × MAX_FRAMES
        //   - CombinedImageSampler: g-buffer samplers (shared) →  5 + headroom
        //   - AccelerationStructure: TLAS                       →  MAX_FRAMES + headroom
        var poolSizes = new DescriptorPoolSize[]
        {
            new() { Type = DescriptorType.UniformBuffer,            DescriptorCount = 24 },
            // Storage buffer budget: bindless mat+instance (2 × MAX_FRAMES), light SSBO
            // (MAX_FRAMES), cull pass (renderables + cmds + instancesOut + count =
            // 4 × MAX_FRAMES), PBR shadow-alpha set (ShadowEntityInfo + global vb + ib),
            // plus the wavefront tracer (set 0 lights/shadow/emissive × MAX_FRAMES, set 1
            // vb/ib, and the 25-binding set-4 SoA working set: ping-ponged shadow records
            // + connectArgs + shadowRadiance on top of the original 18). Round up generously.
            new() { Type = DescriptorType.StorageBuffer,            DescriptorCount = 128 },
            new() { Type = DescriptorType.SampledImage,             DescriptorCount = RenderConfig.MAX_BINDLESS_TEXTURES * RenderConfig.MAX_CONCURRENT_FRAMES },
            new() { Type = DescriptorType.Sampler,                  DescriptorCount = 8 * RenderConfig.MAX_CONCURRENT_FRAMES + 4 },
            // 5 g-buffer samplers (set 1) + 3 IBL samplers × MAX_FRAMES on set 0
            // for both PBR + Transparent pipelines + ImGui viewport + headroom.
            // Probe prefilter sets reuse iblCubeSampler — one CombinedImageSampler
            // each. MaxProbes (16) × MipLevels (9) = 144. Cheap to oversize.
            new() { Type = DescriptorType.CombinedImageSampler,     DescriptorCount = 32 + 200 },
            // PBR (1) + Transparent (1) + PT (MAX_FRAMES) + Wavefront (MAX_FRAMES) +
            // Pick (1) + SelectionMask (1) + headroom.
            new() { Type = DescriptorType.AccelerationStructureKhr, DescriptorCount = 16 },
            // IBL bake passes need StorageImage descriptors — one per dispatch.
            // Worst case is the prefilter chain (1 set × prefilteredCubeMipLevels
            // mips) + equirect→cube + irradiance + BRDF LUT ≈ 11 sets. Reflection
            // probes pre-allocate one set per (slot, mip): MaxProbes (16) × MipLevels
            // (9) = 144. Round up.
            new() { Type = DescriptorType.StorageImage,             DescriptorCount = 24 + 200 },
        };
        // Sizing is an app-level budget (above); GraphicsDevice owns the pool handle.
        // +16 over the historical 48+200 for the wavefront tracer's 5 sets (set 0 ×MAX_FRAMES,
        // sets 1/3/4) with headroom.
        gfx.CreateDescriptorPool(poolSizes, maxSets: 48 + 200 + 16,
            DescriptorPoolCreateFlags.UpdateAfterBindBit | DescriptorPoolCreateFlags.FreeDescriptorSetBit);
    }
   

    

    // Allocates a host-visible, coherent, persistently-mapped SSBO. Optional extra
    // usage bits let callers turn the same buffer into an indirect-cmd / indirect-
    // count source on top of plain storage usage.
    internal void CreateMappedStorageBuffer(ulong sizeBytes, ref UboBuffer ubo,
        BufferUsageFlags extraUsage = 0, bool preferDeviceLocal = false)
        => gfx.CreateMappedStorageBuffer(sizeBytes, ref ubo, extraUsage, preferDeviceLocal);

    // Lights SSBO — owned by GpuScene (L2). These forwarders keep the former
    // Renderer.GetLightStorageBuffer / Renderer.LightCount call sites (LightCull,
    // PBR-deferred, Transparent, PT, RT pipelines) compiling unchanged.

    /// <summary>Buffer holding packed PbrLightGpu records for the given frame
    /// slot. Bound by every pipeline that needs scene lighting. Stable for the
    /// renderer's lifetime.</summary>
    public Buffer GetLightStorageBuffer(uint frame) => gpuScene.GetLightStorageBuffer(frame);

    /// <summary>Number of valid lights packed by the most recent UpdateLights.</summary>
    public uint LightCount => gpuScene.LightCount;

    // Scene -> GPU packing moved to GpuScene.Extract. These forwarders keep the
    // per-frame DrawX call sites + PbrDeferredPipeline.Record compiling unchanged.

    /// <summary>Forwards to <see cref="GpuScene.UpdateMaterials"/>. See there.</summary>
    public uint UpdateMaterials(uint frameIndex, Scene scene) => gpuScene.UpdateMaterials(frameIndex, scene);

    /// <summary>Forwards to <see cref="GpuScene.UpdateLights"/>. See there.</summary>
    public uint  UpdateLights(uint frameIndex, Scene scene) => gpuScene.UpdateLights(frameIndex, scene);

   

    

    /// <summary>
    /// Writes a bindless texture to the given descriptor set at the given binding.
    /// All descriptor sets must use the same layout.
    /// </summary>
    /// <param name="index"></param>
    /// <param name="texture"></param>
    /// <param name="sets"></param>
    /// <param name="binding"></param>
    /// <exception cref="ArgumentOutOfRangeException"></exception>
    public void WriteBindlessTexture(int index, Texture texture, DescriptorSet[] sets, uint binding)
    {
        if (index < 0 || index >= RenderConfig.MAX_BINDLESS_TEXTURES)
            throw new ArgumentOutOfRangeException(nameof(index), $"bindless texture index {index} out of [0, {RenderConfig.MAX_BINDLESS_TEXTURES}).");

        // Mirror into the unified scene set's texture table at the same slot index, so
        // material rows resolve identically once shaders migrate to SceneBindings.
        descriptorRegistry?.SetBindlessSlot(index, texture.View);

        DescriptorImageInfo imgInfo = new()
        {
            ImageView = texture.View,
            ImageLayout = ImageLayout.ShaderReadOnlyOptimal,
        };

        var writes = stackalloc WriteDescriptorSet[sets.Length];
        for (var f = 0; f < sets.Length; f++)
        {
            writes[f] = new WriteDescriptorSet
            {
                SType = StructureType.WriteDescriptorSet,
                DstSet = sets[f],
                DstBinding = binding,
                DstArrayElement = (uint)index,
                DescriptorType = DescriptorType.SampledImage,
                DescriptorCount = 1,
                PImageInfo = &imgInfo,
            };
        }
        vk!.UpdateDescriptorSets(device, (uint)sets.Length, writes, 0, null);
    }
}

// Mirrors PbrUtils.slang::PbrLight under std430 (16B alignment, no padding needed —
// the struct is exactly 4 × float4 = 64B).

// Per-instance row in the geometry pipeline's instance SSBO. Shader sees this as
// PbrUtils.slang::InstanceData (std430 layout, stride 80B). Padding keeps the C#
// struct aligned to the shader struct so a Span<InstanceDataGPU> blits cleanly.
[StructLayout(LayoutKind.Sequential)]
public struct InstanceDataGPU
{
    public Matrix4x4 model;
    public uint materialIndex;
    public uint _pad0, _pad1, _pad2;
}

// Cull-pass input record. GpuScene.ExtractRenderables packs one per OPAQUE/MASK
// renderable into its cull-input SSBO each frame; the compute shader reads it and
// emits an indirect draw + InstanceData when the bounding sphere passes the frustum
// test. Std430 alignment, total 96B.

// CPU-side transparent draw record. Forward+ transparent pass walks a sorted
// list of these (back-to-front by view-space Z) and issues one push-constant +
// vkCmdDrawIndexed per entity. Not a GPU SSBO type — the per-draw model + matidx
// ride in push constants since the count is typically small.
public struct TransparentDraw
{
    public Matrix4x4 Model;
    public uint      MaterialIndex;
    public uint      IndexCount;
    public uint      FirstIndex;     // into Engine.ResourceManager.GlobalIndexBuffer
    public float     ViewDepth;      // view-space Z of the entity's origin; sort key
}

// Mirrors VkDrawIndexedIndirectCommand exactly. Compute writes this struct into the
// indirect-cmd buffer per surviving renderable; vkCmdDrawIndexedIndirectCount
// consumes them.
[StructLayout(LayoutKind.Sequential)]
public struct DrawIndexedIndirectCommandGpu
{
    public uint indexCount;
    public uint instanceCount;
    public uint firstIndex;
    public int  vertexOffset;
    public uint firstInstance;
}


// Per-frame mapped UBO/SSBO bundle. Storage is owned by the renderer's
// GpuMemoryAllocator; Dispose is intentionally absent — callers free through
// Renderer.DestroyBuffer(b.buffer, b.alloc) so both the VkBuffer handle AND
// the suballocation are released together.
unsafe struct UboBuffer
{
    public Buffer    buffer;
    public SubAlloc  alloc;
    public void*     mapped;
}

// Mirrors PbrUtils.slang::PbrMaterial under std430. First 64B holds the core
// glTF metallic-roughness block (laid out so the vec3 emissive factor's
// trailing 4B slack absorbs AlphaCutoff). The next 32B carries KHR extension
// data — transmission, IOR, clearcoat. All scalar / 4B-aligned so std430
// packs them with zero padding. Fields the shaders don't yet consume sit
// here as data only; uploaded every frame regardless. Total: 96B.

public static class PbrMaterialVolume
{
    // Sentinel for "no participating volume" — matches the shader guard
    // (mediumSigmaA treats >= 1e6 as no absorption). glTF's real default is
    // +infinity; a large finite value uploads cleanly and reads identically.
    public const float NoAbsorptionDistance = 1e30f;
}

public enum AlphaMode : uint
{
    Opaque = 0,
    Mask   = 1,
    Blend  = 2,
}

public static class PbrMaterialAlphaExtensions
{
    // BLEND takes precedence over MASK if both bits are accidentally set
    // (shouldn't happen from glTF — they're mutually exclusive there).
    public static AlphaMode GetAlphaMode(this in PbrMaterial mat)
    {
        if ((mat.Flags & 0x4u) != 0u) return AlphaMode.Blend;
        if ((mat.Flags & 0x1u) != 0u) return AlphaMode.Mask;
        return AlphaMode.Opaque;
    }
}

public unsafe class ImageResource : IDisposable
{
    private readonly Vk     _vk;
    private readonly Device _device;
    private readonly GpuMemoryAllocator _allocator;
    private bool            _disposed;
    public bool IsAllocated { get; set; }

    public string _name { get; }
    public Format _format;
    Extent2D _extent;
    ImageUsageFlags _usage;
    public ImageLayout _initialLayout;
    public ImageLayout _finalLayout;


    public VkImage Image;
    public SubAlloc ImageAlloc;
    public ImageView ImageView;


    public ImageResource(
        Vk vk, Device device,
        string name, Format format, Extent2D extent,
        ImageUsageFlags usage,
        ImageLayout initialLayout = ImageLayout.Undefined,
        ImageLayout finalLayout   = ImageLayout.ShaderReadOnlyOptimal)
    {
        _vk = vk;
        _device = device;
        _allocator = Engine.renderer!.memAllocator;
        _name = name;
        _format = format;
        _extent = extent;
        _usage = usage;
        _initialLayout = initialLayout;
        _finalLayout = finalLayout;
    }

    /// <summary>
    /// Adopts already-allocated image handles. Used for resources whose creation
    /// flow doesn't fit the graph's <see cref="Allocate"/> path (e.g. textures
    /// uploaded from disk via a staging buffer). Dispose semantics match the
    /// allocate path — handles are destroyed on Dispose().
    /// </summary>
    public ImageResource(
        Vk vk, Device device,
        string name, Format format, Extent2D extent,
        ImageUsageFlags usage,
        VkImage image, SubAlloc alloc, ImageView view)
    {
        _vk = vk;
        _device = device;
        _allocator = Engine.renderer!.memAllocator;
        _name = name;
        _format = format;
        _extent = extent;
        _usage = usage;
        _initialLayout = ImageLayout.ShaderReadOnlyOptimal;
        _finalLayout   = ImageLayout.ShaderReadOnlyOptimal;
        Image = image;
        ImageAlloc = alloc;
        ImageView = view;
        IsAllocated = true;
    }
    public void Allocate(PhysicalDevice physicalDevice, float priority = GpuMemoryAllocator.PriorityDefault)
    {
        //configure image creation info based on resource properties
        ImageCreateInfo imageInfo = new()
        {
            SType = StructureType.ImageCreateInfo,
            ImageType = ImageType.Type2D,
            Format = _format,
            Extent = new Extent3D(_extent.Width, _extent.Height, 1),
            MipLevels = 1,
            ArrayLayers = 1,
            Samples = SampleCountFlags.Count1Bit,
            Tiling = ImageTiling.Optimal,
            Usage = _usage,
            SharingMode = SharingMode.Exclusive,
            InitialLayout = _initialLayout
        };
        if (_vk.CreateImage(_device, ref imageInfo, null, out Image) != Result.Success)
        {
            throw new Exception("Failed to create image for resource " + _name);
        }

        ImageAlloc = _allocator.AllocateForImage(Image, MemoryPropertyFlags.DeviceLocalBit, ImageTiling.Optimal, priority);

        bool isDepth = _format is Format.D32Sfloat
            or Format.D24UnormS8Uint
            or Format.D16Unorm;

        var viewInfo = new ImageViewCreateInfo()
        {
            SType = StructureType.ImageViewCreateInfo,
            Image = Image,
            ViewType = ImageViewType.Type2D,
            Format = _format,
            SubresourceRange = new ImageSubresourceRange()
            {
                AspectMask = isDepth ? ImageAspectFlags.DepthBit : ImageAspectFlags.ColorBit,
                BaseMipLevel = 0,
                LevelCount = 1,
                BaseArrayLayer = 0,
                LayerCount = 1
            }
        };

        if (_vk.CreateImageView(_device, ref viewInfo, null, out ImageView) != Result.Success)
        {
            throw new Exception("Failed to create image view for resource " + _name);
        };
        IsAllocated = true;

    }



    public void Dispose()
    {
        if (_disposed) return;

        // Destroy in reverse-creation order — mirrors C++ RAII destruction
        if (ImageView.Handle   != 0) _vk.DestroyImageView(_device, ImageView,   null);
        if (Image.Handle  != 0)      _vk.DestroyImage    (_device, Image,  null);
        _allocator.Free(ImageAlloc);

        _disposed = true;
    }

}

/// <summary>
/// Composes an <see cref="ImageResource"/> + <see cref="Sampler"/> as a single
/// shader-readable texture object. Use for sampled textures (loaded from disk,
/// dummy fallbacks, baked LUTs); <see cref="ImageResource"/> alone is reserved
/// for render-graph attachments.
/// </summary>
public unsafe class Texture : IDisposable
{
    private readonly Vk _vk;
    private readonly Device _device;
    private bool _disposed;

    public ImageResource Resource { get; }
    public Sampler Sampler { get; }

    public VkImage Image     => Resource.Image;
    public ImageView View    => Resource.ImageView;

    public Texture(Vk vk, Device device, ImageResource resource, Sampler sampler)
    {
        _vk = vk;
        _device = device;
        Resource = resource;
        Sampler = sampler;
    }

    /// <summary>
    /// Loads a 2D RGBA texture from disk into a device-local image + linear sampler.
    /// </summary>
    public static Texture CreateTextureFromPath(GraphicsDevice gfx, string path, Format format)
    {
        using var img = SixLabors.ImageSharp.Image.Load<SixLabors.ImageSharp.PixelFormats.Rgba32>(path);
        var pixels = new SixLabors.ImageSharp.PixelFormats.Rgba32[img.Width * img.Height];
        img.CopyPixelDataTo(pixels);
        fixed (SixLabors.ImageSharp.PixelFormats.Rgba32* p = pixels)
        {
            return CreateTextureFromMemory(gfx, (byte*)p,
                (uint)img.Width, (uint)img.Height, format,
                new Extent3D((uint)img.Width, (uint)img.Height, 1));
        }
    }

    /// <summary>
    /// Uploads raw RGBA pixel data into a device-local image + linear sampler.
    /// <paramref name="width"/>/<paramref name="height"/> size the staging copy;
    /// <paramref name="extent"/> sizes the destination image.
    /// </summary>
    public static Texture CreateTextureFromMemory(GraphicsDevice gfx, byte* pixels,
        uint width, uint height, Format format, Extent3D extent)
    {
        var vk = gfx.Vk!;
        var device = gfx.Device;
        var physicalDevice = gfx.PhysicalDevice;

        ulong imageSize = (ulong)width * (ulong)height * 4UL;
        uint mipLevels = (uint)Math.Floor(Math.Log2(Math.Max(width, height))) + 1;

        // Stage pixels in a host-visible buffer.
        gfx.CreateBuffer(imageSize, BufferUsageFlags.TransferSrcBit,
            MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit,
            out var staging, out var stagingAlloc);

        void* mapped = gfx.Allocator.GetMapped(stagingAlloc);
        System.Buffer.MemoryCopy(pixels, mapped, (long)imageSize, (long)imageSize);

        const ImageUsageFlags usage = ImageUsageFlags.TransferDstBit |ImageUsageFlags.TransferSrcBit | ImageUsageFlags.SampledBit;
        var extent2D = new Extent2D(extent.Width, extent.Height);

        // Device-local image.
        ImageCreateInfo imageInfo = new()
        {
            SType = StructureType.ImageCreateInfo,
            ImageType = ImageType.Type2D,
            Format = format,
            Extent = extent,
            MipLevels = mipLevels,
            ArrayLayers = 1,
            Samples = SampleCountFlags.Count1Bit,
            Tiling = ImageTiling.Optimal,
            Usage = usage,
            SharingMode = SharingMode.Exclusive,
            InitialLayout = ImageLayout.Undefined,
        };
        if (vk.CreateImage(device, &imageInfo, null, out var image) != Result.Success)
            throw new Exception("Failed to create image for texture");

        var imageAlloc = gfx.Allocator.AllocateForImage(image, MemoryPropertyFlags.DeviceLocalBit);

        // Transition → copy → transition (single-time command).
        var cmd = gfx.BeginSingleTimeCommands();
        gfx.TransitionImageLayout(cmd, image, format, ImageLayout.Undefined, ImageLayout.TransferDstOptimal, mipLevels: mipLevels);
        BufferImageCopy region = new()
        {
            BufferOffset = 0,
            BufferRowLength = 0,
            BufferImageHeight = 0,
            ImageSubresource = new ImageSubresourceLayers
            {
                AspectMask = ImageAspectFlags.ColorBit,
                MipLevel = 0,
                BaseArrayLayer = 0,
                LayerCount = 1,
            },
            ImageOffset = new Offset3D(0, 0, 0),
            ImageExtent = extent,
        };
        vk.CmdCopyBufferToImage(cmd, staging, image, ImageLayout.TransferDstOptimal, 1, &region);
        gfx.GenerateMipMaps(cmd, image, format, width, height,mipLevels);
        gfx.EndSingleTimeCommands(cmd);

        gfx.DestroyBuffer(staging, stagingAlloc);

        ImageViewCreateInfo viewInfo = new()
        {
            SType = StructureType.ImageViewCreateInfo,
            Image = image,
            ViewType = ImageViewType.Type2D,
            Format = format,
            SubresourceRange = new ImageSubresourceRange
            {
                AspectMask = ImageAspectFlags.ColorBit,
                BaseMipLevel = 0, LevelCount = mipLevels,
                BaseArrayLayer = 0, LayerCount = 1,
            },
        };
        vk.CreateImageView(device, &viewInfo, null, out var view);

        vk.GetPhysicalDeviceProperties(physicalDevice, out var props);
        SamplerCreateInfo samplerInfo = new()
        {
            SType = StructureType.SamplerCreateInfo,
            MagFilter = Filter.Linear,
            MinFilter = Filter.Linear,
            AddressModeU = SamplerAddressMode.Repeat,
            AddressModeV = SamplerAddressMode.Repeat,
            AddressModeW = SamplerAddressMode.Repeat,
            AnisotropyEnable = true,
            MaxAnisotropy = props.Limits.MaxSamplerAnisotropy,
            BorderColor = BorderColor.FloatOpaqueBlack,
            UnnormalizedCoordinates = false,
            CompareEnable = false,
            CompareOp = CompareOp.Always,
            MipmapMode = SamplerMipmapMode.Linear,
            MinLod = 0.0f,
            MaxLod = Vk.LodClampNone,
            MipLodBias = 0.0f,
        };
        vk.CreateSampler(device, &samplerInfo, null, out var sampler);

        var resource = new ImageResource(vk, device, "FontTexture", format, extent2D, usage, image, imageAlloc, view);
        return new Texture(vk, device, resource, sampler);
    } 

    /// <summary>True for the BC1/BC3/BC4/BC5 block formats the loader compresses material textures to.</summary>
    public static bool IsBcFormat(Format f) => f switch
    {
        Format.BC1RgbUnormBlock or Format.BC1RgbSrgbBlock or Format.BC1RgbaUnormBlock or Format.BC1RgbaSrgbBlock
            or Format.BC3UnormBlock or Format.BC3SrgbBlock or Format.BC4UnormBlock or Format.BC5UnormBlock => true,
        _ => false,
    };

    /// <summary>The encoder mode that produces <paramref name="f"/>.</summary>
    public static Features.TextureCompression.BcMode BcModeFor(Format f) => f switch
    {
        Format.BC3UnormBlock or Format.BC3SrgbBlock => Features.TextureCompression.BcMode.Bc3,
        Format.BC5UnormBlock                        => Features.TextureCompression.BcMode.Bc5,
        Format.BC4UnormBlock                        => Features.TextureCompression.BcMode.Bc4,
        _                                           => Features.TextureCompression.BcMode.Bc1,
    };

    /// <summary>Uncompressed RGBA8 fallback for a BC format, used when textureCompressionBC is absent.
    /// Preserves the sRGB-ness so colour textures still decode correctly.</summary>
    public static Format BcFallbackFormat(Format f) => f switch
    {
        Format.BC1RgbSrgbBlock or Format.BC1RgbaSrgbBlock or Format.BC3SrgbBlock => Format.R8G8B8A8Srgb,
        _                                                                        => Format.R8G8B8A8Unorm,
    };

    /// <summary>
    /// Uploads raw RGBA pixel data, then GPU-compresses it to <paramref name="bcFormat"/> (Bc1/3/4/5)
    /// via <see cref="Features.TextureCompression.BcEncoder"/>. Flow: stage RGBA8 -> blit-generate the
    /// mip chain (uncompressed, as for any texture) -> compute-encode each mip into a packed buffer ->
    /// copy the blocks into the BC image. The source is read raw-UNORM so the BC image carries the
    /// sRGB-ness for the final sample (no double decode). Two single-time submits: the queue-idle
    /// between them makes the mip chain visible to the encoder without a cross-stage barrier.
    /// </summary>
    public static Texture CreateCompressedTexture(GraphicsDevice gfx,
        Features.TextureCompression.BcEncoder encoder, byte* pixels,
        uint width, uint height, Format bcFormat)
    {
        var vk     = gfx.Vk!;
        var device = gfx.Device;
        var mode   = BcModeFor(bcFormat);

        ulong imageSize = (ulong)width * height * 4UL;
        uint  mipLevels = (uint)Math.Floor(Math.Log2(Math.Max(width, height))) + 1;
        var   extent    = new Extent3D(width, height, 1);
        var   extent2D  = new Extent2D(width, height);

        // ---- 1. Source RGBA8 (raw UNORM) image with a full mip chain -------------------------
        const Format srcFormat = Format.R8G8B8A8Unorm;
        const ImageUsageFlags srcUsage = ImageUsageFlags.TransferDstBit | ImageUsageFlags.TransferSrcBit | ImageUsageFlags.SampledBit;

        gfx.CreateBuffer(imageSize, BufferUsageFlags.TransferSrcBit,
            MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit,
            out var staging, out var stagingAlloc);
        System.Buffer.MemoryCopy(pixels, gfx.Allocator.GetMapped(stagingAlloc), (long)imageSize, (long)imageSize);

        ImageCreateInfo srcInfo = new()
        {
            SType = StructureType.ImageCreateInfo, ImageType = ImageType.Type2D, Format = srcFormat,
            Extent = extent, MipLevels = mipLevels, ArrayLayers = 1, Samples = SampleCountFlags.Count1Bit,
            Tiling = ImageTiling.Optimal, Usage = srcUsage, SharingMode = SharingMode.Exclusive,
            InitialLayout = ImageLayout.Undefined,
        };
        if (vk.CreateImage(device, &srcInfo, null, out var srcImage) != Result.Success)
            throw new Exception("Failed to create BC encode source image");
        var srcAlloc = gfx.Allocator.AllocateForImage(srcImage, MemoryPropertyFlags.DeviceLocalBit);

        var cmd1 = gfx.BeginSingleTimeCommands();
        gfx.TransitionImageLayout(cmd1, srcImage, srcFormat, ImageLayout.Undefined, ImageLayout.TransferDstOptimal, mipLevels);
        BufferImageCopy mip0 = new()
        {
            BufferOffset = 0, BufferRowLength = 0, BufferImageHeight = 0,
            ImageSubresource = new ImageSubresourceLayers { AspectMask = ImageAspectFlags.ColorBit, MipLevel = 0, BaseArrayLayer = 0, LayerCount = 1 },
            ImageOffset = new Offset3D(0, 0, 0), ImageExtent = extent,
        };
        vk.CmdCopyBufferToImage(cmd1, staging, srcImage, ImageLayout.TransferDstOptimal, 1, &mip0);
        gfx.GenerateMipMaps(cmd1, srcImage, srcFormat, width, height, mipLevels);   // leaves ShaderReadOnlyOptimal
        gfx.EndSingleTimeCommands(cmd1);
        gfx.DestroyBuffer(staging, stagingAlloc);

        ImageViewCreateInfo srcViewInfo = new()
        {
            SType = StructureType.ImageViewCreateInfo, Image = srcImage, ViewType = ImageViewType.Type2D, Format = srcFormat,
            SubresourceRange = new ImageSubresourceRange { AspectMask = ImageAspectFlags.ColorBit, BaseMipLevel = 0, LevelCount = mipLevels, BaseArrayLayer = 0, LayerCount = 1 },
        };
        vk.CreateImageView(device, &srcViewInfo, null, out var srcView);

        // ---- 2. BC destination image + packed block buffer -----------------------------------
        ulong packedBytes = Features.TextureCompression.BcEncoder.PackedSize(mode, width, height, mipLevels);
        gfx.CreateBuffer(packedBytes, BufferUsageFlags.StorageBufferBit | BufferUsageFlags.TransferSrcBit,
            MemoryPropertyFlags.DeviceLocalBit, out var blockBuf, out var blockAlloc);

        const ImageUsageFlags dstUsage = ImageUsageFlags.TransferDstBit | ImageUsageFlags.SampledBit;
        ImageCreateInfo dstInfo = new()
        {
            SType = StructureType.ImageCreateInfo, ImageType = ImageType.Type2D, Format = bcFormat,
            Extent = extent, MipLevels = mipLevels, ArrayLayers = 1, Samples = SampleCountFlags.Count1Bit,
            Tiling = ImageTiling.Optimal, Usage = dstUsage, SharingMode = SharingMode.Exclusive,
            InitialLayout = ImageLayout.Undefined,
        };
        if (vk.CreateImage(device, &dstInfo, null, out var bcImage) != Result.Success)
            throw new Exception("Failed to create BC texture image");
        var bcAlloc = gfx.Allocator.AllocateForImage(bcImage, MemoryPropertyFlags.DeviceLocalBit);

        // ---- 3. Encode each mip, then copy the blocks into the BC image ----------------------
        var cmd2 = gfx.BeginSingleTimeCommands();
        encoder.RecordEncode(cmd2, srcView, blockBuf, mode, width, height, mipLevels);

        BufferMemoryBarrier toCopy = new()
        {
            SType = StructureType.BufferMemoryBarrier,
            SrcAccessMask = AccessFlags.ShaderWriteBit, DstAccessMask = AccessFlags.TransferReadBit,
            SrcQueueFamilyIndex = Vk.QueueFamilyIgnored, DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
            Buffer = blockBuf, Offset = 0, Size = Vk.WholeSize,
        };
        vk.CmdPipelineBarrier(cmd2, PipelineStageFlags.ComputeShaderBit, PipelineStageFlags.TransferBit, 0, 0, null, 1, &toCopy, 0, null);

        gfx.TransitionImageLayout(cmd2, bcImage, bcFormat, ImageLayout.Undefined, ImageLayout.TransferDstOptimal, mipLevels);
        for (uint m = 0; m < mipLevels; m++)
        {
            uint mw = Math.Max(1u, width >> (int)m), mh = Math.Max(1u, height >> (int)m);
            BufferImageCopy region = new()
            {
                BufferOffset = Features.TextureCompression.BcEncoder.MipOffset(mode, width, height, m),
                BufferRowLength = 0, BufferImageHeight = 0,
                ImageSubresource = new ImageSubresourceLayers { AspectMask = ImageAspectFlags.ColorBit, MipLevel = m, BaseArrayLayer = 0, LayerCount = 1 },
                ImageOffset = new Offset3D(0, 0, 0), ImageExtent = new Extent3D(mw, mh, 1),
            };
            vk.CmdCopyBufferToImage(cmd2, blockBuf, bcImage, ImageLayout.TransferDstOptimal, 1, &region);
        }
        gfx.TransitionImageLayout(cmd2, bcImage, bcFormat, ImageLayout.TransferDstOptimal, ImageLayout.ShaderReadOnlyOptimal, mipLevels);
        gfx.EndSingleTimeCommands(cmd2);

        // ---- 4. Free the transient source + block buffer; build the sampled BC texture -------
        gfx.DestroyBuffer(blockBuf, blockAlloc);
        vk.DestroyImageView(device, srcView, null);
        vk.DestroyImage(device, srcImage, null);
        gfx.Allocator.Free(srcAlloc);

        ImageViewCreateInfo bcViewInfo = new()
        {
            SType = StructureType.ImageViewCreateInfo, Image = bcImage, ViewType = ImageViewType.Type2D, Format = bcFormat,
            SubresourceRange = new ImageSubresourceRange { AspectMask = ImageAspectFlags.ColorBit, BaseMipLevel = 0, LevelCount = mipLevels, BaseArrayLayer = 0, LayerCount = 1 },
        };
        vk.CreateImageView(device, &bcViewInfo, null, out var bcView);

        vk.GetPhysicalDeviceProperties(gfx.PhysicalDevice, out var props);
        SamplerCreateInfo samplerInfo = new()
        {
            SType = StructureType.SamplerCreateInfo, MagFilter = Filter.Linear, MinFilter = Filter.Linear,
            AddressModeU = SamplerAddressMode.Repeat, AddressModeV = SamplerAddressMode.Repeat, AddressModeW = SamplerAddressMode.Repeat,
            AnisotropyEnable = true, MaxAnisotropy = props.Limits.MaxSamplerAnisotropy, BorderColor = BorderColor.FloatOpaqueBlack,
            UnnormalizedCoordinates = false, CompareEnable = false, CompareOp = CompareOp.Always,
            MipmapMode = SamplerMipmapMode.Linear, MinLod = 0.0f, MaxLod = Vk.LodClampNone, MipLodBias = 0.0f,
        };
        vk.CreateSampler(device, &samplerInfo, null, out var sampler);

        var resource = new ImageResource(vk, device, "BcTexture", bcFormat, extent2D, dstUsage, bcImage, bcAlloc, bcView);
        return new Texture(vk, device, resource, sampler);
    }

    public void Dispose()
    {
        if (_disposed) return;
        if (Sampler.Handle != 0) _vk.DestroySampler(_device, Sampler, null);
        Resource?.Dispose();
        _disposed = true;
    }
}
