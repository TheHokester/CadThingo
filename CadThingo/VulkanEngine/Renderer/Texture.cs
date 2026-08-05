using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer;

/// <summary>
/// A device image plus its view and suballocation, destroyed together on Dispose. Reserved for
/// render-graph attachments; use <see cref="Texture"/> when a sampler belongs with the image.
/// </summary>
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

    /// <summary>Describes an image the caller will later <see cref="Allocate"/>.</summary>
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

    /// <summary>Adopts already-allocated handles, for resources whose creation does not fit the
    /// graph's <see cref="Allocate"/> path, such as a texture staged in from disk. Dispose destroys
    /// them exactly as it does on the allocate path.</summary>
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

    /// <summary>Creates the image, binds device-local memory and builds a single-mip view.</summary>
    /// <param name="priority">Eviction priority handed to the block suballocator.</param>
    public void Allocate(PhysicalDevice physicalDevice, float priority = GpuMemoryAllocator.PriorityDefault)
    {
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

        // Reverse creation order: the view references the image.
        if (ImageView.Handle   != 0) _vk.DestroyImageView(_device, ImageView,   null);
        if (Image.Handle  != 0)      _vk.DestroyImage    (_device, Image,  null);
        _allocator.Free(ImageAlloc);

        _disposed = true;
    }

}

/// <summary>
/// Composes an <see cref="ImageResource"/> and a <see cref="Sampler"/> as one shader-readable
/// texture. Use for sampled textures: loaded from disk, dummy fallbacks, baked LUTs.
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

    /// <summary>Loads a 2D RGBA texture from disk into a device-local image plus linear
    /// sampler.</summary>
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

    /// <summary>Uploads raw RGBA pixels into a device-local image plus linear sampler.</summary>
    /// <param name="width">Sizes the staging copy, alongside <paramref name="height"/>.</param>
    /// <param name="extent">Sizes the destination image.</param>
    public static Texture CreateTextureFromMemory(GraphicsDevice gfx, byte* pixels,
        uint width, uint height, Format format, Extent3D extent)
    {
        var vk = gfx.Vk!;
        var device = gfx.Device;
        var physicalDevice = gfx.PhysicalDevice;

        ulong imageSize = (ulong)width * (ulong)height * 4UL;
        uint mipLevels = (uint)Math.Floor(Math.Log2(Math.Max(width, height))) + 1;

        // Stage the pixels in a host-visible buffer.
        gfx.CreateBuffer(imageSize, BufferUsageFlags.TransferSrcBit,
            MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit,
            out var staging, out var stagingAlloc);

        void* mapped = gfx.Allocator.GetMapped(stagingAlloc);
        System.Buffer.MemoryCopy(pixels, mapped, (long)imageSize, (long)imageSize);

        const ImageUsageFlags usage = ImageUsageFlags.TransferDstBit |ImageUsageFlags.TransferSrcBit | ImageUsageFlags.SampledBit;
        var extent2D = new Extent2D(extent.Width, extent.Height);

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

        // Transition, copy, generate mips, all in one single-time command.
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

    /// <summary>True for the BC1/BC3/BC4/BC5 block formats the loader compresses material textures
    /// to.</summary>
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

    /// <summary>Uncompressed RGBA8 fallback for a BC format, used when textureCompressionBC is
    /// absent. Preserves the sRGB-ness so colour textures still decode correctly.</summary>
    public static Format BcFallbackFormat(Format f) => f switch
    {
        Format.BC1RgbSrgbBlock or Format.BC1RgbaSrgbBlock or Format.BC3SrgbBlock => Format.R8G8B8A8Srgb,
        _                                                                        => Format.R8G8B8A8Unorm,
    };

    /// <summary>
    /// Uploads raw RGBA pixels, then GPU-compresses them to <paramref name="bcFormat"/> through
    /// <see cref="Features.TextureCompression.BcEncoder"/>. Stages RGBA8, blit-generates the mip
    /// chain uncompressed, compute-encodes each mip into a packed buffer, then copies the blocks
    /// into the BC image. The source is read raw-UNORM so the BC image carries the sRGB-ness for
    /// the final sample and nothing decodes twice. Two single-time submits: the queue idle between
    /// them makes the mip chain visible to the encoder without a cross-stage barrier.
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