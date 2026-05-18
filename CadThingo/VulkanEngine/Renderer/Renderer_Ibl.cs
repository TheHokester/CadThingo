using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer;

// Image-Based Lighting resources — split-sum approximation.
//
//   envCube         512² × 6, mipped       full env, sampled by skybox + prefilter
//   irradianceCube   32² × 6                Lambert hemisphere convolution (diffuse)
//   prefilteredCube 128² × 6, 5 mips        GGX importance-sampled specular (per-roughness mip)
//   brdfLut         512² × 1                split-sum 2D LUT (NdotV × roughness)
//
// Phase 1 just allocates and clears them; the compute bake passes that fill the
// content live in Phase 2, the PBR shader that reads them lives in Phase 3.
// The PBR pipeline binds the cube views unconditionally — black content yields
// zero IBL contribution until an HDR is loaded.

public unsafe partial class Renderer
{
    // Face sizes. envCube needs to be large enough that prefilter taps don't
    // alias; 512 is the standard sweet spot. brdfLut at 512 gives plenty of
    // resolution along both axes without measurable cost.
    internal const uint EnvCubeFaceSize        = 512;
    internal const uint IrradianceCubeFaceSize = 32;
    internal const uint PrefilteredCubeFaceSize = 128;
    internal const uint BrdfLutSize            = 512;

    // Image handles. Held as raw Vulkan handles (rather than ImageResource)
    // because ImageResource hard-codes 2D / single layer / single mip — cubemaps
    // need ArrayLayers=6 + CubeCompatible + variable mips.
    internal VkImage        envCubeImage;
    internal DeviceMemory   envCubeMemory;
    internal ImageView      envCubeView;        // VK_IMAGE_VIEW_TYPE_CUBE, all mips
    internal uint           envCubeMipLevels;

    internal VkImage        irradianceCubeImage;
    internal DeviceMemory   irradianceCubeMemory;
    internal ImageView      irradianceCubeView;

    internal VkImage        prefilteredCubeImage;
    internal DeviceMemory   prefilteredCubeMemory;
    internal ImageView      prefilteredCubeView;
    internal uint           prefilteredCubeMipLevels;

    internal VkImage        brdfLutImage;
    internal DeviceMemory   brdfLutMemory;
    internal ImageView      brdfLutView;

    // Two shared samplers — one for the cubemaps (linear + mipmap linear,
    // ClampToEdge so face seams don't bleed at low mips), one for the BRDF LUT
    // (linear, no mips, ClampToEdge).
    internal Sampler        iblCubeSampler;
    internal Sampler        iblLutSampler;

    internal void CreateIblResources()
    {
        envCubeMipLevels         = (uint)System.Math.Floor(System.Math.Log2(EnvCubeFaceSize)) + 1;
        prefilteredCubeMipLevels = (uint)System.Math.Floor(System.Math.Log2(PrefilteredCubeFaceSize)) + 1;

        // ── env cube — sampled, storage-written by EquirectToCube, blitted for mips
        CreateCubemapImage(
            EnvCubeFaceSize, envCubeMipLevels, Format.R16G16B16A16Sfloat,
            ImageUsageFlags.SampledBit | ImageUsageFlags.StorageBit |
            ImageUsageFlags.TransferDstBit | ImageUsageFlags.TransferSrcBit,
            out envCubeImage, out envCubeMemory, out envCubeView);

        // ── irradiance cube — sampled + storage-written
        CreateCubemapImage(
            IrradianceCubeFaceSize, 1, Format.R16G16B16A16Sfloat,
            ImageUsageFlags.SampledBit | ImageUsageFlags.StorageBit | ImageUsageFlags.TransferDstBit,
            out irradianceCubeImage, out irradianceCubeMemory, out irradianceCubeView);

        // ── prefiltered cube — sampled + storage-written per mip
        CreateCubemapImage(
            PrefilteredCubeFaceSize, prefilteredCubeMipLevels, Format.R16G16B16A16Sfloat,
            ImageUsageFlags.SampledBit | ImageUsageFlags.StorageBit | ImageUsageFlags.TransferDstBit,
            out prefilteredCubeImage, out prefilteredCubeMemory, out prefilteredCubeView);

        // ── BRDF LUT — 2D, sampled + storage-written, no mips
        CreateLutImage(
            BrdfLutSize, Format.R16G16Sfloat,
            ImageUsageFlags.SampledBit | ImageUsageFlags.StorageBit | ImageUsageFlags.TransferDstBit,
            out brdfLutImage, out brdfLutMemory, out brdfLutView);

        // ── Samplers ─────────────────────────────────────────────
        SamplerCreateInfo cubeSamplerInfo = new()
        {
            SType            = StructureType.SamplerCreateInfo,
            MagFilter        = Filter.Linear,
            MinFilter        = Filter.Linear,
            AddressModeU     = SamplerAddressMode.ClampToEdge,
            AddressModeV     = SamplerAddressMode.ClampToEdge,
            AddressModeW     = SamplerAddressMode.ClampToEdge,
            AnisotropyEnable = false,
            BorderColor      = BorderColor.FloatOpaqueBlack,
            CompareEnable    = false,
            CompareOp        = CompareOp.Always,
            MipmapMode       = SamplerMipmapMode.Linear,
            MinLod           = 0f,
            MaxLod           = Vk.LodClampNone,
            MipLodBias       = 0f,
        };
        if (vk!.CreateSampler(device, &cubeSamplerInfo, null, out iblCubeSampler) != Result.Success)
            throw new System.Exception("Failed to create IBL cube sampler");

        SamplerCreateInfo lutSamplerInfo = new()
        {
            SType         = StructureType.SamplerCreateInfo,
            MagFilter     = Filter.Linear,
            MinFilter     = Filter.Linear,
            AddressModeU  = SamplerAddressMode.ClampToEdge,
            AddressModeV  = SamplerAddressMode.ClampToEdge,
            AddressModeW  = SamplerAddressMode.ClampToEdge,
            MipmapMode    = SamplerMipmapMode.Linear,
            MinLod        = 0f,
            MaxLod        = 0f,
            CompareEnable = false,
            CompareOp     = CompareOp.Always,
        };
        if (vk!.CreateSampler(device, &lutSamplerInfo, null, out iblLutSampler) != Result.Success)
            throw new System.Exception("Failed to create IBL LUT sampler");

        // Phase 1 leaves the content black so descriptor binds are well-defined
        // even before an HDR is loaded — the PBR shader will read 0s and add 0
        // ambient. Phase 2's compute passes overwrite this content.
        InitializeIblImagesBlack();
    }

    void CreateCubemapImage(
        uint faceSize, uint mipLevels, Format format, ImageUsageFlags usage,
        out VkImage image, out DeviceMemory memory, out ImageView cubeView)
    {
        ImageCreateInfo info = new()
        {
            SType         = StructureType.ImageCreateInfo,
            ImageType     = ImageType.Type2D,
            Format        = format,
            Extent        = new Extent3D(faceSize, faceSize, 1),
            MipLevels     = mipLevels,
            ArrayLayers   = 6,
            Samples       = SampleCountFlags.Count1Bit,
            Tiling        = ImageTiling.Optimal,
            Usage         = usage,
            SharingMode   = SharingMode.Exclusive,
            InitialLayout = ImageLayout.Undefined,
            Flags         = ImageCreateFlags.CreateCubeCompatibleBit,
        };
        if (vk!.CreateImage(device, &info, null, out image) != Result.Success)
            throw new System.Exception("Failed to create IBL cubemap image");

        vk!.GetImageMemoryRequirements(device, image, out var memReqs);
        MemoryAllocateInfo alloc = new()
        {
            SType           = StructureType.MemoryAllocateInfo,
            AllocationSize  = memReqs.Size,
            MemoryTypeIndex = FindMemoryType(vk, physicalDevice, memReqs.MemoryTypeBits,
                MemoryPropertyFlags.DeviceLocalBit),
        };
        if (vk!.AllocateMemory(device, &alloc, null, out memory) != Result.Success)
            throw new System.Exception("Failed to allocate IBL cubemap memory");
        vk!.BindImageMemory(device, image, memory, 0);

        ImageViewCreateInfo viewInfo = new()
        {
            SType    = StructureType.ImageViewCreateInfo,
            Image    = image,
            ViewType = ImageViewType.TypeCube,
            Format   = format,
            SubresourceRange = new ImageSubresourceRange
            {
                AspectMask     = ImageAspectFlags.ColorBit,
                BaseMipLevel   = 0,
                LevelCount     = mipLevels,
                BaseArrayLayer = 0,
                LayerCount     = 6,
            },
        };
        if (vk!.CreateImageView(device, &viewInfo, null, out cubeView) != Result.Success)
            throw new System.Exception("Failed to create IBL cubemap view");
    }

    void CreateLutImage(
        uint size, Format format, ImageUsageFlags usage,
        out VkImage image, out DeviceMemory memory, out ImageView view)
    {
        ImageCreateInfo info = new()
        {
            SType         = StructureType.ImageCreateInfo,
            ImageType     = ImageType.Type2D,
            Format        = format,
            Extent        = new Extent3D(size, size, 1),
            MipLevels     = 1,
            ArrayLayers   = 1,
            Samples       = SampleCountFlags.Count1Bit,
            Tiling        = ImageTiling.Optimal,
            Usage         = usage,
            SharingMode   = SharingMode.Exclusive,
            InitialLayout = ImageLayout.Undefined,
        };
        if (vk!.CreateImage(device, &info, null, out image) != Result.Success)
            throw new System.Exception("Failed to create BRDF LUT image");

        vk!.GetImageMemoryRequirements(device, image, out var memReqs);
        MemoryAllocateInfo alloc = new()
        {
            SType           = StructureType.MemoryAllocateInfo,
            AllocationSize  = memReqs.Size,
            MemoryTypeIndex = FindMemoryType(vk, physicalDevice, memReqs.MemoryTypeBits,
                MemoryPropertyFlags.DeviceLocalBit),
        };
        if (vk!.AllocateMemory(device, &alloc, null, out memory) != Result.Success)
            throw new System.Exception("Failed to allocate BRDF LUT memory");
        vk!.BindImageMemory(device, image, memory, 0);

        ImageViewCreateInfo viewInfo = new()
        {
            SType    = StructureType.ImageViewCreateInfo,
            Image    = image,
            ViewType = ImageViewType.Type2D,
            Format   = format,
            SubresourceRange = new ImageSubresourceRange
            {
                AspectMask     = ImageAspectFlags.ColorBit,
                BaseMipLevel   = 0,
                LevelCount     = 1,
                BaseArrayLayer = 0,
                LayerCount     = 1,
            },
        };
        if (vk!.CreateImageView(device, &viewInfo, null, out view) != Result.Success)
            throw new System.Exception("Failed to create BRDF LUT view");
    }

    /// <summary>
    /// One-shot clear-to-black + transition to ShaderReadOnlyOptimal for all four
    /// IBL images. Phase 2's compute bake passes transition individual mips back
    /// to General as needed.
    /// </summary>
    void InitializeIblImagesBlack()
    {
        var cmd = BeginSingleTimeCommands();

        // Step 1: Undefined → TransferDstOptimal on every layer/mip.
        IblBarrierAll(cmd, envCubeImage,         envCubeMipLevels,         6, ImageLayout.Undefined, ImageLayout.TransferDstOptimal);
        IblBarrierAll(cmd, irradianceCubeImage,  1,                        6, ImageLayout.Undefined, ImageLayout.TransferDstOptimal);
        IblBarrierAll(cmd, prefilteredCubeImage, prefilteredCubeMipLevels, 6, ImageLayout.Undefined, ImageLayout.TransferDstOptimal);
        IblBarrierAll(cmd, brdfLutImage,         1,                        1, ImageLayout.Undefined, ImageLayout.TransferDstOptimal);

        // Step 2: Clear each image to (0,0,0,1).
        ClearColorValue black = new();
        black.Float32_0 = 0f; black.Float32_1 = 0f; black.Float32_2 = 0f; black.Float32_3 = 1f;

        ClearImage(cmd, envCubeImage,         envCubeMipLevels,         6, black);
        ClearImage(cmd, irradianceCubeImage,  1,                        6, black);
        ClearImage(cmd, prefilteredCubeImage, prefilteredCubeMipLevels, 6, black);
        ClearImage(cmd, brdfLutImage,         1,                        1, black);

        // Step 3: TransferDstOptimal → ShaderReadOnlyOptimal so the PBR pipeline
        // can bind these descriptors immediately on initialization.
        IblBarrierAll(cmd, envCubeImage,         envCubeMipLevels,         6, ImageLayout.TransferDstOptimal, ImageLayout.ShaderReadOnlyOptimal);
        IblBarrierAll(cmd, irradianceCubeImage,  1,                        6, ImageLayout.TransferDstOptimal, ImageLayout.ShaderReadOnlyOptimal);
        IblBarrierAll(cmd, prefilteredCubeImage, prefilteredCubeMipLevels, 6, ImageLayout.TransferDstOptimal, ImageLayout.ShaderReadOnlyOptimal);
        IblBarrierAll(cmd, brdfLutImage,         1,                        1, ImageLayout.TransferDstOptimal, ImageLayout.ShaderReadOnlyOptimal);

        EndSingleTimeCommands(cmd);
    }

    void ClearImage(CommandBuffer cmd, VkImage image, uint mipLevels, uint layerCount, ClearColorValue color)
    {
        ImageSubresourceRange range = new()
        {
            AspectMask     = ImageAspectFlags.ColorBit,
            BaseMipLevel   = 0,
            LevelCount     = mipLevels,
            BaseArrayLayer = 0,
            LayerCount     = layerCount,
        };
        vk!.CmdClearColorImage(cmd, image, ImageLayout.TransferDstOptimal, &color, 1, &range);
    }

    /// <summary>
    /// Pipeline barrier covering every mip and every layer of an image. Used for
    /// the bulk Undefined→TransferDst and TransferDst→ShaderRead transitions on
    /// the four IBL images. Phase 2 will need a per-mip variant for the
    /// envCube blit chain and the per-mip prefilter writes.
    /// </summary>
    void IblBarrierAll(CommandBuffer cmd, VkImage image, uint mipLevels, uint layerCount,
        ImageLayout oldLayout, ImageLayout newLayout)
    {
        ImageMemoryBarrier barrier = new()
        {
            SType               = StructureType.ImageMemoryBarrier,
            OldLayout           = oldLayout,
            NewLayout           = newLayout,
            SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
            Image               = image,
            SubresourceRange    = new ImageSubresourceRange(ImageAspectFlags.ColorBit, 0, mipLevels, 0, layerCount),
        };
        PipelineStageFlags srcStage, dstStage;
        if (oldLayout == ImageLayout.Undefined && newLayout == ImageLayout.TransferDstOptimal)
        {
            barrier.SrcAccessMask = 0;
            barrier.DstAccessMask = AccessFlags.TransferWriteBit;
            srcStage = PipelineStageFlags.TopOfPipeBit;
            dstStage = PipelineStageFlags.TransferBit;
        }
        else if (oldLayout == ImageLayout.TransferDstOptimal && newLayout == ImageLayout.ShaderReadOnlyOptimal)
        {
            barrier.SrcAccessMask = AccessFlags.TransferWriteBit;
            barrier.DstAccessMask = AccessFlags.ShaderReadBit;
            srcStage = PipelineStageFlags.TransferBit;
            dstStage = PipelineStageFlags.FragmentShaderBit;
        }
        else
        {
            throw new System.Exception($"IblBarrierAll: unsupported transition {oldLayout} → {newLayout}");
        }
        vk!.CmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, null, 0, null, 1, &barrier);
    }

    void CleanupIblResources()
    {
        if (iblLutSampler.Handle != 0)         vk!.DestroySampler(device, iblLutSampler, null);
        if (iblCubeSampler.Handle != 0)        vk!.DestroySampler(device, iblCubeSampler, null);

        if (brdfLutView.Handle != 0)           vk!.DestroyImageView(device, brdfLutView, null);
        if (brdfLutImage.Handle != 0)          vk!.DestroyImage(device, brdfLutImage, null);
        if (brdfLutMemory.Handle != 0)         vk!.FreeMemory(device, brdfLutMemory, null);

        if (prefilteredCubeView.Handle != 0)   vk!.DestroyImageView(device, prefilteredCubeView, null);
        if (prefilteredCubeImage.Handle != 0)  vk!.DestroyImage(device, prefilteredCubeImage, null);
        if (prefilteredCubeMemory.Handle != 0) vk!.FreeMemory(device, prefilteredCubeMemory, null);

        if (irradianceCubeView.Handle != 0)    vk!.DestroyImageView(device, irradianceCubeView, null);
        if (irradianceCubeImage.Handle != 0)   vk!.DestroyImage(device, irradianceCubeImage, null);
        if (irradianceCubeMemory.Handle != 0)  vk!.FreeMemory(device, irradianceCubeMemory, null);

        if (envCubeView.Handle != 0)           vk!.DestroyImageView(device, envCubeView, null);
        if (envCubeImage.Handle != 0)          vk!.DestroyImage(device, envCubeImage, null);
        if (envCubeMemory.Handle != 0)         vk!.FreeMemory(device, envCubeMemory, null);
    }
}
