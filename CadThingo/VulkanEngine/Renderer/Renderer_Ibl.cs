using System;
using System.Runtime.InteropServices;
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
    internal SubAlloc       envCubeAlloc;
    internal ImageView      envCubeView;        // VK_IMAGE_VIEW_TYPE_CUBE, all mips
    internal uint           envCubeMipLevels;

    internal VkImage        irradianceCubeImage;
    internal SubAlloc       irradianceCubeAlloc;
    internal ImageView      irradianceCubeView;

    internal VkImage        prefilteredCubeImage;
    internal SubAlloc       prefilteredCubeAlloc;
    internal ImageView      prefilteredCubeView;
    internal uint           prefilteredCubeMipLevels;

    internal VkImage        brdfLutImage;
    internal SubAlloc       brdfLutAlloc;
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
            out envCubeImage, out envCubeAlloc, out envCubeView);

        // ── irradiance cube — sampled + storage-written
        CreateCubemapImage(
            IrradianceCubeFaceSize, 1, Format.R16G16B16A16Sfloat,
            ImageUsageFlags.SampledBit | ImageUsageFlags.StorageBit | ImageUsageFlags.TransferDstBit,
            out irradianceCubeImage, out irradianceCubeAlloc, out irradianceCubeView);

        // ── prefiltered cube — sampled + storage-written per mip
        CreateCubemapImage(
            PrefilteredCubeFaceSize, prefilteredCubeMipLevels, Format.R16G16B16A16Sfloat,
            ImageUsageFlags.SampledBit | ImageUsageFlags.StorageBit | ImageUsageFlags.TransferDstBit,
            out prefilteredCubeImage, out prefilteredCubeAlloc, out prefilteredCubeView);

        // ── BRDF LUT — 2D, sampled + storage-written, no mips
        CreateLutImage(
            BrdfLutSize, Format.R16G16Sfloat,
            ImageUsageFlags.SampledBit | ImageUsageFlags.StorageBit | ImageUsageFlags.TransferDstBit,
            out brdfLutImage, out brdfLutAlloc, out brdfLutView);

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
        out VkImage image, out SubAlloc alloc, out ImageView cubeView)
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

        alloc = memAllocator.AllocateForImage(image, MemoryPropertyFlags.DeviceLocalBit);

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
        out VkImage image, out SubAlloc alloc, out ImageView view)
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

        alloc = memAllocator.AllocateForImage(image, MemoryPropertyFlags.DeviceLocalBit);

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

    // ── Bake pipelines ─────────────────────────────────────────────────────
    // Created once at engine init; live for the lifetime of the renderer. The
    // BRDF LUT bake runs immediately after construction (the result is view-
    // independent — same LUT for every environment). The other three pipelines
    // sit idle until LoadEnvironmentHdr is invoked.

    internal IblBakePipeline equirectToCubePipeline    = null!;
    internal IblBakePipeline irradianceConvolvePipeline = null!;
    internal IblBakePipeline prefilterEnvPipeline      = null!;
    internal IblBakePipeline brdfLutGenPipeline        = null!;

    // Push-constant structs. Slang's `cbuffer` push block reads these exactly —
    // member order/types must match the shader's PC struct.
    [StructLayout(LayoutKind.Sequential)]
    struct PcFaceSize { public uint faceSize; }

    [StructLayout(LayoutKind.Sequential)]
    struct PcPrefilter
    {
        public float roughness;
        public uint  mipFaceSize;
        public uint  envCubeSize;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct PcLutSize { public uint lutSize; }

    internal void CreateIblBakePipelines()
    {
        const string shaderDir = @"C:\Users\jamie\RiderProjects\CadThingo\CadThingo\Assets\Shaders\";

        equirectToCubePipeline = new IblBakePipeline(this,
            shaderDir + "EquirectToCube.spv",     hasInputSampler: true,  pushSize: (uint)sizeof(PcFaceSize));
        irradianceConvolvePipeline = new IblBakePipeline(this,
            shaderDir + "IrradianceConvolve.spv", hasInputSampler: true,  pushSize: (uint)sizeof(PcFaceSize));
        prefilterEnvPipeline = new IblBakePipeline(this,
            shaderDir + "PrefilterEnv.spv",       hasInputSampler: true,  pushSize: (uint)sizeof(PcPrefilter));
        brdfLutGenPipeline = new IblBakePipeline(this,
            shaderDir + "BrdfLutGen.spv",         hasInputSampler: false, pushSize: (uint)sizeof(PcLutSize));

        equirectToCubePipeline    .Initialize();
        irradianceConvolvePipeline.Initialize();
        prefilterEnvPipeline      .Initialize();
        brdfLutGenPipeline        .Initialize();
    }

    void DisposeIblBakePipelines()
    {
        brdfLutGenPipeline?.Dispose();
        prefilterEnvPipeline?.Dispose();
        irradianceConvolvePipeline?.Dispose();
        equirectToCubePipeline?.Dispose();
    }

    // ── Storage-image views ────────────────────────────────────────────────
    // The cube ImageViews from Phase 1 are sample-only (ViewType.Cube). Storage
    // writes need a 2D-array view at a specific mip level. Created on demand
    // during a bake, destroyed immediately after.

    ImageView CreateCubeStorageView(VkImage image, Format format, uint mipLevel)
    {
        ImageViewCreateInfo info = new()
        {
            SType    = StructureType.ImageViewCreateInfo,
            Image    = image,
            ViewType = ImageViewType.Type2DArray,
            Format   = format,
            SubresourceRange = new ImageSubresourceRange
            {
                AspectMask     = ImageAspectFlags.ColorBit,
                BaseMipLevel   = mipLevel,
                LevelCount     = 1,
                BaseArrayLayer = 0,
                LayerCount     = 6,
            },
        };
        if (vk!.CreateImageView(device, &info, null, out var view) != Result.Success)
            throw new Exception("Failed to create IBL cube storage view");
        return view;
    }

    ImageView CreateLut2DStorageView(VkImage image, Format format)
    {
        ImageViewCreateInfo info = new()
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
        if (vk!.CreateImageView(device, &info, null, out var view) != Result.Success)
            throw new Exception("Failed to create IBL LUT storage view");
        return view;
    }

    // ── Layout transitions ─────────────────────────────────────────────────
    // Phase 2 dispatches mix sampling (ShaderRead) and storage writes (General)
    // on the same image, sometimes per-mip. This helper takes an explicit
    // subresource range and chooses access/stage masks from old→new layout.

    void IblTransition(CommandBuffer cmd, VkImage image,
        ImageLayout oldLayout, ImageLayout newLayout, in ImageSubresourceRange range)
    {
        var (srcAccess, srcStage) = LayoutToSrc(oldLayout);
        var (dstAccess, dstStage) = LayoutToDst(newLayout);

        ImageMemoryBarrier barrier = new()
        {
            SType               = StructureType.ImageMemoryBarrier,
            OldLayout           = oldLayout,
            NewLayout           = newLayout,
            SrcAccessMask       = srcAccess,
            DstAccessMask       = dstAccess,
            SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
            Image               = image,
            SubresourceRange    = range,
        };
        vk!.CmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, null, 0, null, 1, &barrier);
    }

    static (AccessFlags access, PipelineStageFlags stage) LayoutToSrc(ImageLayout l) => l switch
    {
        ImageLayout.Undefined             => (0,                                 PipelineStageFlags.TopOfPipeBit),
        ImageLayout.General               => (AccessFlags.ShaderWriteBit,        PipelineStageFlags.ComputeShaderBit),
        ImageLayout.TransferDstOptimal    => (AccessFlags.TransferWriteBit,      PipelineStageFlags.TransferBit),
        ImageLayout.TransferSrcOptimal    => (AccessFlags.TransferReadBit,       PipelineStageFlags.TransferBit),
        ImageLayout.ShaderReadOnlyOptimal => (AccessFlags.ShaderReadBit,         PipelineStageFlags.FragmentShaderBit | PipelineStageFlags.ComputeShaderBit),
        _ => throw new Exception($"IblTransition: unsupported src layout {l}")
    };

    static (AccessFlags access, PipelineStageFlags stage) LayoutToDst(ImageLayout l) => l switch
    {
        ImageLayout.General               => (AccessFlags.ShaderWriteBit | AccessFlags.ShaderReadBit, PipelineStageFlags.ComputeShaderBit),
        ImageLayout.TransferDstOptimal    => (AccessFlags.TransferWriteBit,                          PipelineStageFlags.TransferBit),
        ImageLayout.TransferSrcOptimal    => (AccessFlags.TransferReadBit,                           PipelineStageFlags.TransferBit),
        ImageLayout.ShaderReadOnlyOptimal => (AccessFlags.ShaderReadBit,                             PipelineStageFlags.FragmentShaderBit | PipelineStageFlags.ComputeShaderBit),
        _ => throw new Exception($"IblTransition: unsupported dst layout {l}")
    };

    static ImageSubresourceRange CubeRange(uint baseMip, uint levelCount) => new()
    {
        AspectMask     = ImageAspectFlags.ColorBit,
        BaseMipLevel   = baseMip,
        LevelCount     = levelCount,
        BaseArrayLayer = 0,
        LayerCount     = 6,
    };

    static ImageSubresourceRange Lut2DRange() => new()
    {
        AspectMask     = ImageAspectFlags.ColorBit,
        BaseMipLevel   = 0,
        LevelCount     = 1,
        BaseArrayLayer = 0,
        LayerCount     = 1,
    };

    // ── BRDF LUT bake ──────────────────────────────────────────────────────
    // Runs once at engine init. The split-sum integral is view-independent so
    // we never need to re-bake it. ~262k integration runs, milliseconds on any
    // modern GPU.

    public void BakeBrdfLut()
    {
        var cmd = BeginSingleTimeCommands();
        IblTransition(cmd, brdfLutImage,
            ImageLayout.ShaderReadOnlyOptimal, ImageLayout.General, Lut2DRange());

        var view = CreateLut2DStorageView(brdfLutImage, Format.R16G16Sfloat);
        var set  = brdfLutGenPipeline.AllocateDescriptorSet();
        brdfLutGenPipeline.WriteDescriptors(set, view, default, default);

        vk!.CmdBindPipeline(cmd, PipelineBindPoint.Compute, brdfLutGenPipeline.Handle);
        vk!.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute, brdfLutGenPipeline.Layout,
            0, 1, &set, 0, null);

        var pc = new PcLutSize { lutSize = BrdfLutSize };
        vk!.CmdPushConstants(cmd, brdfLutGenPipeline.Layout,
            ShaderStageFlags.ComputeBit, 0, (uint)sizeof(PcLutSize), &pc);

        uint groups = (BrdfLutSize + 15) / 16;
        vk!.CmdDispatch(cmd, groups, groups, 1);

        IblTransition(cmd, brdfLutImage,
            ImageLayout.General, ImageLayout.ShaderReadOnlyOptimal, Lut2DRange());

        EndSingleTimeCommands(cmd);

        vk!.DestroyImageView(device, view, null);
        brdfLutGenPipeline.FreeDescriptorSet(set);
    }

    // ── Full HDR → IBL chain ───────────────────────────────────────────────
    // Drives equirect upload → cube unwrap → mip generation → irradiance →
    // prefiltered specular. Safe to call multiple times (rebake on environment
    // swap); device is waited on so prior IBL reads can't observe a torn state.

    /// <summary>Path of the .hdr most recently loaded via LoadEnvironmentHdr —
    /// empty when no environment has been baked yet. Surfaced in the Renderer
    /// Settings panel as the "currently loaded" line.</summary>
    public string CurrentEnvironmentPath { get; private set; } = string.Empty;

    /// <summary>
    /// Look for any .hdr file in Assets/Textures and bake from the first one
    /// found. Called once at engine init so the scene has a sensible default
    /// environment if the user has dropped an HDRI in there. Silent no-op if
    /// the directory or file isn't present.
    /// </summary>
    internal void TryAutoLoadEnvironment()
    {
        string[] candidates =
        {
            System.IO.Path.Combine(System.AppContext.BaseDirectory, "Assets", "Textures"),
            @"C:\Users\jamie\RiderProjects\CadThingo\CadThingo\Assets\Textures",
        };

        foreach (var dir in candidates)
        {
            if (!System.IO.Directory.Exists(dir)) continue;
            var files = System.IO.Directory.GetFiles(dir, "*.hdr", System.IO.SearchOption.TopDirectoryOnly);
            if (files.Length == 0) continue;
            try
            {
                LoadEnvironmentHdr(files[0]);
                System.Console.WriteLine($"Auto-loaded IBL environment: {System.IO.Path.GetFileName(files[0])}");
                return;
            }
            catch (System.Exception e)
            {
                System.Console.WriteLine($"Auto-load failed for {files[0]}: {e.Message}");
            }
        }
    }

    public void LoadEnvironmentHdr(string path)
    {
        var img = HdrLoader.Load(path);
        vk!.DeviceWaitIdle(device);
        CurrentEnvironmentPath = path;

        // ── 1. Upload equirect as a 2D RGBA16F sampled image ─────────────
        UploadEquirectHalf(img, out var eqImage, out var eqAlloc, out var eqView, out var eqSampler);

        // ── 2. EquirectToCube: write envCube mip 0 layers 0..5 ───────────
        var cubeMip0View = CreateCubeStorageView(envCubeImage, Format.R16G16B16A16Sfloat, mipLevel: 0);
        var eqCubeSet    = equirectToCubePipeline.AllocateDescriptorSet();
        equirectToCubePipeline.WriteDescriptors(eqCubeSet, cubeMip0View, eqView, eqSampler);

        var cmd = BeginSingleTimeCommands();

        // envCube: ShaderRead → General (all mips/layers). Mip 0 receives storage
        // writes; mips 1..N get TransferDst during the blit chain so it's simpler
        // to push the whole image through one transition now.
        IblTransition(cmd, envCubeImage,
            ImageLayout.ShaderReadOnlyOptimal, ImageLayout.General, CubeRange(0, envCubeMipLevels));

        vk!.CmdBindPipeline(cmd, PipelineBindPoint.Compute, equirectToCubePipeline.Handle);
        vk!.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute, equirectToCubePipeline.Layout,
            0, 1, &eqCubeSet, 0, null);
        var pcFace = new PcFaceSize { faceSize = EnvCubeFaceSize };
        vk!.CmdPushConstants(cmd, equirectToCubePipeline.Layout,
            ShaderStageFlags.ComputeBit, 0, (uint)sizeof(PcFaceSize), &pcFace);
        uint cubeGroups = (EnvCubeFaceSize + 15) / 16;
        vk!.CmdDispatch(cmd, cubeGroups, cubeGroups, 6);

        // ── 3. envCube mip chain via vkCmdBlitImage ──────────────────────
        // mip 0 was just written via storage → needs General→TransferSrc.
        // mips 1..N were freshly transitioned to General (TopOfPipe) → need
        // General→TransferDst before the first blit lands on them.
        IblTransition(cmd, envCubeImage, ImageLayout.General, ImageLayout.TransferSrcOptimal,
            CubeRange(0, 1));
        if (envCubeMipLevels > 1)
            IblTransition(cmd, envCubeImage, ImageLayout.General, ImageLayout.TransferDstOptimal,
                CubeRange(1, envCubeMipLevels - 1));

        // BlitCubeMipChain leaves every mip in ShaderReadOnlyOptimal — irradiance
        // + prefilter dispatches below sample envCube directly.
        BlitCubeMipChain(cmd, envCubeImage, EnvCubeFaceSize, envCubeMipLevels);

        // ── 4. Irradiance convolution ────────────────────────────────────
        var irrView = CreateCubeStorageView(irradianceCubeImage, Format.R16G16B16A16Sfloat, mipLevel: 0);
        var irrSet  = irradianceConvolvePipeline.AllocateDescriptorSet();
        irradianceConvolvePipeline.WriteDescriptors(irrSet, irrView, envCubeView, iblCubeSampler);

        IblTransition(cmd, irradianceCubeImage, ImageLayout.ShaderReadOnlyOptimal,
            ImageLayout.General, CubeRange(0, 1));

        vk!.CmdBindPipeline(cmd, PipelineBindPoint.Compute, irradianceConvolvePipeline.Handle);
        vk!.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute, irradianceConvolvePipeline.Layout,
            0, 1, &irrSet, 0, null);
        var pcIrr = new PcFaceSize { faceSize = IrradianceCubeFaceSize };
        vk!.CmdPushConstants(cmd, irradianceConvolvePipeline.Layout,
            ShaderStageFlags.ComputeBit, 0, (uint)sizeof(PcFaceSize), &pcIrr);
        uint irrGroups = (IrradianceCubeFaceSize + 15) / 16;
        vk!.CmdDispatch(cmd, irrGroups, irrGroups, 6);

        IblTransition(cmd, irradianceCubeImage, ImageLayout.General,
            ImageLayout.ShaderReadOnlyOptimal, CubeRange(0, 1));

        // ── 5. Prefilter — per-mip dispatches with per-mip storage views ──
        var prefilterMipViews = new ImageView[prefilteredCubeMipLevels];
        var prefilterSets     = new DescriptorSet[prefilteredCubeMipLevels];

        IblTransition(cmd, prefilteredCubeImage, ImageLayout.ShaderReadOnlyOptimal,
            ImageLayout.General, CubeRange(0, prefilteredCubeMipLevels));

        vk!.CmdBindPipeline(cmd, PipelineBindPoint.Compute, prefilterEnvPipeline.Handle);
        for (uint m = 0; m < prefilteredCubeMipLevels; m++)
        {
            uint mipSize = System.Math.Max(1u, PrefilteredCubeFaceSize >> (int)m);
            float roughness = prefilteredCubeMipLevels == 1
                ? 0f
                : (float)m / (float)(prefilteredCubeMipLevels - 1);

            prefilterMipViews[m] = CreateCubeStorageView(prefilteredCubeImage, Format.R16G16B16A16Sfloat, mipLevel: m);
            prefilterSets[m]     = prefilterEnvPipeline.AllocateDescriptorSet();
            prefilterEnvPipeline.WriteDescriptors(prefilterSets[m], prefilterMipViews[m], envCubeView, iblCubeSampler);

            var set = prefilterSets[m];
            vk!.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute, prefilterEnvPipeline.Layout,
                0, 1, &set, 0, null);
            var pcPre = new PcPrefilter
            {
                roughness   = roughness,
                mipFaceSize = mipSize,
                envCubeSize = EnvCubeFaceSize,
            };
            vk!.CmdPushConstants(cmd, prefilterEnvPipeline.Layout,
                ShaderStageFlags.ComputeBit, 0, (uint)sizeof(PcPrefilter), &pcPre);
            uint mipGroups = (mipSize + 15) / 16;
            vk!.CmdDispatch(cmd, mipGroups, mipGroups, 6);
        }

        IblTransition(cmd, prefilteredCubeImage, ImageLayout.General,
            ImageLayout.ShaderReadOnlyOptimal, CubeRange(0, prefilteredCubeMipLevels));

        EndSingleTimeCommands(cmd);

        // ── 6. Cleanup transient bake resources ──────────────────────────
        vk!.DestroyImageView(device, cubeMip0View, null);
        vk!.DestroyImageView(device, irrView, null);
        for (int m = 0; m < prefilterMipViews.Length; m++)
            vk!.DestroyImageView(device, prefilterMipViews[m], null);

        equirectToCubePipeline.FreeDescriptorSet(eqCubeSet);
        irradianceConvolvePipeline.FreeDescriptorSet(irrSet);
        for (int m = 0; m < prefilterSets.Length; m++)
            prefilterEnvPipeline.FreeDescriptorSet(prefilterSets[m]);

        vk!.DestroySampler(device, eqSampler, null);
        vk!.DestroyImageView(device, eqView, null);
        vk!.DestroyImage(device, eqImage, null);
        memAllocator.Free(eqAlloc);
    }

    /// <summary>
    /// Uploads an HDR equirect (float32 RGBA) into a device-local RGBA16F 2D
    /// image with a linear-clamp sampler. The half conversion saves half the
    /// VRAM with no visible loss for IBL bake — we re-integrate over the cube
    /// anyway, so banding from 11-bit mantissa truncation can't survive.
    /// </summary>
    void UploadEquirectHalf(HdrLoader.HdrImage img,
        out VkImage image, out SubAlloc alloc, out ImageView view, out Sampler sampler)
    {
        int texelCount = img.Width * img.Height;
        ulong byteSize = (ulong)texelCount * 4UL * 2UL; // 4 channels × 2 bytes (Half)

        // CPU-side float32 → float16 conversion. System.Half is a value type; the
        // unsafe block writes straight into the mapped staging buffer.
        CreateBuffer(byteSize, BufferUsageFlags.TransferSrcBit,
            MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit,
            out var staging, out var stagingAlloc);

        void* mapped = memAllocator.GetMapped(stagingAlloc);
        var halfSpan = new Span<Half>(mapped, texelCount * 4);
        for (int i = 0; i < texelCount * 4; i++) halfSpan[i] = (Half)img.Pixels[i];

        ImageCreateInfo info = new()
        {
            SType         = StructureType.ImageCreateInfo,
            ImageType     = ImageType.Type2D,
            Format        = Format.R16G16B16A16Sfloat,
            Extent        = new Extent3D((uint)img.Width, (uint)img.Height, 1),
            MipLevels     = 1,
            ArrayLayers   = 1,
            Samples       = SampleCountFlags.Count1Bit,
            Tiling        = ImageTiling.Optimal,
            Usage         = ImageUsageFlags.TransferDstBit | ImageUsageFlags.SampledBit,
            SharingMode   = SharingMode.Exclusive,
            InitialLayout = ImageLayout.Undefined,
        };
        if (vk!.CreateImage(device, &info, null, out image) != Result.Success)
            throw new Exception("Failed to create equirect upload image");

        alloc = memAllocator.AllocateForImage(image, MemoryPropertyFlags.DeviceLocalBit);

        var cmd = BeginSingleTimeCommands();
        IblTransition(cmd, image, ImageLayout.Undefined, ImageLayout.TransferDstOptimal,
            Lut2DRange());
        BufferImageCopy region = new()
        {
            BufferOffset      = 0,
            BufferRowLength   = 0,
            BufferImageHeight = 0,
            ImageSubresource  = new ImageSubresourceLayers
            {
                AspectMask     = ImageAspectFlags.ColorBit,
                MipLevel       = 0,
                BaseArrayLayer = 0,
                LayerCount     = 1,
            },
            ImageOffset = new Offset3D(0, 0, 0),
            ImageExtent = new Extent3D((uint)img.Width, (uint)img.Height, 1),
        };
        vk!.CmdCopyBufferToImage(cmd, staging, image, ImageLayout.TransferDstOptimal, 1, &region);
        IblTransition(cmd, image, ImageLayout.TransferDstOptimal,
            ImageLayout.ShaderReadOnlyOptimal, Lut2DRange());
        EndSingleTimeCommands(cmd);

        DestroyBuffer(staging, stagingAlloc);

        ImageViewCreateInfo viewInfo = new()
        {
            SType    = StructureType.ImageViewCreateInfo,
            Image    = image,
            ViewType = ImageViewType.Type2D,
            Format   = Format.R16G16B16A16Sfloat,
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
            throw new Exception("Failed to create equirect upload view");

        SamplerCreateInfo samplerInfo = new()
        {
            SType         = StructureType.SamplerCreateInfo,
            MagFilter     = Filter.Linear,
            MinFilter     = Filter.Linear,
            AddressModeU  = SamplerAddressMode.Repeat,        // longitude wraps
            AddressModeV  = SamplerAddressMode.ClampToEdge,   // latitude clamps
            AddressModeW  = SamplerAddressMode.ClampToEdge,
            MipmapMode    = SamplerMipmapMode.Nearest,
            MinLod        = 0f,
            MaxLod        = 0f,
            CompareEnable = false,
        };
        if (vk!.CreateSampler(device, &samplerInfo, null, out sampler) != Result.Success)
            throw new Exception("Failed to create equirect sampler");
    }

    /// <summary>
    /// Generates the envCube mip chain by chained vkCmdBlitImage calls. After
    /// each blit, the source mip is finalized to ShaderReadOnlyOptimal (no more
    /// reads from it) and the just-written destination is flipped to
    /// TransferSrcOptimal to feed the next iteration. Last mip is finalized
    /// after the loop. Entry contract: mip 0 in TransferSrcOptimal, mips 1..N
    /// in TransferDstOptimal. Exit: every mip in ShaderReadOnlyOptimal.
    /// </summary>
    void BlitCubeMipChain(CommandBuffer cmd, VkImage image, uint baseSize, uint mipLevels)
    {
        uint srcSize = baseSize;
        for (uint i = 1; i < mipLevels; i++)
        {
            uint dstSize = System.Math.Max(1u, srcSize / 2);

            ImageBlit blit = new()
            {
                SrcOffsets = { Element0 = new Offset3D(0, 0, 0), Element1 = new Offset3D((int)srcSize, (int)srcSize, 1) },
                SrcSubresource = new ImageSubresourceLayers
                {
                    AspectMask = ImageAspectFlags.ColorBit, MipLevel = i - 1,
                    BaseArrayLayer = 0, LayerCount = 6,
                },
                DstOffsets = { Element0 = new Offset3D(0, 0, 0), Element1 = new Offset3D((int)dstSize, (int)dstSize, 1) },
                DstSubresource = new ImageSubresourceLayers
                {
                    AspectMask = ImageAspectFlags.ColorBit, MipLevel = i,
                    BaseArrayLayer = 0, LayerCount = 6,
                },
            };

            vk!.CmdBlitImage(cmd,
                image, ImageLayout.TransferSrcOptimal,
                image, ImageLayout.TransferDstOptimal,
                1, &blit, Filter.Linear);

            // Source mip is done — never read from again — finalize it.
            IblTransition(cmd, image, ImageLayout.TransferSrcOptimal, ImageLayout.ShaderReadOnlyOptimal,
                CubeRange(i - 1, 1));
            // Destination mip becomes source for the next iteration.
            IblTransition(cmd, image, ImageLayout.TransferDstOptimal, ImageLayout.TransferSrcOptimal,
                CubeRange(i, 1));

            srcSize = dstSize;
        }
        // Last destination mip is still TransferSrc from the loop's tail
        // transition — finalize it. Handles the edge case of a single-mip image
        // by transitioning mip 0 from its TransferSrc entry state.
        IblTransition(cmd, image, ImageLayout.TransferSrcOptimal, ImageLayout.ShaderReadOnlyOptimal,
            CubeRange(mipLevels - 1, 1));
    }

    void CleanupIblResources()
    {
        DisposeIblBakePipelines();

        if (iblLutSampler.Handle != 0)         vk!.DestroySampler(device, iblLutSampler, null);
        if (iblCubeSampler.Handle != 0)        vk!.DestroySampler(device, iblCubeSampler, null);

        if (brdfLutView.Handle != 0)           vk!.DestroyImageView(device, brdfLutView, null);
        if (brdfLutImage.Handle != 0)          vk!.DestroyImage(device, brdfLutImage, null);
        memAllocator.Free(brdfLutAlloc);

        if (prefilteredCubeView.Handle != 0)   vk!.DestroyImageView(device, prefilteredCubeView, null);
        if (prefilteredCubeImage.Handle != 0)  vk!.DestroyImage(device, prefilteredCubeImage, null);
        memAllocator.Free(prefilteredCubeAlloc);

        if (irradianceCubeView.Handle != 0)    vk!.DestroyImageView(device, irradianceCubeView, null);
        if (irradianceCubeImage.Handle != 0)   vk!.DestroyImage(device, irradianceCubeImage, null);
        memAllocator.Free(irradianceCubeAlloc);

        if (envCubeView.Handle != 0)           vk!.DestroyImageView(device, envCubeView, null);
        if (envCubeImage.Handle != 0)          vk!.DestroyImage(device, envCubeImage, null);
        memAllocator.Free(envCubeAlloc);
    }
}
