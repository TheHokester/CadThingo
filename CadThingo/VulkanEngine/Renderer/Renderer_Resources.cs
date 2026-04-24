using System.Numerics;
using ImGuiNET;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer;

public unsafe partial class Renderer
{
    private void CreateDepthResources()
    {
        var width = swapChainExtent.Width;
        var height = swapChainExtent.Height;

        depthImageResource = new ImageResource(vk, device, "Depth", Format.D32Sfloat, new Extent2D(width, height),
            ImageUsageFlags.DepthStencilAttachmentBit | ImageUsageFlags.InputAttachmentBit,
            ImageLayout.Undefined, ImageLayout.DepthStencilAttachmentOptimal);
    }

    private void CreateGBufferResources()
    {
        var width = swapChainExtent.Width;
        var height = swapChainExtent.Height;
        // SampledBit required: the lighting pass samples these via CombinedImageSampler descriptors.
        const ImageUsageFlags gBufferUsage =
            ImageUsageFlags.ColorAttachmentBit | ImageUsageFlags.InputAttachmentBit | ImageUsageFlags.SampledBit;
        gBufferPosition = new ImageResource(vk, device, "GBuffer_Position", Format.R32G32B32A32Sfloat,
            new Extent2D(width, height), gBufferUsage,
            ImageLayout.Undefined, ImageLayout.ShaderReadOnlyOptimal);
        gBufferNormal = new ImageResource(vk, device, "GBuffer_Normal", Format.R32G32B32A32Sfloat,
            new Extent2D(width, height), gBufferUsage,
            ImageLayout.Undefined, ImageLayout.ShaderReadOnlyOptimal);
        gBufferAlbedo = new ImageResource(vk, device, "GBuffer_Albedo", Format.R8G8B8A8Unorm,
            new Extent2D(width, height), gBufferUsage,
            ImageLayout.Undefined, ImageLayout.ShaderReadOnlyOptimal);
        gBufferMaterial = new ImageResource(vk, device, "GBuffer_Material", Format.R8G8B8A8Unorm,
            new Extent2D(width, height), gBufferUsage,
            ImageLayout.Undefined, ImageLayout.ShaderReadOnlyOptimal);
    }

    public CommandBuffer BeginSingleTimeCommands()
    {
        CommandBufferAllocateInfo allocInfo = new()
        {
            SType = StructureType.CommandBufferAllocateInfo,
            CommandPool = commandPool,
            Level = CommandBufferLevel.Primary,
            CommandBufferCount = 1,
        };
        vk!.AllocateCommandBuffers(device, &allocInfo, out CommandBuffer cmd);
        CommandBufferBeginInfo beginInfo = new()
        {
            SType = StructureType.CommandBufferBeginInfo,
            Flags = CommandBufferUsageFlags.OneTimeSubmitBit,
        };
        vk!.BeginCommandBuffer(cmd, &beginInfo);
        return cmd;
    }

    public void EndSingleTimeCommands(CommandBuffer cmd)
    {
        vk!.EndCommandBuffer(cmd);
        SubmitInfo submit = new()
        {
            SType = StructureType.SubmitInfo,
            CommandBufferCount = 1,
            PCommandBuffers = &cmd,
        };
        vk!.QueueSubmit(graphicsQueue, 1, &submit, default);
        vk!.QueueWaitIdle(graphicsQueue);
        vk!.FreeCommandBuffers(device, commandPool, 1, &cmd);
    }

    public void CreateBuffer(ulong size, BufferUsageFlags usage, MemoryPropertyFlags memProps,
        out Buffer buffer, out DeviceMemory memory)
    {
        BufferCreateInfo bufferInfo = new()
        {
            SType = StructureType.BufferCreateInfo,
            Size = size,
            Usage = usage,
            SharingMode = SharingMode.Exclusive,
        };
        if (vk!.CreateBuffer(device, &bufferInfo, null, out buffer) != Result.Success)
            throw new Exception("Failed to create buffer");

        vk!.GetBufferMemoryRequirements(device, buffer, out var memReqs);
        MemoryAllocateInfo allocInfo = new()
        {
            SType = StructureType.MemoryAllocateInfo,
            AllocationSize = memReqs.Size,
            MemoryTypeIndex = FindMemoryType(vk, physicalDevice, memReqs.MemoryTypeBits, memProps),
        };
        if (vk!.AllocateMemory(device, &allocInfo, null, out memory) != Result.Success)
            throw new Exception("Failed to allocate buffer memory");

        vk!.BindBufferMemory(device, buffer, memory, 0);
    }

    public void CopyBuffer(Buffer src, Buffer dst, ulong size)
    {
        var cmd = BeginSingleTimeCommands();
        BufferCopy region = new()
        {
            SrcOffset = 0,
            DstOffset = 0,
            Size = size,
        };
        vk!.CmdCopyBuffer(cmd, src, dst, 1, &region);
        EndSingleTimeCommands(cmd);
    }

    public void UploadBufferData(Buffer dst, ulong dstOffset, void* srcData, ulong size)
    {
        CreateBuffer(size, BufferUsageFlags.TransferSrcBit,
            MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit,
            out var staging, out var stagingMem);

        void* mapped = null;
        vk!.MapMemory(device, stagingMem, 0, size, 0, ref mapped);
        System.Buffer.MemoryCopy(srcData, mapped, (long)size, (long)size);
        vk!.UnmapMemory(device, stagingMem);

        var cmd = BeginSingleTimeCommands();
        BufferCopy region = new() { SrcOffset = 0, DstOffset = dstOffset, Size = size };
        vk!.CmdCopyBuffer(cmd, staging, dst, 1, &region);
        EndSingleTimeCommands(cmd);

        vk!.DestroyBuffer(device, staging, null);
        vk!.FreeMemory(device, stagingMem, null);
    }

    public void DestroyBuffer(Buffer buffer, DeviceMemory memory)
    {
        if (buffer.Handle != 0) vk!.DestroyBuffer(device, buffer, null);
        if (memory.Handle != 0) vk!.FreeMemory(device, memory, null);
    }

    private void CreateDummyWhiteTexture()
    {
        // 1x1 RGBA white for unbound PBR sampler slots. Shader ignores samples when
        // PbrPushConstants.*TextureSet = -1, so this is just a valid fallback.
        const int pixelBytes = 4;

        // Staging buffer (inline — CreateBuffer helper comes in 4a)
        BufferCreateInfo stagingInfo = new()
        {
            SType = StructureType.BufferCreateInfo,
            Size = pixelBytes,
            Usage = BufferUsageFlags.TransferSrcBit,
            SharingMode = SharingMode.Exclusive,
        };
        vk!.CreateBuffer(device, &stagingInfo, null, out var stagingBuffer);
        vk!.GetBufferMemoryRequirements(device, stagingBuffer, out var stagingReqs);
        MemoryAllocateInfo stagingAlloc = new()
        {
            SType = StructureType.MemoryAllocateInfo,
            AllocationSize = stagingReqs.Size,
            MemoryTypeIndex = FindMemoryType(vk, physicalDevice, stagingReqs.MemoryTypeBits,
                MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit),
        };
        vk!.AllocateMemory(device, &stagingAlloc, null, out var stagingMemory);
        vk!.BindBufferMemory(device, stagingBuffer, stagingMemory, 0);
        void* mapped = null;
        vk!.MapMemory(device, stagingMemory, 0, pixelBytes, 0, ref mapped);
        ((byte*)mapped)[0] = 0xFF;
        ((byte*)mapped)[1] = 0xFF;
        ((byte*)mapped)[2] = 0xFF;
        ((byte*)mapped)[3] = 0xFF;
        vk!.UnmapMemory(device, stagingMemory);

        // Device-local 1x1 image
        ImageCreateInfo imageInfo = new()
        {
            SType = StructureType.ImageCreateInfo,
            ImageType = ImageType.Type2D,
            Format = Format.R8G8B8A8Unorm,
            Extent = new Extent3D(1, 1, 1),
            MipLevels = 1,
            ArrayLayers = 1,
            Samples = SampleCountFlags.Count1Bit,
            Tiling = ImageTiling.Optimal,
            Usage = ImageUsageFlags.TransferDstBit | ImageUsageFlags.SampledBit,
            SharingMode = SharingMode.Exclusive,
            InitialLayout = ImageLayout.Undefined,
        };
        if (vk!.CreateImage(device, &imageInfo, null, out dummyWhiteImage) != Result.Success)
            throw new Exception("Failed to create dummy white image");

        vk!.GetImageMemoryRequirements(device, dummyWhiteImage, out var imgReqs);
        MemoryAllocateInfo imgAlloc = new()
        {
            SType = StructureType.MemoryAllocateInfo,
            AllocationSize = imgReqs.Size,
            MemoryTypeIndex = FindMemoryType(vk, physicalDevice, imgReqs.MemoryTypeBits,
                MemoryPropertyFlags.DeviceLocalBit),
        };
        vk!.AllocateMemory(device, &imgAlloc, null, out dummyWhiteImageMemory);
        vk!.BindImageMemory(device, dummyWhiteImage, dummyWhiteImageMemory, 0);

        // Transition → copy → transition
        var cmd = BeginSingleTimeCommands();
        TransitionImageLayout(cmd, dummyWhiteImage, Format.R8G8B8A8Unorm,
            ImageLayout.Undefined, ImageLayout.TransferDstOptimal);

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
            ImageExtent = new Extent3D(1, 1, 1),
        };
        vk!.CmdCopyBufferToImage(cmd, stagingBuffer, dummyWhiteImage,
            ImageLayout.TransferDstOptimal, 1, &region);

        TransitionImageLayout(cmd, dummyWhiteImage, Format.R8G8B8A8Unorm,
            ImageLayout.TransferDstOptimal, ImageLayout.ShaderReadOnlyOptimal);
        EndSingleTimeCommands(cmd);

        vk!.DestroyBuffer(device, stagingBuffer, null);
        vk!.FreeMemory(device, stagingMemory, null);

        // Image view
        ImageViewCreateInfo viewInfo = new()
        {
            SType = StructureType.ImageViewCreateInfo,
            Image = dummyWhiteImage,
            ViewType = ImageViewType.Type2D,
            Format = Format.R8G8B8A8Unorm,
            SubresourceRange = new ImageSubresourceRange
            {
                AspectMask = ImageAspectFlags.ColorBit,
                BaseMipLevel = 0, LevelCount = 1,
                BaseArrayLayer = 0, LayerCount = 1,
            },
        };
        vk!.CreateImageView(device, &viewInfo, null, out dummyWhiteImageView);

        // Sampler
        SamplerCreateInfo samplerInfo = new()
        {
            SType = StructureType.SamplerCreateInfo,
            MagFilter = Filter.Nearest,
            MinFilter = Filter.Nearest,
            AddressModeU = SamplerAddressMode.ClampToEdge,
            AddressModeV = SamplerAddressMode.ClampToEdge,
            AddressModeW = SamplerAddressMode.ClampToEdge,
            MipmapMode = SamplerMipmapMode.Nearest,
            AnisotropyEnable = false,
            UnnormalizedCoordinates = false,
            CompareEnable = false,
            MinLod = 0.0f,
            MaxLod = 0.0f,
        };
        vk!.CreateSampler(device, &samplerInfo, null, out dummyWhiteSampler);
    }

    private void CreateGBufferSampler()
    {
        SamplerCreateInfo samplerInfo = new()
        {
            SType = StructureType.SamplerCreateInfo,
            MagFilter = Filter.Nearest,
            MinFilter = Filter.Nearest,
            AddressModeU = SamplerAddressMode.ClampToEdge,
            AddressModeV = SamplerAddressMode.ClampToEdge,
            AddressModeW = SamplerAddressMode.ClampToEdge,
            AnisotropyEnable = true,
            MaxAnisotropy = 16,
            BorderColor = BorderColor.FloatOpaqueBlack,
            UnnormalizedCoordinates = false,
            CompareEnable = false,
            CompareOp = CompareOp.Always,
            MipmapMode = SamplerMipmapMode.Nearest,
            MinLod = 0.0f,
            MaxLod = 1.0f,
            MipLodBias = 0.0f,
        };
        if(vk!.CreateSampler(device, &samplerInfo, null, out gBufferSampler) != Result.Success)
        {
            throw new Exception("Failed to create gBuffer sampler");
        }
            
    }
    private void CreateUniformBuffers()
    {
        for (var i = 0; i < MAX_CONCURRENT_FRAMES; i++)
        {
            CreateMappedUniformBuffer(sizeof(GeometryUBO), ref GeometryUniformBuffers[i]);
            CreateMappedUniformBuffer(sizeof(LightingUBO), ref LightingUniformBuffers[i]);
        }
    }

    private void CreateMappedUniformBuffer(int sizeBytes, ref UboBuffer ubo)
    {
        BufferCreateInfo bufferInfo = new()
        {
            SType = StructureType.BufferCreateInfo,
            Size = (ulong)sizeBytes,
            Usage = BufferUsageFlags.UniformBufferBit,
            SharingMode = SharingMode.Exclusive,
        };
        vk!.CreateBuffer(device, &bufferInfo, null, out ubo.buffer);

        vk!.GetBufferMemoryRequirements(device, ubo.buffer, out MemoryRequirements memReqs);
        MemoryAllocateInfo allocInfo = new()
        {
            SType = StructureType.MemoryAllocateInfo,
            AllocationSize = memReqs.Size,
            MemoryTypeIndex = FindMemoryType(vk, physicalDevice, memReqs.MemoryTypeBits,
                MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit)
        };
        vk!.AllocateMemory(device, &allocInfo, null, out ubo.memory);
        vk!.BindBufferMemory(device, ubo.buffer, ubo.memory, 0);
        vk!.MapMemory(device, ubo.memory, 0, (ulong)sizeBytes, MemoryMapFlags.None, ref ubo.mapped);
    }
    
    public void TransitionImageLayout( CommandBuffer cmd, Image image, Format format,  ImageLayout oldLayout,
        ImageLayout newLayout, uint mipLevels = 1)
    {
        bool isDepth = format == Format.D32Sfloat || format == Format.D32SfloatS8Uint || format == Format.D24UnormS8Uint;
        ImageMemoryBarrier barrier = new()
        {
            SType = StructureType.ImageMemoryBarrier,
            OldLayout = oldLayout,
            NewLayout = newLayout,
            SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
            Image = image,
            SubresourceRange =
                new ImageSubresourceRange(isDepth ? ImageAspectFlags.DepthBit : ImageAspectFlags.ColorBit, 0, mipLevels,
                    0, 1)
        };
        
        //Initialize pipeline stage tracking for synchronization timing
        //these stages define when operations must complete and when new operations can begin
        PipelineStageFlags sourceStage;
        PipelineStageFlags destinationStage;
        
        //configure sync for undefined -> transfer layout transition
        //common when preparing images
        if (oldLayout == ImageLayout.Undefined && newLayout == ImageLayout.TransferDstOptimal)
        {
            barrier.SrcAccessMask = 0;
            barrier.DstAccessMask = AccessFlags.TransferWriteBit;
             
            sourceStage = PipelineStageFlags.TopOfPipeBit;
            destinationStage = PipelineStageFlags.TransferBit;

        } 
        //configure sync for transfer -> shader read layout transition
        //pattern prepares uploaded images for shader sampling
        else if (oldLayout == ImageLayout.TransferDstOptimal && newLayout == ImageLayout.ShaderReadOnlyOptimal)
        {
            barrier.SrcAccessMask = AccessFlags.TransferWriteBit;
            barrier.DstAccessMask = AccessFlags.ShaderReadBit;
            
            sourceStage = PipelineStageFlags.TransferBit;
            destinationStage = PipelineStageFlags.FragmentShaderBit;
        }
        else
        {
            throw new Exception("Unsupported layout transition");
        }
        
        vk.CmdPipelineBarrier(cmd,
            sourceStage,
            destinationStage,
            0,
            0, null, 
            0, null,
            1, &barrier);
    }


    private void CreateDescriptorPool()
    {
        // Sizing: 4 UBO (2 geometry + 2 lighting), 14 CIS (10 geometry + 4 G-buffer). Using 20/20 for headroom.
        var poolSizes = new DescriptorPoolSize[]
        {
            new DescriptorPoolSize()
            {
                Type = DescriptorType.UniformBuffer,
                DescriptorCount = (uint)20,
            },
            new DescriptorPoolSize()
            {
                Type = DescriptorType.CombinedImageSampler,
                DescriptorCount = (uint)20,
            }
        };
        fixed (DescriptorPoolSize* poolSizesPtr = poolSizes)
        {
            DescriptorPoolCreateInfo poolInfo = new()
            {
                SType = StructureType.DescriptorPoolCreateInfo,
                MaxSets = (uint)20,
                PoolSizeCount = (uint)poolSizes.Length,
                PPoolSizes = poolSizesPtr,
                Flags = DescriptorPoolCreateFlags.UpdateAfterBindBit | DescriptorPoolCreateFlags.FreeDescriptorSetBit,
            };
            if (vk!.CreateDescriptorPool(device, &poolInfo, null, out descriptorPool) != Result.Success)
            {
                throw new Exception("Failed to create descriptor pool");
            }
        }
    }
    private void CreateDescriptorSets()
    {
        // Per-frame geometry descriptor sets from geometryDescriptorSetLayout.
        // Binding 0 = GeometryUBO (written here). Bindings 1-5 = PBR textures (written in 3e).
        var layouts = stackalloc DescriptorSetLayout[(int)MAX_CONCURRENT_FRAMES];
        for (var i = 0; i < MAX_CONCURRENT_FRAMES; i++) layouts[i] = geometryDescriptorSetLayout;

        DescriptorSetAllocateInfo allocateInfo = new()
        {
            SType = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool = descriptorPool,
            DescriptorSetCount = MAX_CONCURRENT_FRAMES,
            PSetLayouts = layouts
        };
        geometryDescriptorSets = new DescriptorSet[MAX_CONCURRENT_FRAMES];
        fixed (DescriptorSet* pSets = geometryDescriptorSets)
        {
            if (vk!.AllocateDescriptorSets(device, &allocateInfo, pSets) != Result.Success)
                throw new Exception("Failed to allocate geometry descriptor sets");
        }

        for (var i = 0; i < MAX_CONCURRENT_FRAMES; i++)
        {
            DescriptorBufferInfo bufferInfo = new()
            {
                Buffer = GeometryUniformBuffers[i].buffer,
                Offset = 0,
                Range = (ulong)sizeof(GeometryUBO),
            };
            WriteDescriptorSet descriptorWrite = new()
            {
                SType = StructureType.WriteDescriptorSet,
                DstSet = geometryDescriptorSets[i],
                DstBinding = 0,
                DstArrayElement = 0,
                DescriptorType = DescriptorType.UniformBuffer,
                DescriptorCount = 1,
                PBufferInfo = &bufferInfo
            };
            vk!.UpdateDescriptorSets(device, 1, &descriptorWrite, 0, null);
        }

        // Write dummy white texture to bindings 1-5 (baseColor, physical, normal, occlusion, emissive).
        // Shader ignores these when PbrPushConstants.*TextureSet = -1.
        DescriptorImageInfo dummyImgInfo = new()
        {
            ImageView = dummyWhiteImageView,
            Sampler = dummyWhiteSampler,
            ImageLayout = ImageLayout.ShaderReadOnlyOptimal,
        };
        var textureWrites = stackalloc WriteDescriptorSet[5];
        for (var i = 0; i < MAX_CONCURRENT_FRAMES; i++)
        {
            for (uint b = 1; b <= 5; b++)
            {
                textureWrites[b - 1] = new WriteDescriptorSet
                {
                    SType = StructureType.WriteDescriptorSet,
                    DstSet = geometryDescriptorSets[i],
                    DstBinding = b,
                    DstArrayElement = 0,
                    DescriptorType = DescriptorType.CombinedImageSampler,
                    DescriptorCount = 1,
                    PImageInfo = &dummyImgInfo,
                };
            }
            vk!.UpdateDescriptorSets(device, 5, textureWrites, 0, null);
        }
    }

    private void CreateLightingDescriptorSets()
    {
        // Set 0: per-frame LightingUBO
        var lightingLayouts = stackalloc DescriptorSetLayout[(int)MAX_CONCURRENT_FRAMES];
        for (var i = 0; i < MAX_CONCURRENT_FRAMES; i++) lightingLayouts[i] = PBRDescriptorSetLayout;

        DescriptorSetAllocateInfo lightingAlloc = new()
        {
            SType = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool = descriptorPool,
            DescriptorSetCount = MAX_CONCURRENT_FRAMES,
            PSetLayouts = lightingLayouts
        };
        lightingDescriptorSets = new DescriptorSet[MAX_CONCURRENT_FRAMES];
        fixed (DescriptorSet* pSets = lightingDescriptorSets)
        {
            if (vk!.AllocateDescriptorSets(device, &lightingAlloc, pSets) != Result.Success)
                throw new Exception("Failed to allocate lighting descriptor sets");
        }
        for (var i = 0; i < MAX_CONCURRENT_FRAMES; i++)
        {
            DescriptorBufferInfo bufInfo = new()
            {
                Buffer = LightingUniformBuffers[i].buffer,
                Offset = 0,
                Range = (ulong)sizeof(LightingUBO),
            };
            WriteDescriptorSet write = new()
            {
                SType = StructureType.WriteDescriptorSet,
                DstSet = lightingDescriptorSets[i],
                DstBinding = 0,
                DstArrayElement = 0,
                DescriptorType = DescriptorType.UniformBuffer,
                DescriptorCount = 1,
                PBufferInfo = &bufInfo,
            };
            vk!.UpdateDescriptorSets(device, 1, &write, 0, null);
        }

        // Set 1: shared G-buffer samplers (same ImageViews every frame)
        var gBufLayout = PBRGBufferDescriptorSetLayout;
        DescriptorSetAllocateInfo gBufAlloc = new()
        {
            SType = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool = descriptorPool,
            DescriptorSetCount = 1,
            PSetLayouts = &gBufLayout,
        };
        if (vk!.AllocateDescriptorSets(device, &gBufAlloc, out gBufferDescriptorSet) != Result.Success)
            throw new Exception("Failed to allocate G-buffer descriptor set");

        var imageInfos = stackalloc DescriptorImageInfo[4]
        {
            new() { ImageView = gBufferPosition.ImageView, Sampler = gBufferSampler, ImageLayout = ImageLayout.ShaderReadOnlyOptimal },
            new() { ImageView = gBufferNormal.ImageView,   Sampler = gBufferSampler, ImageLayout = ImageLayout.ShaderReadOnlyOptimal },
            new() { ImageView = gBufferAlbedo.ImageView,   Sampler = gBufferSampler, ImageLayout = ImageLayout.ShaderReadOnlyOptimal },
            new() { ImageView = gBufferMaterial.ImageView, Sampler = gBufferSampler, ImageLayout = ImageLayout.ShaderReadOnlyOptimal },
        };
        var writes = stackalloc WriteDescriptorSet[4];
        for (uint i = 0; i < 4; i++)
        {
            writes[i] = new WriteDescriptorSet
            {
                SType = StructureType.WriteDescriptorSet,
                DstSet = gBufferDescriptorSet,
                DstBinding = i,
                DstArrayElement = 0,
                DescriptorType = DescriptorType.CombinedImageSampler,
                DescriptorCount = 1,
                PImageInfo = &imageInfos[i],
            };
        }
        vk!.UpdateDescriptorSets(device, 4, writes, 0, null);
    }

    private void UpdateGeometryUBO(uint frameIndex, Entity* entity, Camera camera)
    {
        var transform = entity->GetComponent<TransformComponent>();
        if (transform == null)
        {
            Console.Error.WriteLine("Entity has no transform component");
            return;
        }

        GeometryUBO ubo = new();
        ubo.model = *transform.GetModelMatrix();
        if (camera != null)
        {
            ubo.proj = camera.GetProjectionMatrix((float)swapChainExtent.Width / swapChainExtent.Height, 0.1f, 100.0f);
            ubo.view = camera.GetViewMatrix();
            ubo.proj.M22 *= -1; // Vulkan clip space has Y down
        }
        else
        {
            ubo.view = Matrix4x4.CreateLookAt(new Vector3(2, 2, 2), new Vector3(0, 0, 0), new Vector3(0, 0, 1));
            ubo.proj = Matrix4x4.CreatePerspectiveFieldOfView((float)(45 * Math.PI / 180),
                (float)swapChainExtent.Width / swapChainExtent.Height, 0.1f, 100.0f);
            ubo.proj.M22 *= -1; // flip Y for Vulkan clip space
        }

        void* data = GeometryUniformBuffers[frameIndex].mapped;
        new Span<GeometryUBO>(data, 1).Fill(ubo);
    }

    private void UpdateLightingUBO(uint frameIndex, Camera camera)
    {
        LightingUBO ubo = new();
        // light 1 white from above
        ubo.lightPosition0 = new Vector4(0.0f, 5.0f, 5.0f, 1.0f);
        ubo.lightColor0 = new Vector4(300.0f, 300.0f, 300.0f, 1.0f);
        // light 2 blue from the left
        ubo.lightPosition1 = new Vector4(-5.0f, 0f, 0f, 1.0f);
        ubo.lightColor1 = new Vector4(0.0f, 0.0f, 300.0f, 1.0f);
        // light 3 red from the right
        ubo.lightPosition2 = new Vector4(5.0f, 0f, 0f, 1.0f);
        ubo.lightColor2 = new Vector4(300.0f, 0.0f, 0.0f, 1.0f);
        // light 4 green from behind
        ubo.lightPosition3 = new Vector4(0.0f, -5.0f, 0f, 1.0f);
        ubo.lightColor3 = new Vector4(0.0f, 300.0f, 0.0f, 1.0f);

        ubo.camPos = camera != null ? new Vector4(camera.GetPosition(), 1.0f) : new Vector4(2, 2, 2, 1);
        ubo.exposure = 4.5f;
        ubo.gamma = 2.2f;
        ubo.prefilteredCubeMipLevels = 1.0f;
        ubo.scaleIBLAmbient = 1.0f;

        void* data = LightingUniformBuffers[frameIndex].mapped;
        new Span<LightingUBO>(data, 1).Fill(ubo);
    }

    private void PushMaterialProperties(CommandBuffer cmd, Material material)
    {
        PbrPushConstants pushConstants = new()
        {
            BaseColorFactor = material.baseColorFactor,
            MetallicFactor = material.metallicFactor,
            RoughnessFactor = material.roughnessFactor,
            BaseColorTextureSet = material.baseColorTextureSet,
            PhysicalDescriptorTextureSet = material.physicalDescriptorTextureSet,
            NormalTextureSet = material.normalTextureSet,
            OcclusionTextureSet = material.occlusionTextureSet,
            EmissiveTextureSet = material.emissiveTextureSet,
            AlphaMask = material.alphaMask,
            AlphaMaskCutoff = material.alphaMaskCutoff
        };
        
        //push constants to shader with cmdbuffer
        vk!.CmdPushConstants(cmd, pbrPipelineLayout, ShaderStageFlags.FragmentBit, 0, (uint)sizeof(PbrPushConstants), &pushConstants);
        
    }
    
    private Extent2D ChooseSwapExtent(SurfaceCapabilitiesKHR capabilities)
    {
        if (capabilities.CurrentExtent.Width != uint.MaxValue)
            return capabilities.CurrentExtent;
        else
        {
            var framebufferSize = App.window!.FramebufferSize;
            Extent2D actualExtent = new()
            {
                Width = (uint)framebufferSize.X,
                Height = (uint)framebufferSize.Y
            };
            actualExtent.Width = Math.Clamp(actualExtent.Width, capabilities.MinImageExtent.Width, capabilities.MaxImageExtent.Width);
            actualExtent.Height = Math.Clamp(actualExtent.Height, capabilities.MinImageExtent.Height, capabilities.MaxImageExtent.Height);
            
            return actualExtent;
        }
    }

    private PresentModeKHR ChooseSwapPresentMode(PresentModeKHR[] presentModes)
    {
        foreach (var availablePresentMode in presentModes)
        {
            if(availablePresentMode == PresentModeKHR.MailboxKhr)
                return availablePresentMode;
        }
        
        return PresentModeKHR.FifoKhr;
    }

    private SurfaceFormatKHR ChooseSwapSurfaceFormat(SurfaceFormatKHR[] formats)
    {
        foreach (var availableFormat in formats)
        {
            if(availableFormat.Format == Format.B8G8R8A8Srgb && availableFormat.ColorSpace == ColorSpaceKHR.PaceSrgbNonlinearKhr)
                return availableFormat;
        }
        
        return formats[0];
    }
}

// Matches Geometry.slang's GBufferUBO (binding 0 of geometryDescriptorSetLayout).
struct GeometryUBO
{
    public Matrix4x4 model;
    public Matrix4x4 view;
    public Matrix4x4 proj;
}

// Matches PbrShader.slang's LightingUBO (binding 0 of PBRDescriptorSetLayout).
// Inline fields instead of managed arrays so the struct is blittable.
struct LightingUBO
{
    public Vector4 lightPosition0;
    public Vector4 lightPosition1;
    public Vector4 lightPosition2;
    public Vector4 lightPosition3;
    public Vector4 lightColor0;
    public Vector4 lightColor1;
    public Vector4 lightColor2;
    public Vector4 lightColor3;
    public Vector4 camPos;
    public float exposure;
    public float gamma;
    public float prefilteredCubeMipLevels;
    public float scaleIBLAmbient;
}

unsafe struct UboBuffer : IDisposable
{
    public Device device;//for m alloc cleanup
    
    public Buffer buffer;
    public DeviceMemory memory;
    public void* mapped;
    
    public void Dispose()
    {
        Vk.GetApi().FreeMemory(device, memory, null);
        Vk.GetApi().DestroyBuffer(device, buffer, null);
        
        GC.SuppressFinalize(this);
    }
}

struct PbrPushConstants
{
    public Vector4 BaseColorFactor;
    public float MetallicFactor;
    public float RoughnessFactor;
    public int BaseColorTextureSet;
    public int PhysicalDescriptorTextureSet;
    public int NormalTextureSet;
    public int OcclusionTextureSet;
    public int EmissiveTextureSet;
    public float AlphaMask;
    public float AlphaMaskCutoff;
}