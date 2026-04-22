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
        gBufferPosition = new ImageResource(vk, device, "GBuffer_Position", Format.R32G32B32Sfloat,
            new Extent2D(width, height),
            ImageUsageFlags.ColorAttachmentBit | ImageUsageFlags.InputAttachmentBit,
            ImageLayout.Undefined, ImageLayout.ShaderReadOnlyOptimal);
        gBufferNormal = new ImageResource(vk, device, "GBuffer_Normal", Format.R32G32B32A32Sfloat,
            new Extent2D(width, height),
            ImageUsageFlags.ColorAttachmentBit | ImageUsageFlags.InputAttachmentBit,
            ImageLayout.Undefined, ImageLayout.ShaderReadOnlyOptimal);
        gBufferAlbedo = new ImageResource(vk, device, "GBuffer_Albedo", Format.R8G8B8A8Unorm,
            new Extent2D(width, height),
            ImageUsageFlags.ColorAttachmentBit | ImageUsageFlags.InputAttachmentBit,
            ImageLayout.Undefined, ImageLayout.ShaderReadOnlyOptimal);
        gBufferMaterial = new ImageResource(vk, device, "GBuffer_Material", Format.R8G8B8A8Unorm,
            new Extent2D(width, height),
            ImageUsageFlags.ColorAttachmentBit | ImageUsageFlags.InputAttachmentBit,
            ImageLayout.Undefined, ImageLayout.ShaderReadOnlyOptimal);
        
    }
    private void CreateUniformBuffers()
    {
        int bufferSize = sizeof(UniformBufferObject);
        //create the buffer
        BufferCreateInfo bufferInfo = new()
        {
            SType = StructureType.BufferCreateInfo,
            Size = (ulong)bufferSize,
            Usage = BufferUsageFlags.UniformBufferBit,
            SharingMode = SharingMode.Exclusive,
        };
        for (var i = 0; i < MAX_CONCURRENT_FRAMES; i++)
        {
            vk!.CreateBuffer(device, &bufferInfo, null, out UniformBuffers[i].buffer);
            
            
            //allocate and bind memory
            vk!.GetBufferMemoryRequirements(device, UniformBuffers[i].buffer, out MemoryRequirements memReqs);

            MemoryAllocateInfo allocInfo = new()
            {
                SType = StructureType.MemoryAllocateInfo,
                AllocationSize = memReqs.Size,
                MemoryTypeIndex = FindMemoryType(vk, physicalDevice, memReqs.MemoryTypeBits,
                    MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit)
            };

            vk!.AllocateMemory(device, &allocInfo, null, out UniformBuffers[i].memory);
            vk!.BindBufferMemory(device, UniformBuffers[i].buffer, UniformBuffers[i].memory, 0);
            vk!.MapMemory(device, UniformBuffers[i].memory, 0, (ulong)bufferSize, MemoryMapFlags.None,
                ref UniformBuffers[i].mapped);
            
        }
        
        
    }
    
    public void TransitionImageLayout( CommandBuffer cmd, Image image, Format format,  ImageLayout oldLayout,
        ImageLayout newLayout, uint mipLevels = 1)
    {
        bool isDepth = format == Format.D32Sfloat || format == Format.D32SfloatS8Uint || format == Format.D24UnormS8Uint;
        ImageMemoryBarrier barrier = new()
        {
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
        var poolSizes = new DescriptorPoolSize[]
        {
            new DescriptorPoolSize()
            {
                Type = DescriptorType.UniformBuffer,
                DescriptorCount = (uint)MAX_CONCURRENT_FRAMES,
            },
            new DescriptorPoolSize()
            {
                Type = DescriptorType.CombinedImageSampler,
                DescriptorCount = (uint)MAX_CONCURRENT_FRAMES,
            }
        };
        fixed (DescriptorPoolSize* poolSizesPtr = poolSizes)
        {
            DescriptorPoolCreateInfo poolInfo = new()
            {
                SType = StructureType.DescriptorPoolCreateInfo,
                MaxSets = (uint)MAX_CONCURRENT_FRAMES,
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
        var numLayouts = 2;
        var layouts = stackalloc DescriptorSetLayout[numLayouts];
        for (var i = 0; i < numLayouts; i++) layouts[i] = descriptorSetLayout;

        DescriptorSetAllocateInfo allocateInfo = new()
        {
            SType = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool = descriptorPool,
            DescriptorSetCount = MAX_CONCURRENT_FRAMES,
            PSetLayouts = layouts
        };
        descriptorSets = new DescriptorSet[numLayouts];
        vk!.AllocateDescriptorSets(device, &allocateInfo, descriptorSets);

        DescriptorBufferInfo bufferInfo = new()
        {
            Offset = 0,
            Range = (ulong)sizeof(UniformBufferObject),
        };
        for (var i = 0; i < MAX_CONCURRENT_FRAMES; i++)
        {
            bufferInfo.Buffer = UniformBuffers[i].buffer;
            WriteDescriptorSet descriptorWrite = new()
            {
                SType = StructureType.WriteDescriptorSet,
                DstSet = descriptorSets[i],
                DstBinding = 0,
                DstArrayElement = 0,
                DescriptorType = DescriptorType.UniformBuffer,
                DescriptorCount = 1,
                PBufferInfo = &bufferInfo
                
            };
            
            vk!.UpdateDescriptorSets(device, 1, &descriptorWrite, 0, null);
        }
    }
    
    private void UpdateUniformBuffers(uint currentFrame, Entity* entity, Camera camera)
    {
        //get the transform component from the entity
        var transform = entity->GetComponent<TransformComponent>();
        if (transform == null)
        {
            Console.Error.WriteLine("Entity has no transform component");
            return;
        }
        //Create the UBO
        UniformBufferObject ubo = new();
        ubo.model = *transform->GetModelMatrix();
        if (camera != null)
        {
            ubo.proj = camera.GetProjectionMatrix((float)swapChainExtent.Width/swapChainExtent.Height, 0.1f, 100.0f);
            ubo.view = camera.GetViewMatrix();
        }
        else
        {
            //default view and proj matrices
            ubo.view = Matrix4x4.CreateLookAt(new Vector3(2, 2, 2), new Vector3(0, 0, 0), new Vector3(0, 0, 1));
            ubo.proj = Matrix4x4.CreatePerspectiveFieldOfView((float)(45*Math.PI/180), (float)swapChainExtent.Width/swapChainExtent.Height, 0.1f, 100.0f );
            ubo.proj.M22 *= -1;
        }
        //setup lights
        ubo.lightPositions = new Vector4[4];
        ubo.lightColors = new Vector4[4];
        //light 1 white light from above
        ubo.lightPositions[0] = new Vector4(0.0f, 5.0f, 5.0f, 1.0f);
        ubo.lightColors[0] = new Vector4(300.0f, 300.0f, 300.0f, 1.0f);
        //light 2 blue light from the left
        ubo.lightPositions[1] = new Vector4(-5.0f, 0f, 0f, 1.0f);
        ubo.lightColors[1] = new Vector4(0.0f, 0.0f, 300.0f, 1.0f);
        //light 3 red light from the right
        ubo.lightPositions[2] = new Vector4(5.0f, 0f, 0f, 1.0f);
        ubo.lightColors[2] = new Vector4(300.0f, 0.0f, 0.0f, 1.0f);
        //light 4 green light from behind
        ubo.lightPositions[3] = new Vector4(0.0f, -5.0f, 0f, 1.0f);
        ubo.lightColors[3] = new Vector4(0.0f, 300.0f, 0.0f, 1.0f);
        
        ubo.camPos = new Vector4(camera.GetPosition(), 1.0f);
        //Pbr parameters
        ubo.exposure = 4.5f;
        ubo.gamma = 2.2f;
        ubo.prefilterredCubeMipLevels = 1.0f;
        ubo.scaleIBLAmbient = 1.0f;
        //copy the data to the uniform buffer
        void* data = UniformBuffers[currentFrame].mapped;
        new Span<UniformBufferObject>(data, 1).Fill(ubo);
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

struct UniformBufferObject
{
    public Matrix4x4 model;
    public Matrix4x4 view;
    public Matrix4x4 proj;
    public Vector4[] lightPositions;
    public Vector4[] lightColors;
    public Vector4 camPos;
    public float exposure;
    public float gamma;
    public float prefilterredCubeMipLevels;
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