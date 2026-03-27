using SixLabors.ImageSharp.PixelFormats;

namespace CadThingo.GraphicsPipeline;
using Silk.NET.Vulkan;
using Silk.NET.Vulkan.Extensions.KHR;

using Silk.NET.Maths;
using System.Numerics;
using System.Runtime.InteropServices;
using Silk.NET.Core.Native;
using Silk.NET.Core;


public class Texturing(VulkanContext ctx)
{
    Vk? vk = Globals.vk;

    public unsafe void CreateTextureImage()
    {
        var img = ImageSharp.Image.Load<ImageSharp.PixelFormats.Rgba32>("C:\\Users\\jamie\\RiderProjects\\CadThingo\\CadThingo\\Assets\\Textures\\Dark_Side_of_the_Moon.png");
        
        ulong imageSize = (ulong)img.Width * (ulong)img.Height * 4;
        
        Buffer stagingBuffer = new();
        DeviceMemory stagingBufferMemory = new();
        VulkanRenderer._Buffers.CreateBuffer(imageSize, BufferUsageFlags.TransferSrcBit,
            MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit,
            &stagingBuffer, &stagingBufferMemory);

        void* data;
        vk!.MapMemory(ctx.Device, stagingBufferMemory, 0, imageSize,0 , &data);
        img.CopyPixelDataTo(new Span<Rgba32>(data, img.Width * img.Height));
        vk!.UnmapMemory(ctx.Device, stagingBufferMemory);
        fixed(DeviceMemory* txtrImgMemPtr = &ctx.TextureImageMemory )
        fixed (VkImage* txtrImgPtr = &ctx.TextureImage)
        {
            CreateImage((uint)img.Width, (uint)img.Height, Format.R8G8B8A8Srgb, ImageTiling.Optimal,
                ImageUsageFlags.TransferDstBit | ImageUsageFlags.SampledBit, MemoryPropertyFlags.DeviceLocalBit,
                txtrImgPtr, txtrImgMemPtr);
            
            TransitionImageLayout(ctx.TextureImage, Format.R8G8B8A8Srgb, ImageLayout.Undefined, ImageLayout.TransferDstOptimal);
            CopyBufferToImage(stagingBuffer, ctx.TextureImage, (uint)img.Width, (uint)img.Height);
            TransitionImageLayout(ctx.TextureImage, Format.R8G8B8A8Srgb, ImageLayout.TransferDstOptimal, ImageLayout.ShaderReadOnlyOptimal);
        }
        
        vk!.DestroyBuffer(ctx.Device, stagingBuffer, null);
        vk!.FreeMemory(ctx.Device, stagingBufferMemory, null);
    }

    public void CreateTextureImageView()
    {
        
        ctx.TextureImageView = VulkanRenderer._SwapChain.CreateImageView(ctx.TextureImage, Format.R8G8B8A8Srgb);  
    }

    public unsafe void CreateTextureSampler()
    {
        SamplerCreateInfo samplerInfo = new()
        {
            SType = StructureType.SamplerCreateInfo,
            MagFilter = Filter.Linear,
            MinFilter = Filter.Linear,
            AddressModeU = SamplerAddressMode.Repeat,
            AddressModeV = SamplerAddressMode.Repeat,
            AddressModeW = SamplerAddressMode.Repeat,
            AnisotropyEnable = true
        };
        PhysicalDeviceProperties deviceProperties;
        vk!.GetPhysicalDeviceProperties(ctx.PhysicalDevice, &deviceProperties);
        
        samplerInfo.MaxAnisotropy = deviceProperties.Limits.MaxSamplerAnisotropy;
        samplerInfo.BorderColor = BorderColor.FloatOpaqueBlack;
        samplerInfo.UnnormalizedCoordinates = false;
        
        samplerInfo.CompareEnable = false;
        samplerInfo.CompareOp = CompareOp.Always;
        
        samplerInfo.MipmapMode = SamplerMipmapMode.Linear;
        samplerInfo.MipLodBias = 0.0f;
        samplerInfo.MinLod = 0.0f;
        samplerInfo.MaxLod = 0.0f;

        if (vk!.CreateSampler(ctx.Device, &samplerInfo, null, out ctx.TextureSampler) != Result.Success)
        {
            throw new Exception("Failed to create texture sampler");
        }
    }
    
    

    private unsafe void CreateImage(uint width, uint height, Format format, ImageTiling tiling, ImageUsageFlags usage,
        MemoryPropertyFlags properties, VkImage* image, DeviceMemory* imageMemory)
    {
        ImageCreateInfo imageInfo = new ImageCreateInfo()
        {
            SType = StructureType.ImageCreateInfo,
            ImageType = ImageType.Type2D,
            Extent = new Extent3D(width, height, 1),
            MipLevels = 1,
            ArrayLayers = 1,
            Format = Format.R8G8B8A8Srgb,
            Tiling = ImageTiling.Optimal,
            InitialLayout = ImageLayout.Undefined,
            Usage = usage,
            Samples = SampleCountFlags.Count1Bit,
            SharingMode = SharingMode.Exclusive,
        };
        if (vk!.CreateImage(ctx.Device, &imageInfo, null, image) != Result.Success)
        {
            throw new Exception("Failed to create image");
        }
        vk!.GetImageMemoryRequirements(ctx.Device, *image,  out MemoryRequirements memReqs);

        MemoryAllocateInfo allocateInfo = new()
        {
            SType = StructureType.MemoryAllocateInfo,
            AllocationSize = memReqs.Size,
            MemoryTypeIndex = VulkanRenderer._Buffers.FindMemoryType(memReqs.MemoryTypeBits, properties)
        };

        if (vk!.AllocateMemory(ctx.Device, &allocateInfo, null, imageMemory) != Result.Success)
        {
            throw new Exception("Failed to allocate image memory");
        }
        vk!.BindImageMemory(ctx.Device, *image, *imageMemory, 0);
    }

    private unsafe void TransitionImageLayout(VkImage image, Format format, ImageLayout oldLayout, ImageLayout newLayout)
    {
        CommandBuffer commandBuffer = VulkanRenderer._Commands.BeginSingleTimeCommands();

        ImageMemoryBarrier barrier = new()
        {
            SType = StructureType.ImageMemoryBarrier,
            OldLayout = oldLayout,
            NewLayout = newLayout,
            
            SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
            
            Image = image,
            SubresourceRange = new()
            {
                AspectMask = ImageAspectFlags.ColorBit,
                BaseMipLevel = 0,
                LevelCount = 1,
                BaseArrayLayer = 0,
                LayerCount = 1
            },
        };
        PipelineStageFlags sourceStage;
        PipelineStageFlags destinationStage;

        if (oldLayout == ImageLayout.Undefined && newLayout == ImageLayout.TransferDstOptimal)
        {
             barrier.SrcAccessMask = 0;
             barrier.DstAccessMask = AccessFlags.TransferWriteBit;
             
             sourceStage = PipelineStageFlags.TopOfPipeBit;
             destinationStage = PipelineStageFlags.TransferBit;

        } else if (oldLayout == ImageLayout.TransferDstOptimal && newLayout == ImageLayout.ShaderReadOnlyOptimal)
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
        vk!.CmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0, null, 0, null, 1, &barrier);
        
        VulkanRenderer._Commands.EndSingleTimeCommands(commandBuffer);
    }

    private unsafe void CopyBufferToImage(Buffer buffer, VkImage image, uint width, uint height)
    {
        var commandBuffer = VulkanRenderer._Commands.BeginSingleTimeCommands();

        BufferImageCopy region = new()
        {
            BufferOffset = 0,
            BufferRowLength = 0,
            BufferImageHeight = 0,
            ImageSubresource = new()
            {
                AspectMask = ImageAspectFlags.ColorBit,
                MipLevel = 0,
                BaseArrayLayer = 0,
                LayerCount = 1
            },
            ImageOffset = new Offset3D(0, 0, 0),
            ImageExtent = new Extent3D(width, height, 1)
        };
        
        vk!.CmdCopyBufferToImage(commandBuffer, buffer, image, ImageLayout.TransferDstOptimal, 1, &region);
        VulkanRenderer._Commands.EndSingleTimeCommands(commandBuffer);
    }
}