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
        var img = ImageSharp.Image.Load<ImageSharp.PixelFormats.Rgba32>("C:\\Users\\jamie\\RiderProjects\\CadThingo\\CadThingo\\Assets\\Textures\\viking_room.png");
        
        ulong imageSize = (ulong)img.Width * (ulong)img.Height * 4;
        ctx.MipLevels = (uint)(Math.Floor(Math.Log2(Math.Max(img.Width, img.Height))));
        
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
            CreateImage((uint)img.Width, (uint)img.Height, ctx.MipLevels, SampleCountFlags.Count1Bit, Format.R8G8B8A8Srgb, ImageTiling.Optimal, ImageUsageFlags.TransferSrcBit |
                ImageUsageFlags.TransferDstBit | ImageUsageFlags.SampledBit, MemoryPropertyFlags.DeviceLocalBit,
                out ctx.TextureImage, out ctx.TextureImageMemory);
            
            TransitionImageLayout(ctx.TextureImage, Format.R8G8B8A8Srgb, ImageLayout.Undefined, ImageLayout.TransferDstOptimal, ctx.MipLevels);
            CopyBufferToImage(stagingBuffer, ctx.TextureImage, (uint)img.Width, (uint)img.Height);
            //Transitioned to VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL while generating mipmaps
        }
        
        vk!.DestroyBuffer(ctx.Device, stagingBuffer, null);
        vk!.FreeMemory(ctx.Device, stagingBufferMemory, null);
        
        GenerateMipMaps(ctx.TextureImage, Format.R8G8B8A8Srgb, (int)img.Width, (int)img.Height, ctx.MipLevels);
    }

    public void CreateTextureImageView()
    {
        
        ctx.TextureImageView = VulkanRenderer._SwapChain.CreateImageView(ctx.TextureImage, Format.R8G8B8A8Srgb, ImageAspectFlags.ColorBit, ctx.MipLevels);  
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
        samplerInfo.MinLod = 0;
        samplerInfo.MaxLod = Vk.LodClampNone;

        if (vk!.CreateSampler(ctx.Device, &samplerInfo, null, out ctx.TextureSampler) != Result.Success)
        {
            throw new Exception("Failed to create texture sampler");
        }
    }

    

    public unsafe void CreateImage(uint width, uint height, uint mipLevels,SampleCountFlags numSamples, Format format, ImageTiling tiling, ImageUsageFlags usage,
        MemoryPropertyFlags properties, out VkImage image, out DeviceMemory imageMemory)
    {
        ImageCreateInfo imageInfo = new ImageCreateInfo()
        {
            SType = StructureType.ImageCreateInfo,
            ImageType = ImageType.Type2D,
            Extent = new Extent3D(width, height, 1),
            MipLevels = mipLevels,
            ArrayLayers = 1,
            Format = format,
            Tiling = ImageTiling.Optimal,
            InitialLayout = ImageLayout.Undefined,
            Usage = usage,
            Samples = numSamples,
            SharingMode = SharingMode.Exclusive,
        };
        if (vk!.CreateImage(ctx.Device, &imageInfo, null, out image) != Result.Success)
        {
            throw new Exception("Failed to create image");
        }
        vk!.GetImageMemoryRequirements(ctx.Device, image,  out MemoryRequirements memReqs);

        MemoryAllocateInfo allocateInfo = new()
        {
            SType = StructureType.MemoryAllocateInfo,
            AllocationSize = memReqs.Size,
            MemoryTypeIndex = VulkanRenderer._Buffers.FindMemoryType(memReqs.MemoryTypeBits, properties)
        };

        if (vk!.AllocateMemory(ctx.Device, &allocateInfo, null, out imageMemory) != Result.Success)
        {
            throw new Exception("Failed to allocate image memory");
        }
        vk!.BindImageMemory(ctx.Device, image, imageMemory, 0);
    }

    public unsafe void CreateColorResources()
    {
        Format ColorFormat = ctx.SwapChainImageFormat;
        
        CreateImage(ctx.SwapChainExtent.Width, ctx.SwapChainExtent.Height, 1, ctx.MsaaSamples, ColorFormat,
            ImageTiling.Optimal, ImageUsageFlags.TransientAttachmentBit | ImageUsageFlags.ColorAttachmentBit, 
            MemoryPropertyFlags.DeviceLocalBit,out ctx.ColorImage,out ctx.ColorImageMemory);
        
        ctx.ColorImageView = VulkanRenderer._SwapChain.CreateImageView(ctx.ColorImage, ColorFormat, ImageAspectFlags.ColorBit, 1);
    }

    private unsafe void GenerateMipMaps(VkImage image, Format format, int texWidth, int texHeight, uint mipLevels)
    {
        var commandBuffer = VulkanRenderer._Commands.BeginSingleTimeCommands();

        FormatProperties formatProperties;
        vk!.GetPhysicalDeviceFormatProperties(ctx.PhysicalDevice, format, out formatProperties);
        if ((formatProperties.OptimalTilingFeatures & FormatFeatureFlags.SampledImageFilterLinearBit) == 0)
        {
            throw new Exception("texture image format does not support linear blitting");
        }
        
        var barrier = new ImageMemoryBarrier()
        {
            SType = StructureType.ImageMemoryBarrier,
            Image = image,
            SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
            SubresourceRange = new()
            {
                AspectMask = ImageAspectFlags.ColorBit,
                LevelCount = 1,
                LayerCount = 1
            },
        };
        var mipWidth = texWidth;
        var mipHeight = texHeight;

        for (uint i = 1; i < mipLevels; i++)
        {
            barrier.SubresourceRange.BaseMipLevel = i - 1;
            barrier.OldLayout = ImageLayout.TransferDstOptimal;
            barrier.NewLayout = ImageLayout.TransferSrcOptimal;
            barrier.SrcAccessMask = AccessFlags.TransferWriteBit;
            barrier.DstAccessMask = AccessFlags.TransferReadBit;
            
            vk!.CmdPipelineBarrier(commandBuffer, PipelineStageFlags.TransferBit, PipelineStageFlags.TransferBit, 0 ,
                0, null,
                0, null,
                1, &barrier);

            ImageBlit blit = new()
            {
                SrcOffsets =
                {
                    Element0 = new Offset3D(0,0,0),
                    Element1 = new Offset3D(mipWidth, mipHeight,1),
                },
                SrcSubresource =
                {
                    AspectMask = ImageAspectFlags.ColorBit,
                    MipLevel = i - 1,
                    BaseArrayLayer = 0,
                    LayerCount = 1
                },
                DstOffsets =
                {
                    Element0 = new Offset3D(0, 0, 0),
                    Element1 = new Offset3D((int)(mipWidth > 1 ? mipWidth/2 : 1), (int)(mipHeight > 1 ? mipHeight/2 : 1), 1),
                },
                DstSubresource =
                {
                    AspectMask = ImageAspectFlags.ColorBit,
                    MipLevel = i,
                    BaseArrayLayer = 0,
                    LayerCount = 1
                }
            };
            
            vk!.CmdBlitImage(commandBuffer,
                image, ImageLayout.TransferSrcOptimal,
                image, ImageLayout.TransferDstOptimal,
                1, &blit,
                Filter.Linear);
            
            barrier.OldLayout = ImageLayout.TransferSrcOptimal;
            barrier.NewLayout = ImageLayout.ShaderReadOnlyOptimal;
            barrier.SrcAccessMask = AccessFlags.TransferReadBit;
            barrier.DstAccessMask = AccessFlags.ShaderReadBit;
            
            vk!.CmdPipelineBarrier(commandBuffer, PipelineStageFlags.TransferBit, PipelineStageFlags.FragmentShaderBit, 0,
                0, null,
                0, null,
                1, &barrier);

            if (mipWidth > 1) mipWidth /= 2;
            if (mipHeight > 1) mipHeight /= 2;
        }
        barrier.SubresourceRange.BaseMipLevel = mipLevels -1;
        barrier.OldLayout = ImageLayout.TransferDstOptimal;
        barrier.NewLayout = ImageLayout.ShaderReadOnlyOptimal;
        barrier.SrcAccessMask = AccessFlags.TransferWriteBit;
        barrier.DstAccessMask = AccessFlags.ShaderReadBit;
        
        vk!.CmdPipelineBarrier(commandBuffer, PipelineStageFlags.TransferBit, PipelineStageFlags.FragmentShaderBit, 0,
            0, null,
            0, null,
            1, &barrier);
        
        VulkanRenderer._Commands.EndSingleTimeCommands(commandBuffer);
    }
    private unsafe void TransitionImageLayout(VkImage image, Format format, ImageLayout oldLayout, ImageLayout newLayout, uint mipLevels)
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
                LevelCount = mipLevels,
                BaseArrayLayer = 0,
                LayerCount = 1,
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