using Silk.NET.Vulkan;

namespace CadThingo.GraphicsPipeline;

public class DepthResources(VulkanContext ctx)
{
    Vk? vk = Globals.vk;
    
    public unsafe void CreateDepthResources()
    {
        var depthFormat = FindDepthFormat();
        VulkanRenderer._Texturing.CreateImage(ctx.SwapChainExtent.Width, ctx.SwapChainExtent.Height, 1, ctx.MsaaSamples,depthFormat,
            ImageTiling.Optimal, ImageUsageFlags.DepthStencilAttachmentBit, MemoryPropertyFlags.DeviceLocalBit,
            out ctx.DepthImage, out ctx.DepthImageMemory);
        
        ctx.DepthImageView = VulkanRenderer._SwapChain.CreateImageView(ctx.DepthImage, depthFormat, ImageAspectFlags.DepthBit, 1);
    }

    private Format FindSupportedFormat(Format[] candidates, ImageTiling tiling, FormatFeatureFlags features)
    {
        foreach (var format in candidates)
        {
            FormatProperties props;
            vk!.GetPhysicalDeviceFormatProperties(ctx.PhysicalDevice, format, out props);
            if (tiling == ImageTiling.Linear && (props.LinearTilingFeatures & features) == features)
            {
                return format;
            }else if (tiling == ImageTiling.Optimal && (props.OptimalTilingFeatures & features) == features)
            {
                return format;
            }
        }
        throw new Exception("failed to find suitable format!");
    }

    public Format FindDepthFormat()
    {
        return FindSupportedFormat(new[] { Format.D32Sfloat, Format.D32SfloatS8Uint, Format.D24UnormS8Uint },
            ImageTiling.Optimal, FormatFeatureFlags.DepthStencilAttachmentBit);
    }
}