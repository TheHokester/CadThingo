using CadThingo.GraphicsPipeline;
using Silk.NET.Maths;
using Silk.NET.Vulkan;
using Silk.NET.Vulkan.Extensions.KHR;
using Silk.NET.Vulkan.Extensions.EXT;
using Silk.NET.Windowing;


namespace CadThingo;

public class VulkanSwapchain
{
    private static readonly Vk? vk = Globals.vk;
    private VulkanContext ctx;
    public VulkanSwapchain(VulkanContext context)
        => ctx = context;
    public struct SwapChainSupportDetails
    {
        public SurfaceCapabilitiesKHR Capabilities;
        public SurfaceFormatKHR[] Formats;
        public PresentModeKHR[] PresentModes;

        
    }

    public unsafe void CreateSwapChain()
    {
        var swapChainSupport = QuerySwapChainSupport(ctx.PhysicalDevice, ctx.Surface, ctx.KhrSurface);
        
        var surfaceFormat = ChooseSwapSurfaceFormat(swapChainSupport.Formats);
        var presentMode = ChoosePresentMode(swapChainSupport.PresentModes);
        var extent = ChooseSwapExtent(swapChainSupport.Capabilities);
        
        var imageCount = swapChainSupport.Capabilities.MinImageCount + 1;
        if(swapChainSupport.Capabilities.MaxImageCount > 0 && imageCount > swapChainSupport.Capabilities.MaxImageCount)
            imageCount = swapChainSupport.Capabilities.MaxImageCount;

        SwapchainCreateInfoKHR swapChainInfo = new()
        {
            SType = StructureType.SwapchainCreateInfoKhr,
            Surface = ctx.Surface,

            MinImageCount = imageCount,
            ImageFormat = surfaceFormat.Format,
            ImageColorSpace = surfaceFormat.ColorSpace,
            ImageExtent = extent,
            ImageArrayLayers = 1,
            ImageUsage = ImageUsageFlags.ColorAttachmentBit,
        };
        var indices = VulkanDevice.FindQueueFamilies(ctx.PhysicalDevice, ctx.Surface, ctx.KhrSurface);
        var queueFamilyIndices = stackalloc[] { indices.graphicsFamily!.Value, indices.presentFamily!.Value };

        if (indices.graphicsFamily != indices.presentFamily)
        {
            swapChainInfo = swapChainInfo with
            {
                ImageSharingMode = SharingMode.Concurrent,
                QueueFamilyIndexCount = 2,
                PQueueFamilyIndices = queueFamilyIndices
            };
        }
        else
        {
            swapChainInfo.ImageSharingMode = SharingMode.Exclusive;
        }

        swapChainInfo = swapChainInfo with
        {
            PreTransform = swapChainSupport.Capabilities.CurrentTransform,
            CompositeAlpha = CompositeAlphaFlagsKHR.OpaqueBitKhr,
            PresentMode = presentMode,
            Clipped = true,

            OldSwapchain = default
        };

        if (!vk!.TryGetDeviceExtension(ctx.Instance, ctx.Device, out  ctx.KhrSwapChain))
        {
            throw new Exception("Swapchain ext not found");
        }

        if (ctx.KhrSwapChain!.CreateSwapchain(ctx.Device, &swapChainInfo, null, out ctx.SwapChain) != Result.Success)
            throw new Exception("Failed to create swapchain");
        
        ctx.KhrSwapChain.GetSwapchainImages(ctx.Device, ctx.SwapChain, &imageCount, null);
        ctx.SwapChainImages = new Image[imageCount];
        fixed (Image* imagesPtr = ctx.SwapChainImages)
        {
            ctx.KhrSwapChain.GetSwapchainImages(ctx.Device, ctx.SwapChain, &imageCount, imagesPtr);
        }
        ctx.SwapChainImageFormat = surfaceFormat.Format;
        ctx.SwapChainExtent = extent;
        
        Console.WriteLine($"Created swapchain with {ctx.SwapChainImages.Length} images");
    }

    public void RecreateSwapChain(IWindow? window, VulkanPipeline? pipeline, VulkanRenderer? renderer)
    {
        
        Vector2D<int> framebufferSize = window!.FramebufferSize;

        while (framebufferSize.X == 0 || framebufferSize.Y == 0)
        {
            framebufferSize = window.FramebufferSize;
            window.DoEvents();
        }
        vk!.DeviceWaitIdle(ctx.Device);
        
        CleanupSwapChain();
        
        CreateSwapChain();
        CreateImageViews();
        pipeline!.CreateRenderPass();
        pipeline.CreateGraphicsPipeline();
        CreateFrameBuffer();
        renderer!.CreateCommandBuffers();
        
        ctx.ImagesInFlight = new Fence[ctx.SwapChainImages!.Length];
    }

    public unsafe void CleanupSwapChain()
    {
        foreach (var frameBuffer in ctx.SwapChainFramebuffers!)
        {
            vk!.DestroyFramebuffer(ctx.Device, frameBuffer, null);
        }

        fixed (CommandBuffer* commandBuffers = ctx.CommandBuffers)
        {
            vk!.FreeCommandBuffers(ctx.Device, ctx.CommandPool, (uint)ctx.CommandBuffers!.Length, commandBuffers);
        }
        
        vk!.DestroyRenderPass(ctx.Device, ctx.RenderPass, null);
        vk!.DestroyPipeline(ctx.Device, ctx.Pipeline, null);
        vk!.DestroyPipelineLayout(ctx.Device, ctx.PipelineLayout, null);
        
        foreach (var imageView in ctx.SwapChainImageViews!)
        {
            vk!.DestroyImageView(ctx.Device, imageView, null);
        }
        ctx.KhrSwapChain!.DestroySwapchain(ctx.Device, ctx.SwapChain, null);
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

    private PresentModeKHR ChoosePresentMode(PresentModeKHR[] presentModes)
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

    public static unsafe SwapChainSupportDetails QuerySwapChainSupport(PhysicalDevice physicalDevice, SurfaceKHR surface,  KhrSurface? khrSurface)
    {
        var details = new SwapChainSupportDetails();
        khrSurface!.GetPhysicalDeviceSurfaceCapabilities(physicalDevice, surface, out details.Capabilities);

        uint formatCount = 0;
        khrSurface!.GetPhysicalDeviceSurfaceFormats(physicalDevice, surface, &formatCount, null);

        if (formatCount != 0)
        {
            details.Formats = new SurfaceFormatKHR[formatCount];
            fixed (SurfaceFormatKHR* formatsPtr = details.Formats)
            {
                khrSurface!.GetPhysicalDeviceSurfaceFormats(physicalDevice, surface, &formatCount, formatsPtr);
            }
            
        }
        else
        {
            details.Formats = Array.Empty<SurfaceFormatKHR>();
        }

        uint presentModeCount = 0;
        khrSurface!.GetPhysicalDeviceSurfacePresentModes(physicalDevice, surface, &presentModeCount, null);
        if (presentModeCount != 0)
        {
            details.PresentModes = new PresentModeKHR[presentModeCount];
            fixed (PresentModeKHR* formatsPtr = details.PresentModes)
            {
                khrSurface!.GetPhysicalDeviceSurfacePresentModes(physicalDevice, surface, &presentModeCount,
                    formatsPtr);
            }
        }
        else
        {
            details.PresentModes = Array.Empty<PresentModeKHR>();
        }
        
        return details;
    }

    public unsafe void CreateImageViews()
    {
        ctx.SwapChainImageViews = new ImageView[ctx.SwapChainImages!.Length];

        for (int i = 0; i < ctx.SwapChainImages.Length; i++)
        {
            ImageViewCreateInfo createInfo = new()
            {
                SType = StructureType.ImageViewCreateInfo,
                Image = ctx.SwapChainImages[i],

                ViewType = ImageViewType.Type2D,
                Format = ctx.SwapChainImageFormat,
                Components = new ComponentMapping(ComponentSwizzle.R,
                    ComponentSwizzle.G,
                    ComponentSwizzle.B,
                    ComponentSwizzle.A),
                SubresourceRange = new ImageSubresourceRange(ImageAspectFlags.ColorBit, 0, 1, 0, 1)
            };

            if (vk!.CreateImageView(ctx.Device, &createInfo, null, out ctx.SwapChainImageViews[i]) != Result.Success)
            {
                throw new Exception("Failed to create image view");
            }
        }
        Console.WriteLine("Created image views");
    }

    public unsafe void CreateFrameBuffer()
    {
        ctx.SwapChainFramebuffers = new Framebuffer[ctx.SwapChainImageViews!.Length];

        for (var i = 0; i < ctx.SwapChainImageViews!.Length; i++)
        {
            var attachment = ctx.SwapChainImageViews[i];
            
            FramebufferCreateInfo framebufferInfo = new()
            {
                SType = StructureType.FramebufferCreateInfo,
                RenderPass = ctx.RenderPass,
                AttachmentCount = 1,
                PAttachments = &attachment,
                Width = ctx.SwapChainExtent.Width,
                Height = ctx.SwapChainExtent.Height,
                Layers = 1
            };
            if(vk!.CreateFramebuffer(ctx.Device, &framebufferInfo, null, out ctx.SwapChainFramebuffers[i]) != Result.Success)
                throw new Exception("Failed to create framebuffer");
        }
    }

    public unsafe void DestroyFrameBuffers()
    {
        foreach(var frameBuffer in ctx.SwapChainFramebuffers)
        {
            vk!.DestroyFramebuffer(ctx.Device, frameBuffer, null);
        }
    }
    public unsafe void DestroyImageViews()
    {
        foreach (var imageView in ctx.SwapChainImageViews!)
        {
            vk!.DestroyImageView(ctx.Device, imageView, null);
        }
    }
}