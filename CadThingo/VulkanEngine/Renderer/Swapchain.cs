using Silk.NET.Vulkan;
using Silk.NET.Vulkan.Extensions.KHR;
using Silk.NET.Windowing;
// ReSharper disable InconsistentNaming

namespace CadThingo.VulkanEngine.Renderer;

/// <summary>
/// Owns the swap chain and its image views — the surface-extent-scoped presentation
/// resources. Depends only on <see cref="GraphicsDevice"/> (device, surface, queue
/// families) + the window. Rebuilt on resize via <see cref="Recreate"/>; the
/// orchestrator (Renderer) re-allocates the render targets that ride alongside it.
///
/// Acquire/present live here because they own the swap chain handle and the
/// <c>fixed</c> pointer dance that goes with it — keeping them off the frame body.
/// </summary>
public sealed unsafe class Swapchain : IDisposable
{
    private readonly GraphicsDevice _gfx;
    private readonly IWindow _window;
    private readonly Vk vk;

    private KhrSwapchain swapChainKhr;
    private SwapchainKHR swapChain;
    private Image[] swapChainImages = Array.Empty<Image>();
    private Format swapChainImageFormat;
    private Extent2D swapChainExtent;
    private ImageView[] swapChainImageViews = Array.Empty<ImageView>();

    public Swapchain(GraphicsDevice gfx, IWindow window)
    {
        _gfx = gfx;
        _window = window;
        vk = gfx.Vk;
    }

    public KhrSwapchain KhrSwapchain => swapChainKhr;
    public SwapchainKHR Handle       => swapChain;
    public Image[]      Images        => swapChainImages;
    public Format       ImageFormat   => swapChainImageFormat;
    public Extent2D     Extent        => swapChainExtent;
    public ImageView[]  ImageViews    => swapChainImageViews;

    public void Create()
    {
        //query swapchain support
        SwapChainSupportDetails swapChainSupport = _gfx.QuerySwapChainSupport(_gfx.PhysicalDevice);

        //choose swapsurface format
        SurfaceFormatKHR surfaceFormat = ChooseSwapSurfaceFormat(swapChainSupport.Formats);
        PresentModeKHR presentMode = ChooseSwapPresentMode(swapChainSupport.PresentModes);
        Extent2D extent = ChooseSwapExtent(swapChainSupport.Capabilities);

        //choose image count
        uint imageCount = swapChainSupport.Capabilities.MinImageCount + 1;
        if (swapChainSupport.Capabilities.MaxImageCount > 0 && imageCount > swapChainSupport.Capabilities.MaxImageCount)
        {
            imageCount = swapChainSupport.Capabilities.MaxImageCount;
        }

        var queueFamilyIndices = _gfx.QueueFamilyIndices;
        var surface = _gfx.Surface;
        var device = _gfx.Device;

        //create swapchain info
        SwapchainCreateInfoKHR createInfo = new()
        {
            SType = StructureType.SwapchainCreateInfoKhr,
            Surface = surface,
            MinImageCount = imageCount,
            ImageFormat = surfaceFormat.Format,
            ImageColorSpace = surfaceFormat.ColorSpace,
            ImageExtent = extent,
            ImageArrayLayers = 1,
            ImageUsage = ImageUsageFlags.ColorAttachmentBit | ImageUsageFlags.TransferDstBit,
            PreTransform = swapChainSupport.Capabilities.CurrentTransform,
            CompositeAlpha = CompositeAlphaFlagsKHR.OpaqueBitKhr,
            PresentMode = presentMode,
            Clipped = true,
            OldSwapchain = default
        };
        //set sharing mode
        uint?[] queueFamilyIndiciesLoc =
        {
            queueFamilyIndices.graphicsFamily,
            queueFamilyIndices.presentFamily
        };

        if (queueFamilyIndices.graphicsFamily != queueFamilyIndices.presentFamily)
        {
            createInfo.ImageSharingMode = SharingMode.Concurrent;
            createInfo.QueueFamilyIndexCount = (uint)queueFamilyIndiciesLoc.Length;
            fixed (uint?* pQueueFamilyIndicesLoc = &queueFamilyIndiciesLoc[0])
                createInfo.PQueueFamilyIndices = (uint*)pQueueFamilyIndicesLoc;
        }
        else
        {
            createInfo.ImageSharingMode = SharingMode.Exclusive;
            createInfo.QueueFamilyIndexCount = 0;
            createInfo.PQueueFamilyIndices = null;
        }

        //create swapchain
        if (!vk.TryGetDeviceExtension(_gfx.Instance, device, out swapChainKhr))
        {
            throw new Exception("Failed to load swapchain extension");
        }

        if (swapChainKhr.CreateSwapchain(device, &createInfo, null, out swapChain) != Result.Success)
        {
            throw new Exception("Failed to create swapchain");
        }

        //Get swapchain images
        swapChainKhr.GetSwapchainImages(device, swapChain, &imageCount, null);
        swapChainImages = new Image[imageCount];
        fixed (Image* imagesPtr = swapChainImages)
        {
            swapChainKhr.GetSwapchainImages(device, swapChain, &imageCount, imagesPtr);
        }

        //format and extent
        swapChainImageFormat = surfaceFormat.Format;
        swapChainExtent = extent;
    }

    public void CreateImageViews()
    {
        var device = _gfx.Device;
        swapChainImageViews = new ImageView[swapChainImages.Length];

        ImageViewCreateInfo viewInfo = new()
        {
            SType = StructureType.ImageViewCreateInfo,
            ViewType = ImageViewType.Type2D,
            Format = swapChainImageFormat,
            Components = new ComponentMapping()
            {
                R = ComponentSwizzle.Identity,
                G = ComponentSwizzle.Identity,
                B = ComponentSwizzle.Identity,
                A = ComponentSwizzle.Identity
            },
            SubresourceRange = new ImageSubresourceRange()
            {
                AspectMask = ImageAspectFlags.ColorBit,
                BaseMipLevel = 0,
                LevelCount = 1,
                BaseArrayLayer = 0,
                LayerCount = 1
            }
        };
        //create imageview for each image
        for (var i = 0; i < swapChainImages.Length; i ++)
        {
            viewInfo.Image = swapChainImages[i];
            vk.CreateImageView(device, &viewInfo, null, out swapChainImageViews[i]);
        }
    }

    // Tears down only the swap chain itself + its image views. Size-dependent
    // attachments (depth, g-buffers, FinalColor) are owned elsewhere and torn
    // down by the orchestrator's RecreateSwapChain / Cleanup.
    public void Cleanup()
    {
        var device = _gfx.Device;
        if (swapChainImageViews != null)
        {
            foreach (var iv in swapChainImageViews)
                if (iv.Handle != 0) vk.DestroyImageView(device, iv, null);
            swapChainImageViews = Array.Empty<ImageView>();
        }

        if (swapChain.Handle != 0)
        {
            swapChainKhr.DestroySwapchain(device, swapChain, null);
            swapChain = default;
        }
    }

    /// <summary>Cleanup + Create + CreateImageViews. The orchestrator calls this from
    /// RecreateSwapChain after draining the GPU, then re-allocates render targets.</summary>
    public void Recreate()
    {
        Cleanup();
        Create();
        CreateImageViews();
    }

    public void Dispose() => Cleanup();

    /// <summary>Acquires the next presentable image, signalling <paramref name="imageAvailable"/>.</summary>
    public Result AcquireNextImage(Semaphore imageAvailable, out uint imageIndex)
    {
        uint idx = 0;
        var result = swapChainKhr.AcquireNextImage(_gfx.Device, swapChain, ulong.MaxValue,
            imageAvailable, default, &idx);
        imageIndex = idx;
        return result;
    }

    /// <summary>Presents <paramref name="imageIndex"/> on <paramref name="presentQueue"/>,
    /// waiting on <paramref name="waitSem"/>.</summary>
    public Result Present(Queue presentQueue, Semaphore waitSem, uint imageIndex)
    {
        var sem = waitSem;
        var idx = imageIndex;
        fixed (SwapchainKHR* pSwap = &swapChain)
        {
            var presentInfo = new PresentInfoKHR
            {
                SType = StructureType.PresentInfoKhr,
                WaitSemaphoreCount = 1,
                PWaitSemaphores    = &sem,
                SwapchainCount     = 1,
                PSwapchains        = pSwap,
                PImageIndices      = &idx,
            };
            return swapChainKhr.QueuePresent(presentQueue, &presentInfo);
        }
    }

    private Extent2D ChooseSwapExtent(SurfaceCapabilitiesKHR capabilities)
    {
        if (capabilities.CurrentExtent.Width != uint.MaxValue)
            return capabilities.CurrentExtent;
        else
        {
            var framebufferSize = _window.FramebufferSize;
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