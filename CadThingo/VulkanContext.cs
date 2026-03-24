using Silk.NET.Vulkan;
using Silk.NET.Vulkan.Extensions.KHR;
using Silk.NET.Windowing;
using Silk.NET.Vulkan.Extensions.EXT;

namespace CadThingo;

public unsafe class VulkanContext
{
    
    public bool EnableValidation = true;
    public readonly string[] ValidationLayers =
    [
        "VK_LAYER_KHRONOS_validation"
    ];
    
    public ExtDebugUtils? DebugUtils;
    public DebugUtilsMessengerEXT DebugMessenger;
    public SurfaceKHR Surface;
    public KhrSurface? KhrSurface;
    
    public Instance Instance;
    
    public PhysicalDevice PhysicalDevice;
    public Device Device;
    
    public Queue GraphicsQueue;
    public Queue PresentQueue;
    
    public KhrSwapchain? KhrSwapchain;
    public SwapchainKHR Swapchain;
    public Image[]? SwapchainImages;
    public Format swapchainImageFormat;
    public Extent2D swapchainExtent;
    
}