using CadThingo.Graphics.Assets3D.Geometry;
using CadThingo.GraphicsPipeline;
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

    public readonly string[] DeviceExtensions = 
    [
        KhrSwapchain.ExtensionName
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
    
    public KhrSwapchain? KhrSwapChain;
    public SwapchainKHR SwapChain;
    public Image[]? SwapChainImages;
    public Format SwapChainImageFormat;
    public Extent2D SwapChainExtent;
    public ImageView[]? SwapChainImageViews;
    public Framebuffer[]? SwapChainFramebuffers;
    
    public RenderPass RenderPass;
    public PipelineLayout PipelineLayout;
    public Pipeline Pipeline;

    public CommandPool CommandPool;
    
    public Buffer VertexBuffer;
    public DeviceMemory VertexBufferMemory;
    public Buffer IndexBuffer;
    public DeviceMemory IndexBufferMemory;
    
    public CommandBuffer[]? CommandBuffers;
    
    public Semaphore[]? ImageAvailableSemaphores;
    public Semaphore[]? RenderFinishedSemaphores;
    public Fence[]? InFlightFences;
    public Fence[]? ImagesInFlight;
    public uint CurrentFrame = 0;
    
    public VkVertex[] Vertices;
    public uint[] Indices;
}