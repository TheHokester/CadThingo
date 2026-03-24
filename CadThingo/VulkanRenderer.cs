using Silk.NET.Windowing;
using Silk.NET.Vulkan;
namespace CadThingo;

public class VulkanRenderer
{
    private static IWindow window;
    
    public VulkanRenderer(IWindow _window)
        => window = _window;
    
    
    private static Vk vk;
    private VulkanContext ctx;
    private VulkanInstance VKInstance;
    private VulkanDevice VKDevice;
    
    
    
    public unsafe void OnLoad()
    {
        vk = Globals.vk;
        ctx = new VulkanContext();
        VKInstance = new VulkanInstance(ctx);
        VKDevice = new VulkanDevice(ctx);
        
        VKInstance.CreateInstance(out ctx.EnableValidation);
        VKInstance.SetupDebugMessenger(ctx.EnableValidation);
        // --- surface ---
        VKInstance.CreateSurface();
        // --- physical device ---
        VKDevice.PickPhysicalDevice();
        // --- logical device ---
        VKDevice.CreateLogicalDevice();

        Console.WriteLine("Vulkan initialized");
    }
    public void OnRender(double deltaTime)
    {
        
    }
    public unsafe void OnClose()
    {
        
        vk.DestroyDevice(ctx.Device, null);
        VKInstance.DestroyInstance();
        ctx.KhrSurface!.DestroySurface(ctx.Instance, ctx.Surface, null);
        vk.DestroyInstance(ctx.Instance, null);
        vk.Dispose();
        window.Dispose();
        window.Close();
    }
}
