using Silk.NET.Core;
using Silk.NET.Core.Native;
using Silk.NET.Vulkan;
using Silk.NET.Windowing;
using Silk.NET.Vulkan.Extensions.EXT;
using Silk.NET.Vulkan.Extensions.KHR;

namespace CadThingo;

public unsafe class VulkanInstance
{
    private static Vk? _vk = Globals.vk;
    private static readonly IWindow? Window = App.window;
    
    
    private VulkanContext ctx; 
    public VulkanInstance(VulkanContext context)
    {
        ctx = context;
    }

    
    public unsafe void CreateInstance(out bool enableValidation) 
    {
        
        // Check validation FIRST
        enableValidation = CheckValidationLayer();

        // --- extensions ---
        
        
        
        

        var appNamePtr = SilkMarshal.StringToPtr("App");
        var engineNamePtr = SilkMarshal.StringToPtr("No Engine");
        
        var appInfo = new ApplicationInfo
        {
            SType = StructureType.ApplicationInfo,
            PApplicationName = (byte*)appNamePtr,
            ApplicationVersion = new Version32(1, 0, 0),
            ApiVersion = Vk.Version12,
            EngineVersion = new Version32(1, 0, 0),
            PEngineName = (byte*)engineNamePtr
        };

        var createInfo = new InstanceCreateInfo
        {
            SType = StructureType.InstanceCreateInfo,
            PApplicationInfo = &appInfo,
        };
        
        var extensions = GetRequiredExtensions();
        createInfo.EnabledExtensionCount = (uint)extensions.Length;
        createInfo.PpEnabledExtensionNames = (byte**)SilkMarshal.StringArrayToPtr(extensions);

        if (enableValidation)
        {
            createInfo.EnabledLayerCount = 1;
            createInfo.PpEnabledLayerNames = (byte**)SilkMarshal.StringArrayToPtr(ctx.ValidationLayers);
        }
        else
        {
            createInfo.EnabledLayerCount = 0;
            createInfo.PNext = null;
        }
        

        if (_vk!.CreateInstance(&createInfo, null, out ctx.Instance) != Result.Success)
            throw new Exception("Failed to create Vulkan instance");

        // free memory
        SilkMarshal.Free(engineNamePtr);
        SilkMarshal.Free(appNamePtr);
        SilkMarshal.Free((nint)createInfo.PpEnabledExtensionNames);
        if(enableValidation)
            SilkMarshal.Free((nint)createInfo.PpEnabledLayerNames);
    }

    public void SetupDebugMessenger(bool enableValidation)
    {
         if(!enableValidation) return;

        if (!_vk!.TryGetInstanceExtension(ctx.Instance, out ctx.DebugUtils)) return;
        // ✅ CREATE DEBUG MESSENGER HERE
        var createInfo = new DebugUtilsMessengerCreateInfoEXT
        {
            SType = StructureType.DebugUtilsMessengerCreateInfoExt,
            MessageSeverity = DebugUtilsMessageSeverityFlagsEXT.ErrorBitExt |
                              DebugUtilsMessageSeverityFlagsEXT.WarningBitExt,
            MessageType = DebugUtilsMessageTypeFlagsEXT.GeneralBitExt |
                          DebugUtilsMessageTypeFlagsEXT.ValidationBitExt |
                          DebugUtilsMessageTypeFlagsEXT.PerformanceBitExt,
            PfnUserCallback = (PfnDebugUtilsMessengerCallbackEXT)DebugCallBack
        };

        if (ctx.DebugUtils!.CreateDebugUtilsMessenger(ctx.Instance, &createInfo, null, out ctx.DebugMessenger) != Result.Success)
            throw new Exception("Failed to create debug messenger");
    }

    public unsafe void CreateSurface()
    {
        if(!_vk!.TryGetInstanceExtension<KhrSurface>(ctx.Instance, out ctx.KhrSurface))
            throw new Exception("KHR Surface ext not found");
        ctx.Surface = Window!.VkSurface!.Create<AllocationCallbacks>(ctx.Instance.ToHandle(), null).ToSurface();
    }
    private static unsafe bool CheckValidationLayer()
    {
        uint layerCount = 0;
        _vk!.EnumerateInstanceLayerProperties(&layerCount, null);

        var layers = stackalloc LayerProperties[(int)layerCount];
        _vk!.EnumerateInstanceLayerProperties(&layerCount, layers);

        for (int i = 0; i < layerCount; i++)
        {
            var name = SilkMarshal.PtrToString((nint)layers[i].LayerName);
            if (name == "VK_LAYER_KHRONOS_validation")
            {
                return true;
            }
        }
        return false;
    }
    
    private static unsafe uint DebugCallBack(
        DebugUtilsMessageSeverityFlagsEXT severity,
        DebugUtilsMessageTypeFlagsEXT types,
        DebugUtilsMessengerCallbackDataEXT* data,
        void* userData)
    {
        var message = SilkMarshal.PtrToString((nint)data->PMessage);
        Console.WriteLine($"[VULKAN] {message}");
        return Vk.False;
    }
   
    private string[] GetRequiredExtensions()
    {
        
        var glfwExtensions = Window!.VkSurface!.GetRequiredExtensions(out var glfwExtensionCount);
        var extensions = SilkMarshal.PtrToStringArray((nint)glfwExtensions, (int)glfwExtensionCount);

        if (ctx.EnableValidation)
            return extensions.Append(ExtDebugUtils.ExtensionName).ToArray();

        return extensions;
    }
   
    
    
}