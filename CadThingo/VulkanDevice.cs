using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Silk.NET.Core;
using Silk.NET.Core.Native;
using Silk.NET.Vulkan;
using Silk.NET.Vulkan.Extensions.KHR;
namespace CadThingo;

public class VulkanDevice
{
    private static readonly Vk? vk = Globals.vk;
    private VulkanContext ctx;

    public VulkanDevice(VulkanContext context)
    {
        ctx = context;
    }

    public unsafe void PickPhysicalDevice()
    {
        uint deviceCount = 0;
        vk!.EnumeratePhysicalDevices(ctx.Instance, &deviceCount, null);

        if (deviceCount == 0)
            throw new Exception("No Vulkan devices found");

        var devices = stackalloc PhysicalDevice[(int)deviceCount];
        vk!.EnumeratePhysicalDevices(ctx.Instance, &deviceCount, devices);

        for (int i = 0; i < deviceCount; i++)
        {
            var device = devices[i];
            if (IsDeviceSuitable(device))
            {
                ctx.PhysicalDevice = device;
                break;
            }
        }

        if (ctx.PhysicalDevice.Handle == 0)
            throw new Exception("No suitable GPU found"); // throw new Exception("No suitable Vulkan device found");
        vk!.GetPhysicalDeviceProperties(ctx.PhysicalDevice, out var deviceProperties);
        string gpuName = Marshal.PtrToStringAnsi((IntPtr)deviceProperties.DeviceName)!;
        Console.WriteLine("device:" + gpuName );
    }
    private bool IsDeviceSuitable(PhysicalDevice device)
    {
        var indices = FindQueueFamilies(device, ctx.Surface, ctx.KhrSurface);
        var extensionsSupported = CheckDeviceExtensionSupport(device);
        
        var swapChainAdequate = false;
        if (extensionsSupported)
        {
            VulkanSwapchain.SwapChainSupportDetails SwapChainSupport = VulkanSwapchain.QuerySwapChainSupport(device, ctx.Surface, ctx.KhrSurface);
            swapChainAdequate = SwapChainSupport.Formats.Length != 0 && SwapChainSupport.PresentModes.Length != 0;
        }
        return indices.IsComplete() && extensionsSupported && swapChainAdequate;
    }
    
    private unsafe bool CheckDeviceExtensionSupport(PhysicalDevice device)
    {
        uint extensionCount = 0;
        vk!.EnumerateDeviceExtensionProperties(device, (byte*)null, &extensionCount, null);
        var availableExtensions = stackalloc ExtensionProperties[(int)extensionCount];
        vk!.EnumerateDeviceExtensionProperties(device, (byte*)null, &extensionCount, availableExtensions);
        
        HashSet<string> requiredExtensions = new(ctx.DeviceExtensions);

        for (var i = 0; i < extensionCount; i++)
        {
            requiredExtensions.Remove(SilkMarshal.PtrToString((nint)availableExtensions[i].ExtensionName));
        }
        return requiredExtensions.Count == 0;
    }


    public unsafe void CreateLogicalDevice()
    {
        var indices = FindQueueFamilies(ctx.PhysicalDevice, ctx.Surface, ctx.KhrSurface);

        var uniqueQueueFamilies = new[] { indices.graphicsFamily!.Value, indices.presentFamily!.Value };
        uniqueQueueFamilies = uniqueQueueFamilies.Distinct().ToArray();

        using var mem = GlobalMemory.Allocate(uniqueQueueFamilies.Length * sizeof(DeviceQueueCreateInfo));
        var queueCreateInfos = (DeviceQueueCreateInfo*)Unsafe.AsPointer(ref mem.GetPinnableReference());
        float priority = 1.0f;
        for (var i = 0; i < uniqueQueueFamilies.Length; i++){
            queueCreateInfos[i] = new()
            {
                SType = StructureType.DeviceQueueCreateInfo,
                QueueFamilyIndex = uniqueQueueFamilies[i],
                QueueCount = 1,
                PQueuePriorities = &priority
            };
        }

        PhysicalDeviceFeatures features = new();
        
        
        var deviceCreateInfo = new DeviceCreateInfo
        {
            SType = StructureType.DeviceCreateInfo,
            QueueCreateInfoCount = 1,
            PQueueCreateInfos = queueCreateInfos,
            
            PEnabledFeatures = &features,
            
            EnabledExtensionCount = (uint)ctx.DeviceExtensions.Length,
            PpEnabledExtensionNames = (byte**)SilkMarshal.StringArrayToPtr(ctx.DeviceExtensions)
        };
        if (ctx.EnableValidation)
        {
            deviceCreateInfo.EnabledLayerCount = (uint)ctx.ValidationLayers.Length;
            deviceCreateInfo.PpEnabledLayerNames = (byte**)SilkMarshal.StringArrayToPtr(ctx.ValidationLayers);
        }
        else
        {
            deviceCreateInfo.EnabledLayerCount = 0;
        }
        
        if (vk!.CreateDevice(ctx.PhysicalDevice, &deviceCreateInfo, null, out ctx.Device) != Result.Success)
            throw new Exception("Failed to create logical device");
        vk!.GetDeviceQueue(ctx.Device, indices.graphicsFamily!.Value, 0, out ctx.GraphicsQueue);
        vk!.GetDeviceQueue(ctx.Device, indices.presentFamily!.Value, 0, out ctx.PresentQueue);
    }

    public struct QueueFamilyIndices
    {
        public QueueFamilyIndices()
        {
        }

        public uint? graphicsFamily { get; set; }
        public uint? presentFamily { get; set; }
        
        public bool IsComplete()
        {
            return graphicsFamily.HasValue && presentFamily.HasValue;
        }
    }

    public static unsafe QueueFamilyIndices FindQueueFamilies(PhysicalDevice device,SurfaceKHR surface ,KhrSurface? khrSurface = null) 
    {
        var indices = new QueueFamilyIndices();
        
        uint count = 0;
        vk!.GetPhysicalDeviceQueueFamilyProperties(device, &count, null);
        var queueFamilies = new QueueFamilyProperties[(int)count];
        fixed (QueueFamilyProperties* pQueueFamilies = queueFamilies)
        {
            vk!.GetPhysicalDeviceQueueFamilyProperties(device, &count, pQueueFamilies);
        }
        
        
        
        for (uint i = 0; i < count; i++)
        {
            if (queueFamilies[i].QueueFlags.HasFlag(QueueFlags.GraphicsBit))
            {
                indices.graphicsFamily = i;
            }

            khrSurface!.GetPhysicalDeviceSurfaceSupport(device, i, surface, out var presentSupport);
            
            if (presentSupport)
                indices.presentFamily = i;
            
            if (indices.IsComplete())
                break;
            
        }

        return indices;
    }
}