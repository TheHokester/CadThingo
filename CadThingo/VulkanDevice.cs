using System.Runtime.CompilerServices;
using Silk.NET.Core;
using Silk.NET.Core.Native;
using Silk.NET.Vulkan;
using Silk.NET.Vulkan.Extensions.KHR;
namespace CadThingo;

public class VulkanDevice
{
    private static readonly Vk vk = Globals.vk;
    private VulkanContext ctx;

    public VulkanDevice(VulkanContext context)
    {
        ctx = context;
    }

    public unsafe void PickPhysicalDevice()
    {
        uint deviceCount = 0;
        vk.EnumeratePhysicalDevices(ctx.Instance, &deviceCount, null);

        if (deviceCount == 0)
            throw new Exception("No Vulkan devices found");

        var devices = stackalloc PhysicalDevice[(int)deviceCount];
        vk.EnumeratePhysicalDevices(ctx.Instance, &deviceCount, devices);

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
        bool IsDeviceSuitable(PhysicalDevice device)
        {
            var indices = FindQueueFamilies(device);
            return indices.IsComplete();
        }
        
    }
    
    
    public unsafe void CreateLogicalDevice()
    {
        var indices = FindQueueFamilies(ctx.PhysicalDevice);

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
            EnabledExtensionCount = 0
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
        
        if (vk.CreateDevice(ctx.PhysicalDevice, &deviceCreateInfo, null, out ctx.Device) != Result.Success)
            throw new Exception("Failed to create logical device");
        vk.GetDeviceQueue(ctx.Device, indices.graphicsFamily!.Value, 0, out ctx.GraphicsQueue);
        vk.GetDeviceQueue(ctx.Device, indices.presentFamily!.Value, 0, out ctx.PresentQueue);
    }

    private struct QueueFamilyIndices
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
    private unsafe QueueFamilyIndices FindQueueFamilies(PhysicalDevice device) 
    {
        var indices = new QueueFamilyIndices();
        
        uint count = 0;
        vk.GetPhysicalDeviceQueueFamilyProperties(device, &count, null);
        var queueFamilies = new QueueFamilyProperties[(int)count];
        fixed (QueueFamilyProperties* pQueueFamilies = queueFamilies)
        {
            vk.GetPhysicalDeviceQueueFamilyProperties(device, &count, pQueueFamilies);
        }
        
        
        
        for (uint i = 0; i < count; i++)
        {
            if (queueFamilies[i].QueueFlags.HasFlag(QueueFlags.GraphicsBit))
            {
                indices.graphicsFamily = i;
            }

            ctx.KhrSurface!.GetPhysicalDeviceSurfaceSupport(device, i, ctx.Surface, out var presentSupport);
            
            if (presentSupport)
                indices.presentFamily = i;
            
            if (indices.IsComplete())
                break;
            
        }

        return indices;
    }
}