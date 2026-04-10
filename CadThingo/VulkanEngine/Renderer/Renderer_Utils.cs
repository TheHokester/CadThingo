using Silk.NET.Core.Native;
using Silk.NET.Vulkan;
using Silk.NET.Vulkan.Extensions.EXT;
using Silk.NET.Windowing;

namespace CadThingo.VulkanEngine.Renderer;

public unsafe partial class Renderer
{
    /// <summary>
    /// Finds a memory type that matches the given properties.
    /// </summary>
    /// <param name="_vk">parse to static context</param>
    /// <param name="physDev">parse to static context</param>
    /// <param name="typeFilter">filter</param>
    /// <param name="props">propertyFlags of desired memory type</param>
    /// <returns>uint of the found memory type matching the properites and filter</returns>
    /// <exception cref="Exception"> Couldn't find a suitable memory type</exception>
    public static uint FindMemoryType(Vk _vk,
        PhysicalDevice physDev, uint typeFilter, MemoryPropertyFlags props)
    {
        _vk.GetPhysicalDeviceMemoryProperties(physDev, out var memProps);
        for (uint i = 0; i < memProps.MemoryTypeCount; i++)
            if ((typeFilter & (1u << (int)i)) != 0 &&
                (memProps.MemoryTypes[(int)i].PropertyFlags & props) == props)
                return i;
        throw new Exception("No suitable memory type found.");
    }

    private Format FindDepthFormat()
    {
        Format depthFormat =
            FindSupportedFormat(new[] { Format.D32Sfloat, Format.D32SfloatS8Uint, Format.D24UnormS8Uint },
                ImageTiling.Optimal, FormatFeatureFlags.DepthStencilAttachmentBit);
        
        if (depthFormat == Format.Undefined)
        {
            Console.Error.WriteLine("failed to find suitable depth format, falling back to d32sFloat");
            return Format.D32Sfloat;
        }
        return depthFormat;
    }
    /// <summary>
    /// Finds a format that supports the given features.
    /// </summary>
    /// <param name="formats"></param>
    /// <param name="tiling"></param>
    /// <param name="features"></param>
    /// <returns>first format the supports the given features</returns>
    private Format FindSupportedFormat(Format[] formats, ImageTiling tiling, FormatFeatureFlags features = FormatFeatureFlags.None)
    {
        foreach (var format in formats)
        {
            vk!.GetPhysicalDeviceFormatProperties(physicalDevice, format, out var props);
            if(tiling == ImageTiling.Linear && (props.LinearTilingFeatures & features) != 0)
                return format;
            else if(tiling == ImageTiling.Optimal &&(props.OptimalTilingFeatures & features) != 0)
                return format;
        }
        Console.Error.WriteLine("failed to find suitable format!");
        return Format.Undefined;
    }


    private string[] GetRequiredExtensions()
    {
        
        var glfwExtensions = window!.VkSurface!.GetRequiredExtensions(out var glfwExtensionCount);
        var extensions = SilkMarshal.PtrToStringArray((nint)glfwExtensions, (int)glfwExtensionCount);
        
        if (enableValidationLayers)
            return extensions.Append(ExtDebugUtils.ExtensionName).ToArray();

        return extensions;
    }

    private QueueFamilyIndices FindQueueFamilies(PhysicalDevice device)
    {
        QueueFamilyIndices indices = new();
        
        //Get queue families props
        
        uint count = 0;
        vk!.GetPhysicalDeviceQueueFamilyProperties(device, &count, null);
        var queueFamilies = new QueueFamilyProperties[(int)count];
        fixed (QueueFamilyProperties* pQueueFamilies = queueFamilies)
        {
            vk!.GetPhysicalDeviceQueueFamilyProperties(device, &count, pQueueFamilies);
        }

        for (uint i = 0; i < count; i++)
        {
            var qf = queueFamilies[i];
            //check for graphics support
            if (qf.QueueFlags.HasFlag(QueueFlags.GraphicsBit) && !indices.graphicsFamily.HasValue)
            {
                indices.graphicsFamily = i;
            }

            //check for present support
            khrSurface!.GetPhysicalDeviceSurfaceSupport(device, i, surface, out var presentSupport);
            if (presentSupport && !indices.presentFamily.HasValue)
                indices.presentFamily = i;

            //check for compute support
            if (qf.QueueFlags.HasFlag(QueueFlags.ComputeBit) && !indices.computeFamily.HasValue)
            {
                indices.computeFamily = i;
            }

            //find dedicated transfer queue
            if (qf.QueueFlags.HasFlag(QueueFlags.TransferBit) && !qf.QueueFlags.HasFlag(QueueFlags.GraphicsBit))
            {
                if (!indices.transferFamily.HasValue)
                    indices.transferFamily = i;
            }

            //if we have found all the required queue families
            if (indices.IsComplete() && indices.transferFamily.HasValue)
                break;
        }

        //fallback if no transfer family is found
        if (!indices.transferFamily.HasValue && indices.graphicsFamily.HasValue)
        {
            indices.transferFamily = indices.graphicsFamily;
        }
        return indices;
    }

    public unsafe SwapChainSupportDetails QuerySwapChainSupport(PhysicalDevice physicalDevice)
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

    private bool CheckDeviceExtensionSupport(PhysicalDevice device)
    {
        uint extensionCount = 0;
         vk!.EnumerateDeviceExtensionProperties(device, (byte*)null, &extensionCount, null);
        var availableExtensions = stackalloc ExtensionProperties[(int)extensionCount];
        vk!.EnumerateDeviceExtensionProperties(device, (byte*)null, &extensionCount, availableExtensions);
        
        HashSet<string> requiredExtensions = new(deviceExtensions);
        for (var i = 0; i < extensionCount; i++)
        {
            requiredExtensions.Remove(SilkMarshal.PtrToString((nint)availableExtensions[i].ExtensionName));
        }
        return requiredExtensions.Count == 0;
    }

    private bool IsDeviceSuitable(PhysicalDevice device)
    {
        return false;
    }

    private ShaderModule CreateShaderModule(byte[] shaderCode)
    {
        //Create shader module
        ShaderModuleCreateInfo createInfo = new()
        {
            SType = StructureType.ShaderModuleCreateInfo,
            CodeSize = (nuint)shaderCode.Length,
        };
        fixed (byte* pCode = shaderCode)
        {
            createInfo.PCode = (uint*)pCode;
        }

        if (vk!.CreateShaderModule(device, &createInfo, null, out var shaderModule) != Result.Success)
        {
            throw new Exception("Failed to create shader module");
        }
        return shaderModule;
    }
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
}
public struct QueueFamilyIndices
{
    public QueueFamilyIndices()
    {
    }
    public uint? graphicsFamily { get; set; }
    public uint? presentFamily { get; set; }
    public uint? computeFamily { get; set; }
    public uint? transferFamily { get; set; }//optional
    public bool IsComplete()
    {
        return graphicsFamily.HasValue && presentFamily.HasValue && computeFamily.HasValue;
    }
}
public struct SwapChainSupportDetails
{
    public SurfaceCapabilitiesKHR Capabilities;
    public SurfaceFormatKHR[] Formats;
    public PresentModeKHR[] PresentModes;

    
}