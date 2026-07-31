using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer;



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