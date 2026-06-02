using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer;

public unsafe partial class Renderer
{
    // Device-level queries / shader-module creation moved to GraphicsDevice (L1).
    // These forwarders keep the former Renderer.* call sites (pipelines, swapchain
    // creation, IBL bake) compiling while the deeper repoints land.
    internal Format FindDepthFormat() => gfx.FindDepthFormat();

    internal ShaderModule CreateShaderModule(byte[] shaderCode) => gfx.CreateShaderModule(shaderCode);

    public SwapChainSupportDetails QuerySwapChainSupport(PhysicalDevice physicalDevice)
        => gfx.QuerySwapChainSupport(physicalDevice);
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