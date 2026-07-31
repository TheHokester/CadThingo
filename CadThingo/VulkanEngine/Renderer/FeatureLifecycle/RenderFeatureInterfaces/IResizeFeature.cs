using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.FeatureLifecycle.RenderFeatureInterfaces;

/// <summary>
/// A feature holding state sized to the render extent (transient targets, storage images, SoA
/// working sets). The host reallocates the shared render targets and issues DeviceWaitIdle before
/// pumping this, so it is safe to destroy and rebuild GPU objects here.
/// </summary>
public interface IResizeFeature : IRenderFeature
{
    void Resize(Extent2D extent);
}
