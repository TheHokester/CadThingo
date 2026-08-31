using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.FeatureLifecycle.RenderFeatureInterfaces;

/// <summary>
/// A feature holding state sized to the render extent (transient targets, storage images, SoA
/// working sets). The host reallocates the shared render targets and issues DeviceWaitIdle before
/// pumping this, so it is safe to destroy and rebuild GPU objects here.
///
/// This runs once at boot too, straight after BuildAll's Initialize pass. So Initialize builds only
/// what is extent-independent, and everything sized to the render target is built HERE - one
/// allocation site per feature rather than an init copy and a resize copy that drift apart.
/// </summary>
public interface IResizeFeature : IRenderFeature
{
    void Resize(in HostTargets targets);
}
