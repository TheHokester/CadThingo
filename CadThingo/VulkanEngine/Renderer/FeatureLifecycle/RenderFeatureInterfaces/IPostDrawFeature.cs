using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.FeatureLifecycle.RenderFeatureInterfaces;

/// <summary>
/// A feature recording into the frame's command buffer AFTER the active core has produced
/// FinalColor and before the swapchain blit (the selection outline composite). FinalColor is in
/// ShaderReadOnlyOptimal on entry and must be left that way.
/// </summary>
public interface IPostDrawFeature : IRenderFeature
{
    void PostDraw(CommandBuffer cmd, in RenderView view);
}
