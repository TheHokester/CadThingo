using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer;

/// <summary>Host-owned render-target state, snapshotted per resize: the current extent and the
/// shared FinalColor. Cache the struct, refresh it on every resize, never cache the ImageResource
/// out of it.</summary>
public readonly record struct HostTargets(Extent2D Extent, ImageResource FinalColor);