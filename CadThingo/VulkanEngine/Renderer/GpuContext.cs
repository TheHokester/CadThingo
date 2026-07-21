using CadThingo.VulkanEngine.Renderer.Descriptors;
using CadThingo.VulkanEngine.Renderer.Shaders;

namespace CadThingo.VulkanEngine.Renderer;

/// <summary>The device-services channel : the three handles a
/// consumer needs for RHI verbs, bindable resources, and shaders. Deliberately capped at these
/// three and never grows - anything frame-scoped arrives at record time, constants are static,
/// and feature collaborators resolve by contract interface.</summary>
public readonly record struct GpuContext(
    GraphicsDevice Gfx,
    DescriptorRegistry Registry,
    ShaderLibrary Shaders);