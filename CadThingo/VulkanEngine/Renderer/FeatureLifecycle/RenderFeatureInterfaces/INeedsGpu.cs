namespace CadThingo.VulkanEngine.Renderer.FeatureLifecycle.RenderFeatureInterfaces;

/// <summary>
/// A feature needing device services: RHI verbs, the descriptor registry, the shader library. Set
/// by the host's wiring pass before Initialize, so the setter should only stash it in a field.
///
/// Set-only by design. The context is the host's to hand out, not a property of the feature, and a
/// getter would let one feature pull it off another.
/// </summary>
public interface INeedsGpu
{
    GpuContext Gpu { set; }
}
