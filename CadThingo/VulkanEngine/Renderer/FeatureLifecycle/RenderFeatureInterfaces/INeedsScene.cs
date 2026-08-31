namespace CadThingo.VulkanEngine.Renderer.FeatureLifecycle.RenderFeatureInterfaces;

/// <summary>
/// A feature needing the renderer's view of the scene: the GPU mirror, the renderable identity map,
/// and the extract path that fills them. Set by the host's wiring pass before Initialize, so the
/// setter should only stash it in a field.
///
/// Infrastructure, not a collaborator - it is constructed by the host and handed down, so it takes a
/// direct channel rather than <see cref="INeedsFeature{T}"/>. It is also NOT frame state: the AS
/// rebuild and the deferred graph build both reach it with no <see cref="RenderView"/> in hand,
/// which is why it cannot ride the frame snapshot.
///
/// Set-only, for the same reason as <see cref="INeedsGpu"/>: a getter would let one feature pull the
/// scene off another.
/// </summary>
public interface INeedsScene
{
    GpuScene Scene { set; }
}