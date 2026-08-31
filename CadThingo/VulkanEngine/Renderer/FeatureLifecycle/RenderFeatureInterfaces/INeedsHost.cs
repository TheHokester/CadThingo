namespace CadThingo.VulkanEngine.Renderer.FeatureLifecycle.RenderFeatureInterfaces;

/// <summary>
/// TRANSITIONAL. A feature that has not yet been unpicked from the <see cref="Renderer"/> and still
/// reaches shared pipelines / render targets through it.
///
/// This exists so the debt is visible and finite: it rides the same wiring pass as
/// <see cref="INeedsGpu"/>, but every feature still carrying it says so in its declaration list, so
/// "what is left to decouple" is a search for this interface rather than an audit of constructor
/// bodies. When the last implementer drops it, the interface is deleted and the compiler proves
/// nothing reaches the host any more.
///
/// Do not add new uses. A collaborator belongs behind <c>INeedsFeature&lt;T&gt;</c>; a bindable
/// resource belongs in the descriptor registry; a constant belongs in RenderConfig; per-frame state
/// arrives in the RenderView at record time.
/// </summary>
public interface INeedsHost
{
    Renderer Host { set; }
}
