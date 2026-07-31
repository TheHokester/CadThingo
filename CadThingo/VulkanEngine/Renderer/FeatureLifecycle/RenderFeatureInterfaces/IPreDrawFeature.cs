namespace CadThingo.VulkanEngine.Renderer.FeatureLifecycle.RenderFeatureInterfaces;

/// <summary>
/// A feature doing out-of-band work before the frame's command buffer is recorded - work that
/// submits on its own (selection's pick ray) or that stages CPU data the active core will read.
/// Runs after extraction, so the RenderView it is handed is this frame's.
/// </summary>
public interface IPreDrawFeature : IRenderFeature
{
    void PreDraw(in RenderView view);
}
