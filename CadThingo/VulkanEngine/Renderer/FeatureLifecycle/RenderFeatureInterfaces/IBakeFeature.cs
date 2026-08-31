namespace CadThingo.VulkanEngine.Renderer.FeatureLifecycle.RenderFeatureInterfaces;

/// <summary>
/// A feature with view-independent GPU content that is computed once and reused across frames
/// (IBL irradiance/prefilter, reflection probes, the scene acceleration structure).
///
/// Bake is request-driven, not per-frame: the host services pending bakes at the top of DrawFrame,
/// the same place the pipeline-rebuild and TLAS-dirty flags are consumed. Descriptor Order decides
/// bake order, which is what lets a prober bake against an already-baked IBL.
/// </summary>
public interface IBakeFeature : IRenderFeature
{
    /// <summary>True when this feature's baked content is stale and must be recomputed.</summary>
    bool BakePending { get; }

    /// <summary>Recomputes the baked content and clears <see cref="BakePending"/>.</summary>
    void Bake();
}
