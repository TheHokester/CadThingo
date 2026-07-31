using CadThingo.VulkanEngine.Renderer.Features.SceneAcceleration;

namespace CadThingo.VulkanEngine.Renderer;

// What is left of the old Renderer_Ray_Query partial. The acceleration structure - clustering,
// rebuild cadence, instance / shadow-info / emissive packing, and the buffers behind them - now
// belongs to the gated SceneAS feature, and the Vulkan verbs it drives belong to gfx.As. This file
// is only the bridge: the editor panels and the path-trace pipelines still name the Renderer, so
// their calls land here and forward.
//
// Every member is null-tolerant on purpose. SceneAS is gated out on a device without ray query, and
// that is not an error condition - it is a scene with no ray infrastructure, where a pick finds
// nothing and a path tracer sees no emissive triangles. Each of these deletes itself when its last
// caller stops naming the Renderer.
public unsafe partial class Renderer
{
    // Assigned right after FeatureHost.BuildAll. Null when the device gated the feature out.
    private SceneAS? _sceneAs;

    /// <summary>Flags the acceleration structure as stale. The rebuild is serviced once per frame
    /// by the bake pump, however many edits accumulated - which is why the editor calls this from
    /// slider callbacks rather than rebuilding per tick.</summary>
    public void MarkTlasDirty() => _sceneAs?.MarkDirty();

    /// <summary>True while a rebuild is owed.</summary>
    public bool IsTlasDirty => _sceneAs?.BakePending ?? false;

    /// <summary>
    /// Rebuilds the acceleration structure immediately rather than waiting for the bake pump.
    /// The asset paths (scene load, mesh destroy) use this because they want the new geometry
    /// traceable before they return.
    /// </summary>
    public void OnSceneEntitiesChanged()
    {
        if (!initialized) return;

        // With no AS on this device there is still an in-progress integration to invalidate:
        // changing the scene changes the image whether or not anything traces it.
        if (_sceneAs is null) { MarkAccumulatorDirty(); return; }

        _sceneAs.Rebuild();
    }

    /// <summary>
    /// Pairs with file destroy in the editor. Cluster BLASes are world-space and rebuilt wholesale
    /// on the next rebuild, so there is nothing mesh-keyed to free here — just mark the AS stale so
    /// the freed geometry is dropped on the next rebuild.
    /// </summary>
    public void DestroyBlasFor(IEnumerable<nint> meshPtrs)
    {
        _ = meshPtrs;
        MarkTlasDirty();
    }

    /// <summary>True when the full ray-query stack is usable this frame: the feature exists on this
    /// device and has a built TLAS. Selection gates on it before picking or outlining.</summary>
    internal bool RayInfraReady => _sceneAs?.Ready ?? false;

    /// <summary>Emissive area-light scalars the path tracers push as uniforms. Zero when there is
    /// no AS, which is exactly what makes the shaders skip their NEE loop.</summary>
    public uint  EmissiveTriangleCount => _sceneAs?.EmissiveTriangleCount ?? 0u;
    public float TotalEmissivePower    => _sceneAs?.TotalEmissivePower    ?? 0f;
}
