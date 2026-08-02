namespace CadThingo.VulkanEngine.Renderer.FeatureLifecycle.RenderFeatureInterfaces;

/// <summary>
/// A feature needing a collaborator, named by CONTRACT rather than by concrete type
/// (<c>INeedsFeature&lt;IIblProvider&gt;</c>, not <c>INeedsFeature&lt;IblSystem&gt;</c>). The host
/// resolves it from the built feature set during the wiring pass.
///
/// Implement it once per collaborator, with explicit-interface setters when there are several:
/// <code>
/// sealed class ReflectionProbeSystem : IRenderFeature, INeedsGpu, INeedsFeature&lt;IIblProvider&gt;
/// {
///     IIblProvider _ibl = null!;
///     IIblProvider INeedsFeature&lt;IIblProvider&gt;.Dependency { set =&gt; _ibl = value; }
/// }
/// </code>
///
/// What arrives is a REFERENCE, available from wiring onwards - not a guarantee that the
/// collaborator has initialized or baked. If you need it ready, use it in Initialize or Bake and
/// give yourself a higher Order; that ordering is the whole reason the descriptor carries one.
///
/// Only non-bindable data belongs here (scalars, queries). A collaborator's bindable GPU resources
/// go through the descriptor registry by name, so the consumer never names the producer at all.
/// </summary>
public interface INeedsFeature<in T> where T : class
{
    T Dependency { set; }
}

/// <summary>
/// The same wiring for a collaborator that may be gated out: it arrives as null and the build
/// carries on. Take it where the consumer has a defined answer for absence - a device with no
/// acceleration structure traces nothing, which is a valid frame.
///
/// A consumer that cannot cope with absence takes <see cref="INeedsFeature{T}"/> and fails at
/// wiring, or gates itself the same way its collaborator does.
/// </summary>
public interface INeedsOptionalFeature<in T> where T : class
{
    T? Dependency { set; }
}
