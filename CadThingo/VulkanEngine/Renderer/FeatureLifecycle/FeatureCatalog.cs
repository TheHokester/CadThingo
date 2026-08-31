using CadThingo.VulkanEngine.Renderer.FeatureLifecycle.RenderFeatureInterfaces;

namespace CadThingo.VulkanEngine.Renderer.FeatureLifecycle;

/// <summary>Decides whether this device can run the feature at all. Returning false means the
/// feature is never constructed - not constructed-then-disabled - so its resources are never
/// allocated on hardware that cannot use them.</summary>
public delegate bool FeatureGate(GpuContext gpu);

/// <summary>Constructs the feature. Takes nothing on purpose: phase 1 runs when no other feature is
/// guaranteed to exist, and everything a feature needs arrives in the phase-2 wiring pass, so a
/// constructor has nothing useful to receive and nothing useful to do.</summary>
public delegate IRenderFeature FeatureMake();

/// <summary>
/// One feature's registration. <see cref="Order"/> drives construction, Initialize and phase-pump
/// order (and, for cores, their index in the mode combo); dispose runs in reverse.
/// </summary>
public readonly record struct FeatureDesc(uint Order, FeatureGate Gate, FeatureMake Make);

/// <summary>
/// The assembly-wide registry of feature descriptors, filled by each feature's module initializer
/// before Main runs. The <see cref="FeatureHost"/> is the only reader.
///
/// The tradeoff of scattering Order across feature files is losing the single readable boot-order
/// list. <see cref="FeatureHost.Dump"/> replaces it with a runtime manifest that also shows what
/// the gates excluded on this device - which the static list never could.
/// </summary>
public static class FeatureCatalog
{
    private static readonly List<FeatureDesc> _descriptors = [];

    public static IReadOnlyList<FeatureDesc> Descriptors => _descriptors;

    /// <summary>Registers <typeparamref name="T"/>'s own descriptor. The type constraint is the
    /// enforcement: a feature with no <c>Desc</c> cannot be passed here.</summary>
    public static void Register<T>() where T : ISelfRegisteringFeature<T> => _descriptors.Add(T.Desc);

    /// <summary>Escape hatch for a descriptor not owned by a single type. Prefer
    /// <see cref="Register{T}"/> - it is the one the compiler checks.</summary>
    public static void Add(in FeatureDesc desc) => _descriptors.Add(desc);
}
