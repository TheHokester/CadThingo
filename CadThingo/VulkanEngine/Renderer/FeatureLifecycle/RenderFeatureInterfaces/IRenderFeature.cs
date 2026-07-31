namespace CadThingo.VulkanEngine.Renderer.FeatureLifecycle.RenderFeatureInterfaces;

/// <summary>
/// A renderer subsystem with a host-managed lifecycle. The <see cref="FeatureHost"/> constructs it,
/// wires its dependencies, initializes it in order, pumps whichever phase interfaces it also
/// implements, and disposes it. No edit to Renderer, DrawFrame or Cleanup.
/// </summary>
public interface IRenderFeature : IDisposable
{
    /// <summary>Human-readable label. Used by the host's boot manifest and the debug panels.</summary>
    string Name { get; }

    /// <summary>
    /// Where a feature builds everything it owns. Runs after EVERY gated feature has been
    /// constructed and wired, so by now <c>Gpu</c> and every <c>INeedsFeature&lt;T&gt;</c>
    /// collaborator are set - which is why this takes no arguments. Constructors should do nothing
    /// but exist: at construction time the injected fields are still null, and it is the descriptor
    /// Order, not construction order, that guarantees a collaborator initialized before this one.
    /// </summary>
    void Initialize();
}

/// <summary>
/// A feature that describes its own registration. <c>static abstract</c> is what makes this a real
/// contract: the compiler refuses to build an implementer that does not supply a
/// <see cref="Desc"/>, so a feature cannot exist without declaring its order, its device gate and
/// how to construct it.
///
/// It has to be static ABSTRACT and generic over the implementer rather than a plain static member
/// on <see cref="IRenderFeature"/>. A plain static interface member is one storage slot shared by
/// every implementer -- six cores would overwrite each other's descriptor. <c>static abstract</c>
/// resolves per implementing type, so <c>DeferredCore.Desc</c> and <c>WavefrontPTCore.Desc</c> are
/// genuinely different values.
///
/// The matching half is the one-line module initializer in each feature's file:
/// <code>
/// [ModuleInitializer]
/// internal static void _Reg() => FeatureCatalog.Register&lt;DeferredCore&gt;();
/// </code>
/// That line cannot silently drift -- <c>Register&lt;T&gt;</c> only accepts a type that satisfies
/// this interface, so it will not compile without a <see cref="Desc"/> to register. It is per-file
/// rather than inherited because [ModuleInitializer] runs once per ASSEMBLY, not once per type;
/// there is no per-type startup hook in .NET short of an assembly-wide reflection scan.
/// </summary>
/// <typeparam name="TSelf">The implementing type itself (curiously-recurring, so
/// <c>Register&lt;T&gt;</c> can read the right type's descriptor).</typeparam>
public interface ISelfRegisteringFeature<TSelf> : IRenderFeature
    where TSelf : ISelfRegisteringFeature<TSelf>
{
    /// <summary>This feature's registration: construction order, device gate, and factory.</summary>
    static abstract FeatureDesc Desc { get; }
}
