using System.Runtime.CompilerServices;
using CadThingo.VulkanEngine.Renderer.FeatureLifecycle;
using CadThingo.VulkanEngine.Renderer.FeatureLifecycle.RenderFeatureInterfaces;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Features.Forward;

/// <summary>
/// Placeholder core for the forward+ technique (RenderMode.ForwardPlus). The technique is not
/// implemented yet -- this preserves the prior behaviour of the empty <c>DrawRayQueried</c> (a
/// no-op frame: FinalColor keeps whatever the last active core left it as, in ShaderReadOnly, so
/// the host post-stack stays valid). It exists so the enum -> core mapping is total without
/// inventing behaviour; the real forward+ chain lands here later as its own module/graph.
/// </summary>
internal sealed class ForwardPlusCore : IRenderCore, ISelfRegisteringFeature<ForwardPlusCore>
{
    public static FeatureDesc Desc =>
        new(Order: 40, Gate: _ => true, Make: () => new ForwardPlusCore());

    [ModuleInitializer]
    internal static void _Reg() => FeatureCatalog.Register<ForwardPlusCore>();

    // Nothing to build - the technique is a stub. Kept explicit rather than defaulted so the
    // no-op is a decision rather than an omission.
    public void Initialize() { }

    public string Name => "Forward+ (stub)";
    public Renderer.RenderMode Mode => Renderer.RenderMode.ForwardPlus;

    // No scene-colour image of its own yet, so leave tonemap pointed wherever the previous core
    // bound it (a valid image either way) -- nothing to rebind.
    public void Activate() { }

    // No-op frame (matches the former empty DrawRayQueried). FinalColor is untouched.
    public void Render(in RenderFrame frame) { }

    public void Resize(Extent2D extent) { }

    public void Dispose() { }
}