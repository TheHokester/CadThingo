using CadThingo.VulkanEngine.Renderer.RenderCores;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Features.Forward;

/// <summary>
/// Placeholder core for the forward+ technique (RenderMode.ForwardPlus). The technique is not
/// implemented yet -- this preserves the prior behaviour of the empty <c>DrawRayQueried</c> (a
/// no-op frame: FinalColor keeps whatever the last active core left it as, in ShaderReadOnly, so
/// the host post-stack stays valid). It exists so the enum -> core mapping is total without
/// inventing behaviour; the real forward+ chain lands here later as its own module/graph.
/// </summary>
internal sealed class ForwardPlusCore : IRenderCore
{
    private readonly Renderer _host;

    public ForwardPlusCore(Renderer host)
    {
        _host = host;
        host.RegisterCore(this);   // cores add themselves to the host's render-core registry
    }

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