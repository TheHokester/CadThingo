using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.RenderCores;

/// <summary>
/// The per-frame context handed to an <see cref="IRenderCore"/> (renderer-refactor.md L3).
/// A thin composition of the recording command buffer + the host <see cref="Renderer.FrameContext"/>
/// (camera / frame index / scene / render extent). It folds into L2's RenderView later -- the
/// FrameGraph already accepts a FrameContext in Execute, so the seam is forward-compatible.
/// </summary>
internal readonly ref struct RenderFrame
{
    public CommandBuffer        Cmd   { get; init; }
    public Renderer.FrameContext Frame { get; init; }
}

/// <summary>
/// One pluggable rendering technique (renderer-refactor.md L3). The host (<see cref="Renderer"/>)
/// owns the shared resources + the frame skeleton (acquire -> extract -> [core] -> outline -> blit
/// -> present); a core owns exactly one technique and records it into <see cref="RenderFrame.Cmd"/>,
/// leaving FinalColor in <c>ShaderReadOnlyOptimal</c> -- the layout the host post-stack
/// (RecordSelectionOutline + swapchain blit + ImGui viewport sampler) already assumes from every
/// path today.
///
/// This replaces the four parallel <c>DrawX</c> methods + the <c>switch(renderMode)</c> in
/// <c>DrawFrame</c>: a mode change becomes a swap of the active core. Cores are built eagerly (all
/// technique pipelines are already constructed up front in <c>Renderer.Initialize</c>).
/// </summary>
internal interface IRenderCore : IDisposable
{
    /// <summary>Human-readable label (debug / DOT / stats panels).</summary>
    string Name { get; }

    /// <summary>The <see cref="Renderer.RenderMode"/> this core services.</summary>
    Renderer.RenderMode Mode { get; }

    /// <summary>
    /// Called once when this core becomes the active core (mode switch / first activation /
    /// after a tonemap or PBR pipeline rebuild). The host has already issued DeviceWaitIdle, so
    /// it is safe to re-point shared descriptor sets. Crucially the core rebinds tonemap's
    /// HDR-input source to its own scene-colour image here -- this is what replaces the per-frame
    /// <c>_lastRenderMode</c> check the old <c>DrawFrame</c> ran.
    /// </summary>
    void Activate();

    /// <summary>Records the full technique into <see cref="RenderFrame.Cmd"/>, leaving FinalColor
    /// in <c>ShaderReadOnlyOptimal</c> for the host post-stack.</summary>
    void Render(in RenderFrame frame);

    /// <summary>Rebuilds / rebinds any size-dependent technique state after a render-target
    /// resize (fresh extent -> fresh transients / storage images). The host has issued
    /// DeviceWaitIdle and reallocated the shared targets before calling this.</summary>
    void Resize(Extent2D extent);
}