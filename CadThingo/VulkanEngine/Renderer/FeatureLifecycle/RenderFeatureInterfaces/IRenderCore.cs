using CadThingo.VulkanEngine.Renderer.FrameGraph;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.FeatureLifecycle.RenderFeatureInterfaces;

/// <summary>
/// Optional capability for cores that drive a <see cref="FrameGraph.FrameGraph"/> (deferred,
/// wavefront PT). The host's render-graph debug surface (Stats panel) queries the ACTIVE core
/// through this so the panel reflects whatever technique is running, not a hard-wired one. Cores
/// without a graph (the megakernel PT cores, forward+) simply don't implement it -> the host's
/// `_activeCore as IGraphCore` is null and the panel shows nothing.
/// </summary>
internal interface IGraphCore
{
    /// <summary>Last-frame per-pass GPU/CPU timings + counts, or null before first compile.</summary>
    GraphStats? GraphStats { get; }
    /// <summary>Graphviz dump of the compiled graph (for "Copy DOT").</summary>
    string ToDot();
    /// <summary>Runtime toggle for the graph's pipeline-statistics collection.</summary>
    bool CollectPipelineStats { get; set; }
    /// <summary>True when the active graph has async-compute passes and Execute deferred its gfx
    /// chunks. The host must call <see cref="SubmitGfxChunks"/> instead of its own QueueSubmit.</summary>
    bool HasPendingGfxChunks { get; }
    /// <summary>Submit the deferred graphics chunks plus <paramref name="hostCmd"/> in one
    /// vkQueueSubmit2, with the frame-pacing binary semaphores and fence attached to the host
    /// cmd's slot. Only valid when <see cref="HasPendingGfxChunks"/> is true.</summary>
    void SubmitGfxChunks(Queue gfxQueue, SemaphoreSubmitInfo imgAvailWait,
        SemaphoreSubmitInfo renderDoneSignal, CommandBuffer hostCmd, Fence fence);
}

/// <summary>
/// The per-frame context handed to an <see cref="IRenderCore"/>: the recording command buffer plus
/// the host's <see cref="RenderView"/> snapshot (camera / frame index / render extent / the counts
/// the draw loop's extraction produced). A core reads the view; it never re-derives what is in it.
/// </summary>
internal readonly ref struct RenderFrame
{
    public CommandBuffer Cmd  { get; init; }
    public RenderView    View { get; init; }
}

/// <summary>
/// One pluggable rendering technique (renderer-refactor.md L3). The host (<see cref="Renderer"/>)
/// owns the shared resources + the frame skeleton (acquire -> extract -> [core] -> outline -> blit
/// -> present); a core owns exactly one technique and records it into <see cref="RenderFrame.Cmd"/>,
/// leaving FinalColor in <c>ShaderReadOnlyOptimal</c> -- the layout the host post-stack
/// (SelectionSystem.RecordOutline + swapchain blit + ImGui viewport sampler) already assumes from every
/// path today.
///
/// This replaces the four parallel <c>DrawX</c> methods + the <c>switch(renderMode)</c> in
/// <c>DrawFrame</c>: a mode change becomes a swap of the active core.
///
/// A core is a FEATURE with two extra properties: exactly one is active at a time, and the active
/// one produces the frame. Everything else - registration, gating, ordered construction, wiring,
/// Initialize, resize, reverse dispose - it inherits from <see cref="IResizeFeature"/> rather than
/// getting its own parallel machinery. Its descriptor Order doubles as its index in the mode combo,
/// so the lowest-Order core is the boot default.
/// </summary>
internal interface IRenderCore : IResizeFeature
{
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
}