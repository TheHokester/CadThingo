using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer;

/// <summary>
/// The immutable per-frame scene snapshot handed down the whole recording path: render cores,
/// their frame graphs, and every pass body inside them.
///
/// The draw loop builds exactly one of these per frame, AFTER it has run the frame's extraction
/// (world-transform cache open -> materials -> lights -> renderables). Everything a consumer needs
/// to know about "what is being drawn this frame" is a field here, so nothing downstream re-walks
/// the ECS or re-triggers an extract to find out. A core that wants the opaque count reads
/// <see cref="RenderableCount"/>; it does not call ExtractRenderables and get the count back.
///
/// Stable per-renderer resources (buffers, descriptor sets, pipelines) deliberately do NOT live
/// here - they are not frame state. Only what changes frame to frame belongs.
/// </summary>
public readonly ref struct RenderView
{
    /// <summary>Frame-in-flight slot index. Indexes every per-frame buffer / descriptor set.</summary>
    public uint FrameIndex { get; init; }
    
    public ulong FrameCounter { get; init; }

    /// <summary>Camera the frame is being recorded for. Null only in headless / pre-camera paths.</summary>
    public Camera Camera { get; init; }

    /// <summary>Authoring scene. Consumers that only need render data should read the counts below
    /// instead - this is here for the subsystems (probe capture) that still walk entities.</summary>
    public Scene Scene { get; init; }

    /// <summary>Size of the offscreen render target chain this frame. NOT the swapchain extent -
    /// the viewport panel drives it independently.</summary>
    public Extent2D RenderExtent { get; init; }

    // Extraction outputs. Produced once by the draw loop before the active core runs.

    /// <summary>OPAQUE/MASK renderables packed into GpuScene's cull-input SSBO this frame. Drives
    /// the cull dispatch size and maxDrawCount.</summary>
    public uint RenderableCount { get; init; }

    /// <summary>Lights packed into GpuScene's light SSBO this frame.</summary>
    public uint LightCount { get; init; }

    /// <summary>Materials packed into the material SSBO this frame, excluding the fallback slot
    /// that sits at [MaterialCount].</summary>
    public uint MaterialCount { get; init; }

    /// <summary>BLEND-mode renderables extracted this frame, unsorted. The back-to-front sort is
    /// view-dependent, so it stays downstream in the cull pass which owns the camera.</summary>
    public IReadOnlyList<TransparentDraw> TransparentCandidates { get; init; }
}