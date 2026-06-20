using System.Numerics;
using CadThingo.VulkanEngine.Renderer.RenderCores;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Features.PathTracer;

/// <summary>
/// Shared base for the two progressive path-trace cores (renderer-refactor.md L3): the compute
/// (ray-query) path and the RT-pipeline path. The two former <c>DrawPathtraced</c> /
/// <c>DrawRayTraced</c> methods were ~80 near-identical lines that differed only in the technique
/// pipeline and the writer pipeline stage (compute vs ray-tracing) -- this base is the dedupe.
///
/// The render skeleton: camera-motion -> accumulator restart; per-frame light + material SSBO
/// refresh; bridge the host's accumulator-dirty flag into the pipeline's reset; UpdatePerFrame;
/// the two-image (accumulator + ptOutColor) pre-dispatch barrier; dispatch (Record); then tonemap
/// ptOutColor -> FinalColor with the General &lt;-&gt; ShaderReadOnly layout dance, leaving FinalColor
/// in <c>ShaderReadOnlyOptimal</c> for the host post-stack. The accumulator / ptOutColor images and
/// FinalColor are host-owned (RenderTargets); this core owns only the camera snapshot + the
/// orchestration. Subclasses bind the concrete pipeline via the protected hooks.
/// </summary>
internal abstract unsafe class PathTraceCoreBase : IRenderCore
{
    protected readonly Renderer _host;

    // Previous-frame camera snapshot. Any change restarts progressive integration. Identity/zero
    // defaults mean the first frame after switching to this core always restarts (intended).
    private Matrix4x4 _lastCamView = Matrix4x4.Identity;
    private Vector3   _lastCamPos  = Vector3.Zero;
    // FOV doesn't move the view matrix, so it needs its own snapshot.
    private float     _lastCamFov;

    protected PathTraceCoreBase(Renderer host)
    {
        _host = host;
        host.RegisterCore(this);   // cores add themselves to the host's render-core registry
    }

    public abstract string Name { get; }
    public abstract Renderer.RenderMode Mode { get; }

    // ---- Pipeline-specific hooks (forwarded to PTComputePipeline / RTPipeline) ------------------
    /// <summary>The stage that writes the accumulator / ptOutColor: compute vs ray-tracing. Used for
    /// the pre-dispatch barrier's src (|fragment) + dst masks.</summary>
    protected abstract PipelineStageFlags WriterStage { get; }
    protected abstract void PipelineMarkAccumulatorDirty();
    protected abstract bool PipelineUpdatePerFrame(uint frameIndex, Camera camera, uint lightCount, Extent2D renderExtent);
    protected abstract void PipelineRecord(CommandBuffer cmd, in Renderer.FrameContext ctx);
    protected abstract void PipelineWriteStorageImages(ImageView accumView, ImageView outColorView);

    /// <summary>Re-point tonemap's shared HDR-input descriptor at ptOutColor (the PT scene-colour
    /// image). Called by the host on mode switch / after a tonemap rebuild -- replaces the per-frame
    /// _lastRenderMode rebind.</summary>
    public void Activate()
    {
        _host.tonemapPipeline.WriteHdrInputDescriptor(
            _host.renderTargets.PtOutColor.ImageView, _host.gBufferSampler);
        // The accumulator + ptOutColor are shared across all PT cores (compute / RT / wavefront),
        // so on a switch into this core they still hold the previous core's progressive result.
        // Restart integration deterministically here (after the host's DeviceWaitIdle) rather than
        // leaning on the host accumulator-dirty flag, which otherwise lands a frame late and can
        // leave the stale image visible.
        PipelineMarkAccumulatorDirty();
    }

    /// <summary>Rebind the pipeline's storage-image descriptors to the freshly-reallocated
    /// accumulator + ptOutColor (WriteStorageImageDescriptors marks the accumulator dirty
    /// internally, so the next dispatch clears the fresh memory rather than reading garbage).</summary>
    public void Resize(Extent2D extent) =>
        PipelineWriteStorageImages(
            _host.renderTargets.PtAccumulator.ImageView, _host.renderTargets.PtOutColor.ImageView);

    public void Render(in RenderFrame frame)
    {
        var cmd = frame.Cmd;
        var ctx = frame.Frame;
        var camera = _host.Camera;
        var scene  = _host.Scene;
        uint currentFrame  = ctx.FrameIndex;
        var renderExtent   = ctx.RenderExtent;

        var ptAccumulator = _host.renderTargets.PtAccumulator;
        var ptOutColor    = _host.renderTargets.PtOutColor;

        // Camera motion -> restart accumulation. Cheap structural-equality check against last
        // frame's snapshot. If Camera ever grows a built-in dirty flag, swap this out for that.
        if (camera != null)
        {
            var view = camera.GetViewMatrix();
            var pos  = camera.GetPosition();
            var fov  = camera.Fov;
            if (view != _lastCamView || pos != _lastCamPos || fov != _lastCamFov)
            {
                _host.MarkAccumulatorDirty();
                _lastCamView = view; _lastCamPos = pos; _lastCamFov = fov;
            }
        }

        // Refresh per-frame SSBOs — lights AND materials (so inspector edits land in PT mode).
        uint lightCount = _host.UpdateLights(currentFrame, scene);
        _host.UpdateMaterials(currentFrame, scene);

        // Bridge the host's dirty signal into the pipeline's accumulator reset.
        if (_host.AccumulatorDirty)
        {
            PipelineMarkAccumulatorDirty();
            _host.ClearAccumulatorDirty();
        }

        PipelineUpdatePerFrame(currentFrame, camera!, lightCount, renderExtent);

        //  Pre-dispatch barrier
        // ptAccumulator: previous frame's compute read+write -> this frame's same.
        // ptOutColor:    previous frame's tonemap fragment read (or first-frame Undefined->General)
        //                -> this frame's write. Both stay in General the whole time; tonemap reads
        // ptOutColor in ShaderReadOnly between dispatch and end-of-frame, re-stamped to General
        // before the next dispatch lands here.
        var preBarriers = stackalloc ImageMemoryBarrier[2];
        preBarriers[0] = new ImageMemoryBarrier
        {
            SType = StructureType.ImageMemoryBarrier,
            OldLayout = ImageLayout.General,
            NewLayout = ImageLayout.General,
            SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
            Image = ptAccumulator.Image,
            SrcAccessMask = AccessFlags.ShaderReadBit | AccessFlags.ShaderWriteBit,
            DstAccessMask = AccessFlags.ShaderReadBit | AccessFlags.ShaderWriteBit,
            SubresourceRange = new ImageSubresourceRange(ImageAspectFlags.ColorBit, 0, 1, 0, 1),
        };
        preBarriers[1] = new ImageMemoryBarrier
        {
            SType = StructureType.ImageMemoryBarrier,
            OldLayout = ImageLayout.General,
            NewLayout = ImageLayout.General,
            SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
            Image = ptOutColor.Image,
            SrcAccessMask = AccessFlags.ShaderReadBit,
            DstAccessMask = AccessFlags.ShaderWriteBit,
            SubresourceRange = new ImageSubresourceRange(ImageAspectFlags.ColorBit, 0, 1, 0, 1),
        };
        _host.vk!.CmdPipelineBarrier(cmd,
            WriterStage | PipelineStageFlags.FragmentShaderBit,
            WriterStage,
            0, 0, null, 0, null, 2, preBarriers);

        PipelineRecord(cmd, ctx);

        // Tonemap into FinalColor. Tonemap's HDR input descriptor was rebound to ptOutColor in
        // Activate (mode switch). It expects the image in ShaderReadOnly and writes FinalColor as a
        // color attachment.
        _host.TransitionImageLayout(cmd, ptOutColor.Image, ptOutColor._format,
            ImageLayout.General, ImageLayout.ShaderReadOnlyOptimal);

        var finalColor = _host.renderTargets.FinalColor;
        _host.TransitionImageLayout(cmd, finalColor.Image, finalColor._format,
            ImageLayout.ShaderReadOnlyOptimal, ImageLayout.ColorAttachmentOptimal);

        _host.tonemapPipeline.Record(cmd, ctx, finalColor.ImageView);

        _host.TransitionImageLayout(cmd, finalColor.Image, finalColor._format,
            ImageLayout.ColorAttachmentOptimal, ImageLayout.ShaderReadOnlyOptimal);

        // ptOutColor back to General so next frame's dispatch + pre-barrier path are layout-uniform.
        _host.TransitionImageLayout(cmd, ptOutColor.Image, ptOutColor._format,
            ImageLayout.ShaderReadOnlyOptimal, ImageLayout.General);
    }

    // PT cores own no GPU resources (pipelines + images are host-owned); nothing to free.
    public void Dispose() { }
}
