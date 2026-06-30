using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using CadThingo.VulkanEngine.GLTF;
using CadThingo.VulkanEngine.Renderer.Pipelines;
using CadThingo.VulkanEngine.Renderer.FrameGraph;   // GraphStats (deferred-graph debug surface)
using CadThingo.VulkanEngine.Renderer.RenderCores;  // IRenderCore, RenderFrame
using CadThingo.VulkanEngine.Renderer.Features.WavefrontPathTracer;  // WavefrontPTCore (active-core type test)
using CadThingo.VulkanEngine.Renderer.Features.Selection;  // PickPipeline.PickNone
using Silk.NET.Core.Native;
using Silk.NET.Vulkan;
using Silk.NET.Vulkan.Extensions.KHR;

namespace CadThingo.VulkanEngine.Renderer;

public unsafe partial class Renderer
{
    private void SetupDynamicRendering()
    {
        //create color attachment
        ClearValue backgroundValue = new()
        {
            Color = new ClearColorValue() { Float32_0 = 0.0f, Float32_1 = 0.0f, Float32_2 = 0.0f, Float32_3 = 1.0f }
        };
        colorAttachments = new()
        {
            new RenderingAttachmentInfo()
            {
                ImageLayout = ImageLayout.ColorAttachmentOptimal,
                LoadOp = AttachmentLoadOp.Clear,
                StoreOp = AttachmentStoreOp.Store,
                ClearValue = backgroundValue,
            }
        };
        //create depth attachment
        ClearValue depthValue = new()
        {
            DepthStencil = new ClearDepthStencilValue(1.0f, 0)
        };
        depthAttachment = new()
        {
            ImageLayout = ImageLayout.DepthStencilAttachmentOptimal,
            LoadOp = AttachmentLoadOp.Clear,
            StoreOp = AttachmentStoreOp.Store,
            ClearValue = depthValue,
        };
        //Create rendering info
        fixed (RenderingAttachmentInfo* pColorAttachment = &colorAttachments.ToArray()[0])
        fixed(RenderingAttachmentInfo* pDepthAttachment = &depthAttachment)
        {
            renderingInfo = new()
            {
                RenderArea = new Rect2D(new Offset2D(0, 0), swapChainExtent),
                LayerCount = 1,
                ColorAttachmentCount = (uint)colorAttachments.Count,
                PColorAttachments = pColorAttachment,
                PDepthAttachment = pDepthAttachment
            };
            
        }
    }
    
    // Rebuilds everything tied to the surface extent: swap chain, attachment
    // images, render graph (which captures width/height in its pass closures),
    // and the lighting pass's g-buffer descriptor writes.
    // Public so Engine can call it from window.Resize — otherwise the only
    // trigger is AcquireNextImage returning ErrorOutOfDateKhr, which leaves
    // one frame at the wrong dimensions on some platforms.
    public void RecreateSwapChain()
    {
        // Block while the window is minimized (zero framebuffer). DoEvents pumps
        // the message loop so we still respond to restore.
        var fb = window!.FramebufferSize;
        while (fb.X == 0 || fb.Y == 0)
        {
            window.DoEvents();
            fb = window.FramebufferSize;
        }
        vk!.DeviceWaitIdle(device);

        swapchain.Recreate();

        // Window resized — bring renderExtent back to swapchain size by default.
        // ViewportPanel will then drive it back down to panel size next frame if
        // the panel is smaller. Setting both equal here avoids a brief frame at
        // the wrong size before the panel's request lands.
        RebuildRenderTargets(swapChainExtent.Width, swapChainExtent.Height);
        imGuiUtils?.UpdateScreenSize(swapChainExtent.Width, swapChainExtent.Height);
    }

    /// <summary>
    /// Reallocates depth + g-buffers + render graph at (<paramref name="width"/>,
    /// <paramref name="height"/>) and re-binds every descriptor whose ImageView
    /// changed. Caller must have invoked vkDeviceWaitIdle beforehand — none of
    /// the disposed resources can be in flight.
    /// </summary>
    private void RebuildRenderTargets(uint width, uint height)
    {
        // Diagnostic: count full render-target rebuilds so the Stats panel can
        // correlate spp/s degradation with resize churn (see GpuMemoryAllocator
        // retained-block investigation).
        _renderTargetRebuilds++;

        // The cores own their size-dependent technique state. Reallocate the shared host targets
        // first (FinalColor + PT accumulator/out + selection), then rebuild each core: DeferredCore
        // re-compiles its graph (fresh g-buffer/depth/HDR transients, re-imports FinalColor,
        // re-binds its lighting + tonemap descriptors); the PT cores rebind their storage images
        // (which marks the accumulator dirty so the next dispatch clears fresh memory).
        renderTargets.ReallocateSizeDependent(new Extent2D(width, height));
        // Every registered core rebuilds its size-dependent state: DeferredCore/WavefrontPTCore
        // re-compile their graphs (fresh transients, re-import FinalColor, re-bind descriptors);
        // the PT/RT cores rebind their storage images (which marks the accumulator dirty so the
        // next dispatch clears fresh memory).
        foreach (var core in _renderCores) core.Resize(renderExtent);

        // Re-point tonemap's HDR input at the ACTIVE core's fresh scene-colour source (deferred
        // HDR vs PT ptOutColor). Subsumes the old per-mode branch: DeferredCore.Resize just bound
        // tonemap to the new deferred HDR, so in a PT mode we must override it back to ptOutColor.
        _activeCore.Activate();

        // FinalColor ImageView is fresh — re-bind it for the ImGui viewport panel.
        imGuiUtils?.WriteViewportDescriptor(renderTargets.FinalColor.ImageView);

        // Selection mask view is fresh — re-bind it on both the compute (storage
        // image) and outline (sampled image) sides.
        selectionMaskPipeline?.WriteMaskImageDescriptor(selectionMask.ImageView);
        outlinePipeline?.WriteMaskDescriptor(selectionMask.ImageView);

        // Diagnostic: sample allocator occupancy now that old targets are freed and
        // new ones allocated — the steady post-resize state for the history plot.
        RecordMemorySample();
    }

    /// <summary>
    /// Public entry: blocks on vkDeviceWaitIdle and reallocates render targets at
    /// the requested extent. Safe to call from outside the renderer (e.g.
    /// ViewportPanel via EditorState → DrawFrame top-of-frame check).
    /// </summary>
    public void ResizeRenderTargets(uint width, uint height)
    {
        if (width == 0 || height == 0) return;
        if (width == renderExtent.Width && height == renderExtent.Height) return;

        vk!.DeviceWaitIdle(device);
        RebuildRenderTargets(width, height);
    }
    public readonly ref struct FrameContext
    {
        public readonly uint     FrameIndex { get; init;}
        public readonly Camera   Camera { get; init;}
        public readonly Scene    Scene { get; init;}
        public readonly Extent2D RenderExtent { get; init;}
        // bindless descriptor set, current view/proj/inv matrices precomputed, etc.
    }
    public void DrawFrame()
    {
        
        
        
        // 0. Apply any pending render-extent request from the ViewportPanel.
        //    Requests are 1 frame stale (the panel computes size during ImGui
        //    construction inside the previous frame's DrawFrame); that's fine.
        //    ResizeRenderTargets is a no-op when the request matches current.
        if (ImGui.EditorState.RequestedRenderExtent is var req && req.HasValue)
        {
            ResizeRenderTargets(req.Value.w, req.Value.h);
            ImGui.EditorState.RequestedRenderExtent = null;
        }

        // 0b. Apply any pending pipeline rebuilds posted by the Renderer Settings
        //     panel. Has to happen here (before command-buffer recording) because
        //     mid-frame disposal of a bound pipeline would corrupt the in-flight
        //     command buffer's references.
        if (pendingPbrRebuild)     { RebuildPbrPipelines();     pendingPbrRebuild     = false; }
        if (pendingTonemapRebuild) { RebuildTonemapPipeline();  pendingTonemapRebuild = false; }

        // 0c. Stale TLAS flush. tlasDirty is flipped by editor mutations
        //     (transform sliders, visibility toggles) and consumed here so we
        //     do at most one RebuildTlas per frame regardless of how many
        //     slider ticks landed last frame. OnSceneEntitiesChanged also
        //     re-runs the BLAS build pass for any new meshes and re-binds the
        //     TLAS / shadow-info descriptors.
        if (tlasDirty)
        {
            OnSceneEntitiesChanged();
            // RebuildTlas already clears tlasDirty on success; for hardware
            // paths where ray shadows aren't supported (RebuildTlas never runs)
            // we still need to clear it so the next frame doesn't re-enter.
            tlasDirty = false;
        }

        // 0c-bis. Object pick. Consumes a click posted by ViewportPanel and runs
        //     a one-ray compute dispatch against the (now up-to-date) TLAS,
        //     setting EditorState.SelectedEntity. Out-of-band single-time submit,
        //     so it sits here before the per-frame command buffer is recorded.
        ProcessPickRequest();

        // 0d. Render-mode change: swap the active core. Its Activate() rebinds tonemap's HDR
        //     input to the core's scene-colour source (deferred HDR vs PT ptOutColor).
        //     DeviceWaitIdle ensures no in-flight frame is still using the old binding. Mode
        //     flips are user-driven (ImGui combo), so the hitch is acceptable. This replaces the
        //     old per-frame _lastRenderMode tonemap-rebind check.
        var desiredCore = _renderCores[_desiredCoreIndex];
        if (!ReferenceEquals(desiredCore, _activeCore))
        {
            vk!.DeviceWaitIdle(device);
            _activeCore = desiredCore;
            _activeCore.Activate();
        }

        var ctx = new FrameContext()
        {
            FrameIndex = currentFrame,
            Camera = camera,
            RenderExtent = renderExtent,
            Scene = scene
        };
        // 1. CPU/GPU sync for this slot
        vk!.WaitForFences(device, 1, ref inFlightFences[currentFrame], true, ulong.MaxValue);

        // 2. Acquire swapchain image
        var acquireResult = swapchain.AcquireNextImage(imageAvailableSemaphores[currentFrame], out uint imageIndex);
        if (acquireResult == Result.ErrorOutOfDateKhr) { RecreateSwapChain(); return; }

        // 3. Reset fence — we're about to submit work that will signal it
        vk!.ResetFences(device, 1, ref inFlightFences[currentFrame]);

        // 4. Reset + begin command buffer
        var cmd = commandBuffers[currentFrame];
        vk!.ResetCommandBuffer(cmd, 0);
        var beginInfo = new CommandBufferBeginInfo
        {
            SType = StructureType.CommandBufferBeginInfo,
            Flags = CommandBufferUsageFlags.OneTimeSubmitBit,
        };
        if (vk!.BeginCommandBuffer(cmd, &beginInfo) != Result.Success)
            throw new Exception("Failed to begin command buffer");

        // Open this frame's world-transform cache (L2 step 5) - one reset covers
        // every render mode below; the active DrawX's extract reads served from it.
        // (RebuildTlas, which runs out-of-band above, manages its own window.)
        gpuScene.BeginTransforms();

        // Dispatch the active render core (L3). It records its technique into cmd and leaves
        // FinalColor in ShaderReadOnlyOptimal for the host post-stack below. The core was selected
        // (by list index, _renderCores[_desiredCoreIndex]) + Activated at step 0d; RT-only cores
        // never registered on devices without the RT pipeline, so they can't be the desired core.
        _activeCore.Render(new RenderFrame { Cmd = cmd, Frame = ctx });

        // 6b. Selection outline. Both render modes leave FinalColor in
        //     ShaderReadOnly here; this composites the outline overlay in place
        //     (no-op when nothing's selected) and restores that layout.
        RecordSelectionOutline(cmd);

        // 7. Blit FinalColor -> swapchain image
        var swapImage = swapChainImages[imageIndex];

        // 7a. Swapchain Undefined -> TransferDstOptimal
        var toTransferDst = new ImageMemoryBarrier
        {
            SType = StructureType.ImageMemoryBarrier,
            OldLayout = ImageLayout.Undefined,
            NewLayout = ImageLayout.TransferDstOptimal,
            SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
            Image = swapImage,
            SrcAccessMask = 0,
            DstAccessMask = AccessFlags.TransferWriteBit,
            SubresourceRange = new ImageSubresourceRange(ImageAspectFlags.ColorBit, 0, 1, 0, 1),
        };
        vk!.CmdPipelineBarrier(cmd,
            PipelineStageFlags.TopOfPipeBit,
            PipelineStageFlags.TransferBit,
            0, 0, null, 0, null, 1, &toTransferDst);

        // 7b. Blit FinalColor → swapchain.
        // FinalColor's graph-declared finalLayout is now ShaderReadOnlyOptimal (so the
        // viewport panel can sample it), so we dance it through TransferSrcOptimal
        // here and back to ShaderReadOnlyOptimal after the blit. The graph's layout
        // tracker stays consistent because we end where the graph thinks it ends.
        var finalColor = renderTargets.FinalColor;
        TransitionImageLayout(cmd, finalColor.Image, finalColor._format,
            ImageLayout.ShaderReadOnlyOptimal, ImageLayout.TransferSrcOptimal);
        var blit = new ImageBlit
        {
            SrcSubresource = new ImageSubresourceLayers
            {
                AspectMask = ImageAspectFlags.ColorBit,
                MipLevel = 0,
                BaseArrayLayer = 0,
                LayerCount = 1,
            },
            DstSubresource = new ImageSubresourceLayers
            {
                AspectMask = ImageAspectFlags.ColorBit,
                MipLevel = 0,
                BaseArrayLayer = 0,
                LayerCount = 1,
            },
        };
        // Src is FinalColor at renderExtent; Dst is the swapchain image at
        // swapChainExtent. CmdBlitImage handles the scale when they differ.
        blit.SrcOffsets[0] = new Offset3D(0, 0, 0);
        blit.SrcOffsets[1] = new Offset3D((int)renderExtent.Width, (int)renderExtent.Height, 1);
        blit.DstOffsets[0] = new Offset3D(0, 0, 0);
        blit.DstOffsets[1] = new Offset3D((int)swapChainExtent.Width, (int)swapChainExtent.Height, 1);

        vk!.CmdBlitImage(cmd,
            finalColor.Image, ImageLayout.TransferSrcOptimal,
            swapImage, ImageLayout.TransferDstOptimal,
            1, &blit, Filter.Linear);

        // 7b-post. Return FinalColor to ShaderReadOnlyOptimal so the viewport ImGui
        //     panel (and the render graph's layout tracker) can rely on it.
        TransitionImageLayout(cmd, finalColor.Image, finalColor._format,
            ImageLayout.TransferSrcOptimal, ImageLayout.ShaderReadOnlyOptimal);

        // 7c. Swapchain TransferDstOptimal → ColorAttachmentOptimal so the UI overlay
        //     can render on top of the blitted scene.
        TransitionImageLayout(cmd, swapImage, swapChainImageFormat,
            ImageLayout.TransferDstOptimal, ImageLayout.ColorAttachmentOptimal);

        // 7d. UI overlay (no-op when imGuiUtils is unwired).
        if (imGuiUtils != null)
        {
            imGuiUtils.newFrame();
            imGuiUtils.updateBuffers(currentFrame);
        }
        imGuiUtils?.DrawFrame(cmd, swapChainImageViews[imageIndex], currentFrame);

        // 7e. Swapchain ColorAttachmentOptimal → PresentSrcKhr.
        TransitionImageLayout(cmd, swapImage, swapChainImageFormat,
            ImageLayout.ColorAttachmentOptimal, ImageLayout.PresentSrcKhr);

        // 8. End command buffer
        if (vk!.EndCommandBuffer(cmd) != Result.Success)
            throw new Exception("Failed to end command buffer");

        // 9. Submit: wait imageAvailable @ Transfer, signal renderFinished, fence inFlight.
        // Always uses vkQueueSubmit2. When the active core has deferred gfx chunks (wavefront
        // async path), SubmitGfxChunks merges them with the host cmd so all graphics work lands
        // in one unified submission -- making every graph pass visible in Nsight's timeline bar.
        var imgAvailWait = new SemaphoreSubmitInfo
        {
            SType     = StructureType.SemaphoreSubmitInfo,
            Semaphore = imageAvailableSemaphores[currentFrame],
            Value     = 0,
            StageMask = PipelineStageFlags2.TransferBit,
        };
        var renderDoneSignal = new SemaphoreSubmitInfo
        {
            SType     = StructureType.SemaphoreSubmitInfo,
            Semaphore = renderFinishedSemaphores[currentFrame],
            Value     = 0,
            StageMask = PipelineStageFlags2.AllCommandsBit,
        };

        if (ActiveGraphCore is { HasPendingGfxChunks: true } graphCore)
        {
            graphCore.SubmitGfxChunks(graphicsQueue, imgAvailWait, renderDoneSignal,
                cmd, inFlightFences[currentFrame]);
        }
        else
        {
            var hostCmdInfo = new CommandBufferSubmitInfo
            {
                SType = StructureType.CommandBufferSubmitInfo,
                CommandBuffer = cmd,
            };
            var submitInfo2 = new SubmitInfo2
            {
                SType                    = StructureType.SubmitInfo2,
                WaitSemaphoreInfoCount   = 1,
                PWaitSemaphoreInfos      = &imgAvailWait,
                CommandBufferInfoCount   = 1,
                PCommandBufferInfos      = &hostCmdInfo,
                SignalSemaphoreInfoCount = 1,
                PSignalSemaphoreInfos    = &renderDoneSignal,
            };
            if (vk!.QueueSubmit2(graphicsQueue, 1, &submitInfo2, inFlightFences[currentFrame]) != Result.Success)
                throw new Exception("Queue submit failed");
        }

        // 10. Present — wait on renderFinished
        swapchain.Present(presentQueue, renderFinishedSemaphores[currentFrame], imageIndex);

        // Advance to the next frame-in-flight slot + bump the monotonic counter.
        frameRing.Advance();
    }
    /// <summary>
    /// Consumes a pending viewport pick (posted by ViewportPanel as a render-
    /// target pixel) and resolves it to an entity by casting one ray through the
    /// TLAS in a compute dispatch. The pick pass returns the hit's
    /// ShadowEntityInfo.EntityIndex — a stable RenderableHandle slot (see
    /// RebuildTlas) — which <see cref="GpuScene.ResolveSlot"/> maps back to the
    /// entity. Runs as a self-contained single-time submit (QueueWaitIdle), so the
    /// host-visible result is ready immediately. No-op unless ray queries are
    /// supported and a TLAS exists.
    /// </summary>
    private void ProcessPickRequest()
    {
        var req = ImGui.EditorState.RequestedPick;
        if (!req.HasValue) return;
        ImGui.EditorState.RequestedPick = null;

        if (!RayShadowsSupported || khrAccelStruct == null || tlas.Handle == 0 || pickPipeline == null)
            return;

        uint px = req.Value.x;
        uint py = req.Value.y;
        if (px >= renderExtent.Width || py >= renderExtent.Height) return;

        // Same Y-flipped projection the geometry / lighting / light-cull passes
        // use, so the pick ray lines up with the rasterized image (and the PT
        // image, which flips the same way).
        Matrix4x4 view = camera.GetViewMatrix();
        Matrix4x4 proj = camera.GetProjectionMatrix(
            (float)renderExtent.Width / renderExtent.Height, 0.1f, 100.0f);
        proj.M22 *= -1f;
        if (!Matrix4x4.Invert(view * proj, out Matrix4x4 invVP)) return;

        var cmd = BeginSingleTimeCommands();
        pickPipeline.Record(cmd, invVP, camera.GetPosition(),
            new Vector2(renderExtent.Width, renderExtent.Height), px, py);
        EndSingleTimeCommands(cmd);   // QueueWaitIdle — result buffer is now valid

        uint idx = pickPipeline.ReadResult();
        ImGui.EditorState.SelectedEntity =
            idx == PickPipeline.PickNone ? null : gpuScene.ResolveSlot(idx);
    }

    /// <summary>
    /// Composites the selection outline into FinalColor. Runs after the active
    /// render mode has produced FinalColor (left in ShaderReadOnly by both the
    /// deferred and PT paths) and before the swapchain blit, so one insertion
    /// point covers every mode. Ray-queries the TLAS into the coverage mask,
    /// then draws an outer ring around the selected entity's silhouette. No-op
    /// unless a mesh-bearing entity is selected and ray queries are available;
    /// leaves FinalColor in ShaderReadOnly exactly as it found it.
    /// </summary>
    private void RecordSelectionOutline(CommandBuffer cmd)
    {
        if (ImGui.EditorState.SelectedEntity == null) return;
        if (!RayShadowsSupported || khrAccelStruct == null || tlas.Handle == 0) return;
        if (selectionMaskPipeline == null || outlinePipeline == null) return;

        // Resolve the selection to its RenderableHandle slot — the same token the
        // mask shader compares against ShadowEntityInfo.EntityIndex (L2 step 6). No
        // handle ⇒ not a renderable (e.g. a light picked in the outliner) ⇒ no
        // outline, which is correct.
        if (!gpuScene.TryGetHandle(ImGui.EditorState.SelectedEntity, out var selHandle)) return;
        uint idx = selHandle.Index;

        Matrix4x4 view = camera.GetViewMatrix();
        Matrix4x4 proj = camera.GetProjectionMatrix(
            (float)renderExtent.Width / renderExtent.Height, 0.1f, 100.0f);
        proj.M22 *= -1f;
        if (!Matrix4x4.Invert(view * proj, out Matrix4x4 invVP)) return;

        // Mask sits in ShaderReadOnly between frames. Flip to General before the
        // compute write; this fragment-read→compute-write barrier also serializes
        // the previous frame's outline read against this frame's overwrite.
        TransitionImageLayout(cmd, selectionMask.Image, selectionMask._format,
            ImageLayout.ShaderReadOnlyOptimal, ImageLayout.General);

        selectionMaskPipeline.Record(cmd, invVP, camera.GetPosition(), renderExtent, idx);

        // compute write → outline fragment read
        TransitionImageLayout(cmd, selectionMask.Image, selectionMask._format,
            ImageLayout.General, ImageLayout.ShaderReadOnlyOptimal);

        var finalColor = renderTargets.FinalColor;
        TransitionImageLayout(cmd, finalColor.Image, finalColor._format,
            ImageLayout.ShaderReadOnlyOptimal, ImageLayout.ColorAttachmentOptimal);

        outlinePipeline.Record(cmd, renderExtent, finalColor.ImageView);

        // Back to ShaderReadOnly — exactly where the swapchain blit (step 7) and
        // the ImGui viewport sampler expect FinalColor to be.
        TransitionImageLayout(cmd, finalColor.Image, finalColor._format,
            ImageLayout.ColorAttachmentOptimal, ImageLayout.ShaderReadOnlyOptimal);
    }


    // ---- Active-core FrameGraph debug surface (read by the Stats panel) ----------------------
    // The graph is owned by whichever core is active (DeferredCore, WavefrontPTCore, ...), so these
    // forward to `_activeCore` through IGraphCore. Cores with no graph (megakernel PT, forward+)
    // aren't IGraphCore -> null, and the panel shows nothing.
    private IGraphCore? ActiveGraphCore => _activeCore as IGraphCore;
    /// <summary>Last-frame per-pass GPU/CPU timings + counts, or null when the active core has no graph.</summary>
    public GraphStats? ActiveGraphStats => ActiveGraphCore?.GraphStats;
    /// <summary>The active core's human-readable name (for the panel header).</summary>
    public string ActiveCoreName => _activeCore?.Name ?? "";

    // ---- Resize-churn diagnostics (GpuMemoryAllocator retained-block check) -----
    private int _renderTargetRebuilds;
    /// <summary>Count of full render-target rebuilds since launch (one per resize).</summary>
    public int RenderTargetRebuilds => _renderTargetRebuilds;
    /// <summary>Live allocator occupancy — reserved (held from driver) vs. actually used.</summary>
    public AllocatorStats GpuMemoryStats => gfx.Allocator.GetStats();
    /// <summary>Driver-reported WDDM budget/usage for the device-local heap (VK_EXT_memory_budget),
    /// or Available=false when the extension isn't enabled. This is the authoritative number our
    /// hand-rolled reserved/used counters approximate - it includes the OS budget our committed
    /// allocations are racing against.</summary>
    public MemoryBudgetInfo GpuMemoryBudget => gfx.QueryMemoryBudget();

    // Per-rebuild MB history (sampled at the END of each RebuildRenderTargets, i.e.
    // steady post-resize state). The SHAPE discriminates the diagnosis: a monotonic
    // climb with rebuild count == per-resize leak; a step-then-plateau == one-time
    // high-water (retained empty block) with no leak.
    private const int MemHistoryLen = 256;
    private readonly float[] _usedMbHistory     = new float[MemHistoryLen];
    private readonly float[] _reservedMbHistory = new float[MemHistoryLen];
    private int _memHistoryHead;
    public float[] UsedMbHistory     => _usedMbHistory;
    public float[] ReservedMbHistory => _reservedMbHistory;
    public int     MemHistoryHead    => _memHistoryHead;
    public int     MemHistoryLength  => MemHistoryLen;

    private void RecordMemorySample()
    {
        var s = gfx.Allocator.GetStats();
        const float MB = 1024f * 1024f;
        _usedMbHistory[_memHistoryHead]     = s.UsedBytes     / MB;
        _reservedMbHistory[_memHistoryHead] = s.ReservedBytes / MB;
        _memHistoryHead = (_memHistoryHead + 1) % MemHistoryLen;
    }
    /// <summary>Graphviz dump of the active core's compiled graph (for "Copy DOT").</summary>
    public string ActiveGraphDot() => ActiveGraphCore?.ToDot() ?? "(active core has no frame graph)";
    /// <summary>Runtime toggle for the active core's graph pipeline-statistics collection.</summary>
    public bool ActiveGraphPipelineStats
    {
        get => ActiveGraphCore?.CollectPipelineStats ?? false;
        set { if (ActiveGraphCore is { } g) g.CollectPipelineStats = value; }
    }

    /// <summary>TEMP debug: per-bounce [extend, shade, connect] indirect workgroup counts for the
    /// wavefront tracer (compaction visualization), or null when wavefront isn't the active core.
    /// ~a frame stale, best-effort. Remove with the pipeline's _argsReadback readback feature.</summary>
    public uint[]? WavefrontDispatchCounts =>
        _activeCore is WavefrontPTCore && wavefrontPipeline != null
            ? wavefrontPipeline.ReadDispatchArgs() : null;







    
    
    
    /// <summary>
    /// 
    /// </summary>
   
}

unsafe class CullingSystem(Camera camera)
{
    private Camera camera;
    private List<Entity> visibleEntities;

    void SetCamera(Camera camera) => this.camera = camera;


    public void CullScene(List<Entity> allEntities)
    {
        visibleEntities.Clear();

        if (camera == null) return;

        //Get camera frustum
        //TODO: Make culling system use the new camera 
        Frustum frustum = camera.GetFrustum();

        //check each entity against frustum
        foreach (var entity in allEntities)
        {
            if (!entity.IsActive) continue;

            var meshComponent = entity.GetComponent<MeshComponent>();
            if (meshComponent == null) continue;

            var transformComponent = entity.GetComponent<TransformComponent>();
            if (transformComponent == null) continue;

            //Get bouding box of the mesh
            BoundingBox boundingBox = meshComponent.GetBoundingBox();
            //transform the bounding box by entity transform
            boundingBox.Transform(transformComponent.GetModelMatrix());

            //check if bounding box is visible
            if (frustum.Intersects(boundingBox))
            {
                visibleEntities.Add(entity);
            }
        }
    }

    public List<Entity> GetVisibleEntities() => visibleEntities;
}




[StructLayout(LayoutKind.Sequential)]
public unsafe struct Plane
{
    // xyz = unit normal, w = signed distance from origin
    public Vector4 Data;
 
    public Vector3 Normal   => new(Data.X, Data.Y, Data.Z);
    public float   Distance => Data.W;
 
    /// <summary>
    /// Build a plane from the raw coefficients (A, B, C, D) where
    /// the plane equation is Ax + By + Cz + D = 0.
    /// The normal is normalised so distance comparisons are in world units.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Plane FromCoefficients(float a, float b, float c, float d)
    {
        float invLen = 1f / MathF.Sqrt(a * a + b * b + c * c);
        return new Plane
        {
            Data = new Vector4(a * invLen, b * invLen, c * invLen, d * invLen)
        };
    }
 
    /// <summary>
    /// Signed distance from a point to this plane.
    /// Positive = in front (same side as normal).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float SignedDistance(Vector3 point) =>
        Data.X * point.X + Data.Y * point.Y + Data.Z * point.Z + Data.W;
}




 
[StructLayout(LayoutKind.Sequential)]
public unsafe struct Frustum
{
    // Six planes in a fixed inline array — no heap allocation,
    // GC never touches this struct.
    // Order matches the standard OpenGL/Vulkan convention:
    //   0 Left  1 Right  2 Bottom  3 Top  4 Near  5 Far
    private fixed float _planeData[6 * 4]; // 6 planes × 4 floats (Vector4)
 
    // Index constants
    public const int Left   = 0;
    public const int Right  = 1;
    public const int Bottom = 2;
    public const int Top    = 3;
    public const int Near   = 4;
    public const int Far    = 5;
 
    // Plane accessors
 
    /// <summary>
    /// Read or write a plane by index.
    /// Uses a pointer into the fixed array — no copy.
    /// </summary>
    public Plane GetPlane(int index)
    {
        fixed (float* p = _planeData)
        {
            float* slot = p + index * 4;
            return new Plane
            {
                Data = new Vector4(slot[0], slot[1], slot[2], slot[3])
            };
        }
    }
 
    public void SetPlane(int index, Plane plane)
    {
        fixed (float* p = _planeData)
        {
            float* slot = p + index * 4;
            slot[0] = plane.Data.X;
            slot[1] = plane.Data.Y;
            slot[2] = plane.Data.Z;
            slot[3] = plane.Data.W;
        }
    }
 
    // Convenience properties for named access
    public Plane PlaneLeft   { get => GetPlane(Left);   set => SetPlane(Left,   value); }
    public Plane PlaneRight  { get => GetPlane(Right);  set => SetPlane(Right,  value); }
    public Plane PlaneBottom { get => GetPlane(Bottom); set => SetPlane(Bottom, value); }
    public Plane PlaneTop    { get => GetPlane(Top);    set => SetPlane(Top,    value); }
    public Plane PlaneNear   { get => GetPlane(Near);   set => SetPlane(Near,   value); }
    public Plane PlaneFar    { get => GetPlane(Far);    set => SetPlane(Far,    value); }
 
    // Construction
 
    /// <summary>
    /// Extracts the six frustum planes from a combined view-projection matrix.
    /// Works for both OpenGL (NDC z: -1..1) and Vulkan (NDC z: 0..1) —
    /// pass <paramref name="vulkanNDC"/> = true for Vulkan.
    ///
    /// Uses the Gribb &amp; Hartmann method — directly reads rows/columns of
    /// the combined VP matrix, no trig or ray-casting required.
    /// </summary>
    public static Frustum FromViewProjection(Matrix4x4 vp, bool vulkanNDC = true)
    {
        // Matrix4x4 in System.Numerics is row-major.
        // Gribb/Hartmann extracts planes by adding/subtracting rows.
        //
        // Row vectors (M.MiN notation, 1-based):
        //   r1 = (M11, M12, M13, M14)
        //   r2 = (M21, M22, M23, M24)
        //   r3 = (M31, M32, M33, M34)
        //   r4 = (M41, M42, M43, M44)
        //
        // Plane normals:
        //   Left   =  r4 + r1
        //   Right  =  r4 - r1
        //   Bottom =  r4 + r2
        //   Top    =  r4 - r2
        //   Near   =  r3           (Vulkan: z in [0,1])  or  r4 + r3 (OpenGL)
        //   Far    =  r4 - r3
 
        Frustum f = default;
 
        f.SetPlane(Left,   Plane.FromCoefficients(
            vp.M14 + vp.M11, vp.M24 + vp.M21, vp.M34 + vp.M31, vp.M44 + vp.M41));
 
        f.SetPlane(Right,  Plane.FromCoefficients(
            vp.M14 - vp.M11, vp.M24 - vp.M21, vp.M34 - vp.M31, vp.M44 - vp.M41));
 
        f.SetPlane(Bottom, Plane.FromCoefficients(
            vp.M14 + vp.M12, vp.M24 + vp.M22, vp.M34 + vp.M32, vp.M44 + vp.M42));
 
        f.SetPlane(Top,    Plane.FromCoefficients(
            vp.M14 - vp.M12, vp.M24 - vp.M22, vp.M34 - vp.M32, vp.M44 - vp.M42));
 
        if (vulkanNDC)
        {
            // Vulkan NDC z in [0, 1]: near plane is just row 3
            f.SetPlane(Near, Plane.FromCoefficients(
                vp.M13, vp.M23, vp.M33, vp.M43));
        }
        else
        {
            // OpenGL NDC z in [-1, 1]: near plane is r4 + r3
            f.SetPlane(Near, Plane.FromCoefficients(
                vp.M14 + vp.M13, vp.M24 + vp.M23, vp.M34 + vp.M33, vp.M44 + vp.M43));
        }
 
        f.SetPlane(Far, Plane.FromCoefficients(
            vp.M14 - vp.M13, vp.M24 - vp.M23, vp.M34 - vp.M33, vp.M44 - vp.M43));
 
        return f;
    }
 
 
    /// <summary>
    /// Tests whether an axis-aligned bounding box intersects or is inside
    /// this frustum.
    ///
    /// Uses the "positive vertex" (p-vertex) method:
    ///   For each plane, find the corner of the AABB that is furthest in
    ///   the direction of the plane normal (the p-vertex).  If that corner
    ///   is on the negative side of any plane, the whole box is outside.
    ///
    /// Returns false (outside) as early as possible — on average only
    /// 1–2 planes are tested before rejection on a typical scene.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool Intersects(BoundingBox box)
    {
        // Work directly from the fixed array via a pointer — avoids
        // copying each Plane struct through the GetPlane() accessor
        // in the hot path.
        fixed (float* p = _planeData)
        {
            for (int i = 0; i < 6; i++)
            {
                float* slot = p + i * 4;
 
                float nx = slot[0];
                float ny = slot[1];
                float nz = slot[2];
                float d  = slot[3];
 
                // P-vertex: the AABB corner furthest along the plane normal.
                // For each axis, pick Max if the normal component is positive,
                // Min if negative — this is the most likely inside point.
                float px = nx >= 0f ? box.Max.X : box.Min.X;
                float py = ny >= 0f ? box.Max.Y : box.Min.Y;
                float pz = nz >= 0f ? box.Max.Z : box.Min.Z;
 
                // If the p-vertex is behind this plane the whole box is outside.
                if (nx * px + ny * py + nz * pz + d < 0f)
                    return false;
            }
        }
        return true;
    }
 
    /// <summary>
    /// Pointer overload — for callers that already have a BoundingBox*
    /// (e.g. iterating an unmanaged scene object array) and want to avoid
    /// copying the struct onto the managed stack.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool Intersects(BoundingBox* box) => Intersects(*box);
}
