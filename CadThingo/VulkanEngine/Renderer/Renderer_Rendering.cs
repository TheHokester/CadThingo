using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using CadThingo.VulkanEngine.GLTF;
using CadThingo.VulkanEngine.Renderer.Pipelines;
using CadThingo.VulkanEngine.Renderer.FrameGraph;   // PassType, QueueClass, ResourceUsage, GraphImage, ImageDesc, PassResources
using Silk.NET.Core.Native;
using Silk.NET.Vulkan;
using Silk.NET.Vulkan.Extensions.KHR;

// The graph type shares its name with its namespace; alias it so bare `FrameGraph`
// never has to disambiguate type-vs-namespace.
using DeferredGraph = CadThingo.VulkanEngine.Renderer.FrameGraph.FrameGraph;

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

        // The deferred FrameGraph owns the g-buffer/depth/HDR transients and imports
        // FinalColor. Reallocate the imported targets first (FinalColor + PT/selection),
        // then rebuild the graph: that disposes the old transients, allocates fresh ones at
        // the new extent, re-imports FinalColor, and re-binds the lighting g-buffer + tonemap
        // HDR descriptors from the new views.
        renderTargets.ReallocateSizeDependent(new Extent2D(width, height));
        BuildDeferredFrameGraph();

        // BuildDeferredFrameGraph bound tonemap's HDR input to the fresh HDR transient (the
        // deferred source). In a PT mode the input must instead be ptOutColor — also fresh
        // post-rebuild — so re-point it; otherwise PT-mode resize reads the (correct but
        // unused) HDR view. Both PT modes (compute + RT pipeline) write ptOutColor.
        bool ptMode = renderMode == RenderMode.RayCompute || renderMode == RenderMode.RayTrace;
        if (ptMode)
            tonemapPipeline.WriteHdrInputDescriptor(ptOutColor.ImageView, gBufferSampler);

        // FinalColor ImageView is fresh — re-bind it for the ImGui viewport panel.
        imGuiUtils?.WriteViewportDescriptor(renderTargets.FinalColor.ImageView);

        // PT storage image views are fresh too. WriteStorageImageDescriptors calls
        // MarkAccumulatorDirty internally so the next dispatch clears the
        // (also-fresh) accumulator instead of reading garbage memory.
        ptComputePipeline?.WriteStorageImageDescriptors(ptAccumulator.ImageView, ptOutColor.ImageView);
        rtPipeline?.WriteStorageImageDescriptors(ptAccumulator.ImageView, ptOutColor.ImageView);

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

        // 0d. Render-mode change: rebind tonemap's HDR input so its single
        //     shared descriptor set points at the right source. DeviceWaitIdle
        //     ensures no in-flight frame is still using the old binding.
        //     Mode flips are user-driven (ImGui combo), so the hitch is
        //     acceptable; per-frame double-buffered descriptors would avoid it.
        if (renderMode != _lastRenderMode)
        {
            vk!.DeviceWaitIdle(device);
            ImageView source = (renderMode == RenderMode.RayCompute || renderMode == RenderMode.RayTrace)
                ? ptOutColor.ImageView
                : _deferredHdrView;
            tonemapPipeline.WriteHdrInputDescriptor(source, gBufferSampler);
            _lastRenderMode = renderMode;
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

        // Open this frame's world-transform cache (L2 step 5) — one reset covers
        // every render mode below; the active DrawX's extract reads served from it.
        // (RebuildTlas, which runs out-of-band above, manages its own window.)
        gpuScene.BeginTransforms();

        switch (renderMode)
        {
            case RenderMode.Deferred:
                DrawDeferred(cmd, ctx);
                break;
            case RenderMode.ForwardPlus:
                DrawRayQueried(cmd, ctx);
                break;
            case RenderMode.RayCompute:
                DrawPathtraced(cmd, ctx);
                break;
            case RenderMode.RayTrace:
                // Opt-in RT-pipeline path tracer; falls back to deferred if the
                // device didn't expose the feature (rtPipeline == null).
                if (rtPipeline != null) DrawRayTraced(cmd, ctx);
                else                    DrawDeferred(cmd, ctx);
                break;
            default:
                DrawDeferred(cmd, ctx);
                break;
        }

        // 6b. Selection outline. Both render modes leave FinalColor in
        //     ShaderReadOnly here; this composites the outline overlay in place
        //     (no-op when nothing's selected) and restores that layout.
        RecordSelectionOutline(cmd);

        // 7. Blit FinalColor → swapchain image
        var swapImage = swapChainImages[imageIndex];

        // 7a. Swapchain Undefined → TransferDstOptimal
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
        
        // 9. Submit: wait imageAvailable @ Transfer (blit consumes swapchain), signal renderFinished, fence inFlight
        var waitSem = imageAvailableSemaphores[currentFrame];
        var sigSem = renderFinishedSemaphores[currentFrame];
        var waitStage = PipelineStageFlags.TransferBit;
        var submitCmd = cmd;
        var submitInfo = new SubmitInfo
        {
            SType = StructureType.SubmitInfo,
            WaitSemaphoreCount = 1,
            PWaitSemaphores = &waitSem,
            PWaitDstStageMask = &waitStage,
            CommandBufferCount = 1,
            PCommandBuffers = &submitCmd,
            SignalSemaphoreCount = 1,
            PSignalSemaphores = &sigSem,
        };
        if (vk!.QueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != Result.Success)
            throw new Exception("Queue submit failed");

        // 10. Present — wait on renderFinished
        swapchain.Present(presentQueue, renderFinishedSemaphores[currentFrame], imageIndex);

        // Advance to the next frame-in-flight slot + bump the monotonic counter.
        frameRing.Advance();
    }
    /// <summary>
    /// Refactor of existing rendering code in DrawFrame(), the logic unique to DrawFrame()
    /// </summary>
    private void DrawDeferred(CommandBuffer cmd, FrameContext frameContext)
    {
        // Reflection-probe scheduler bookkeeping. Cheap CPU-only walk; the
        //     capture draw + prefilter dispatch land in Phase 4. Runs before the
        //     geometry pass so the captured cube is visible to downstream
        //     shaders within the same frame once it's hooked up.
        reflectionProbeSystem.Tick(frameCounter, scene);

        // Per-frame material SSBO snapshot — needs to land before the geometry
        // pass reads it. PT does the same in DrawPathtraced so inspector edits
        // are visible in either renderer.
        UpdateMaterials(currentFrame, scene);

        geometryPipeline.UpdateUbo(currentFrame, camera);
        var (lightCount, tileCountX, tileCountY) = PbrDeferredPipeline.UpdatePerFrame(currentFrame, camera, scene);
        transparentPipeline.UpdatePerFrame(currentFrame, camera, lightCount, tileCountX, tileCountY);
        // Stash for the LightCullPass body (it runs inside the FrameGraph, below).
        _frameLightCount = lightCount; _frameTileCountX = tileCountX; _frameTileCountY = tileCountY;
        // Skybox always updates — pass body is conditional on EditorState.SkyboxEnabled
        // so we can flip the toggle without re-recording the graph.
        skyboxPipeline.UpdatePerFrame(currentFrame, camera, ImGui.EditorState.SkyboxIntensity);

        // 5a-ter. Reflection-probe cluster cull. Tile-only today (zSlices=1);
        // when 3D froxels land the shader-side index math stays identical.
        // Cheap CPU work (~0.1ms for 16 probes × ~8000 tiles at 1080p).
        float aspect = (float)renderExtent.Width / renderExtent.Height;
        reflectionProbeSystem.BuildClusters(currentFrame, camera, aspect, 0.1f, 100f,
            tileCountX, tileCountY);
        // Refresh the per-probe SSBO read by the PBR lighting shader.
        reflectionProbeSystem.WriteProbeRecords(currentFrame);

        // 5a-bis. Reflection-probe capture for the next dirty probe (if any).
        // Bounded to one probe per frame; the actual draw recording happens
        // here so the captured cube is visible to any later sampling pass
        // within the same command buffer. No-op if no probe needs capturing
        // or if the capture shader wasn't compiled into ProbeCapture.spv.
        reflectionProbeSystem.RecordCapture(cmd, currentFrame, frameCounter, scene);

        // 5b. Scene extraction (L2 step 4) — the single ECS walk that packs the
        // cull-input SSBO + classifies BLEND candidates. Must precede the cull pass.
        // (The host-coherent write is visible to the cull dispatch on submit.)
        gpuScene.ExtractRenderables(currentFrame, scene);

        // 6. Record the deferred chain via the FrameGraph: cull -> light-cull -> geometry ->
        //    lighting -> skybox -> transparent -> tonemap. Cull/light-cull run as compute
        //    passes and every barrier is derived from the usage table.
        _deferredFrameGraph!.Execute(cmd, frameContext);
    }
    /// <summary>
    /// Uses RT in a compute shader
    /// </summary>
    private void DrawRayQueried(CommandBuffer cmd, FrameContext frameContext)
    {

    }

    /// <summary>
    /// Progressive ray-query pathtracer. Dispatches PTCompute.spv into
    /// ptAccumulator + ptOutColor, then blits ptOutColor into FinalColor so the
    /// viewport panel (which samples FinalColor) shows the result.
    ///
    /// Does NOT run the render graph — the deferred chain is bypassed entirely.
    /// FinalColor's layout tracker stays valid because the graph's _currentLayout
    /// settles at ShaderReadOnly per its declared finalLayout, and this method
    /// leaves FinalColor in that same state.
    /// </summary>
    private void DrawPathtraced(CommandBuffer cmd, FrameContext frameContext)
    {
        // Camera motion → restart accumulation. Cheap structural-equality check
        // against last frame's snapshot; only runs when PT is active. If the
        // camera class ever grows a built-in dirty flag, swap this out for that.
        if (camera != null)
        {
            var view = camera.GetViewMatrix();
            var pos  = camera.GetPosition();
            var fov  = camera.Fov;
            if (view != _ptLastCameraView || pos != _ptLastCameraPos || fov != _ptLastCameraFov)
            {
                MarkAccumulatorDirty();
                _ptLastCameraView = view;
                _ptLastCameraPos  = pos;
                _ptLastCameraFov  = fov;
            }
        }

        // Refresh per-frame SSBOs — lights AND materials. PT used to skip the
        // material upload (only the deferred geometry pass did it), which is
        // why inspector edits never landed in PT mode.
        uint lightCount = UpdateLights(currentFrame, scene);
        UpdateMaterials(currentFrame, scene);

        // Bridge Renderer's dirty signal into the pipeline's accumulator-reset.
        if (accumulatorDirty)
        {
            ptComputePipeline.MarkAccumulatorDirty();
            accumulatorDirty = false;
        }

        ptComputePipeline.UpdatePerFrame(currentFrame, camera, lightCount, renderExtent);

        //  Pre-dispatch barrier 
        // ptAccumulator: previous frame's compute read+write → this frame's same.
        // ptOutColor:    previous frame's tonemap fragment read (or first-frame
        //                Undefined→General) → this frame's compute write.
        // Both stay in General the whole time; tonemap reads ptOutColor in
        // ShaderReadOnly between dispatch and end-of-frame, but is re-stamped
        // back to General before the next dispatch lands here.
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
        vk!.CmdPipelineBarrier(cmd,
            PipelineStageFlags.ComputeShaderBit | PipelineStageFlags.FragmentShaderBit,
            PipelineStageFlags.ComputeShaderBit,
            0, 0, null, 0, null, 2, preBarriers);

        ptComputePipeline.Record(cmd, frameContext);

        // Tonemap into FinalColor
        // Tonemap's HDR input descriptor was rebound to ptOutColor at the top of
        // DrawFrame on the mode-change boundary. It expects the image in
        // ShaderReadOnly and writes FinalColor as a color attachment.
        TransitionImageLayout(cmd, ptOutColor.Image, ptOutColor._format,
            ImageLayout.General, ImageLayout.ShaderReadOnlyOptimal);

        var finalColor = renderTargets.FinalColor;
        TransitionImageLayout(cmd, finalColor.Image, finalColor._format,
            ImageLayout.ShaderReadOnlyOptimal, ImageLayout.ColorAttachmentOptimal);

        tonemapPipeline.Record(cmd, frameContext, finalColor.ImageView);

        TransitionImageLayout(cmd, finalColor.Image, finalColor._format,
            ImageLayout.ColorAttachmentOptimal, ImageLayout.ShaderReadOnlyOptimal);

        // ptOutColor back to General so next frame's dispatch + pre-barrier
        // path are layout-uniform.
        TransitionImageLayout(cmd, ptOutColor.Image, ptOutColor._format,
            ImageLayout.ShaderReadOnlyOptimal, ImageLayout.General);
    }

    /// <summary>
    /// RT-pipeline path tracer. Mirrors DrawPathtraced but dispatches via
    /// CmdTraceRays into the SAME ptAccumulator + ptOutColor images, then blits
    /// through the tonemap into FinalColor. Only reached when rtPipeline != null
    /// (RayTracePipelineSupported).
    /// </summary>
    private void DrawRayTraced(CommandBuffer cmd, FrameContext frameContext)
    {
        var rt = rtPipeline!;

        // Camera motion → restart accumulation (shared snapshot with the compute path).
        if (camera != null)
        {
            var view = camera.GetViewMatrix();
            var pos  = camera.GetPosition();
            var fov  = camera.Fov;
            if (view != _ptLastCameraView || pos != _ptLastCameraPos || fov != _ptLastCameraFov)
            {
                MarkAccumulatorDirty();
                _ptLastCameraView = view;
                _ptLastCameraPos  = pos;
                _ptLastCameraFov  = fov;
            }
        }

        uint lightCount = UpdateLights(currentFrame, scene);
        UpdateMaterials(currentFrame, scene);

        if (accumulatorDirty)
        {
            rt.MarkAccumulatorDirty();
            accumulatorDirty = false;
        }

        rt.UpdatePerFrame(currentFrame, camera, lightCount, renderExtent);

        //  Pre-dispatch barrier — same as the compute path but the writer stage
        //  is the ray-tracing stage rather than compute. Both images stay in
        //  General; tonemap reads ptOutColor in ShaderReadOnly between trace and
        //  end-of-frame, re-stamped to General before the next trace.
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
        vk!.CmdPipelineBarrier(cmd,
            PipelineStageFlags.RayTracingShaderBitKhr | PipelineStageFlags.FragmentShaderBit,
            PipelineStageFlags.RayTracingShaderBitKhr,
            0, 0, null, 0, null, 2, preBarriers);

        rt.Record(cmd, frameContext);

        // Tonemap into FinalColor (HDR input rebound to ptOutColor on the mode flip).
        TransitionImageLayout(cmd, ptOutColor.Image, ptOutColor._format,
            ImageLayout.General, ImageLayout.ShaderReadOnlyOptimal);

        var finalColor = renderTargets.FinalColor;
        TransitionImageLayout(cmd, finalColor.Image, finalColor._format,
            ImageLayout.ShaderReadOnlyOptimal, ImageLayout.ColorAttachmentOptimal);

        tonemapPipeline.Record(cmd, frameContext, finalColor.ImageView);

        TransitionImageLayout(cmd, finalColor.Image, finalColor._format,
            ImageLayout.ColorAttachmentOptimal, ImageLayout.ShaderReadOnlyOptimal);

        TransitionImageLayout(cmd, ptOutColor.Image, ptOutColor._format,
            ImageLayout.ShaderReadOnlyOptimal, ImageLayout.General);
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


    // The frame graph driving the deferred chain (render-graph.md Phase 1 / §1.8). It OWNS
    // the g-buffers / depth / HDR as transients (allocated in Compile, freed in Dispose) and
    // imports only FinalColor (RenderTargets-owned, consumed by the swapchain blit + ImGui
    // viewport). All sync is derived from the usage table. Rebuilt on resize.
    private DeferredGraph? _deferredFrameGraph;

    // Cached view of the FrameGraph's HDR transient, refreshed each BuildDeferredFrameGraph
    // (the transient is reallocated on every Compile). Tonemap's HDR-input descriptor is
    // bound to it there; the PT⇄deferred mode flip and the tonemap-operator rebuild read it
    // to restore the deferred source after pointing tonemap at ptOutColor.
    private ImageView _deferredHdrView;

    // Cached views of the FrameGraph's 5 g-buffer transients (pos/normal/albedo/material/
    // emissive), refreshed each BuildDeferredFrameGraph. The lighting set is bound to them
    // there; RebuildPbrPipelines also rebinds from these onto its fresh descriptor set
    // (Initialize no longer writes the g-buffer set — the views live on the graph).
    private readonly ImageView[] _deferredGBufferViews = new ImageView[5];

    // Per-frame values the LightCullPass body needs (computed in DrawDeferred from
    // PbrDeferredPipeline.UpdatePerFrame, before the graph executes). Stashed on the
    // renderer because the pass closures are built once but these change every frame.
    private uint _frameLightCount;
    private uint _frameTileCountX;
    private uint _frameTileCountY;

    // ---- Deferred FrameGraph debug surface (read by the Stats panel) -----------
    /// <summary>Last-frame per-pass GPU/CPU timings + counts, or null before first compile.</summary>
    public GraphStats? DeferredGraphStats => _deferredFrameGraph?.Stats;

    // ---- Resize-churn diagnostics (GpuMemoryAllocator retained-block check) -----
    private int _renderTargetRebuilds;
    /// <summary>Count of full render-target rebuilds since launch (one per resize).</summary>
    public int RenderTargetRebuilds => _renderTargetRebuilds;
    /// <summary>Live allocator occupancy — reserved (held from driver) vs. actually used.</summary>
    public AllocatorStats GpuMemoryStats => gfx.Allocator.GetStats();

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
    /// <summary>Graphviz dump of the compiled deferred graph (for "Copy DOT").</summary>
    public string DeferredGraphDot() => _deferredFrameGraph?.ToDot() ?? "(no deferred frame graph)";
    /// <summary>Runtime toggle for the deferred graph's pipeline-statistics collection.</summary>
    public bool DeferredGraphPipelineStats
    {
        get => _deferredFrameGraph?.CollectPipelineStats ?? false;
        set { if (_deferredFrameGraph != null) _deferredFrameGraph.CollectPipelineStats = value; }
    }

    /// <summary>
    /// (Re)builds the deferred chain as the <see cref="DeferredGraph"/>. The graph OWNS the
    /// g-buffers / depth / HDR as transients (allocated in Compile) and imports only
    /// FinalColor (RenderTargets-owned, consumed outside the graph). Must be called after the
    /// pipelines it binds exist (PbrDeferred / tonemap, for the post-Compile descriptor
    /// rebinds) and again after every <see cref="RebuildRenderTargets"/> (fresh extent →
    /// fresh transients). Pass bodies are closures over the renderer's pipelines — they only
    /// need to exist by the time Execute runs, not now.
    /// </summary>
    private void BuildDeferredFrameGraph()
    {
        _deferredFrameGraph?.Dispose();
        var fg  = new DeferredGraph(gfx);
        var ext = renderExtent;

        // Deferred intermediates the graph OWNS as transients: it allocates their VkImages in
        // Compile() and frees them in Dispose(). Produced and consumed entirely within the
        // chain (geometry writes; lighting/skybox/transparent/tonemap read), so nothing
        // outside needs a stable handle. Seeded Undefined each frame (a legal discard — every
        // producing pass fully regenerates its target).
        GraphImage Color(string name, Format fmt) => fg.CreateImage(new ImageDesc
        {
            Format = fmt, Extent = ext, Mips = 1, Layers = 1,
            Usage = ImageUsageFlags.ColorAttachmentBit | ImageUsageFlags.SampledBit,
        }, name);

        var pos      = Color("GBuffer_Position", Format.R32G32B32A32Sfloat);
        var normal   = Color("GBuffer_Normal",   Format.R32G32B32A32Sfloat);
        var albedo   = Color("GBuffer_Albedo",   Format.R8G8B8A8Unorm);
        var material = Color("GBuffer_Material",  Format.R8G8B8A8Unorm);
        var emissive = Color("GBuffer_Emissive", Format.R16G16B16A16Sfloat);
        var hdr      = Color("HDRColor",          Format.R16G16B16A16Sfloat);
        var depth    = fg.CreateImage(new ImageDesc
        {
            Format = Format.D32Sfloat, Extent = ext, Mips = 1, Layers = 1,
            Usage = ImageUsageFlags.DepthStencilAttachmentBit,
        }, "Depth");

        // FinalColor is the one IMPORTED image: the swapchain blit + ImGui viewport sample it,
        // so RenderTargets owns it and the graph just adopts the handle, handing it back in
        // ShaderReadOnly each frame. Undefined incoming = discard (tonemap rewrites every pixel).
        var finalRes  = renderTargets.FinalColor;
        var finalDesc = new ImageDesc { Format = finalRes._format, Mips = 1, Layers = 1 };
        var final = fg.ImportImage(finalRes.Image, finalRes.ImageView, in finalDesc,
            ImageLayout.Undefined, "FinalColor", ImageLayout.ShaderReadOnlyOptimal);

        // Per-frame compute-output buffers, imported so the graph derives the cull→geometry
        // and light-cull→lighting barriers + ordering. (The compute INPUTS — renderables and
        // lights — are host-coherent mapped writes, visible on submit, so they need no graph
        // barrier and aren't declared.) BufferDesc is unused for imports.
        var indirectCmdF   = new Buffer[MAX_CONCURRENT_FRAMES];
        var indirectCountF = new Buffer[MAX_CONCURRENT_FRAMES];
        var instanceF      = new Buffer[MAX_CONCURRENT_FRAMES];
        var tileCountF     = new Buffer[MAX_CONCURRENT_FRAMES];
        var tileIndicesF   = new Buffer[MAX_CONCURRENT_FRAMES];
        for (uint i = 0; i < MAX_CONCURRENT_FRAMES; i++)
        {
            indirectCmdF[i]   = drawCullPipeline.GetIndirectCmdBuffer(i);
            indirectCountF[i] = drawCullPipeline.GetIndirectCountBuffer(i);
            instanceF[i]      = Engine.ResourceManager.GetInstanceBuffer(i);
            tileCountF[i]     = lightCullPipeline.GetTileLightCountBuffer(i);
            tileIndicesF[i]   = lightCullPipeline.GetTileLightIndicesBuffer(i);
        }
        var indirectCmd   = fg.ImportBufferPerFrame(indirectCmdF,   default, "IndirectCmd");
        var indirectCount = fg.ImportBufferPerFrame(indirectCountF, default, "IndirectCount");
        var instance      = fg.ImportBufferPerFrame(instanceF,      default, "InstanceData");
        var tileCount     = fg.ImportBufferPerFrame(tileCountF,     default, "TileLightCount");
        var tileIndices   = fg.ImportBufferPerFrame(tileIndicesF,   default, "TileLightIndices");

        // Cull (compute): frustum-tests renderables → post-cull indirect draw commands +
        // instance data. Its declared writes both keep it alive (writes imported buffers)
        // and order it before geometry (RAW on the indirect + instance buffers). Record()
        // also runs the view-dependent transparent sort consumed by TransparentPass.
        fg.AddPass("CullPass", PassType.Compute, QueueClass.Graphics,
            b =>
            {
                indirectCmd   = b.Write(indirectCmd,   ResourceUsage.StorageWriteCompute);
                indirectCount = b.Write(indirectCount, ResourceUsage.StorageWriteCompute);
                instance      = b.Write(instance,      ResourceUsage.StorageWriteCompute);
            },
            (CommandBuffer cmd, PassResources res, in FrameContext f) =>
                drawCullPipeline.Record(cmd, f.FrameIndex, f.Camera));

        // Light-cull (compute): bins lights into the per-tile lists the lighting FS reads.
        // Tile counts/light count are computed in DrawDeferred and stashed on the renderer.
        fg.AddPass("LightCullPass", PassType.Compute, QueueClass.Graphics,
            b =>
            {
                tileCount   = b.Write(tileCount,   ResourceUsage.StorageWriteCompute);
                tileIndices = b.Write(tileIndices, ResourceUsage.StorageWriteCompute);
            },
            (CommandBuffer cmd, PassResources res, in FrameContext f) =>
                lightCullPipeline.Record(cmd, f.FrameIndex, f.Camera,
                    _frameLightCount, _frameTileCountX, _frameTileCountY));

        // Geometry → g-buffers + depth. Reads the post-cull indirect buffers (IndirectArg)
        // + instance data (vertex storage read) → RAW edges order it after CullPass.
        fg.AddPass("GeometryPass", PassType.Graphics, QueueClass.Graphics,
            b =>
            {
                b.Read(indirectCmd,   ResourceUsage.IndirectArg);
                b.Read(indirectCount, ResourceUsage.IndirectArg);
                b.Read(instance,      ResourceUsage.StorageReadVertex);
                pos      = b.Write(pos,      ResourceUsage.ColorAttachment);
                normal   = b.Write(normal,   ResourceUsage.ColorAttachment);
                albedo   = b.Write(albedo,   ResourceUsage.ColorAttachment);
                material = b.Write(material, ResourceUsage.ColorAttachment);
                emissive = b.Write(emissive, ResourceUsage.ColorAttachment);
                depth    = b.Write(depth,    ResourceUsage.DepthAttachment);
            },
            (CommandBuffer cmd, PassResources res, in FrameContext f) =>
            {
                var indirectCmd   = drawCullPipeline.GetIndirectCmdBuffer(f.FrameIndex);
                var indirectCount = drawCullPipeline.GetIndirectCountBuffer(f.FrameIndex);
                var drawCount     = drawCullPipeline.LastRenderableCount;
                var attachments = new GeometryPipeline.Attachments(
                    res.View(pos), res.View(normal), res.View(albedo),
                    res.View(material), res.View(emissive), res.View(depth));
                geometryPipeline.Record(cmd, f, indirectCmd, indirectCount, drawCount, attachments);
            });

        // Lighting samples the five g-buffers (ShaderReadOnly) and writes HDRColor@v1. It
        // does NOT bind depth — the deferred lighting pass reconstructs position from the
        // g-buffer, so depth is left untouched in DepthStencilAttachmentOptimal for the
        // skybox/transparent depth tests. (The legacy graph listed Depth as a lighting
        // input, which forced a pointless Attachment->ShaderReadOnly->Attachment round-trip.)
        fg.AddPass("LightingPass", PassType.Graphics, QueueClass.Graphics,
            b =>
            {
                b.Read(pos,      ResourceUsage.SampledFragment);
                b.Read(normal,   ResourceUsage.SampledFragment);
                b.Read(albedo,   ResourceUsage.SampledFragment);
                b.Read(material, ResourceUsage.SampledFragment);
                b.Read(emissive, ResourceUsage.SampledFragment);
                // Per-tile light lists from LightCullPass, read by the lighting FS → RAW
                // edge orders lighting after light-cull.
                b.Read(tileCount,   ResourceUsage.StorageReadFragment);
                b.Read(tileIndices, ResourceUsage.StorageReadFragment);
                hdr = b.Write(hdr, ResourceUsage.ColorAttachment);
            },
            (CommandBuffer cmd, PassResources res, in FrameContext f) =>
                PbrDeferredPipeline.Record(cmd, f, res.View(hdr)));

        // Skybox / Transparent both LOAD HDRColor and depth-test (DepthWriteEnable=false)
        // against the geometry depth. Their CmdBeginRendering binds depth as a
        // DepthStencilAttachmentOptimal attachment, so depth is declared DepthAttachment
        // here (matching that layout) — not DepthRead/ReadOnlyOptimal, which would mismatch
        // the attachment layout. The depth WAW chain geometry→skybox→transparent and the
        // HDRColor version chain Lighting→Skybox→Transparent both fall out of the ledger.
        fg.AddPass("SkyboxPass", PassType.Graphics, QueueClass.Graphics,
            b =>
            {
                hdr   = b.Write(hdr,   ResourceUsage.ColorAttachment);
                depth = b.Write(depth, ResourceUsage.DepthAttachment);
            },
            (CommandBuffer cmd, PassResources res, in FrameContext f) =>
                skyboxPipeline.Record(cmd, f, new SkyboxPipeline.Attachments(res.View(hdr), res.View(depth))));

        fg.AddPass("TransparentPass", PassType.Graphics, QueueClass.Graphics,
            b =>
            {
                hdr   = b.Write(hdr,   ResourceUsage.ColorAttachment);
                depth = b.Write(depth, ResourceUsage.DepthAttachment);
            },
            (CommandBuffer cmd, PassResources res, in FrameContext f) =>
                transparentPipeline.Record(cmd, f, drawCullPipeline.LastTransparentDraws,
                    new TransparentPipeline.Attachments(res.View(hdr), res.View(depth))));

        // Tonemap reads HDRColor@v3, writes FinalColor.
        fg.AddPass("TonemapPass", PassType.Graphics, QueueClass.Graphics,
            b =>
            {
                b.Read(hdr, ResourceUsage.SampledFragment);
                final = b.Write(final, ResourceUsage.ColorAttachment);
            },
            (CommandBuffer cmd, PassResources res, in FrameContext f) =>
                tonemapPipeline.Record(cmd, f, res.View(final)));

        fg.MarkOutput(final);
        fg.Compile();
        _deferredFrameGraph = fg;

        // Compile() just (re)allocated the g-buffer + HDR transients. (Re)bind the only graph
        // resources read through a PRE-BAKED descriptor set rather than a per-pass res.View:
        // the lighting pass's 5 g-buffer samplers and tonemap's HDR input. (Geometry / skybox
        // / transparent / tonemap reach their attachments via res.View, which resolves the
        // fresh handle automatically.) The cached HDR view also feeds the PT <->deferred and
        // tonemap-operator rebind paths.
        _deferredGBufferViews[0] = fg.ResolveView(pos);
        _deferredGBufferViews[1] = fg.ResolveView(normal);
        _deferredGBufferViews[2] = fg.ResolveView(albedo);
        _deferredGBufferViews[3] = fg.ResolveView(material);
        _deferredGBufferViews[4] = fg.ResolveView(emissive);
        PbrDeferredPipeline.WriteGBufferDescriptors(
            _deferredGBufferViews[0], _deferredGBufferViews[1], _deferredGBufferViews[2],
            _deferredGBufferViews[3], _deferredGBufferViews[4]);
        _deferredHdrView = fg.ResolveView(hdr);
        tonemapPipeline.WriteHdrInputDescriptor(_deferredHdrView, gBufferSampler);
    }






    
    
    
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
