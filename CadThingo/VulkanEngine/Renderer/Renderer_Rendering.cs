using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using CadThingo.VulkanEngine.GLTF;
using CadThingo.VulkanEngine.Renderer.Pipelines;
using Silk.NET.Core.Native;
using Silk.NET.Vulkan;
using Silk.NET.Vulkan.Extensions.KHR;

namespace CadThingo.VulkanEngine.Renderer;

public unsafe partial class Renderer
{
    private void CreateSwapChain()
    {
        //query swapchain support
        SwapChainSupportDetails swapChainSupport = QuerySwapChainSupport(physicalDevice);
        
        //choose swapsurface format
        SurfaceFormatKHR surfaceFormat = ChooseSwapSurfaceFormat(swapChainSupport.Formats);
        PresentModeKHR presentMode = ChooseSwapPresentMode(swapChainSupport.PresentModes);
        Extent2D extent = ChooseSwapExtent(swapChainSupport.Capabilities);
        
        
        
        //choose image count
        uint imageCount = swapChainSupport.Capabilities.MinImageCount + 1;
        if (swapChainSupport.Capabilities.MaxImageCount > 0 && imageCount > swapChainSupport.Capabilities.MaxImageCount)
        {
            imageCount = swapChainSupport.Capabilities.MaxImageCount;
        }
        
        //create swapchain info
        SwapchainCreateInfoKHR createInfo = new()
        {
            SType = StructureType.SwapchainCreateInfoKhr,
            Surface = surface,
            MinImageCount = imageCount,
            ImageFormat = surfaceFormat.Format,
            ImageColorSpace = surfaceFormat.ColorSpace,
            ImageExtent = extent,
            ImageArrayLayers = 1,
            ImageUsage = ImageUsageFlags.ColorAttachmentBit | ImageUsageFlags.TransferDstBit,
            PreTransform = swapChainSupport.Capabilities.CurrentTransform,
            CompositeAlpha = CompositeAlphaFlagsKHR.OpaqueBitKhr,
            PresentMode = presentMode,
            Clipped = true,
            OldSwapchain = default
        };
        //set sharing mode
        uint?[] queueFamilyIndiciesLoc =
        {
            queueFamilyIndices.graphicsFamily,
            queueFamilyIndices.presentFamily
        };
        
        if (queueFamilyIndices.graphicsFamily != queueFamilyIndices.presentFamily)
        {
            createInfo.ImageSharingMode = SharingMode.Concurrent;
            createInfo.QueueFamilyIndexCount = (uint)queueFamilyIndiciesLoc.Length;
            fixed (uint?* pQueueFamilyIndicesLoc = &queueFamilyIndiciesLoc[0])
                createInfo.PQueueFamilyIndices = (uint*)pQueueFamilyIndicesLoc;
        }
        else
        {
            createInfo.ImageSharingMode = SharingMode.Exclusive;
            createInfo.QueueFamilyIndexCount = 0;
            createInfo.PQueueFamilyIndices = null;
        }
        
        //create swapchain
        if (!vk!.TryGetDeviceExtension(instance, device, out swapChainKhr))
        {
            throw new Exception("Failed to load swapchain extension");
        }

        if (swapChainKhr.CreateSwapchain(device, &createInfo, null, out swapChain) != Result.Success)
        {
            throw new Exception("Failed to create swapchain");
        }
        
        //Get swapchain images
        swapChainKhr.GetSwapchainImages(device, swapChain, &imageCount, null);
        swapChainImages = new Image[imageCount];
        fixed (Image* imagesPtr = swapChainImages)
        {
            swapChainKhr.GetSwapchainImages(device, swapChain, &imageCount, imagesPtr);
        }
        
        //swapchain images start with no layout
        var imageLayouts = new ImageLayout[imageCount];
        for (var i = 0; i < imageCount; i++) imageLayouts[i] = ImageLayout.Undefined;
        swapChainImageLayouts = imageLayouts;
        
        //format and extent
        swapChainImageFormat = surfaceFormat.Format;
        swapChainExtent = extent;
    }

    private void CreateImageViews()
    {
        swapChainImageViews = new ImageView[swapChainImages.Length];

        ImageViewCreateInfo viewInfo = new()
        {
            SType = StructureType.ImageViewCreateInfo,
            ViewType = ImageViewType.Type2D,
            Format = swapChainImageFormat,
            Components = new ComponentMapping()
            {
                R = ComponentSwizzle.Identity,
                G = ComponentSwizzle.Identity,
                B = ComponentSwizzle.Identity,
                A = ComponentSwizzle.Identity
            },
            SubresourceRange = new ImageSubresourceRange()
            {
                AspectMask = ImageAspectFlags.ColorBit,
                BaseMipLevel = 0,
                LevelCount = 1,
                BaseArrayLayer = 0,
                LayerCount = 1
            }
        };
        //create imageview for each image
        for (var i = 0; i < swapChainImages.Length; i ++)
        {
            viewInfo.Image = swapChainImages[i];
            vk!.CreateImageView(device, &viewInfo, null, out swapChainImageViews[i]);
        }
        
    }

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
    
    private void CreateCommandPool()
    {
        //Create command pool info
        CommandPoolCreateInfo poolCreateInfo = new()
        {
            SType = StructureType.CommandPoolCreateInfo,
            Flags = CommandPoolCreateFlags.ResetCommandBufferBit,
            QueueFamilyIndex = queueFamilyIndices.graphicsFamily!.Value
        };

        if (vk!.CreateCommandPool(device, &poolCreateInfo, null, out commandPool) != Result.Success)
        {
            throw new Exception("Failed to create command pool");
        }
        
        
    }

    private void CreateCommandBuffers()
    {
        commandBuffers = new CommandBuffer[swapChainImages!.Length];
        CommandBufferAllocateInfo allocateInfo = new()
        {
            SType = StructureType.CommandBufferAllocateInfo,
            CommandPool = commandPool,
            Level = CommandBufferLevel.Primary,
            CommandBufferCount = 1
        };
        

        for (int i = 0; i < commandBuffers.Length; i++)
        {
            if (vk!.AllocateCommandBuffers(device, &allocateInfo, out commandBuffers[i]) != Result.Success)
            {
                throw new Exception("Failed to allocate command buffers");
            } 
        }
    }

    private void CreateSyncObjects()
    {
        imageAvailableSemaphores = new Semaphore[MAX_CONCURRENT_FRAMES];
        renderFinishedSemaphores = new Semaphore[MAX_CONCURRENT_FRAMES];
        inFlightFences = new Fence[MAX_CONCURRENT_FRAMES];
        
        SemaphoreCreateInfo semaphoreCreateInfo = new()
        {
            SType = StructureType.SemaphoreCreateInfo,
        };
        FenceCreateInfo fenceCreateInfo = new()
        {
            SType = StructureType.FenceCreateInfo,
            Flags = FenceCreateFlags.SignaledBit
        };

        for (var i = 0; i < MAX_CONCURRENT_FRAMES; i++)
        {
            if (vk!.CreateSemaphore(device, &semaphoreCreateInfo, null, out imageAvailableSemaphores[i]) != Result.Success ||
                vk!.CreateSemaphore(device, &semaphoreCreateInfo, null, out renderFinishedSemaphores[i]) != Result.Success ||
                vk!.CreateFence(device, &fenceCreateInfo, null, out inFlightFences[i]) != Result.Success)
            {
                throw new Exception("Failed to create synchronization objects for a frame");
            }
        }
    }


    // Tears down only the swap chain itself + its image views. Size-dependent
    // attachments (depth, g-buffers, FinalColor) are owned elsewhere and torn
    // down by RecreateSwapChain / Cleanup.
    private void CleanupSwapChain()
    {
        if (swapChainImageViews != null)
        {
            foreach (var iv in swapChainImageViews)
                if (iv.Handle != 0) vk!.DestroyImageView(device, iv, null);
            swapChainImageViews = Array.Empty<ImageView>();
        }

        if (swapChain.Handle != 0)
        {
            swapChainKhr.DestroySwapchain(device, swapChain, null);
            swapChain = default;
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

        CleanupSwapChain();
        CreateSwapChain();
        CreateImageViews();

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
        // Order matters: render graph holds references to depth + g-buffer
        // ImageResources, so dispose it first. The ImageResource Dispose guard
        // makes the explicit calls below idempotent.
        scene.renderGraph.Dispose();
        depthImageResource.Dispose();
        gBufferPosition.Dispose();
        gBufferNormal.Dispose();
        gBufferAlbedo.Dispose();
        gBufferMaterial.Dispose();
        gBufferEmissive.Dispose();

        // Pathtracer storage images live outside the graph but share the
        // renderExtent — recreate them too so per-pixel work stays in sync.
        ptAccumulator?.Dispose();
        ptOutColor?.Dispose();

        renderExtent = new Extent2D(width, height);

        CreateDepthResources();
        CreateGBufferResources();
        CreatePathTracingResources(renderExtent.Width, renderExtent.Height);

        scene.renderGraph = new RenderGraph(vk!, device, physicalDevice);
        SetupDeferredRenderer(scene.renderGraph, renderExtent.Width, renderExtent.Height);

        // G-buffer ImageViews are fresh — re-bind them on the lighting pass set.
        PbrDeferredPipeline.WriteGBufferDescriptors();

        // HDRColor ImageView is also fresh after the graph rebuild — re-bind it on
        // the tonemap pass's sampler binding.
        tonemapPipeline.WriteHdrInputDescriptor(
            scene.renderGraph.GetResource("HDRColor").ImageView, gBufferSampler);

        // FinalColor ImageView is fresh — re-bind it for the ImGui viewport panel.
        imGuiUtils?.WriteViewportDescriptor(scene.renderGraph.GetResource("FinalColor").ImageView);

        // PT storage image views are fresh too. WriteStorageImageDescriptors calls
        // MarkAccumulatorDirty internally so the next dispatch clears the
        // (also-fresh) accumulator instead of reading garbage memory.
        ptComputePipeline?.WriteStorageImageDescriptors(ptAccumulator.ImageView, ptOutColor.ImageView);
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

        // 0c. Render-mode change: rebind tonemap's HDR input so its single
        //     shared descriptor set points at the right source. DeviceWaitIdle
        //     ensures no in-flight frame is still using the old binding.
        //     Mode flips are user-driven (ImGui combo), so the hitch is
        //     acceptable; per-frame double-buffered descriptors would avoid it.
        if (renderMode != _lastRenderMode)
        {
            vk!.DeviceWaitIdle(device);
            ImageView source = renderMode == RenderMode.RayCompute
                ? ptOutColor.ImageView
                : scene.renderGraph.GetResource("HDRColor").ImageView;
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
        uint imageIndex = 0;
        var acquireResult = swapChainKhr.AcquireNextImage(device, swapChain, ulong.MaxValue,
            imageAvailableSemaphores[currentFrame], default, &imageIndex);
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
                // RT pipeline path not implemented — fall through to deferred so the
                // viewport stays alive.
                DrawDeferred(cmd, ctx);
                break;
            default:
                DrawDeferred(cmd, ctx);
                break;
        }

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
        var finalColor = scene.renderGraph.GetResource("FinalColor");
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
        fixed (SwapchainKHR* pSwap = &swapChain)
        {
            var presentSig = renderFinishedSemaphores[currentFrame];
            var presentInfo = new PresentInfoKHR
            {
                SType = StructureType.PresentInfoKhr,
                WaitSemaphoreCount = 1,
                PWaitSemaphores    = &presentSig,
                SwapchainCount     = 1,
                PSwapchains        = pSwap,
                PImageIndices      = &imageIndex,
            };
            swapChainKhr.QueuePresent(presentQueue, &presentInfo);
        }

        currentFrame = (currentFrame + 1) % MAX_CONCURRENT_FRAMES;
        frameCounter++;
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

        geometryPipeline.UpdateUbo(currentFrame, camera);
        var (lightCount, tileCountX, tileCountY) = PbrDeferredPipeline.UpdatePerFrame(currentFrame, camera, scene);
        transparentPipeline.UpdatePerFrame(currentFrame, camera, lightCount, tileCountX, tileCountY);
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

        // 5b. GPU cull pass — runs before the geometry pass and produces the
        // indirect command + post-cull instance buffers the geometry pass consumes.
        drawCullPipeline.Record(cmd, currentFrame, camera, scene);

        // 5c. Tiled light-cull — bins lights into per-tile slot lists for the
        // lighting fragment shader. Buffer barrier inside makes the writes
        // visible to the fragment stage of the lighting pass.
        lightCullPipeline.Record(cmd, currentFrame, camera, lightCount, tileCountX, tileCountY);

        // 6. Record all render-graph passes (geometry → lighting), record-only.
        scene.renderGraph.Execute(cmd, frameContext);
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
            if (view != _ptLastCameraView || pos != _ptLastCameraPos)
            {
                MarkAccumulatorDirty();
                _ptLastCameraView = view;
                _ptLastCameraPos  = pos;
            }
        }

        // Refresh the per-frame lights SSBO — renderer-owned, shared with
        // every rendering path. No tile-cull or PbrDeferred-UBO work needed
        // for the pathtracer.
        uint lightCount = UpdateLights(currentFrame, scene);

        // Bridge Renderer's dirty signal into the pipeline's accumulator-reset.
        if (accumulatorDirty)
        {
            ptComputePipeline.MarkAccumulatorDirty();
            accumulatorDirty = false;
        }

        ptComputePipeline.UpdatePerFrame(currentFrame, camera, lightCount, renderExtent);

        // ── Pre-dispatch barrier ──────────────────────────────────────────────
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

        // ── Dispatch ──────────────────────────────────────────────────────────
        ptComputePipeline.Record(cmd, frameContext);

        // ── Tonemap into FinalColor ───────────────────────────────────────────
        // Tonemap's HDR input descriptor was rebound to ptOutColor at the top of
        // DrawFrame on the mode-change boundary. It expects the image in
        // ShaderReadOnly and writes FinalColor as a color attachment.
        TransitionImageLayout(cmd, ptOutColor.Image, ptOutColor._format,
            ImageLayout.General, ImageLayout.ShaderReadOnlyOptimal);

        var finalColor = scene.renderGraph.GetResource("FinalColor");
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
    /// placeholder for future RTPipeline implementation<br/>
    /// </summary>
    private void DrawRayTraced()
    {

    }
    
    
    /// <summary>
    /// Comprehensive deferred renderer setup demonstrating rendergraph resource management<br/>
    /// this implementation shows how to efficiently organise resources for multiple passes<br/>
    /// </summary>
    /// <param name="graph"></param>
    /// <param name="width"></param>
    /// <param name="height"></param>
    private void SetupDeferredRenderer(RenderGraph graph, uint width, uint height)
    {
        //configure posiiton buffer for world-space vertex positions
        //High precision format preserves positional accuracy for lighting calculations
        graph.AddResource(gBufferPosition);
        
        
        //Configure normal buffer for surface orientation data
        //High precision normals enable accurate lighting and reflections
        graph.AddResource(gBufferNormal);
        
        
        //configure albedo buffer for surface color information
        //standard 8bit precision sufficient for color data with gamma encoding
        graph.AddResource(gBufferAlbedo);
        
        
        graph.AddResource(gBufferMaterial);
        // Emissive — sampled by the lighting pass and added to the final color before tone-mapping.
        graph.AddResource(gBufferEmissive);
        //configure depth buffer for accurate depth information
        //standard 32bit depth format preserves accurate depth information
        graph.AddResource(depthImageResource);
        
        // HDR linear scene target — produced by the lighting pass (and later the
        // forward+ transparent pass). Consumed by the tone-map pass which writes
        // the LDR FinalColor. SampledBit lets the tonemap pass bind it as a texture.
        graph.AddResource("HDRColor", Format.R16G16B16A16Sfloat, new Extent2D(width, height),
            ImageUsageFlags.ColorAttachmentBit | ImageUsageFlags.SampledBit,
            ImageLayout.Undefined, ImageLayout.ShaderReadOnlyOptimal);

        //configure finalcolor buffer for the completed lighting results
        // SampledBit lets the ImGui viewport panel bind FinalColor as a CombinedImageSampler.
        // finalLayout settles to ShaderReadOnlyOptimal each frame so ImGui draws sample valid
        // contents — the per-frame blit to the swapchain transitions in and out manually.
        // TransferDstBit lets the pathtracer path blit its ptOutColor into FinalColor —
        // the viewport panel samples FinalColor regardless of render mode, so PT
        // pixels need to land here for the user to see them.
        graph.AddResource("FinalColor", Format.R8G8B8A8Unorm, new Extent2D(width, height),
            ImageUsageFlags.ColorAttachmentBit | ImageUsageFlags.TransferSrcBit | ImageUsageFlags.TransferDstBit | ImageUsageFlags.SampledBit,
            ImageLayout.Undefined, ImageLayout.ShaderReadOnlyOptimal);
        
        //set names of FInal and hdr color for debug
        
        //configure geometry pass for G-buffer population
        //this pass renders all geometry and stores intermediate data for lighting calculations
        graph.AddPass("GeometryPass", default,
            new List<string> { "GBuffer_Position", "GBuffer_Normal", "GBuffer_Albedo", "GBuffer_Material", "GBuffer_Emissive", "Depth" },
            (buffer, frameContext) => {
                var indirectCmd = drawCullPipeline.GetIndirectCmdBuffer(frameContext.FrameIndex);
                var indirectCount = drawCullPipeline.GetIndirectCountBuffer(frameContext.FrameIndex);
                var drawCount = drawCullPipeline.LastRenderableCount;
                var attachments = new GeometryPipeline.Attachments(
                    position:  graph.GetResource("GBuffer_Position").ImageView,
                    normal:    graph.GetResource("GBuffer_Normal").ImageView,
                    albedo:    graph.GetResource("GBuffer_Albedo").ImageView,
                    material:  graph.GetResource("GBuffer_Material").ImageView,
                    emissive:  graph.GetResource("GBuffer_Emissive").ImageView,
                    depth:     graph.GetResource("Depth").ImageView);
                
                geometryPipeline.Record(buffer, frameContext, indirectCmd, indirectCount, drawCount, attachments);
            });
        
        
        graph.AddPass("LightingPass", new List<string>{"GBuffer_Position", "GBuffer_Normal", "GBuffer_Albedo", "GBuffer_Material", "GBuffer_Emissive", "Depth"},
            new List<string>{"HDRColor"}, (buffer, frameContext) =>
            {
                PbrDeferredPipeline.Record(buffer, frameContext, graph.GetResource("HDRColor").ImageView);
            });

        // Skybox pass — runs between lighting and transparent. Depth-tested
        // EQUAL @ 1.0, no depth write; writes only to pixels where the geometry
        // pass left the cleared far-plane depth (i.e. true sky). PBR shader
        // emits black for those same pixels via the WorldPos.w sky-gate, so
        // here LOAD_OP_LOAD preserves the lit framebuffer and the skybox just
        // overwrites the black sky.
        graph.AddPass("SkyboxPass", new List<string>(),
            new List<string>{"HDRColor", "Depth"}, (buffer, frameContext) =>
            {
                SkyboxPipeline.Attachments attachments = new(
                    hdrColor: graph.GetResource("HDRColor").ImageView,
                    depth: graph.GetResource("Depth").ImageView
                );
                skyboxPipeline.Record(buffer, frameContext, attachments);
            });

        // Forward+ transparent pass — blends BLEND-mode materials into HDRColor on
        // top of the deferred lighting result, depth-tested LE against the geometry
        // pass's Depth (no depth write). Declares HDRColor + Depth as writes so the
        // graph's writer→writer edges serialize this after LightingPass + GeometryPass
        // and the current-layout tracker preserves HDR contents for blending.
        // Body is a stub for stage 3 — pipeline + attachments bind but no draws.
        graph.AddPass("TransparentPass", new List<string>(),
            new List<string>{"HDRColor", "Depth"}, (buffer, frameContext) =>
            {
                TransparentPipeline.Attachments attachments = new(
                    hdrColor: graph.GetResource("HDRColor").ImageView,
                    depth: graph.GetResource("Depth").ImageView);
                transparentPipeline.Record(buffer, frameContext, drawCullPipeline.LastTransparentDraws, attachments);
            });

        // Post-process: sample HDRColor, apply exposure + Reinhard + inverse gamma,
        // write LDR into FinalColor. FinalColor's graph-declared finalLayout
        // (TransferSrcOptimal) is honoured by the post-pass barrier, keeping
        // the existing blit-to-swapchain path unchanged.
        graph.AddPass("TonemapPass", new List<string>{"HDRColor"},
            new List<string>{"FinalColor"}, (buffer, frameContext) =>
            {
                tonemapPipeline.Record(buffer, frameContext, graph.GetResource("FinalColor").ImageView);
            });

        graph.Compile();
    }


    private void CreatePathTracingResources(uint width, uint height)
    {
        // ptOutColor needs TransferSrcBit so DrawPathtraced can blit it into
        // FinalColor (the viewport's source) at the end of the dispatch.
        ptAccumulator = new ImageResource(vk, device, "accumulator", Format.R32G32B32A32Sfloat,
            new Extent2D(width, height),
            ImageUsageFlags.StorageBit | ImageUsageFlags.SampledBit,
            ImageLayout.Undefined, ImageLayout.General);

        ptOutColor = new ImageResource(vk, device, "outColor", Format.R32G32B32A32Sfloat,
            new Extent2D(width, height),
            ImageUsageFlags.StorageBit | ImageUsageFlags.SampledBit | ImageUsageFlags.TransferSrcBit,
            ImageLayout.Undefined, ImageLayout.General);

        // Render graph normally calls Allocate inside Compile — these images live
        // outside the graph, so allocate explicitly.
        ptAccumulator.Allocate(physicalDevice);
        ptOutColor.Allocate(physicalDevice);

        // One-shot Undefined → General transition so the very first dispatch can
        // do imageStore without a per-frame "first use" branch. Subsequent
        // dispatches keep both images in General between frames.
        var cmd = BeginSingleTimeCommands();
        TransitionImageLayout(cmd, ptAccumulator.Image, ptAccumulator._format,
            ImageLayout.Undefined, ImageLayout.General);
        TransitionImageLayout(cmd, ptOutColor.Image, ptOutColor._format,
            ImageLayout.Undefined, ImageLayout.General);
        EndSingleTimeCommands(cmd);
    }

    


    
    
    
    /// <summary>
    /// 
    /// </summary>
   
}

/// <summary>
/// Workflow<br/>
/// 1. AddResource() - declare all images the graph will use <br/>
/// 2. AddPass() - declare all passes and there read/write sets<br/>
/// 3. Compile() - topological sort + allocate gpu resources<br/>
/// 4. Execute() - record all passes into the command buffer in order<br/>
/// 5. Dispose() - free all gpu resources <br/>
/// </summary>
public unsafe class RenderGraph : IDisposable
{
    private readonly Vk _vk;
    private readonly Device _device;
    private readonly PhysicalDevice _physicalDevice;
    private bool _disposed;
    private bool _compiled;

    private readonly Dictionary<string, ImageResource> _imageResources = new();
    private List<Pass> passes;
    private List<int> executionOrder;

    // Tracks each resource's current Vulkan layout across passes within a frame and
    // across frames. Source layout for every transition barrier comes from this dict
    // so blends / post-process passes that consume previous-pass output read the
    // correct contents rather than racing an Undefined→X discard.
    private readonly Dictionary<string, ImageLayout> _currentLayout = new();

    public RenderGraph(Vk vk, Device device, PhysicalDevice physicalDevice)
    {
        _vk = vk;
        _device = device;
        _physicalDevice = physicalDevice;
        passes = new();
        executionOrder = new();
    }


    /// <summary>
    /// Creates a new image resource and adds it to the render graph
    /// </summary>
    /// <param name="name"></param>
    /// <param name="format"></param>
    /// <param name="extent"></param>
    /// <param name="usage"></param>
    /// <param name="initialLayout"></param>
    /// <param name="finalLayout"></param>
    public void AddResource(string name, Format format, Extent2D extent, ImageUsageFlags usage,
        ImageLayout initialLayout = ImageLayout.Undefined,
        ImageLayout finalLayout = ImageLayout.ShaderReadOnlyOptimal)
    {
        ImageResource resource = new(_vk, _device, name, format, extent, usage, initialLayout, finalLayout);
        AddResource(resource);
    }

    /// <summary>
    /// Add a resource to the render graph
    /// </summary>
    /// <param name="resource"></param>
    public void AddResource(ImageResource resource)
    {
        _imageResources[resource._name] = resource;
    }

    public ImageResource GetResource(string name)
    {
        if (!_imageResources.TryGetValue(name, out var resource))
        {
            throw new Exception($"Resource {name} not found");
        }

        return resource;
    }

    /// <summary>
    /// Add a pass to the render graph
    /// </summary>
    /// <param name="pass"></param>
    public void AddPass(Pass pass) => passes.Add(pass);

    /// <summary>
    /// Creates a new pass and adds it to the render graph
    /// </summary>
    /// <param name="name"></param>
    /// <param name="inputs"></param>
    /// <param name="outputs"></param>
    /// <param name="executeFunc"></param>
    public void AddPass(string name, List<string> inputs, List<string> outputs, Action<CommandBuffer, Renderer.FrameContext> executeFunc)
    {
        Pass pass = new(name, inputs, outputs)
        {
            ExecuteFunc = executeFunc
        };
        AddPass(pass);
    }

    /// <summary>
    /// Rendergraph compilation - Transforms declarative descriptions into executable pipelines<br/>
    /// This method performs dependency analysis, resource allocation, and execution planning<br/>
    /// </summary>
    public void Compile()
    {
        if (_compiled)
        {
            throw new InvalidOperationException("RenderGraph already compiled");
        }

        int n = passes.Count;

        // adjacency[i] = list of pass indices that depend on pass i
        var adjacency = new List<int>[n];
        var inDegree = new int[n];
        for (int i = 0; i < n; i++) adjacency[i] = new List<int>();

        // Build edges: for each resource, find (writer → reader) pairs.
        // Restricted to j > i: a pass only reads from writers declared before it.
        // Without this restriction, a later-declared writer of the same resource
        // (e.g. TransparentPass writing Depth that LightingPass also reads) creates
        // a back-edge into an earlier pass, and combined with writer→writer edges
        // for other shared resources produces a cycle that strands the topo sort.
        for (var i = 0; i < n; i++)
        {
            foreach (var output in passes[i]._outputs)
            {
                for (int j = i + 1; j < n; j++)
                {
                    if (passes[j]._inputs.Contains(output))
                    {
                        adjacency[i].Add(j);
                        inDegree[j]++;
                    }
                }
            }
        }

        // Writer → writer edges: when two passes both write the same resource,
        // serialize them in declaration order (earlier-declared runs first).
        // Without this the topo sort would put them in arbitrary order and
        // blend / accumulation chains (LightingPass → TransparentPass into
        // HDRColor) would break non-deterministically.
        for (int i = 0; i < n; i++)
        {
            foreach (var output in passes[i]._outputs)
            {
                for (int j = i + 1; j < n; j++)
                {
                    if (passes[j]._outputs.Contains(output) && !adjacency[i].Contains(j))
                    {
                        adjacency[i].Add(j);
                        inDegree[j]++;
                    }
                }
            }
        }

        //Topological sort for optimal execution order
        //this is done by Kahn's algorithm
        //BFS from all zero-in-degree nodes
        var queue = new Queue<int>();
        for (var i = 0; i < n; i++)
            if (inDegree[i] == 0)
                queue.Enqueue(i);

        while (queue.Count > 0)
        {
            int node = queue.Dequeue();
            executionOrder.Add(node);
            foreach (var adj in adjacency[node])
            {
                if (--inDegree[adj] == 0)
                    queue.Enqueue(adj);
            }
        }

        if (executionOrder.Count == 0)
        {
            throw new Exception("RenderGraph contains a cycle, check pass dependencies");
        }

        // Single command buffer / single queue: pipeline barriers between passes (in Execute)
        // handle ordering — no semaphores needed inside the graph.

        foreach (var resource in _imageResources.Values)
            resource.Allocate(_physicalDevice);

        // Seed the tracker. Resources start in their declared _initialLayout
        // (typically Undefined for fresh allocations); subsequent transitions
        // update the tracker so each pass sources from the actual prior layout.
        foreach (var kv in _imageResources)
        {
            _currentLayout[kv.Key] = kv.Value._initialLayout;
            
        }
        _compiled = true;
    }

    /// <summary>
    /// Records all passes (input barriers → pass callback → output barriers → final-layout barriers)
    /// into the provided command buffer. Caller owns Begin/End/Submit and the
    /// imageAvailable/renderFinished semaphores.
    /// </summary>
    public void Execute(CommandBuffer cmd, Renderer.FrameContext frameContext)
    {
        if (!_compiled)
            throw new InvalidOperationException("Call Compile() before Execute().");

        foreach (var passIndex in executionOrder)
        {
            var pass = passes[passIndex];
            
            // ----- Barrier transition inputs -> ShaderReadOnlyOptimal
            foreach (var inputName in pass._inputs)
            {
                ImageResource resource = _imageResources[inputName];
                bool isDepth =
                    resource._format is Format.D32Sfloat or Format.D24UnormS8Uint or Format.D16UnormS8Uint;
                if (_imageResources[inputName].IsAllocated)
                {
                    var barrier = new ImageMemoryBarrier()
                    {
                        SType = StructureType.ImageMemoryBarrier,
                        OldLayout = _currentLayout[inputName],
                        NewLayout = ImageLayout.ShaderReadOnlyOptimal,
                        SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
                        DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
                        Image = resource.Image,
                        SrcAccessMask = AccessFlags.MemoryReadBit,
                        DstAccessMask = AccessFlags.ShaderReadBit,
                        SubresourceRange = new ImageSubresourceRange()
                        {
                            AspectMask = isDepth ? ImageAspectFlags.DepthBit : ImageAspectFlags.ColorBit,
                            BaseMipLevel = 0,
                            LevelCount = 1,
                            BaseArrayLayer = 0,
                            LayerCount = 1
                        }
                    };

                    _vk.CmdPipelineBarrier(cmd,
                        PipelineStageFlags.AllCommandsBit,
                        PipelineStageFlags.FragmentShaderBit,
                        DependencyFlags.ByRegionBit,
                        0, null,
                        0, null,
                        1, ref barrier);

                    _currentLayout[inputName] = ImageLayout.ShaderReadOnlyOptimal;
                }
            }

            // ----- Barrier transition outputs -> ColorAttachmentOptimal / DepthStencilAttachmentOptimal
            foreach (var outputName in pass._outputs)
            {
                if (_imageResources[outputName].IsAllocated)
                {
                    ImageResource resource = _imageResources[outputName];
                    bool isDepth =
                        resource._format is Format.D32Sfloat or Format.D24UnormS8Uint or Format.D16UnormS8Uint;
                    var targetAttachmentLayout = isDepth
                        ? ImageLayout.DepthStencilAttachmentOptimal
                        : ImageLayout.ColorAttachmentOptimal;
                    var barrier = new ImageMemoryBarrier()
                    {
                        SType = StructureType.ImageMemoryBarrier,
                        OldLayout = _currentLayout[outputName],
                        NewLayout = targetAttachmentLayout,
                        SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
                        DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
                        Image = resource.Image,
                        SrcAccessMask = AccessFlags.MemoryReadBit,
                        DstAccessMask = isDepth
                            ? AccessFlags.DepthStencilAttachmentWriteBit
                            : AccessFlags.ColorAttachmentWriteBit,
                        SubresourceRange = new ImageSubresourceRange()
                        {
                            AspectMask = isDepth ? ImageAspectFlags.DepthBit : ImageAspectFlags.ColorBit,
                            BaseMipLevel = 0,
                            LevelCount = 1,
                            BaseArrayLayer = 0,
                            LayerCount = 1
                        }
                    };

                    _vk.CmdPipelineBarrier(cmd,
                        PipelineStageFlags.AllCommandsBit,
                        isDepth
                            ? PipelineStageFlags.EarlyFragmentTestsBit
                            : PipelineStageFlags.ColorAttachmentOutputBit,
                        DependencyFlags.ByRegionBit,
                        0, null,
                        0, null,
                        1, ref barrier);

                    _currentLayout[outputName] = targetAttachmentLayout;
                }
            }

            //----- Execute pass-------------------
            pass.ExecuteFunc(cmd, frameContext);

            //----- Barrier: transition outputs -> resource._finalLayout
            foreach (var outputName in pass._outputs)
            {
                if (!_imageResources[outputName].IsAllocated) continue;
                ImageResource resource = _imageResources[outputName];
                bool isDepth =
                    resource._format is Format.D32Sfloat or Format.D24UnormS8Uint or Format.D16UnormS8Uint;
                var barrier = new ImageMemoryBarrier
                {
                    SType = StructureType.ImageMemoryBarrier,
                    OldLayout = _currentLayout[outputName],
                    NewLayout = resource._finalLayout,
                    SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
                    DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
                    Image = resource.Image,
                    SrcAccessMask = isDepth
                        ? AccessFlags.DepthStencilAttachmentWriteBit
                        : AccessFlags.ColorAttachmentWriteBit,
                    DstAccessMask = AccessFlags.MemoryReadBit,
                    SubresourceRange = new ImageSubresourceRange()
                    {
                        AspectMask = isDepth ? ImageAspectFlags.DepthBit : ImageAspectFlags.ColorBit,
                        BaseMipLevel = 0,
                        LevelCount = 1,
                        BaseArrayLayer = 0,
                        LayerCount = 1
                    }
                };
                
                _vk.CmdPipelineBarrier(
                    cmd,
                    isDepth
                        ? PipelineStageFlags.LateFragmentTestsBit
                        : PipelineStageFlags.ColorAttachmentOutputBit,
                    PipelineStageFlags.AllCommandsBit, // before any subsequent work
                    DependencyFlags.ByRegionBit,
                    0, null, 0, null, 1, ref barrier);

                _currentLayout[outputName] = resource._finalLayout;
            }
        }
    }


//IDisposable ------------------
    public void Dispose()
    {
        if (_disposed) return;

        foreach (var resource in _imageResources.Values)
            resource.Dispose();
        _imageResources.Clear();

        _disposed = true;
        GC.SuppressFinalize(this);
    }
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




// ────────────────────────────────────────────────────────────
// Frustum
// ────────────────────────────────────────────────────────────
 
[StructLayout(LayoutKind.Sequential)]
public unsafe struct Frustum
{
    // Six planes in a fixed inline array — no heap allocation,
    // GC never touches this struct.
    // Order matches the standard OpenGL/Vulkan convention:
    //   0 Left  1 Right  2 Bottom  3 Top  4 Near  5 Far
    private fixed float _planeData[6 * 4]; // 6 planes × 4 floats (Vector4)
 
    // ── Index constants ──────────────────────────────────────
    public const int Left   = 0;
    public const int Right  = 1;
    public const int Bottom = 2;
    public const int Top    = 3;
    public const int Near   = 4;
    public const int Far    = 5;
 
    // ── Plane accessors ──────────────────────────────────────
 
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
 
    // ── Construction ─────────────────────────────────────────
 
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
 
    // ── Intersection test ─────────────────────────────────────
 
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

public unsafe struct Pass
{
    public string _name;
    public List<string> _inputs;//names of the resources this pass reads from
    public List<string> _outputs;//names of the resources this pass writes to

    public Pass(string name, List<string> inputs, List<string> outputs)
    {
        _name = name;
        _inputs = inputs ?? new List<string>();
        _outputs = outputs ?? new List<string>();
    }
    
    public void AddInput(string input) => _inputs.Add(input);
    public void AddOutput(string output) => _outputs.Add(output);

    public Action<CommandBuffer, Renderer.FrameContext> ExecuteFunc { get; set; } = (_, _) => { };
    
    
    // Builder helpers — lets callers chain 
    public Pass ReadsFrom (string resourceName) { _inputs .Add(resourceName); return this; }
    public Pass WritesTo  (string resourceName) { _outputs.Add(resourceName); return this; }
    public Pass Executes  (Action<CommandBuffer, Renderer.FrameContext> fn) { ExecuteFunc = fn; return this; }
}