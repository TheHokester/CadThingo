using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
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
    private void RecreateSwapChain()
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

        // Order matters: render graph holds references to depth + g-buffer
        // ImageResources, so dispose it first. The ImageResource Dispose guard
        // makes the explicit calls below idempotent.
        scene.renderGraph.Dispose();
        depthImageResource.Dispose();
        gBufferPosition.Dispose();
        gBufferNormal.Dispose();
        gBufferAlbedo.Dispose();
        gBufferMaterial.Dispose();

        CleanupSwapChain();

        CreateSwapChain();
        CreateImageViews();
        CreateDepthResources();
        CreateGBufferResources();

        scene.renderGraph = new RenderGraph(vk!, device, physicalDevice);
        SetupDeferredRenderer(scene.renderGraph, swapChainExtent.Width, swapChainExtent.Height);

        // G-buffer ImageViews are fresh — re-bind them on the lighting pass set.
        UpdateGBufferDescriptorSet();
    }

    public void DrawFrame()
    {
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

        // 5. Update per-frame UBOs
        // UpdateGeometryUBO is per-entity — wired in Phase 5/6 once entities exist.
        UpdateLightingUBO(currentFrame, camera);

        // 6. Record all render-graph passes (geometry → lighting), record-only.
        scene.renderGraph.Execute(cmd);

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

        // 7b. Blit FinalColor (TransferSrcOptimal — set by render-graph final-layout barrier) → swapchain
        var finalColor = scene.renderGraph.GetResource("FinalColor");
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
        blit.SrcOffsets[0] = new Offset3D(0, 0, 0);
        blit.SrcOffsets[1] = new Offset3D((int)swapChainExtent.Width, (int)swapChainExtent.Height, 1);
        blit.DstOffsets[0] = new Offset3D(0, 0, 0);
        blit.DstOffsets[1] = new Offset3D((int)swapChainExtent.Width, (int)swapChainExtent.Height, 1);

        vk!.CmdBlitImage(cmd,
            finalColor.Image, ImageLayout.TransferSrcOptimal,
            swapImage, ImageLayout.TransferDstOptimal,
            1, &blit, Filter.Linear);

        // 7c. Swapchain TransferDstOptimal → ColorAttachmentOptimal so the UI overlay
        //     can render on top of the blitted scene.
        TransitionImageLayout(cmd, swapImage, swapChainImageFormat,
            ImageLayout.TransferDstOptimal, ImageLayout.ColorAttachmentOptimal);

        // 7d. UI overlay (no-op when imGuiUtils is unwired).
        if (imGuiUtils != null)
        {
            imGuiUtils.newFrame();
            imGuiUtils.updateBuffers();
        }
        imGuiUtils?.DrawFrame(cmd, swapChainImageViews[imageIndex]);

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
        //configure depth buffer for accurate depth information
        //standard 32bit depth format preserves accurate depth information
        graph.AddResource(depthImageResource);
        
        //configure finalcolor buffer for the completed lighting results
        //standard color format with transfer capability for presentation or post processing
        graph.AddResource("FinalColor", Format.R8G8B8A8Unorm, new Extent2D(width, height),
            ImageUsageFlags.ColorAttachmentBit | ImageUsageFlags.TransferSrcBit,
            ImageLayout.Undefined, ImageLayout.TransferSrcOptimal);
        
        
        //configure geometry pass for G-buffer population
        //this pass renders all geometry and stores intermediate data for lighting calculations
        graph.AddPass("GeometryPass", default,
            new List<string> { "GBuffer_Position", "GBuffer_Normal", "GBuffer_Albedo", "GBuffer_Material", "Depth" }, buffer =>
            {
                //Configure multiple render target attachments for GBuffer output
                //each attachment corresponds to a seperate geometric property
                RenderingAttachmentInfoKHR* colorAttachments = stackalloc RenderingAttachmentInfoKHR[4];
                colorAttachments[0] = new()
                {
                    SType = StructureType.RenderingAttachmentInfoKhr,
                    ImageView = graph.GetResource("GBuffer_Position").ImageView,
                    ImageLayout = ImageLayout.ColorAttachmentOptimal,
                    LoadOp = AttachmentLoadOp.Clear,
                    StoreOp = AttachmentStoreOp.Store
                };
                colorAttachments[1] = new()
                {
                    SType = StructureType.RenderingAttachmentInfoKhr,
                    ImageView = graph.GetResource("GBuffer_Normal").ImageView,
                    ImageLayout = ImageLayout.ColorAttachmentOptimal,
                    LoadOp = AttachmentLoadOp.Clear,
                    StoreOp = AttachmentStoreOp.Store
                };
                colorAttachments[2] = new()
                {
                    SType = StructureType.RenderingAttachmentInfoKhr,
                    ImageView = graph.GetResource("GBuffer_Albedo").ImageView,
                    ImageLayout = ImageLayout.ColorAttachmentOptimal,
                    LoadOp = AttachmentLoadOp.Clear,
                    StoreOp = AttachmentStoreOp.Store
                };
                colorAttachments[3] = new()
                {
                    SType = StructureType.RenderingAttachmentInfoKhr,
                    ImageView = graph.GetResource("GBuffer_Material").ImageView,
                    ImageLayout = ImageLayout.ColorAttachmentOptimal,
                    LoadOp = AttachmentLoadOp.Clear,
                    StoreOp = AttachmentStoreOp.Store
                };
                //Configure depth attachment for occlusion culling
                RenderingAttachmentInfoKHR depthAttachment = new()
                {
                    SType = StructureType.RenderingAttachmentInfoKhr,
                    ImageView = graph.GetResource("Depth").ImageView,
                    ImageLayout = ImageLayout.DepthStencilAttachmentOptimal,
                    LoadOp = AttachmentLoadOp.Clear,
                    StoreOp = AttachmentStoreOp.Store,
                    ClearValue = new ClearValue() { DepthStencil = new ClearDepthStencilValue(1.0f, 0) }
                };
                //assemble complete rendering config
                RenderingInfoKHR renderingInfo = new()
                {
                    SType = StructureType.RenderingInfoKhr,
                    RenderArea = new Rect2D(new Offset2D(0, 0), new Extent2D(width, height)),
                    LayerCount = 1,
                    ColorAttachmentCount = 4,
                    PColorAttachments = (RenderingAttachmentInfo*)colorAttachments,
                    PDepthAttachment = (RenderingAttachmentInfo*)&depthAttachment
                };
                
                //execute geometry rendering with dynamic rendering
                vk!.CmdBeginRendering(buffer, (RenderingInfo*)&renderingInfo);

                // Pipeline + dynamic state
                vk!.CmdBindPipeline(buffer, PipelineBindPoint.Graphics, geometryPipeline);

                Viewport vp = new()
                {
                    X = 0, Y = 0,
                    Width = width, Height = height,
                    MinDepth = 0.0f, MaxDepth = 1.0f,
                };
                Rect2D scissor = new(new Offset2D(0, 0), new Extent2D(width, height));
                vk!.CmdSetViewport(buffer, 0, 1, &vp);
                vk!.CmdSetScissor(buffer, 0, 1, &scissor);

                // Bind per-frame geometry descriptor set (UBO + dummy samplers)
                var dset = geometryDescriptorSets[currentFrame];
                vk!.CmdBindDescriptorSets(buffer, PipelineBindPoint.Graphics,
                    geometryPipelineLayout, 0, 1, &dset, 0, null);

                // Bind global VB/IB once — every mesh is packed into these
                var vb = Engine.ResourceManager.GlobalVertexBuffer;
                var ib = Engine.ResourceManager.GlobalIndexBuffer;
                ulong vbOffset = 0;
                vk!.CmdBindVertexBuffers(buffer, 0, 1, &vb, &vbOffset);
                vk!.CmdBindIndexBuffer(buffer, ib, 0, IndexType.Uint32);

                // Default PBR push constants. Binding 1 is wired to a real base color
                // texture at descriptor-set creation time, so BaseColorTextureSet = 0.
                // Remaining texture slots fall back to dummy whites and stay disabled.
                PbrPushConstants pcDefaults = new()
                {
                    BaseColorFactor = new Vector4(1, 1, 1, 1),
                    MetallicFactor = 0.3f,
                    RoughnessFactor = 0.7f,
                    BaseColorTextureSet = 0,
                    PhysicalDescriptorTextureSet = -1,
                    NormalTextureSet = -1,
                    OcclusionTextureSet = -1,
                    EmissiveTextureSet = -1,
                    AlphaMask = 0f,
                    AlphaMaskCutoff = 0f,
                };

                // Iterate entities with MeshComponent. Model matrix is pushed via the mapped UBO
                // (single-entity correct; multi-entity = last-write-wins until push-constant model lands).
                for (int i = 0; i < scene.EntityCount; i++)
                {
                    Entity* e = scene.GetEntity(i);
                    if (e == null) continue;
                    var meshComp = e->GetComponent<MeshComponent>();
                    if (meshComp == null) continue;
                    if (currentFrame == 0)
                    {
                        
                    }
                    var meshPtr = meshComp.mesh;
                    if (meshPtr == null) continue;

                    UpdateGeometryUBO(currentFrame, e, camera);

                    PbrPushConstants pc = pcDefaults;
                    vk!.CmdPushConstants(buffer, geometryPipelineLayout,
                        ShaderStageFlags.VertexBit | ShaderStageFlags.FragmentBit,
                        0, (uint)sizeof(PbrPushConstants), &pc);

                    Mesh m = *meshComp.mesh;
                    vk!.CmdDrawIndexed(buffer, (uint)m.count, 1, (uint)m.offset, 0, 0);
                }

                vk!.CmdEndRendering(buffer);
            });
        
        
        graph.AddPass("LightingPass", new List<string>{"GBuffer_Position", "GBuffer_Normal", "GBuffer_Albedo", "GBuffer_Material","Depth"},
            new List<string>{"FinalColor"}, buffer =>
            {
                //configure single color output for final lighting result
                RenderingAttachmentInfoKHR colorAttachment = new()
                {
                    SType = StructureType.RenderingAttachmentInfoKhr,
                    ImageView = graph.GetResource("FinalColor").ImageView,
                    ImageLayout = ImageLayout.ColorAttachmentOptimal,
                    LoadOp = AttachmentLoadOp.Clear,
                    StoreOp = AttachmentStoreOp.Store,
                    ClearValue = new ClearValue() { Color = new ClearColorValue(0.2f, 0.3f, 0.8f, 1.0f) }

                };
                //Configure lighting pass rendeirng without depth testing 
                //unnecessary since we're processing each pixel exactly once
                RenderingInfoKHR renderingInfo = new()
                {
                    SType = StructureType.RenderingInfoKhr,
                    RenderArea = new Rect2D(new Offset2D(0, 0), new Extent2D(width, height)),
                    LayerCount = 1,
                    ColorAttachmentCount = 1,
                    PColorAttachments = (RenderingAttachmentInfo*)&colorAttachment
                };
                
                //execute screen space lighting calcs.
                vk!.CmdBeginRendering(buffer, (RenderingInfo*)&renderingInfo);

                vk!.CmdBindPipeline(buffer, PipelineBindPoint.Graphics, pbrLightingPipeline);

                Viewport vp = new()
                {
                    X = 0, Y = 0,
                    Width = width, Height = height,
                    MinDepth = 0.0f, MaxDepth = 1.0f,
                };
                Rect2D scissor = new(new Offset2D(0, 0), new Extent2D(width, height));
                vk!.CmdSetViewport(buffer, 0, 1, &vp);
                vk!.CmdSetScissor(buffer, 0, 1, &scissor);

                // Set 0 = per-frame LightingUBO, Set 1 = shared G-buffer samplers
                var sets = stackalloc DescriptorSet[2]
                {
                    lightingDescriptorSets[currentFrame],
                    gBufferDescriptorSet,
                };
                vk!.CmdBindDescriptorSets(buffer, PipelineBindPoint.Graphics,
                    pbrPipelineLayout, 0, 2, sets, 0, null);

                // Shader declares PushConstants but PSMain doesn't read it in the lighting pass;
                // push defaults so the range is initialized.
                PbrPushConstants pc = new()
                {
                    BaseColorFactor = new Vector4(1, 1, 1, 1),
                    MetallicFactor = 0.3f,
                    RoughnessFactor = 0.7f,
                    BaseColorTextureSet = -1,
                    PhysicalDescriptorTextureSet = -1,
                    NormalTextureSet = -1,
                    OcclusionTextureSet = -1,
                    EmissiveTextureSet = -1,
                    AlphaMask = 0f,
                    AlphaMaskCutoff = 0f,
                };
                vk!.CmdPushConstants(buffer, pbrPipelineLayout,
                    ShaderStageFlags.VertexBit | ShaderStageFlags.FragmentBit,
                    0, (uint)sizeof(PbrPushConstants), &pc);

                // Fullscreen triangle — VSMain synthesizes 3 verts from SV_VertexID
                vk!.CmdDraw(buffer, 3, 1, 0, 0);

                vk!.CmdEndRendering(buffer);
            });
        
        graph.Compile();
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
    public void AddPass(string name, List<string> inputs, List<string> outputs, Action<CommandBuffer> executeFunc)
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

        // Build edges: for each resource, find (writer → reader) pairs
        for (var i = 0; i < n; i++)
        {
            foreach (var output in passes[i]._outputs)
            {
                for (int j = 0; j < n; j++)
                {
                    if (i == j) continue;
                    if (passes[j]._inputs.Contains(output))
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

        _compiled = true;
    }

    /// <summary>
    /// Records all passes (input barriers → pass callback → output barriers → final-layout barriers)
    /// into the provided command buffer. Caller owns Begin/End/Submit and the
    /// imageAvailable/renderFinished semaphores.
    /// </summary>
    public void Execute(CommandBuffer cmd)
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
                        OldLayout = resource._initialLayout,
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
                    var barrier = new ImageMemoryBarrier()
                    {
                        SType = StructureType.ImageMemoryBarrier,
                        OldLayout = resource._initialLayout,
                        NewLayout = isDepth
                            ? ImageLayout.DepthStencilAttachmentOptimal
                            : ImageLayout.ColorAttachmentOptimal,
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
                }
            }

            //----- Execute pass-------------------
            pass.ExecuteFunc(cmd);

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
                    OldLayout = isDepth
                        ? ImageLayout.DepthStencilAttachmentOptimal
                        : ImageLayout.ColorAttachmentOptimal,
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

    public Action<CommandBuffer> ExecuteFunc { get; set; } = _ => { };
    
    
    // Builder helpers — lets callers chain 
    public Pass ReadsFrom (string resourceName) { _inputs .Add(resourceName); return this; }
    public Pass WritesTo  (string resourceName) { _outputs.Add(resourceName); return this; }
    public Pass Executes  (Action<CommandBuffer> fn) { ExecuteFunc = fn; return this; }
}