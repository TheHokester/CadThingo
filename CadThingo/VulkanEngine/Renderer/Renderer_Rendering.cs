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

    private void SetupRenderPasses()
    {
        //Create geometry pass
        var gPassName = "GeometryPass";
        var geometryPass = renderPassManager.AddRenderPass<GeometryPass>(gPassName,(string name) => new GeometryPass(device, name, cullingSystem));
        //Create lighting pass
        var lPassName = "LightingPass";
        var lightingPass = renderPassManager.AddRenderPass<LightingPass>(lPassName,(string name) =>  new LightingPass(device, name, geometryPass));
        //Create postprocess pass
        var ppPassName = "PostProcessPass";
        var postProcessPass = renderPassManager.AddRenderPass<PostProcessPass>(ppPassName, (string name) => new PostProcessPass(device, name, lightingPass));
        
    }
    
    private void CleanupSwapChain()
    {
        
    }

    private void RecreateSwapChain()
    {
        
    }
    /// <summary>
    /// DO NOT LIKE this as is, but it's a start.
    /// </summary>
    /// <param name="entities"></param>
    public void Render(List<Entity> entities)
    {
        vk!.WaitForFences(device, 1, ref inFlightFences[currentFrame], true, ulong.MaxValue);
        vk!.ResetFences(device, 1, ref inFlightFences[currentFrame]);
        
        vk!.ResetCommandBuffer(commandBuffers[0], CommandBufferResetFlags.ReleaseResourcesBit);
        
        //perform culling
        cullingSystem.CullScene(entities);
        
        //record commands
        CommandBufferBeginInfo beginInfo = new();
        vk!.BeginCommandBuffer(commandBuffers[0], &beginInfo);
        
        //execute render passes
        renderPassManager.Execute(commandBuffers[0]);

        vk!.EndCommandBuffer(commandBuffers[0]);
        Semaphore imageAvailableSemaphore = imageAvailableSemaphores[currentFrame];
        Semaphore renderFinishedSemaphore = renderFinishedSemaphores[currentFrame];
        var waitStages = stackalloc []{ PipelineStageFlags.ColorAttachmentOutputBit };
        fixed (CommandBuffer* pCmdBuffer = &commandBuffers[0])
        {
            var submitInfo = new SubmitInfo()
            {
                SType = StructureType.SubmitInfo,
                CommandBufferCount = 1,
                PCommandBuffers = pCmdBuffer,
                WaitSemaphoreCount = 1,
                PWaitSemaphores = &imageAvailableSemaphore,
                PWaitDstStageMask = waitStages,
                SignalSemaphoreCount = 1,
                PSignalSemaphores = &renderFinishedSemaphore
            };
            vk!.QueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]);
        }
        
        
    }
    
    public void RenderFrame(Device device, Queue graphicsQueue, Queue presentQueue)
    {
        //Synchronise with previos frame completion
        //prevent cpu from submitting work faster than gpu can process it
        Result result = vk!.WaitForFences(device, 1, ref inFlightFences[currentFrame] , true, ulong.MaxValue);
        
        //Reset fence for this frames completion tracking
        //prepare the fence to signal when this frames gpu work completes
        vk.ResetFences(device,1, ref inFlightFences[currentFrame]);
        
        //acquire next available image from the swapchain
        //This operation coordinates with the presentation engine and display system
        uint imageIndex = 0;
        result = swapChainKhr.AcquireNextImage(device, swapChain, long.MaxValue, imageAvailableSemaphores[currentFrame], default, &imageIndex);
        if (result == Result.ErrorOutOfDateKhr)
        {
            RecreateSwapChain();
        }
        else if (result != Result.Success && result != Result.SuboptimalKhr)
        {
            throw new Exception("Failed to acquire swap chain image");
        }
        //record commands for this frames rendering

        
        
        
        SubmitInfo submitInfo = new()
        {
            SType = StructureType.SubmitInfo,
        };
        var waitSemaphores = stackalloc[] { imageAvailableSemaphores[currentFrame] };
        var waitStages = stackalloc[] { PipelineStageFlags.ColorAttachmentOutputBit };
        
        var buffer = commandBuffers![imageIndex];
        
        submitInfo = submitInfo with
        {
            WaitSemaphoreCount = 1,
            PWaitSemaphores = waitSemaphores,
            PWaitDstStageMask = waitStages,

            CommandBufferCount = 1,
            PCommandBuffers = &buffer
        };

        var signalSemaphores = stackalloc[] { renderFinishedSemaphores![currentFrame] };
        submitInfo = submitInfo with
        {
            SignalSemaphoreCount = 1,
            PSignalSemaphores = signalSemaphores
        };
        //submit work to gpu with fence-based completion tracking
        //the fence allows cpu to know when this frames gpu work is complete
        vk!.QueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]);
        fixed (SwapchainKHR* pSwapChain = &swapChain)
        {
            PresentInfoKHR presentInfo = new()
            {
                WaitSemaphoreCount = 1,
                PWaitSemaphores = waitSemaphores,
                SwapchainCount = 1,
                PSwapchains = pSwapChain,
                PImageIndices = &imageIndex,
            };
            //submit presentation request to the presentation queue
            result = swapChainKhr.QueuePresent(presentQueue, &presentInfo);
            if (result == Result.ErrorOutOfDateKhr || result == Result.SuboptimalKhr)
            {
                RecreateSwapChain();
            }
        }
            
        }
    
    
    
    
    
    /// <summary>
    /// Comprehensive deferred renderer setup demonstrating rendergraph resource management<br/>
    /// this implementation shows how to efficiently organise resources for multiple passes<br/>
    /// </summary>
    /// <param name="renderGraph"></param>
    /// <param name="width"></param>
    /// <param name="height"></param>
    private void SetupDeferredRenderer(RenderGraph graph, uint width, uint height)
    {
        //configure posiiton buffer for world-space vertex positions
        //High precision format preserves positional accuracy for lighting calculations
        graph.AddResource("GBuffer_Position", Format.R32G32B32Sfloat, new Extent2D(width, height),
            ImageUsageFlags.ColorAttachmentBit | ImageUsageFlags.InputAttachmentBit,
            ImageLayout.Undefined, ImageLayout.ShaderReadOnlyOptimal);
        
        
        //Configure normal buffer for surface orientation data
        //High precision normals enable accurate lighting and reflections
        graph.AddResource("GBuffer_Normal", Format.R32G32B32A32Sfloat, new Extent2D(width, height),
            ImageUsageFlags.ColorAttachmentBit | ImageUsageFlags.InputAttachmentBit,
            ImageLayout.Undefined, ImageLayout.ShaderReadOnlyOptimal);
        
        
        //configure albedo buffer for surface color information
        //standard 8bit precision sufficient for color data with gamma encoding
        graph.AddResource("GBuffer_Albedo", Format.R8G8B8A8Unorm, new Extent2D(width, height),
            ImageUsageFlags.ColorAttachmentBit | ImageUsageFlags.InputAttachmentBit,
            ImageLayout.Undefined, ImageLayout.ShaderReadOnlyOptimal);
        
        
        //configure depth buffer for accurate depth information
        //standard 32bit depth format preserves accurate depth information
        graph.AddResource("Depth", Format.D32Sfloat, new Extent2D(width, height),
            ImageUsageFlags.DepthStencilAttachmentBit | ImageUsageFlags.InputAttachmentBit,
            ImageLayout.Undefined, ImageLayout.DepthStencilAttachmentOptimal);
        
        
        //configure finalcolor buffer for the completed lighting results
        //standard color format with transfer capability for presentation or post processing
        graph.AddResource("FinalColor", Format.R8G8B8A8Unorm, new Extent2D(width, height),
            ImageUsageFlags.ColorAttachmentBit | ImageUsageFlags.TransferSrcBit,
            ImageLayout.Undefined, ImageLayout.TransferSrcOptimal);
        
        
        //configure geometry pass for G-buffer population
        //this pass renders all geometry and stores intermediate data for lighting calculations
        graph.AddPass("GeometryPass", default,
            new List<string> { "GBuffer_Position", "GBuffer_Normal", "GBuffer_Albedo", "Depth" }, buffer =>
            {
                //Configure multiple render target attachments for GBuffer output
                //each attachment corresponds to a seperate geometric property
                RenderingAttachmentInfoKHR* colorAttachments = stackalloc RenderingAttachmentInfoKHR[3];
                colorAttachments[0] = new()
                {
                    ImageView = graph.GetResource("GBuffer_Position").ImageView,
                    ImageLayout = ImageLayout.ColorAttachmentOptimal,
                    LoadOp = AttachmentLoadOp.Clear,
                    StoreOp = AttachmentStoreOp.Store
                };
                colorAttachments[1] = new()
                {
                    ImageView = graph.GetResource("GBuffer_Normal").ImageView,
                    ImageLayout = ImageLayout.ColorAttachmentOptimal,
                    LoadOp = AttachmentLoadOp.Clear,
                    StoreOp = AttachmentStoreOp.Store
                };
                colorAttachments[2] = new()
                {
                    ImageView = graph.GetResource("GBuffer_Albedo").ImageView,
                    ImageLayout = ImageLayout.ColorAttachmentOptimal,
                    LoadOp = AttachmentLoadOp.Clear,
                    StoreOp = AttachmentStoreOp.Store
                };
                //Configure depth attachment for occlusion culling
                RenderingAttachmentInfoKHR depthAttachment = new()
                {
                    ImageView = graph.GetResource("Depth").ImageView,
                    ImageLayout = ImageLayout.DepthStencilAttachmentOptimal,
                    LoadOp = AttachmentLoadOp.Clear,
                    StoreOp = AttachmentStoreOp.Store,
                    ClearValue = new ClearValue() { DepthStencil = new ClearDepthStencilValue(1.0f, 0) }
                };
                //assemble complete rendering config
                RenderingInfoKHR renderingInfo = new()
                {
                    RenderArea = new Rect2D(new Offset2D(0, 0), new Extent2D(width, height)),
                    LayerCount = 1,
                    ColorAttachmentCount = 3,
                    PColorAttachments = (RenderingAttachmentInfo*)colorAttachments,
                    PDepthAttachment = (RenderingAttachmentInfo*)&depthAttachment
                };
                
                //execute geometry rendering with dynamic rendering
                vk!.CmdBeginRendering(buffer, (RenderingInfo*)&renderingInfo);
                
                //Bind geometry pipeline and render all scene objects
                //each draw call populates position normal and albedo for visible fragments
                //implement here -------------------------------------------------------------
                
                vk!.CmdEndRendering(buffer);
            });
        
        
        graph.AddPass("LightingPass", new List<string>{"GBuffer_Position", "GBuffer_Normal", "GBuffer_Albedo", "Depth"},
            new List<string>{"FinalColor"}, buffer =>
            {
                //configure single color output for final lighting result
                RenderingAttachmentInfoKHR colorAttachment = new()
                {
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
                    RenderArea = new Rect2D(new Offset2D(0, 0), new Extent2D(width, height)),
                    LayerCount = 1,
                    ColorAttachmentCount = 1,
                    PColorAttachments = (RenderingAttachmentInfo*)&colorAttachment
                };
                
                //execute screen space lighting calcs.
                vk!.CmdBeginRendering(buffer, (RenderingInfo*)&renderingInfo);
                
                //Bind lighting pipeline and render all visible fragments
                //fragment shader reads gbuffer textures and computes lighting for each
                // all scene lights are processed in a single screenspace render pass
                //implement here ---------------------------------------------------------
                
                vk!.CmdEndRendering(buffer);
            });
        
        graph.Compile();
    }
    
    
    
    
    
    
    
    /// <summary>
    /// Workflow<br/>
    /// 1. AddResource() - declare all images the graph will use <br/>
    /// 2. AddPass() - declare all passes and there read/write sets<br/>
    /// 3. Compile() - topological sort + allocate gpu resources<br/>
    /// 4. Execute() - record all passes into the command buffer in order<br/>
    /// 5. Dispose() - free all gpu resources <br/>
    /// </summary>
    class RenderGraph : IDisposable
    {
        private readonly Vk _vk;
        private readonly Device _device;
        private readonly PhysicalDevice _physicalDevice;
        private bool _disposed;
        private bool _compiled;

        private readonly Dictionary<string, ImageResource> _imageResources = new();
        private List<Pass> passes;
        private List<int> executionOrder;

        //Automatic synchronization management
        //these objects ensure correct GPU execution order without manual barriers
        private List<Semaphore> semaphores;
        private List<SemaphorePair> semaphorePairs;

        public RenderGraph(Vk vk, Device device)
        {
            _vk = vk;
            _device = device;
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

            //Automatic Synchronization object creations
            //Generate semaphore for all dependencies identified
            var semaphoreInfo = new SemaphoreCreateInfo()
            {
                SType = StructureType.SemaphoreCreateInfo
            };
            for (int i = 0; i < passes.Count; i++)
            {
                foreach (var output in passes[i]._outputs)
                {
                    for (int j = 0; j < passes.Count; j++)
                    {
                        if (i == j) continue;
                        if (!passes[j]._inputs.Contains(output)) continue;

                        if (_vk.CreateSemaphore(_device, ref semaphoreInfo, null, out var semaphore) != Result.Success)
                        {
                            throw new Exception("Failed to create render graph semaphore");
                        }

                        semaphores.Add(semaphore);
                        semaphorePairs.Add(new SemaphorePair()
                        {
                            SignalPassIndex = i,
                            WaitPassIndex = j,
                        });

                    }
                }
            }
            
            foreach (var resource in _imageResources.Values)
                resource.Allocate(_physicalDevice);

            _compiled = true;
        }

        public void Execute(CommandBuffer cmd, Queue queue)
        {
            if (!_compiled)
                throw new InvalidOperationException("Call Compile() before Execute().");
            //pre pass temporaries
            var waitSemaphores = new List<Semaphore>();
            var waitStages = new List<PipelineStageFlags>();
            var signalSemaphores = new List<Semaphore>();


            foreach (var passIndex in executionOrder)
            {
                var pass = passes[passIndex];

                //------------collect wait semaphores for this pass----------------------
                waitSemaphores.Clear();
                waitStages.Clear();

                for (int i = 0; i < semaphorePairs.Count; i++)
                {
                    if (semaphorePairs[i].WaitPassIndex == passIndex)
                    {
                        waitSemaphores.Add(semaphores[i]);
                        waitStages.Add(PipelineStageFlags.ColorAttachmentOutputBit);
                    }
                }

                //---------collect signal semaphores for this pass-----------------------
                signalSemaphores.Clear();

                for (int i = 0; i < semaphorePairs.Count; i++)
                    if (semaphorePairs[i].SignalPassIndex == passIndex)
                        signalSemaphores.Add(semaphores[i]);


                //------------begin command recording ------------------------------

                var beginInfo = new CommandBufferBeginInfo()
                {
                    SType = StructureType.CommandBufferBeginInfo,
                    Flags = CommandBufferUsageFlags.OneTimeSubmitBit,

                };
                if (_vk.BeginCommandBuffer(cmd, ref beginInfo) != Result.Success)
                    throw new Exception("Failed to begin command buffer");

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
                                AspectMask = isDepth ? ImageAspectFlags.ColorBit : ImageAspectFlags.DepthBit,
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

                // ----- Barrier transition outputs -> ColorAttachmentOptimal
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
                            NewLayout = ImageLayout.ColorAttachmentOptimal,
                            SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
                            DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
                            Image = resource.Image,
                            SrcAccessMask = AccessFlags.MemoryReadBit,
                            DstAccessMask = AccessFlags.ColorAttachmentWriteBit,
                            SubresourceRange = new ImageSubresourceRange()
                            {
                                AspectMask = isDepth ? ImageAspectFlags.ColorBit : ImageAspectFlags.DepthBit,
                                BaseMipLevel = 0,
                                LevelCount = 1,
                                BaseArrayLayer = 0,
                                LayerCount = 1
                            }
                        };

                        _vk.CmdPipelineBarrier(cmd,
                            PipelineStageFlags.AllCommandsBit,
                            PipelineStageFlags.ColorAttachmentOutputBit,
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
                    bool isDepth = resource._format is Format.D32Sfloat or Format.D24UnormS8Uint or Format.D16UnormS8Uint;
                    var barrier = new ImageMemoryBarrier
                    {
                        SType = StructureType.ImageMemoryBarrier,
                        OldLayout = ImageLayout.ColorAttachmentOptimal,
                        NewLayout = resource._finalLayout,
                        SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
                        DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
                        Image = resource.Image,
                        SrcAccessMask = AccessFlags.ColorAttachmentWriteBit,
                        DstAccessMask = AccessFlags.MemoryReadBit,
                        SubresourceRange = new ImageSubresourceRange()
                        {
                            AspectMask = isDepth ? ImageAspectFlags.ColorBit : ImageAspectFlags.DepthBit,
                            BaseMipLevel = 0,
                            LevelCount = 1,
                            BaseArrayLayer = 0,
                            LayerCount = 1
                        }
                    };

                    _vk.CmdPipelineBarrier(
                        cmd,
                        PipelineStageFlags.ColorAttachmentOutputBit,
                        PipelineStageFlags.AllCommandsBit, // before any subsequent work
                        DependencyFlags.ByRegionBit,
                        0, null, 0, null, 1, barrier);
                }
                //----- End command recording ------------------------------
                if (_vk.EndCommandBuffer(cmd) != Result.Success)
                {
                    throw new Exception("Failed to end command buffer for pass name" + pass._name);
                };
                
                
                //---- Submit with semaphore wiring --------------------
                int waitCount = waitSemaphores.Count;
                int signalCount = signalSemaphores.Count;

                Span<Semaphore> waitSemSpan = stackalloc Semaphore[waitCount];
                Span<PipelineStageFlags> waitStageSpan = stackalloc PipelineStageFlags[waitCount];
                Span<Semaphore> sigSemSpan = stackalloc Semaphore[signalCount];
                
                
                for(int i = 0; i < waitCount; i++) waitSemSpan[i] = waitSemaphores[i];
                for(int i = 0; i < waitCount; i++) waitStageSpan[i] = waitStages[i];
                for(int i = 0 ; i < signalCount; i++) sigSemSpan[i] = signalSemaphores[i];
                
                fixed(Semaphore* pWaitSem = waitSemSpan)
                fixed(PipelineStageFlags* pWaitStage = waitStageSpan)
                fixed (Semaphore* pSigSem = sigSemSpan)
                {
                    var cb = cmd; //local copy so we can take its address

                    var submitInfo = new SubmitInfo()
                    {
                        SType = StructureType.SubmitInfo,
                        WaitSemaphoreCount = (uint)waitCount,
                        PWaitSemaphores = pWaitSem,
                        PWaitDstStageMask = pWaitStage,
                        CommandBufferCount = 1,
                        PCommandBuffers = &cb,
                        SignalSemaphoreCount = (uint)signalCount,
                        PSignalSemaphores = pSigSem,
                    };
                    //queue submit
                    if (_vk.QueueSubmit(queue, 1, submitInfo, default) != Result.Success)
                    {
                        throw new Exception("Queue submit failed for pass: " + pass._name);
                    }
                    
                }
            } 
            
        }
    


//IDisposable ------------------
        public void Dispose()
        {
            if (_disposed) return;
            foreach (var sem in semaphores)
            {
                _vk.DestroySemaphore(_device, sem, null);
            }
            semaphores.Clear();
            
            foreach (var resource in _imageResources.Values)
                resource.Dispose();
            _imageResources.Clear();
            
            _disposed = true;
            GC.SuppressFinalize(this);
        }
        
    }

    /// <summary>
    /// 
    /// </summary>
    class RenderPassManager
    {
        private readonly Vk _vk;
        private readonly Device _device;
        private readonly PhysicalDevice _physicalDevice;
        
        Dictionary<string, RenderPass> renderPasses = new();
        List<RenderPass> sortedPasses = new(); 
        bool dirty = true;
        
        public RenderPassManager(Vk vk, Device device, PhysicalDevice physicalDevice)
        {
            _vk = vk;
            _device = device;
            _physicalDevice = physicalDevice;
        }
        /// <summary>
        /// Creates a new render pass and adds it to the render pass manager
        /// </summary>
        /// <param name="name">Pass name</param>
        /// <param name="factory"> Constructor for the renderpass object of desired subtype</param>
        /// <typeparam name="T">Render pass subtype</typeparam>
        /// <returns>Pointer to new render pass</returns>
        public T* AddRenderPass<T>(string name, Func<string, T> factory) where T : RenderPass
        {
            if (renderPasses.TryGetValue(name, out var pass))
            {
                return (T*)&pass;
            }
            var newPass = factory(name);
            renderPasses[name] = newPass;
            dirty = true;
            
            return (T*)&newPass;
        }


        public RenderPass* GetRenderPass(string name)
        {
            if (!renderPasses.TryGetValue(name, out var pass))
            {
                return null;
            }
            return (RenderPass*)&pass;
        }

        public void RemoveRenderPass(string name)
        {
           renderPasses.Remove(name); 
        }
        
        
        /// <summary>
        /// Compile passes into render pipelines, passes are executed in dependency order <br/>
        /// 
        /// </summary>
        /// <param name="buffer">Buffer to execute commands with</param>
        public void Execute(CommandBuffer buffer)
        {
            if (dirty)
            {
                SortPasses();
                dirty = false;
            }

            foreach (var pass in sortedPasses)
            {
                pass.Execute(buffer);
            }
        }

        private void SortPasses()
        {
            //topologically sort render passes based on dependencies
            sortedPasses.Clear();
            
            //create a copy of render passes for sorting
            var passMap = new Dictionary<string, RenderPass>(renderPasses);
            
            //perform topological sort
            HashSet<string> visited = new();
            HashSet<string> visiting = new();

            foreach (var pair in passMap)
            {
                if (visited.Contains(pair.Key))
                {
                    TopologicalSort(pair.Key, passMap, ref visited, ref visiting);
                }
            }

            void TopologicalSort(string name, Dictionary<string, RenderPass> map,ref HashSet<string> visited,
                ref HashSet<string> visiting)
            {
                //Cycle detected
                if (visiting.Contains(name))
                {
                    throw new InvalidOperationException($"Cycle detected at {name}");
                }

                // Already processed
                if (visited.Contains(name))
                    return;

                visiting.Add(name);

                var pass = map[name];
                foreach (var dependency in pass.GetDependencies())
                {
                    TopologicalSort(dependency, map, ref visited, ref visiting);
                }

                visiting.Remove(name);
                visited.Add(name);
                sortedPasses.Add(pass);
            }
        }
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
            BoundingBox boundingBox = meshComponent->GetBoundingBox();
            //transform the bounding box by entity transform
            boundingBox.Transform(transformComponent->GetTransformMatrix());

            //check if bounding box is visible
            if (frustum.Intersects(boundingBox))
            {
                visibleEntities.Add(entity);
            }
        }
    }

    public List<Entity> GetVisibleEntities() => visibleEntities;
}

unsafe class RenderPass : IDisposable
{
    protected Vk vk = Vk.GetApi();
    private Device device;
    
    private readonly string _name;
    private readonly List<string> dependencies;
    private RenderTarget* _renderTarget = null;
    bool enabled = true;

    public RenderPass(Device device, string name)
    {
        this.device = device;
        _name = name;
    }

    public string GetName() => _name;
    public RenderTarget* GetRenderTarget() => _renderTarget;
    public void SetRenderTarget(RenderTarget* renderTarget) => _renderTarget = renderTarget;
    public void SetEnabled(bool enabled) => this.enabled = enabled; 
    public bool IsEnabled() => enabled;
    public List<string> GetDependencies() => dependencies;
    
    public void AddDependency(string dependency) => dependencies.Add(dependency);

    public virtual void Execute(CommandBuffer buffer)
    {
        if(!enabled) return;

        BeginPass(buffer);
        Render(buffer);
        EndPass(buffer);
    }
    //With dynamic rendering, beginpass call vkCmdBeginRendering isntead of CmdBeginRenderPass
    protected virtual void BeginPass(CommandBuffer buffer){}
    protected virtual void Render(CommandBuffer buffer){}
    //with dynamic rendering, endpass call vkCmdEndRendering instead of CmdEndRenderPass
    protected virtual void EndPass(CommandBuffer buffer){}
    
    
    
    
    
    public virtual void Dispose()
    {
    }
}

/// <summary>
/// Geometry pass for deferred rendering
/// </summary>
unsafe class GeometryPass : RenderPass
{
    private readonly Vk _vk = Vk.GetApi();
    CullingSystem _cullingSystem;
    
    RenderTarget gBuffer;
    public GeometryPass(Device device, string name, CullingSystem culling) : base(device, name)
    {
        //Create gbuffer render target
        gBuffer = new RenderTarget(device, 1920, 1080);//example resolution
        fixed(RenderTarget* pGBuffer = &gBuffer)
            SetRenderTarget(pGBuffer);
    }

    public override void Dispose()
    {
        base.Dispose();
        gBuffer.Dispose();
        
        GC.SuppressFinalize(this);
    }

    protected override void BeginPass(CommandBuffer buffer)
    {
        //begin rendering with dynamic rendering
        RenderingInfoKHR renderingInfo;
        
        //Color attachment
        RenderingAttachmentInfoKHR colorAttachmentInfo = new()
        {
            ImageView = gBuffer.GetColorImageView(),
            ImageLayout = ImageLayout.ColorAttachmentOptimal,
            LoadOp = AttachmentLoadOp.Clear,
            StoreOp = AttachmentStoreOp.Store,
            ClearValue = new ClearValue() { Color = new ClearColorValue(0, 0, 0, 1) }
        };
        
        //Depth attachment
        RenderingAttachmentInfoKHR depthAttachmentInfo = new()
        {
            ImageView = gBuffer.GetDepthImageView(),
            ImageLayout = ImageLayout.DepthStencilAttachmentOptimal,
            LoadOp = AttachmentLoadOp.Clear,
            StoreOp = AttachmentStoreOp.Store,
            ClearValue = new ClearValue() { DepthStencil = new ClearDepthStencilValue(1, 0) }
        };

        renderingInfo = new()
        {
            RenderArea = new Rect2D(){Offset = {X = 0, Y = 0}, Extent = {Width = 1920, Height = 1080}},
            LayerCount = 1,
            ColorAttachmentCount = 1,
            PColorAttachments = (RenderingAttachmentInfo*)&colorAttachmentInfo,
            PDepthAttachment = (RenderingAttachmentInfo*)&depthAttachmentInfo
        };
        //begin dynamic rendering
        vk!.CmdBeginRendering(buffer, (RenderingInfo*)&renderingInfo);
    }

    protected override void Render(CommandBuffer buffer)
    {
        //Get visisble entities
        var visibleEntities = _cullingSystem.GetVisibleEntities();
        foreach (var entity in visibleEntities)
        {
            var meshComponent = entity.GetComponent<MeshComponent>();
            var transformComponent = entity.GetComponent<TransformComponent>();

            if (meshComponent != null && transformComponent != null) 
            {
                //bind pipeline for Gbuffer rendering
                //...
                
                //Set model matrix
                //...
                
                //Draw mesh
                //...
            }
        }
    }

    protected override void EndPass(CommandBuffer buffer)
    {
        //End dynamic rendering
        vk!.CmdEndRendering(buffer);
    }
}
//lighting pass for deferred rendering
unsafe class LightingPass : RenderPass
{
    private GeometryPass* _geometryPass;
    
    private List<Light> lights;
    public LightingPass(Device device, string name, GeometryPass* gPass) : base(device, name)
    {
        //add dependency to geometry pass
        AddDependency(gPass->GetName());
    }
    public void AddLight(Light light) => lights.Add(light);
    public void RemoveLight(Light light) => lights.Remove(light);

    protected override void BeginPass(CommandBuffer buffer)
    {
        //Begin rendering with dynamic rendering
        RenderingInfoKHR renderingInfo;
        //setup color attachment
        RenderingAttachmentInfoKHR colorAttachment = new()
        {
            ImageView = GetRenderTarget()->GetColorImageView(),
            ImageLayout = ImageLayout.ColorAttachmentOptimal,
            LoadOp = AttachmentLoadOp.Clear,
            StoreOp = AttachmentStoreOp.Store,
            ClearValue = new ClearValue() { Color = new ClearColorValue(0, 0, 0, 1) }
        };
        
        //configure rendering info
        renderingInfo = new()
        {
            RenderArea = new Rect2D(){Offset = {X = 0, Y = 0}, Extent = {Width = 1920, Height = 1080}},
            LayerCount = 1,
            ColorAttachmentCount = 1,
            PColorAttachments = (RenderingAttachmentInfo*)&colorAttachment
        };
        //begin dynamic rendering
        vk!.CmdBeginRendering(buffer, (RenderingInfo*)&renderingInfo);
        
        
    }

    protected override void Render(CommandBuffer buffer)
    {
        //bind gbuffer textures from the geometry pass
        var gBuffer = _geometryPass->GetRenderTarget();
        
        //set up descriptor sets for gbuffer textures
        //With dynamic rendering we access the gbuffer textures directly as shader resources
        //rather than subpass inputs
        
        //Render fullscreen quad with lighting shader
        
        //for each light 
        foreach (var light in lights)
        {
            //set light properties
            //...
            
            //draw light volume
            //...
        }
    }

    protected override void EndPass(CommandBuffer buffer)
    {
        //end dynamic rendering
        vk!.CmdEndRendering(buffer);
    }
}

unsafe class PostProcessPass : RenderPass
{
    LightingPass* _lightingPass;
    List<PostProcessEffect> postProcessEffects;
    public PostProcessPass(Device device, string name, LightingPass* lPass) : base(device, name)
    {
        //add dependency to lighting pass
        AddDependency(lPass->GetName());
    }
    
    public void AddEffect(PostProcessEffect effect) => postProcessEffects.Add(effect);
    public void RemoveEffect(PostProcessEffect effect) => postProcessEffects.Remove(effect);

    protected override void BeginPass(CommandBuffer buffer)
    {
        //begin rednering with dynamic rendering
        RenderingInfoKHR renderingInfo;
        
        //Set up color attachment
        RenderingAttachmentInfoKHR colorAttachment = new()
        {
            ImageView = GetRenderTarget()->GetColorImageView(),
            ImageLayout = ImageLayout.ColorAttachmentOptimal,
            LoadOp = AttachmentLoadOp.Clear,
            StoreOp = AttachmentStoreOp.Store,
            ClearValue = new ClearValue() { Color = new ClearColorValue(0, 0, 0, 1) }
        };
        
        //configure rendering info
        renderingInfo = new()
        {
            RenderArea = new Rect2D() { Offset = { X = 0, Y = 0 }, Extent = { Width = 1920, Height = 1080 } },
            LayerCount = 1,
            ColorAttachmentCount = 1,
            PColorAttachments = (RenderingAttachmentInfo*)&colorAttachment
        };
        
        //begin dynamic rendering
        vk!.CmdBeginRendering(buffer, (RenderingInfo*)&renderingInfo);
    }

    protected override void Render(CommandBuffer buffer)
    {
        //with dynamic rendering each effect can set up its own rendering state
        //and access input textures directly as shader resources
        
        //apply each post process effect
        foreach (var effect in postProcessEffects)
        {
            effect.Apply(buffer);
        }
    }

    protected override void EndPass(CommandBuffer buffer)
    {
        //end dynamic rendering
        vk!.CmdEndRendering(buffer);
    }
}

unsafe class RenderTarget : IDisposable
{
    private Vk? vk = Vk.GetApi();
    Device device;
    
    
    private Image colorImage = default;
    private DeviceMemory colorImageMemory = default;
    private ImageView colorImageView = default;
    
    private Image depthImage = default;
    private DeviceMemory depthImageMemory = default;
    private ImageView depthImageView = default;

    private uint width;
    private uint height;

    public RenderTarget(Device device, uint w, uint h)
    {
        this.device = device; 
        width = w;
        height = h;
        //Create color and depth images
        CreateColorResources();
        CreateDepthResources();
        
        //Note: with dynamic rendering we dont need to create VkRenderPass or VkFrameBuffer objects.
        //Instead we just create the images and imageviews that will be used directly with vkCmdBeginRendering
        
    }
    
    public ImageView GetColorImageView() => colorImageView;
    public ImageView GetDepthImageView() => depthImageView;
    
    uint GetWidth() => width;
    uint GetHeight() => height;

    //TODO: implement Image creations here later
    private void CreateColorResources()
    {
        
    }

    private void CreateDepthResources()
    {
        
    }
    
    
    
    
    
    public void Dispose()
    {
        if(colorImageView.Handle != 0) vk.DestroyImageView(device, colorImageView, null); 
        if(colorImageMemory.Handle != 0) vk.FreeMemory(device, colorImageMemory, null); 
        if(colorImage.Handle != 0) vk.DestroyImage(device, colorImage, null); 
        
        if(depthImageView.Handle != 0) vk.DestroyImageView(device, depthImageView, null); 
        if(depthImageMemory.Handle != 0) vk.FreeMemory(device, depthImageMemory, null); 
        if(depthImage.Handle != 0) vk.DestroyImage(device, depthImage, null); 
        
    }
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

[StructLayout(LayoutKind.Sequential)]
public unsafe struct ImageResource : IDisposable
{
    private readonly Vk     _vk;
    private readonly Device _device;
    private bool            _disposed;
    public bool IsAllocated { get; set; }

    public string _name { get; }
    public Format _format;
    Extent2D _extent;
    ImageUsageFlags _usage;
    public ImageLayout _initialLayout;
    public ImageLayout _finalLayout;
    
    
    public VkImage Image;
    public DeviceMemory ImageMemory;
    public ImageView ImageView;
    

    public ImageResource(
        Vk vk, Device device,
        string name, Format format, Extent2D extent,
        ImageUsageFlags usage,
        ImageLayout initialLayout = ImageLayout.Undefined,
        ImageLayout finalLayout   = ImageLayout.ShaderReadOnlyOptimal)
    {
        _vk = vk;
        _device = device;
        _name = name;
        _format = format;
        _extent = extent;
        _usage = usage;
        _initialLayout = initialLayout;
        _finalLayout = finalLayout;
    }
    public void Allocate(PhysicalDevice physicalDevice)
    {
        //configure image creation info based on resource properties
        ImageCreateInfo imageInfo = new()
        {
            SType = StructureType.ImageCreateInfo,
            ImageType = ImageType.Type2D,
            Format = _format,
            Extent = new Extent3D(_extent.Width, _extent.Height, 1),
            MipLevels = 1,
            ArrayLayers = 1,
            Samples = SampleCountFlags.Count1Bit,
            Tiling = ImageTiling.Optimal,
            Usage = _usage,
            SharingMode = SharingMode.Exclusive,
            InitialLayout = _initialLayout
        };
        if (_vk.CreateImage(_device, ref imageInfo, null, out Image) != Result.Success)
        {
            throw new Exception("Failed to create image for resource " + _name);
        }
        
        //alloc backing memory
        _vk.GetImageMemoryRequirements(_device, Image, out var memReqs);

        var allocInfo = new MemoryAllocateInfo()
        {
            SType = StructureType.MemoryAllocateInfo,
            AllocationSize = memReqs.Size,
            MemoryTypeIndex = Renderer.FindMemoryType(_vk, physicalDevice, memReqs.MemoryTypeBits,
                MemoryPropertyFlags.DeviceLocalBit)
        };

        if (_vk.AllocateMemory(_device, ref allocInfo, null, out ImageMemory) != Result.Success)
        {
            throw new Exception("Failed to allocate memory for image " + _name);
        };
        if (_vk.BindImageMemory(_device, Image, ImageMemory, 0) != Result.Success)
        {
            throw new Exception("Failed to bind image memory for resource " + _name);
        }
        
        // ── Create ImageView ──────────────────────────────────
        bool isDepth = _format is Format.D32Sfloat
            or Format.D24UnormS8Uint
            or Format.D16Unorm;
        
        //Create image view
        var viewInfo = new ImageViewCreateInfo()
        {
            SType = StructureType.ImageViewCreateInfo,
            Image = Image,
            ViewType = ImageViewType.Type2D,
            Format = _format,
            SubresourceRange = new ImageSubresourceRange()
            {
                AspectMask = isDepth ? ImageAspectFlags.DepthBit : ImageAspectFlags.ColorBit,
                BaseMipLevel = 0,
                LevelCount = 1,
                BaseArrayLayer = 0,
                LayerCount = 1
            }
        };

        if (_vk.CreateImageView(_device, ref viewInfo, null, out ImageView) != Result.Success)
        {
            throw new Exception("Failed to create image view for resource " + _name);
        };
        IsAllocated = true;

    }

    

    public void Dispose()
    {
        if (_disposed) return;
 
        // Destroy in reverse-creation order — mirrors C++ RAII destruction
        if (ImageView.Handle   != 0) _vk.DestroyImageView(_device, ImageView,   null);
        if (ImageMemory.Handle != 0) _vk.FreeMemory      (_device, ImageMemory, null);
        if (Image.Handle  != 0)      _vk.DestroyImage    (_device, Image,  null);
 
        _disposed = true;
    }

}

public unsafe struct Pass
{
    public string _name;
    public List<string> _inputs;//names of the resources this pass reads from
    public List<string> _outputs;//names of the resources this pass writes to

    public Pass(string name, List<string> inputs, List<string> outputs)
    {
        _name = name;
        _inputs = inputs;
        _outputs = outputs;
    }
    
    public void AddInput(string input) => _inputs.Add(input);
    public void AddOutput(string output) => _outputs.Add(output);

    public Action<CommandBuffer> ExecuteFunc { get; set; } = _ => { };
    
    
    // Builder helpers — lets callers chain 
    public Pass ReadsFrom (string resourceName) { _inputs .Add(resourceName); return this; }
    public Pass WritesTo  (string resourceName) { _outputs.Add(resourceName); return this; }
    public Pass Executes  (Action<CommandBuffer> fn) { ExecuteFunc = fn; return this; }
}

/// <summary>
/// Stores which pass signals the semaphore and which pass waits on it.
/// </summary>
public readonly struct SemaphorePair
{
    public int SignalPassIndex { get; init; }
    public int WaitPassIndex   { get; init; }
}