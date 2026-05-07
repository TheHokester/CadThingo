using System.Numerics;
using CadThingo.VulkanEngine.Renderer;
using Silk.NET.Vulkan;
using ImGuiNET;
using Silk.NET.Core.Native;

namespace CadThingo.VulkanEngine.ImGui;

public unsafe class ImGuiVulkanUtils : IDisposable, IEventListener
{
    Vk vk = Globals.vk;

    //Core vulkan resources for UI rendering
    Buffer vertexBuffer; //Contains all the vertices of the UI

    Buffer indexBuffer; //Contains all the indices of the UI

    //geometry buffers memory
    DeviceMemory vertexBufferMemory;
    DeviceMemory indexBufferMemory;
    void* vertexBufferMapped;
    void* indexBufferMapped;

    uint vertexCount; //current number of vertices in the UI for draw cmds
    uint indexCount; //current number of indices in the UI for Draw cmds

    //texture for the UI font, contains the sampler, image, image view and memory.
    Texture fontTexture;

    //Vulkan pipeline infrastructure for UI rendering
    PipelineCache pipelineCache; //for fast loading of pipelines
    PipelineLayout pipelineLayout; //UI pipeline layout
    Pipeline pipeline; //UI pipeline
    DescriptorPool descriptorPool; //for allocating descriptor sets
    DescriptorSetLayout descriptorSetLayout; //layout defining shader bindings for UI
    DescriptorSet descriptorSet; //actual resource bindings for font tex

    //Vulkan Engine context 
    //references connect our ui system to the rest of the engine
    Renderer.Renderer renderer;

    Device device; //Primary device for resource allocation
    PhysicalDevice physicalDevice; //for validation queries
    Queue graphicsQueue; //for work submission
    uint graphicsQueueFamilyIndex = 0; //for validation

    // UI state management and rendering configuration
    // These members control the visual appearance and dynamic behavior of the UI system
    ImGuiStylePtr vulkanStyle = new();

    // Push constants for efficient per-frame parameter updates
    // This structure enables fast updates of transformation and styling data
    struct PushConstBlock
    {
        public Vector2 scale; // UI scaling factors for different screen sizes
        public Vector2 translate; // Translation offset for UI positioning
    }

    PushConstBlock pC;

    // Dynamic state tracking for performance optimization
    bool needsUpdateBuffers = false; // Flag indicating buffer resize requirements

    // Tracks which ImGuiKeys we've already reported as down. Silk.NET's GLFW backend
    // fires KeyDown on every OS-level key-repeat, but ImGui has its own internal
    // repeat handling for IsKeyPressed(repeat=true). Forwarding repeats compounds
    // the two and produces "ludicrous speed" deletes/backspaces in InputText. We
    // only forward true transitions and let ImGui drive repeat internally.
    readonly HashSet<ImGuiKey> _keysDown = new();

    // Modern Vulkan rendering configuration
    PipelineRenderingCreateInfo renderingInfo; //dynamic rendering setup info\
    // Pipeline color format — mirrors renderer.swapChainImageFormat at construction so the
    // UI pipeline matches whatever surface format the swapchain negotiated. Cached because
    // the swapchain format is fixed for the lifetime of the swapchain; if the swapchain is
    // recreated with a different format, this object must be recreated too.
    Format colorFormat;

    uint width;
    uint height;

    public ImGuiVulkanUtils(Renderer.Renderer renderer, uint graphicsQueueFamilyIndex)
    {
        this.renderer = renderer;
        this.device = renderer.device;
        this.physicalDevice = renderer.physicalDevice;
        this.graphicsQueue = renderer.graphicsQueue;
        this.graphicsQueueFamilyIndex = graphicsQueueFamilyIndex;
        this.colorFormat = renderer.swapChainImageFormat;
    }

    public void Dispose()
    {
        Engine.EventBus.RemoveListener(this);

        //Destroy resources in reverse creation order
        fontTexture.Dispose();
        
        vk!.DestroyBuffer(device, indexBuffer, null);
        vk!.FreeMemory(device, indexBufferMemory, null);
        vk!.DestroyBuffer(device, vertexBuffer, null);
        vk!.FreeMemory(device, vertexBufferMemory, null);
        
        vk!.DestroyPipeline(device, pipeline, null);
        vk!.DestroyPipelineLayout(device, pipelineLayout, null);
        
        fixed(DescriptorSet* pDS = &descriptorSet)
            vk!.FreeDescriptorSets(device, descriptorPool, 1, pDS);
        vk!.DestroyDescriptorSetLayout(device, descriptorSetLayout, null);
        vk!.DestroyDescriptorPool(device, descriptorPool, null);
        
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Initialize the ImGui context
    /// </summary>
    /// <param name="width"></param>
    /// <param name="height"></param>
    public void init(float width, float height)
    {
        this.width = (uint)width;
        this.height = (uint)height;
        //initialize ImGui context
        ImGuiNET.ImGui.CreateContext();

        //Config Imgui
        ImGuiIOPtr io = ImGuiNET.ImGui.GetIO();
        io.ConfigFlags |= ImGuiConfigFlags.NavEnableKeyboard;
        io.ConfigFlags |= ImGuiConfigFlags.DockingEnable;

        //Set display size
        io.DisplaySize = new Vector2(width, height);
        io.DisplayFramebufferScale = new Vector2(1, 1);

        //Setup Style
        vulkanStyle = ImGuiNET.ImGui.GetStyle();
        vulkanStyle.Colors[(int)ImGuiCol.TitleBg] = new Vector4(0.1f, 0.1f, 0.1f, 1.0f);
        vulkanStyle.Colors[(int)ImGuiCol.TitleBgActive] = new Vector4(0.1f, 0.1f, 0.1f, 1.0f);
        vulkanStyle.Colors[(int)ImGuiCol.MenuBarBg] = new Vector4(0.1f, 0.1f, 0.1f, 1.0f);
        vulkanStyle.Colors[(int)ImGuiCol.Header] = new Vector4(0.2f, 0.2f, 0.2f, 1.0f);
        vulkanStyle.Colors[(int)ImGuiCol.CheckMark] = new Vector4(0.7f, 0.7f, 0.7f, 1.0f);

        //Apply default style
        SetStyle(0);
        initResources();

        // Subscribe to keyboard, mouse-move, mouse-button, and scroll events.
        // EventCategory.Input matches all of those (Keyboard | Mouse | MouseButton
        // are all flagged with Input).
        Engine.EventBus.AddListener(this, EventCategory.Input);

        // Hook Silk.NET's KeyChar directly. The OS/GLFW text-input pipeline produces
        // proper Unicode codepoints here (handles shift, layouts, IME, etc.) — that
        // is the only correct source for AddInputCharacter. Doing it from KeyDown
        // by casting an ImGuiKey enum value gives garbage codepoints (rendered as ?).
        if (Engine.keyboard != null)
        {
            Engine.keyboard.KeyChar += (_, c) =>
            {
                ImGuiNET.ImGui.GetIO().AddInputCharacter(c);
            };
        }
    }

    /// <summary>
    /// Initialize the vulkan resource for rendering
    /// </summary>
    public void initResources()
    {
        ImGuiIOPtr io = ImGuiNET.ImGui.GetIO();
        io.Fonts.GetTexDataAsRGBA32(out byte* fontData, out var texWidth, out var texHeight);

        //calculate memory reqs.
        ulong uploadSize = (ulong)texWidth * (ulong)texHeight * 4;

        Extent3D fontExtent = new()
        {
            Width = (uint)texWidth,
            Height = (uint)texHeight,
            Depth = 1
        };
        fontTexture = Texture.CreateTextureFromMemory(renderer, fontData, (uint)texWidth,
            (uint)texHeight, Format.R8G8B8A8Unorm, fontExtent);

        // Eagerly allocate vertex/index buffers at a reasonable starting size so the
        // first frame doesn't pay create+map cost. Sizing covers a typical demo UI;
        // updateBuffers() resizes on demand if a frame outgrows them.
        const uint initialVertexCount = 8192;
        const uint initialIndexCount = 16384;
        ulong initialVertexBytes = initialVertexCount * (ulong)sizeof(ImDrawVert);
        ulong initialIndexBytes = initialIndexCount * sizeof(ushort);

        renderer.CreateBuffer(initialVertexBytes, BufferUsageFlags.VertexBufferBit,
            MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit,
            out vertexBuffer, out vertexBufferMemory);
        vertexCount = initialVertexCount;
        void* vMapped = null;
        vk!.MapMemory(device, vertexBufferMemory, 0, initialVertexBytes, 0, ref vMapped);
        vertexBufferMapped = vMapped;

        renderer.CreateBuffer(initialIndexBytes, BufferUsageFlags.IndexBufferBit,
            MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit,
            out indexBuffer, out indexBufferMemory);
        indexCount = initialIndexCount;
        void* iMapped = null;
        vk!.MapMemory(device, indexBufferMemory, 0, initialIndexBytes, 0, ref iMapped);
        indexBufferMapped = iMapped;

        CreateDescriptorResources();
        CreatePipelineResources();
        CreateImGuiPipeline();
    }

    /// <summary>
    /// Set the style of the UI
    /// </summary>
    /// <param name="index"></param>
    public void SetStyle(uint index)
    {
        ImGuiStylePtr style = ImGuiNET.ImGui.GetStyle();

        switch (index)
        {
            case 0: style = vulkanStyle; break;
            case 1: ImGuiNET.ImGui.StyleColorsClassic(); break;
            case 2: ImGuiNET.ImGui.StyleColorsDark(); break;
            case 3: ImGuiNET.ImGui.StyleColorsLight(); break;
            default: style = vulkanStyle; break;
        }
    }

    //Frame by frame rendering operations

    /// <summary>
    /// Begin a new ImGui frame and generate geometry
    /// </summary>
    /// <returns></returns>
    public bool newFrame()
    {
        // Without a real DeltaTime, ImGui's clock advances at a fixed 1/60 per
        // NewFrame. At very high framerates that makes its internal timers (key
        // repeat, hover delays, animations) fire many times faster than wall-clock
        // — e.g. "ludicrous speed" backspace/delete. Clamp guards against the
        // first frame and any pause-induced spikes that would jump ImGui state.
        ImGuiIOPtr io = ImGuiNET.ImGui.GetIO();
        io.DeltaTime = Math.Clamp(Engine.DeltaTime, 1.0f / 10000.0f, 1.0f / 15.0f);

        ImGuiNET.ImGui.NewFrame();
        ImGuiNET.ImGui.ShowDemoWindow();
        ImGuiNET.ImGui.Render();

        var drawData = ImGuiNET.ImGui.GetDrawData();
        if (drawData.NativePtr != null && drawData.CmdListsCount > 0)
        {
            if (drawData.TotalVtxCount > vertexCount || drawData.TotalIdxCount > indexCount)
            {
                needsUpdateBuffers = true;
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Upload updated geometry buffers to GPU.
    ///
    /// Buffers are HostVisible+HostCoherent and persistently mapped, so each
    /// CmdList copies straight into mapped memory and the GPU reads from there
    /// — no staging buffer, no transfer command, one memcpy per CmdList.
    /// Standard ImGui Vulkan-backend pattern; vertex counts are small enough
    /// that fetching from host memory is well below noise.
    /// </summary>
    public void updateBuffers()
    {
        ImDrawDataPtr drawData = ImGuiNET.ImGui.GetDrawData();
        if (drawData.NativePtr == null || drawData.CmdListsCount == 0)
        {
            return;
        }

        ulong vertexBufferSize = (ulong)drawData.TotalVtxCount * (ulong)sizeof(ImDrawVert);
        ulong indexBufferSize = (ulong)drawData.TotalIdxCount * sizeof(ushort);

        // (Re)create + map vertex buffer when the frame's geometry outgrows it.
        if (drawData.TotalVtxCount > vertexCount)
        {
            if (vertexBufferMemory.Handle != 0) vk!.UnmapMemory(device, vertexBufferMemory);
            renderer.DestroyBuffer(vertexBuffer, vertexBufferMemory);
            renderer.CreateBuffer(vertexBufferSize, BufferUsageFlags.VertexBufferBit,
                MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit,
                out vertexBuffer, out vertexBufferMemory);
            vertexCount = (uint)drawData.TotalVtxCount;
            void* mapped = null;
            vk!.MapMemory(device, vertexBufferMemory, 0, vertexBufferSize, 0, ref mapped);
            vertexBufferMapped = mapped;
        }

        if (drawData.TotalIdxCount > indexCount)
        {
            if (indexBufferMemory.Handle != 0) vk!.UnmapMemory(device, indexBufferMemory);
            renderer.DestroyBuffer(indexBuffer, indexBufferMemory);
            renderer.CreateBuffer(indexBufferSize, BufferUsageFlags.IndexBufferBit,
                MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit,
                out indexBuffer, out indexBufferMemory);
            indexCount = (uint)drawData.TotalIdxCount;
            void* mapped = null;
            vk!.MapMemory(device, indexBufferMemory, 0, indexBufferSize, 0, ref mapped);
            indexBufferMapped = mapped;
        }

        // One memcpy per CmdList, straight into mapped GPU memory.
        var vtxDst = new Span<ImDrawVert>(vertexBufferMapped, drawData.TotalVtxCount);
        var idxDst = new Span<ushort>(indexBufferMapped, drawData.TotalIdxCount);
        int vtxOffset = 0;
        int idxOffset = 0;
        for (int i = 0; i < drawData.CmdListsCount; i++)
        {
            var cmdList = drawData.CmdLists[i];
            var srcVtx = new ReadOnlySpan<ImDrawVert>((void*)cmdList.VtxBuffer.Data, cmdList.VtxBuffer.Size);
            var srcIdx = new ReadOnlySpan<ushort>((void*)cmdList.IdxBuffer.Data, cmdList.IdxBuffer.Size);
            srcVtx.CopyTo(vtxDst.Slice(vtxOffset));
            srcIdx.CopyTo(idxDst.Slice(idxOffset));
            vtxOffset += srcVtx.Length;
            idxOffset += srcIdx.Length;
        }
    }
    public void OnEvent(Event e)
    {
        ImGuiIOPtr io = ImGuiNET.ImGui.GetIO();
        switch (e)
        {
            case KeyPressEvent kp:
                if (ImGuiHelpers.TryMapGlfwKey(kp.GetKeyCode, out var imKeyDown))
                {
                    // Suppress OS key-repeats: only forward true transitions. ImGui
                    // runs its own repeat logic for IsKeyPressed(repeat=true).
                    if (_keysDown.Add(imKeyDown))
                    {
                        io.AddKeyEvent(imKeyDown, true);
                        UpdateImGuiModifiers(io);
                    }
                    // Text input is delivered via Silk's KeyChar event (see init()).
                }
                break;
            case KeyReleaseEvent kr:
                if (ImGuiHelpers.TryMapGlfwKey(kr.GetKeyCode, out var imKeyUp))
                {
                    if (_keysDown.Remove(imKeyUp))
                    {
                        io.AddKeyEvent(imKeyUp, false);
                        UpdateImGuiModifiers(io);
                    }
                }
                break;
            case MouseMoveEvent mm:
                io.AddMousePosEvent(mm.GetAbsX(), mm.GetAbsY());
                break;
            case MouseKeyDownEvent mbd:
                io.AddMouseButtonEvent((int)mbd.GetButton, true);
                break;
            case MouseKeyReleaseEvent mbu:
                io.AddMouseButtonEvent((int)mbu.GetButton, false);
                break;
            case MouseScrollEvent ms:
                io.AddMouseWheelEvent(ms.GetX(), ms.GetY());
                break;
        }
    }

    // Mirror modifier state to ImGui's dedicated mod keys. ImGui needs these set
    // for shortcuts like Ctrl+A / Shift+Arrow / Alt+click inside InputText.
    void UpdateImGuiModifiers(ImGuiIOPtr io)
    {
        io.AddKeyEvent(ImGuiKey.ModCtrl,
            _keysDown.Contains(ImGuiKey.LeftCtrl)  || _keysDown.Contains(ImGuiKey.RightCtrl));
        io.AddKeyEvent(ImGuiKey.ModShift,
            _keysDown.Contains(ImGuiKey.LeftShift) || _keysDown.Contains(ImGuiKey.RightShift));
        io.AddKeyEvent(ImGuiKey.ModAlt,
            _keysDown.Contains(ImGuiKey.LeftAlt)   || _keysDown.Contains(ImGuiKey.RightAlt));
        io.AddKeyEvent(ImGuiKey.ModSuper,
            _keysDown.Contains(ImGuiKey.LeftSuper) || _keysDown.Contains(ImGuiKey.RightSuper));
    }
    
    /// <summary>
    /// Record rendering commands to command buffer.
    /// </summary>
    /// <param name="cmdBuffer">buffer to record on</param>
    /// <param name="targetView">color attachment image view to render the UI into.
    /// Caller is responsible for transitioning the underlying image into
    /// <see cref="ImageLayout.ColorAttachmentOptimal"/> beforehand and out of it
    /// afterward — this method only records the rendering pass, not barriers.</param>
    public void DrawFrame(CommandBuffer cmdBuffer, ImageView targetView)
    {
        ImDrawDataPtr drawData = ImGuiNET.ImGui.GetDrawData();
        if (drawData.NativePtr == null || drawData.CmdListsCount == 0)
        {
            return;
        }
        
        // LoadOp.Load preserves the FinalColor blit underneath; the alpha-blended
        // pipeline composites the UI on top.
        RenderingAttachmentInfo colorAttachment = new()
        {
            SType = StructureType.RenderingAttachmentInfo,
            ImageView = targetView,
            ImageLayout = ImageLayout.ColorAttachmentOptimal,
            LoadOp = AttachmentLoadOp.Load,
            StoreOp = AttachmentStoreOp.Store,
        };
        RenderingInfo renderInfo;

        renderInfo = new()
        {
            SType = StructureType.RenderingInfo,
            RenderArea = new Rect2D(new(0, 0), new(width, height)),
            LayerCount = 1,
            ColorAttachmentCount = 1,
            PColorAttachments = &colorAttachment
        };

        vk!.CmdBeginRendering(cmdBuffer, &renderInfo);

        //bind pipeline
        vk!.CmdBindPipeline(cmdBuffer, PipelineBindPoint.Graphics, pipeline);

        //config viewport
        Viewport viewport = new()
        {
            Width = (float)width!,
            Height = (float)height!,
            MinDepth = 0.0f,
            MaxDepth = 1.0f
        };
        vk!.CmdSetViewport(cmdBuffer, 0, 1, &viewport);
        //convert from imgui coords to NDC via simple scale/translate
        pC.scale = new Vector2(2.0f / drawData.DisplaySize.X, 2.0f / drawData.DisplaySize.Y);
        pC.translate = new Vector2(-1) - drawData.DisplayPos * pC.scale;
        PushConstBlock* pPC = stackalloc PushConstBlock[]
        {
            pC
        };
        vk!.CmdPushConstants(cmdBuffer, pipelineLayout, ShaderStageFlags.VertexBit, 0, (uint)sizeof(PushConstBlock),
            pPC);
        //bind buffers
        Buffer* pVB = stackalloc Buffer[] { vertexBuffer };
        ulong offset = 0;
        vk!.CmdBindVertexBuffers(cmdBuffer, 0, 1, pVB, &offset);
        vk!.CmdBindIndexBuffer(cmdBuffer, indexBuffer, 0, IndexType.Uint16);
        
        //bind font (and any UI texs) for this draw, move inside loop if needed in future.
        fixed (DescriptorSet* pDS = &descriptorSet)
        {
            vk!.CmdBindDescriptorSets(cmdBuffer, PipelineBindPoint.Graphics, pipelineLayout,
                0u, 1u, pDS, null);
        }
        
        int vertexOffset = 0;
        int indexOffset = 0;
        for (int i = 0; i < drawData.CmdListsCount; i++)
        {
            var cmdList = drawData.CmdLists[i];
            for (int j = 0; j < cmdList.CmdBuffer.Size; j++)
            {
                var cmd = cmdList.CmdBuffer[j];
                //clip per draw call
                Rect2D scissor = new();
                int sX = Math.Max((int)cmd.ClipRect.X, 0);
                int sY = Math.Max((int)cmd.ClipRect.Y, 0);
                scissor.Offset.X = sX;
                scissor.Offset.Y = sY;
                scissor.Extent.Width = (uint)(cmd.ClipRect.Z - sX);
                scissor.Extent.Height = (uint)(cmd.ClipRect.W - sY);
                vk!.CmdSetScissor(cmdBuffer, 0, 1, &scissor);

                

                //issue indexed draw for this UI data
                vk!.CmdDrawIndexed(cmdBuffer, cmd.ElemCount, 1, (uint)indexOffset, vertexOffset, 0);
                indexOffset += (int)cmd.ElemCount;
            }
            
            vertexOffset += cmdList.VtxBuffer.Size;
        }
        vk!.CmdEndRendering(cmdBuffer);
    }
    
    
    
    private void CreateDescriptorResources()
    {
        //Create descriptor pool
        var poolSize = new DescriptorPoolSize
        {
            Type = DescriptorType.CombinedImageSampler,
            DescriptorCount = 1
        };
        var poolInfo = new DescriptorPoolCreateInfo
        {
            SType = StructureType.DescriptorPoolCreateInfo,
            MaxSets = 1,
            PoolSizeCount = 1,
            PPoolSizes = &poolSize
        };
        vk.CreateDescriptorPool(device, &poolInfo, null, out descriptorPool);

        //Create descriptor set layout defining shader resources interface
        //Must match layout defined in imgui shader
        DescriptorSetLayoutBinding fontBinding = new()
        {
            DescriptorType = DescriptorType.CombinedImageSampler,
            DescriptorCount = 1,
            Binding = 0,
            StageFlags = ShaderStageFlags.FragmentBit
        };
        DescriptorSetLayoutCreateInfo layoutInfo = new()
        {
            SType = StructureType.DescriptorSetLayoutCreateInfo,
            BindingCount = 1,
            PBindings = &fontBinding,
        };
        vk!.CreateDescriptorSetLayout(device, &layoutInfo, null, out descriptorSetLayout);

        //Allocate descriptor set from pool with defined layout
        DescriptorSetAllocateInfo allocInfo = new()
        {
            SType = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool = descriptorPool,
            DescriptorSetCount = 1
        };
        var layouts = stackalloc DescriptorSetLayout[] { descriptorSetLayout };
        allocInfo.PSetLayouts = layouts;

        vk!.AllocateDescriptorSets(device, &allocInfo, out descriptorSet);

        //update descriptorset with font tex and sampler
        DescriptorImageInfo fontInfo = new()
        {
            ImageLayout = ImageLayout.ShaderReadOnlyOptimal,
            ImageView = fontTexture.View,
            Sampler = fontTexture.Sampler
        };

        var writes = stackalloc WriteDescriptorSet[]
        {
            new WriteDescriptorSet
            {
                SType = StructureType.WriteDescriptorSet,
                DstSet = descriptorSet,
                DstBinding = 0,
                DstArrayElement = 0,
                DescriptorCount = 1,
                DescriptorType = DescriptorType.CombinedImageSampler,
                PImageInfo = &fontInfo
            }
        };
        vk!.UpdateDescriptorSets(device, 1, writes, 0, null);
    }

    private void CreatePipelineResources()
    {
        PipelineCacheCreateInfo cacheInfo = new()
        {
            SType = StructureType.PipelineCacheCreateInfo
        };
        vk!.CreatePipelineCache(device, &cacheInfo, null, out pipelineCache);

        //Create pipeline layout
        PushConstantRange pcr = new()
        {
            Offset = 0,
            StageFlags = ShaderStageFlags.VertexBit,
            Size = (uint)sizeof(PushConstBlock)
        };
        var layouts = stackalloc DescriptorSetLayout[] { descriptorSetLayout };
        PipelineLayoutCreateInfo layoutInfo = new()
        {
            SType = StructureType.PipelineLayoutCreateInfo,
            SetLayoutCount = 1,
            PSetLayouts = layouts,
            PushConstantRangeCount = 1,
            PPushConstantRanges = &pcr
        };

        vk!.CreatePipelineLayout(device, &layoutInfo, null, out pipelineLayout);
    }

    private void CreateImGuiPipeline()
    {
        byte[] shaderCode = File.ReadAllBytes
            ("C:\\Users\\jamie\\RiderProjects\\CadThingo\\CadThingo\\Assets\\Shaders\\ImGui.spv");

        var shaderInfo = new ShaderModuleCreateInfo()
        {
            SType = StructureType.ShaderModuleCreateInfo,
            CodeSize = (nuint)shaderCode.Length,
        };
        fixed (byte* pCode = shaderCode)
        {
            shaderInfo.PCode = (uint*)pCode;
        }

        if (vk!.CreateShaderModule(device, &shaderInfo, null, out var shader) != Result.Success)
        {
            throw new Exception("Failed to create shader module");
        }

        //config vertex stage 
        PipelineShaderStageCreateInfo vertShaderInfo = new()
        {
            SType = StructureType.PipelineShaderStageCreateInfo,
            Stage = ShaderStageFlags.VertexBit,
            Module = shader,
            PName = (byte*)SilkMarshal.StringToPtr("VSMain")
        };
        //config frag stage
        PipelineShaderStageCreateInfo fragShaderInfo = new()
        {
            SType = StructureType.PipelineShaderStageCreateInfo,
            Stage = ShaderStageFlags.FragmentBit,
            Module = shader,
            PName = (byte*)SilkMarshal.StringToPtr("PSMain")
        };
        var shaderStageCount = 2;
        var shaderStages = stackalloc PipelineShaderStageCreateInfo[]
        {
            vertShaderInfo,
            fragShaderInfo
        };

        // ImDrawVert: pos(vec2 @0), uv(vec2 @8), col(packed RGBA8 @16) — 20 bytes.
        VertexInputBindingDescription vertexBinding = new()
        {
            Binding = 0,
            Stride = (uint)sizeof(ImDrawVert),
            InputRate = VertexInputRate.Vertex
        };
        var vertexAttribs = stackalloc VertexInputAttributeDescription[3]
        {
            new() { Location = 0, Binding = 0, Format = Format.R32G32Sfloat, Offset = 0 },
            new() { Location = 1, Binding = 0, Format = Format.R32G32Sfloat, Offset = 8 },
            new() { Location = 2, Binding = 0, Format = Format.R8G8B8A8Unorm, Offset = 16 },
        };
        PipelineVertexInputStateCreateInfo vertexInputInfo = new()
        {
            SType = StructureType.PipelineVertexInputStateCreateInfo,
            VertexBindingDescriptionCount = 1,
            PVertexBindingDescriptions = &vertexBinding,
            VertexAttributeDescriptionCount = 3,
            PVertexAttributeDescriptions = vertexAttribs
        };

        PipelineInputAssemblyStateCreateInfo inputAssemblyInfo = new()
        {
            SType = StructureType.PipelineInputAssemblyStateCreateInfo,
            Topology = PrimitiveTopology.TriangleList,
            PrimitiveRestartEnable = false
        };

        // Viewport + scissor are dynamic — drawFrame sets them from DisplaySize and per-cmd ClipRect.
        PipelineViewportStateCreateInfo viewportInfo = new()
        {
            SType = StructureType.PipelineViewportStateCreateInfo,
            ViewportCount = 1,
            ScissorCount = 1
        };

        // 2D UI: no culling, no depth bias.
        PipelineRasterizationStateCreateInfo rasterizer = new()
        {
            SType = StructureType.PipelineRasterizationStateCreateInfo,
            DepthClampEnable = false,
            RasterizerDiscardEnable = false,
            PolygonMode = PolygonMode.Fill,
            CullMode = CullModeFlags.None,
            FrontFace = FrontFace.CounterClockwise,
            DepthBiasEnable = false,
            LineWidth = 1.0f
        };

        PipelineMultisampleStateCreateInfo multisampleInfo = new()
        {
            SType = StructureType.PipelineMultisampleStateCreateInfo,
            SampleShadingEnable = false,
            RasterizationSamples = SampleCountFlags.Count1Bit
        };

        // UI sits on top of the scene — no depth test, no depth writes.
        PipelineDepthStencilStateCreateInfo depthStencilInfo = new()
        {
            SType = StructureType.PipelineDepthStencilStateCreateInfo,
            DepthTestEnable = false,
            DepthWriteEnable = false,
            DepthCompareOp = CompareOp.Always,
            DepthBoundsTestEnable = false,
            StencilTestEnable = false
        };

        // Standard SrcAlpha / OneMinusSrcAlpha blending so anti-aliased edges and
        // translucent panels composite correctly over the underlying frame.
        PipelineColorBlendAttachmentState colorBlendAttachment = new()
        {
            BlendEnable = true,
            SrcColorBlendFactor = BlendFactor.SrcAlpha,
            DstColorBlendFactor = BlendFactor.OneMinusSrcAlpha,
            ColorBlendOp = BlendOp.Add,
            SrcAlphaBlendFactor = BlendFactor.One,
            DstAlphaBlendFactor = BlendFactor.OneMinusSrcAlpha,
            AlphaBlendOp = BlendOp.Add,
            ColorWriteMask = ColorComponentFlags.RBit | ColorComponentFlags.GBit |
                             ColorComponentFlags.BBit | ColorComponentFlags.ABit
        };
        PipelineColorBlendStateCreateInfo colorBlendInfo = new()
        {
            SType = StructureType.PipelineColorBlendStateCreateInfo,
            LogicOpEnable = false,
            LogicOp = LogicOp.Copy,
            AttachmentCount = 1,
            PAttachments = &colorBlendAttachment
        };

        var dynamicStates = stackalloc DynamicState[]
        {
            DynamicState.Viewport,
            DynamicState.Scissor
        };
        PipelineDynamicStateCreateInfo dynamicStateInfo = new()
        {
            SType = StructureType.PipelineDynamicStateCreateInfo,
            DynamicStateCount = 2,
            PDynamicStates = dynamicStates
        };

        // Dynamic rendering — no render pass object. UI targets a single color attachment
        // matching colorFormat; depth/stencil are unused.
        fixed (Format* pColorFormat = &colorFormat)
        {
            renderingInfo = new()
            {
                SType = StructureType.PipelineRenderingCreateInfo,
                ColorAttachmentCount = 1,
                PColorAttachmentFormats = pColorFormat,
                DepthAttachmentFormat = Format.Undefined,
                StencilAttachmentFormat = Format.Undefined
            };

            fixed (PipelineRenderingCreateInfo* pRenderingInfo = &renderingInfo)
            {
                GraphicsPipelineCreateInfo pipelineInfo = new()
                {
                    SType = StructureType.GraphicsPipelineCreateInfo,
                    PNext = pRenderingInfo,
                    StageCount = (uint)shaderStageCount,
                    PStages = shaderStages,
                    PVertexInputState = &vertexInputInfo,
                    PInputAssemblyState = &inputAssemblyInfo,
                    PViewportState = &viewportInfo,
                    PRasterizationState = &rasterizer,
                    PMultisampleState = &multisampleInfo,
                    PDepthStencilState = &depthStencilInfo,
                    PColorBlendState = &colorBlendInfo,
                    PDynamicState = &dynamicStateInfo,
                    Layout = pipelineLayout,
                    RenderPass = default,
                    Subpass = 0,
                    BasePipelineHandle = default,
                    BasePipelineIndex = -1
                };

                if (vk!.CreateGraphicsPipelines(device, pipelineCache, 1, &pipelineInfo, null, out pipeline) !=
                    Result.Success)
                {
                    throw new Exception("Failed to create ImGui graphics pipeline");
                }
            }
        }

        vk!.DestroyShaderModule(device, shader, null);
    }
}