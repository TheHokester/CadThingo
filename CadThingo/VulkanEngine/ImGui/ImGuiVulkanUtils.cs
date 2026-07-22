using System.Numerics;
using CadThingo.VulkanEngine.Renderer;
using Silk.NET.Vulkan;
using ImGuiNET;
using Silk.NET.Core.Native;

namespace CadThingo.VulkanEngine.ImGui;

public unsafe class ImGuiVulkanUtils : IDisposable, IEventListener
{
    Vk vk = Globals.vk;

    // Per-frame vertex/index buffer ring. One slot per frame-in-flight so
    // updateBuffers() can destroy + recreate its own slot freely — the
    // renderer's WaitForFences(inFlightFences[currentFrame]) at the top of
    // DrawFrame guarantees the GPU is done with this slot before we touch it,
    // while the OTHER slots remain untouched and safe for any frame the GPU is
    // still executing. Capacities track per-slot allocated count (not the
    // current frame's required count) so we only grow, never thrash on
    // alternating big/small frames.
    readonly Buffer[]       vertexBuffers      = new Buffer[RenderConfig.MAX_CONCURRENT_FRAMES];
    readonly Buffer[]       indexBuffers       = new Buffer[RenderConfig.MAX_CONCURRENT_FRAMES];
    readonly SubAlloc[]     vertexBufferAllocs = new SubAlloc[RenderConfig.MAX_CONCURRENT_FRAMES];
    readonly SubAlloc[]     indexBufferAllocs  = new SubAlloc[RenderConfig.MAX_CONCURRENT_FRAMES];
    // Mapped pointers held as nint because C# doesn't allow void*[] fields.
    readonly nint[]         vertexBufferMapped = new nint[RenderConfig.MAX_CONCURRENT_FRAMES];
    readonly nint[]         indexBufferMapped  = new nint[RenderConfig.MAX_CONCURRENT_FRAMES];
    readonly uint[]         vertexCapacities   = new uint[RenderConfig.MAX_CONCURRENT_FRAMES];
    readonly uint[]         indexCapacities    = new uint[RenderConfig.MAX_CONCURRENT_FRAMES];

    //texture for the UI font, contains the sampler, image, image view and memory.
    Texture fontTexture;

    //Vulkan pipeline infrastructure for UI rendering
    PipelineCache pipelineCache; //for fast loading of pipelines
    PipelineLayout pipelineLayout; //UI pipeline layout
    Pipeline pipeline; //UI pipeline
    DescriptorPool descriptorPool; //for allocating descriptor sets
    DescriptorSetLayout descriptorSetLayout; //layout defining shader bindings for UI
    DescriptorSet descriptorSet; //actual resource bindings for font tex

    // Second descriptor set sharing the font layout (binding 0 = CombinedImageSampler).
    // Written lazily by the renderer once FinalColor's ImageView exists; sampled by
    // the viewport panel via ImGui.Image(ViewportTextureId, size). Re-written on
    // swapchain recreate because the underlying ImageView changes.
    DescriptorSet viewportDescriptorSet;
    Sampler       viewportSampler;

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

    /// <summary>
    /// Texture ID for the viewport ImGui.Image call. Encodes the viewport
    /// VkDescriptorSet handle so the per-cmd bind loop in DrawFrame can route
    /// the right descriptor for that draw. Zero until WriteViewportDescriptor
    /// has been called with a valid ImageView.
    /// </summary>
    public nint ViewportTextureId =>
        viewportDescriptorSet.Handle == 0 ? 0 : (nint)viewportDescriptorSet.Handle;

    /// <summary>
    /// (Re-)binds the viewport descriptor set to the supplied <paramref name="view"/>.
    /// Caller (renderer) is responsible for invoking this after the render graph's
    /// FinalColor ImageView has been allocated (initial bind) and after every
    /// swapchain recreation (the view changes whenever the graph is rebuilt).
    /// Must be called outside any in-flight frame — i.e. after vkDeviceWaitIdle
    /// or during initialization.
    /// </summary>
    public void WriteViewportDescriptor(ImageView view)
    {
        if (viewportDescriptorSet.Handle == 0) return;

        DescriptorImageInfo info = new()
        {
            ImageLayout = ImageLayout.ShaderReadOnlyOptimal,
            ImageView   = view,
            Sampler     = viewportSampler,
        };
        WriteDescriptorSet write = new()
        {
            SType           = StructureType.WriteDescriptorSet,
            DstSet          = viewportDescriptorSet,
            DstBinding      = 0,
            DstArrayElement = 0,
            DescriptorCount = 1,
            DescriptorType  = DescriptorType.CombinedImageSampler,
            PImageInfo      = &info,
        };
        vk!.UpdateDescriptorSets(device, 1, &write, 0, null);
    }

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

        for (int i = 0; i < RenderConfig.MAX_CONCURRENT_FRAMES; i++)
        {
            if (indexBufferAllocs[i].IsValid)
                renderer.DestroyBuffer(indexBuffers[i], indexBufferAllocs[i]);
            if (vertexBufferAllocs[i].IsValid)
                renderer.DestroyBuffer(vertexBuffers[i], vertexBufferAllocs[i]);
        }
        
        vk!.DestroyPipeline(device, pipeline, null);
        vk!.DestroyPipelineLayout(device, pipelineLayout, null);
        
        // DestroyDescriptorPool implicitly frees every descriptor set allocated
        // from it (font + viewport), so no explicit FreeDescriptorSets call is
        // needed for the viewport set.
        fixed(DescriptorSet* pDS = &descriptorSet)
            vk!.FreeDescriptorSets(device, descriptorPool, 1, pDS);
        vk!.DestroyDescriptorSetLayout(device, descriptorSetLayout, null);
        vk!.DestroyDescriptorPool(device, descriptorPool, null);

        if (viewportSampler.Handle != 0)
            vk!.DestroySampler(device, viewportSampler, null);

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
        fontTexture = Texture.CreateTextureFromMemory(renderer.gfx, fontData, (uint)texWidth,
            (uint)texHeight, Format.R8G8B8A8Unorm, fontExtent);

        // Eagerly allocate one vertex/index buffer per frame-in-flight at a
        // reasonable starting size so the first frames don't pay create+map
        // cost. updateBuffers() grows them on demand at 2x per growth.
        const uint initialVertexCount = 8192;
        const uint initialIndexCount  = 16384;
        for (int i = 0; i < RenderConfig.MAX_CONCURRENT_FRAMES; i++)
        {
            AllocateVertexSlot(i, initialVertexCount);
            AllocateIndexSlot(i,  initialIndexCount);
        }

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
        ImGuiUI.Draw();
        ImGuiNET.ImGui.Render();

        // Per-slot resize is now handled inside updateBuffers(); no need to
        // gate it from here. Return value is unused by callers — kept so the
        // signature doesn't churn unrelated code.
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
    public void updateBuffers(uint frameIndex)
    {
        ImDrawDataPtr drawData = ImGuiNET.ImGui.GetDrawData();
        if (drawData.NativePtr == null || drawData.CmdListsCount == 0)
        {
            return;
        }

        int slot = (int)frameIndex;

        // Grow this slot's buffers when geometry outgrows capacity. Safe to
        // destroy+recreate because the caller has already waited on this slot's
        // in-flight fence — the GPU is done with it. Other slots are
        // untouched, so frames still in flight there keep their valid handles.
        uint requiredVtx = (uint)drawData.TotalVtxCount;
        if (requiredVtx > vertexCapacities[slot])
        {
            uint newCap = GrowCapacity(vertexCapacities[slot], requiredVtx);
            FreeVertexSlot(slot);
            AllocateVertexSlot(slot, newCap);
        }

        uint requiredIdx = (uint)drawData.TotalIdxCount;
        if (requiredIdx > indexCapacities[slot])
        {
            uint newCap = GrowCapacity(indexCapacities[slot], requiredIdx);
            FreeIndexSlot(slot);
            AllocateIndexSlot(slot, newCap);
        }

        // One memcpy per CmdList, straight into mapped GPU memory.
        var vtxDst = new Span<ImDrawVert>((void*)vertexBufferMapped[slot], drawData.TotalVtxCount);
        var idxDst = new Span<ushort>   ((void*)indexBufferMapped[slot],  drawData.TotalIdxCount);
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

    // 2x exponential growth — doubles until capacity covers the requirement.
    // First-time growth from 0 just sizes exactly to required.
    static uint GrowCapacity(uint current, uint required)
    {
        if (current == 0) return required;
        uint cap = current;
        while (cap < required) cap *= 2;
        return cap;
    }

    void AllocateVertexSlot(int slot, uint count)
    {
        ulong bytes = count * (ulong)sizeof(ImDrawVert);
        renderer.CreateBuffer(bytes, BufferUsageFlags.VertexBufferBit,
            MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit,
            out vertexBuffers[slot], out vertexBufferAllocs[slot]);
        vertexCapacities[slot] = count;
        vertexBufferMapped[slot] = (nint)renderer.memAllocator.GetMapped(vertexBufferAllocs[slot]);
    }

    void AllocateIndexSlot(int slot, uint count)
    {
        ulong bytes = count * sizeof(ushort);
        renderer.CreateBuffer(bytes, BufferUsageFlags.IndexBufferBit,
            MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit,
            out indexBuffers[slot], out indexBufferAllocs[slot]);
        indexCapacities[slot] = count;
        indexBufferMapped[slot] = (nint)renderer.memAllocator.GetMapped(indexBufferAllocs[slot]);
    }

    void FreeVertexSlot(int slot)
    {
        if (!vertexBufferAllocs[slot].IsValid) return;
        renderer.DestroyBuffer(vertexBuffers[slot], vertexBufferAllocs[slot]);
        vertexBuffers[slot]       = default;
        vertexBufferAllocs[slot]  = default;
        vertexBufferMapped[slot]  = 0;
        vertexCapacities[slot]    = 0;
    }

    void FreeIndexSlot(int slot)
    {
        if (!indexBufferAllocs[slot].IsValid) return;
        renderer.DestroyBuffer(indexBuffers[slot], indexBufferAllocs[slot]);
        indexBuffers[slot]       = default;
        indexBufferAllocs[slot]  = default;
        indexBufferMapped[slot]  = 0;
        indexCapacities[slot]    = 0;
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
    /// <param name="frameIndex">Which frame-in-flight slot's vertex/index buffer
    /// to bind. Must match the slot fed to <see cref="updateBuffers"/> this
    /// frame and must be guarded by the caller's per-slot fence wait.</param>
    public void DrawFrame(CommandBuffer cmdBuffer, ImageView targetView, uint frameIndex)
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
        //bind buffers — this frame's slot only; other slots may still be
        //in use by the previous in-flight frame and must not be touched.
        int slot = (int)frameIndex;
        Buffer* pVB = stackalloc Buffer[] { vertexBuffers[slot] };
        ulong offset = 0;
        vk!.CmdBindVertexBuffers(cmdBuffer, 0, 1, pVB, &offset);
        vk!.CmdBindIndexBuffer(cmdBuffer, indexBuffers[slot], 0, IndexType.Uint16);
        
        // Per-cmd descriptor binding: font set by default; switch to whichever
        // VkDescriptorSet handle ImGui.Image() encoded into cmd.TextureId, then
        // back. Tracks last-bound handle so font→font sequences don't rebind.
        DescriptorSet boundDs = descriptorSet;
        vk!.CmdBindDescriptorSets(cmdBuffer, PipelineBindPoint.Graphics, pipelineLayout,
            0u, 1u, &boundDs, 0u, null);
        ulong lastBoundHandle = descriptorSet.Handle;

        int vertexOffset = 0;
        int indexOffset = 0;
        for (int i = 0; i < drawData.CmdListsCount; i++)
        {
            var cmdList = drawData.CmdLists[i];
            for (int j = 0; j < cmdList.CmdBuffer.Size; j++)
            {
                var cmd = cmdList.CmdBuffer[j];

                // Resolve which descriptor set this cmd wants. TextureId == 0
                // means "no Image() override" → use the font set.
                ulong wantHandle = cmd.TextureId == 0
                    ? descriptorSet.Handle
                    : (ulong)cmd.TextureId;
                if (wantHandle != lastBoundHandle)
                {
                    boundDs.Handle = wantHandle;
                    vk!.CmdBindDescriptorSets(cmdBuffer, PipelineBindPoint.Graphics, pipelineLayout,
                        0u, 1u, &boundDs, 0u, null);
                    lastBoundHandle = wantHandle;
                }

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

    /// <summary>
    /// Updates ImGui's surface size after a window resize. ImGui's coordinate
    /// space (DisplaySize) and the dynamic-rendering pass dimensions
    /// (width/height fields used by DrawFrame's viewport + RenderArea) both
    /// need this — without the field update the UI would still record into the
    /// pre-resize rectangle and get clipped or stretched against the new
    /// swapchain extent.
    /// </summary>
    public void UpdateScreenSize(uint width, uint height)
    {
        this.width  = width;
        this.height = height;
        ImGuiNET.ImGui.GetIO().DisplaySize = new Vector2(width, height);
    }
    
    private void CreateDescriptorResources()
    {
        //Create descriptor pool — sized for two sets (font + viewport scene-image),
        // both bound to the same single-binding layout below.
        var poolSize = new DescriptorPoolSize
        {
            Type = DescriptorType.CombinedImageSampler,
            DescriptorCount = 2
        };
        var poolInfo = new DescriptorPoolCreateInfo
        {
            SType = StructureType.DescriptorPoolCreateInfo,
            MaxSets = 2,
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

        // Allocate the viewport descriptor set from the same pool & layout.
        // Contents are filled in later via WriteViewportDescriptor — the renderer
        // calls that after FinalColor's ImageView exists (post-graph-compile).
        vk!.AllocateDescriptorSets(device, &allocInfo, out viewportDescriptorSet);

        // Linear filter + clamp-to-edge: matches how DCC tools show a viewport
        // texture — no wrapping artefacts at panel edges, smooth on scale changes.
        SamplerCreateInfo samplerInfo = new()
        {
            SType                  = StructureType.SamplerCreateInfo,
            MagFilter              = Filter.Linear,
            MinFilter              = Filter.Linear,
            MipmapMode             = SamplerMipmapMode.Linear,
            AddressModeU           = SamplerAddressMode.ClampToEdge,
            AddressModeV           = SamplerAddressMode.ClampToEdge,
            AddressModeW           = SamplerAddressMode.ClampToEdge,
            AnisotropyEnable       = false,
            MaxAnisotropy          = 1.0f,
            BorderColor            = BorderColor.IntOpaqueBlack,
            UnnormalizedCoordinates= false,
            CompareEnable          = false,
            CompareOp              = CompareOp.Always,
            MinLod                 = 0,
            MaxLod                 = 0,
            MipLodBias             = 0,
        };
        vk!.CreateSampler(device, &samplerInfo, null, out viewportSampler);

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
        // ImGui.slang compiled at runtime through the shader library, like every engine pipeline.
        // The reflected route emits one SPIR-V blob per entry point, so this is two modules where
        // the old build-time .spv was one shared module holding both stages.
        var program = renderer.Gpu.Shaders.GetProgram(
            new Renderer.Shaders.ShaderCompileRequest("ImGui", ["VSMain", "PSMain"], [], []));

        ShaderModule vertShader = renderer.gfx.CreateShaderModule(program.Spirv(0).ToArray());
        ShaderModule fragShader = renderer.gfx.CreateShaderModule(program.Spirv(1).ToArray());

        //config vertex stage
        PipelineShaderStageCreateInfo vertShaderInfo = new()
        {
            SType = StructureType.PipelineShaderStageCreateInfo,
            Stage = ShaderStageFlags.VertexBit,
            Module = vertShader,
            PName = (byte*)SilkMarshal.StringToPtr("VSMain")
        };
        //config frag stage
        PipelineShaderStageCreateInfo fragShaderInfo = new()
        {
            SType = StructureType.PipelineShaderStageCreateInfo,
            Stage = ShaderStageFlags.FragmentBit,
            Module = fragShader,
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

        vk!.DestroyShaderModule(device, vertShader, null);
        vk!.DestroyShaderModule(device, fragShader, null);
    }
}