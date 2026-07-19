using Silk.NET.Core.Native;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Pipelines;

public abstract unsafe class GraphicsPipeline : PipelineBase
{
    public override PipelineBindPoint BindPoint => PipelineBindPoint.Graphics;

    protected GraphicsPipeline(in GpuContext gpu, Renderer renderer) : base(gpu, renderer) { }

    // Build-time .spv for the legacy route; null on the reflected route (see PipelineBase.Program).
    protected virtual  string?  ShaderPath              => null;
    protected abstract Format[] ColorAttachmentFormats  { get; }

    // Optional hooks (defaults match the common case)
    protected virtual Format DepthAttachmentFormat { get; init; } = Format.Undefined;

    // Multiview view mask threaded into VkPipelineRenderingCreateInfo.viewMask.
    // Zero (the default) disables multiview — single-view pipelines work as
    // before. Set to a bitmask with N bits to fan each draw across N layers,
    // which probe capture uses for the 6 cube faces (0x3F).
    protected virtual uint RenderingViewMask => 0u;

    protected virtual (ShaderStageFlags Stage, string EntryPoint)[] ShaderStages => new[]
    {
        (ShaderStageFlags.VertexBit,   "VSMain"),
        (ShaderStageFlags.FragmentBit, "PSMain"),
    };

    protected virtual VertexInputBindingDescription[]   GetVertexInputBindings()   => Array.Empty<VertexInputBindingDescription>();
    protected virtual VertexInputAttributeDescription[] GetVertexInputAttributes() => Array.Empty<VertexInputAttributeDescription>();

    protected virtual PipelineInputAssemblyStateCreateInfo BuildInputAssembly() => new()
    {
        SType                  = StructureType.PipelineInputAssemblyStateCreateInfo,
        Topology               = PrimitiveTopology.TriangleList,
        PrimitiveRestartEnable = false,
    };

    protected virtual PipelineViewportStateCreateInfo BuildViewportState() => new()
    {
        SType         = StructureType.PipelineViewportStateCreateInfo,
        ViewportCount = 1,
        ScissorCount  = 1,
    };

    protected virtual PipelineRasterizationStateCreateInfo BuildRasterizer() => new()
    {
        SType                   = StructureType.PipelineRasterizationStateCreateInfo,
        DepthClampEnable        = false,
        RasterizerDiscardEnable = false,
        PolygonMode             = PolygonMode.Fill,
        LineWidth               = 1.0f,
        CullMode                = CullModeFlags.BackBit,
        FrontFace               = FrontFace.CounterClockwise,
        DepthBiasEnable         = false,
    };

    protected virtual PipelineMultisampleStateCreateInfo BuildMultisample() => new()
    {
        SType                = StructureType.PipelineMultisampleStateCreateInfo,
        SampleShadingEnable  = false,
        RasterizationSamples = SampleCountFlags.Count1Bit,
    };

    protected virtual PipelineDepthStencilStateCreateInfo BuildDepthStencil() => new()
    {
        SType                 = StructureType.PipelineDepthStencilStateCreateInfo,
        DepthTestEnable       = true,
        DepthWriteEnable      = true,
        DepthCompareOp        = CompareOp.Less,
        DepthBoundsTestEnable = false,
        StencilTestEnable     = false,
        MinDepthBounds        = 0.0f,
        MaxDepthBounds        = 1.0f,
    };

    // One no-blend attachment per color target by default. Override for
    // additive / alpha / G-buffer-with-mixed-formats cases.
    protected virtual PipelineColorBlendAttachmentState[] BuildColorBlendAttachments()
    {
        var att = new PipelineColorBlendAttachmentState[ColorAttachmentFormats.Length];
        for (int i = 0; i < att.Length; i++)
        {
            att[i] = new()
            {
                BlendEnable    = false,
                ColorWriteMask = ColorComponentFlags.RBit | ColorComponentFlags.GBit |
                                 ColorComponentFlags.BBit | ColorComponentFlags.ABit,
            };
        }
        return att;
    }

    protected virtual DynamicState[] DynamicStates => new[] { DynamicState.Viewport, DynamicState.Scissor };

    // Drives the pipeline build from the hooks above. Sealed because concrete
    // pipelines should configure via overrides rather than re-implementing
    // the whole assembly — that's the main thing keeping the sprawl out.
    protected sealed override void CreatePipeline()
    {
        // Reflected route: one module per entry point, stages named by reflection. Legacy route:
        // one build-time .spv holding every entry point, stages named by the ShaderStages hook.
        var stageDefs = Reflected != null
            ? Reflected.Reflection.EntryPoints.Select(e => (e.Stage, EntryPoint: e.Name)).ToArray()
            : ShaderStages;

        var modules = new ShaderModule[stageDefs.Length];
        if (Reflected != null)
            for (int i = 0; i < modules.Length; i++) modules[i] = CreateReflectedModule(i);
        else
        {
            var shared = Gfx.CreateShaderModule(File.ReadAllBytes(
                ShaderPath ?? throw new InvalidOperationException(
                    $"{GetType().Name}: needs either a Program or a ShaderPath.")));
            Array.Fill(modules, shared);
        }

        var stages    = stackalloc PipelineShaderStageCreateInfo[stageDefs.Length];
        var entryPtrs = stackalloc nint[stageDefs.Length];

        // Stack scratch for spec constants — alive for the duration of this
        // method, which covers the CreateGraphicsPipelines call below.
        var specInfos   = stackalloc SpecializationInfo[stageDefs.Length];
        var specEntries = stackalloc SpecializationMapEntry[stageDefs.Length * SpecScratchEntries];
        var specData    = stackalloc byte[stageDefs.Length * SpecScratchBytes];

        for (int i = 0; i < stageDefs.Length; i++)
        {
            entryPtrs[i] = SilkMarshal.StringToPtr(stageDefs[i].EntryPoint);
            stages[i] = new()
            {
                SType  = StructureType.PipelineShaderStageCreateInfo,
                Stage  = stageDefs[i].Stage,
                Module = modules[i],
                PName  = (byte*)entryPtrs[i],
            };

            var entriesSlot = &specEntries[i * SpecScratchEntries];
            var dataSlot    = &specData[i * SpecScratchBytes];
            int filled = FillStageSpecialization(i, entriesSlot, dataSlot, out uint dataSize);
            if (filled > 0)
            {
                specInfos[i] = new SpecializationInfo
                {
                    MapEntryCount = (uint)filled,
                    PMapEntries   = entriesSlot,
                    DataSize      = (UIntPtr)dataSize,
                    PData         = dataSlot,
                };
                stages[i].PSpecializationInfo = &specInfos[i];
            }
        }

        var vertexInputBindings   = GetVertexInputBindings();
        var vertexInputAttributes = GetVertexInputAttributes();
        var inputAssembly         = BuildInputAssembly();
        var viewportState         = BuildViewportState();
        var rasterizer            = BuildRasterizer();
        var multisample           = BuildMultisample();
        var depthStencil          = BuildDepthStencil();
        var blendAttachments      = BuildColorBlendAttachments();
        var dynamicStates         = DynamicStates;
        var colorFormats          = ColorAttachmentFormats;

        fixed (VertexInputBindingDescription*     pBindings  = vertexInputBindings)
        fixed (VertexInputAttributeDescription*   pAttribs   = vertexInputAttributes)
        fixed (PipelineColorBlendAttachmentState* pBlend     = blendAttachments)
        fixed (DynamicState*                      pDyn       = dynamicStates)
        fixed (Format*                            pColorFmts = colorFormats)
        {
            var vertexInput = new PipelineVertexInputStateCreateInfo
            {
                SType                           = StructureType.PipelineVertexInputStateCreateInfo,
                VertexBindingDescriptionCount   = (uint)vertexInputBindings.Length,
                PVertexBindingDescriptions      = pBindings,
                VertexAttributeDescriptionCount = (uint)vertexInputAttributes.Length,
                PVertexAttributeDescriptions    = pAttribs,
            };

            var colorBlend = new PipelineColorBlendStateCreateInfo
            {
                SType           = StructureType.PipelineColorBlendStateCreateInfo,
                LogicOpEnable   = false,
                LogicOp         = LogicOp.Copy,
                AttachmentCount = (uint)blendAttachments.Length,
                PAttachments    = pBlend,
            };

            var dynamic = new PipelineDynamicStateCreateInfo
            {
                SType             = StructureType.PipelineDynamicStateCreateInfo,
                DynamicStateCount = (uint)dynamicStates.Length,
                PDynamicStates    = pDyn,
            };

            var renderingInfo = new PipelineRenderingCreateInfo
            {
                SType                   = StructureType.PipelineRenderingCreateInfo,
                ViewMask                = RenderingViewMask,
                ColorAttachmentCount    = (uint)colorFormats.Length,
                PColorAttachmentFormats = pColorFmts,
                DepthAttachmentFormat   = DepthAttachmentFormat,
                StencilAttachmentFormat = Format.Undefined,
            };

            var info = new GraphicsPipelineCreateInfo
            {
                SType               = StructureType.GraphicsPipelineCreateInfo,
                PNext               = &renderingInfo,
                StageCount          = (uint)stageDefs.Length,
                PStages             = stages,
                PVertexInputState   = &vertexInput,
                PInputAssemblyState = &inputAssembly,
                PViewportState      = &viewportState,
                PRasterizationState = &rasterizer,
                PMultisampleState   = &multisample,
                PDepthStencilState  = &depthStencil,
                PColorBlendState    = &colorBlend,
                PDynamicState       = &dynamic,
                Layout              = PipelineLayoutHandle,
                RenderPass          = default,
                Subpass             = 0,
                BasePipelineHandle  = default,
                BasePipelineIndex   = -1,
            };

            if (Vk.CreateGraphicsPipelines(Device, PipelineCacheHandle, 1, &info, null, out PipelineHandle) != Result.Success)
                throw new Exception($"Failed to create graphics pipeline for {GetType().Name}");
        }

        for (int i = 0; i < stageDefs.Length; i++) SilkMarshal.Free(entryPtrs[i]);
        // Distinct() because the legacy route shares one module across every stage.
        foreach (var m in modules.Distinct()) Vk.DestroyShaderModule(Device, m, null);
    }

    protected void BeginRendering(
        CommandBuffer cmd,
        Extent2D extent,
        ReadOnlySpan<ImageView> colorViews,
        ImageView depthView = default,
        AttachmentLoadOp colorLoad = AttachmentLoadOp.Clear,
        AttachmentLoadOp depthLoad = AttachmentLoadOp.Clear,
        ReadOnlySpan<ClearValue> clearValues = default)
    {
        var colorAttachments = stackalloc RenderingAttachmentInfoKHR[colorViews.Length];
        for (int i = 0; i < colorViews.Length; i++)
        {
            colorAttachments[i] = new()
            {
                SType = StructureType.RenderingAttachmentInfoKhr,
                ImageView = colorViews[i],
                ImageLayout = ImageLayout.ColorAttachmentOptimal,
                LoadOp = colorLoad,
                StoreOp = AttachmentStoreOp.Store
            };
            if (clearValues.Length > i) colorAttachments[i].ClearValue = clearValues[i];
        }
        var depthAttachment = new RenderingAttachmentInfoKHR()
        {
            SType = StructureType.RenderingAttachmentInfoKhr,
            ImageView = depthView,
            ImageLayout = ImageLayout.DepthStencilAttachmentOptimal,
            LoadOp = depthLoad,
            StoreOp = AttachmentStoreOp.Store,
            ClearValue = new ClearValue() { DepthStencil = new ClearDepthStencilValue(1.0f, 0) }
        };

        var renderingInfo = new RenderingInfoKHR()
        {
            SType = StructureType.RenderingInfoKhr,
            RenderArea = new Rect2D(new Offset2D(0, 0), extent),
            LayerCount = 1,
            ColorAttachmentCount = (uint)colorViews.Length,
            PColorAttachments = (RenderingAttachmentInfo*)colorAttachments,
            PDepthAttachment = (RenderingAttachmentInfo*)&depthAttachment,
        };
        Vk.CmdBeginRendering(cmd, (RenderingInfo*)&renderingInfo);
    }

    protected void EndRendering(CommandBuffer cmd) => Vk.CmdEndRendering(cmd);
    
    
}