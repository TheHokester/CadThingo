using Silk.NET.Core.Native;
using Silk.NET.Vulkan;
using Silk.NET.Vulkan.Extensions.KHR;

namespace CadThingo.VulkanEngine.Renderer.Pipelines;


//  Pipeline wrapper layout
//
//  Three layers:
//    1. PipelineBase      — owns the handle, layout, cache, descriptor set
//                           layouts, push-constant ranges, and the lifecycle.
//    2. GraphicsPipeline  — assembles GraphicsPipelineCreateInfo from a set of
//                           protected virtual hooks (vertex input, raster,
//                           blend, depth, dynamic rendering formats, …) so
//                           concrete pipelines override only what differs.
//    3. ComputePipeline   — single-shader-stage compute equivalent.
//
//  Concrete pipelines (Geometry, PbrDeferred, DrawCull, LightCull, …)
//  inherit from layer 2/3 and own their own SSBOs/UBOs, push-constant
//  structs, descriptor sets, and Record(...) entry points.

public abstract unsafe class PipelineBase : IDisposable
{
    // Single reference back to the renderer — pipelines call into things like
    // Renderer.CreateMappedStorageBuffer / CreateShaderModule / FindDepthFormat
    // directly. The fields and methods on Renderer that pipelines need are
    // 'internal' for same-assembly access.
    protected readonly Renderer Renderer;

    // Convenience accessors so subclass bodies stay short — these just forward
    // to the renderer's vk / device.
    protected Vk     Vk     => Renderer.vk!;
    protected Device Device => Renderer.device;

    protected Pipeline       PipelineHandle;
    protected PipelineLayout PipelineLayoutHandle;
    protected PipelineCache  PipelineCacheHandle;

    // Subclasses populate these in CreateDescriptorSetLayouts(). The default
    // CreatePipelineLayout() reads them to build the VkPipelineLayout.
    protected DescriptorSetLayout[] DescriptorSetLayouts = Array.Empty<DescriptorSetLayout>();

    // Indices into DescriptorSetLayouts that this pipeline OWNS and should
    // destroy on Dispose. Subclasses that borrow a layout from elsewhere
    // (e.g. ResourceManager's bindless layout) leave that index out.
    protected int[] OwnedDescriptorSetLayoutIndices = Array.Empty<int>();

    /// <summary>
    /// Descriptor sets for the pipeline <br/>
    /// DescriptorSets[layoutNum][frame]
    /// </summary>
    protected DescriptorSet[][] DescriptorSets = Array.Empty<DescriptorSet[]>();
    protected PushConstantRange[]   PushConstantRanges   = Array.Empty<PushConstantRange>();

    public Pipeline                  Handle    => PipelineHandle;
    public PipelineLayout            Layout    => PipelineLayoutHandle;

    public DescriptorSet GetDescriptorSet(int layoutNum, uint frame) => DescriptorSets[layoutNum][frame];
    public abstract PipelineBindPoint BindPoint { get; }

    protected PipelineBase(Renderer renderer)
    {
        Renderer = renderer;
    }

    // Called once by the owner (Renderer) after construction. Each step is a
    // virtual hook so concrete pipelines slot in their own logic without
    // re-implementing the whole flow.
    public void Initialize()
    {
        CreateDescriptorSetLayouts();
        CreatePipelineLayout();
        CreatePipeline();
        CreateResources();
        CreateDescriptorSets();
        WriteDescriptors();
    }

    // Required: populate descriptorSetLayouts and (optionally) pushConstantRanges.
    protected abstract void CreateDescriptorSetLayouts();

    // Required: build the VkPipeline itself. GraphicsPipeline / ComputePipeline
    // seal this and drive it from their hooks; only override directly if a
    // pipeline doesn't fit either category.
    protected abstract void CreatePipeline();

    protected virtual void CreatePipelineLayout()
    {
        fixed (DescriptorSetLayout* pLayouts = DescriptorSetLayouts)
        fixed (PushConstantRange*   pRanges  = PushConstantRanges)
        {
            PipelineLayoutCreateInfo info = new()
            {
                SType                  = StructureType.PipelineLayoutCreateInfo,
                SetLayoutCount         = (uint)DescriptorSetLayouts.Length,
                PSetLayouts            = pLayouts,
                PushConstantRangeCount = (uint)PushConstantRanges.Length,
                PPushConstantRanges    = pRanges,
            };
            if (Vk.CreatePipelineLayout(Device, &info, null, out PipelineLayoutHandle) != Result.Success)
                throw new Exception($"Failed to create pipeline layout for {GetType().Name}");
        }
    }

    // Concrete pipelines override these to allocate their owned SSBOs/UBOs,
    // allocate descriptor sets from the pool, and write the initial bindings.
    protected virtual void CreateResources()          { }
    protected virtual void CreateDescriptorSets()     { }
    protected virtual void WriteDescriptors()         { }

    public virtual void Dispose()
    {
        if (PipelineHandle.Handle       != 0) Vk.DestroyPipeline(Device, PipelineHandle, null);
        if (PipelineLayoutHandle.Handle != 0) Vk.DestroyPipelineLayout(Device, PipelineLayoutHandle, null);
        // Only destroy DSLs we own — borrowed layouts (e.g. ResourceManager's bindless
        // layout) are torn down by their owner.
        foreach (var idx in OwnedDescriptorSetLayoutIndices)
        {
            if (idx < DescriptorSetLayouts.Length && DescriptorSetLayouts[idx].Handle != 0)
                Vk.DestroyDescriptorSetLayout(Device, DescriptorSetLayouts[idx], null);
        }
        if (PipelineCacheHandle.Handle  != 0) Vk.DestroyPipelineCache(Device, PipelineCacheHandle, null);
    }
}

public abstract unsafe class GraphicsPipeline : PipelineBase
{
    public override PipelineBindPoint BindPoint => PipelineBindPoint.Graphics;

    protected GraphicsPipeline(Renderer renderer) : base(renderer) { }

    protected abstract string   ShaderPath              { get; }
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

    // Specialization-constant hook. Default = none. Subclasses override to wire
    // per-stage spec constants. Stack scratch is supplied by the base
    // CreatePipeline so the data outlives the vkCreateGraphicsPipelines call.
    // Return the number of map entries written for this stage.
    protected const int SpecScratchEntries = 8;   // max spec entries per stage
    protected const int SpecScratchBytes   = 64;  // max spec data bytes per stage

    protected virtual int FillSpecializationData(
        int stageIdx,
        SpecializationMapEntry* entries,
        byte* data,
        out uint dataSize)
    {
        dataSize = 0;
        return 0;
    }

    // Drives the pipeline build from the hooks above. Sealed because concrete
    // pipelines should configure via overrides rather than re-implementing
    // the whole assembly — that's the main thing keeping the sprawl out.
    protected sealed override void CreatePipeline()
    {
        byte[] code   = File.ReadAllBytes(ShaderPath);
        var    module = Renderer.CreateShaderModule(code);

        var stageDefs = ShaderStages;
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
                Module = module,
                PName  = (byte*)entryPtrs[i],
            };

            var entriesSlot = &specEntries[i * SpecScratchEntries];
            var dataSlot    = &specData[i * SpecScratchBytes];
            int filled = FillSpecializationData(i, entriesSlot, dataSlot, out uint dataSize);
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
        Vk.DestroyShaderModule(Device, module, null);
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

public abstract unsafe class ComputePipeline : PipelineBase
{
    public override PipelineBindPoint BindPoint => PipelineBindPoint.Compute;

    protected ComputePipeline(Renderer renderer) : base(renderer) { }

    protected abstract string ShaderPath { get; }

    // slangc emits the SPIR-V OpEntryPoint as "main" regardless of the source
    // function name. Override only if the .spv was produced by a toolchain
    // that preserves a different entry-point symbol.
    protected virtual string EntryPoint => "main";

    protected sealed override void CreatePipeline()
    {
        byte[] code   = File.ReadAllBytes(ShaderPath);
        var    module = Renderer.CreateShaderModule(code);
        var    entry  = SilkMarshal.StringToPtr(EntryPoint);

        var stage = new PipelineShaderStageCreateInfo
        {
            SType  = StructureType.PipelineShaderStageCreateInfo,
            Stage  = ShaderStageFlags.ComputeBit,
            Module = module,
            PName  = (byte*)entry,
        };

        var info = new ComputePipelineCreateInfo
        {
            SType  = StructureType.ComputePipelineCreateInfo,
            Stage  = stage,
            Layout = PipelineLayoutHandle,
        };

        if (Vk.CreateComputePipelines(Device, PipelineCacheHandle, 1, &info, null, out PipelineHandle) != Result.Success)
            throw new Exception($"Failed to create compute pipeline for {GetType().Name}");
        SilkMarshal.Free(entry);
        Vk.DestroyShaderModule(Device, module, null);
    }
}


public abstract unsafe class RtPipeline : PipelineBase
{
    protected RtPipeline(Renderer renderer) : base(renderer)
    {
    }
    
    public override PipelineBindPoint BindPoint => PipelineBindPoint.RayTracingNV;
    public abstract string ShaderPath { get; }

    protected override void CreatePipeline()
    {
        
    }
}



