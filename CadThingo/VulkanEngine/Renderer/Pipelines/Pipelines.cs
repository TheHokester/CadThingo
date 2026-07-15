using CadThingo.VulkanEngine.Renderer.Descriptors;
using CadThingo.VulkanEngine.Renderer.Shaders;
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
    // Reference back to the renderer — concrete pipelines still reach renderer-owned
    // technique/scene data through it (IBL views, light SSBOs, g-buffers, the probe
    // system, per-frame scene packing). That residual coupling is what L2/L3 remove.
    protected readonly Renderer Renderer;

    // The injected device-services channel. Anything RHI, descriptor, or shader related goes
    // through these three handles rather than the Renderer god object; Renderer above stays
    // only for the technique/scene reaches that have no other home yet.
    protected readonly GpuContext Gpu;

    protected GraphicsDevice     Gfx      => Gpu.Gfx;
    protected DescriptorRegistry Registry => Gpu.Registry;
    protected ShaderLibrary      Shaders  => Gpu.Shaders;

    // Convenience accessors so subclass bodies stay short — forward to the device.
    protected Vk     Vk     => Gfx.Vk;
    protected Device Device => Gfx.Device;

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

    // ---- reflected-program route ---------------------------------------------------------
    // Non-null routes this pipeline through ShaderLibrary: SPIR-V, push-constant ranges, spec
    // constants and set layouts are all derived from the .slang source rather than a build-time
    // .spv plus hand-transcribed declarations. Null keeps the legacy ShaderPath route.
    protected virtual ShaderCompileRequest? Program => null;

    /// The resolved program on the reflected route, else null. Valid from the start of Initialize.
    protected ShaderProgram? Reflected { get; private set; }

    /// Values for the program's reflected spec constants, keyed by name. Read at each build.
    protected virtual SpecValues? Specialization => null;

    /// The immutable sampler to bake into a reflected layout binding, or null for none.
    /// Reflection reports that a binding is a combined image sampler but not which sampler the
    /// layout pins, so the pipeline still answers that.
    protected virtual Sampler? ImmutableSamplerFor(in BindingDesc binding) => null;

    /// Stages to OR onto a reflected binding's visibility. Reflection only knows the stages of the
    /// program it walked, so a pipeline that builds SIBLING PSOs of a different kind against the
    /// same layout (ReSTIR's compute passes on the RT layout) must name those stages here, or the
    /// sibling cannot bind the set.
    protected virtual ShaderStageFlags ExtraStagesFor(in BindingDesc binding) => 0;

    /// Stages to OR onto every reflected push-constant range, for the same reason as
    /// <see cref="ExtraStagesFor"/>.
    protected virtual ShaderStageFlags ExtraPushStages => 0;

    /// The program's reflected bindings for one descriptor set, ordered by binding index.
    protected BindingDesc[] ReflectedBindings(uint set)
    {
        var program = Reflected ?? throw new InvalidOperationException(
            $"{GetType().Name}: ReflectedBindings requires a Program.");
        return program.Reflection.Bindings.Where(b => b.Set == set).OrderBy(b => b.Binding).ToArray();
    }

    /// Builds a VkDescriptorSetLayout for one reflected set. The shader declaration is the whole
    /// contract: binding index, descriptor type, and count all come from reflection.
    protected DescriptorSetLayout CreateReflectedSetLayout(uint set)
    {
        var reflected = ReflectedBindings(set);
        if (reflected.Length == 0)
            throw new InvalidOperationException($"{GetType().Name}: program reflects no bindings in set {set}.");

        var bindings = new DescriptorSetLayoutBinding[reflected.Length];
        // One slot per binding; taking the address of a list element requires it to stay put,
        // hence a flat array pinned for the whole create call.
        var samplers = new Sampler[reflected.Length];
        fixed (Sampler* pSamplers = samplers)
        {
            for (int i = 0; i < reflected.Length; i++)
            {
                var b = reflected[i];
                bindings[i] = new DescriptorSetLayoutBinding
                {
                    Binding         = b.Binding,
                    DescriptorType  = b.Type,
                    DescriptorCount = b.Count,
                    StageFlags      = b.Stages | ExtraStagesFor(in b),
                };
                if (ImmutableSamplerFor(in b) is { } s)
                {
                    samplers[i] = s;
                    bindings[i].PImmutableSamplers = &pSamplers[i];
                }
            }

            fixed (DescriptorSetLayoutBinding* pBindings = bindings)
            {
                var info = new DescriptorSetLayoutCreateInfo
                {
                    SType        = StructureType.DescriptorSetLayoutCreateInfo,
                    BindingCount = (uint)bindings.Length,
                    PBindings    = pBindings,
                };
                if (Vk.CreateDescriptorSetLayout(Device, &info, null, out var layout) != Result.Success)
                    throw new Exception($"{GetType().Name}: failed to create reflected set {set} layout");
                return layout;
            }
        }
    }

    // Builds the VkSpecializationInfo payload for one stage: joins the pipeline's named values
    // with the program's reflected constant ids, keeping only ids this stage's SPIR-V declares
    // (Slang strips constants an entry point never reads). Returns the entry count.
    private int FillReflectedSpecialization(
        int entryIndex, SpecializationMapEntry* entries, byte* data, out uint dataSize)
    {
        dataSize = 0;
        var program = Reflected!;
        if (Specialization is not { } values || values.Bits.Count == 0) return 0;

        var declared = SpirvUtil.SpecConstantIds(program.Spirv(entryIndex).Span);
        int filled = 0;
        foreach (var (name, bits) in values.Bits)
        {
            var match = program.Reflection.SpecConstants.FirstOrDefault(c => c.Name == name);
            if (match.Name != name)
                throw new InvalidOperationException(
                    $"{GetType().Name}: spec constant '{name}' is not declared by {program.Desc.Module} " +
                    $"(declared: {string.Join(", ", program.Reflection.SpecConstants.Select(c => c.Name))})");
            if (!declared.Contains(match.ConstantId)) continue;

            entries[filled] = new SpecializationMapEntry
            {
                ConstantID = match.ConstantId,
                Offset     = dataSize,
                Size       = sizeof(uint),
            };
            *(uint*)(data + dataSize) = bits;
            dataSize += sizeof(uint);
            filled++;
        }
        return filled;
    }

    // Creates the shader module for one entry point of the reflected program.
    protected ShaderModule CreateReflectedModule(int entryIndex)
        => Gfx.CreateShaderModule(Reflected!.Spirv(entryIndex).ToArray());

    // Shared by GraphicsPipeline / ComputePipeline: build a stage's spec info on whichever route
    // is active. Scratch must outlive the vkCreate*Pipelines call.
    private protected int FillStageSpecialization(
        int stageIdx, SpecializationMapEntry* entries, byte* data, out uint dataSize)
        => Reflected != null
            ? FillReflectedSpecialization(stageIdx, entries, data, out dataSize)
            : FillSpecializationData(stageIdx, entries, data, out dataSize);

    // Legacy specialization hook - stage-indexed and unsafe. Pipelines on the reflected route
    // override Specialization instead.
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

    protected PipelineBase(in GpuContext gpu, Renderer renderer)
    {
        Gpu = gpu;
        Renderer = renderer;
    }

    // Called once by the owner (Renderer) after construction. Each step is a
    // virtual hook so concrete pipelines slot in their own logic without
    // re-implementing the whole flow.
    public void Initialize()
    {
        // Resolve first: CreateDescriptorSetLayouts reads reflection on the new route, and the
        // push ranges reflection yields feed CreatePipelineLayout below.
        Reflected = Program is { } request ? Shaders.GetProgram(request) : null;
        if (Reflected != null)
            PushConstantRanges = Reflected.Reflection.PushConstants
                .Select(p => new PushConstantRange
                {
                    StageFlags = p.Stages | ExtraPushStages, Offset = 0, Size = p.Size,
                })
                .ToArray();

        CreateDescriptorSetLayouts();
        CreatePipelineLayout();
        CreatePipeline();
        CreateResources();
        CreateDescriptorSets();
        WriteDescriptors();
    }

    /// <summary>Rebuilds this pipeline's GPU objects in place, preserving object identity so
    /// any held reference (render-graph module fields, cross-pipeline bindings) stays valid:
    /// tears everything down via <see cref="ReleaseGpuResources"/> and recreates it via
    /// <see cref="Initialize"/> on the same instance. Deliberately does NOT route through
    /// <see cref="Dispose"/> -- reusing an object after Dispose violates the IDisposable
    /// contract and would break the moment a Dispose override adds a _disposed guard /
    /// finalizer / SuppressFinalize. The caller must have idled the device (e.g. after
    /// toggling a spec-constant setting) and is responsible for re-applying any external
    /// descriptor writes Initialize does not perform (cross-pipeline / graph-transient binds).
    /// Replaces the old "Dispose + new + Initialize" pattern, which orphaned references the
    /// owner had handed out.</summary>
    public void Rebuild()
    {
        ReleaseGpuResources();
        Initialize();
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

    public virtual void Dispose() => ReleaseGpuResources();

    /// <summary>The single GPU-teardown path, shared by <see cref="Dispose"/> (terminal) and
    /// <see cref="Rebuild"/> (followed by a fresh <see cref="Initialize"/>). Concrete pipelines
    /// override THIS -- not Dispose -- to free their own owned resources (UBOs/SSBOs/samplers),
    /// then call <c>base.ReleaseGpuResources()</c>. Routing Rebuild through here rather than
    /// Dispose keeps re-init off the IDisposable path.</summary>
    protected virtual void ReleaseGpuResources()
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

public abstract unsafe class ComputePipeline : PipelineBase
{
    public override PipelineBindPoint BindPoint => PipelineBindPoint.Compute;

    protected ComputePipeline(in GpuContext gpu, Renderer renderer) : base(gpu, renderer) { }

    // Build-time .spv for the legacy route; null on the reflected route (see PipelineBase.Program).
    protected virtual string? ShaderPath => null;

    // Legacy-route entry-point symbol: slangc emits the SPIR-V OpEntryPoint as "main" regardless
    // of the source function name when compiling a single-entry kernel. The reflected route takes
    // the name from reflection instead and ignores this.
    protected virtual string EntryPoint => "main";

    protected sealed override void CreatePipeline()
    {
        ShaderModule module;
        string entryName;
        if (Reflected != null)
        {
            var ep = Reflected.Reflection.EntryPoints[0];
            module = CreateReflectedModule(0);
            entryName = ep.Name;
        }
        else
        {
            module = Gfx.CreateShaderModule(File.ReadAllBytes(
                ShaderPath ?? throw new InvalidOperationException(
                    $"{GetType().Name}: needs either a Program or a ShaderPath.")));
            entryName = EntryPoint;
        }
        var entry = SilkMarshal.StringToPtr(entryName);

        var stage = new PipelineShaderStageCreateInfo
        {
            SType  = StructureType.PipelineShaderStageCreateInfo,
            Stage  = ShaderStageFlags.ComputeBit,
            Module = module,
            PName  = (byte*)entry,
        };

        // Scratch must outlive the create call below.
        var specInfo    = new SpecializationInfo();
        var specEntries = stackalloc SpecializationMapEntry[SpecScratchEntries];
        var specData    = stackalloc byte[SpecScratchBytes];
        int filled = FillStageSpecialization(0, specEntries, specData, out uint dataSize);
        if (filled > 0)
        {
            specInfo = new SpecializationInfo
            {
                MapEntryCount = (uint)filled,
                PMapEntries   = specEntries,
                DataSize      = (UIntPtr)dataSize,
                PData         = specData,
            };
            stage.PSpecializationInfo = &specInfo;
        }

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


// Base for the (additive, opt-in) ray-tracing-pipeline path. Owns the
// VK_KHR_ray_tracing_pipeline dispatch table and the SBT-layout properties every
// RT pipeline needs to pack its shader binding table — deliberately kept off the
// Renderer so the RT path stays self-contained and the compute path tracer is
// untouched. The concrete RTPipeline (the path tracer itself: shader groups, SBT,
// CmdTraceRays) inherits from this in a later phase. Construction is gated by the
// Renderer on RayTracePipelineSupported.
public abstract unsafe class RtPipeline : PipelineBase
{
    public override PipelineBindPoint BindPoint => PipelineBindPoint.RayTracingKhr;

    // Build-time .spv for the legacy route; null on the reflected route (see PipelineBase.Program).
    protected virtual string? ShaderPath => null;

    // VK_KHR_ray_tracing_pipeline dispatch table + SBT-layout properties, loaded
    // once per instance from the device. Null / zero if the extension failed to
    // load (defensive — the Renderer gate should prevent that).
    protected KhrRayTracingPipeline? KhrRtPipeline;
    protected uint ShaderGroupHandleSize;        // bytes per shader-group handle
    protected uint ShaderGroupBaseAlignment;     // raygen/miss/hit SBT region base alignment
    protected uint ShaderGroupHandleAlignment;   // per-handle stride alignment within a region
    protected uint MaxRayRecursionDepth;         // device cap; the path tracer targets depth 1

    protected RtPipeline(in GpuContext gpu, Renderer renderer) : base(gpu, renderer)
    {
        LoadDispatchAndProperties();
    }

    // Loads the dispatch table and queries PhysicalDeviceRayTracingPipelinePropertiesKHR
    // for the SBT alignment/handle sizes. Safe to call even if unsupported — it
    // just leaves KhrRtPipeline null and the properties zero.
    private void LoadDispatchAndProperties()
    {
        if (!Vk.TryGetDeviceExtension(Gfx.GetVkInstance(), Device, out KhrRtPipeline))
        {
            Console.Error.WriteLine("[RtPipeline] KhrRayTracingPipeline dispatch table failed to load");
            KhrRtPipeline = null;
            return;
        }

        var rtProps = new PhysicalDeviceRayTracingPipelinePropertiesKHR
        {
            SType = StructureType.PhysicalDeviceRayTracingPipelinePropertiesKhr,
        };
        var props2 = new PhysicalDeviceProperties2
        {
            SType = StructureType.PhysicalDeviceProperties2,
            PNext = &rtProps,
        };
        Vk.GetPhysicalDeviceProperties2(Gfx.PhysicalDevice, &props2);

        ShaderGroupHandleSize      = rtProps.ShaderGroupHandleSize;
        ShaderGroupBaseAlignment   = rtProps.ShaderGroupBaseAlignment;
        ShaderGroupHandleAlignment = rtProps.ShaderGroupHandleAlignment;
        MaxRayRecursionDepth       = rtProps.MaxRayRecursionDepth;
    }
}



