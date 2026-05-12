using System.Numerics;
using System.Runtime.InteropServices;
using Silk.NET.Core.Native;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer;

// ────────────────────────────────────────────────────────────────────────────
//  Pipeline wrapper layout
// ────────────────────────────────────────────────────────────────────────────
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
// ────────────────────────────────────────────────────────────────────────────

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

    // ── Required overrides ──────────────────────────────────────────────────
    protected abstract string   ShaderPath              { get; }
    protected abstract Format[] ColorAttachmentFormats  { get; }

    // ── Optional hooks (defaults match the common case) ─────────────────────
    protected virtual Format DepthAttachmentFormat { get; init; } = Format.Undefined;

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

// ────────────────────────────────────────────────────────────────────────────
//  Geometry pass — writes the G-buffer
// ────────────────────────────────────────────────────────────────────────────

public sealed unsafe class GeometryPipeline : GraphicsPipeline
{
    struct GeometryUBO
    {
        public Matrix4x4 view;
        public Matrix4x4 proj;
    }
    //Per frame uniform buffers for geometry pipeline
    private UboBuffer[] GeometryUniformBuffers = new UboBuffer[Renderer.MAX_CONCURRENT_FRAMES];

    protected override string ShaderPath { get; } =
        @"C:\Users\jamie\RiderProjects\CadThingo\CadThingo\Assets\Shaders\Geometry.spv";
    protected override Format[] ColorAttachmentFormats { get; } =
    [
        Format.R32G32B32A32Sfloat, // Position
        Format.R32G32B32A32Sfloat, // Normal
        Format.R8G8B8A8Unorm, // Albedo
        Format.R8G8B8A8Unorm, // Material
        Format.R8G8B8A8Unorm // Emissive
    ];


    public GeometryPipeline(Renderer renderer) : base(renderer)
    {
        DepthAttachmentFormat = renderer.FindDepthFormat();
    }

    public override void Dispose()
    {
        foreach (var ubo in GeometryUniformBuffers) ubo.Dispose();
        base.Dispose();
    }
    protected override void CreateDescriptorSetLayouts()
    {
        CreateFrameDescriptorSetLayout(out var frameDSL);
        // Slot 0 is OWNED here; slot 1 is borrowed from ResourceManager (don't destroy on Dispose).
        DescriptorSetLayouts = new[] { frameDSL,  Engine.ResourceManager.GetBindlessLayout()};
        OwnedDescriptorSetLayoutIndices = new[] { 0 };
    }

    protected override void CreateResources()
    {
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            Renderer.CreateMappedUniformBuffer(sizeof(GeometryUBO), ref GeometryUniformBuffers[i]);
        }
    }

    protected override void CreateDescriptorSets()
    {
        // Per-frame "frame" descriptor sets (set 0): only binding 0 = FrameUBO (view+proj).
        // Set 1 (bindless materials/instances/textures/samplers) is owned and bound by
        // ResourceManager.
        var layouts = stackalloc DescriptorSetLayout[(int)Renderer.MAX_CONCURRENT_FRAMES];
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++) layouts[i] = DescriptorSetLayouts[0];

        DescriptorSetAllocateInfo allocateInfo = new()
        {
            SType = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool = Renderer.descriptorPool,
            DescriptorSetCount = Renderer.MAX_CONCURRENT_FRAMES,
            PSetLayouts = layouts
        };
        DescriptorSets = new DescriptorSet[1][];
        DescriptorSets[0] = new DescriptorSet[Renderer.MAX_CONCURRENT_FRAMES];
        fixed (DescriptorSet* pDS = DescriptorSets[0])
        {
            if (Vk.AllocateDescriptorSets(Device, &allocateInfo, pDS) != Result.Success)
                throw new Exception("Failed to allocate descriptor sets using layout 0 for geometry pipeline");
        }
    }

    protected override void WriteDescriptors()
    {
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            DescriptorBufferInfo bufferInfo = new()
            {
                Buffer = GeometryUniformBuffers[i].buffer,
                Offset = 0,
                Range = (ulong)sizeof(GeometryUBO),
            };
            WriteDescriptorSet descriptorWrite = new()
            {
                SType = StructureType.WriteDescriptorSet,
                DstSet = DescriptorSets[0][i],
                DstBinding = 0,
                DstArrayElement = 0,
                DescriptorType = DescriptorType.UniformBuffer,
                DescriptorCount = 1,
                PBufferInfo = &bufferInfo
            };
            Vk.UpdateDescriptorSets(Device, 1, &descriptorWrite, 0, null);
        }
    }

    protected override VertexInputBindingDescription[] GetVertexInputBindings()
    {
        return [Vertex.GetBindingDescription()];
    }

    protected override VertexInputAttributeDescription[] GetVertexInputAttributes()
    {
        return Vertex.GetAttributeDescriptions();
    }


    // Set 0 of the geometry pipeline. One binding (the per-frame FrameUBO with view+proj),
    // bound once at the start of the geometry pass and reused for every draw.
    private void CreateFrameDescriptorSetLayout(out DescriptorSetLayout layout)
    {
        var binding = new DescriptorSetLayoutBinding
        {
            Binding = 0,
            DescriptorType = DescriptorType.UniformBuffer,
            DescriptorCount = 1,
            StageFlags = ShaderStageFlags.VertexBit,
            PImmutableSamplers = null,
        };

        DescriptorSetLayoutBindingFlagsCreateInfo flagsCreateInfo = new()
            { SType = StructureType.DescriptorSetLayoutBindingFlagsCreateInfo };
        var flag = DescriptorBindingFlags.UpdateAfterBindBit |
                   DescriptorBindingFlags.UpdateUnusedWhilePendingBit;

        if (Renderer.descriptorIndexEnabled)
        {
            flagsCreateInfo.BindingCount = 1;
            flagsCreateInfo.PBindingFlags = &flag;
        }

        DescriptorSetLayoutCreateInfo layoutInfo = new()
        {
            SType = StructureType.DescriptorSetLayoutCreateInfo,
            BindingCount = 1,
            PBindings = &binding,
        };
        if (Renderer.descriptorIndexEnabled)
        {
            layoutInfo.Flags |= DescriptorSetLayoutCreateFlags.UpdateAfterBindPoolBit;
            layoutInfo.PNext = &flagsCreateInfo;
        }

        if (Vk.CreateDescriptorSetLayout(Device, &layoutInfo, null, out layout) !=
            Result.Success)
            throw new Exception("Failed to create geometry frame descriptor set layout");
    }

    // Writes the per-frame view+proj into GeometryUniformBuffers[frameIndex].
    // Called once per frame in DrawFrame; per-draw model matrix lives in the instance SSBO.
    public void UpdateUbo(uint frameIndex, Camera camera)
    {
        GeometryUBO ubo = new();
        if (camera != null)
        {
            ubo.proj = camera.GetProjectionMatrix((float)Renderer.swapChainExtent.Width / Renderer.swapChainExtent.Height, 0.1f, 100.0f);
            ubo.view = camera.GetViewMatrix();
            ubo.proj.M22 *= -1; // Vulkan clip space has Y down
        }
        else
        {
            ubo.view = Matrix4x4.CreateLookAt(new Vector3(2, 2, 2), new Vector3(0, 0, 0), new Vector3(0, 0, 1));
            ubo.proj = Matrix4x4.CreatePerspectiveFieldOfView((float)(45 * Math.PI / 180),
                (float)Renderer.swapChainExtent.Width / Renderer.swapChainExtent.Height, 0.1f, 100.0f);
            ubo.proj.M22 *= -1; // flip Y for Vulkan clip space
        }

        void* data = GeometryUniformBuffers[frameIndex].mapped;
        new Span<GeometryUBO>(data, 1).Fill(ubo);
    }
}

// ────────────────────────────────────────────────────────────────────────────
//  Draw-cull compute pass — frustum-tests scene renderables and emits
//  VkDrawIndexedIndirectCommand[] consumed by the geometry pass.
// ────────────────────────────────────────────────────────────────────────────

public sealed unsafe class DrawCullPipeline : ComputePipeline
{
    // Push constants pushed at every dispatch. 100 bytes — within the
    // 128B Vulkan minimum (maxPushConstantsSize).
    [StructLayout(LayoutKind.Sequential)]
    private struct CullPushConstants
    {
        public Vector4 PlaneL;
        public Vector4 PlaneR;
        public Vector4 PlaneB;
        public Vector4 PlaneT;
        public Vector4 PlaneN;
        public Vector4 PlaneF;
        public uint    RenderableCount;
        public uint    _pad0;
        public uint    _pad1;
        public uint    _pad2;
    }

    protected override string ShaderPath { get; } =
        @"C:\Users\jamie\RiderProjects\CadThingo\CadThingo\Assets\Shaders\CullDraws.spv";

    // Per-frame buffers owned by this pipeline. RenderableInput is the CPU input
    // list filled in Record(); IndirectCmd holds the post-cull
    // VkDrawIndexedIndirectCommand array; IndirectCount holds one uint the cull
    // shader InterlockedAdds into and the rasterizer reads via
    // vkCmdDrawIndexedIndirectCount.
    private UboBuffer[] RenderableInputBuffers = new UboBuffer[Renderer.MAX_CONCURRENT_FRAMES];
    private UboBuffer[] IndirectCmdBuffers     = new UboBuffer[Renderer.MAX_CONCURRENT_FRAMES];
    private UboBuffer[] IndirectCountBuffers   = new UboBuffer[Renderer.MAX_CONCURRENT_FRAMES];

    public Buffer GetIndirectCmdBuffer  (uint frame) => IndirectCmdBuffers[frame].buffer;
    public Buffer GetIndirectCountBuffer(uint frame) => IndirectCountBuffers[frame].buffer;

    /// <summary>Renderables packed in the most recent Record() call — drives
    /// maxDrawCount on vkCmdDrawIndexedIndirectCount.</summary>
    public uint LastRenderableCount { get; private set; }

    // BLEND-mode entities partitioned out of the cull input during Record.
    // Sorted back-to-front by view-space depth so the transparent pass renders
    // far-first.
    private readonly List<TransparentDraw> _transparentDraws = new();

    /// <summary>BLEND-mode draws captured this frame, sorted back-to-front by view-space Z.
    /// Consumed by the TransparentPass; empty when no scene material is BLEND-mode.</summary>
    public IReadOnlyList<TransparentDraw> LastTransparentDraws => _transparentDraws;

    public DrawCullPipeline(Renderer renderer) : base(renderer)
    {
        PushConstantRanges = new[]
        {
            new PushConstantRange
            {
                StageFlags = ShaderStageFlags.ComputeBit,
                Offset     = 0,
                Size       = (uint)sizeof(CullPushConstants),
            }
        };
    }

    public override void Dispose()
    {
        foreach (var b in RenderableInputBuffers) b.Dispose();
        foreach (var b in IndirectCmdBuffers)     b.Dispose();
        foreach (var b in IndirectCountBuffers)   b.Dispose();
        base.Dispose();
    }

    protected override void CreateDescriptorSetLayouts()
    {
        // Four storage buffers — all writable except binding 0 (input renderables).
        // We mark them all StorageBuffer with no UpdateAfterBind — the sets are
        // written once at startup (the buffers themselves are persistently mapped).
        var bindings = stackalloc DescriptorSetLayoutBinding[4];
        for (uint b = 0; b < 4; b++)
        {
            bindings[b] = new DescriptorSetLayoutBinding
            {
                Binding         = b,
                DescriptorType  = DescriptorType.StorageBuffer,
                DescriptorCount = 1,
                StageFlags      = ShaderStageFlags.ComputeBit,
                PImmutableSamplers = null,
            };
        }
        DescriptorSetLayoutCreateInfo info = new()
        {
            SType        = StructureType.DescriptorSetLayoutCreateInfo,
            BindingCount = 4,
            PBindings    = bindings,
        };
        if (Vk.CreateDescriptorSetLayout(Device, &info, null, out var layout) != Result.Success)
            throw new Exception("Failed to create cull descriptor set layout");
        DescriptorSetLayouts = new[] { layout };
        OwnedDescriptorSetLayoutIndices = new[] { 0 };
    }

    protected override void CreateResources()
    {
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            Renderer.CreateMappedStorageBuffer(
                (ulong)(Renderer.MAX_INSTANCES * (uint)sizeof(RenderableInputGpu)),
                ref RenderableInputBuffers[i]);

            // Indirect-command buffer also needs IndirectBuffer usage so the
            // vkCmdDraw...IndirectCount call can read it without validation errors.
            Renderer.CreateMappedStorageBuffer(
                (ulong)(Renderer.MAX_INSTANCES * (uint)sizeof(DrawIndexedIndirectCommandGpu)),
                ref IndirectCmdBuffers[i],
                BufferUsageFlags.IndirectBufferBit);

            // Count buffer is one uint. Needs IndirectBuffer for the count read and
            // TransferDst so vkCmdFillBuffer can reset it to 0 every frame.
            Renderer.CreateMappedStorageBuffer(
                sizeof(uint),
                ref IndirectCountBuffers[i],
                BufferUsageFlags.IndirectBufferBit | BufferUsageFlags.TransferDstBit);
        }
    }

    protected override void CreateDescriptorSets()
    {
        var layouts = stackalloc DescriptorSetLayout[(int)Renderer.MAX_CONCURRENT_FRAMES];
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++) layouts[i] = DescriptorSetLayouts[0];

        DescriptorSetAllocateInfo alloc = new()
        {
            SType              = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool     = Renderer.descriptorPool,
            DescriptorSetCount = Renderer.MAX_CONCURRENT_FRAMES,
            PSetLayouts        = layouts,
        };
        DescriptorSets = new DescriptorSet[1][];
        DescriptorSets[0] = new DescriptorSet[Renderer.MAX_CONCURRENT_FRAMES];
        fixed (DescriptorSet* pSets = DescriptorSets[0])
        {
            if (Vk.AllocateDescriptorSets(Device, &alloc, pSets) != Result.Success)
                throw new Exception("Failed to allocate cull descriptor sets");
        }
    }

    protected override void WriteDescriptors()
    {
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            DescriptorBufferInfo bufIn = new()
            {
                Buffer = RenderableInputBuffers[i].buffer, Offset = 0,
                Range  = (ulong)(Renderer.MAX_INSTANCES * (uint)sizeof(RenderableInputGpu)),
            };
            DescriptorBufferInfo bufCmd = new()
            {
                Buffer = IndirectCmdBuffers[i].buffer, Offset = 0,
                Range  = (ulong)(Renderer.MAX_INSTANCES * (uint)sizeof(DrawIndexedIndirectCommandGpu)),
            };
            DescriptorBufferInfo bufInst = new()
            {
                Buffer = Engine.ResourceManager.GetInstanceBuffer((uint)i), Offset = 0,
                Range  = (ulong)(Renderer.MAX_INSTANCES * (uint)sizeof(InstanceDataGPU)),
            };
            DescriptorBufferInfo bufCount = new()
            {
                Buffer = IndirectCountBuffers[i].buffer, Offset = 0,
                Range  = sizeof(uint),
            };

            var writes = stackalloc WriteDescriptorSet[4];
            for (uint b = 0; b < 4; b++)
            {
                writes[b] = new WriteDescriptorSet
                {
                    SType           = StructureType.WriteDescriptorSet,
                    DstSet          = DescriptorSets[0][i],
                    DstBinding      = b,
                    DescriptorType  = DescriptorType.StorageBuffer,
                    DescriptorCount = 1,
                };
            }
            writes[0].PBufferInfo = &bufIn;
            writes[1].PBufferInfo = &bufCmd;
            writes[2].PBufferInfo = &bufInst;
            writes[3].PBufferInfo = &bufCount;

            Vk.UpdateDescriptorSets(Device, 4, writes, 0, null);
        }
    }

    // CPU side of the cull pass. Walks scene entities, fills the input buffer,
    // then records the dispatch + barriers. Returns the renderable count packed
    // this frame (cached as LastRenderableCount for the geometry pass).
    public uint Record(CommandBuffer cmd, uint frameIndex, Camera cam, Scene scene)
    {
        // ── Pack RenderableInput rows from the scene ─────────────────────
        // Opaque (OPAQUE + MASK) entities go through the GPU cull → indirect-draw path.
        // BLEND entities are siphoned into _transparentDraws for the forward+ pass.
        RenderableInputGpu* inputPtr = (RenderableInputGpu*)RenderableInputBuffers[frameIndex].mapped;
        uint count = 0;
        int matCount = scene.MaterialCount;
        int fallbackMatIdx = matCount; // matches the fallback slot written by the geometry pass

        _transparentDraws.Clear();
        Matrix4x4 viewMat = cam != null ? cam.GetViewMatrix() : Matrix4x4.Identity;

        for (int i = 0; i < scene.EntityCount; i++)
        {
            Entity* e = scene.GetEntity(i);
            if (e == null) continue;
            var meshComp = e->GetComponent<MeshComponent>();
            if (meshComp == null || meshComp.mesh == null) continue;
            var transform = e->GetComponent<TransformComponent>();
            if (transform == null) continue;

            int  matIdx = meshComp.materialIndex >= 0 ? meshComp.materialIndex : fallbackMatIdx;
            Mesh m      = *meshComp.mesh;
            var  world  = *transform.GetWorldMatrix();

            // Fallback material is implicitly opaque. Real materials carry their
            // mode in the Flags bitfield.
            AlphaMode mode = AlphaMode.Opaque;
            if (matIdx >= 0 && matIdx < matCount)
                mode = scene.Materials[matIdx].GetAlphaMode();

            if (mode == AlphaMode.Blend)
            {
                // System.Numerics row-vector convention: Vector.Transform(v, m) = v * m.
                // CreateLookAt produces -Z-forward view space, so farther entities
                // have more-negative Z. Sort ascending for back-to-front order.
                var worldOrigin = new Vector4(world.M41, world.M42, world.M43, 1f);
                float viewZ = Vector4.Transform(worldOrigin, viewMat).Z;

                _transparentDraws.Add(new TransparentDraw
                {
                    Model         = world,
                    MaterialIndex = (uint)matIdx,
                    IndexCount    = (uint)m.count,
                    FirstIndex    = (uint)m.offset,
                    ViewDepth     = viewZ,
                });
                continue;
            }

            if (count >= Renderer.MAX_INSTANCES)
                throw new InvalidOperationException($"Renderable count exceeds MAX_INSTANCES ({Renderer.MAX_INSTANCES}).");

            inputPtr[count] = new RenderableInputGpu
            {
                model         = world,
                sphereLocal   = m.sphereLocal,
                indexCount    = (uint)m.count,
                firstIndex    = (uint)m.offset,
                materialIndex = (uint)matIdx,
            };
            count++;
        }

        // Back-to-front: far (lowest view-space Z) first.
        _transparentDraws.Sort((a, b) => a.ViewDepth.CompareTo(b.ViewDepth));

        LastRenderableCount = count;
        if (count == 0) return 0;

        // ── 1. Reset the count buffer to 0 via vkCmdFillBuffer ───────────
        Vk.CmdFillBuffer(cmd, IndirectCountBuffers[frameIndex].buffer, 0, sizeof(uint), 0);

        // ── 2. Barrier: transfer write -> compute shader access on count buffer ─
        var fillBarrier = new BufferMemoryBarrier
        {
            SType         = StructureType.BufferMemoryBarrier,
            SrcAccessMask = AccessFlags.TransferWriteBit,
            DstAccessMask = AccessFlags.ShaderReadBit | AccessFlags.ShaderWriteBit,
            SrcQueueFamilyIndex = Silk.NET.Vulkan.Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Silk.NET.Vulkan.Vk.QueueFamilyIgnored,
            Buffer = IndirectCountBuffers[frameIndex].buffer,
            Offset = 0,
            Size   = sizeof(uint),
        };
        Vk.CmdPipelineBarrier(cmd,
            PipelineStageFlags.TransferBit,
            PipelineStageFlags.ComputeShaderBit,
            0, 0, null, 1, &fillBarrier, 0, null);

        // ── 3. Bind pipeline + descriptor set + push constants, dispatch ──
        Vk.CmdBindPipeline(cmd, PipelineBindPoint.Compute, PipelineHandle);
        var dset = DescriptorSets[0][frameIndex];
        Vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute,
            PipelineLayoutHandle, 0, 1, &dset, 0, null);

        // Build frustum from the camera's view*proj. Deliberately use a non-Y-flipped
        // projection here: the visible volume is the same in both conventions, and
        // Frustum.FromViewProjection assumes standard row-major Vulkan NDC.
        Matrix4x4 view = cam.GetViewMatrix();
        Matrix4x4 proj = cam.GetProjectionMatrix(
            (float)Renderer.swapChainExtent.Width / Renderer.swapChainExtent.Height, 0.1f, 100.0f);
        Matrix4x4 vp   = view * proj;
        var frustum    = Frustum.FromViewProjection(vp, vulkanNDC: true);

        var push = new CullPushConstants
        {
            PlaneL = frustum.PlaneLeft.Data,
            PlaneR = frustum.PlaneRight.Data,
            PlaneB = frustum.PlaneBottom.Data,
            PlaneT = frustum.PlaneTop.Data,
            PlaneN = frustum.PlaneNear.Data,
            PlaneF = frustum.PlaneFar.Data,
            RenderableCount = count,
        };
        Vk.CmdPushConstants(cmd, PipelineLayoutHandle, ShaderStageFlags.ComputeBit,
            0, (uint)sizeof(CullPushConstants), &push);

        // 64 threads per group; ceil-divide so the last group covers the tail.
        uint groups = (count + 63u) / 64u;
        Vk.CmdDispatch(cmd, groups, 1, 1);

        // ── 4. Barrier: compute writes -> indirect/vertex stage reads ────
        var postBarriers = stackalloc BufferMemoryBarrier[3];
        postBarriers[0] = new BufferMemoryBarrier
        {
            SType         = StructureType.BufferMemoryBarrier,
            SrcAccessMask = AccessFlags.ShaderWriteBit,
            DstAccessMask = AccessFlags.IndirectCommandReadBit,
            SrcQueueFamilyIndex = Silk.NET.Vulkan.Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Silk.NET.Vulkan.Vk.QueueFamilyIgnored,
            Buffer = IndirectCmdBuffers[frameIndex].buffer,
            Offset = 0,
            Size   = Silk.NET.Vulkan.Vk.WholeSize,
        };
        postBarriers[1] = new BufferMemoryBarrier
        {
            SType         = StructureType.BufferMemoryBarrier,
            SrcAccessMask = AccessFlags.ShaderWriteBit,
            DstAccessMask = AccessFlags.IndirectCommandReadBit,
            SrcQueueFamilyIndex = Silk.NET.Vulkan.Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Silk.NET.Vulkan.Vk.QueueFamilyIgnored,
            Buffer = IndirectCountBuffers[frameIndex].buffer,
            Offset = 0,
            Size   = sizeof(uint),
        };
        postBarriers[2] = new BufferMemoryBarrier
        {
            SType         = StructureType.BufferMemoryBarrier,
            SrcAccessMask = AccessFlags.ShaderWriteBit,
            DstAccessMask = AccessFlags.ShaderReadBit,
            SrcQueueFamilyIndex = Silk.NET.Vulkan.Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Silk.NET.Vulkan.Vk.QueueFamilyIgnored,
            Buffer = Engine.ResourceManager.GetInstanceBuffer(frameIndex),
            Offset = 0,
            Size   = Silk.NET.Vulkan.Vk.WholeSize,
        };
        Vk.CmdPipelineBarrier(cmd,
            PipelineStageFlags.ComputeShaderBit,
            PipelineStageFlags.DrawIndirectBit | PipelineStageFlags.VertexShaderBit,
            0, 0, null, 3, postBarriers, 0, null);

        return count;
    }
}

// ────────────────────────────────────────────────────────────────────────────
//  Tiled light-cull compute pass — bins active lights into per-tile slot
//  lists so the deferred lighting shader iterates only the lights that
//  actually touch each 16×16 tile.
// ────────────────────────────────────────────────────────────────────────────

public sealed unsafe class LightCullPipeline : ComputePipeline
{
    // 64B invViewProj + 16 camPos + 8 screenSize + 8 tileCounts + 4 lightCount
    // + 12 pad = 112B, well under the 128B Vulkan minimum.
    [StructLayout(LayoutKind.Sequential)]
    private struct LightCullPushConstants
    {
        public Matrix4x4 InvViewProj;
        public Vector4   CamPos;
        public Vector2   ScreenSize;
        public uint      TileCountX;
        public uint      TileCountY;
        public uint      LightCount; 
        public uint      _pad0;
        public uint      _pad1;
        public uint      _pad2;
    }

    protected override string ShaderPath { get; } =
        @"C:\Users\jamie\RiderProjects\CadThingo\CadThingo\Assets\Shaders\LightCulling.spv";

    // Tile-cull per-frame outputs owned by this pipeline. TileLightCount[tileIdx]
    // is the number of lights overlapping each tile; TileLightIndices[tileIdx*MAX + slot]
    // is the flat index into the lights SSBO. The lighting pass reads both, keyed by
    // tileIdx = (gl_FragCoord / TILE_SIZE).
    private UboBuffer[] TileLightCountBuffers   = new UboBuffer[Renderer.MAX_CONCURRENT_FRAMES];
    private UboBuffer[] TileLightIndicesBuffers = new UboBuffer[Renderer.MAX_CONCURRENT_FRAMES];

    public Buffer GetTileLightCountBuffer  (uint frame) => TileLightCountBuffers[frame].buffer;
    public Buffer GetTileLightIndicesBuffer(uint frame) => TileLightIndicesBuffers[frame].buffer;

    public LightCullPipeline(Renderer renderer) : base(renderer)
    {
        PushConstantRanges = new[]
        {
            new PushConstantRange
            {
                StageFlags = ShaderStageFlags.ComputeBit,
                Offset     = 0,
                Size       = (uint)sizeof(LightCullPushConstants),
            }
        };
    }

    public override void Dispose()
    {
        foreach (var b in TileLightCountBuffers)   b.Dispose();
        foreach (var b in TileLightIndicesBuffers) b.Dispose();
        base.Dispose();
    }

    protected override void CreateDescriptorSetLayouts()
    {
        // 3 storage buffers: lights (read), tileLightCount (write), tileLightIndices (write).
        var bindings = stackalloc DescriptorSetLayoutBinding[3];
        for (uint b = 0; b < 3; b++)
        {
            bindings[b] = new DescriptorSetLayoutBinding
            {
                Binding         = b,
                DescriptorType  = DescriptorType.StorageBuffer,
                DescriptorCount = 1,
                StageFlags      = ShaderStageFlags.ComputeBit,
                PImmutableSamplers = null,
            };
        }
        DescriptorSetLayoutCreateInfo info = new()
        {
            SType        = StructureType.DescriptorSetLayoutCreateInfo,
            BindingCount = 3,
            PBindings    = bindings,
        };
        if (Vk.CreateDescriptorSetLayout(Device, &info, null, out var layout) != Result.Success)
            throw new Exception("Failed to create light-cull descriptor set layout");
        DescriptorSetLayouts = new[] { layout };
        OwnedDescriptorSetLayoutIndices = new[] { 0 };
    }

    protected override void CreateResources()
    {
        // Tile-cull buffers sized for worst-case tile count (MAX_TILE_COUNT).
        // Per frame: TileLightCount = MAX × 4B, TileLightIndices = MAX × MAX_LIGHTS_PER_TILE × 4B.
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            Renderer.CreateMappedStorageBuffer(
                (ulong)(Renderer.MAX_TILE_COUNT * sizeof(uint)),
                ref TileLightCountBuffers[i]);
            Renderer.CreateMappedStorageBuffer(
                (ulong)(Renderer.MAX_TILE_COUNT * Renderer.MAX_LIGHTS_PER_TILE * sizeof(uint)),
                ref TileLightIndicesBuffers[i]);
        }
    }

    protected override void CreateDescriptorSets()
    {
        var layouts = stackalloc DescriptorSetLayout[(int)Renderer.MAX_CONCURRENT_FRAMES];
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++) layouts[i] = DescriptorSetLayouts[0];

        DescriptorSetAllocateInfo alloc = new()
        {
            SType              = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool     = Renderer.descriptorPool,
            DescriptorSetCount = Renderer.MAX_CONCURRENT_FRAMES,
            PSetLayouts        = layouts,
        };
        DescriptorSets = new DescriptorSet[1][];
        DescriptorSets[0] = new DescriptorSet[Renderer.MAX_CONCURRENT_FRAMES];
        fixed (DescriptorSet* pSets = DescriptorSets[0])
        {
            if (Vk.AllocateDescriptorSets(Device, &alloc, pSets) != Result.Success)
                throw new Exception("Failed to allocate light-cull descriptor sets");
        }
    }

    protected override void WriteDescriptors()
    {
        // PbrDeferredPipeline owns the lights SSBO; this binding is the consumer side.
        var pbr = Renderer.PbrDeferredPipeline
                  ?? throw new InvalidOperationException(
                      "LightCullPipeline must be initialized AFTER PbrDeferredPipeline so the lights SSBO is available.");

        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            DescriptorBufferInfo bufLights = new()
            {
                Buffer = pbr.GetLightStorageBuffer((uint)i), Offset = 0,
                Range  = (ulong)(Renderer.MAX_LIGHTS * (uint)sizeof(PbrLightGpu)),
            };
            DescriptorBufferInfo bufTileCount = new()
            {
                Buffer = TileLightCountBuffers[i].buffer, Offset = 0,
                Range  = (ulong)(Renderer.MAX_TILE_COUNT * sizeof(uint)),
            };
            DescriptorBufferInfo bufTileIdx = new()
            {
                Buffer = TileLightIndicesBuffers[i].buffer, Offset = 0,
                Range  = (ulong)(Renderer.MAX_TILE_COUNT * Renderer.MAX_LIGHTS_PER_TILE * sizeof(uint)),
            };

            var writes = stackalloc WriteDescriptorSet[3];
            for (uint b = 0; b < 3; b++)
            {
                writes[b] = new WriteDescriptorSet
                {
                    SType           = StructureType.WriteDescriptorSet,
                    DstSet          = DescriptorSets[0][i],
                    DstBinding      = b,
                    DescriptorType  = DescriptorType.StorageBuffer,
                    DescriptorCount = 1,
                };
            }
            writes[0].PBufferInfo = &bufLights;
            writes[1].PBufferInfo = &bufTileCount;
            writes[2].PBufferInfo = &bufTileIdx;

            Vk.UpdateDescriptorSets(Device, 3, writes, 0, null);
        }
    }

    // CPU side. Computes invViewProj + tile counts from the current
    // camera/swapchain extent, pushes them, dispatches one group per tile, and
    // barriers compute-write -> fragment-read on the two tile buffers.
    public void Record(CommandBuffer cmd, uint frameIndex, Camera cam,
                       uint lightCount, uint tileCountX, uint tileCountY)
    {
        if (tileCountX == 0 || tileCountY == 0) return;

        Vk.CmdBindPipeline(cmd, PipelineBindPoint.Compute, PipelineHandle);
        var dset = DescriptorSets[0][frameIndex];
        Vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute,
            PipelineLayoutHandle, 0, 1, &dset, 0, null);

        Matrix4x4 view = cam.GetViewMatrix();
        Matrix4x4 proj = cam.GetProjectionMatrix(
            (float)Renderer.swapChainExtent.Width / Renderer.swapChainExtent.Height, 0.1f, 100.0f);
        // The lighting fragment shader sees a Y-flipped projection (the geometry
        // pipeline flips proj.M22 in UpdateUbo). Build invViewProj from the SAME
        // flipped matrix so the cull frustum lines up with where pixels actually
        // sample world positions from the g-buffer.
        proj.M22 *= -1f;
        Matrix4x4 vp = view * proj;
        if (!Matrix4x4.Invert(vp, out Matrix4x4 invVP))
            invVP = Matrix4x4.Identity;

        var push = new LightCullPushConstants
        {
            InvViewProj = invVP,
            CamPos      = new Vector4(cam.GetPosition(), 1f),
            ScreenSize  = new Vector2(Renderer.swapChainExtent.Width, Renderer.swapChainExtent.Height),
            TileCountX  = tileCountX,
            TileCountY  = tileCountY,
            LightCount  = lightCount,
        };
        Vk.CmdPushConstants(cmd, PipelineLayoutHandle, ShaderStageFlags.ComputeBit,
            0, (uint)sizeof(LightCullPushConstants), &push);

        // One thread group per tile (each group is 16×16 = 256 threads).
        Vk.CmdDispatch(cmd, tileCountX, tileCountY, 1);

        // Barrier: compute writes -> fragment shader reads of the tile buffers.
        var postBarriers = stackalloc BufferMemoryBarrier[2];
        postBarriers[0] = new BufferMemoryBarrier
        {
            SType         = StructureType.BufferMemoryBarrier,
            SrcAccessMask = AccessFlags.ShaderWriteBit,
            DstAccessMask = AccessFlags.ShaderReadBit,
            SrcQueueFamilyIndex = Silk.NET.Vulkan.Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Silk.NET.Vulkan.Vk.QueueFamilyIgnored,
            Buffer = TileLightCountBuffers[frameIndex].buffer,
            Offset = 0,
            Size   = Silk.NET.Vulkan.Vk.WholeSize,
        };
        postBarriers[1] = new BufferMemoryBarrier
        {
            SType         = StructureType.BufferMemoryBarrier,
            SrcAccessMask = AccessFlags.ShaderWriteBit,
            DstAccessMask = AccessFlags.ShaderReadBit,
            SrcQueueFamilyIndex = Silk.NET.Vulkan.Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Silk.NET.Vulkan.Vk.QueueFamilyIgnored,
            Buffer = TileLightIndicesBuffers[frameIndex].buffer,
            Offset = 0,
            Size   = Silk.NET.Vulkan.Vk.WholeSize,
        };
        Vk.CmdPipelineBarrier(cmd,
            PipelineStageFlags.ComputeShaderBit,
            PipelineStageFlags.FragmentShaderBit,
            0, 0, null, 2, postBarriers, 0, null);
    }
}

// ────────────────────────────────────────────────────────────────────────────
//  PBR deferred lighting pass — fullscreen triangle, samples G-buffer +
//  per-tile light list, optional ray-queried shadows.
// ────────────────────────────────────────────────────────────────────────────

public sealed unsafe class PbrDeferredPipeline : GraphicsPipeline
{
    // Matches PbrShader.slang's LightingFrameUBO (binding 0 of set 0).
    [StructLayout(LayoutKind.Sequential)]
    struct LightingFrameUBO
    {
        public Vector4 camPos;
        public float _padExposure;          // formerly exposure — tone-map moved to TonemapPipeline
        public float _padGamma;             // formerly gamma — tone-map moved to TonemapPipeline
        public float prefilteredCubeMipLevels;
        public float scaleIBLAmbient;
        public uint lightCount;
        public uint tileCountX;
        public uint tileCountY;
        public uint _pad0;
        public Vector2 screenSize;
        public uint _pad1;
        public uint _pad2;
    }

    protected override string ShaderPath { get; } =
        @"C:\Users\jamie\RiderProjects\CadThingo\CadThingo\Assets\Shaders\PBR.spv";

    // Lighting writes linear HDR scene-referred color; tone-map + gamma run in
    // the separate TonemapPipeline pass that consumes this attachment.
    protected override Format[] ColorAttachmentFormats { get; } = new[] { Format.R16G16B16A16Sfloat };

    // Set 0 — per-frame lighting (UBO, lights SSBO, TLAS, tile cull buffers).
    // Set 1 — shared G-buffer samplers (one allocation reused every frame).
    // The bindless-style "set index" matches what the PBR shader expects.
    private const int SetLighting = 0;
    private const int SetGBuffer  = 1;

    // Per-frame data owned by this pipeline. UpdatePerFrame fills both each frame.
    private UboBuffer[] LightingUniformBuffers = new UboBuffer[Renderer.MAX_CONCURRENT_FRAMES];
    private UboBuffer[] LightStorageBuffers    = new UboBuffer[Renderer.MAX_CONCURRENT_FRAMES];

    public Buffer GetLightStorageBuffer(uint frame) => LightStorageBuffers[frame].buffer;

    /// <summary>True = wire the PCSS-style soft-shadow specialization constant on,
    /// pulled into the fragment shader as <c>constant_id 0</c>. Read once at
    /// pipeline build; toggling requires Dispose + rebuild.</summary>
    public bool SoftShadowsEnabled { get; init; } = true;

    // Reusable scratch list filled by Scene.EnumerateLights — keeps allocations
    // out of the per-frame path.
    private readonly List<LightComponent> _lightScratch = new();

    public PbrDeferredPipeline(Renderer renderer) : base(renderer) { }

    public override void Dispose()
    {
        foreach (var b in LightingUniformBuffers) b.Dispose();
        foreach (var b in LightStorageBuffers)    b.Dispose();
        base.Dispose();
    }

    // ── Shader-stage overrides ─────────────────────────────────────────────
    // Lighting pass is a fullscreen triangle synthesized by SV_VertexID; depth
    // test is off, alpha blending off.

    protected override PipelineDepthStencilStateCreateInfo BuildDepthStencil() => new()
    {
        SType                 = StructureType.PipelineDepthStencilStateCreateInfo,
        DepthTestEnable       = false,
        DepthWriteEnable      = false,
        DepthCompareOp        = CompareOp.Always,
        DepthBoundsTestEnable = false,
        StencilTestEnable     = false,
    };

    protected override PipelineRasterizationStateCreateInfo BuildRasterizer() => new()
    {
        SType                   = StructureType.PipelineRasterizationStateCreateInfo,
        DepthClampEnable        = false,
        RasterizerDiscardEnable = false,
        PolygonMode             = PolygonMode.Fill,
        LineWidth               = 1.0f,
        CullMode                = CullModeFlags.None,
        FrontFace               = FrontFace.CounterClockwise,
        DepthBiasEnable         = false,
    };

    // Wire constant_id 0 (SOFT_SHADOWS) on the fragment stage. Vulkan bool spec
    // constants are 32-bit, so we pack the value into a uint.
    protected override int FillSpecializationData(
        int stageIdx,
        SpecializationMapEntry* entries,
        byte* data,
        out uint dataSize)
    {
        // ShaderStages default: [0]=VS, [1]=FS. Spec constant lives on FS only.
        if (stageIdx == 1)
        {
            entries[0] = new SpecializationMapEntry
            {
                ConstantID = 0,
                Offset     = 0,
                Size       = sizeof(uint),
            };
            *(uint*)data = SoftShadowsEnabled ? 1u : 0u;
            dataSize = sizeof(uint);
            return 1;
        }
        dataSize = 0;
        return 0;
    }

    // ── Descriptor set layouts ─────────────────────────────────────────────

    protected override void CreateDescriptorSetLayouts()
    {
        DescriptorSetLayouts = new DescriptorSetLayout[2];
        OwnedDescriptorSetLayoutIndices = new[] { 0, 1 };

        // ── Set 0: LightingFrameUBO + Light SSBO + TLAS + Tile cull buffers ───
        // Only the fragment shader reads lights, camPos, exposure, gamma etc.
        // The vertex shader is a procedural fullscreen triangle (SV_VertexID only).
        var set0Bindings = new DescriptorSetLayoutBinding[]
        {
            new()
            {
                Binding = 0,
                DescriptorType = DescriptorType.UniformBuffer,
                DescriptorCount = 1,
                StageFlags = ShaderStageFlags.FragmentBit,
                PImmutableSamplers = null,
            },
            new() // StructuredBuffer<PbrLight> — per-frame, rewritten in UpdatePerFrame.
            {
                Binding = 1,
                DescriptorType = DescriptorType.StorageBuffer,
                DescriptorCount = 1,
                StageFlags = ShaderStageFlags.FragmentBit,
                PImmutableSamplers = null,
            },
            new() // TLAS for ray-traced shadows. Bound once at startup; no per-frame update.
            {
                Binding = 2,
                DescriptorType = DescriptorType.AccelerationStructureKhr,
                DescriptorCount = 1,
                StageFlags = ShaderStageFlags.FragmentBit,
                PImmutableSamplers = null,
            },
            new() // tileLightCount[tileIdx] — produced by LightCulling.slang.
            {
                Binding = 3,
                DescriptorType = DescriptorType.StorageBuffer,
                DescriptorCount = 1,
                StageFlags = ShaderStageFlags.FragmentBit,
                PImmutableSamplers = null,
            },
            new() // tileLightIndices[tileIdx*MAX + slot] — produced by LightCulling.slang.
            {
                Binding = 4,
                DescriptorType = DescriptorType.StorageBuffer,
                DescriptorCount = 1,
                StageFlags = ShaderStageFlags.FragmentBit,
                PImmutableSamplers = null,
            },
        };

        // ── Set 1: G-Buffer inputs ────────────────────────────────────────────
        // Five samplers written by the geometry pass, read here for lighting.
        var set1Bindings = new DescriptorSetLayoutBinding[]
        {
            new() { Binding = 0, DescriptorType = DescriptorType.CombinedImageSampler, DescriptorCount = 1, StageFlags = ShaderStageFlags.FragmentBit, PImmutableSamplers = null },
            new() { Binding = 1, DescriptorType = DescriptorType.CombinedImageSampler, DescriptorCount = 1, StageFlags = ShaderStageFlags.FragmentBit, PImmutableSamplers = null },
            new() { Binding = 2, DescriptorType = DescriptorType.CombinedImageSampler, DescriptorCount = 1, StageFlags = ShaderStageFlags.FragmentBit, PImmutableSamplers = null },
            new() { Binding = 3, DescriptorType = DescriptorType.CombinedImageSampler, DescriptorCount = 1, StageFlags = ShaderStageFlags.FragmentBit, PImmutableSamplers = null },
            new() { Binding = 4, DescriptorType = DescriptorType.CombinedImageSampler, DescriptorCount = 1, StageFlags = ShaderStageFlags.FragmentBit, PImmutableSamplers = null },
        };

        DescriptorSetLayoutBindingFlagsCreateInfo set0FlagsInfo = new()
            { SType = StructureType.DescriptorSetLayoutBindingFlagsCreateInfo };
        DescriptorSetLayoutBindingFlagsCreateInfo set1FlagsInfo = new()
            { SType = StructureType.DescriptorSetLayoutBindingFlagsCreateInfo };

        var set0Flags = stackalloc DescriptorBindingFlags[set0Bindings.Length];
        var set1Flags = stackalloc DescriptorBindingFlags[set1Bindings.Length];

        fixed (DescriptorSetLayoutBinding* pSet0 = set0Bindings)
        fixed (DescriptorSetLayoutBinding* pSet1 = set1Bindings)
        {
            if (Renderer.descriptorIndexEnabled)
            {
                var updateFlags = DescriptorBindingFlags.UpdateAfterBindBit |
                                  DescriptorBindingFlags.UpdateUnusedWhilePendingBit;

                for (int i = 0; i < set0Bindings.Length; i++)
                {
                    // AccelerationStructureKhr and StorageBuffer bindings can't carry
                    // UpdateAfterBindBit unless their respective UpdateAfterBind features
                    // (descriptorBindingAccelerationStructureUpdateAfterBind /
                    // descriptorBindingStorageBufferUpdateAfterBind) are enabled — we
                    // don't request either. The light SSBO is per-frame mapped memory,
                    // so per-frame vkUpdateDescriptorSets isn't needed anyway.
                    if (set0Bindings[i].DescriptorType == DescriptorType.AccelerationStructureKhr) continue;
                    if (set0Bindings[i].DescriptorType == DescriptorType.StorageBuffer) continue;
                    set0Flags[i] = updateFlags;
                }
                set0FlagsInfo.BindingCount = (uint)set0Bindings.Length;
                set0FlagsInfo.PBindingFlags = set0Flags;

                for (int i = 0; i < set1Bindings.Length; i++) set1Flags[i] = updateFlags;
                set1FlagsInfo.BindingCount = (uint)set1Bindings.Length;
                set1FlagsInfo.PBindingFlags = set1Flags;
            }

            DescriptorSetLayoutCreateInfo set0LayoutInfo = new()
            {
                SType = StructureType.DescriptorSetLayoutCreateInfo,
                BindingCount = (uint)set0Bindings.Length,
                PBindings = pSet0,
            };
            if (Renderer.descriptorIndexEnabled)
            {
                set0LayoutInfo.Flags |= DescriptorSetLayoutCreateFlags.UpdateAfterBindPoolBit;
                set0LayoutInfo.PNext = &set0FlagsInfo;
            }
            if (Vk.CreateDescriptorSetLayout(Device, &set0LayoutInfo, null, out DescriptorSetLayouts[SetLighting]) != Result.Success)
                throw new Exception("Failed to create PBR set 0 descriptor set layout");

            DescriptorSetLayoutCreateInfo set1LayoutInfo = new()
            {
                SType = StructureType.DescriptorSetLayoutCreateInfo,
                BindingCount = (uint)set1Bindings.Length,
                PBindings = pSet1,
            };
            if (Renderer.descriptorIndexEnabled)
            {
                set1LayoutInfo.Flags |= DescriptorSetLayoutCreateFlags.UpdateAfterBindPoolBit;
                set1LayoutInfo.PNext = &set1FlagsInfo;
            }
            if (Vk.CreateDescriptorSetLayout(Device, &set1LayoutInfo, null, out DescriptorSetLayouts[SetGBuffer]) != Result.Success)
                throw new Exception("Failed to create PBR set 1 (GBuffer) descriptor set layout");
        }
    }

    // ── Resources ──────────────────────────────────────────────────────────

    protected override void CreateResources()
    {
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            Renderer.CreateMappedUniformBuffer(sizeof(LightingFrameUBO), ref LightingUniformBuffers[i]);
            Renderer.CreateMappedStorageBuffer(
                (ulong)(Renderer.MAX_LIGHTS * (uint)sizeof(PbrLightGpu)),
                ref LightStorageBuffers[i]);
        }
    }

    // ── Descriptor sets ────────────────────────────────────────────────────

    protected override void CreateDescriptorSets()
    {
        // Set 0 — per-frame; one set per frame in flight.
        var lightingLayouts = stackalloc DescriptorSetLayout[(int)Renderer.MAX_CONCURRENT_FRAMES];
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++) lightingLayouts[i] = DescriptorSetLayouts[SetLighting];

        DescriptorSetAllocateInfo lightingAlloc = new()
        {
            SType              = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool     = Renderer.descriptorPool,
            DescriptorSetCount = Renderer.MAX_CONCURRENT_FRAMES,
            PSetLayouts        = lightingLayouts,
        };

        DescriptorSets = new DescriptorSet[2][];
        DescriptorSets[SetLighting] = new DescriptorSet[Renderer.MAX_CONCURRENT_FRAMES];
        fixed (DescriptorSet* pSets = DescriptorSets[SetLighting])
        {
            if (Vk.AllocateDescriptorSets(Device, &lightingAlloc, pSets) != Result.Success)
                throw new Exception("Failed to allocate PBR lighting descriptor sets");
        }

        // Set 1 — shared g-buffer set, one allocation reused by every frame.
        var gBufLayout = DescriptorSetLayouts[SetGBuffer];
        DescriptorSetAllocateInfo gBufAlloc = new()
        {
            SType              = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool     = Renderer.descriptorPool,
            DescriptorSetCount = 1,
            PSetLayouts        = &gBufLayout,
        };
        DescriptorSets[SetGBuffer] = new DescriptorSet[1];
        fixed (DescriptorSet* pSet = DescriptorSets[SetGBuffer])
        {
            if (Vk.AllocateDescriptorSets(Device, &gBufAlloc, pSet) != Result.Success)
                throw new Exception("Failed to allocate PBR g-buffer descriptor set");
        }
    }

    // ── Descriptor writes ──────────────────────────────────────────────────
    // Writes split into three phases so cross-pipeline / TLAS deps can be wired
    // post-Initialize:
    //   - WriteDescriptors  (auto from Initialize): bindings 0,1 of set 0 + set 1 g-buffer.
    //   - WriteTileBufferDescriptors(lightCull):    bindings 3,4 of set 0.
    //   - WriteTlasDescriptor(tlas):                binding 2 of set 0.
    // Set 1 (g-buffer) is also written by WriteGBufferDescriptors after swapchain recreate.

    protected override void WriteDescriptors()
    {
        WriteLightFrameDescriptors();
        WriteGBufferDescriptors();
    }

    private void WriteLightFrameDescriptors()
    {
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            DescriptorBufferInfo frameInfo = new()
            {
                Buffer = LightingUniformBuffers[i].buffer,
                Offset = 0,
                Range  = (ulong)sizeof(LightingFrameUBO),
            };
            DescriptorBufferInfo lightsInfo = new()
            {
                Buffer = LightStorageBuffers[i].buffer,
                Offset = 0,
                Range  = (ulong)(Renderer.MAX_LIGHTS * (uint)sizeof(PbrLightGpu)),
            };
            var writes = stackalloc WriteDescriptorSet[2];
            writes[0] = new WriteDescriptorSet
            {
                SType           = StructureType.WriteDescriptorSet,
                DstSet          = DescriptorSets[SetLighting][i],
                DstBinding      = 0,
                DstArrayElement = 0,
                DescriptorType  = DescriptorType.UniformBuffer,
                DescriptorCount = 1,
                PBufferInfo     = &frameInfo,
            };
            writes[1] = new WriteDescriptorSet
            {
                SType           = StructureType.WriteDescriptorSet,
                DstSet          = DescriptorSets[SetLighting][i],
                DstBinding      = 1,
                DstArrayElement = 0,
                DescriptorType  = DescriptorType.StorageBuffer,
                DescriptorCount = 1,
                PBufferInfo     = &lightsInfo,
            };
            Vk.UpdateDescriptorSets(Device, 2, writes, 0, null);
        }
    }

    /// <summary>Writes bindings 3 + 4 (per-tile light count and indices) on the
    /// per-frame lighting sets. Called once after the light-cull pipeline is
    /// initialized — its output buffers are this pipeline's inputs.</summary>
    public void WriteTileBufferDescriptors(LightCullPipeline lightCull)
    {
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            DescriptorBufferInfo tileCountInfo = new()
            {
                Buffer = lightCull.GetTileLightCountBuffer((uint)i),
                Offset = 0,
                Range  = (ulong)(Renderer.MAX_TILE_COUNT * sizeof(uint)),
            };
            DescriptorBufferInfo tileIdxInfo = new()
            {
                Buffer = lightCull.GetTileLightIndicesBuffer((uint)i),
                Offset = 0,
                Range  = (ulong)(Renderer.MAX_TILE_COUNT * Renderer.MAX_LIGHTS_PER_TILE * sizeof(uint)),
            };

            var writes = stackalloc WriteDescriptorSet[2];
            writes[0] = new WriteDescriptorSet
            {
                SType           = StructureType.WriteDescriptorSet,
                DstSet          = DescriptorSets[SetLighting][i],
                DstBinding      = 3,
                DstArrayElement = 0,
                DescriptorType  = DescriptorType.StorageBuffer,
                DescriptorCount = 1,
                PBufferInfo     = &tileCountInfo,
            };
            writes[1] = new WriteDescriptorSet
            {
                SType           = StructureType.WriteDescriptorSet,
                DstSet          = DescriptorSets[SetLighting][i],
                DstBinding      = 4,
                DstArrayElement = 0,
                DescriptorType  = DescriptorType.StorageBuffer,
                DescriptorCount = 1,
                PBufferInfo     = &tileIdxInfo,
            };
            Vk.UpdateDescriptorSets(Device, 2, writes, 0, null);
        }
    }

    /// <summary>Writes the current TLAS handle into binding 2 of every per-frame
    /// lighting set. Called once at startup after InitRayQuery, and again whenever
    /// the TLAS handle is recreated. Skips silently when ray queries aren't
    /// available — the layout still has the binding declared but the shader path
    /// that reads it is gated by a specialization constant.</summary>
    public void WriteTlasDescriptor(AccelerationStructureKHR tlas)
    {
        if (tlas.Handle == 0) return;

        var tlasHandle = tlas;
        var asWrite = new WriteDescriptorSetAccelerationStructureKHR
        {
            SType = StructureType.WriteDescriptorSetAccelerationStructureKhr,
            AccelerationStructureCount = 1,
            PAccelerationStructures = &tlasHandle,
        };

        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            // AS writes carry no buffer/image info — resolved via the chained
            // WriteDescriptorSetAccelerationStructureKHR.
            var write = new WriteDescriptorSet
            {
                SType           = StructureType.WriteDescriptorSet,
                PNext           = &asWrite,
                DstSet          = DescriptorSets[SetLighting][i],
                DstBinding      = 2,
                DstArrayElement = 0,
                DescriptorType  = DescriptorType.AccelerationStructureKhr,
                DescriptorCount = 1,
            };
            Vk.UpdateDescriptorSets(Device, 1, &write, 0, null);
        }
    }

    /// <summary>Re-writes the 5 g-buffer sampler bindings on set 1. Run after
    /// the renderer recreates the g-buffer ImageViews (initial create or
    /// swapchain recreate).</summary>
    public void WriteGBufferDescriptors()
    {
        var sampler = Renderer.gBufferSampler;
        var imageInfos = stackalloc DescriptorImageInfo[5]
        {
            new() { ImageView = Renderer.gBufferPosition.ImageView, Sampler = sampler, ImageLayout = ImageLayout.ShaderReadOnlyOptimal },
            new() { ImageView = Renderer.gBufferNormal  .ImageView, Sampler = sampler, ImageLayout = ImageLayout.ShaderReadOnlyOptimal },
            new() { ImageView = Renderer.gBufferAlbedo  .ImageView, Sampler = sampler, ImageLayout = ImageLayout.ShaderReadOnlyOptimal },
            new() { ImageView = Renderer.gBufferMaterial.ImageView, Sampler = sampler, ImageLayout = ImageLayout.ShaderReadOnlyOptimal },
            new() { ImageView = Renderer.gBufferEmissive.ImageView, Sampler = sampler, ImageLayout = ImageLayout.ShaderReadOnlyOptimal },
        };
        var writes = stackalloc WriteDescriptorSet[5];
        for (uint i = 0; i < 5; i++)
        {
            writes[i] = new WriteDescriptorSet
            {
                SType           = StructureType.WriteDescriptorSet,
                DstSet          = DescriptorSets[SetGBuffer][0],
                DstBinding      = i,
                DstArrayElement = 0,
                DescriptorType  = DescriptorType.CombinedImageSampler,
                DescriptorCount = 1,
                PImageInfo      = &imageInfos[i],
            };
        }
        Vk.UpdateDescriptorSets(Device, 5, writes, 0, null);
    }

    // ── Per-frame upload ────────────────────────────────────────────────────
    // Walks scene lights into the per-frame Light SSBO and updates the lighting
    // UBO with camera + tile counts. Returns (lightCount, tileX, tileY) so the
    // renderer can drive the light-cull dispatch without recomputing.

    public (uint lightCount, uint tileCountX, uint tileCountY) UpdatePerFrame(
        uint frameIndex, Camera camera, Scene scene)
    {
        scene.EnumerateLights(_lightScratch);

        // ── Pack lights ────────────────────────────────────────
        PbrLightGpu* lightPtr = (PbrLightGpu*)LightStorageBuffers[frameIndex].mapped;
        uint count = 0;
        foreach (var light in _lightScratch)
        {
            if (count >= Renderer.MAX_LIGHTS) break;

            // World-space position from the owner transform if present.
            Vector3 worldPos = Vector3.Zero;
            if (light.Owner != null)
            {
                var t = light.Owner->GetComponent<TransformComponent>();
                if (t != null)
                {
                    var w = *t.GetWorldMatrix();
                    worldPos = new Vector3(w.M41, w.M42, w.M43);
                }
            }

            // Normalize direction — guard against zero-vector default.
            Vector3 dir = light.Direction.LengthSquared() > 1e-8f
                ? Vector3.Normalize(light.Direction)
                : new Vector3(0, -1, 0);

            // Range: -1 sentinel marks directional lights so the shader can branch
            // on attenuation without inspecting Type for the most common test.
            float range = light.Type == LightType.Directional ? -1f : light.Range;

            lightPtr[count] = new PbrLightGpu
            {
                positionRange  = new Vector4(worldPos, range),
                colorIntensity = new Vector4(light.Color, light.Intensity),
                directionType  = new Vector4(dir, (float)(uint)light.Type),
                spotCones      = new Vector4(light.InnerConeCos, light.OuterConeCos,
                                             light.CastShadows ? 1f : 0f, light.Radius),
            };
            count++;
        }

        // ── Frame UBO ──────────────────────────────────────────
        uint tileX = (Renderer.swapChainExtent.Width  + Renderer.TILE_SIZE - 1) / Renderer.TILE_SIZE;
        uint tileY = (Renderer.swapChainExtent.Height + Renderer.TILE_SIZE - 1) / Renderer.TILE_SIZE;

        LightingFrameUBO ubo = new();
        ubo.camPos = camera != null ? new Vector4(camera.GetPosition(), 1.0f) : new Vector4(2, 2, 2, 1);
        ubo.prefilteredCubeMipLevels = 1.0f;
        ubo.scaleIBLAmbient = 1.0f;
        ubo.lightCount = count;
        ubo.tileCountX = tileX;
        ubo.tileCountY = tileY;
        ubo.screenSize = new Vector2(Renderer.swapChainExtent.Width, Renderer.swapChainExtent.Height);

        void* data = LightingUniformBuffers[frameIndex].mapped;
        new Span<LightingFrameUBO>(data, 1).Fill(ubo);

        return (count, tileX, tileY);
    }
}


// ────────────────────────────────────────────────────────────────────────────
//  Tone-map / post pass — samples HDRColor, writes FinalColor (LDR)
// ────────────────────────────────────────────────────────────────────────────

// Matches the TONEMAP_OPERATOR spec constant in Tonemap.slang. Read once at
// pipeline build; changing the operator requires Dispose + rebuild.
public enum TonemapOperator : uint
{
    Reinhard = 0,
    Filmic   = 1,
}

public sealed unsafe class TonemapPipeline : GraphicsPipeline
{
    // Push constants — fragment-only, 8 bytes total, well under the 128B
    // guaranteed minimum. Kept off the per-frame UBO because exposure/gamma
    // change rarely and don't deserve a descriptor binding of their own.
    [StructLayout(LayoutKind.Sequential)]
    struct TonemapPushConstants
    {
        public float Exposure;
        public float Gamma;
    }

    protected override string ShaderPath { get; } =
        @"C:\Users\jamie\RiderProjects\CadThingo\CadThingo\Assets\Shaders\Tonemap.spv";

    protected override Format[] ColorAttachmentFormats { get; } = new[] { Format.R8G8B8A8Unorm };

    // Surfaced as plain properties so the renderer (or imgui later) can adjust.
    // Defaults preserve the visual output of the pre-refactor inline tone-map.
    public float Exposure { get; set; } = 4.5f;
    public float Gamma    { get; set; } = 2.0f;

    /// <summary>Selects the tone-map curve via <c>constant_id 0</c> in
    /// Tonemap.slang. Read once at pipeline build; toggling requires
    /// Dispose + rebuild.</summary>
    public TonemapOperator Operator { get; init; } = TonemapOperator.Filmic;

    public TonemapPipeline(Renderer renderer) : base(renderer)
    {
        PushConstantRanges = new[]
        {
            new PushConstantRange
            {
                StageFlags = ShaderStageFlags.FragmentBit,
                Offset     = 0,
                Size       = (uint)sizeof(TonemapPushConstants),
            }
        };
    }

    // Fullscreen triangle — no vertex inputs, no depth, no blend, no culling.

    protected override PipelineDepthStencilStateCreateInfo BuildDepthStencil() => new()
    {
        SType                 = StructureType.PipelineDepthStencilStateCreateInfo,
        DepthTestEnable       = false,
        DepthWriteEnable      = false,
        DepthCompareOp        = CompareOp.Always,
        DepthBoundsTestEnable = false,
        StencilTestEnable     = false,
    };

    protected override PipelineRasterizationStateCreateInfo BuildRasterizer() => new()
    {
        SType                   = StructureType.PipelineRasterizationStateCreateInfo,
        DepthClampEnable        = false,
        RasterizerDiscardEnable = false,
        PolygonMode             = PolygonMode.Fill,
        LineWidth               = 1.0f,
        CullMode                = CullModeFlags.None,
        FrontFace               = FrontFace.CounterClockwise,
        DepthBiasEnable         = false,
    };

    // Wire constant_id 0 (TONEMAP_OPERATOR) on the fragment stage. The slang
    // declaration is uint, so we pack the enum value into a 4-byte slot.
    protected override int FillSpecializationData(
        int stageIdx,
        SpecializationMapEntry* entries,
        byte* data,
        out uint dataSize)
    {
        // ShaderStages default: [0]=VS, [1]=FS. Spec constant lives on FS only.
        if (stageIdx == 1)
        {
            entries[0] = new SpecializationMapEntry
            {
                ConstantID = 0,
                Offset     = 0,
                Size       = sizeof(uint),
            };
            *(uint*)data = (uint)Operator;
            dataSize = sizeof(uint);
            return 1;
        }
        dataSize = 0;
        return 0;
    }

    // ── Descriptor layout — set 0, binding 0 = HDR input sampler ───────────

    protected override void CreateDescriptorSetLayouts()
    {
        DescriptorSetLayouts = new DescriptorSetLayout[1];
        OwnedDescriptorSetLayoutIndices = new[] { 0 };

        var binding = new DescriptorSetLayoutBinding
        {
            Binding         = 0,
            DescriptorType  = DescriptorType.CombinedImageSampler,
            DescriptorCount = 1,
            StageFlags      = ShaderStageFlags.FragmentBit,
        };
        var layoutInfo = new DescriptorSetLayoutCreateInfo
        {
            SType        = StructureType.DescriptorSetLayoutCreateInfo,
            BindingCount = 1,
            PBindings    = &binding,
        };
        if (Vk.CreateDescriptorSetLayout(Device, &layoutInfo, null, out DescriptorSetLayouts[0]) != Result.Success)
            throw new Exception("Failed to create tonemap descriptor set layout");
    }

    // No per-frame mapped buffers — the HDR image is graph-owned, single-buffered,
    // and tunables ride in push constants.
    protected override void CreateResources() { }

    protected override void CreateDescriptorSets()
    {
        // Single shared set — HDR image is single-buffered like FinalColor.
        var layout = DescriptorSetLayouts[0];
        DescriptorSetAllocateInfo allocInfo = new()
        {
            SType              = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool     = Renderer.descriptorPool,
            DescriptorSetCount = 1,
            PSetLayouts        = &layout,
        };
        DescriptorSets = new DescriptorSet[1][];
        DescriptorSets[0] = new DescriptorSet[1];
        fixed (DescriptorSet* pSet = DescriptorSets[0])
        {
            if (Vk.AllocateDescriptorSets(Device, &allocInfo, pSet) != Result.Success)
                throw new Exception("Failed to allocate tonemap descriptor set");
        }
    }

    // Descriptor target (HDRColor view) only exists after the render graph compiles,
    // so the renderer calls WriteHdrInputDescriptor explicitly post-setup and on
    // every swapchain recreate.
    protected override void WriteDescriptors() { }

    public void WriteHdrInputDescriptor(ImageView hdrView, Sampler sampler)
    {
        DescriptorImageInfo imageInfo = new()
        {
            ImageView   = hdrView,
            Sampler     = sampler,
            ImageLayout = ImageLayout.ShaderReadOnlyOptimal,
        };
        WriteDescriptorSet write = new()
        {
            SType           = StructureType.WriteDescriptorSet,
            DstSet          = DescriptorSets[0][0],
            DstBinding      = 0,
            DstArrayElement = 0,
            DescriptorType  = DescriptorType.CombinedImageSampler,
            DescriptorCount = 1,
            PImageInfo      = &imageInfo,
        };
        Vk.UpdateDescriptorSets(Device, 1, &write, 0, null);
    }

    public void PushConstants(CommandBuffer cmd)
    {
        var pc = new TonemapPushConstants { Exposure = Exposure, Gamma = Gamma };
        Vk.CmdPushConstants(cmd, Layout,
            ShaderStageFlags.FragmentBit,
            0, (uint)sizeof(TonemapPushConstants), &pc);
    }
}


// ────────────────────────────────────────────────────────────────────────────
//  Transparent forward+ pass — renders BLEND-mode materials into HDRColor with
//  src-alpha / one-minus-src-alpha blending, depth-tested LE against the
//  geometry pass's depth buffer (no depth write).
// ────────────────────────────────────────────────────────────────────────────

public sealed unsafe class TransparentPipeline : GraphicsPipeline
{
    // Matches Transparent.slang::FrameUBO. View+proj feed the VS; camPos +
    // tile state feed the FS.
    [StructLayout(LayoutKind.Sequential)]
    struct TransparentFrameUBO
    {
        public Matrix4x4 view;
        public Matrix4x4 proj;
        public Vector4 camPos;
        public uint    lightCount;
        public uint    tileCountX;
        public uint    tileCountY;
        public uint    _pad0;
        public Vector2 screenSize;
        public uint    _pad1;
        public uint    _pad2;
    }

    // Matches Transparent.slang::DrawPC. 80B; well under the 128B Vulkan minimum.
    [StructLayout(LayoutKind.Sequential)]
    struct TransparentPushConstants
    {
        public Matrix4x4 Model;
        public uint      MaterialIndex;
        public uint      _pad0;
        public uint      _pad1;
        public uint      _pad2;
    }

    protected override string ShaderPath { get; } =
        @"C:\Users\jamie\RiderProjects\CadThingo\CadThingo\Assets\Shaders\Transparent.spv";

    protected override Format[] ColorAttachmentFormats { get; } = new[] { Format.R16G16B16A16Sfloat };

    public bool SoftShadowsEnabled { get; init; } = true;

    private const int SetFrame    = 0;
    private const int SetBindless = 1;

    private UboBuffer[] FrameUniformBuffers = new UboBuffer[Renderer.MAX_CONCURRENT_FRAMES];

    public TransparentPipeline(Renderer renderer) : base(renderer)
    {
        DepthAttachmentFormat = renderer.FindDepthFormat();
        PushConstantRanges = new[]
        {
            new PushConstantRange
            {
                StageFlags = ShaderStageFlags.VertexBit | ShaderStageFlags.FragmentBit,
                Offset     = 0,
                Size       = (uint)sizeof(TransparentPushConstants),
            }
        };
    }

    public override void Dispose()
    {
        foreach (var b in FrameUniformBuffers) b.Dispose();
        base.Dispose();
    }

    // Wire constant_id 0 (SOFT_SHADOWS) on the fragment stage — mirrors PbrDeferredPipeline.
    protected override int FillSpecializationData(
        int stageIdx,
        SpecializationMapEntry* entries,
        byte* data,
        out uint dataSize)
    {
        if (stageIdx == 1)
        {
            entries[0] = new SpecializationMapEntry
            {
                ConstantID = 0,
                Offset     = 0,
                Size       = sizeof(uint),
            };
            *(uint*)data = SoftShadowsEnabled ? 1u : 0u;
            dataSize = sizeof(uint);
            return 1;
        }
        dataSize = 0;
        return 0;
    }

    // ── Pipeline state overrides ───────────────────────────────────────────

    protected override PipelineDepthStencilStateCreateInfo BuildDepthStencil() => new()
    {
        SType                 = StructureType.PipelineDepthStencilStateCreateInfo,
        DepthTestEnable       = true,
        DepthWriteEnable      = false,                       // multiple transparent layers stack
        DepthCompareOp        = CompareOp.LessOrEqual,
        DepthBoundsTestEnable = false,
        StencilTestEnable     = false,
        MinDepthBounds        = 0.0f,
        MaxDepthBounds        = 1.0f,
    };

    protected override PipelineRasterizationStateCreateInfo BuildRasterizer() => new()
    {
        SType                   = StructureType.PipelineRasterizationStateCreateInfo,
        DepthClampEnable        = false,
        RasterizerDiscardEnable = false,
        PolygonMode             = PolygonMode.Fill,
        LineWidth               = 1.0f,
        CullMode                = CullModeFlags.None,         // most transparents need both sides visible
        FrontFace               = FrontFace.CounterClockwise,
        DepthBiasEnable         = false,
    };

    // Standard src-alpha / one-minus-src-alpha. Dest alpha tracks accumulated
    // coverage in case anything downstream wants to sample it.
    protected override PipelineColorBlendAttachmentState[] BuildColorBlendAttachments()
    {
        return new[]
        {
            new PipelineColorBlendAttachmentState
            {
                BlendEnable         = true,
                SrcColorBlendFactor = BlendFactor.SrcAlpha,
                DstColorBlendFactor = BlendFactor.OneMinusSrcAlpha,
                ColorBlendOp        = BlendOp.Add,
                SrcAlphaBlendFactor = BlendFactor.One,
                DstAlphaBlendFactor = BlendFactor.OneMinusSrcAlpha,
                AlphaBlendOp        = BlendOp.Add,
                ColorWriteMask      = ColorComponentFlags.RBit | ColorComponentFlags.GBit |
                                      ColorComponentFlags.BBit | ColorComponentFlags.ABit,
            },
        };
    }

    protected override VertexInputBindingDescription[]   GetVertexInputBindings()   => [Vertex.GetBindingDescription()];
    protected override VertexInputAttributeDescription[] GetVertexInputAttributes() => Vertex.GetAttributeDescriptions();

    // ── Descriptor layouts ─────────────────────────────────────────────────

    protected override void CreateDescriptorSetLayouts()
    {
        DescriptorSetLayouts = new DescriptorSetLayout[2];
        OwnedDescriptorSetLayoutIndices = new[] { 0 };

        // Set 0 — own frame UBO + cross-pipeline lighting handles.
        var set0Bindings = new DescriptorSetLayoutBinding[]
        {
            new() { Binding = 0, DescriptorType = DescriptorType.UniformBuffer,            DescriptorCount = 1, StageFlags = ShaderStageFlags.VertexBit | ShaderStageFlags.FragmentBit },
            new() { Binding = 1, DescriptorType = DescriptorType.StorageBuffer,            DescriptorCount = 1, StageFlags = ShaderStageFlags.FragmentBit },
            new() { Binding = 2, DescriptorType = DescriptorType.AccelerationStructureKhr, DescriptorCount = 1, StageFlags = ShaderStageFlags.FragmentBit },
            new() { Binding = 3, DescriptorType = DescriptorType.StorageBuffer,            DescriptorCount = 1, StageFlags = ShaderStageFlags.FragmentBit },
            new() { Binding = 4, DescriptorType = DescriptorType.StorageBuffer,            DescriptorCount = 1, StageFlags = ShaderStageFlags.FragmentBit },
        };

        fixed (DescriptorSetLayoutBinding* pSet0 = set0Bindings)
        {
            var set0LayoutInfo = new DescriptorSetLayoutCreateInfo
            {
                SType        = StructureType.DescriptorSetLayoutCreateInfo,
                BindingCount = (uint)set0Bindings.Length,
                PBindings    = pSet0,
            };
            if (Vk.CreateDescriptorSetLayout(Device, &set0LayoutInfo, null, out DescriptorSetLayouts[SetFrame]) != Result.Success)
                throw new Exception("Failed to create transparent set 0 layout");
        }

        // Set 1 — bindless, borrowed from ResourceManager (matches GeometryPipeline).
        DescriptorSetLayouts[SetBindless] = Engine.ResourceManager.GetBindlessLayout();
    }

    protected override void CreateResources()
    {
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            Renderer.CreateMappedUniformBuffer(sizeof(TransparentFrameUBO), ref FrameUniformBuffers[i]);
        }
    }

    protected override void CreateDescriptorSets()
    {
        var layouts = stackalloc DescriptorSetLayout[(int)Renderer.MAX_CONCURRENT_FRAMES];
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++) layouts[i] = DescriptorSetLayouts[SetFrame];

        var allocInfo = new DescriptorSetAllocateInfo
        {
            SType              = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool     = Renderer.descriptorPool,
            DescriptorSetCount = Renderer.MAX_CONCURRENT_FRAMES,
            PSetLayouts        = layouts,
        };
        DescriptorSets = new DescriptorSet[1][];
        DescriptorSets[0] = new DescriptorSet[Renderer.MAX_CONCURRENT_FRAMES];
        fixed (DescriptorSet* pSets = DescriptorSets[0])
        {
            if (Vk.AllocateDescriptorSets(Device, &allocInfo, pSets) != Result.Success)
                throw new Exception("Failed to allocate transparent descriptor sets");
        }
    }

    // Only binding 0 (own UBO) is written here. Bindings 1–4 (lights / TLAS / tile
    // buffers) point at PbrDeferredPipeline / LightCullPipeline's buffers and are
    // written by the renderer after those pipelines exist.
    protected override void WriteDescriptors()
    {
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            DescriptorBufferInfo frameInfo = new()
            {
                Buffer = FrameUniformBuffers[i].buffer,
                Offset = 0,
                Range  = (ulong)sizeof(TransparentFrameUBO),
            };
            var write = new WriteDescriptorSet
            {
                SType           = StructureType.WriteDescriptorSet,
                DstSet          = DescriptorSets[0][i],
                DstBinding      = 0,
                DstArrayElement = 0,
                DescriptorType  = DescriptorType.UniformBuffer,
                DescriptorCount = 1,
                PBufferInfo     = &frameInfo,
            };
            Vk.UpdateDescriptorSets(Device, 1, &write, 0, null);
        }
    }

    /// <summary>Bind the per-frame lights SSBO + the tile cull output buffers
    /// produced by LightCullPipeline. Call once at startup after both producer
    /// pipelines exist.</summary>
    public void WriteSharedLightingDescriptors(PbrDeferredPipeline pbr, LightCullPipeline lightCull)
    {
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            DescriptorBufferInfo lightsInfo = new()
            {
                Buffer = pbr.GetLightStorageBuffer((uint)i),
                Offset = 0,
                Range  = (ulong)(Renderer.MAX_LIGHTS * (uint)sizeof(PbrLightGpu)),
            };
            DescriptorBufferInfo tileCountInfo = new()
            {
                Buffer = lightCull.GetTileLightCountBuffer((uint)i),
                Offset = 0,
                Range  = (ulong)(Renderer.MAX_TILE_COUNT * sizeof(uint)),
            };
            DescriptorBufferInfo tileIdxInfo = new()
            {
                Buffer = lightCull.GetTileLightIndicesBuffer((uint)i),
                Offset = 0,
                Range  = (ulong)(Renderer.MAX_TILE_COUNT * Renderer.MAX_LIGHTS_PER_TILE * sizeof(uint)),
            };

            var writes = stackalloc WriteDescriptorSet[3];
            writes[0] = new() { SType = StructureType.WriteDescriptorSet, DstSet = DescriptorSets[0][i], DstBinding = 1, DescriptorType = DescriptorType.StorageBuffer, DescriptorCount = 1, PBufferInfo = &lightsInfo };
            writes[1] = new() { SType = StructureType.WriteDescriptorSet, DstSet = DescriptorSets[0][i], DstBinding = 3, DescriptorType = DescriptorType.StorageBuffer, DescriptorCount = 1, PBufferInfo = &tileCountInfo };
            writes[2] = new() { SType = StructureType.WriteDescriptorSet, DstSet = DescriptorSets[0][i], DstBinding = 4, DescriptorType = DescriptorType.StorageBuffer, DescriptorCount = 1, PBufferInfo = &tileIdxInfo };
            Vk.UpdateDescriptorSets(Device, 3, writes, 0, null);
        }
    }

    /// <summary>Mirror of PbrDeferredPipeline.WriteTlasDescriptor — call after InitRayQuery
    /// and on every TLAS recreate.</summary>
    public void WriteTlasDescriptor(AccelerationStructureKHR tlas)
    {
        if (tlas.Handle == 0) return;

        var tlasHandle = tlas;
        var asWrite = new WriteDescriptorSetAccelerationStructureKHR
        {
            SType = StructureType.WriteDescriptorSetAccelerationStructureKhr,
            AccelerationStructureCount = 1,
            PAccelerationStructures = &tlasHandle,
        };
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            var write = new WriteDescriptorSet
            {
                SType           = StructureType.WriteDescriptorSet,
                PNext           = &asWrite,
                DstSet          = DescriptorSets[0][i],
                DstBinding      = 2,
                DescriptorType  = DescriptorType.AccelerationStructureKhr,
                DescriptorCount = 1,
            };
            Vk.UpdateDescriptorSets(Device, 1, &write, 0, null);
        }
    }

    /// <summary>Fill the per-frame FrameUBO from the current camera + tile counts.
    /// Call once per frame from DrawFrame; tileCount / lightCount come from
    /// PbrDeferredPipeline.UpdatePerFrame so the two pipelines stay coherent.</summary>
    public void UpdatePerFrame(uint frameIndex, Camera camera, uint lightCount, uint tileCountX, uint tileCountY)
    {
        TransparentFrameUBO ubo = new();
        if (camera != null)
        {
            ubo.proj = camera.GetProjectionMatrix(
                (float)Renderer.swapChainExtent.Width / Renderer.swapChainExtent.Height, 0.1f, 100.0f);
            ubo.view = camera.GetViewMatrix();
            ubo.proj.M22 *= -1;
            ubo.camPos = new Vector4(camera.GetPosition(), 1.0f);
        }
        else
        {
            ubo.view   = Matrix4x4.CreateLookAt(new Vector3(2, 2, 2), Vector3.Zero, new Vector3(0, 0, 1));
            ubo.proj   = Matrix4x4.CreatePerspectiveFieldOfView((float)(45 * Math.PI / 180),
                (float)Renderer.swapChainExtent.Width / Renderer.swapChainExtent.Height, 0.1f, 100.0f);
            ubo.proj.M22 *= -1;
            ubo.camPos = new Vector4(2, 2, 2, 1);
        }
        ubo.lightCount = lightCount;
        ubo.tileCountX = tileCountX;
        ubo.tileCountY = tileCountY;
        ubo.screenSize = new Vector2(Renderer.swapChainExtent.Width, Renderer.swapChainExtent.Height);

        void* data = FrameUniformBuffers[frameIndex].mapped;
        new Span<TransparentFrameUBO>(data, 1).Fill(ubo);
    }

    /// <summary>Push the per-draw model matrix + material index. Called once per transparent draw.</summary>
    public void PushDrawConstants(CommandBuffer cmd, in Matrix4x4 model, uint materialIndex)
    {
        var pc = new TransparentPushConstants
        {
            Model         = model,
            MaterialIndex = materialIndex,
        };
        Vk.CmdPushConstants(cmd, Layout,
            ShaderStageFlags.VertexBit | ShaderStageFlags.FragmentBit,
            0, (uint)sizeof(TransparentPushConstants), &pc);
    }
}