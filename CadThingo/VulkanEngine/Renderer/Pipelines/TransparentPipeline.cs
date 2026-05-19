using System.Numerics;
using System.Runtime.InteropServices;
using CadThingo.VulkanEngine.ImGui;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Pipelines;


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
        // Repurposed from former trailing 8B pad — matches LightingFrameUBO IBL
        // params byte-for-byte so the same Renderer.prefilteredCubeMipLevels +
        // scaleIBLAmbient story applies on the transparent pass.
        public float   prefilteredCubeMipLevels;
        public float   scaleIBLAmbient;
        // Probe cluster grid dims (1 for Z when tile-only) + cubemap-array mip count.
        public uint    probeClusterDimsX;
        public uint    probeClusterDimsY;
        public uint    probeClusterDimsZ;
        public float   probeMipLevels;
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
        foreach (var b in FrameUniformBuffers) Renderer.DestroyBuffer(b.buffer, b.alloc);
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

        // Set 0 — own frame UBO + cross-pipeline lighting handles + IBL samplers.
        var set0Bindings = new DescriptorSetLayoutBinding[]
        {
            new() { Binding = 0, DescriptorType = DescriptorType.UniformBuffer,            DescriptorCount = 1, StageFlags = ShaderStageFlags.VertexBit | ShaderStageFlags.FragmentBit },
            new() { Binding = 1, DescriptorType = DescriptorType.StorageBuffer,            DescriptorCount = 1, StageFlags = ShaderStageFlags.FragmentBit },
            new() { Binding = 2, DescriptorType = DescriptorType.AccelerationStructureKhr, DescriptorCount = 1, StageFlags = ShaderStageFlags.FragmentBit },
            new() { Binding = 3, DescriptorType = DescriptorType.StorageBuffer,            DescriptorCount = 1, StageFlags = ShaderStageFlags.FragmentBit },
            new() { Binding = 4, DescriptorType = DescriptorType.StorageBuffer,            DescriptorCount = 1, StageFlags = ShaderStageFlags.FragmentBit },
            // IBL — mirrors PbrDeferredPipeline slots 5/6/7. Mirror is intentional:
            // PBR + Transparent each own a distinct set 0 layout, both pointing at
            // the same renderer-wide IBL VkImages.
            new() { Binding = 5, DescriptorType = DescriptorType.CombinedImageSampler,     DescriptorCount = 1, StageFlags = ShaderStageFlags.FragmentBit },
            new() { Binding = 6, DescriptorType = DescriptorType.CombinedImageSampler,     DescriptorCount = 1, StageFlags = ShaderStageFlags.FragmentBit },
            new() { Binding = 7, DescriptorType = DescriptorType.CombinedImageSampler,     DescriptorCount = 1, StageFlags = ShaderStageFlags.FragmentBit },
            // Reflection probes — mirrors PbrDeferredPipeline slots 8/9/10/11.
            new() { Binding = 8,  DescriptorType = DescriptorType.CombinedImageSampler,    DescriptorCount = 1, StageFlags = ShaderStageFlags.FragmentBit },
            new() { Binding = 9,  DescriptorType = DescriptorType.StorageBuffer,           DescriptorCount = 1, StageFlags = ShaderStageFlags.FragmentBit },
            new() { Binding = 10, DescriptorType = DescriptorType.StorageBuffer,           DescriptorCount = 1, StageFlags = ShaderStageFlags.FragmentBit },
            new() { Binding = 11, DescriptorType = DescriptorType.StorageBuffer,           DescriptorCount = 1, StageFlags = ShaderStageFlags.FragmentBit },
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

    /// <summary>Mirror of PbrDeferredPipeline.WriteIblDescriptors — same renderer-
    /// wide images, separate descriptor sets. Called once at startup after the
    /// IBL resources exist; doesn't need re-running on rebake because the VkImage
    /// handles persist.</summary>
    public void WriteIblDescriptors()
    {
        var imageInfos = stackalloc DescriptorImageInfo[3]
        {
            new() { ImageView = Renderer.irradianceCubeView,  Sampler = Renderer.iblCubeSampler, ImageLayout = ImageLayout.ShaderReadOnlyOptimal },
            new() { ImageView = Renderer.prefilteredCubeView, Sampler = Renderer.iblCubeSampler, ImageLayout = ImageLayout.ShaderReadOnlyOptimal },
            new() { ImageView = Renderer.brdfLutView,         Sampler = Renderer.iblLutSampler,  ImageLayout = ImageLayout.ShaderReadOnlyOptimal },
        };

        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            var writes = stackalloc WriteDescriptorSet[3];
            for (uint b = 0; b < 3; b++)
            {
                writes[b] = new WriteDescriptorSet
                {
                    SType           = StructureType.WriteDescriptorSet,
                    DstSet          = DescriptorSets[0][i],
                    DstBinding      = 5 + b,
                    DstArrayElement = 0,
                    DescriptorType  = DescriptorType.CombinedImageSampler,
                    DescriptorCount = 1,
                    PImageInfo      = &imageInfos[b],
                };
            }
            Vk.UpdateDescriptorSets(Device, 3, writes, 0, null);
        }
    }

    /// <summary>Mirror of PbrDeferredPipeline.WriteProbeDescriptors — writes
    /// bindings 8/9/10/11 on each per-frame transparent set. Call once after
    /// the probe system exists; underlying handles are stable for the
    /// renderer's lifetime so no per-frame rewrites needed.</summary>
    public void WriteProbeDescriptors()
    {
        var probeSys = Renderer.reflectionProbeSystem;
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            DescriptorImageInfo cubeArrayInfo = new()
            {
                Sampler     = probeSys.prefilteredArraySampler,
                ImageView   = probeSys.prefilteredArrayView,
                ImageLayout = ImageLayout.ShaderReadOnlyOptimal,
            };
            DescriptorBufferInfo recordsInfo = new()
            {
                Buffer = probeSys.probeRecordBuffers[i].buffer,
                Offset = 0,
                Range  = ReflectionProbeSystem.MaxProbes * (ulong)sizeof(ProbeGpuRecord),
            };
            DescriptorBufferInfo clusterRangeInfo = new()
            {
                Buffer = probeSys.clusterGrid.GetClusterRangeBuffer((uint)i),
                Offset = 0,
                Range  = probeSys.clusterGrid.ClusterRangeBufferSize,
            };
            DescriptorBufferInfo indexListInfo = new()
            {
                Buffer = probeSys.clusterGrid.GetProbeIndexBuffer((uint)i),
                Offset = 0,
                Range  = probeSys.clusterGrid.ProbeIndexBufferSize,
            };

            var writes = stackalloc WriteDescriptorSet[4];
            writes[0] = new WriteDescriptorSet
            {
                SType = StructureType.WriteDescriptorSet, DstSet = DescriptorSets[0][i],
                DstBinding = 8, DescriptorType = DescriptorType.CombinedImageSampler,
                DescriptorCount = 1, PImageInfo = &cubeArrayInfo,
            };
            writes[1] = new WriteDescriptorSet
            {
                SType = StructureType.WriteDescriptorSet, DstSet = DescriptorSets[0][i],
                DstBinding = 9, DescriptorType = DescriptorType.StorageBuffer,
                DescriptorCount = 1, PBufferInfo = &recordsInfo,
            };
            writes[2] = new WriteDescriptorSet
            {
                SType = StructureType.WriteDescriptorSet, DstSet = DescriptorSets[0][i],
                DstBinding = 10, DescriptorType = DescriptorType.StorageBuffer,
                DescriptorCount = 1, PBufferInfo = &clusterRangeInfo,
            };
            writes[3] = new WriteDescriptorSet
            {
                SType = StructureType.WriteDescriptorSet, DstSet = DescriptorSets[0][i],
                DstBinding = 11, DescriptorType = DescriptorType.StorageBuffer,
                DescriptorCount = 1, PBufferInfo = &indexListInfo,
            };
            Vk.UpdateDescriptorSets(Device, 4, writes, 0, null);
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
                (float)Renderer.renderExtent.Width / Renderer.renderExtent.Height, 0.1f, 100.0f);
            ubo.view = camera.GetViewMatrix();
            ubo.proj.M22 *= -1;
            ubo.camPos = new Vector4(camera.GetPosition(), 1.0f);
        }
        else
        {
            ubo.view   = Matrix4x4.CreateLookAt(new Vector3(2, 2, 2), Vector3.Zero, new Vector3(0, 0, 1));
            ubo.proj   = Matrix4x4.CreatePerspectiveFieldOfView((float)(45 * Math.PI / 180),
                (float)Renderer.renderExtent.Width / Renderer.renderExtent.Height, 0.1f, 100.0f);
            ubo.proj.M22 *= -1;
            ubo.camPos = new Vector4(2, 2, 2, 1);
        }
        ubo.lightCount = lightCount;
        ubo.tileCountX = tileCountX;
        ubo.tileCountY = tileCountY;
        ubo.screenSize = new Vector2(Renderer.renderExtent.Width, Renderer.renderExtent.Height);
        ubo.prefilteredCubeMipLevels = Renderer.prefilteredCubeMipLevels;
        ubo.scaleIBLAmbient          = EditorState.IblIntensity;

        // Probe cluster dims — built once per frame by ReflectionProbeSystem.
        // The transparent pass uses the same grid as PbrDeferred so cluster
        // indices stay consistent across opaque and transparent samples.
        var grid = Renderer.reflectionProbeSystem.clusterGrid;
        ubo.probeClusterDimsX = grid.DimsX;
        ubo.probeClusterDimsY = grid.DimsY;
        ubo.probeClusterDimsZ = grid.DimsZ;
        ubo.probeMipLevels    = ReflectionProbeSystem.ProbeMipLevels;

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