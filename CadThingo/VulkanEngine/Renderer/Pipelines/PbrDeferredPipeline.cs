using System.Numerics;
using System.Runtime.InteropServices;
using CadThingo.VulkanEngine.ImGui;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Pipelines;

//  PBR deferred lighting pass — fullscreen triangle, samples G-buffer +
//  per-tile light list, optional ray-queried shadows.
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
        // Probe cluster grid dims (1 for Z when tile-only) + cubemap-array mip count.
        public uint probeClusterDimsX;
        public uint probeClusterDimsY;
        public uint probeClusterDimsZ;
        public float probeMipLevels;
    }

    protected override string ShaderPath { get; } =
        @"C:\Users\jamie\RiderProjects\CadThingo\CadThingo\Assets\Shaders\PBR.spv";

    // Lighting writes linear HDR scene-referred color; tone-map + gamma run in
    // the separate TonemapPipeline pass that consumes this attachment.
    protected override Format[] ColorAttachmentFormats { get; } = new[] { Format.R16G16B16A16Sfloat };

    // Set 0 — per-frame lighting (UBO, lights SSBO, TLAS, tile cull buffers).
    // Set 1 — shared G-buffer samplers (one allocation reused every frame).
    // Set 2 — shadow-alpha plumbing (ShadowEntityInfo + global vb/ib as
    //         ByteAddressBuffers; one shared allocation, rewritten when the
    //         shadow info VkBuffer is reallocated).
    // Set 3 — bindless materials/instances/textures/samplers — borrowed from
    //         ResourceManager so the shadow-alpha path can sample baseColor
    //         alpha at hit time without duplicating the bindless allocation.
    // The set indices match what PbrShader.slang expects.
    private const int SetLighting     = 0;
    private const int SetGBuffer      = 1;
    private const int SetShadowAlpha  = 2;
    private const int SetBindless     = 3;

    // Per-frame data owned by this pipeline. The lights SSBO it formerly owned
    // now lives on Renderer (Renderer.lightStorageBuffers) so the pathtracer +
    // forward+ pass can read it without depending on this pipeline existing.
    private UboBuffer[] LightingUniformBuffers = new UboBuffer[Renderer.MAX_CONCURRENT_FRAMES];

    /// <summary>True = wire the PCSS-style soft-shadow specialization constant on,
    /// pulled into the fragment shader as <c>constant_id 0</c>. Read once at
    /// pipeline build; toggling requires Dispose + rebuild.</summary>
    public bool SoftShadowsEnabled { get; init; } = true;

    public PbrDeferredPipeline(Renderer renderer) : base(renderer) { }

    public override void Dispose()
    {
        foreach (var b in LightingUniformBuffers) Gfx.DestroyBuffer(b.buffer, b.alloc);
        base.Dispose();
    }

    internal void Record(CommandBuffer cmd, in Renderer.FrameContext ctx, ImageView HdrTarget)
    {
        //configure single color output for final lighting result

        BeginRendering(cmd, ctx.RenderExtent, [HdrTarget]);
        Vk!.CmdBindPipeline(cmd, PipelineBindPoint.Graphics, Handle);

        Viewport vp = new()
        {
            X = 0, Y = 0,
            Width = ctx.RenderExtent.Width, Height = ctx.RenderExtent.Height,
            MinDepth = 0.0f, MaxDepth = 1.0f,
        };
        Rect2D scissor = new(new Offset2D(0, 0), ctx.RenderExtent);
        Vk!.CmdSetViewport(cmd, 0, 1, &vp);
        Vk!.CmdSetScissor(cmd, 0, 1, &scissor);

        // Set 0 = per-frame LightingUBO + Lights SSBO + TLAS + tile cull buffers.
        // Set 1 = shared G-buffer samplers (single allocation reused every frame).
        // No push constants — lighting pass reads everything from descriptor bindings.
        // Set 2 = shadow-alpha (ShadowEntityInfo + global vb/ib).
        // Set 3 = bindless materials/textures/samplers, owned by ResourceManager.
        // The shadow-ray Proceed loop in PbrShader.slang reads both at hit time.
        var sets = stackalloc DescriptorSet[4]
        {
            GetDescriptorSet(0, ctx.FrameIndex),
            GetDescriptorSet(1, 0),
            GetDescriptorSet(2, 0),
            Engine.ResourceManager.GetBindlessSet(ctx.FrameIndex),
        };
        Vk!.CmdBindDescriptorSets(cmd, PipelineBindPoint.Graphics,
            Layout, 0, 4, sets, 0, null);

        // Fullscreen triangle — VSMain synthesizes 3 verts from SV_VertexID
        Vk!.CmdDraw(cmd, 3, 1, 0, 0);
        
        EndRendering(cmd);
    }
    // Shader-stage overrides 
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


    protected override void CreateDescriptorSetLayouts()
    {
        // Layouts 0/1/2 are owned by this pipeline. Layout 3 (bindless) is
        // borrowed from ResourceManager and must NOT be destroyed in Dispose.
        DescriptorSetLayouts = new DescriptorSetLayout[4];
        OwnedDescriptorSetLayoutIndices = new[] { 0, 1, 2 };

        //  Set 0: LightingFrameUBO + Light SSBO + TLAS + Tile cull buffers 
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
            //  IBL split-sum (Phase 1: bound to black-cleared images; Phase 3
            // adds the matching declarations to PbrShader.slang). The shader is
            // free to declare fewer bindings than the layout exposes, so adding
            // these now causes no SPIR-V change.
            new() // irradianceCube — Lambert hemisphere convolution (diffuse IBL).
            {
                Binding = 5,
                DescriptorType = DescriptorType.CombinedImageSampler,
                DescriptorCount = 1,
                StageFlags = ShaderStageFlags.FragmentBit,
                PImmutableSamplers = null,
            },
            new() // prefilteredCube — per-roughness GGX-prefiltered environment.
            {
                Binding = 6,
                DescriptorType = DescriptorType.CombinedImageSampler,
                DescriptorCount = 1,
                StageFlags = ShaderStageFlags.FragmentBit,
                PImmutableSamplers = null,
            },
            new() // brdfLut — split-sum 2D LUT (NdotV × roughness → scale,bias).
            {
                Binding = 7,
                DescriptorType = DescriptorType.CombinedImageSampler,
                DescriptorCount = 1,
                StageFlags = ShaderStageFlags.FragmentBit,
                PImmutableSamplers = null,
            },
            // Reflection-probe bindings (Phase 7)
            // 8: probeCubeArray (sampled cube array of every probe's prefiltered specular)
            // 9: probes         (per-probe ProbeGpuRecord SSBO; world pos + radius + slot)
            // 10: clusterRange  (per-cluster (offset,count) into probeIndexList)
            // 11: probeIndexList (flat uint list of probe indices per cluster)
            new()
            {
                Binding = 8,
                DescriptorType = DescriptorType.CombinedImageSampler,
                DescriptorCount = 1,
                StageFlags = ShaderStageFlags.FragmentBit,
                PImmutableSamplers = null,
            },
            new()
            {
                Binding = 9,
                DescriptorType = DescriptorType.StorageBuffer,
                DescriptorCount = 1,
                StageFlags = ShaderStageFlags.FragmentBit,
                PImmutableSamplers = null,
            },
            new()
            {
                Binding = 10,
                DescriptorType = DescriptorType.StorageBuffer,
                DescriptorCount = 1,
                StageFlags = ShaderStageFlags.FragmentBit,
                PImmutableSamplers = null,
            },
            new()
            {
                Binding = 11,
                DescriptorType = DescriptorType.StorageBuffer,
                DescriptorCount = 1,
                StageFlags = ShaderStageFlags.FragmentBit,
                PImmutableSamplers = null,
            },
        };

        // Set 1: G-Buffer inputs
        // Five samplers written by the geometry pass, read here for lighting.
        var set1Bindings = new DescriptorSetLayoutBinding[]
        {
            new() { Binding = 0, DescriptorType = DescriptorType.CombinedImageSampler, DescriptorCount = 1, StageFlags = ShaderStageFlags.FragmentBit, PImmutableSamplers = null },
            new() { Binding = 1, DescriptorType = DescriptorType.CombinedImageSampler, DescriptorCount = 1, StageFlags = ShaderStageFlags.FragmentBit, PImmutableSamplers = null },
            new() { Binding = 2, DescriptorType = DescriptorType.CombinedImageSampler, DescriptorCount = 1, StageFlags = ShaderStageFlags.FragmentBit, PImmutableSamplers = null },
            new() { Binding = 3, DescriptorType = DescriptorType.CombinedImageSampler, DescriptorCount = 1, StageFlags = ShaderStageFlags.FragmentBit, PImmutableSamplers = null },
            new() { Binding = 4, DescriptorType = DescriptorType.CombinedImageSampler, DescriptorCount = 1, StageFlags = ShaderStageFlags.FragmentBit, PImmutableSamplers = null },
        };

        // Set 2: shadow-alpha plumbing
        // ShadowEntityInfo (per-entity index/material lookup), globalIndices,
        // globalVertices. All three are read in the fragment stage by the
        // ray-query Proceed loop only when the candidate is non-opaque.
        var set2Bindings = new DescriptorSetLayoutBinding[]
        {
            new() { Binding = 0, DescriptorType = DescriptorType.StorageBuffer, DescriptorCount = 1, StageFlags = ShaderStageFlags.FragmentBit, PImmutableSamplers = null },
            new() { Binding = 1, DescriptorType = DescriptorType.StorageBuffer, DescriptorCount = 1, StageFlags = ShaderStageFlags.FragmentBit, PImmutableSamplers = null },
            new() { Binding = 2, DescriptorType = DescriptorType.StorageBuffer, DescriptorCount = 1, StageFlags = ShaderStageFlags.FragmentBit, PImmutableSamplers = null },
        };

        DescriptorSetLayoutBindingFlagsCreateInfo set0FlagsInfo = new()
            { SType = StructureType.DescriptorSetLayoutBindingFlagsCreateInfo };
        DescriptorSetLayoutBindingFlagsCreateInfo set1FlagsInfo = new()
            { SType = StructureType.DescriptorSetLayoutBindingFlagsCreateInfo };

        var set0Flags = stackalloc DescriptorBindingFlags[set0Bindings.Length];
        var set1Flags = stackalloc DescriptorBindingFlags[set1Bindings.Length];

        fixed (DescriptorSetLayoutBinding* pSet0 = set0Bindings)
        fixed (DescriptorSetLayoutBinding* pSet1 = set1Bindings)
        fixed (DescriptorSetLayoutBinding* pSet2 = set2Bindings)
        {
            if (Gfx.DescriptorIndexingEnabled)
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
            if (Gfx.DescriptorIndexingEnabled)
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
            if (Gfx.DescriptorIndexingEnabled)
            {
                set1LayoutInfo.Flags |= DescriptorSetLayoutCreateFlags.UpdateAfterBindPoolBit;
                set1LayoutInfo.PNext = &set1FlagsInfo;
            }
            if (Vk.CreateDescriptorSetLayout(Device, &set1LayoutInfo, null, out DescriptorSetLayouts[SetGBuffer]) != Result.Success)
                throw new Exception("Failed to create PBR set 1 (GBuffer) descriptor set layout");

            // Set 2: shadow-alpha plumbing. Pure StorageBuffers — no
            // UpdateAfterBind chain (the underlying VkBuffer reallocates rarely
            // and the renderer pauses the device before rewriting). Standard
            // pool flags are fine.
            DescriptorSetLayoutCreateInfo set2LayoutInfo = new()
            {
                SType        = StructureType.DescriptorSetLayoutCreateInfo,
                BindingCount = (uint)set2Bindings.Length,
                PBindings    = pSet2,
            };
            if (Vk.CreateDescriptorSetLayout(Device, &set2LayoutInfo, null, out DescriptorSetLayouts[SetShadowAlpha]) != Result.Success)
                throw new Exception("Failed to create PBR set 2 (ShadowAlpha) descriptor set layout");
        }

        // Set 3: borrow ResourceManager's bindless layout — owned there, freed
        // there. The pipeline layout still needs it slotted in.
        DescriptorSetLayouts[SetBindless] = Engine.ResourceManager.GetBindlessLayout();
    }

    // Resources

    protected override void CreateResources()
    {
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            Gfx.CreateMappedUniformBuffer(sizeof(LightingFrameUBO), ref LightingUniformBuffers[i]);
        }
    }


    protected override void CreateDescriptorSets()
    {
        // Set 0 — per-frame; one set per frame in flight.
        var lightingLayouts = stackalloc DescriptorSetLayout[(int)Renderer.MAX_CONCURRENT_FRAMES];
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++) lightingLayouts[i] = DescriptorSetLayouts[SetLighting];

        DescriptorSetAllocateInfo lightingAlloc = new()
        {
            SType              = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool     = Gfx.DescriptorPool,
            DescriptorSetCount = Renderer.MAX_CONCURRENT_FRAMES,
            PSetLayouts        = lightingLayouts,
        };

        DescriptorSets = new DescriptorSet[4][];
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
            DescriptorPool     = Gfx.DescriptorPool,
            DescriptorSetCount = 1,
            PSetLayouts        = &gBufLayout,
        };
        DescriptorSets[SetGBuffer] = new DescriptorSet[1];
        fixed (DescriptorSet* pSet = DescriptorSets[SetGBuffer])
        {
            if (Vk.AllocateDescriptorSets(Device, &gBufAlloc, pSet) != Result.Success)
                throw new Exception("Failed to allocate PBR g-buffer descriptor set");
        }

        // Set 2 — shared shadow-alpha set. The bound buffers are renderer-wide
        // singletons (one ShadowEntityInfo SSBO, one global vb, one global ib),
        // so frame-in-flight duplication isn't needed.
        var shadowLayout = DescriptorSetLayouts[SetShadowAlpha];
        DescriptorSetAllocateInfo shadowAlloc = new()
        {
            SType              = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool     = Gfx.DescriptorPool,
            DescriptorSetCount = 1,
            PSetLayouts        = &shadowLayout,
        };
        DescriptorSets[SetShadowAlpha] = new DescriptorSet[1];
        fixed (DescriptorSet* pSet = DescriptorSets[SetShadowAlpha])
        {
            if (Vk.AllocateDescriptorSets(Device, &shadowAlloc, pSet) != Result.Success)
                throw new Exception("Failed to allocate PBR shadow-alpha descriptor set");
        }

        // Set 3 — bindless set is owned by ResourceManager. We don't allocate
        // here; callers bind ResourceManager.GetBindlessSet(frame) at draw time.
        DescriptorSets[SetBindless] = null;
    }

    // Descriptor writes
    // Writes split into phases so cross-pipeline / TLAS deps can be wired post-Initialize:
    //   - WriteDescriptors  (auto from Initialize): bindings 0,1 of set 0.
    //   - WriteTileBufferDescriptors(lightCull):    bindings 3,4 of set 0.
    //   - WriteTlasDescriptor(tlas):                binding 2 of set 0.
    // Set 1 (the g-buffer samplers) is written by WriteGBufferDescriptors from the deferred
    // FrameGraph's freshly-allocated transient views - initial build + every resize. The
    // views don't exist at Initialize time, so it is NOT called from here.

    protected override void WriteDescriptors()
    {
        WriteLightFrameDescriptors();
        WriteIblDescriptors();
    }

    /// <summary>Writes bindings 5/6/7 (irradiance cube, prefiltered cube, BRDF LUT)
    /// on every per-frame lighting set. The underlying VkImage handles don't
    /// change when content is rebaked, so this only needs to be called when the
    /// set is first allocated — and again if the renderer ever reallocates the
    /// IBL images.</summary>
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
                    DstSet          = DescriptorSets[SetLighting][i],
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
                Buffer = Renderer.GetLightStorageBuffer((uint)i),
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

    /// <summary>Writes bindings 8/9/10/11 (probe cube array + probe records SSBO
    /// + cluster range SSBO + probe index list SSBO) on each per-frame lighting
    /// set. Called once after the probe system is initialized; the underlying
    /// VkImage / VkBuffer handles are stable for the renderer's lifetime so
    /// per-frame rewrites aren't needed.</summary>
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
                SType = StructureType.WriteDescriptorSet,
                DstSet = DescriptorSets[SetLighting][i],
                DstBinding = 8, DstArrayElement = 0,
                DescriptorType = DescriptorType.CombinedImageSampler,
                DescriptorCount = 1, PImageInfo = &cubeArrayInfo,
            };
            writes[1] = new WriteDescriptorSet
            {
                SType = StructureType.WriteDescriptorSet,
                DstSet = DescriptorSets[SetLighting][i],
                DstBinding = 9, DstArrayElement = 0,
                DescriptorType = DescriptorType.StorageBuffer,
                DescriptorCount = 1, PBufferInfo = &recordsInfo,
            };
            writes[2] = new WriteDescriptorSet
            {
                SType = StructureType.WriteDescriptorSet,
                DstSet = DescriptorSets[SetLighting][i],
                DstBinding = 10, DstArrayElement = 0,
                DescriptorType = DescriptorType.StorageBuffer,
                DescriptorCount = 1, PBufferInfo = &clusterRangeInfo,
            };
            writes[3] = new WriteDescriptorSet
            {
                SType = StructureType.WriteDescriptorSet,
                DstSet = DescriptorSets[SetLighting][i],
                DstBinding = 11, DstArrayElement = 0,
                DescriptorType = DescriptorType.StorageBuffer,
                DescriptorCount = 1, PBufferInfo = &indexListInfo,
            };
            Vk.UpdateDescriptorSets(Device, 4, writes, 0, null);
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

    /// <summary>Binds set 2: ShadowEntityInfo SSBO + global vertex/index
    /// buffers (as raw storage buffers for ByteAddressBuffer access in the
    /// shader). Call once after InitRayQuery has built the SSBO, and again any
    /// time the ShadowEntityInfo VkBuffer is reallocated by RebuildTlas.</summary>
    public void WriteShadowAlphaDescriptors()
    {
        var rm = Engine.ResourceManager;
        var shadowBuf = Renderer.ShadowInfoBuffer;
        if (shadowBuf.Handle == 0) return;

        DescriptorBufferInfo shadowInfo = new()
        {
            Buffer = shadowBuf,
            Offset = 0,
            Range  = Renderer.ShadowInfoBufferSize,
        };
        DescriptorBufferInfo indexInfo = new()
        {
            Buffer = rm.GlobalIndexBuffer,
            Offset = 0,
            Range  = Vk.WholeSize,
        };
        DescriptorBufferInfo vertexInfo = new()
        {
            Buffer = rm.GlobalVertexBuffer,
            Offset = 0,
            Range  = Vk.WholeSize,
        };

        var writes = stackalloc WriteDescriptorSet[3];
        writes[0] = new WriteDescriptorSet
        {
            SType           = StructureType.WriteDescriptorSet,
            DstSet          = DescriptorSets[SetShadowAlpha][0],
            DstBinding      = 0,
            DstArrayElement = 0,
            DescriptorType  = DescriptorType.StorageBuffer,
            DescriptorCount = 1,
            PBufferInfo     = &shadowInfo,
        };
        writes[1] = new WriteDescriptorSet
        {
            SType           = StructureType.WriteDescriptorSet,
            DstSet          = DescriptorSets[SetShadowAlpha][0],
            DstBinding      = 1,
            DstArrayElement = 0,
            DescriptorType  = DescriptorType.StorageBuffer,
            DescriptorCount = 1,
            PBufferInfo     = &indexInfo,
        };
        writes[2] = new WriteDescriptorSet
        {
            SType           = StructureType.WriteDescriptorSet,
            DstSet          = DescriptorSets[SetShadowAlpha][0],
            DstBinding      = 2,
            DstArrayElement = 0,
            DescriptorType  = DescriptorType.StorageBuffer,
            DescriptorCount = 1,
            PBufferInfo     = &vertexInfo,
        };
        Vk.UpdateDescriptorSets(Device, 3, writes, 0, null);
    }

    /// <summary>Re-writes the 5 g-buffer sampler bindings on set 1 from the deferred
    /// FrameGraph's transient g-buffer views. Called from BuildDeferredFrameGraph after
    /// Compile (initial build + every resize), since those views are freshly allocated each
    /// Compile and don't exist when the pipeline first initializes.</summary>
    public void WriteGBufferDescriptors(ImageView position, ImageView normal,
        ImageView albedo, ImageView material, ImageView emissive)
    {
        var sampler = Renderer.gBufferSampler;
        var imageInfos = stackalloc DescriptorImageInfo[5]
        {
            new() { ImageView = position, Sampler = sampler, ImageLayout = ImageLayout.ShaderReadOnlyOptimal },
            new() { ImageView = normal,   Sampler = sampler, ImageLayout = ImageLayout.ShaderReadOnlyOptimal },
            new() { ImageView = albedo,   Sampler = sampler, ImageLayout = ImageLayout.ShaderReadOnlyOptimal },
            new() { ImageView = material, Sampler = sampler, ImageLayout = ImageLayout.ShaderReadOnlyOptimal },
            new() { ImageView = emissive, Sampler = sampler, ImageLayout = ImageLayout.ShaderReadOnlyOptimal },
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

    // Per-frame upload
    // Walks scene lights into the per-frame Light SSBO and updates the lighting
    // UBO with camera + tile counts. Returns (lightCount, tileX, tileY) so the
    // renderer can drive the light-cull dispatch without recomputing.

    public (uint lightCount, uint tileCountX, uint tileCountY) UpdatePerFrame(
        uint frameIndex, Camera camera, Scene scene)
    {
        // Lights SSBO is renderer-owned; this just refreshes its contents from
        // the current scene. Other rendering paths call the same method.
        uint count = Renderer.UpdateLights(frameIndex, scene);

        uint tileX = (Renderer.renderExtent.Width  + Renderer.TILE_SIZE - 1) / Renderer.TILE_SIZE;
        uint tileY = (Renderer.renderExtent.Height + Renderer.TILE_SIZE - 1) / Renderer.TILE_SIZE;

        LightingFrameUBO ubo = new();
        ubo.camPos = camera != null ? new Vector4(camera.GetPosition(), 1.0f) : new Vector4(2, 2, 2, 1);
        // Used by PbrShader.slang to scale roughness into the prefiltered mip chain.
        // Renderer.prefilteredCubeMipLevels is set in CreateIblResources and never
        // changes — IBL bakes overwrite content, not metadata.
        ubo.prefilteredCubeMipLevels = Renderer.prefilteredCubeMipLevels;
        ubo.scaleIBLAmbient = EditorState.IblIntensity;
        ubo.lightCount = count;
        ubo.tileCountX = tileX;
        ubo.tileCountY = tileY;
        ubo.screenSize = new Vector2(Renderer.renderExtent.Width, Renderer.renderExtent.Height);

        // Probe cluster dims — the cluster grid is rebuilt earlier in DrawFrame
        // with the same tile counts so its dims always match the lighting tile grid.
        var grid = Renderer.reflectionProbeSystem.clusterGrid;
        ubo.probeClusterDimsX = grid.DimsX;
        ubo.probeClusterDimsY = grid.DimsY;
        ubo.probeClusterDimsZ = grid.DimsZ;
        ubo.probeMipLevels    = ReflectionProbeSystem.ProbeMipLevels;

        void* data = LightingUniformBuffers[frameIndex].mapped;
        new Span<LightingFrameUBO>(data, 1).Fill(ubo);

        return (count, tileX, tileY);
    }
}