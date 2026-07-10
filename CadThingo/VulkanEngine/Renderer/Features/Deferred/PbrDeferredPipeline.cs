using System.Numerics;
using System.Runtime.InteropServices;
using CadThingo.VulkanEngine.ImGui;
using CadThingo.VulkanEngine.Renderer.FrameGraph;
using CadThingo.VulkanEngine.Renderer.Pipelines;
using CadThingo.VulkanEngine.Renderer.Shaders;
using CadThingo.VulkanEngine.Renderer.Features.Forward;
using CadThingo.VulkanEngine.Renderer.Features.IBL;   // ReflectionProbeSystem, ProbeGpuRecord
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Features.Deferred;

//  PBR deferred lighting pass — fullscreen triangle, samples G-buffer +
//  per-tile light list, optional ray-queried shadows.
// Base type qualified: the dead VulkanTut `CadThingo.GraphicsPipeline` namespace would
// otherwise shadow the GraphicsPipeline base via enclosing-namespace lookup under Features.
public sealed unsafe class PbrDeferredPipeline : Pipelines.GraphicsPipeline
{
    // Matches PBR.slang's LightingFrameUBO - pushed into the scene set's (0,0)
    // constant arena slot each frame.
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

    protected override string ShaderPath { get; } = ShaderPaths.Kernel("Deferred", "PBR");

    // Lighting writes linear HDR scene-referred color; tone-map + gamma run in
    // the separate TonemapPipeline pass that consumes this attachment.
    protected override Format[] ColorAttachmentFormats { get; } = new[] { Format.R16G16B16A16Sfloat };

    // Set 0 — unified scene set (registry-owned): lights, TLAS, shadow
    //         entity-info, global vb/ib, bindless materials/textures/samplers.
    //         Per-frame constants ride its (0,0) dynamic slot.
    // Set 1 — shared G-buffer samplers (one allocation reused every frame).
    // Set 2 — pass-local lighting inputs: tile-cull outputs, global IBL
    //         split-sum, reflection probes. Per-frame (tile/probe buffers are
    //         per-frame allocations).
    private const int SetScene          = 0;
    private const int SetGBuffer        = 1;
    private const int SetLightingInputs = 2;

    // Graph-baked pass set (set 1): the five g-buffer transients, filled by the deferred
    // FrameGraph. Names match the LightingPass Read binds; the sampler is immutable in the
    // layout (see CreateDescriptorSetLayouts), so only the views are written.
    private static readonly BindingDesc[] _gBufferBindings =
    {
        new("gPosition", SetGBuffer, 0, DescriptorType.CombinedImageSampler, 1, ShaderStageFlags.FragmentBit),
        new("gNormal",   SetGBuffer, 1, DescriptorType.CombinedImageSampler, 1, ShaderStageFlags.FragmentBit),
        new("gAlbedo",   SetGBuffer, 2, DescriptorType.CombinedImageSampler, 1, ShaderStageFlags.FragmentBit),
        new("gMaterial", SetGBuffer, 3, DescriptorType.CombinedImageSampler, 1, ShaderStageFlags.FragmentBit),
        new("gEmissive", SetGBuffer, 4, DescriptorType.CombinedImageSampler, 1, ShaderStageFlags.FragmentBit),
    };

    public PassSetSpec PassSet => new(SetIndex: SetGBuffer, DescriptorSetLayouts[SetGBuffer], _gBufferBindings);

    // Frame constants staged by UpdatePerFrame, pushed into the constant arena
    // by Record (which runs later the same frame inside the graph).
    private LightingFrameUBO _frameUbo;

    /// <summary>True = wire the PCSS-style soft-shadow specialization constant on,
    /// pulled into the fragment shader as <c>constant_id 0</c>. Read at each pipeline
    /// build; set then call <see cref="PipelineBase.Rebuild"/> to apply a change.</summary>
    public bool SoftShadowsEnabled { get; set; } = true;

    public PbrDeferredPipeline(Renderer renderer) : base(renderer) { }

    internal void Record(CommandBuffer cmd, in Renderer.FrameContext ctx, ImageView HdrTarget, DescriptorSet gBufferSet)
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

        // Set 0 = scene set with the frame constants' dynamic offset (arena push
        // of the UBO staged by UpdatePerFrame). Set 1 = shared g-buffer samplers.
        // Set 2 = tile cull + IBL + probe inputs. No push constants.
        var registry = Renderer.descriptorRegistry;
        uint frameConstants = registry.ConstantArena.Push(ctx.FrameIndex, _frameUbo);
        var sets = stackalloc DescriptorSet[3]
        {
            registry.SceneSet(ctx.FrameIndex),
            gBufferSet,                               // graph-baked (set 1)
            GetDescriptorSet(SetLightingInputs, ctx.FrameIndex),
        };
        Vk!.CmdBindDescriptorSets(cmd, PipelineBindPoint.Graphics,
            Layout, 0, 3, sets, 1, &frameConstants);

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
        // Set 0 is borrowed from DescriptorRegistry (never destroyed here);
        // sets 1 and 2 are owned by this pipeline.
        DescriptorSetLayouts = new DescriptorSetLayout[3];
        OwnedDescriptorSetLayoutIndices = new[] { SetGBuffer, SetLightingInputs };
        DescriptorSetLayouts[SetScene] = Renderer.descriptorRegistry.SceneSetLayout;

        // Set 1: G-Buffer inputs. Five combined-image-samplers reading the geometry pass's
        // g-buffer transients. The g-buffer sampler is baked in as an IMMUTABLE sampler so the
        // deferred FrameGraph (which owns + writes this set - see PassSet) only writes the views,
        // no sampler plumbing and no update-after-bind. The pipeline owns the LAYOUT; the graph
        // owns the SETS allocated from it.
        Sampler gSampler = Renderer.gBufferSampler;
        var set1Bindings = new DescriptorSetLayoutBinding[5];
        for (uint b = 0; b < 5; b++)
        {
            set1Bindings[b] = new DescriptorSetLayoutBinding
            {
                Binding            = b,
                DescriptorType     = DescriptorType.CombinedImageSampler,
                DescriptorCount    = 1,
                StageFlags         = ShaderStageFlags.FragmentBit,
                PImmutableSamplers = &gSampler,
            };
        }

        // Set 2: pass-local lighting inputs. Written only at Initialize /
        // Rebuild / cross-pipeline wiring, all under an idle device, so no
        // update-after-bind flags are needed.
        //   0 tileLightCount   1 tileLightIndices   (LightCulling outputs)
        //   2 irradianceCube   3 prefilteredCube    4 brdfLut   (global IBL)
        //   5 probeCubeArray   6 probes   7 probeClusterRange   8 probeIndexList
        var set2Types = new[]
        {
            DescriptorType.StorageBuffer,
            DescriptorType.StorageBuffer,
            DescriptorType.CombinedImageSampler,
            DescriptorType.CombinedImageSampler,
            DescriptorType.CombinedImageSampler,
            DescriptorType.CombinedImageSampler,
            DescriptorType.StorageBuffer,
            DescriptorType.StorageBuffer,
            DescriptorType.StorageBuffer,
        };
        var set2Bindings = new DescriptorSetLayoutBinding[set2Types.Length];
        for (uint b = 0; b < set2Types.Length; b++)
        {
            set2Bindings[b] = new DescriptorSetLayoutBinding
            {
                Binding            = b,
                DescriptorType     = set2Types[b],
                DescriptorCount    = 1,
                StageFlags         = ShaderStageFlags.FragmentBit,
                PImmutableSamplers = null,
            };
        }

        fixed (DescriptorSetLayoutBinding* pSet1 = set1Bindings)
        fixed (DescriptorSetLayoutBinding* pSet2 = set2Bindings)
        {
            // No update-after-bind: the graph writes this set once per Compile (init/resize),
            // both under device-idle, and only the views change (sampler is immutable).
            DescriptorSetLayoutCreateInfo set1LayoutInfo = new()
            {
                SType        = StructureType.DescriptorSetLayoutCreateInfo,
                BindingCount = (uint)set1Bindings.Length,
                PBindings    = pSet1,
            };
            if (Vk.CreateDescriptorSetLayout(Device, &set1LayoutInfo, null, out DescriptorSetLayouts[SetGBuffer]) != Result.Success)
                throw new Exception("Failed to create PBR set 1 (GBuffer) descriptor set layout");

            DescriptorSetLayoutCreateInfo set2LayoutInfo = new()
            {
                SType        = StructureType.DescriptorSetLayoutCreateInfo,
                BindingCount = (uint)set2Bindings.Length,
                PBindings    = pSet2,
            };
            if (Vk.CreateDescriptorSetLayout(Device, &set2LayoutInfo, null, out DescriptorSetLayouts[SetLightingInputs]) != Result.Success)
                throw new Exception("Failed to create PBR set 2 (LightingInputs) descriptor set layout");
        }
    }

    protected override void CreateDescriptorSets()
    {
        DescriptorSets = new DescriptorSet[3][];

        // Set 0 - scene set is owned by DescriptorRegistry; Record binds
        // Renderer.descriptorRegistry.SceneSet(frame) directly.
        DescriptorSets[SetScene] = null;

        // Set 1 - g-buffer set is owned by the deferred FrameGraph (graph-baked, allocated from
        // this pipeline's layout). Record binds the set the graph hands it.
        DescriptorSets[SetGBuffer] = null;

        // Set 2 - per-frame: the tile-cull and probe buffers it points at are
        // per-frame allocations.
        var inputLayouts = stackalloc DescriptorSetLayout[(int)Renderer.MAX_CONCURRENT_FRAMES];
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
            inputLayouts[i] = DescriptorSetLayouts[SetLightingInputs];

        DescriptorSetAllocateInfo inputsAlloc = new()
        {
            SType              = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool     = Gfx.DescriptorPool,
            DescriptorSetCount = Renderer.MAX_CONCURRENT_FRAMES,
            PSetLayouts        = inputLayouts,
        };
        DescriptorSets[SetLightingInputs] = new DescriptorSet[Renderer.MAX_CONCURRENT_FRAMES];
        fixed (DescriptorSet* pSets = DescriptorSets[SetLightingInputs])
        {
            if (Vk.AllocateDescriptorSets(Device, &inputsAlloc, pSets) != Result.Success)
                throw new Exception("Failed to allocate PBR lighting-inputs descriptor sets");
        }
    }

    // Descriptor writes
    // The scene set (lights, TLAS, shadow-alpha buffers, bindless) is registry-
    // maintained; only the pipeline-owned sets are written here, split into
    // phases so cross-pipeline deps can be wired post-Initialize:
    //   - WriteDescriptors (auto from Initialize): IBL bindings of set 2.
    //   - WriteTileBufferDescriptors(lightCull):   bindings 0,1 of set 2.
    //   - WriteProbeDescriptors():                 bindings 5-8 of set 2.
    // Set 1 (the g-buffer) is graph-baked: the deferred FrameGraph allocates it from this
    // pipeline's layout and writes the transient views (initial build + every resize), so no
    // g-buffer write lives here.

    protected override void WriteDescriptors()
    {
        WriteIblDescriptors();
    }

    /// <summary>Writes bindings 2/3/4 of set 2 (irradiance cube, prefiltered cube,
    /// BRDF LUT) on every per-frame lighting-inputs set. The underlying VkImage
    /// handles don't change when content is rebaked, so this only needs to be
    /// called when the set is first allocated — and again if the renderer ever
    /// reallocates the IBL images.</summary>
    public void WriteIblDescriptors()
    {
        var imageInfos = stackalloc DescriptorImageInfo[3]
        {
            new() { ImageView = Renderer.Ibl.irradianceCubeView,  Sampler = Renderer.Ibl.iblCubeSampler, ImageLayout = ImageLayout.ShaderReadOnlyOptimal },
            new() { ImageView = Renderer.Ibl.prefilteredCubeView, Sampler = Renderer.Ibl.iblCubeSampler, ImageLayout = ImageLayout.ShaderReadOnlyOptimal },
            new() { ImageView = Renderer.Ibl.brdfLutView,         Sampler = Renderer.Ibl.iblLutSampler,  ImageLayout = ImageLayout.ShaderReadOnlyOptimal },
        };

        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            var writes = stackalloc WriteDescriptorSet[3];
            for (uint b = 0; b < 3; b++)
            {
                writes[b] = new WriteDescriptorSet
                {
                    SType           = StructureType.WriteDescriptorSet,
                    DstSet          = DescriptorSets[SetLightingInputs][i],
                    DstBinding      = 2 + b,
                    DstArrayElement = 0,
                    DescriptorType  = DescriptorType.CombinedImageSampler,
                    DescriptorCount = 1,
                    PImageInfo      = &imageInfos[b],
                };
            }
            Vk.UpdateDescriptorSets(Device, 3, writes, 0, null);
        }
    }

    /// <summary>Writes bindings 5/6/7/8 of set 2 (probe cube array + probe records
    /// SSBO + cluster range SSBO + probe index list SSBO) on each per-frame
    /// lighting-inputs set. Called once after the probe system is initialized;
    /// the underlying VkImage / VkBuffer handles are stable for the renderer's
    /// lifetime so per-frame rewrites aren't needed.</summary>
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
                DstSet = DescriptorSets[SetLightingInputs][i],
                DstBinding = 5, DstArrayElement = 0,
                DescriptorType = DescriptorType.CombinedImageSampler,
                DescriptorCount = 1, PImageInfo = &cubeArrayInfo,
            };
            writes[1] = new WriteDescriptorSet
            {
                SType = StructureType.WriteDescriptorSet,
                DstSet = DescriptorSets[SetLightingInputs][i],
                DstBinding = 6, DstArrayElement = 0,
                DescriptorType = DescriptorType.StorageBuffer,
                DescriptorCount = 1, PBufferInfo = &recordsInfo,
            };
            writes[2] = new WriteDescriptorSet
            {
                SType = StructureType.WriteDescriptorSet,
                DstSet = DescriptorSets[SetLightingInputs][i],
                DstBinding = 7, DstArrayElement = 0,
                DescriptorType = DescriptorType.StorageBuffer,
                DescriptorCount = 1, PBufferInfo = &clusterRangeInfo,
            };
            writes[3] = new WriteDescriptorSet
            {
                SType = StructureType.WriteDescriptorSet,
                DstSet = DescriptorSets[SetLightingInputs][i],
                DstBinding = 8, DstArrayElement = 0,
                DescriptorType = DescriptorType.StorageBuffer,
                DescriptorCount = 1, PBufferInfo = &indexListInfo,
            };
            Vk.UpdateDescriptorSets(Device, 4, writes, 0, null);
        }
    }

    /// <summary>Writes bindings 0 + 1 of set 2 (per-tile light count and indices)
    /// on the per-frame lighting-inputs sets. Called once after the light-cull
    /// pipeline is initialized — its output buffers are this pipeline's inputs.</summary>
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
                DstSet          = DescriptorSets[SetLightingInputs][i],
                DstBinding      = 0,
                DstArrayElement = 0,
                DescriptorType  = DescriptorType.StorageBuffer,
                DescriptorCount = 1,
                PBufferInfo     = &tileCountInfo,
            };
            writes[1] = new WriteDescriptorSet
            {
                SType           = StructureType.WriteDescriptorSet,
                DstSet          = DescriptorSets[SetLightingInputs][i],
                DstBinding      = 1,
                DstArrayElement = 0,
                DescriptorType  = DescriptorType.StorageBuffer,
                DescriptorCount = 1,
                PBufferInfo     = &tileIdxInfo,
            };
            Vk.UpdateDescriptorSets(Device, 2, writes, 0, null);
        }
    }

    // Per-frame upload
    // Walks scene lights into the per-frame Light SSBO and stages the frame
    // constants for Record's arena push. Returns (lightCount, tileX, tileY) so
    // the renderer can drive the light-cull dispatch without recomputing.

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
        // Used by PBR.slang to scale roughness into the prefiltered mip chain.
        // Renderer.Ibl.prefilteredCubeMipLevels is set when IblSystem is constructed
        // and never changes - IBL bakes overwrite content, not metadata.
        ubo.prefilteredCubeMipLevels = Renderer.Ibl.prefilteredCubeMipLevels;
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

        _frameUbo = ubo;

        return (count, tileX, tileY);
    }
}