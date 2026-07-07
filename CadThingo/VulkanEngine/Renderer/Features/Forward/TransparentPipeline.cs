using System.Numerics;
using System.Runtime.InteropServices;
using CadThingo.VulkanEngine.ImGui;
using CadThingo.VulkanEngine.Renderer.Pipelines;
using CadThingo.VulkanEngine.Renderer.Features.IBL;   // ReflectionProbeSystem, ProbeGpuRecord
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Features.Forward;


//
//  Transparent forward+ pass — renders BLEND-mode materials into HDRColor with
//  src-alpha / one-minus-src-alpha blending, depth-tested LE against the
//  geometry pass's depth buffer (no depth write).
//
// Base type qualified: the dead VulkanTut `CadThingo.GraphicsPipeline` namespace would
// otherwise shadow the GraphicsPipeline base via enclosing-namespace lookup under Features.
public sealed unsafe class TransparentPipeline : Pipelines.GraphicsPipeline
{
    // Matches Transparent.slang::FrameUBO — pushed into the scene set's (0,0)
    // constant arena slot each frame. View+proj feed the VS; camPos + tile
    // state feed the FS.
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
        // params byte-for-byte so the same Renderer.Ibl.prefilteredCubeMipLevels +
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

    protected override string ShaderPath { get; } = ShaderPaths.Kernel("Forward", "Transparent");

    protected override Format[] ColorAttachmentFormats { get; } = new[] { Format.R16G16B16A16Sfloat };

    public bool SoftShadowsEnabled { get; set; } = true;

    // Set 0 - unified scene set (registry-owned): lights, TLAS, bindless
    //         materials/textures/samplers. Frame constants ride its (0,0)
    //         dynamic slot.
    // Set 1 - pass-local lighting inputs: tile-cull outputs, global IBL
    //         split-sum, reflection probes. Per-frame (tile/probe buffers are
    //         per-frame allocations). Same layout/order as PBR's set 2.
    private const int SetScene          = 0;
    private const int SetLightingInputs = 1;

    // Frame constants staged by UpdatePerFrame, pushed into the constant arena
    // by Record (which runs later the same frame inside the graph).
    private TransparentFrameUBO _frameUbo;

    public TransparentPipeline(Renderer renderer) : base(renderer)
    {
        DepthAttachmentFormat = Gfx.FindDepthFormat();
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

    internal readonly ref struct Attachments(ImageView hdrColor, ImageView depth)
    {
        internal readonly ImageView HdrColor = hdrColor;
        internal readonly ImageView Depth = depth;
    }

    internal void Record(CommandBuffer cmd, Renderer.FrameContext ctx, IReadOnlyList<TransparentDraw> transparentDraws,Attachments attachments)
    {
        BeginRendering(cmd,
            ctx.RenderExtent,
            [attachments.HdrColor],
            depthView: attachments.Depth,
            colorLoad: AttachmentLoadOp.Load,
            depthLoad: AttachmentLoadOp.Load
            );
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
        // of the UBO staged by UpdatePerFrame). Set 1 = tile cull + IBL + probes.
        var registry = Renderer.descriptorRegistry;
        uint frameConstants = registry.ConstantArena.Push(ctx.FrameIndex, _frameUbo);
        var sets = stackalloc DescriptorSet[2]
        {
            registry.SceneSet(ctx.FrameIndex),
            GetDescriptorSet(SetLightingInputs, ctx.FrameIndex),
        };
        Vk!.CmdBindDescriptorSets(cmd, PipelineBindPoint.Graphics,
            Layout, 0, 2, sets, 1, &frameConstants);

        // Bind global VB/IB once — every BLEND entity references offsets into these.
        var vb = Engine.ResourceManager.GlobalVertexBuffer;
        var ib = Engine.ResourceManager.GlobalIndexBuffer;
        ulong vbOffset = 0;
        Vk!.CmdBindVertexBuffers(cmd, 0, 1, &vb, &vbOffset);
        Vk!.CmdBindIndexBuffer(cmd, ib, 0, IndexType.Uint32);

        // One push-constant + draw per BLEND entity, in back-to-front order
        // set by DrawCullPipeline.Record.

        for (int di = 0; di < transparentDraws.Count; di++)
        {
            var d = transparentDraws[di];
            PushDrawConstants(cmd, d.Model, d.MaterialIndex);
            Vk!.CmdDrawIndexed(cmd, d.IndexCount, 1, d.FirstIndex, 0, 0);
        }

        EndRendering(cmd);
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

    // Pipeline state overrides

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


    protected override void CreateDescriptorSetLayouts()
    {
        // Set 0 is borrowed from DescriptorRegistry (never destroyed here);
        // set 1 is owned by this pipeline.
        DescriptorSetLayouts = new DescriptorSetLayout[2];
        OwnedDescriptorSetLayoutIndices = new[] { SetLightingInputs };
        DescriptorSetLayouts[SetScene] = Renderer.descriptorRegistry.SceneSetLayout;

        // Set 1: pass-local lighting inputs. Written only at Initialize /
        // Rebuild / cross-pipeline wiring, all under an idle device, so no
        // update-after-bind flags are needed.
        //   0 tileLightCount   1 tileLightIndices   (LightCulling outputs)
        //   2 irradianceCube   3 prefilteredCube    4 brdfLut   (global IBL)
        //   5 probeCubeArray   6 probes   7 probeClusterRange   8 probeIndexList
        var set1Types = new[]
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
        var set1Bindings = new DescriptorSetLayoutBinding[set1Types.Length];
        for (uint b = 0; b < set1Types.Length; b++)
        {
            set1Bindings[b] = new DescriptorSetLayoutBinding
            {
                Binding            = b,
                DescriptorType     = set1Types[b],
                DescriptorCount    = 1,
                StageFlags         = ShaderStageFlags.FragmentBit,
                PImmutableSamplers = null,
            };
        }

        fixed (DescriptorSetLayoutBinding* pSet1 = set1Bindings)
        {
            var set1LayoutInfo = new DescriptorSetLayoutCreateInfo
            {
                SType        = StructureType.DescriptorSetLayoutCreateInfo,
                BindingCount = (uint)set1Bindings.Length,
                PBindings    = pSet1,
            };
            if (Vk.CreateDescriptorSetLayout(Device, &set1LayoutInfo, null, out DescriptorSetLayouts[SetLightingInputs]) != Result.Success)
                throw new Exception("Failed to create transparent set 1 (LightingInputs) layout");
        }
    }

    protected override void CreateDescriptorSets()
    {
        DescriptorSets = new DescriptorSet[2][];

        // Set 0 — scene set is owned by DescriptorRegistry; Record binds
        // Renderer.descriptorRegistry.SceneSet(frame) directly.
        DescriptorSets[SetScene] = null;

        // Set 1 — per-frame: the tile-cull and probe buffers it points at are
        // per-frame allocations.
        var layouts = stackalloc DescriptorSetLayout[(int)Renderer.MAX_CONCURRENT_FRAMES];
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++) layouts[i] = DescriptorSetLayouts[SetLightingInputs];

        var allocInfo = new DescriptorSetAllocateInfo
        {
            SType              = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool     = Gfx.DescriptorPool,
            DescriptorSetCount = Renderer.MAX_CONCURRENT_FRAMES,
            PSetLayouts        = layouts,
        };
        DescriptorSets[SetLightingInputs] = new DescriptorSet[Renderer.MAX_CONCURRENT_FRAMES];
        fixed (DescriptorSet* pSets = DescriptorSets[SetLightingInputs])
        {
            if (Vk.AllocateDescriptorSets(Device, &allocInfo, pSets) != Result.Success)
                throw new Exception("Failed to allocate transparent lighting-inputs descriptor sets");
        }
    }

    // Only the IBL bindings are written here. Bindings 0/1 (tile buffers) point
    // at LightCullPipeline's buffers and are written by the renderer after that
    // pipeline exists; the probe quartet after the probe system exists. The
    // scene set (lights / TLAS / bindless) is registry-maintained.
    protected override void WriteDescriptors()
    {
        WriteIblDescriptors();
    }

    /// <summary>Writes bindings 0 + 1 of set 1 (per-tile light count and indices)
    /// on the per-frame lighting-inputs sets. Call once at startup after the
    /// light-cull pipeline exists — its output buffers are this pipeline's inputs.</summary>
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
            writes[0] = new() { SType = StructureType.WriteDescriptorSet, DstSet = DescriptorSets[SetLightingInputs][i], DstBinding = 0, DescriptorType = DescriptorType.StorageBuffer, DescriptorCount = 1, PBufferInfo = &tileCountInfo };
            writes[1] = new() { SType = StructureType.WriteDescriptorSet, DstSet = DescriptorSets[SetLightingInputs][i], DstBinding = 1, DescriptorType = DescriptorType.StorageBuffer, DescriptorCount = 1, PBufferInfo = &tileIdxInfo };
            Vk.UpdateDescriptorSets(Device, 2, writes, 0, null);
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

    /// <summary>Mirror of PbrDeferredPipeline.WriteProbeDescriptors — writes
    /// bindings 5/6/7/8 of set 1 on each per-frame transparent set. Call once
    /// after the probe system exists; underlying handles are stable for the
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
                SType = StructureType.WriteDescriptorSet, DstSet = DescriptorSets[SetLightingInputs][i],
                DstBinding = 5, DescriptorType = DescriptorType.CombinedImageSampler,
                DescriptorCount = 1, PImageInfo = &cubeArrayInfo,
            };
            writes[1] = new WriteDescriptorSet
            {
                SType = StructureType.WriteDescriptorSet, DstSet = DescriptorSets[SetLightingInputs][i],
                DstBinding = 6, DescriptorType = DescriptorType.StorageBuffer,
                DescriptorCount = 1, PBufferInfo = &recordsInfo,
            };
            writes[2] = new WriteDescriptorSet
            {
                SType = StructureType.WriteDescriptorSet, DstSet = DescriptorSets[SetLightingInputs][i],
                DstBinding = 7, DescriptorType = DescriptorType.StorageBuffer,
                DescriptorCount = 1, PBufferInfo = &clusterRangeInfo,
            };
            writes[3] = new WriteDescriptorSet
            {
                SType = StructureType.WriteDescriptorSet, DstSet = DescriptorSets[SetLightingInputs][i],
                DstBinding = 8, DescriptorType = DescriptorType.StorageBuffer,
                DescriptorCount = 1, PBufferInfo = &indexListInfo,
            };
            Vk.UpdateDescriptorSets(Device, 4, writes, 0, null);
        }
    }

    /// <summary>Fill the frame constants from the current camera + tile counts,
    /// staged for Record's arena push. Call once per frame from DrawFrame;
    /// tileCount / lightCount come from PbrDeferredPipeline.UpdatePerFrame so
    /// the two pipelines stay coherent.</summary>
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
        ubo.prefilteredCubeMipLevels = Renderer.Ibl.prefilteredCubeMipLevels;
        ubo.scaleIBLAmbient          = EditorState.IblIntensity;

        // Probe cluster dims — built once per frame by ReflectionProbeSystem.
        // The transparent pass uses the same grid as PbrDeferred so cluster
        // indices stay consistent across opaque and transparent samples.
        var grid = Renderer.reflectionProbeSystem.clusterGrid;
        ubo.probeClusterDimsX = grid.DimsX;
        ubo.probeClusterDimsY = grid.DimsY;
        ubo.probeClusterDimsZ = grid.DimsZ;
        ubo.probeMipLevels    = ReflectionProbeSystem.ProbeMipLevels;

        _frameUbo = ubo;
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