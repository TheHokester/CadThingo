using System.Numerics;
using System.Runtime.InteropServices;
using CadThingo.VulkanEngine.ImGui;
using CadThingo.VulkanEngine.Renderer.FrameGraph;
using CadThingo.VulkanEngine.Renderer.Pipelines;
using CadThingo.VulkanEngine.Renderer.Shaders;
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
    //         materials/textures/samplers. Frame constants ride its (0,0) dynamic slot.
    // Set 1 - graph-baked pass set: the tile-cull outputs.
    // Set 3 - FeatureIBL (registry-owned): global IBL split-sum + reflection probes.
    //         (Set 2 is the graph-shared slot this pass doesn't use - a gap in the layout.)
    private const int SetScene = 0;
    private const int SetTile  = 1;
    private const string FeatureIbl = "FeatureIBL";

    // Graph-baked pass set (set 1): the two tile-cull outputs (graph resources the
    // TransparentPass reads). Names match the TransparentPass Read binds.
    private static readonly BindingDesc[] _passBindings =
    {
        new("tileLightCount",   SetTile, 0, DescriptorType.StorageBuffer, 1, ShaderStageFlags.FragmentBit),
        new("tileLightIndices", SetTile, 1, DescriptorType.StorageBuffer, 1, ShaderStageFlags.FragmentBit),
    };

    public PassSetSpec PassSet => new(SetIndex: SetTile, DescriptorSetLayouts[SetTile], _passBindings);

    // Frame constants staged by UpdatePerFrame, pushed into the constant arena
    // by Record (which runs later the same frame inside the graph).
    private TransparentFrameUBO _frameUbo;

    public TransparentPipeline(GpuContext gpu, Renderer renderer) : base(gpu, renderer)
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

    internal void Record(CommandBuffer cmd, Renderer.FrameContext ctx, IReadOnlyList<TransparentDraw> transparentDraws, Attachments attachments, DescriptorSet tileSet)
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

        // Set 0 = scene set with the frame constants' dynamic offset (arena push of the UBO
        // staged by UpdatePerFrame). Set 1 = graph-baked tile pass set. FeatureIBL sits at its
        // own reflected index (set 3) with a gap at set 2, so it binds separately.
        uint frameConstants = Registry.ConstantArena.Push(ctx.FrameIndex, _frameUbo);
        var sets = stackalloc DescriptorSet[2]
        {
            Registry.SceneSet(ctx.FrameIndex),
            tileSet,
        };
        Vk!.CmdBindDescriptorSets(cmd, PipelineBindPoint.Graphics,
            Layout, 0, 2, sets, 1, &frameConstants);

        var iblSet = Registry.FeatureSet(FeatureIbl, ctx.FrameIndex);
        Vk!.CmdBindDescriptorSets(cmd, PipelineBindPoint.Graphics,
            Layout, Registry.FeatureSetIndex(FeatureIbl), 1, &iblSet, 0, null);

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
        // Pass set (set 1): the two tile-cull output storage buffers. The pipeline owns this
        // LAYOUT; the deferred FrameGraph owns + writes the SETS allocated from it.
        var passBindings = new DescriptorSetLayoutBinding[2];
        for (uint b = 0; b < 2; b++)
            passBindings[b] = new DescriptorSetLayoutBinding
            {
                Binding = b, DescriptorType = DescriptorType.StorageBuffer, DescriptorCount = 1,
                StageFlags = ShaderStageFlags.FragmentBit,
            };

        DescriptorSetLayout passLayout;
        fixed (DescriptorSetLayoutBinding* pPass = passBindings)
        {
            var info = new DescriptorSetLayoutCreateInfo
            {
                SType = StructureType.DescriptorSetLayoutCreateInfo,
                BindingCount = (uint)passBindings.Length,
                PBindings = pPass,
            };
            if (Vk.CreateDescriptorSetLayout(Device, &info, null, out passLayout) != Result.Success)
                throw new Exception("Failed to create transparent pass-set (set 1) layout");
        }

        // Assemble [scene(0), pass(1), empty(2), FeatureIBL(3)]; the pipeline owns only the pass layout.
        DescriptorSetLayouts = Registry.BuildPipelineSetLayouts(passLayout, FeatureIbl);
        OwnedDescriptorSetLayoutIndices = new[] { SetTile };
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