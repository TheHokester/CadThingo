using System.Numerics;
using System.Runtime.InteropServices;
using CadThingo.VulkanEngine.ImGui;
using CadThingo.VulkanEngine.Renderer.Descriptors;
using CadThingo.VulkanEngine.Renderer.FrameGraph;
using CadThingo.VulkanEngine.Renderer.Pipelines;
using CadThingo.VulkanEngine.Renderer.Features.IBL;
using CadThingo.VulkanEngine.Renderer.Slang; // ReflectionProbeSystem, ProbeGpuRecord
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Features.Forward;


//
//  Transparent forward+ pass — renders BLEND-mode materials into HDRColor with
//  src-alpha / one-minus-src-alpha blending, depth-tested LE against the
//  geometry pass's depth buffer (no depth write).
//
public sealed unsafe class TransparentPipeline : GraphicsPipeline
{
    private IIblProvider _ibl;
    private IReflectionProbeProvider _reflProbes;
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
        // params byte-for-byte so the same Renderer.PrefilteredCubeMipLevels +
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

    // Ray-queried shadows need the capability at compile time even when SOFT_SHADOWS is off:
    // the traceRayInline call sites are in the shader either way.
    protected override ShaderCompileRequest? Program =>
        new("Forward/Transparent", ["VSMain", "PSMain"], [], ["spvRayQueryKHR"]);

    protected override Format[] ColorAttachmentFormats { get; } = new[] { Format.R16G16B16A16Sfloat };

    public bool SoftShadowsEnabled { get; set; } = true;

    // Set 0 - unified scene set (registry-owned): lights, TLAS, bindless
    //         materials/textures/samplers. Frame constants ride its (0,0) dynamic slot.
    // Set 1 - graph-baked pass set: the tile-cull outputs.
    // Set 3 - FeatureIBL (registry-owned): global IBL split-sum + reflection probes.
    //         (Set 2 is the graph-shared slot this pass doesn't use - a gap in the layout.)
    private const string FeatureIbl = "FeatureIBL";

    // Graph-baked pass set: the two tile-cull outputs (the only graph resources the
    // TransparentPass reads). Both the layout and the names the graph matches against come from
    // Transparent.slang's set-1 declarations, so the TransparentPass Read binds name the shader
    // globals.
    public PassSetSpec PassSet =>
        new(ShaderSets.Pass, DescriptorSetLayouts[ShaderSets.Pass], ReflectedBindings(ShaderSets.Pass));

    // Frame constants staged by UpdatePerFrame, pushed into the constant arena
    // by Record (which runs later the same frame inside the graph).
    private TransparentFrameUBO _frameUbo;

    public TransparentPipeline(GpuContext gpu, IIblProvider ibl, IReflectionProbeProvider reflProbes) : base(gpu)
    {
        DepthAttachmentFormat = Gfx.FindDepthFormat();
        _ibl = ibl;
        _reflProbes = reflProbes;
    }

    // No owned buffers — frame constants ride the scene set's arena and per-draw state is pushed.
    // The size check is the one thing reflection cannot enforce on its own: the C# mirror of
    // DrawPC has to keep matching the shader.
    protected override void CreateResources()
    {
        uint reflected = PushConstantRanges[0].Size;
        if (reflected != (uint)sizeof(TransparentPushConstants))
            throw new Exception(
                $"TransparentPushConstants is {sizeof(TransparentPushConstants)} bytes but " +
                $"Transparent.slang reflects {reflected}");
    }

    internal readonly ref struct Attachments(ImageView hdrColor, ImageView depth)
    {
        internal readonly ImageView HdrColor = hdrColor;
        internal readonly ImageView Depth = depth;
    }

    internal void Record(CommandBuffer cmd, RenderView ctx, IReadOnlyList<TransparentDraw> transparentDraws, Attachments attachments, DescriptorSet tileSet)
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
    protected override SpecValues? Specialization =>
        new SpecValues().Set("SOFT_SHADOWS", SoftShadowsEnabled);

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
        // Assemble [scene(0), pass(1), empty(2), FeatureIBL(3)]. Scene + FeatureIBL are
        // registry-owned; the pipeline owns only the pass layout, built from Transparent.slang's
        // set-1 declarations. The deferred FrameGraph owns + writes the SETS allocated from it.
        var passLayout = CreateReflectedSetLayout(ShaderSets.Pass);
        DescriptorSetLayouts = Registry.BuildPipelineSetLayouts(passLayout, FeatureIbl);
        OwnedDescriptorSetLayoutIndices = new[] { (int)ShaderSets.Pass };
    }

    
    /// <summary>Fill the frame constants from the current camera + tile counts,
    /// staged for Record's arena push. Call once per frame from DrawFrame;
    /// tileCount / lightCount come from PbrDeferredPipeline.UpdatePerFrame so
    /// the two pipelines stay coherent.</summary>
    public void UpdatePerFrame(RenderView f, uint tileCountX, uint tileCountY)
    {
        var camera = f.Camera;
        var renderExtent = f.RenderExtent;
        var lightCount = f.LightCount;
        
        TransparentFrameUBO ubo = new();
        
        var aspect = (float)renderExtent.Width / renderExtent.Height;
        if (camera != null)
        {
            ubo.proj = camera.GetProjectionMatrix(
                aspect, 0.1f, 100.0f);
            ubo.view = camera.GetViewMatrix();
            ubo.proj.M22 *= -1;
            ubo.camPos = new Vector4(camera.GetPosition(), 1.0f);
        }
        else
        {
            ubo.view   = Matrix4x4.CreateLookAt(new Vector3(2, 2, 2), Vector3.Zero, new Vector3(0, 0, 1));
            ubo.proj   = Matrix4x4.CreatePerspectiveFieldOfView((float)(45 * Math.PI / 180),
                aspect, 0.1f, 100.0f);
            ubo.proj.M22 *= -1;
            ubo.camPos = new Vector4(2, 2, 2, 1);
        }
        ubo.lightCount = lightCount;
        ubo.tileCountX = tileCountX;
        ubo.tileCountY = tileCountY;
        ubo.screenSize = new Vector2(renderExtent.Width, renderExtent.Height);
        ubo.prefilteredCubeMipLevels = _ibl.PrefilteredCubeMipLevels;
        ubo.scaleIBLAmbient          = EditorState.IblIntensity;

        // Probe cluster dims - built once per frame by ReflectionProbeSystem.
        // The transparent pass uses the same grid as PbrDeferred so cluster
        // indices stay consistent across opaque and transparent samples.
        
        var grid = _reflProbes.ClusterGrid;
        ubo.probeClusterDimsX = grid.DimsX;
        ubo.probeClusterDimsY = grid.DimsY;
        ubo.probeClusterDimsZ = grid.DimsZ;
        ubo.probeMipLevels    = _reflProbes.ProbeMipLevels;

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