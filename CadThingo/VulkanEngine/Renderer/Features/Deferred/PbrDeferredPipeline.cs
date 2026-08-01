using System.Numerics;
using System.Runtime.InteropServices;
using CadThingo.VulkanEngine.ImGui;
using CadThingo.VulkanEngine.Renderer.Descriptors;
using CadThingo.VulkanEngine.Renderer.FrameGraph;
using CadThingo.VulkanEngine.Renderer.Pipelines;
using CadThingo.VulkanEngine.Renderer.Features.Forward;
using CadThingo.VulkanEngine.Renderer.Features.IBL;
using CadThingo.VulkanEngine.Renderer.Slang; // ReflectionProbeSystem, ProbeGpuRecord
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Features.Deferred;

//  PBR deferred lighting pass — fullscreen triangle, samples G-buffer +
//  per-tile light list, optional ray-queried shadows.
public sealed unsafe class PbrDeferredPipeline : GraphicsPipeline
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

    // Ray-queried shadows need the capability at compile time even when SOFT_SHADOWS is off:
    // the traceRayInline call sites are in the shader either way.
    protected override ShaderCompileRequest? Program =>
        new("Deferred/PBR", ["VSMain", "PSMain"], [], ["spvRayQueryKHR"]);

    // Lighting writes linear HDR scene-referred color; tone-map + gamma run in
    // the separate TonemapPipeline pass that consumes this attachment.
    protected override Format[] ColorAttachmentFormats { get; } = new[] { Format.R16G16B16A16Sfloat };

    // Set 0 - unified scene set (registry-owned): lights, TLAS, shadow
    //         entity-info, global vb/ib, bindless materials/textures/samplers.
    //         Per-frame constants ride its (0,0) dynamic slot.
    // Set 1 - graph-baked pass set: the five g-buffer transients + the two tile-cull
    //         outputs (all graph resources the LightingPass reads).
    // Set 3 - FeatureIBL (registry-owned): global IBL split-sum + reflection probes.
   
    private const string FeatureIbl = "FeatureIBL";

    // Graph-baked pass set: five g-buffer transients (immutable-sampler CIS) plus the two
    // tile-cull output buffers. Both the layout and the names the graph matches against come from
    // PBR.slang's set-1 declarations, so the LightingPass Read binds name the shader globals.
    public PassSetSpec PassSet =>
        new(ShaderSets.Pass, DescriptorSetLayouts[ShaderSets.Pass], ReflectedBindings(ShaderSets.Pass));

    // Frame constants staged by UpdatePerFrame, pushed into the constant arena
    // by Record (which runs later the same frame inside the graph).
    private LightingFrameUBO _frameUbo;

    /// <summary>True = wire the PCSS-style soft-shadow path on via the SOFT_SHADOWS spec
    /// constant. Read at each pipeline build; set then call
    /// <see cref="PipelineBase.Rebuild"/> to apply a change.</summary>
    public bool SoftShadowsEnabled { get; set; } = true;

    public PbrDeferredPipeline(GpuContext gpu, Renderer renderer) : base(gpu, renderer) { }

    internal void Record(CommandBuffer cmd, in RenderView ctx, ImageView HdrTarget, DescriptorSet gBufferSet)
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

        // Set 0 = scene set with the frame constants' dynamic offset (arena push of the UBO
        // staged by UpdatePerFrame). Set 1 = graph-baked g-buffer + tile pass set. FeatureIBL
        // sits at its own reflected index (set 3) with a gap at set 2, so it binds separately.
        uint frameConstants = Registry.ConstantArena.Push(ctx.FrameIndex, _frameUbo);
        var sets = stackalloc DescriptorSet[2]
        {
            Registry.SceneSet(ctx.FrameIndex),
            gBufferSet,                               // graph-baked (set 1)
        };
        Vk!.CmdBindDescriptorSets(cmd, PipelineBindPoint.Graphics,
            Layout, ShaderSets.Scene, 2, sets, 1, &frameConstants);

        var iblSet = Registry.FeatureSet(FeatureIbl, ctx.FrameIndex);
        Vk!.CmdBindDescriptorSets(cmd, PipelineBindPoint.Graphics,
            Layout, Registry.FeatureSetIndex(FeatureIbl), 1, &iblSet, 0, null);

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

    // The g-buffer sampler is baked into every set-1 image binding as an IMMUTABLE sampler, so the
    // graph writes only the views - no sampler plumbing, no update-after-bind. The tile buffers
    // are storage buffers and take none.
    protected override Sampler? ImmutableSamplerFor(in BindingDesc binding)
        => binding.Type == DescriptorType.CombinedImageSampler ? Renderer.gBufferSampler : null;

    protected override SpecValues? Specialization =>
        new SpecValues().Set("SOFT_SHADOWS", SoftShadowsEnabled);

    protected override void CreateDescriptorSetLayouts()
    {
        // Assemble [scene(0), pass(1), empty(2), FeatureIBL(3)]. Scene + FeatureIBL are
        // registry-owned; the pipeline owns only the pass layout, built from PBR.slang's set-1
        // declarations.
        var passLayout = CreateReflectedSetLayout(ShaderSets.Pass);
        DescriptorSetLayouts = Registry.BuildPipelineSetLayouts(passLayout, FeatureIbl);
        OwnedDescriptorSetLayoutIndices = new[] { (int)ShaderSets.Pass };
    }

    // Per-frame upload
    // Stages the frame constants for Record's arena push. Takes the light count the draw loop's
    // extraction already produced - packing the light SSBO is not this pipeline's job. Returns
    // (tileX, tileY) so the caller can drive the light-cull dispatch without recomputing.

    public (uint tileCountX, uint tileCountY) UpdatePerFrame(
        RenderView f)
    {
        var camera = f.Camera;
        var renderExtent = f.RenderExtent;
        var lightCount = f.LightCount;
        
        uint tileX = (renderExtent.Width  + RenderConfig.TILE_SIZE - 1) / RenderConfig.TILE_SIZE;
        uint tileY = (renderExtent.Height + RenderConfig.TILE_SIZE - 1) / RenderConfig.TILE_SIZE;

        LightingFrameUBO ubo = new();
        ubo.camPos = camera != null ? new Vector4(camera.GetPosition(), 1.0f) : new Vector4(2, 2, 2, 1);
        // Used by PBR.slang to scale roughness into the prefiltered mip chain.
        // Read from the IBL provider every frame rather than cached: it is fixed today (bakes
        // overwrite content, not metadata) but a stale copy would be silent if that ever changed.
        ubo.prefilteredCubeMipLevels = Renderer.PrefilteredCubeMipLevels;
        ubo.scaleIBLAmbient = EditorState.IblIntensity;
        ubo.lightCount = lightCount;
        ubo.tileCountX = tileX;
        ubo.tileCountY = tileY;
        ubo.screenSize = new Vector2(renderExtent.Width, renderExtent.Height);

        // Probe cluster dims — the cluster grid is rebuilt earlier in DrawFrame
        // with the same tile counts so its dims always match the lighting tile grid.
        var grid = Renderer.reflectionProbeSystem.ClusterGrid;
        ubo.probeClusterDimsX = grid.DimsX;
        ubo.probeClusterDimsY = grid.DimsY;
        ubo.probeClusterDimsZ = grid.DimsZ;
        ubo.probeMipLevels    = Renderer.reflectionProbeSystem.ProbeMipLevels;

        _frameUbo = ubo;

        return (tileX, tileY);
    }
}