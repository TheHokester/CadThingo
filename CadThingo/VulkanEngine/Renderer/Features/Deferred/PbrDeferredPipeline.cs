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

    // Set 0 - unified scene set (registry-owned): lights, TLAS, shadow
    //         entity-info, global vb/ib, bindless materials/textures/samplers.
    //         Per-frame constants ride its (0,0) dynamic slot.
    // Set 1 - graph-baked pass set: the five g-buffer transients + the two tile-cull
    //         outputs (all graph resources the LightingPass reads).
    // Set 2 - FeatureIBL (registry-owned): global IBL split-sum + reflection probes.
    private const int SetScene   = 0;
    private const int SetGBuffer = 1;
    private const string FeatureIbl = "FeatureIBL";

    // Graph-baked pass set (set 1): five g-buffer transients (immutable-sampler CIS) plus the
    // two tile-cull output buffers. Names match the LightingPass Read binds; the graph fills the
    // set, so only the views/buffers are written (sampler is immutable in the layout).
    private static readonly BindingDesc[] _passBindings =
    {
        new("gPosition",        SetGBuffer, 0, DescriptorType.CombinedImageSampler, 1, ShaderStageFlags.FragmentBit),
        new("gNormal",          SetGBuffer, 1, DescriptorType.CombinedImageSampler, 1, ShaderStageFlags.FragmentBit),
        new("gAlbedo",          SetGBuffer, 2, DescriptorType.CombinedImageSampler, 1, ShaderStageFlags.FragmentBit),
        new("gMaterial",        SetGBuffer, 3, DescriptorType.CombinedImageSampler, 1, ShaderStageFlags.FragmentBit),
        new("gEmissive",        SetGBuffer, 4, DescriptorType.CombinedImageSampler, 1, ShaderStageFlags.FragmentBit),
        new("tileLightCount",   SetGBuffer, 5, DescriptorType.StorageBuffer,        1, ShaderStageFlags.FragmentBit),
        new("tileLightIndices", SetGBuffer, 6, DescriptorType.StorageBuffer,        1, ShaderStageFlags.FragmentBit),
    };

    public PassSetSpec PassSet => new(SetIndex: SetGBuffer, DescriptorSetLayouts[SetGBuffer], _passBindings);

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

        // Set 0 = scene set with the frame constants' dynamic offset (arena push of the UBO
        // staged by UpdatePerFrame). Set 1 = graph-baked g-buffer + tile pass set. Set 2 =
        // FeatureIBL (registry-owned). No push constants.
        var registry = Renderer.descriptorRegistry;
        uint frameConstants = registry.ConstantArena.Push(ctx.FrameIndex, _frameUbo);
        var sets = stackalloc DescriptorSet[3]
        {
            registry.SceneSet(ctx.FrameIndex),
            gBufferSet,                               // graph-baked (set 1)
            registry.FeatureSet(FeatureIbl, ctx.FrameIndex),
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
        // Pass set (set 1): five g-buffer combined-image-samplers with the g-buffer sampler baked
        // in as an IMMUTABLE sampler (the graph writes only the views, no sampler plumbing, no
        // update-after-bind), plus the two tile-cull output storage buffers. The pipeline owns
        // this LAYOUT; the deferred FrameGraph owns + writes the SETS allocated from it.
        Sampler gSampler = Renderer.gBufferSampler;
        var passBindings = new DescriptorSetLayoutBinding[7];
        for (uint b = 0; b < 5; b++)
            passBindings[b] = new DescriptorSetLayoutBinding
            {
                Binding = b, DescriptorType = DescriptorType.CombinedImageSampler, DescriptorCount = 1,
                StageFlags = ShaderStageFlags.FragmentBit, PImmutableSamplers = &gSampler,
            };
        for (uint b = 5; b < 7; b++)
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
                throw new Exception("Failed to create PBR pass-set (set 1) descriptor set layout");
        }

        // Assemble [scene(0), pass(1), FeatureIBL(2)]. Scene + FeatureIBL are registry-owned; the
        // pipeline owns only the pass layout it just built.
        DescriptorSetLayouts = Renderer.descriptorRegistry.BuildPipelineSetLayouts(passLayout, FeatureIbl);
        OwnedDescriptorSetLayoutIndices = new[] { SetGBuffer };
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