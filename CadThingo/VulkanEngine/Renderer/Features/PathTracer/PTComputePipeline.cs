using System.Numerics;
using System.Runtime.InteropServices;
using CadThingo.VulkanEngine.ImGui;
using CadThingo.VulkanEngine.Renderer.Pipelines;
using Silk.NET.Core.Native;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Features.PathTracer;

//
//  Path-tracer compute pipeline (ray query + progressive accumulation).
//
//  Owns:    4 PSOs (one per camera mode) baked via spec constant id=3;
//           sets 1 (storage-image IO) and 2 (IBL).
//  Scene set (registry): TLAS, lights, ShadowEntityInfo, global VB/IB,
//           emissive tables, bindless materials/textures/samplers. The
//           PathFrameUBO rides the scene set's (0,0) constant-arena slot.
//  Externally supplied: accumulator + outColor storage image views — the
//           wiring lives in the renderer (storage image creation + GENERAL
//           layout transitions before/after the dispatch).
//
//  Inherits from PipelineBase (not ComputePipeline) because the base's
//  CreatePipeline is sealed and has no spec-constant hook — we need full
//  control to build one PSO per camera mode at init.
public sealed unsafe class PTComputePipeline : PipelineBase, IPathTracerCamera
{
    // Matches PTCompute.slang::PathFrameUBO byte-for-byte.
    [StructLayout(LayoutKind.Sequential)]
    private struct PathFrameUBO
    {
        public Matrix4x4 invView;            // 64B
        public Matrix4x4 invProj;            // 64B
        public Vector4   camPos;             // 16B
        public uint      frameIndex;         //  4B - sample index for accumulator + RNG seed
        public uint      bounceCap;          //  4B - clamped below MAX_BOUNCES spec const
        public uint      lightCount;
        public uint      resetAccum;         //  4B - 1 = overwrite accumulator this dispatch
        public Vector2   screenSize;         //  8B
        public float     fov;                //  4B - radians, used by Panini + Fisheye
        public float     tanHalfFov;         //  4B - used by Pinhole + ThinLens
        public float     prefilteredCubeMipLevels;
        public float     scaleIBLAmbient;
        public float     focusDistance;
        public float     aperture;
        public float     paniniDistance;
        public float     verticalCompression;
        public uint      emissiveTriCount;    // emissive area-light triangles in the alias table
        public float     totalEmissivePower;  // sum( area·luminance(Le)) - alias-table normaliser
    }

    public enum CameraMode : uint
    {
        Pinhole      = 0,
        ThinLens     = 1,
        Panini       = 2,
        Fisheye      = 3,
        Orthographic = 4,   // not yet implemented in PTUtils.slang - falls back to Pinhole PSO
    }

    public override PipelineBindPoint BindPoint => PipelineBindPoint.Compute;

    private const int SetScene = 0;
    private const int SetIO    = 1;
    private const string FeatureEnv = "FeatureEnv";   // set 4 (registry-owned): envCube

    private static readonly string ShaderPath = ShaderPaths.Kernel("PathTracer", "PTCompute");

    /// <summary>Multi-PSO pattern: one pipeline per camera mode, mode baked via spec
    ///constant id=3. Mode switch at Record time = `CmdBindPipeline` only, no
    /// rebuild. Indices match CameraMode values 0..3 (Orthographic re-uses 0).</summary>
    private Pipeline[] _modePipelines = new Pipeline[4];

    //  Compile-time tunables (baked into spec constants at Initialize)
    // Change after Initialize requires Dispose + rebuild.
    public bool EnableNee         { get; init; } = true;
    public bool RussianRoulette   { get; init; } = true;
    public uint MaxBouncesHardCap { get; init; } = 8;

    //  Per-frame runtime state (staged into the constant arena each frame)
    public CameraMode Mode                { get; set; } = CameraMode.Pinhole;
    public uint       BounceCap           { get; set; } = 8;
    public float      Aperture            { get; set; } = 0.0f;
    public float      FocusDistance       { get; set; } = 5.0f;
    public float      PaniniDistance      { get; set; } = 1.0f;
    public float      VerticalCompression { get; set; } = 0.0f;
    public float      IblIntensity        { get; set; } = 1.0f;

    // Frame constants staged by UpdatePerFrame, pushed into the constant arena
    // by Record (which runs later the same frame).
    private PathFrameUBO _frameUbo;

    // Progressive-accumulation bookkeeping. Renderer flips _accumDirty on
    // camera move / scene edit / extent change via MarkAccumulatorDirty();
    // UpdatePerFrame consumes it and emits resetAccum=1 on the next dispatch.
    private uint _accumSamples;
    private bool _accumDirty = true;

    public PTComputePipeline(Renderer renderer) : base(renderer) { }

    /// <summary>Force the next dispatch to overwrite (not add to) the
    /// accumulator. Call from input / scene-edit handlers.</summary>
    public void MarkAccumulatorDirty() => _accumDirty = true;

    public uint CurrentSampleCount => _accumSamples;


    // Descriptor set layouts
    protected override void CreateDescriptorSetLayouts()
    {
        // Set 1: accumulator + outColor storage images (pipeline-owned).
        var set1 = stackalloc DescriptorSetLayoutBinding[2];
        for (uint b = 0; b < 2; b++)
            set1[b] = new() { Binding = b, DescriptorType = DescriptorType.StorageImage, DescriptorCount = 1, StageFlags = ShaderStageFlags.ComputeBit };
        CreateLayout(set1, 2, out var ioLayout);

        // [scene(0), IO(1), empty(2), empty(3), FeatureEnv(4)]. envCube is registry-owned on
        // FeatureEnv; the graph-shared and FeatureIBL slots are unused gaps for the tracer.
        DescriptorSetLayouts            = Renderer.descriptorRegistry.BuildPipelineSetLayouts(ioLayout, FeatureEnv);
        OwnedDescriptorSetLayoutIndices = new[] { SetIO };
    }

    private void CreateLayout(DescriptorSetLayoutBinding* bindings, uint count, out DescriptorSetLayout layout)
    {
        DescriptorSetLayoutCreateInfo info = new()
        {
            SType        = StructureType.DescriptorSetLayoutCreateInfo,
            BindingCount = count,
            PBindings    = bindings,
        };
        if (Vk.CreateDescriptorSetLayout(Device, &info, null, out layout) != Result.Success)
            throw new Exception("Failed to create pathtracer descriptor set layout");
    }


    // Pipeline build: one PSO per camera mode
    protected override void CreatePipeline()
    {
        byte[] code     = File.ReadAllBytes(ShaderPath);
        var    module   = Gfx.CreateShaderModule(code);
        var    entryPtr = SilkMarshal.StringToPtr("main");

        // 4 spec entries packed into a 16B blob:
        //   id 0 = ENABLE_NEE        (bool → uint)
        //   id 1 = RUSSIAN_ROULETTE  (bool → uint)
        //   id 2 = MAX_BOUNCES       (uint)
        //   id 3 = MODE              (uint) — patched per pipeline below
        var specEntries = stackalloc SpecializationMapEntry[4];
        specEntries[0] = new() { ConstantID = 0, Offset = 0,  Size = sizeof(uint) };
        specEntries[1] = new() { ConstantID = 1, Offset = 4,  Size = sizeof(uint) };
        specEntries[2] = new() { ConstantID = 2, Offset = 8,  Size = sizeof(uint) };
        specEntries[3] = new() { ConstantID = 3, Offset = 12, Size = sizeof(uint) };

        var specData = stackalloc uint[4];
        specData[0] = EnableNee       ? 1u : 0u;
        specData[1] = RussianRoulette ? 1u : 0u;
        specData[2] = MaxBouncesHardCap;
        // specData[3] patched per-mode in the loop.

        var specInfo = new SpecializationInfo
        {
            MapEntryCount = 4,
            PMapEntries   = specEntries,
            DataSize      = (UIntPtr)(4u * sizeof(uint)),
            PData         = specData,
        };

        for (uint mode = 0; mode < 4; mode++)
        {
            specData[3] = mode;

            var stage = new PipelineShaderStageCreateInfo
            {
                SType               = StructureType.PipelineShaderStageCreateInfo,
                Stage               = ShaderStageFlags.ComputeBit,
                Module              = module,
                PName               = (byte*)entryPtr,
                PSpecializationInfo = &specInfo,
            };

            var info = new ComputePipelineCreateInfo
            {
                SType  = StructureType.ComputePipelineCreateInfo,
                Stage  = stage,
                Layout = PipelineLayoutHandle,
            };

            if (Vk.CreateComputePipelines(Device, PipelineCacheHandle, 1, &info, null, out _modePipelines[mode]) != Result.Success)
                throw new Exception($"Failed to create pathtracer compute pipeline (mode {mode})");
        }

        // PipelineHandle is the "default" the base tracks for disposal — point
        // it at mode 0 so base.Dispose() destroys exactly one PSO via that path
        // and we destroy modes 1-3 ourselves.
        PipelineHandle = _modePipelines[0];

        SilkMarshal.Free(entryPtr);
        Vk.DestroyShaderModule(Device, module, null);
    }


    // Descriptor set allocation
    protected override void CreateDescriptorSets()
    {
        DescriptorSets = new DescriptorSet[2][];

        // Set 0 - scene set is owned by DescriptorRegistry; Record binds
        // Renderer.descriptorRegistry.SceneSet(frame) directly.
        DescriptorSets[SetScene] = null;

        // Set 1 - single shared (both frames dispatch into the same images).
        var ioLayout = DescriptorSetLayouts[SetIO];
        DescriptorSetAllocateInfo alloc1 = new()
        {
            SType              = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool     = Gfx.DescriptorPool,
            DescriptorSetCount = 1,
            PSetLayouts        = &ioLayout,
        };
        DescriptorSets[SetIO] = new DescriptorSet[1];
        fixed (DescriptorSet* p = DescriptorSets[SetIO])
            if (Vk.AllocateDescriptorSets(Device, &alloc1, p) != Result.Success)
                throw new Exception("Failed to allocate pathtracer set 1");
    }

   

    /// <summary>Set 1 bindings 0/1: accumulator + outColor storage images.
    /// Both VkImages must be in ImageLayout.General before Record runs. Call
    /// after the renderer creates the images and again on render-extent
    /// resize.</summary>
    public void WriteStorageImageDescriptors(ImageView accumView, ImageView outColorView)
    {
        DescriptorImageInfo accumInfo = new() { ImageView = accumView,    ImageLayout = ImageLayout.General };
        DescriptorImageInfo outInfo   = new() { ImageView = outColorView, ImageLayout = ImageLayout.General };
        var writes = stackalloc WriteDescriptorSet[2];
        writes[0] = new() { SType = StructureType.WriteDescriptorSet, DstSet = DescriptorSets[SetIO][0], DstBinding = 0, DescriptorType = DescriptorType.StorageImage, DescriptorCount = 1, PImageInfo = &accumInfo };
        writes[1] = new() { SType = StructureType.WriteDescriptorSet, DstSet = DescriptorSets[SetIO][0], DstBinding = 1, DescriptorType = DescriptorType.StorageImage, DescriptorCount = 1, PImageInfo = &outInfo };
        Vk.UpdateDescriptorSets(Device, 2, writes, 0, null);

        // Image identity changed (resize), so any in-progress accumulation is
        // invalid by construction — drop the sample count.
        MarkAccumulatorDirty();
    }

    // Per-frame constants
    /// <summary>Stages the PathFrameUBO for Record's arena push. Returns true
    /// when this dispatch will reset the accumulator (camera move / scene edit /
    /// extent change since the last sample).</summary>
    public bool UpdatePerFrame(uint frameIndex, Camera camera, uint lightCount, Extent2D renderExtent)
    {
        bool reset = _accumDirty;
        if (reset)
        {
            _accumSamples = 0;
            _accumDirty   = false;
        }
        else
        {
            _accumSamples++;
        }

        Matrix4x4 view = camera != null ? camera.GetViewMatrix() : Matrix4x4.Identity;
        Matrix4x4 proj = camera != null
            ? camera.GetProjectionMatrix((float)renderExtent.Width / renderExtent.Height, 0.1f, 100.0f)
            : Matrix4x4.Identity;
        // Y-flip for Vulkan NDC, matching the convention used everywhere else
        // in the renderer.
        proj.M22 *= -1;

        Matrix4x4.Invert(view, out var invView);
        Matrix4x4.Invert(proj, out var invProj);

        // FOV comes from the camera - the single source of truth shared with the
        // raster paths (GetProjectionMatrix), so switching into PT doesn't change
        // the framing. Every projection mode (pinhole / thin-lens / panini /
        // fisheye) derives its angle from this vertical FOV.
        float fovDeg     = camera != null ? camera.Fov : 60.0f;
        float fovRad     = fovDeg * (float)(Math.PI / 180.0);
        float tanHalfFov = MathF.Tan(fovRad * 0.5f);

        _frameUbo = new PathFrameUBO
        {
            invView                  = invView,
            invProj                  = invProj,
            camPos                   = camera != null ? new Vector4(camera.GetPosition(), 1.0f) : new Vector4(2, 2, 2, 1),
            frameIndex               = _accumSamples,
            bounceCap                = BounceCap,
            lightCount               = lightCount,
            resetAccum               = reset ? 1u : 0u,
            screenSize               = new Vector2(renderExtent.Width, renderExtent.Height),
            fov                      = fovRad,
            tanHalfFov               = tanHalfFov,
            prefilteredCubeMipLevels = Renderer.Ibl.prefilteredCubeMipLevels,
            scaleIBLAmbient          = EditorState.IblIntensity,
            focusDistance            = FocusDistance,
            aperture                 = Aperture,
            paniniDistance           = PaniniDistance,
            verticalCompression      = VerticalCompression,
            emissiveTriCount         = Renderer.EmissiveTriangleCount,
            totalEmissivePower       = Renderer.TotalEmissivePower,
        };
        return reset;
    }


    // Record
    /// <summary>Records the dispatch. The caller is responsible for:
    ///   - Having called UpdatePerFrame this frame.
    ///   - Having written the storage-image + IBL descriptors at least once
    ///     before the first Record.
    ///   - Transitioning the accumulator and outColor images to GENERAL
    ///     before this call (and to ShaderReadOnly afterwards if Tonemap
    ///     samples outColor as a CombinedImageSampler).</summary>
    public void Record(CommandBuffer cmd, in Renderer.FrameContext ctx)
    {
        // Orthographic falls back to Pinhole until the helper lands in PTUtils.
        int modeIdx = (int)Mode;
        if (modeIdx >= _modePipelines.Length) modeIdx = 0;

        Vk.CmdBindPipeline(cmd, PipelineBindPoint.Compute, _modePipelines[modeIdx]);

        // Scene(0) + IO(1) with the frame constants' dynamic offset, then FeatureEnv at its own
        // reflected index (set 4); the intervening slots are gaps, never bound.
        var registry = Renderer.descriptorRegistry;
        uint frameConstants = registry.ConstantArena.Push(ctx.FrameIndex, _frameUbo);
        var sets = stackalloc DescriptorSet[2]
        {
            registry.SceneSet(ctx.FrameIndex),
            DescriptorSets[SetIO][0],
        };
        Vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute, Layout, 0, 2, sets, 1, &frameConstants);

        var envSet = registry.FeatureSet(FeatureEnv, ctx.FrameIndex);
        Vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute, Layout, registry.FeatureSetIndex(FeatureEnv), 1, &envSet, 0, null);

        // 8×8 workgroup matches [numthreads(8,8,1)] in PTCompute.slang.
        uint gx = (ctx.RenderExtent.Width  + 7u) / 8u;
        uint gy = (ctx.RenderExtent.Height + 7u) / 8u;
        Vk.CmdDispatch(cmd, gx, gy, 1);
    }


    public override void Dispose()
    {
        // Mode 0 is aliased to PipelineHandle — base.Dispose destroys it.
        // Modes 1-3 are ours to clean up.
        for (int i = 1; i < _modePipelines.Length; i++)
        {
            if (_modePipelines[i].Handle != 0)
                Vk.DestroyPipeline(Device, _modePipelines[i], null);
        }
        base.Dispose();
    }
}