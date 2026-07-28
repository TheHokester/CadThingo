using System.Numerics;
using System.Runtime.InteropServices;
using CadThingo.VulkanEngine.ImGui;
using CadThingo.VulkanEngine.Renderer.Pipelines;
using CadThingo.VulkanEngine.Renderer.Shaders;
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

    private const string FeatureEnv  = "FeatureEnv";    // set 4 (registry-owned): envCube
    private const string FeaturePtIo = "FeaturePTIO";   // set 5 (registry-owned): accumulator + outColor

    protected override ShaderCompileRequest? Program =>
        new("PathTracer/PTCompute", ["main"], [], ["spvRayQueryKHR"]);

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

    public PTComputePipeline(GpuContext gpu, Renderer renderer) : base( gpu, renderer) { }

    /// <summary>Force the next dispatch to overwrite (not add to) the
    /// accumulator. Call from input / scene-edit handlers.</summary>
    public void MarkAccumulatorDirty() => _accumDirty = true;

    public uint CurrentSampleCount => _accumSamples;


    // Descriptor set layouts
    protected override void CreateDescriptorSetLayouts()
    {
        // Nothing pipeline-owned: scene(0), FeatureEnv(4) and FeaturePTIO(5) are all registry-owned,
        // and the slots between them are unused gaps. The accumulator / out-color pair used to be a
        // pass set built and written here; it is the same two images for every tracer, so it moved
        // to the registry.
        DescriptorSetLayouts = Registry.BuildPipelineSetLayouts(null, FeatureEnv, FeaturePtIo);
    }


    // The camera mode the PSO currently being built bakes in. The build loop patches this between
    // pipelines; every other spec value is constant across the four.
    private uint _specMode;

    protected override SpecValues? Specialization => new SpecValues()
        .Set("ENABLE_NEE",       EnableNee)
        .Set("RUSSIAN_ROULETTE", RussianRoulette)
        .Set("MAX_BOUNCES",      MaxBouncesHardCap)
        .Set("MODE",             _specMode);

    // Pipeline build: one PSO per camera mode, all from one cached SPIR-V - the modes differ only
    // in the MODE spec constant.
    protected override void CreatePipeline()
    {
        var entryPoint = Reflected!.Reflection.EntryPoints[0];
        var module     = CreateReflectedModule(0);
        var entryPtr   = SilkMarshal.StringToPtr(entryPoint.Name);

        var specEntries = stackalloc SpecializationMapEntry[SpecScratchEntries];
        var specData    = stackalloc byte[SpecScratchBytes];

        for (uint mode = 0; mode < 4; mode++)
        {
            _specMode = mode;
            int filled = FillStageSpecialization(0, specEntries, specData, out uint dataSize);
            var specInfo = new SpecializationInfo
            {
                MapEntryCount = (uint)filled,
                PMapEntries   = specEntries,
                DataSize      = (UIntPtr)dataSize,
                PData         = specData,
            };

            var stage = new PipelineShaderStageCreateInfo
            {
                SType               = StructureType.PipelineShaderStageCreateInfo,
                Stage               = entryPoint.Stage,
                Module              = module,
                PName               = (byte*)entryPtr,
                PSpecializationInfo = filled > 0 ? &specInfo : null,
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
    public void Record(CommandBuffer cmd, in RenderView ctx)
    {
        // Orthographic falls back to Pinhole until the helper lands in PTUtils.
        int modeIdx = (int)Mode;
        if (modeIdx >= _modePipelines.Length) modeIdx = 0;

        Vk.CmdBindPipeline(cmd, PipelineBindPoint.Compute, _modePipelines[modeIdx]);

        // Scene(0) carrying the frame constants' dynamic offset, then FeatureEnv and FeaturePTIO at
        // their own reflected indices; the slots between are gaps, never bound.
        uint frameConstants = Registry.ConstantArena.Push(ctx.FrameIndex, _frameUbo);
        var sceneSet = Registry.SceneSet(ctx.FrameIndex);
        Vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute, Layout, 0, 1, &sceneSet, 1, &frameConstants);

        var envSet = Registry.FeatureSet(FeatureEnv, ctx.FrameIndex);
        Vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute, Layout, Registry.FeatureSetIndex(FeatureEnv), 1, &envSet, 0, null);

        var ioSet = Registry.FeatureSet(FeaturePtIo, ctx.FrameIndex);
        Vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute, Layout, Registry.FeatureSetIndex(FeaturePtIo), 1, &ioSet, 0, null);

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