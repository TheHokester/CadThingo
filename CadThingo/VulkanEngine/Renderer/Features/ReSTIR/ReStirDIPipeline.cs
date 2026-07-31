using System.IO;
using System.Numerics;
using System.Runtime.InteropServices;
using CadThingo.VulkanEngine.Renderer.Descriptors;
using CadThingo.VulkanEngine.Renderer.FrameGraph;
using CadThingo.VulkanEngine.Renderer.Pipelines;
using CadThingo.VulkanEngine.Renderer.Features.PathTracer;
using CadThingo.VulkanEngine.Renderer.Slang;
using Silk.NET.Core.Native;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Features.ReSTIR;

//
//  ReSTIR direct-illumination pipeline (VK_KHR_ray_tracing_pipeline + a compute sibling).
//
//  Subclass of RTPipeline reusing its SBT / descriptor-set layout / per-frame UBO /
//  CmdTraceRays dispatch, swapping the shader module for the forked ReStirDI tracer. On top it owns
//  SET 2 (the graph-shared ReSTIR working set) + a push constant, AND a second VkPipeline -- the SpatialShade
//  COMPUTE shader -- built on the SAME pipeline layout (set 0 carries ComputeBit, so the exact
//  descriptor sets the RT Trace pass uses are bindable from compute too).
//
//  Two-pass deferred ReSTIR DI:
//    * Trace (RT): build + temporally reuse the reservoir, write cur reservoir + fat G-buffer
//      (PackedHit) + sceneRadiance (everything EXCEPT analytic direct).
//    * SpatialShade (compute): spatially reuse same-frame neighbours, shade the analytic sample
//      (opaque shadow ray), add to sceneRadiance, fold into the accumulator.
//
internal sealed unsafe class ReStirDIPipeline : RTPipeline
{
    // Installed in the base's graph-shared slot (scene 0, IO 1, working 2; FeatureEnv at set 4).
    private const int SetReStir = 2;

    // Per-pixel byte strides -- MUST match the Slang StructuredBuffer element STRIDE, which is
    // std430, NOT the tight field-sum: a float3 has 16-byte alignment and rounds the struct up to a
    // multiple of 16. Strides confirmed via `slangc -reflection-json` (samplePos lands at offset 16).
    //   Reservoir { uint type,index; float3 pos@16; float3 le@32; float esPdf; uint M; float wSum,W,pHat } = 64B
    //   PackedHit { float3 worldPos; uint x7 }  -> 40B of data, rounded up to 16 (the float3 align) = 48B
    //   sceneRadiance : float4 (no float3 -> no rounding)                                                 = 16B
    // Mis-sizing these undersizes the buffer: the shader strides past the end on the bottom rows, which
    // read back zero (robust access) -> the G-buffer reports "no hit" and the deferred shade is skipped.
    private const ulong ReservoirBytes = 64;
    private const ulong PackedHitBytes = 48;
    private const ulong SceneRadBytes  = 16;

    // Push constant uploaded each frame; mirrors `struct ReStirPush` in the shaders (80B).
    [StructLayout(LayoutKind.Sequential)]
    private struct ReStirPush
    {
        public Matrix4x4 PrevViewProj;   // 64
        public uint      FrameParity;    //  4 — monotonic frame & 1
        public uint      HasHistory;     //  4 — 0 on the first frame after (re)alloc
        public uint      Pad0, Pad1;     //  8
    }

    private Buffer   _resA, _resB, _gbufA, _gbufB, _sceneRad;
    private SubAlloc _resAAlloc, _resBAlloc, _gbufAAlloc, _gbufBAlloc, _sceneRadAlloc;
    private uint     _pixelCount;

    private ReStirPush _push;
    private Matrix4x4  _prevViewProj = Matrix4x4.Identity;
    private uint       _frameCounter;

    private Pipeline _buildTemporalPipeline;   // BuildTemporal compute (reservoir build + temporal)
    private Pipeline _spatialPipeline;         // SpatialShade compute (spatial reuse + shade)

    public ReStirDIPipeline(GpuContext gpu, Renderer renderer) : base(gpu, renderer) { }

    // Public handles for the graph module to import (barrier ordering across passes + frames).
    public Buffer ReservoirA    => _resA;
    public Buffer ReservoirB    => _resB;
    public Buffer GBufferA      => _gbufA;
    public Buffer GBufferB      => _gbufB;
    public Buffer SceneRadiance => _sceneRad;

    // The working set (set 2) is graph-owned: the pipeline owns the buffers + the layout, but the
    // FrameGraph allocates + writes the descriptor set from this spec and hands it back after Compile.
    private DescriptorSet _sharedSet;   // set by the core via SetGraphSharedSet after the graph compiles
    public void SetGraphSharedSet(DescriptorSet set) => _sharedSet = set;

    /// The graph-shared working-set spec (layout + current buffer handles). Read by the module at
    /// Build time; on resize the buffers realloc first, so the rebuilt graph bakes the new handles.
    /// Binding index and type come from ReStirBindings' set-2 declarations; only the buffer each
    /// name maps to is ours to state.
    public GraphSharedSetSpec GraphSharedSpec => new(
        ShaderSets.GraphShared, DescriptorSetLayouts[SetReStir],
        ReflectedBindings(SetReStir)
            .Select(b => new GraphSharedBinding(b.Binding, b.Type, BufferFor(b.Name)))
            .ToArray());

    // Throws rather than binding a default handle if the shader grows a set-2 buffer with no C#
    // owner - a stale/null descriptor there would be a silent corruption.
    private Buffer BufferFor(string name) => name switch
    {
        "reservoirA"    => _resA,
        "reservoirB"    => _resB,
        "gbufferA"      => _gbufA,
        "gbufferB"      => _gbufB,
        "sceneRadiance" => _sceneRad,
        _ => throw new InvalidOperationException(
            $"ReStirDIPipeline: ReStirBindings declares set-{SetReStir} buffer '{name}' with no backing buffer"),
    };

    protected override ShaderCompileRequest? Program => RtProgram("ReSTIR/ReStirDI");

    // The two compute passes run on the SHARED RT pipeline layout, so they are separate programs
    // rather than a second Program override (a pipeline has exactly one).
    private static readonly ShaderCompileRequest BuildTemporalReq = new("ReSTIR/BuildTemporal", ["main"], [], []);
    private static readonly ShaderCompileRequest SpatialShadeReq  = new("ReSTIR/SpatialShade", ["main"], [], ["spvRayQueryKHR"]);

    // The IO set (accumulator/outColor) is shared with the SpatialShade compute pipeline (it folds
    // the analytic direct into the accumulator) -> needs ComputeBit. The scene set is already
    // All-stages via the registry, so it needs no override.
    protected override ShaderStageFlags OwnedSetExtraStages => ShaderStageFlags.ComputeBit;


    // ---- Working-set layout (set 2, the base's graph-shared slot) + push-constant range ---------
    protected override void CreateDescriptorSetLayouts()
    {
        base.CreateDescriptorSetLayouts();   // [scene, IO, empty(2), empty(3), FeatureEnv(4)]; IO set now Raygen|...|Compute

        // Overwrite the empty placeholder the base left in the graph-shared slot (set 2) with the
        // working set reflected from ReStirBindings. ExtraStagesFor adds ComputeBit here too: the
        // RT program reflects RT stages only, but both compute passes bind this set.
        DescriptorSetLayouts[SetReStir] = CreateReflectedSetLayout(SetReStir);

        var owned = OwnedDescriptorSetLayoutIndices;
        Array.Resize(ref owned, owned.Length + 1);
        owned[^1] = SetReStir;
        OwnedDescriptorSetLayoutIndices = owned;
    }

    // Build the RT pipeline + SBT (base), then the two ReSTIR compute pipelines (BuildTemporal +
    // SpatialShade) on the SAME shared pipeline layout (set 0/1 carry ComputeBit, set 2 bindless is
    // compute-capable). Deferred order at record time: Trace (RT) -> BuildTemporal -> SpatialShade.
    protected override void CreatePipeline()
    {
        base.CreatePipeline();
        _buildTemporalPipeline = CreateComputePipeline(BuildTemporalReq);
        _spatialPipeline       = CreateComputePipeline(SpatialShadeReq);
    }

    private Pipeline CreateComputePipeline(in ShaderCompileRequest request)
    {
        var program = Shaders.GetProgram(request);
        var entryPoint = program.Reflection.EntryPoints[0];
        var module = Gfx.CreateShaderModule(program.Spirv(0).ToArray());
        var entry  = SilkMarshal.StringToPtr(entryPoint.Name);
        var stage = new PipelineShaderStageCreateInfo
        {
            SType = StructureType.PipelineShaderStageCreateInfo,
            Stage = entryPoint.Stage, Module = module, PName = (byte*)entry,
        };
        var info = new ComputePipelineCreateInfo
        {
            SType = StructureType.ComputePipelineCreateInfo, Stage = stage, Layout = PipelineLayoutHandle,
        };
        if (Vk.CreateComputePipelines(Device, PipelineCacheHandle, 1, &info, null, out Pipeline pipeline) != Result.Success)
            throw new Exception($"ReStirDIPipeline: failed to create {request.Module} compute pipeline");
        SilkMarshal.Free(entry);
        Vk.DestroyShaderModule(Device, module, null);
        return pipeline;
    }

    protected override void CreateResources()
    {
        base.CreateResources();   // per-frame UBO
        AllocBuffers(Renderer.RenderExtent);
    }

    // ---- ReSTIR working-set buffers (extent-sized, device-local, ping-ponged) -------------------
    private void AllocBuffers(Extent2D extent)
    {
        _pixelCount = extent.Width * extent.Height;
        ulong n = _pixelCount;
        Gfx.CreateBuffer(n * ReservoirBytes, BufferUsageFlags.StorageBufferBit, MemoryPropertyFlags.DeviceLocalBit, out _resA,     out _resAAlloc);
        Gfx.CreateBuffer(n * ReservoirBytes, BufferUsageFlags.StorageBufferBit, MemoryPropertyFlags.DeviceLocalBit, out _resB,     out _resBAlloc);
        Gfx.CreateBuffer(n * PackedHitBytes, BufferUsageFlags.StorageBufferBit, MemoryPropertyFlags.DeviceLocalBit, out _gbufA,    out _gbufAAlloc);
        Gfx.CreateBuffer(n * PackedHitBytes, BufferUsageFlags.StorageBufferBit, MemoryPropertyFlags.DeviceLocalBit, out _gbufB,    out _gbufBAlloc);
        Gfx.CreateBuffer(n * SceneRadBytes,  BufferUsageFlags.StorageBufferBit, MemoryPropertyFlags.DeviceLocalBit, out _sceneRad, out _sceneRadAlloc);
        _frameCounter = 0;   // fresh buffers hold garbage -> first frame must not reuse history
    }

    private void FreeBuffers()
    {
        if (_resA.Handle     != 0) Gfx.DestroyBuffer(_resA,     _resAAlloc);
        if (_resB.Handle     != 0) Gfx.DestroyBuffer(_resB,     _resBAlloc);
        if (_gbufA.Handle    != 0) Gfx.DestroyBuffer(_gbufA,    _gbufAAlloc);
        if (_gbufB.Handle    != 0) Gfx.DestroyBuffer(_gbufB,    _gbufBAlloc);
        if (_sceneRad.Handle != 0) Gfx.DestroyBuffer(_sceneRad, _sceneRadAlloc);
        _resA = default; _resB = default; _gbufA = default; _gbufB = default; _sceneRad = default;
    }

    /// <summary>Resize path: reallocate the per-pixel working set to the new extent. Resets the
    /// history gate. Called by the core's Resize before the graph rebuild, which re-bakes the
    /// graph-owned set 2 from the fresh buffer handles (see GraphSharedSpec).</summary>
    public void ReallocReStirBuffers(Extent2D extent)
    {
        FreeBuffers();
        AllocBuffers(extent);
    }


    // ---- Per-frame state: temporal reprojection matrix + ping-pong parity -----------------------
    public override bool UpdatePerFrame(uint frameIndex, Camera camera, uint lightCount, Extent2D renderExtent)
    {
        bool reset = base.UpdatePerFrame(frameIndex, camera, lightCount, renderExtent);

        Matrix4x4 view = camera != null ? camera.GetViewMatrix() : Matrix4x4.Identity;
        Matrix4x4 proj = camera != null
            ? camera.GetProjectionMatrix((float)renderExtent.Width / renderExtent.Height, 0.1f, 100.0f)
            : Matrix4x4.Identity;
        proj.M22 *= -1f;
        Matrix4x4 viewProj = view * proj;

        _push = new ReStirPush
        {
            PrevViewProj = _prevViewProj,
            FrameParity  = _frameCounter & 1u,
            HasHistory   = _frameCounter > 0u ? 1u : 0u,
        };

        _prevViewProj = viewProj;
        _frameCounter++;
        return reset;
    }

    protected override void RecordPushConstants(CommandBuffer cmd)
    {
        // Must cover the range's FULL stage mask, not just the stage doing the work: the range is
        // shared with the compute passes, and vkCmdPushConstants requires the mask to include every
        // stage of each range it overlaps.
        var p = _push;
        Vk.CmdPushConstants(cmd, Layout, PushConstantRanges[0].StageFlags, 0, (uint)sizeof(ReStirPush), &p);
    }

    /// <summary>Records the BuildTemporal compute pass: reconstruct the surface from the G-buffer,
    /// build the unified DI reservoir (RIS) + temporal reuse, write the cur reservoir. Runs after
    /// Trace (G-buffer RAW), before SpatialShade (reservoir RAW).</summary>
    public void RecordBuildTemporal(CommandBuffer cmd, in RenderView ctx)
        => RecordComputePass(cmd, ctx, _buildTemporalPipeline);

    /// <summary>Records the SpatialShade compute pass: spatial reuse + deferred shade + accumulation.</summary>
    public void RecordSpatialShade(CommandBuffer cmd, in RenderView ctx)
        => RecordComputePass(cmd, ctx, _spatialPipeline);

    // Binds the SAME descriptor sets the Trace pass uses on the shared layout, pushes the per-frame
    // ReSTIR state, and dispatches a full-screen 8x8 grid. Both ReSTIR compute passes share this;
    // which buffers each actually touches is decided by its shader's declared bindings. The scene set
    // carries the frame-UBO arena slot, so its bind supplies the same dynamic offset the RT Trace
    // pass uses. envCube (FeatureEnv) binds at set 4 for the passes that sample the background.
    private void RecordComputePass(CommandBuffer cmd, in RenderView ctx, Pipeline pipeline)
    {
        Vk.CmdBindPipeline(cmd, PipelineBindPoint.Compute, pipeline);

        // Set 1 is an unused gap since the IO pair moved to the registry, so scene and the
        // graph-owned working set bind separately rather than as one contiguous run.
        uint frameConstants = PushFrameConstants(ctx.FrameIndex);
        var sceneSet = Registry.SceneSet(ctx.FrameIndex);  // frame UBO + lights / tlas / entityInfo / vb+ib / emissive / bindless
        Vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute, Layout, ShaderSets.Scene, 1, &sceneSet, 1, &frameConstants);

        var shared = _sharedSet;                           // reservoirs / gbuffers / sceneRadiance (graph-owned)
        Vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute, Layout, ShaderSets.GraphShared, 1, &shared, 0, null);

        var envSet = Registry.FeatureSet(FeatureEnv, ctx.FrameIndex);
        Vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute, Layout, Registry.FeatureSetIndex(FeatureEnv), 1, &envSet, 0, null);

        var ioSet = Registry.FeatureSet(FeaturePtIo, ctx.FrameIndex);
        Vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute, Layout, Registry.FeatureSetIndex(FeaturePtIo), 1, &ioSet, 0, null);

        var p = _push;
        Vk.CmdPushConstants(cmd, Layout, PushConstantRanges[0].StageFlags, 0, (uint)sizeof(ReStirPush), &p);

        Vk.CmdDispatch(cmd, (ctx.RenderExtent.Width + 7) / 8, (ctx.RenderExtent.Height + 7) / 8, 1);
    }


    // ---- Install the graph-owned working set in the base's graph-shared slot (set 2, RTPipeline.Record) --
    protected override uint ExtraSetCount => 1u;
    protected override void WriteExtraSets(DescriptorSet* dst, uint frame)
    {
        dst[0] = _sharedSet;
    }

    protected override void ReleaseGpuResources()
    {
        if (_buildTemporalPipeline.Handle != 0) { Vk.DestroyPipeline(Device, _buildTemporalPipeline, null); _buildTemporalPipeline = default; }
        if (_spatialPipeline.Handle != 0) { Vk.DestroyPipeline(Device, _spatialPipeline, null); _spatialPipeline = default; }
        FreeBuffers();
        base.ReleaseGpuResources();
    }
}