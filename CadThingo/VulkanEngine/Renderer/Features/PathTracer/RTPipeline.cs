using System.Numerics;
using System.Runtime.InteropServices;
using CadThingo.VulkanEngine.ImGui;
using CadThingo.VulkanEngine.Renderer.Pipelines;
using CadThingo.VulkanEngine.Renderer.Descriptors;
using CadThingo.VulkanEngine.Renderer.Shaders;
using Silk.NET.Core.Native;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Features.PathTracer;

//
//  Ray-tracing-PIPELINE path tracer (VK_KHR_ray_tracing_pipeline).
//
//  Additive, opt-in sibling of PTComputePipeline. Shares the SAME acceleration
//  structure, scene buffers, IBL, bindless materials, PathFrameUBO layout, and
//  accumulator/outColor storage images as the compute path tracer — the only
//  differences are the pipeline object (RT groups + SBT) and the dispatch
//  (CmdTraceRays).
//
//  Scene set (registry): TLAS, lights, ShadowEntityInfo, global VB/IB, emissive
//  tables, bindless materials/textures/samplers. The PathFrameUBO rides the
//  scene set's (0,0) constant-arena slot. This pipeline owns only set 1 (the
//  accumulator/outColor storage images); envCube rides the registry-owned FeatureEnv set (set 4).
//
//  Inherits RtPipeline, which owns the VK_KHR_ray_tracing_pipeline dispatch
//  table + SBT-layout properties (loaded in its constructor from the device).
//  Not sealed: the ReSTIR DI pipeline (Features/ReSTIR) subclasses this, overriding only
//  ShaderPath to point at its forked tracer .spv while reusing the SBT / descriptor / dispatch
//  machinery unchanged. Later ReSTIR phases extend it with reservoir buffers + reuse passes.
public unsafe class RTPipeline : RtPipeline
{
    // Matches PathTraceRT.slang / PTComputePipeline PathFrameUBO byte-for-byte.
    [StructLayout(LayoutKind.Sequential)]
    private struct PathFrameUBO
    {
        public Matrix4x4 invView;
        public Matrix4x4 invProj;
        public Vector4   camPos;
        public uint      frameIndex;
        public uint      bounceCap;
        public uint      lightCount;
        public uint      resetAccum;
        public Vector2   screenSize;
        public float     fov;
        public float     tanHalfFov;
        public float     prefilteredCubeMipLevels;
        public float     scaleIBLAmbient;
        public float     focusDistance;
        public float     aperture;
        public float     paniniDistance;
        public float     verticalCompression;
        public uint      emissiveTriCount;
        public float     totalEmissivePower;
    }

    // Set 2 is the graph-shared slot: unused by the base megakernel, installed by the ReSTIR
    // subclass for its working set. envCube comes from the registry-owned FeatureEnv set (set 4),
    // accumulator + outColor from FeaturePTIO (set 5).
    protected const string FeatureEnv  = "FeatureEnv";
    protected const string FeaturePtIo = "FeaturePTIO";

    // Every RT stage; descriptor bindings declare the union (a binding may name
    // more stages than actually read it — simpler and valid).
    private const ShaderStageFlags RtAll =
        ShaderStageFlags.RaygenBitKhr | ShaderStageFlags.MissBitKhr |
        ShaderStageFlags.ClosestHitBitKhr | ShaderStageFlags.AnyHitBitKhr;

    // SBT group order. The request's entry order drives both Reflection.EntryPoints[i] and
    // Spirv(i), so stage i lands in the group that names it below; reordering this list silently
    // rewires the SBT, hence the stage assert in CreatePipeline.
    private static readonly string[] RtEntries = ["rayGenMain", "missMain", "closestHitMain", "anyHitMain"];

    /// <summary>The RT program for <paramref name="module"/>, SER-specialized for this device. SER
    /// (HitObject + ReorderThread + Invoke raygen) is now a define on the SAME source rather than a
    /// separate _SER.slang output, so the variant is a runtime capability decision.</summary>
    protected ShaderCompileRequest RtProgram(string module) => Gfx.SerSupported
        ? new(module, RtEntries, ["USE_SER=1"], ["spvRayTracingKHR", "spvShaderInvocationReorderNV"])
        : new(module, RtEntries, [], ["spvRayTracingKHR"]);

    protected override ShaderCompileRequest? Program => RtProgram("PathTracer/PathTraceRT");

    // SBT: one buffer holding [raygen][miss][hit] regions, each padded to the
    // device's shaderGroupBaseAlignment; CmdTraceRays reads the strided regions.
    private Buffer   _sbtBuffer;
    private SubAlloc _sbtAlloc;
    private StridedDeviceAddressRegionKHR _raygenRegion;
    private StridedDeviceAddressRegionKHR _missRegion;
    private StridedDeviceAddressRegionKHR _hitRegion;
    private StridedDeviceAddressRegionKHR _callableRegion;   // unused (no callables)

    // Frame constants staged by UpdatePerFrame, pushed into the constant arena by Record.
    private PathFrameUBO _frameUbo;

    // Per-frame runtime state (mirrors the compute path).
    public uint  BounceCap     { get; set; } = 8;
    public float FocusDistance { get; set; } = 5.0f;
    public float Aperture      { get; set; } = 0.0f;

    private uint _accumSamples;
    private bool _accumDirty = true;

    public RTPipeline(GpuContext gpu, Renderer renderer) : base(gpu, renderer) { }

    public void MarkAccumulatorDirty() => _accumDirty = true;
    public uint CurrentSampleCount => _accumSamples;


    // Extra stage flags OR'd onto the owned IO set (accumulator/outColor). A subclass that shares the
    // storage images with a compute sibling (ReSTIR's SpatialShade, which folds the analytic direct
    // into the accumulator) returns ComputeBit so the set is bindable from both. Base returns 0 -> the
    // megakernel RT pipeline's IO set stays RT-stages-only.
    // Reflection cannot discover this: it only walks the RT program, so it never sees the sibling
    // compute PSOs built against this same layout.
    protected virtual ShaderStageFlags OwnedSetExtraStages => 0;

    // Only consulted for sets this pipeline reflects and owns (IO, plus ReSTIR's working set) -
    // registry-owned scene/feature sets never route through CreateReflectedSetLayout and are
    // All-stages already, so this needs no per-set gate.
    protected override ShaderStageFlags ExtraStagesFor(in BindingDesc binding) => OwnedSetExtraStages;

    protected override ShaderStageFlags ExtraPushStages => OwnedSetExtraStages;

    // Descriptor set layouts, all registry-borrowed: scene (set 0), FeatureEnv (4), FeaturePTIO (5).
    // A subclass may install its own graph-shared working layout into the set-2 gap (ReSTIR does).
    protected override void CreateDescriptorSetLayouts()
        => DescriptorSetLayouts = Registry.BuildPipelineSetLayouts(null, FeatureEnv, FeaturePtIo);


    // Pipeline build: one RT pipeline (raygen + miss + closest/any-hit group) + SBT.
    protected override void CreatePipeline()
    {
        if (KhrRtPipeline == null)
            throw new Exception("RTPipeline: VK_KHR_ray_tracing_pipeline dispatch table not loaded");

        // Slang compiles one SPIR-V module PER ENTRY POINT, so this is 4 modules rather than the
        // single multi-entry .spv the build-time route produced. Entry i is parallel to
        // RtEntries[i], which is what the group table below assumes.
        var reflectedEntries = Reflected!.Reflection.EntryPoints;
        ReadOnlySpan<ShaderStageFlags> expected =
        [
            ShaderStageFlags.RaygenBitKhr, ShaderStageFlags.MissBitKhr,
            ShaderStageFlags.ClosestHitBitKhr, ShaderStageFlags.AnyHitBitKhr,
        ];
        if (reflectedEntries.Length != expected.Length)
            throw new Exception($"RTPipeline: expected {expected.Length} entry points, reflected {reflectedEntries.Length}");

        var modules   = stackalloc ShaderModule[4];
        var entryPtrs = stackalloc nint[4];
        var stages    = stackalloc PipelineShaderStageCreateInfo[4];
        for (int i = 0; i < 4; i++)
        {
            // A silent SBT misorder is unrecoverable at runtime, so pin the contract here.
            if (reflectedEntries[i].Stage != expected[i])
                throw new Exception(
                    $"RTPipeline: entry {i} '{reflectedEntries[i].Name}' is {reflectedEntries[i].Stage}, " +
                    $"expected {expected[i]} - SBT group order would be wrong");
            modules[i]   = CreateReflectedModule(i);
            entryPtrs[i] = SilkMarshal.StringToPtr(reflectedEntries[i].Name);
            stages[i]    = Stage(expected[i], modules[i], entryPtrs[i]);
        }

        const uint UNUSED = uint.MaxValue;   // VK_SHADER_UNUSED_KHR
        var groups = stackalloc RayTracingShaderGroupCreateInfoKHR[3];
        // Group 0: raygen (general, stage 0)
        groups[0] = new() { SType = StructureType.RayTracingShaderGroupCreateInfoKhr, Type = RayTracingShaderGroupTypeKHR.GeneralKhr,
                            GeneralShader = 0, ClosestHitShader = UNUSED, AnyHitShader = UNUSED, IntersectionShader = UNUSED };
        // Group 1: miss (general, stage 1)
        groups[1] = new() { SType = StructureType.RayTracingShaderGroupCreateInfoKhr, Type = RayTracingShaderGroupTypeKHR.GeneralKhr,
                            GeneralShader = 1, ClosestHitShader = UNUSED, AnyHitShader = UNUSED, IntersectionShader = UNUSED };
        // Group 2: hit group (closest=stage 2, any=stage 3)
        groups[2] = new() { SType = StructureType.RayTracingShaderGroupCreateInfoKhr, Type = RayTracingShaderGroupTypeKHR.TrianglesHitGroupKhr,
                            GeneralShader = UNUSED, ClosestHitShader = 2, AnyHitShader = 3, IntersectionShader = UNUSED };

        var info = new RayTracingPipelineCreateInfoKHR
        {
            SType                        = StructureType.RayTracingPipelineCreateInfoKhr,
            StageCount                   = 4,
            PStages                      = stages,
            GroupCount                   = 3,
            PGroups                      = groups,
            // Iterative bounce loop in raygen + inline ray query for shadows keeps
            // this at 1; raising it is a hard perf cliff.
            MaxPipelineRayRecursionDepth = 1,
            Layout                       = PipelineLayoutHandle,
        };

        Pipeline pipeline;
        var res = KhrRtPipeline.CreateRayTracingPipelines(
            Device, default, PipelineCacheHandle, 1, &info, null, &pipeline);
        if (res != Result.Success)
            throw new Exception($"Failed to create ray tracing pipeline: {res}");
        PipelineHandle = pipeline;

        for (int i = 0; i < 4; i++)
        {
            SilkMarshal.Free(entryPtrs[i]);
            Vk.DestroyShaderModule(Device, modules[i], null);
        }

        BuildShaderBindingTable();
    }

    private PipelineShaderStageCreateInfo Stage(ShaderStageFlags stage, ShaderModule module, nint entry) => new()
    {
        SType  = StructureType.PipelineShaderStageCreateInfo,
        Stage  = stage,
        Module = module,
        PName  = (byte*)entry,
    };

    private static uint AlignUp(uint v, uint a) => (v + a - 1) & ~(a - 1);

    private ulong BufferDeviceAddress(Buffer buf)
    {
        var info = new BufferDeviceAddressInfo { SType = StructureType.BufferDeviceAddressInfo, Buffer = buf };
        return Vk.GetBufferDeviceAddress(Device, &info);
    }

    // Packs the 3 group handles into [raygen][miss][hit] regions per the SBT
    // alignment rules (each region base-aligned, handles handle-aligned; raygen
    // size must equal its stride).
    private void BuildShaderBindingTable()
    {
        uint handleSize    = ShaderGroupHandleSize;
        uint handleAligned = AlignUp(handleSize, ShaderGroupHandleAlignment);
        uint baseAlign     = ShaderGroupBaseAlignment;

        const uint raygenCount = 1, missCount = 1, hitCount = 1;
        const uint groupCount  = raygenCount + missCount + hitCount;   // 3

        _raygenRegion.Stride = AlignUp(handleAligned, baseAlign);
        _raygenRegion.Size   = _raygenRegion.Stride;                   // raygen: size == stride
        _missRegion.Stride   = handleAligned;
        _missRegion.Size     = AlignUp(missCount * handleAligned, baseAlign);
        _hitRegion.Stride    = handleAligned;
        _hitRegion.Size      = AlignUp(hitCount * handleAligned, baseAlign);

        ulong sbtSize = _raygenRegion.Size + _missRegion.Size + _hitRegion.Size;

        Gfx.CreateBuffer(sbtSize,
            BufferUsageFlags.ShaderBindingTableBitKhr | BufferUsageFlags.ShaderDeviceAddressBit | BufferUsageFlags.TransferSrcBit,
            MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit,
            out _sbtBuffer, out _sbtAlloc);

        byte* sbt     = (byte*)Gfx.Allocator.GetMapped(_sbtAlloc);
        var   handles = new byte[groupCount * handleSize];
        fixed (byte* pHandles = handles)
        {
            if (KhrRtPipeline!.GetRayTracingShaderGroupHandles(
                    Device, PipelineHandle, 0, groupCount, (nuint)handles.Length, pHandles) != Result.Success)
                throw new Exception("Failed to fetch RT shader group handles");

            // raygen → 0, miss → after raygen region, hit → after miss region.
            System.Buffer.MemoryCopy(pHandles + 0u * handleSize, sbt,                                                       handleSize, handleSize);
            System.Buffer.MemoryCopy(pHandles + 1u * handleSize, sbt + _raygenRegion.Size,                                  handleSize, handleSize);
            System.Buffer.MemoryCopy(pHandles + 2u * handleSize, sbt + _raygenRegion.Size + _missRegion.Size,              handleSize, handleSize);
        }

        ulong baseAddr = BufferDeviceAddress(_sbtBuffer);
        _raygenRegion.DeviceAddress = baseAddr;
        _missRegion.DeviceAddress   = baseAddr + _raygenRegion.Size;
        _hitRegion.DeviceAddress    = baseAddr + _raygenRegion.Size + _missRegion.Size;
        _callableRegion             = default;
    }


    // No descriptor sets to allocate or write: every set this pipeline binds (scene, FeatureEnv,
    // FeaturePTIO) is registry-owned, and ReSTIR's set-2 working set is graph-owned.


    public virtual bool UpdatePerFrame(uint frameIndex, Camera camera, uint lightCount, Extent2D renderExtent)
    {
        bool reset = _accumDirty;
        if (reset) { _accumSamples = 0; _accumDirty = false; } else { _accumSamples++; }

        Matrix4x4 view = camera != null ? camera.GetViewMatrix() : Matrix4x4.Identity;
        Matrix4x4 proj = camera != null
            ? camera.GetProjectionMatrix((float)renderExtent.Width / renderExtent.Height, 0.1f, 100.0f)
            : Matrix4x4.Identity;
        proj.M22 *= -1;
        Matrix4x4.Invert(view, out var invView);
        Matrix4x4.Invert(proj, out var invProj);

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
            paniniDistance           = 1.0f,
            verticalCompression      = 0.0f,
            emissiveTriCount         = Renderer.EmissiveTriangleCount,
            totalEmissivePower       = Renderer.TotalEmissivePower,
        };
        return reset;
    }


    /// <summary>Records the CmdTraceRays dispatch. Caller must have run
    /// UpdatePerFrame, written the storage-image + IBL descriptors, and
    /// transitioned the accumulator/outColor images to GENERAL beforehand.</summary>
    // Extra descriptor sets a subclass binds in the graph-shared slot right after the base sets
    // (0-1), i.e. starting at set 2. ReSTIR installs its working set (reservoirs + G-buffer) there.
    // Base pipeline binds none.
    protected virtual uint ExtraSetCount => 0u;
    protected virtual void WriteExtraSets(DescriptorSet* dst, uint frame) { }

    // Push-constant upload hook, called by Record after the descriptor sets are bound and before
    // CmdTraceRays. Base pipeline declares no push constants; ReSTIR pushes its per-frame state
    // (prevViewProj + ping-pong parity) here. The subclass must also declare the matching range in
    // CreateDescriptorSetLayouts (PushConstantRanges) so the pipeline layout includes it.
    protected virtual void RecordPushConstants(CommandBuffer cmd) { }

    public void Record(CommandBuffer cmd, in RenderView ctx)
    {
        Vk.CmdBindPipeline(cmd, PipelineBindPoint.RayTracingKhr, PipelineHandle);

        // Four separate binds, because the occupied sets are no longer contiguous: set 1 is now an
        // unused gap (the IO pair moved to the registry), and the subclass working set sits at 2.
        uint frameConstants = Registry.ConstantArena.Push(ctx.FrameIndex, _frameUbo);
        var sceneSet = Registry.SceneSet(ctx.FrameIndex);
        Vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.RayTracingKhr, Layout, ShaderSets.Scene, 1, &sceneSet, 1, &frameConstants);

        if (ExtraSetCount > 0u)
        {
            var extra = stackalloc DescriptorSet[(int)ExtraSetCount];
            WriteExtraSets(extra, ctx.FrameIndex);
            Vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.RayTracingKhr, Layout, ShaderSets.GraphShared, ExtraSetCount, extra, 0, null);
        }

        var envSet = Registry.FeatureSet(FeatureEnv, ctx.FrameIndex);
        Vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.RayTracingKhr, Layout, Registry.FeatureSetIndex(FeatureEnv), 1, &envSet, 0, null);

        var ioSet = Registry.FeatureSet(FeaturePtIo, ctx.FrameIndex);
        Vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.RayTracingKhr, Layout, Registry.FeatureSetIndex(FeaturePtIo), 1, &ioSet, 0, null);

        RecordPushConstants(cmd);

        var rg = _raygenRegion; var ms = _missRegion; var ht = _hitRegion; var cl = _callableRegion;
        KhrRtPipeline!.CmdTraceRays(cmd, &rg, &ms, &ht, &cl,
            ctx.RenderExtent.Width, ctx.RenderExtent.Height, 1);
    }

    // Pushes the staged PathFrameUBO into the per-frame constant arena and returns its dynamic
    // offset. ReSTIR's compute passes (built on the shared layout) call this so they can bind the
    // scene set with the same frame constants the RT Trace pass uses. Keeps PathFrameUBO private.
    protected uint PushFrameConstants(uint frameIndex) =>
        Registry.ConstantArena.Push(frameIndex, _frameUbo);


    public override void Dispose()
    {
        if (_sbtBuffer.Handle != 0) Gfx.DestroyBuffer(_sbtBuffer, _sbtAlloc);
        base.Dispose();
    }
}
