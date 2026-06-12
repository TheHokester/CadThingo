using System.Numerics;
using System.Runtime.InteropServices;
using CadThingo.VulkanEngine.ImGui;
using CadThingo.VulkanEngine.Renderer.Pipelines;
using Silk.NET.Core.Native;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Features.WavefrontPathTracer;

//
//  Wavefront path-tracer pipeline (graph-resident, dispatchIndirect, SoA).
//
//  Owns:    PathFrameUBO per frame; 9 compute PSOs (Generate x4 by camera-mode
//           spec id 0, plus Extend / Shade / Connect / Finalize / PrepareArgs);
//           the descriptor-set-4 SoA working-set buffers (device-local, single-
//           buffered, sized to the render extent); sets 0/1/3/4 layouts.
//  Borrows: sets 0-3 contents -- TLAS, lights SSBO, ShadowEntityInfo, global
//           VB/IB, IBL cubes, bindless materials/textures -- written verbatim
//           from the same renderer-owned sources the megakernel uses.
//
//  Inherits PipelineBase (not ComputePipeline) for full control over the multi-
//  PSO build, exactly like PTComputePipeline. All 9 PSOs share one pipeline
//  layout (5 sets + a single 16-byte ComputeBit push-constant range that covers
//  both WavefrontPush (workers) and ArgsPc (PrepareArgs)).
public sealed unsafe class WavefrontPTPipeline : PipelineBase, IPathTracerCamera
{
    // Matches WavefrontBindings / PTUtils PathFrameUBO byte-for-byte (same as
    // PTComputePipeline's private copy).
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

    // Matches WavefrontBindings::WavefrontPush. maxBounces lets Shade's fused tail guard the
    // extend-args[bounce+1] write on the last bounce (the args buffer has no slot past it).
    [StructLayout(LayoutKind.Sequential)]
    private struct WavefrontPush { public uint bounce; public uint srcParity; public uint argsClass; public uint maxBounces; }

    public override PipelineBindPoint BindPoint => PipelineBindPoint.Compute;

    private const int SetFrame     = 0;
    private const int SetGeom      = 1;
    private const int SetBindless  = 2;
    private const int SetIbl       = 3;
    private const int SetWavefront = 4;

    // ---- Material-sorted shading (P3): C routing classes (must match WavefrontBindings.WF_SHADE_CLASSES) -
    public const uint ShadeClasses = 4u;

    // ---- Counter slot indices (must match WavefrontBindings.slang) -----------
    // shadeCount occupies SHADE_COUNT_0 .. SHADE_COUNT_0+ShadeClasses-1 (slots 1..4), one per class;
    // SHADOW_COUNT/NEXT_RAY_COUNT/COMPLETED_WG sit past them.
    public const uint RAY_COUNT = 0, SHADE_COUNT_0 = 1, SHADOW_COUNT = 5, NEXT_RAY_COUNT = 6;

    // THE bounce-count knob. Single source of truth: the module unrolls this many bounce bodies,
    // the dispatchArgs buffer + readback are sized from it, so changing it here keeps everything
    // consistent (no drift / OOB). Empirically ~all paths terminate by bounce 2-3 (RR + escapes),
    // so 4 trades almost nothing in quality for fewer near-empty deep-bounce passes. Bump it for
    // glass-heavy scenes that need more enter/exit refraction events (multilayered glass truncates
    // at MaxBounces). Structural (the graph is unrolled at build), so a change needs a graph rebuild
    // -- already covered by WavefrontPTCore's ctor / Resize.
    public const uint MaxBounces = 4u;

    // ---- dispatchArgs byte layout (PER BOUNCE) -------------------------------
    // Each VkDispatchIndirectCommand is 12 bytes (x,y,z). Each bounce b owns a contiguous block of
    // STAGES_PER_BOUNCE commands: [extend | shade[0..C-1] | connect] (P3: the shade stage fans out
    // to one indirect command per material class). The buffer RETAINS every bounce's launch size, so
    // a RenderDoc/Nsight capture (or the readback) shows the indirect args shrinking down the chain
    // -- the visible proof of compaction. Offsets MUST mirror WavefrontBindings.wf*ArgsOffset.
    public  const uint  StagesPerBounce  = 2u + ShadeClasses;   // extend + C shade + connect
    private const uint  ArgStride        = 12u;      // sizeof(VkDispatchIndirectCommand)
    public static uint  ExtendArgsOffset (uint bounce)            => (bounce * StagesPerBounce + 0u)             * ArgStride;
    public static uint  ShadeArgsOffset  (uint bounce, uint cls)  => (bounce * StagesPerBounce + 1u + cls)       * ArgStride;
    public static uint  ConnectArgsOffset(uint bounce)            => (bounce * StagesPerBounce + 1u + ShadeClasses) * ArgStride;
    private const ulong DispatchArgsBytes = (ulong)MaxBounces * StagesPerBounce * ArgStride;  // 4*6*12 = 288
    private const ulong CountersBytes     = 32;   // 8 uints; indices reach COMPLETED_WG=7

    private static readonly string GenerateSpv = ShaderPaths.Kernel("WavefrontPathTracer", "Generate");
    private static readonly string ExtendSpv   = ShaderPaths.Kernel("WavefrontPathTracer", "Extend");
    private static readonly string ShadeSpv    = ShaderPaths.Kernel("WavefrontPathTracer", "Shade");
    private static readonly string ConnectSpv  = ShaderPaths.Kernel("WavefrontPathTracer", "Connect");
    private static readonly string FinalizeSpv = ShaderPaths.Kernel("WavefrontPathTracer", "Finalize");

    // Generate is baked per camera mode (spec id 0); Shade is baked per material class (P3b
    // SHADING_CLASS spec id 0 -> lobe stripping); Extend/Connect/Finalize are single PSOs. (No
    // PrepareArgs PSO anymore -- arg generation is fused onto the producers' tails.)
    private readonly Pipeline[] _generatePsos = new Pipeline[4];
    private readonly Pipeline[] _shadePsos    = new Pipeline[ShadeClasses];
    private Pipeline _extendPso, _connectPso, _finalizePso;

    // Camera / DoF controls (IPathTracerCamera). Generate bakes the same four
    // CameraMode PSOs as the megakernel (spec id 0); these feed PathFrameUBO so
    // the same lens math drives both tracers. Mode picks the PSO in RecordGenerate.
    public PTComputePipeline.CameraMode Mode { get; set; } = PTComputePipeline.CameraMode.Pinhole;
    public float Aperture            { get; set; } = 0.0f;
    public float FocusDistance       { get; set; } = 5.0f;
    public float PaniniDistance      { get; set; } = 1.0f;
    public float VerticalCompression { get; set; } = 0.0f;

    // Runtime bounce cap is unused by the wavefront chain: bounce count is
    // STRUCTURAL (the module unrolls MaxBounces bodies at graph-build), not a
    // per-frame UBO clamp the way the megakernel uses it. Kept only so the UBO
    // field has a defined value; the panel surfaces MaxBounces read-only instead.
    public uint BounceCap { get; set; } = MaxBounces;

    // ---- Set 4 SoA buffers (binding index = array index; see WavefrontBindings) -
    private const int B_PS_RAY_ORIGIN = 0, B_PS_RAY_DIR = 1, B_PS_THROUGHPUT = 2, B_PS_RADIANCE = 3,
                      B_PS_RNG = 4, B_PS_SIGMA_A = 5, B_HIT_REC_PRIM = 6, B_HIT_T = 7, B_HIT_BARY = 8,
                      B_RAY_QUEUE0 = 9, B_RAY_QUEUE1 = 10, B_SHADE_QUEUE = 11,
                      B_SHADOW_PATH = 12, B_SHADOW_ORIGIN = 13, B_SHADOW_DIR = 14, B_SHADOW_LE = 15,
                      B_COUNTERS = 16, B_DISPATCH_ARGS = 17;
    private const int Set4BindingCount = 18;

    private readonly Buffer[]   _set4Buf   = new Buffer[Set4BindingCount];
    private readonly SubAlloc[] _set4Alloc = new SubAlloc[Set4BindingCount];

    // Handles imported by the module for barrier derivation.
    public Buffer PsRayOrigin   => _set4Buf[B_PS_RAY_ORIGIN];
    public Buffer PsRayDir      => _set4Buf[B_PS_RAY_DIR];
    public Buffer PsThroughput  => _set4Buf[B_PS_THROUGHPUT];
    public Buffer PsRadiance    => _set4Buf[B_PS_RADIANCE];
    public Buffer PsRng         => _set4Buf[B_PS_RNG];
    public Buffer PsSigmaA      => _set4Buf[B_PS_SIGMA_A];   // P2.6 Beer-Lambert medium absorption
    public Buffer HitRecPrim    => _set4Buf[B_HIT_REC_PRIM];
    public Buffer HitT          => _set4Buf[B_HIT_T];
    public Buffer HitBary       => _set4Buf[B_HIT_BARY];
    public Buffer RayQueue0     => _set4Buf[B_RAY_QUEUE0];
    public Buffer RayQueue1     => _set4Buf[B_RAY_QUEUE1];
    public Buffer ShadeQueue    => _set4Buf[B_SHADE_QUEUE];
    public Buffer ShadowPath    => _set4Buf[B_SHADOW_PATH];
    public Buffer ShadowOrigin  => _set4Buf[B_SHADOW_ORIGIN];
    public Buffer ShadowDir     => _set4Buf[B_SHADOW_DIR];
    public Buffer ShadowLe      => _set4Buf[B_SHADOW_LE];
    public Buffer Counters      => _set4Buf[B_COUNTERS];
    public Buffer DispatchArgsBuffer => _set4Buf[B_DISPATCH_ARGS];

    private UboBuffer[] _frameUbos = new UboBuffer[Renderer.MAX_CONCURRENT_FRAMES];

    // ---- TEMP/debug: per-bounce indirect-args readback (compaction visualization) -----------
    // dispatchArgs is copied here (host-visible, mapped) each frame so the Stats panel can show
    // the per-bounce launch sizes in-app, no external capture needed. Single-buffered + ~a frame
    // stale + best-effort (a debug counter, not a sync-critical path). To DROP the feature, delete
    // this field, CreateResources' staging alloc, RecordArgsReadback, ReadDispatchArgs, the Dispose
    // line, the WavefrontPTCore.Render call, and the Renderer/StatsPanel hooks.
    private UboBuffer _argsReadback;

    // Progressive-accumulation bookkeeping (mirrors PTComputePipeline).
    private uint _accumSamples;
    private bool _accumDirty = true;
    public void MarkAccumulatorDirty() => _accumDirty = true;
    public uint CurrentSampleCount => _accumSamples;

    public WavefrontPTPipeline(Renderer renderer) : base(renderer)
    {
        // One 16-byte ComputeBit range for WavefrontPush (the workers; Generate/Finalize don't
        // read it and Slang dead-strips it from their modules).
        PushConstantRanges = new[]
        {
            new PushConstantRange
            {
                StageFlags = ShaderStageFlags.ComputeBit,
                Offset     = 0,
                Size       = (uint)sizeof(WavefrontPush),   // 16 (4 uints)
            }
        };
    }


    // ---- Descriptor-set layouts (sets 0-3 verbatim from PTComputePipeline) ----
    protected override void CreateDescriptorSetLayouts()
    {
        DescriptorSetLayouts            = new DescriptorSetLayout[5];
        OwnedDescriptorSetLayoutIndices = new[] { SetFrame, SetGeom, SetIbl, SetWavefront };

        // Set 0: UBO + lights + TLAS + shadow info + accumulator + outColor + emissive tris + alias.
        var set0 = stackalloc DescriptorSetLayoutBinding[8];
        set0[0] = Binding(0, DescriptorType.UniformBuffer);
        set0[1] = Binding(1, DescriptorType.StorageBuffer);
        set0[2] = Binding(2, DescriptorType.AccelerationStructureKhr);
        set0[3] = Binding(3, DescriptorType.StorageBuffer);
        set0[4] = Binding(4, DescriptorType.StorageImage);
        set0[5] = Binding(5, DescriptorType.StorageImage);
        set0[6] = Binding(6, DescriptorType.StorageBuffer);
        set0[7] = Binding(7, DescriptorType.StorageBuffer);
        CreateLayout(set0, 8, out DescriptorSetLayouts[SetFrame]);

        // Set 1: globalVertices (1) + globalIndices (2). Binding 0 intentionally unused.
        var set1 = stackalloc DescriptorSetLayoutBinding[2];
        set1[0] = Binding(1, DescriptorType.StorageBuffer);
        set1[1] = Binding(2, DescriptorType.StorageBuffer);
        CreateLayout(set1, 2, out DescriptorSetLayouts[SetGeom]);

        // Set 2: borrowed bindless layout.
        DescriptorSetLayouts[SetBindless] = Engine.ResourceManager.GetBindlessLayout();

        // Set 3: IBL cubes + BRDF LUT + full-res envCube (4 combined image samplers).
        var set3 = stackalloc DescriptorSetLayoutBinding[4];
        for (uint b = 0; b < 4; b++) set3[b] = Binding(b, DescriptorType.CombinedImageSampler);
        CreateLayout(set3, 4, out DescriptorSetLayouts[SetIbl]);

        // Set 4: 18 storage buffers (the SoA working set).
        var set4 = stackalloc DescriptorSetLayoutBinding[Set4BindingCount];
        for (uint b = 0; b < Set4BindingCount; b++) set4[b] = Binding(b, DescriptorType.StorageBuffer);
        CreateLayout(set4, Set4BindingCount, out DescriptorSetLayouts[SetWavefront]);
    }

    private static DescriptorSetLayoutBinding Binding(uint binding, DescriptorType type) => new()
    {
        Binding = binding, DescriptorType = type, DescriptorCount = 1, StageFlags = ShaderStageFlags.ComputeBit,
    };

    private void CreateLayout(DescriptorSetLayoutBinding* bindings, uint count, out DescriptorSetLayout layout)
    {
        DescriptorSetLayoutCreateInfo info = new()
        {
            SType = StructureType.DescriptorSetLayoutCreateInfo, BindingCount = count, PBindings = bindings,
        };
        if (Vk.CreateDescriptorSetLayout(Device, &info, null, out layout) != Result.Success)
            throw new Exception("Failed to create wavefront descriptor set layout");
    }


    // ---- PSO build: 4 Generate (camera-mode spec id 0) + 5 single PSOs --------
    protected override void CreatePipeline()
    {
        var specEntry = new SpecializationMapEntry { ConstantID = 0, Offset = 0, Size = sizeof(uint) };
        uint specData = 0;
        var specInfo = new SpecializationInfo
        {
            MapEntryCount = 1, PMapEntries = &specEntry, DataSize = sizeof(uint), PData = &specData,
        };
        for (uint mode = 0; mode < 4; mode++)
        {
            specData = mode;
            _generatePsos[mode] = CreateComputePso(GenerateSpv, &specInfo);
        }

        // Shade: one PSO per material class, SHADING_CLASS baked via the same spec id 0 (Shade is a
        // distinct shader, so reusing the entry/info is fine). P3b lobe-strips per class; in P3a the
        // shader ignored the constant so all four were byte-identical.
        for (uint cls = 0; cls < ShadeClasses; cls++)
        {
            specData = cls;
            _shadePsos[cls] = CreateComputePso(ShadeSpv, &specInfo);
        }

        _extendPso   = CreateComputePso(ExtendSpv,   null);
        _connectPso  = CreateComputePso(ConnectSpv,  null);
        _finalizePso = CreateComputePso(FinalizeSpv, null);

        // Alias mode 0 to PipelineHandle so base.Dispose destroys exactly one PSO
        // via that path; everything else is torn down in our Dispose override.
        PipelineHandle = _generatePsos[0];
    }

    private Pipeline CreateComputePso(string spvPath, SpecializationInfo* specInfo)
    {
        byte[] code   = File.ReadAllBytes(spvPath);
        var    module = Gfx.CreateShaderModule(code);
        var    entry  = SilkMarshal.StringToPtr("main");

        var stage = new PipelineShaderStageCreateInfo
        {
            SType               = StructureType.PipelineShaderStageCreateInfo,
            Stage               = ShaderStageFlags.ComputeBit,
            Module              = module,
            PName               = (byte*)entry,
            PSpecializationInfo = specInfo,
        };
        var info = new ComputePipelineCreateInfo
        {
            SType = StructureType.ComputePipelineCreateInfo, Stage = stage, Layout = PipelineLayoutHandle,
        };
        if (Vk.CreateComputePipelines(Device, PipelineCacheHandle, 1, &info, null, out var pso) != Result.Success)
            throw new Exception($"Failed to create wavefront compute pipeline from {Path.GetFileName(spvPath)}");

        SilkMarshal.Free(entry);
        Vk.DestroyShaderModule(Device, module, null);
        return pso;
    }


    // ---- Owned resources: per-frame UBO + the set-4 SoA buffers ---------------
    protected override void CreateResources()
    {
        for (int i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
            Gfx.CreateMappedUniformBuffer(sizeof(PathFrameUBO), ref _frameUbos[i]);

        AllocSet4(PathCount(Renderer.RenderExtent));

        // TEMP/debug compaction readback staging (host-visible, mapped). See _argsReadback.
        Gfx.CreateMappedStorageBuffer(DispatchArgsBytes, ref _argsReadback, BufferUsageFlags.TransferDstBit);
    }

    private static uint PathCount(Extent2D e) => e.Width * e.Height;

    /// <summary>Allocate the device-local SoA working set at <paramref name="pathCount"/> paths.</summary>
    private void AllocSet4(uint pathCount)
    {
        ulong N  = pathCount;
        ulong f4 = N * 16, f2 = N * 8, f1 = N * 4, u1 = N * 4, u2 = N * 8;

        CreateField(B_PS_RAY_ORIGIN, f4);
        CreateField(B_PS_RAY_DIR,    f4);
        CreateField(B_PS_THROUGHPUT, f4);
        CreateField(B_PS_RADIANCE,   f4);
        CreateField(B_PS_RNG,        u1);
        CreateField(B_PS_SIGMA_A,    f4);   // P1: allocated + bound, never written
        CreateField(B_HIT_REC_PRIM,  u2);
        CreateField(B_HIT_T,         f1);
        CreateField(B_HIT_BARY,      f2);
        CreateField(B_RAY_QUEUE0,    u1);
        CreateField(B_RAY_QUEUE1,    u1);
        CreateField(B_SHADE_QUEUE,   ShadeClasses * u1);   // P3: C bins, class c owns [c*N, (c+1)*N)
        CreateField(B_SHADOW_PATH,   u1);
        CreateField(B_SHADOW_ORIGIN, f4);
        CreateField(B_SHADOW_DIR,    f4);
        CreateField(B_SHADOW_LE,     f4);
        CreateField(B_COUNTERS,      CountersBytes);
        // dispatchArgs is also the indirect source for the three worker dispatches. Reset
        // in-shader (PrepareArgs) so no TransferDst / CmdFillBuffer is needed. TransferSrc is
        // for the TEMP/debug RecordArgsReadback copy (drop it if that feature is removed).
        CreateField(B_DISPATCH_ARGS, DispatchArgsBytes,
            BufferUsageFlags.IndirectBufferBit | BufferUsageFlags.TransferSrcBit);
    }

    private void CreateField(int binding, ulong size, BufferUsageFlags extra = 0) =>
        Gfx.CreateBuffer(size, BufferUsageFlags.StorageBufferBit | extra,
            MemoryPropertyFlags.DeviceLocalBit,
            out _set4Buf[binding], out _set4Alloc[binding], GpuMemoryAllocator.PriorityHigh);

    private void FreeSet4()
    {
        for (int i = 0; i < Set4BindingCount; i++)
        {
            if (_set4Buf[i].Handle != 0) Gfx.DestroyBuffer(_set4Buf[i], _set4Alloc[i]);
            _set4Buf[i] = default; _set4Alloc[i] = default;
        }
    }

    /// <summary>Resize path: reallocate the SoA working set to the new extent + rewrite set 4.
    /// Marks the accumulator dirty (the freshly-allocated memory holds garbage).</summary>
    public void ReallocSet4(Extent2D extent)
    {
        FreeSet4();
        AllocSet4(PathCount(extent));
        WriteSet4Descriptors();
        MarkAccumulatorDirty();
    }


    // ---- Descriptor-set allocation -------------------------------------------
    protected override void CreateDescriptorSets()
    {
        DescriptorSets = new DescriptorSet[5][];

        // Set 0 — per frame-in-flight.
        DescriptorSets[SetFrame] = AllocSets(SetFrame, Renderer.MAX_CONCURRENT_FRAMES);
        // Sets 1 / 3 / 4 — single shared (handles are renderer/pipeline-wide singletons).
        DescriptorSets[SetGeom]      = AllocSets(SetGeom, 1);
        DescriptorSets[SetBindless]  = null;   // borrowed
        DescriptorSets[SetIbl]       = AllocSets(SetIbl, 1);
        DescriptorSets[SetWavefront] = AllocSets(SetWavefront, 1);
    }

    private DescriptorSet[] AllocSets(int layoutIdx, uint count)
    {
        var layouts = stackalloc DescriptorSetLayout[(int)count];
        for (int i = 0; i < count; i++) layouts[i] = DescriptorSetLayouts[layoutIdx];
        DescriptorSetAllocateInfo alloc = new()
        {
            SType = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool = Gfx.DescriptorPool, DescriptorSetCount = count, PSetLayouts = layouts,
        };
        var sets = new DescriptorSet[count];
        fixed (DescriptorSet* p = sets)
            if (Vk.AllocateDescriptorSets(Device, &alloc, p) != Result.Success)
                throw new Exception($"Failed to allocate wavefront descriptor set {layoutIdx}");
        return sets;
    }


    // Owned writes at init; everything external has a public Write* the renderer calls.
    protected override void WriteDescriptors()
    {
        WriteFrameUboDescriptors();
        WriteGeometryDescriptors();
        WriteSet4Descriptors();
    }

    private void WriteFrameUboDescriptors()
    {
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            DescriptorBufferInfo info = new() { Buffer = _frameUbos[i].buffer, Offset = 0, Range = (ulong)sizeof(PathFrameUBO) };
            var write = new WriteDescriptorSet
            {
                SType = StructureType.WriteDescriptorSet, DstSet = DescriptorSets[SetFrame][i], DstBinding = 0,
                DescriptorType = DescriptorType.UniformBuffer, DescriptorCount = 1, PBufferInfo = &info,
            };
            Vk.UpdateDescriptorSets(Device, 1, &write, 0, null);
        }
    }

    /// <summary>Set 4 bindings 0-17: the SoA working set. Stable pipeline-owned handles,
    /// so write once at init + after a resize realloc.</summary>
    public void WriteSet4Descriptors()
    {
        var set = DescriptorSets[SetWavefront][0];
        var infos  = stackalloc DescriptorBufferInfo[Set4BindingCount];
        var writes = stackalloc WriteDescriptorSet[Set4BindingCount];
        for (uint b = 0; b < Set4BindingCount; b++)
        {
            infos[b] = new DescriptorBufferInfo { Buffer = _set4Buf[b], Offset = 0, Range = Vk.WholeSize };
            writes[b] = new WriteDescriptorSet
            {
                SType = StructureType.WriteDescriptorSet, DstSet = set, DstBinding = b,
                DescriptorType = DescriptorType.StorageBuffer, DescriptorCount = 1, PBufferInfo = &infos[b],
            };
        }
        Vk.UpdateDescriptorSets(Device, Set4BindingCount, writes, 0, null);
    }

    /// <summary>Set 1 bindings 1/2: globalVertices + globalIndices. Renderer-wide singletons.</summary>
    public void WriteGeometryDescriptors()
    {
        var rm = Engine.ResourceManager;
        DescriptorBufferInfo vbInfo = new() { Buffer = rm.GlobalVertexBuffer, Offset = 0, Range = Vk.WholeSize };
        DescriptorBufferInfo ibInfo = new() { Buffer = rm.GlobalIndexBuffer,  Offset = 0, Range = Vk.WholeSize };
        var writes = stackalloc WriteDescriptorSet[2];
        writes[0] = new() { SType = StructureType.WriteDescriptorSet, DstSet = DescriptorSets[SetGeom][0], DstBinding = 1, DescriptorType = DescriptorType.StorageBuffer, DescriptorCount = 1, PBufferInfo = &vbInfo };
        writes[1] = new() { SType = StructureType.WriteDescriptorSet, DstSet = DescriptorSets[SetGeom][0], DstBinding = 2, DescriptorType = DescriptorType.StorageBuffer, DescriptorCount = 1, PBufferInfo = &ibInfo };
        Vk.UpdateDescriptorSets(Device, 2, writes, 0, null);
    }

    /// <summary>Set 0 binding 1: PbrLight SSBO. Borrowed from Renderer.</summary>
    public void WriteLightsDescriptor()
    {
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            DescriptorBufferInfo info = new()
            {
                Buffer = Renderer.GetLightStorageBuffer((uint)i), Offset = 0,
                Range = (ulong)(Renderer.MAX_LIGHTS * (uint)sizeof(PbrLightGpu)),
            };
            var write = new WriteDescriptorSet { SType = StructureType.WriteDescriptorSet, DstSet = DescriptorSets[SetFrame][i], DstBinding = 1, DescriptorType = DescriptorType.StorageBuffer, DescriptorCount = 1, PBufferInfo = &info };
            Vk.UpdateDescriptorSets(Device, 1, &write, 0, null);
        }
    }

    /// <summary>Set 0 binding 2: TLAS. Call after InitRayQuery + every TLAS rebuild.</summary>
    public void WriteTlasDescriptor(AccelerationStructureKHR tlas)
    {
        if (tlas.Handle == 0) return;
        var tlasH = tlas;
        var asWrite = new WriteDescriptorSetAccelerationStructureKHR
        {
            SType = StructureType.WriteDescriptorSetAccelerationStructureKhr,
            AccelerationStructureCount = 1, PAccelerationStructures = &tlasH,
        };
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            var write = new WriteDescriptorSet
            {
                SType = StructureType.WriteDescriptorSet, PNext = &asWrite, DstSet = DescriptorSets[SetFrame][i],
                DstBinding = 2, DescriptorType = DescriptorType.AccelerationStructureKhr, DescriptorCount = 1,
            };
            Vk.UpdateDescriptorSets(Device, 1, &write, 0, null);
        }
    }

    /// <summary>Set 0 binding 3: ShadowEntityInfo SSBO. Re-call on every reallocation.</summary>
    public void WriteShadowInfoDescriptor()
    {
        var buf = Renderer.ShadowInfoBuffer;
        if (buf.Handle == 0) return;
        DescriptorBufferInfo info = new() { Buffer = buf, Offset = 0, Range = Renderer.ShadowInfoBufferSize };
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            var write = new WriteDescriptorSet { SType = StructureType.WriteDescriptorSet, DstSet = DescriptorSets[SetFrame][i], DstBinding = 3, DescriptorType = DescriptorType.StorageBuffer, DescriptorCount = 1, PBufferInfo = &info };
            Vk.UpdateDescriptorSets(Device, 1, &write, 0, null);
        }
    }

    /// <summary>Set 0 bindings 6/7: emissive-triangle SSBO + alias table. Borrowed from Renderer.</summary>
    public void WriteEmissiveDescriptors()
    {
        var triBuf   = Renderer.EmissiveTriBuffer;
        var aliasBuf = Renderer.EmissiveAliasBuffer;
        if (triBuf.Handle == 0 || aliasBuf.Handle == 0) return;
        DescriptorBufferInfo triInfo   = new() { Buffer = triBuf,   Offset = 0, Range = Renderer.EmissiveTriBufferSize };
        DescriptorBufferInfo aliasInfo = new() { Buffer = aliasBuf, Offset = 0, Range = Renderer.EmissiveAliasBufferSize };
        var writes = stackalloc WriteDescriptorSet[2];
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            writes[0] = new() { SType = StructureType.WriteDescriptorSet, DstSet = DescriptorSets[SetFrame][i], DstBinding = 6, DescriptorType = DescriptorType.StorageBuffer, DescriptorCount = 1, PBufferInfo = &triInfo };
            writes[1] = new() { SType = StructureType.WriteDescriptorSet, DstSet = DescriptorSets[SetFrame][i], DstBinding = 7, DescriptorType = DescriptorType.StorageBuffer, DescriptorCount = 1, PBufferInfo = &aliasInfo };
            Vk.UpdateDescriptorSets(Device, 2, writes, 0, null);
        }
    }

    /// <summary>Set 0 bindings 4/5: accumulator + outColor storage images (both in General).</summary>
    public void WriteStorageImageDescriptors(ImageView accumView, ImageView outColorView)
    {
        DescriptorImageInfo accumInfo = new() { ImageView = accumView,    ImageLayout = ImageLayout.General };
        DescriptorImageInfo outInfo   = new() { ImageView = outColorView, ImageLayout = ImageLayout.General };
        var writes = stackalloc WriteDescriptorSet[2];
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            writes[0] = new() { SType = StructureType.WriteDescriptorSet, DstSet = DescriptorSets[SetFrame][i], DstBinding = 4, DescriptorType = DescriptorType.StorageImage, DescriptorCount = 1, PImageInfo = &accumInfo };
            writes[1] = new() { SType = StructureType.WriteDescriptorSet, DstSet = DescriptorSets[SetFrame][i], DstBinding = 5, DescriptorType = DescriptorType.StorageImage, DescriptorCount = 1, PImageInfo = &outInfo };
            Vk.UpdateDescriptorSets(Device, 2, writes, 0, null);
        }
        MarkAccumulatorDirty();
    }

    /// <summary>Set 3 bindings 0/1/2/3: irradiance + prefiltered + BRDF LUT + full-res envCube.</summary>
    public void WriteIblDescriptors()
    {
        var imageInfos = stackalloc DescriptorImageInfo[4]
        {
            new() { ImageView = Renderer.irradianceCubeView,  Sampler = Renderer.iblCubeSampler, ImageLayout = ImageLayout.ShaderReadOnlyOptimal },
            new() { ImageView = Renderer.prefilteredCubeView, Sampler = Renderer.iblCubeSampler, ImageLayout = ImageLayout.ShaderReadOnlyOptimal },
            new() { ImageView = Renderer.brdfLutView,         Sampler = Renderer.iblLutSampler,  ImageLayout = ImageLayout.ShaderReadOnlyOptimal },
            new() { ImageView = Renderer.envCubeView,         Sampler = Renderer.iblCubeSampler, ImageLayout = ImageLayout.ShaderReadOnlyOptimal },
        };
        var writes = stackalloc WriteDescriptorSet[4];
        for (uint b = 0; b < 4; b++)
            writes[b] = new WriteDescriptorSet { SType = StructureType.WriteDescriptorSet, DstSet = DescriptorSets[SetIbl][0], DstBinding = b, DescriptorType = DescriptorType.CombinedImageSampler, DescriptorCount = 1, PImageInfo = &imageInfos[b] };
        Vk.UpdateDescriptorSets(Device, 4, writes, 0, null);
    }


    // ---- Per-frame UBO upload (mirrors PTComputePipeline.UpdatePerFrame) -------
    public bool UpdatePerFrame(uint frameIndex, Camera camera, uint lightCount, Extent2D renderExtent)
    {
        bool reset = _accumDirty;
        if (reset) { _accumSamples = 0; _accumDirty = false; }
        else       { _accumSamples++; }

        Matrix4x4 view = camera != null ? camera.GetViewMatrix() : Matrix4x4.Identity;
        Matrix4x4 proj = camera != null
            ? camera.GetProjectionMatrix((float)renderExtent.Width / renderExtent.Height, 0.1f, 100.0f)
            : Matrix4x4.Identity;
        proj.M22 *= -1;   // Vulkan NDC Y-flip, matching the rest of the renderer

        Matrix4x4.Invert(view, out var invView);
        Matrix4x4.Invert(proj, out var invProj);

        float fovDeg     = camera != null ? camera.Fov : 60.0f;
        float fovRad     = fovDeg * (float)(Math.PI / 180.0);
        float tanHalfFov = MathF.Tan(fovRad * 0.5f);

        PathFrameUBO ubo = new()
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
            prefilteredCubeMipLevels = Renderer.prefilteredCubeMipLevels,
            scaleIBLAmbient          = EditorState.IblIntensity,
            focusDistance            = FocusDistance,
            aperture                 = Aperture,
            paniniDistance           = PaniniDistance,
            verticalCompression      = VerticalCompression,
            emissiveTriCount         = Renderer.EmissiveTriangleCount,
            totalEmissivePower       = Renderer.TotalEmissivePower,
        };
        void* data = _frameUbos[frameIndex].mapped;
        new Span<PathFrameUBO>(data, 1).Fill(ubo);
        return reset;
    }


    // ---- Record helpers (the module's pass bodies call these) -----------------
    private void BindSets(CommandBuffer cmd, uint frameIndex)
    {
        var sets = stackalloc DescriptorSet[5]
        {
            DescriptorSets[SetFrame][frameIndex],
            DescriptorSets[SetGeom][0],
            Engine.ResourceManager.GetBindlessSet(frameIndex),
            DescriptorSets[SetIbl][0],
            DescriptorSets[SetWavefront][0],
        };
        Vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute, Layout, 0, 5, sets, 0, null);
    }

    private void PushWavefront(CommandBuffer cmd, uint bounce, uint argsClass)
    {
        var pc = new WavefrontPush { bounce = bounce, srcParity = bounce & 1u, argsClass = argsClass, maxBounces = MaxBounces };
        Vk.CmdPushConstants(cmd, Layout, ShaderStageFlags.ComputeBit, 0, (uint)sizeof(WavefrontPush), &pc);
    }

    /// <summary>Dense primary-ray generation (one PSO per camera mode).</summary>
    public void RecordGenerate(CommandBuffer cmd, in Renderer.FrameContext ctx)
    {
        int modeIdx = (int)Mode;
        if (modeIdx < 0 || modeIdx >= _generatePsos.Length) modeIdx = 0;
        Vk.CmdBindPipeline(cmd, PipelineBindPoint.Compute, _generatePsos[modeIdx]);
        BindSets(cmd, ctx.FrameIndex);
        Vk.CmdDispatch(cmd, (ctx.RenderExtent.Width + 7u) / 8u, (ctx.RenderExtent.Height + 7u) / 8u, 1);
    }

    public void RecordExtend(CommandBuffer cmd, uint frameIndex, uint bounce)
    {
        Vk.CmdBindPipeline(cmd, PipelineBindPoint.Compute, _extendPso);
        BindSets(cmd, frameIndex);
        PushWavefront(cmd, bounce, 0u);
        Vk.CmdDispatchIndirect(cmd, DispatchArgsBuffer, ExtendArgsOffset(bounce));
    }

    /// <summary>Shade one material-class bin (P3). <paramref name="shadingClass"/> selects the
    /// queue slice + counter slot via the push constant and the per-class indirect args. In P3a all
    /// classes bind the same (FULL) PSO; P3b will bake a per-class lobe-stripped PSO.</summary>
    public void RecordShade(CommandBuffer cmd, uint frameIndex, uint bounce, uint shadingClass)
    {
        uint cls = shadingClass < ShadeClasses ? shadingClass : ShadeClasses - 1u;
        Vk.CmdBindPipeline(cmd, PipelineBindPoint.Compute, _shadePsos[cls]);
        BindSets(cmd, frameIndex);
        PushWavefront(cmd, bounce, cls);
        Vk.CmdDispatchIndirect(cmd, DispatchArgsBuffer, ShadeArgsOffset(bounce, cls));
    }

    public void RecordConnect(CommandBuffer cmd, uint frameIndex, uint bounce)
    {
        Vk.CmdBindPipeline(cmd, PipelineBindPoint.Compute, _connectPso);
        BindSets(cmd, frameIndex);
        PushWavefront(cmd, bounce, 0u);   // Connect ignores argsClass, but keeps the range covered
        Vk.CmdDispatchIndirect(cmd, DispatchArgsBuffer, ConnectArgsOffset(bounce));
    }

    /// <summary>Dense finalize: accumulate radiance, normalize into outColor.</summary>
    public void RecordFinalize(CommandBuffer cmd, in Renderer.FrameContext ctx)
    {
        Vk.CmdBindPipeline(cmd, PipelineBindPoint.Compute, _finalizePso);
        BindSets(cmd, ctx.FrameIndex);
        Vk.CmdDispatch(cmd, (ctx.RenderExtent.Width + 7u) / 8u, (ctx.RenderExtent.Height + 7u) / 8u, 1);
    }

    // ---- TEMP/debug: indirect-args readback (see _argsReadback) ------------------------------
    /// <summary>Copy the per-bounce dispatchArgs into the host-visible staging buffer. Record once
    /// after the graph executes; the values are readable (a frame or two stale) via
    /// <see cref="ReadDispatchArgs"/>.</summary>
    public void RecordArgsReadback(CommandBuffer cmd)
    {
        // dispatchArgs was written by PrepArgs (compute) + read as indirect args by the workers;
        // make both visible to the transfer copy.
        var bar = new BufferMemoryBarrier
        {
            SType = StructureType.BufferMemoryBarrier,
            SrcAccessMask = AccessFlags.ShaderWriteBit | AccessFlags.IndirectCommandReadBit,
            DstAccessMask = AccessFlags.TransferReadBit,
            SrcQueueFamilyIndex = Silk.NET.Vulkan.Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Silk.NET.Vulkan.Vk.QueueFamilyIgnored,
            Buffer = DispatchArgsBuffer, Offset = 0, Size = DispatchArgsBytes,
        };
        Vk.CmdPipelineBarrier(cmd,
            PipelineStageFlags.ComputeShaderBit | PipelineStageFlags.DrawIndirectBit,
            PipelineStageFlags.TransferBit, 0, 0, null, 1, &bar, 0, null);
        var region = new BufferCopy { SrcOffset = 0, DstOffset = 0, Size = DispatchArgsBytes };
        Vk.CmdCopyBuffer(cmd, DispatchArgsBuffer, _argsReadback.buffer, 1, &region);
    }

    /// <summary>Per-bounce [extend, shade, connect] indirect workgroup counts (the .x of each
    /// VkDispatchIndirectCommand). Length = MaxBounces*3. ~a frame stale; best-effort.</summary>
    public uint[] ReadDispatchArgs()
    {
        var result = new uint[MaxBounces * StagesPerBounce];
        var src = new ReadOnlySpan<uint>(_argsReadback.mapped, (int)(DispatchArgsBytes / 4));
        for (uint b = 0; b < MaxBounces; b++)
            for (uint s = 0; s < StagesPerBounce; s++)
                result[b * StagesPerBounce + s] = src[(int)((b * StagesPerBounce + s) * 3)];   // .x
        return result;
    }


    public override void Dispose()
    {
        for (int i = 1; i < _generatePsos.Length; i++)
            if (_generatePsos[i].Handle != 0) Vk.DestroyPipeline(Device, _generatePsos[i], null);
        foreach (var pso in _shadePsos)
            if (pso.Handle != 0) Vk.DestroyPipeline(Device, pso, null);
        if (_extendPso.Handle   != 0) Vk.DestroyPipeline(Device, _extendPso, null);
        if (_connectPso.Handle  != 0) Vk.DestroyPipeline(Device, _connectPso, null);
        if (_finalizePso.Handle != 0) Vk.DestroyPipeline(Device, _finalizePso, null);

        FreeSet4();
        foreach (var b in _frameUbos) Gfx.DestroyBuffer(b.buffer, b.alloc);
        Gfx.DestroyBuffer(_argsReadback.buffer, _argsReadback.alloc);   // TEMP/debug readback staging
        base.Dispose();   // destroys mode-0 PSO (= PipelineHandle), layout, owned DSLs, cache
    }
}