using System.Numerics;
using System.Runtime.InteropServices;
using CadThingo.VulkanEngine.ImGui;
using CadThingo.VulkanEngine.Renderer.Descriptors;
using CadThingo.VulkanEngine.Renderer.FrameGraph;
using CadThingo.VulkanEngine.Renderer.Pipelines;
using CadThingo.VulkanEngine.Renderer.Features.PathTracer;
using CadThingo.VulkanEngine.Renderer.Slang;
using Silk.NET.Core.Native;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Features.WavefrontPathTracer;

//
//  Wavefront path-tracer pipeline (graph-resident, dispatchIndirect, SoA).
//
//  Owns:    12 compute PSOs (Generate x4 by camera mode, Shade x4 by material class,
//           plus Extend / Connect / Finalize / Tail); the SoA working-set buffers
//           (device-local, single-buffered, sized to the render extent); the set-1 (IO)
//           and set-2 (working) layouts. PathFrameUBO is staged per frame and pushed into
//           the scene set's (0,0) constant-arena slot at record.
//  Borrows: the scene set (set 0, registry-owned) -- TLAS, lights, ShadowEntityInfo,
//           global VB/IB, emissive tables, bindless materials/textures/samplers.
//
//  Inherits PipelineBase (not ComputePipeline) because ComputePipeline builds the single
//  PSO its Program reflects; this pipeline bakes twelve from six programs. Program (Shade)
//  still drives the shared layout: all 12 PSOs bind one pipeline layout (5 set slots + the
//  single reflected 16-byte ComputeBit push range for WavefrontPush).
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

    // Set 0 borrowed from DescriptorRegistry (scene resources + the (0,0) constant-arena slot
    // carrying PathFrameUBO); sets 1 (IO) / 2 (SoA working, the graph-shared slot) are
    // pipeline-owned; envCube rides the registry-owned FeatureEnv set (set 4).
    private const int SetWavefront = 2;
    private const string FeatureEnv  = "FeatureEnv";
    private const string FeaturePtIo = "FeaturePTIO";

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
    public const uint MaxBounces = 8u;

    // Megakernel tail cutoff. The wavefront chain runs bounces [0, TailStartBounce); a single
    // TailMegakernel dispatch then loops [TailStartBounce, MaxBounces) per surviving path, so the
    // sparse deep bounces (mostly transmissive) stop paying per-bounce Extend/Shade/Connect +
    // arg-gen + barrier overhead. Must satisfy 1 <= TailStartBounce <= MaxBounces; set it equal to
    // MaxBounces to disable the tail (pure wavefront, every bounce a full chain step).
    public const uint TailStartBounce = 3u;
    public static bool TailEnabled => TailStartBounce < MaxBounces;

    // ---- dispatchArgs byte layout (PER BOUNCE) -------------------------------
    // Each VkDispatchIndirectCommand is 12 bytes (x,y,z). Each bounce b owns a contiguous block of
    // STAGES_PER_BOUNCE commands: [extend | shade[0..C-1] | connect] (P3: the shade stage fans out
    // to one indirect command per material class). The buffer RETAINS every bounce's launch size, so
    // a RenderDoc/Nsight capture (or the readback) shows the indirect args shrinking down the chain
    // -- the visible proof of compaction. Offsets MUST mirror WavefrontBindings.wf*ArgsOffset.
    // NOTE: the connect slot is DEBUG-ONLY (kept so the Stats-panel readback layout is stable);
    // Connect actually launches from ConnectArgsBuffer (see B_CONNECT_ARGS / RecordConnect).
    public  const uint  StagesPerBounce  = 2u + ShadeClasses;   // extend + C shade + connect
    private const uint  ArgStride        = 12u;      // sizeof(VkDispatchIndirectCommand)
    public static uint  ExtendArgsOffset (uint bounce)            => (bounce * StagesPerBounce + 0u)             * ArgStride;
    public static uint  ShadeArgsOffset  (uint bounce, uint cls)  => (bounce * StagesPerBounce + 1u + cls)       * ArgStride;
    public static uint  ConnectArgsOffset(uint bounce)            => (bounce * StagesPerBounce + 1u + ShadeClasses) * ArgStride;
    private const ulong DispatchArgsBytes = (ulong)MaxBounces * StagesPerBounce * ArgStride;  // 4*6*12 = 288
    private const ulong CountersBytes     = 32;   // 8 uints; indices reach COMPLETED_WG=7

    // Each kernel is its own Slang module (they share the layout only by importing
    // WavefrontBindings), so each resolves its own program. All six are single-entry compute with
    // the ray-query capability the traceRayInline call sites need.
    private static ShaderCompileRequest Kernel(string name) =>
        new($"WavefrontPathTracer/{name}", ["main"], [], ["spvRayQueryKHR"]);

    private static readonly ShaderCompileRequest GenerateReq = Kernel("Generate");
    private static readonly ShaderCompileRequest ExtendReq   = Kernel("Extend");
    private static readonly ShaderCompileRequest ShadeReq    = Kernel("Shade");
    private static readonly ShaderCompileRequest ConnectReq  = Kernel("Connect");
    private static readonly ShaderCompileRequest FinalizeReq = Kernel("Finalize");
    private static readonly ShaderCompileRequest TailReq     = Kernel("TailMegakernel");

    // Shade drives the shared layout: PipelineBase reflects the push range + the set layouts from
    // Program alone, and Shade is the kernel that actually reads WavefrontPush AND touches the full
    // binding set, so nothing the other eleven PSOs need can be missing from what it reflects.
    // Every kernel imports WavefrontBindings, so all six reflect the same sets either way.
    protected override ShaderCompileRequest? Program => ShadeReq;

    // Generate is baked per camera mode (its `mode` constant); Shade per material class
    // (SHADING_CLASS -> P3b lobe stripping); Extend/Connect/Finalize are single PSOs. (No
    // PrepareArgs PSO anymore -- arg generation is fused onto the producers' tails.)
    private readonly Pipeline[] _generatePsos = new Pipeline[4];
    private readonly Pipeline[] _shadePsos    = new Pipeline[ShadeClasses];
    private Pipeline _extendPso, _connectPso, _finalizePso;
    private Pipeline _tailPso;   // megakernel tail (loops the deep bounces per surviving path)

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
                      B_SHADOW_PATH0 = 12, B_SHADOW_ORIGIN0 = 13, B_SHADOW_DIR0 = 14, B_SHADOW_LE0 = 15,
                      B_COUNTERS = 16, B_DISPATCH_ARGS = 17, B_CONNECT_ARGS0 = 18,
                      // Ping-pong partners (bounce parity 1) + Connect's private NEE accumulator.
                      B_SHADOW_PATH1 = 19, B_SHADOW_ORIGIN1 = 20, B_SHADOW_DIR1 = 21, B_SHADOW_LE1 = 22,
                      B_CONNECT_ARGS1 = 23, B_SHADOW_RADIANCE = 24;
    private const int Set4BindingCount = 25;

    private readonly Buffer[]   _set4Buf   = new Buffer[Set4BindingCount];
    private readonly SubAlloc[] _set4Alloc = new SubAlloc[Set4BindingCount];

    // The SoA working set (set 2) is graph-owned: the pipeline owns the buffers + the layout, but the
    // FrameGraph allocates + writes the descriptor set from this spec and hands it back after Compile.
    private DescriptorSet _sharedSet;   // set by the core via SetGraphSharedSet after the graph compiles
    public void SetGraphSharedSet(DescriptorSet set) => _sharedSet = set;

    /// The graph-shared working-set spec (layout + current buffer handles). Read by the module at
    /// Build time; on resize the buffers realloc first, so the rebuilt graph bakes the new handles.
    public GraphSharedSetSpec GraphSharedSpec
    {
        get
        {
            // Binding index + descriptor type come from WavefrontBindings' set-2 declarations; the
            // buffer filling each is _set4Buf[binding] (the B_* constants ARE binding numbers, and
            // CreateDescriptorSetLayouts asserts the declaration count still matches the array).
            var declared = ReflectedBindings((uint)SetWavefront);
            var bindings = new GraphSharedBinding[declared.Length];
            for (int i = 0; i < declared.Length; i++)
                bindings[i] = new(declared[i].Binding, declared[i].Type, _set4Buf[declared[i].Binding]);
            return new(ShaderSets.GraphShared, DescriptorSetLayouts[SetWavefront], bindings);
        }
    }

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
    public Buffer ShadowPath0   => _set4Buf[B_SHADOW_PATH0];
    public Buffer ShadowOrigin0 => _set4Buf[B_SHADOW_ORIGIN0];
    public Buffer ShadowDir0    => _set4Buf[B_SHADOW_DIR0];
    public Buffer ShadowLe0     => _set4Buf[B_SHADOW_LE0];
    public Buffer ShadowPath1   => _set4Buf[B_SHADOW_PATH1];
    public Buffer ShadowOrigin1 => _set4Buf[B_SHADOW_ORIGIN1];
    public Buffer ShadowDir1    => _set4Buf[B_SHADOW_DIR1];
    public Buffer ShadowLe1     => _set4Buf[B_SHADOW_LE1];
    public Buffer ShadowRadiance => _set4Buf[B_SHADOW_RADIANCE];
    public Buffer Counters      => _set4Buf[B_COUNTERS];
    public Buffer DispatchArgsBuffer => _set4Buf[B_DISPATCH_ARGS];
    public Buffer ConnectArgsBuffer0 => _set4Buf[B_CONNECT_ARGS0];
    public Buffer ConnectArgsBuffer1 => _set4Buf[B_CONNECT_ARGS1];

    // Frame constants staged by UpdatePerFrame, pushed once per frame into the scene set's
    // (0,0) constant arena; the returned dynamic offset is cached for every BindSets this frame.
    private PathFrameUBO _frameUbo;
    private uint         _frameConstants;

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

    // The single 16-byte ComputeBit range for WavefrontPush is reflected from Program by
    // Initialize; CreateResources asserts the C# mirror still matches it.
    public WavefrontPTPipeline(GpuContext gpu, Renderer renderer) : base(gpu, renderer) { }


    // ---- Descriptor-set layouts. Scene (set 0) + FeatureEnv (4) + FeaturePTIO (5) come from the
    // registry; only the SoA working set (2) is built here, from WavefrontBindings' declarations.
    // Array is [scene, empty, working, empty, FeatureEnv, FeaturePTIO] -- sets 1 and 3 are gaps.
    protected override void CreateDescriptorSetLayouts()
    {
        // BuildPipelineSetLayouts covers scene + the pass slot + the features; the graph-shared
        // slot has no place in its signature, so the working layout drops into index 2 after.
        DescriptorSetLayouts = Registry.BuildPipelineSetLayouts(null, FeatureEnv, FeaturePtIo);
        DescriptorSetLayouts[SetWavefront] = CreateReflectedSetLayout((uint)SetWavefront);
        OwnedDescriptorSetLayoutIndices = new[] { SetWavefront };

        // The SoA buffers are indexed BY binding number (the B_* constants ARE the bindings they
        // fill), so a binding added to set 2 without a matching field here would silently leave a
        // stale descriptor. Reflection cannot allocate the buffer for us; it can refuse to start.
        int declared = ReflectedBindings((uint)SetWavefront).Length;
        if (declared != Set4BindingCount)
            throw new Exception(
                $"WavefrontBindings declares {declared} bindings in set {SetWavefront} but the " +
                $"pipeline allocates {Set4BindingCount} SoA buffers for them");
    }


    // ---- PSO build: 4 Generate (camera mode) + 4 Shade (material class) + 4 single PSOs -------
    // Twelve PSOs over six programs on one shared layout. PipelineBase resolves only Program
    // (Shade); the others resolve here. ShaderLibrary memoizes by request, so a Rebuild re-reads
    // the same programs rather than recompiling them.
    protected override void CreatePipeline()
    {
        var generate = Shaders.GetProgram(GenerateReq);
        var shade    = Reflected!;

        // Generate and Shade each bake ONE uint constant, and each declares it in its OWN module.
        // The base's Specialization hook resolves names against Program alone, which cannot see
        // Generate's constant, so the by-name lookup runs per program here instead. Same contract
        // either way: C# names the constant, the shader owns the id.
        uint specData = 0;
        var specEntry = new SpecializationMapEntry
        {
            ConstantID = SpecId(generate, "mode"), Offset = 0, Size = sizeof(uint),
        };
        var specInfo = new SpecializationInfo
        {
            MapEntryCount = 1, PMapEntries = &specEntry, DataSize = sizeof(uint), PData = &specData,
        };
        for (uint mode = 0; mode < 4; mode++)
        {
            specData = mode;
            _generatePsos[mode] = CreateComputePso(generate, &specInfo);
        }

        // Shade: one PSO per material class. P3b lobe-strips per class; in P3a the shader ignored
        // the constant so all four were byte-identical.
        specEntry.ConstantID = SpecId(shade, "SHADING_CLASS");
        for (uint cls = 0; cls < ShadeClasses; cls++)
        {
            specData = cls;
            _shadePsos[cls] = CreateComputePso(shade, &specInfo);
        }

        _extendPso   = CreateComputePso(Shaders.GetProgram(ExtendReq),   null);
        _connectPso  = CreateComputePso(Shaders.GetProgram(ConnectReq),  null);
        _finalizePso = CreateComputePso(Shaders.GetProgram(FinalizeReq), null);
        if (TailEnabled) _tailPso = CreateComputePso(Shaders.GetProgram(TailReq), null);

        // Alias mode 0 to PipelineHandle so base.Dispose destroys exactly one PSO
        // via that path; everything else is torn down in our Dispose override.
        PipelineHandle = _generatePsos[0];
    }

    // A kernel's spec-constant id, by NAME. Mirrors what PipelineBase does for Program, against
    // whichever program actually declares the constant.
    private static uint SpecId(ShaderProgram program, string name)
    {
        var match = program.Reflection.SpecConstants.FirstOrDefault(c => c.Name == name);
        if (match.Name != name)
            throw new InvalidOperationException(
                $"WavefrontPTPipeline: spec constant '{name}' is not declared by {program.Desc.Module} " +
                $"(declared: {string.Join(", ", program.Reflection.SpecConstants.Select(c => c.Name))})");
        return match.ConstantId;
    }

    // Entry name + stage both come from reflection: slangc now preserves the source entry symbol
    // into the SPIR-V, so a hardcoded "main" here would be a second, silently-drifting contract.
    private Pipeline CreateComputePso(ShaderProgram program, SpecializationInfo* specInfo)
    {
        var ep     = program.Reflection.EntryPoints[0];
        var module = Gfx.CreateShaderModule(program.Spirv(0).ToArray());
        var entry  = SilkMarshal.StringToPtr(ep.Name);

        var stage = new PipelineShaderStageCreateInfo
        {
            SType               = StructureType.PipelineShaderStageCreateInfo,
            Stage               = ep.Stage,
            Module              = module,
            PName               = (byte*)entry,
            PSpecializationInfo = specInfo,
        };
        var info = new ComputePipelineCreateInfo
        {
            SType = StructureType.ComputePipelineCreateInfo, Stage = stage, Layout = PipelineLayoutHandle,
        };
        if (Vk.CreateComputePipelines(Device, PipelineCacheHandle, 1, &info, null, out var pso) != Result.Success)
            throw new Exception($"Failed to create wavefront compute pipeline from {program.Desc.Module}");

        SilkMarshal.Free(entry);
        Vk.DestroyShaderModule(Device, module, null);
        return pso;
    }


    // ---- Owned resources: the set-3 SoA buffers (frame UBO rides the scene arena) -----
    protected override void CreateResources()
    {
        // All twelve PSOs share the layout's single reflected push range, but only the worker
        // kernels read it -- so this asserts both that the C# mirror of WavefrontPush still matches
        // the shader AND that Program's reflection carried the range at all (a Program whose module
        // never declared `pc` would silently strip it from the shared layout).
        if (PushConstantRanges.Length != 1 || PushConstantRanges[0].Size != (uint)sizeof(WavefrontPush))
            throw new Exception(
                $"WavefrontPathTracer/Shade must reflect exactly one {sizeof(WavefrontPush)}-byte push " +
                $"range for WavefrontPush; got [{string.Join(", ", PushConstantRanges.Select(p => p.Size))}]");

        // The SoA working set is extent-sized, so it is allocated by ReallocSet4, which the core
        // drives from its Resize (primed once at boot).

        // TEMP/debug compaction readback staging (host-visible, mapped). See _argsReadback.
        Gfx.CreateMappedStorageBuffer(DispatchArgsBytes, ref _argsReadback, BufferUsageFlags.TransferDstBit);
    }

    private static uint PathCount(Extent2D e) => e.Width * e.Height;

    /// <summary>Allocate the device-local SoA working set at <paramref name="pathCount"/> paths.
    /// h4/h2 fields are fp16-PACKED (uint2 = half4, uint = half2; see WavefrontBindings'
    /// wfPackHalf4) -- ~44B/path off the footprint and half the payload traffic in the hot
    /// kernels. Ray origins/dirs, shadowOrigin and hitT stay fp32 (trace precision; the MIS
    /// pdf in psRayDir.w exceeds half range at low roughness).</summary>
    private void AllocSet4(uint pathCount)
    {
        ulong N  = pathCount;
        ulong f4 = N * 16, f1 = N * 4, u1 = N * 4, u2 = N * 8, h4 = N * 8, h2 = N * 4;

        CreateField(B_PS_RAY_ORIGIN, f4);
        CreateField(B_PS_RAY_DIR,    f4);
        CreateField(B_PS_THROUGHPUT, h4);
        CreateField(B_PS_RADIANCE,   h4, crossQueue: true);   // Connect (async queue) accumulates NEE
        CreateField(B_PS_RNG,        u1);
        CreateField(B_PS_SIGMA_A,    h4);
        CreateField(B_HIT_REC_PRIM,  u2);
        CreateField(B_HIT_T,         f1);
        CreateField(B_HIT_BARY,      h2);
        CreateField(B_RAY_QUEUE0,    u1);
        CreateField(B_RAY_QUEUE1,    u1);
        CreateField(B_SHADE_QUEUE,   ShadeClasses * u1);   // P3: C bins, class c owns [c*N, (c+1)*N)
        // Shadow records + connect-args are PING-PONGED by bounce parity (set b&1): Connect(b)
        // reads its set while Shade(b+1) fills the other, so the async Connect's downstream
        // consumer moves from Shade(b+1) to Shade(b+2) -- a full bounce of overlap window.
        CreateField(B_SHADOW_PATH0,   u1, crossQueue: true);
        CreateField(B_SHADOW_ORIGIN0, f4, crossQueue: true);
        CreateField(B_SHADOW_DIR0,    h4, crossQueue: true);
        CreateField(B_SHADOW_LE0,     h4, crossQueue: true);
        CreateField(B_SHADOW_PATH1,   u1, crossQueue: true);
        CreateField(B_SHADOW_ORIGIN1, f4, crossQueue: true);
        CreateField(B_SHADOW_DIR1,    h4, crossQueue: true);
        CreateField(B_SHADOW_LE1,     h4, crossQueue: true);
        // Connect's private NEE accumulator (packed half4): Generate zeroes it, the Connects
        // += into it, Finalize folds it -- no graphics pass ever touches it mid-frame.
        CreateField(B_SHADOW_RADIANCE, h4, crossQueue: true);
        CreateField(B_COUNTERS,      CountersBytes);
        // dispatchArgs is also the indirect source for the three worker dispatches. Reset
        // in-shader (PrepareArgs) so no TransferDst / CmdFillBuffer is needed. TransferSrc is
        // for the TEMP/debug RecordArgsReadback copy (drop it if that feature is removed).
        CreateField(B_DISPATCH_ARGS, DispatchArgsBytes,
            BufferUsageFlags.IndirectBufferBit | BufferUsageFlags.TransferSrcBit);
        // Connect's own indirect sources (decoupling): 16B per bounce = [groups,1,1,count],
        // written by Shade's fused tail, so Connect(b)'s launch never barriers against
        // Extend(b+1)'s dispatchArgs tail-write. The 4th word is the kernel's thread gate.
        CreateField(B_CONNECT_ARGS0, (ulong)MaxBounces * 16u,
            BufferUsageFlags.IndirectBufferBit, crossQueue: true);
        CreateField(B_CONNECT_ARGS1, (ulong)MaxBounces * 16u,
            BufferUsageFlags.IndirectBufferBit, crossQueue: true);
    }

    // crossQueue: the buffer is touched by Connect, which the graph runs on the dedicated
    // async-compute queue when the device has one -- CONCURRENT sharing skips queue-family
    // ownership transfers (the graph's timeline semaphores carry the memory dependency).
    // Collapses to EXCLUSIVE automatically when the compute family aliases graphics.
    // NOTE: Connect also READS frame-global exclusive resources cross-queue (TLAS, entity
    // info, materials, bindless textures, global VB/IB, frame UBO). Spec-strictly those want
    // ownership transfers too; they are immutable mid-frame and desktop GPUs (this engine
    // targets NVIDIA) have coherent caches across queues, so v1 leaves them exclusive.
    private void CreateField(int binding, ulong size, BufferUsageFlags extra = 0, bool crossQueue = false)
    {
        uint[]? families = crossQueue && Gfx.QueueFamilyIndices is { graphicsFamily: { } g, computeFamily: { } c }
            ? [g, c] : null;
        Gfx.CreateBuffer(size, BufferUsageFlags.StorageBufferBit | extra,
            MemoryPropertyFlags.DeviceLocalBit,
            out _set4Buf[binding], out _set4Alloc[binding], families, GpuMemoryAllocator.PriorityHigh);
    }

    private void FreeSet4()
    {
        for (int i = 0; i < Set4BindingCount; i++)
        {
            if (_set4Buf[i].Handle != 0) Gfx.DestroyBuffer(_set4Buf[i], _set4Alloc[i]);
            _set4Buf[i] = default; _set4Alloc[i] = default;
        }
    }

    /// <summary>Allocates the SoA working set at <paramref name="extent"/> + rewrites set 4, freeing
    /// any previous one. Marks the accumulator dirty (the fresh memory holds garbage). The only
    /// construction site - the first call comes from the boot-time resize.</summary>
    public void ReallocSet4(Extent2D extent)
    {
        FreeSet4();
        AllocSet4(PathCount(extent));
        MarkAccumulatorDirty();
        // The graph re-bakes the graph-owned set 2 from the fresh handles on the following rebuild.
    }


    // No pipeline-owned descriptor sets: scene (0), FeatureEnv (4) and FeaturePTIO (5) are
    // registry-owned, and the SoA working set (2) is graph-owned (see GraphSharedSpec).


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
            prefilteredCubeMipLevels = Renderer.PrefilteredCubeMipLevels,
            scaleIBLAmbient          = EditorState.IblIntensity,
            focusDistance            = FocusDistance,
            aperture                 = Aperture,
            paniniDistance           = PaniniDistance,
            verticalCompression      = VerticalCompression,
            emissiveTriCount         = Renderer.EmissiveTriangleCount,
            totalEmissivePower       = Renderer.TotalEmissivePower,
        };
        // Push once per frame; every BindSets this frame reuses the returned dynamic offset.
        // Safe here: the registry reset this frame's arena slice in BeginFrame, before Render.
        _frameConstants = Registry.ConstantArena.Push(frameIndex, _frameUbo);
        return reset;
    }


    // ---- Record helpers (the module's pass bodies call these) -----------------
    private void BindSets(CommandBuffer cmd, uint frameIndex)
    {
        uint frameConstants = _frameConstants;
        var registry = Registry;

        // Set 1 is a gap now (the IO pair moved to the registry), so scene and the graph-owned
        // working set bind separately instead of as one contiguous run.
        var sceneSet = registry.SceneSet(frameIndex);
        Vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute, Layout, ShaderSets.Scene, 1, &sceneSet, 1, &frameConstants);

        var shared = _sharedSet;        // SoA working set (set 2), graph-owned
        Vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute, Layout, ShaderSets.GraphShared, 1, &shared, 0, null);

        // envCube (FeatureEnv) and the accumulator / out-color pair (FeaturePTIO) each sit at their
        // own reflected index, with a gap at set 3.
        var envSet = registry.FeatureSet(FeatureEnv, frameIndex);
        Vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute, Layout, registry.FeatureSetIndex(FeatureEnv), 1, &envSet, 0, null);

        var ioSet = registry.FeatureSet(FeaturePtIo, frameIndex);
        Vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute, Layout, registry.FeatureSetIndex(FeaturePtIo), 1, &ioSet, 0, null);
    }

    private void PushWavefront(CommandBuffer cmd, uint bounce, uint argsClass)
    {
        var pc = new WavefrontPush { bounce = bounce, srcParity = bounce & 1u, argsClass = argsClass, maxBounces = MaxBounces };
        // Stages come from the reflected range the layout was built from -- vkCmdPushConstants
        // requires the two to agree, so neither side names a stage mask of its own.
        Vk.CmdPushConstants(cmd, Layout, PushConstantRanges[0].StageFlags, 0, (uint)sizeof(WavefrontPush), &pc);
    }

    /// <summary>Dense primary-ray generation (one PSO per camera mode).</summary>
    public void RecordGenerate(CommandBuffer cmd, in RenderView ctx)
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

    /// <summary>Megakernel tail: loop bounces [TailStartBounce, MaxBounces) for the bounce-cap
    /// survivors in one dispatch. Launched over the survivor count via extend-args[TailStartBounce]
    /// (the dims Shade(TailStartBounce-1)'s fused tail promoted); srcParity selects the survivor
    /// queue. The push bounce = TailStartBounce is the loop start; maxBounces is the loop end.</summary>
    public void RecordTail(CommandBuffer cmd, uint frameIndex)
    {
        Vk.CmdBindPipeline(cmd, PipelineBindPoint.Compute, _tailPso);
        BindSets(cmd, frameIndex);
        PushWavefront(cmd, TailStartBounce, 0u);
        Vk.CmdDispatchIndirect(cmd, DispatchArgsBuffer, ExtendArgsOffset(TailStartBounce));
    }

    public void RecordConnect(CommandBuffer cmd, uint frameIndex, uint bounce)
    {
        Vk.CmdBindPipeline(cmd, PipelineBindPoint.Compute, _connectPso);
        BindSets(cmd, frameIndex);
        PushWavefront(cmd, bounce, 0u);   // Connect ignores argsClass, but keeps the range covered
        // Launch from the bounce-parity connectArgs (NOT dispatchArgs); 16B stride.
        Vk.CmdDispatchIndirect(cmd,
            (bounce & 1u) == 0u ? ConnectArgsBuffer0 : ConnectArgsBuffer1, bounce * 16u);
    }

    /// <summary>Dense finalize: accumulate radiance, normalize into outColor.</summary>
    public void RecordFinalize(CommandBuffer cmd, in RenderView ctx)
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
        if (_tailPso.Handle     != 0) Vk.DestroyPipeline(Device, _tailPso, null);

        FreeSet4();
        Gfx.DestroyBuffer(_argsReadback.buffer, _argsReadback.alloc);   // TEMP/debug readback staging
        base.Dispose();   // destroys mode-0 PSO (= PipelineHandle), layout, owned DSLs, cache
    }
}