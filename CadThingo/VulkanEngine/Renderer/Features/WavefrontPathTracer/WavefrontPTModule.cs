using CadThingo.VulkanEngine.Renderer.FrameGraph;
using CadThingo.VulkanEngine.Renderer.Pipelines;
using CadThingo.VulkanEngine.Renderer.Features.Tonemapping;
using Silk.NET.Vulkan;
using FrameContext = CadThingo.VulkanEngine.Renderer.Renderer.FrameContext;

namespace CadThingo.VulkanEngine.Renderer.Features.WavefrontPathTracer;

/// <summary>
/// The wavefront path tracer as a composable graph module (wavefront-pathtracer-impl.md P1.8).
/// Imports the pipeline-owned set-4 SoA working set + the host-owned accumulator / out-color /
/// FinalColor, then unrolls the indirect chain:
///
///   Generate -> [ Extend -> Shade -> Connect ] x MAX_BOUNCES -> Finalize -> Tonemap
///
/// Each stage's indirect-args generation is fused onto the tail of its producer (Generate writes
/// extend-args[0]; Extend's tail writes shade-args; Shade's tail writes connect-args + the next
/// bounce's extend-args), so there are no standalone PrepareArgs dispatches -- the last workgroup
/// to finish a producer writes the downstream launch dims (wfFuseTailElect in WavefrontBindings).
///
/// "Import for barriers, bind for access": the kernels touch the set-4 buffers + storage images
/// through the bound descriptor sets (pipeline-owned, written once); the graph imports the SAME
/// physical handles purely so it derives the Sync2 barriers + pass ordering from the declared
/// Read/Write usages. Every pass writes some imported resource, so none are culled; deep empty
/// bounces simply dispatch (0,1,1) via the shrinking indirect args.
///
/// The worker chain (Generate/Extend/Shade) is serialized through the shared `counters` buffer
/// (declared RW on each); Connect is DECOUPLED from it -- Connect(b) is declared between
/// Extend(b+1) and Shade(b+1) with an empty barrier batch so it overlaps the trace (see
/// AddConnectPass + the Extend pass's phantom declarations). Tonemap is a DIRECT pass --
/// not the TonemapModule, whose ExpectImage demands R16F while PtOutColor is R32F -- reading the
/// imported out-color (its HDR-input descriptor is re-pointed to PtOutColor in the core's Activate)
/// and writing the imported FinalColor.
/// </summary>
public sealed class WavefrontPTModule : IGraphModule<WavefrontPTModule.Inputs, WavefrontPTModule.Outputs>
{
    /// <summary>Host-owned targets the module imports: the PT accumulator + out-color (General,
    /// preserved across frames so accumulation survives) and FinalColor (the graph output).</summary>
    public readonly record struct Inputs(ImageResource Accumulator, ImageResource OutColor, ImageResource FinalColor);

    /// <summary>The written FinalColor handle, for the host to MarkOutput.</summary>
    public readonly record struct Outputs(GraphImage Final);

    // Fixes the unrolled pass list (~6/bounce + Generate/Finalize/Tonemap). The bounce-count knob
    // lives on the pipeline (it also sizes the per-bounce dispatchArgs buffer + readback); reference
    // it here so the unroll count and the buffer layout can never drift. Change it in WavefrontPTPipeline.
    private const uint MAX_BOUNCES = WavefrontPTPipeline.MaxBounces;
    // P3: material-sorted shading fans the shade stage out to C bins -- C dispatches recorded
    // inside ONE Shade pass per bounce (no barriers between the class dispatches; see the pass).
    private const uint SHADE_CLASSES = WavefrontPTPipeline.ShadeClasses;

    private readonly WavefrontPTPipeline _pipe;
    private readonly TonemapPipeline     _tonemap;

    public WavefrontPTModule(WavefrontPTPipeline pipe, TonemapPipeline tonemap)
    {
        _pipe = pipe;
        _tonemap = tonemap;
    }

    public void Build(GraphScope scope, in Inputs inp, out Outputs o)
    {
        // ---- Import the pipeline-owned SoA working set ----
        var psRayOrigin  = scope.ImportBuffer(_pipe.PsRayOrigin,  default, "psRayOrigin");
        var psRayDir     = scope.ImportBuffer(_pipe.PsRayDir,     default, "psRayDir");
        var psThroughput = scope.ImportBuffer(_pipe.PsThroughput, default, "psThroughput");
        var psRadiance   = scope.ImportBuffer(_pipe.PsRadiance,   default, "psRadiance");
        var psRng        = scope.ImportBuffer(_pipe.PsRng,        default, "psRng");
        var psSigmaA     = scope.ImportBuffer(_pipe.PsSigmaA,     default, "psSigmaA");   // P2.6 Beer-Lambert
        var hitRecPrim   = scope.ImportBuffer(_pipe.HitRecPrim,   default, "hitRecPrim");
        var hitT         = scope.ImportBuffer(_pipe.HitT,         default, "hitT");
        var hitBary      = scope.ImportBuffer(_pipe.HitBary,      default, "hitBary");
        var rayQ0        = scope.ImportBuffer(_pipe.RayQueue0,    default, "rayQueue0");
        var rayQ1        = scope.ImportBuffer(_pipe.RayQueue1,    default, "rayQueue1");
        var shadeQ       = scope.ImportBuffer(_pipe.ShadeQueue,   default, "shadeQueue");
        // Shadow records + connect-args ping-pong by bounce parity (bounce b uses set b&1):
        // Connect(b) reads its set while Shade(b+1) fills the other, so Connect's downstream
        // consumer is Shade(b+2) -- a full bounce of async overlap window.
        var shadowPath   = new[] { scope.ImportBuffer(_pipe.ShadowPath0,   default, "shadowPath0"),
                                   scope.ImportBuffer(_pipe.ShadowPath1,   default, "shadowPath1") };
        var shadowOrigin = new[] { scope.ImportBuffer(_pipe.ShadowOrigin0, default, "shadowOrigin0"),
                                   scope.ImportBuffer(_pipe.ShadowOrigin1, default, "shadowOrigin1") };
        var shadowDir    = new[] { scope.ImportBuffer(_pipe.ShadowDir0,    default, "shadowDir0"),
                                   scope.ImportBuffer(_pipe.ShadowDir1,    default, "shadowDir1") };
        var shadowLe     = new[] { scope.ImportBuffer(_pipe.ShadowLe0,     default, "shadowLe0"),
                                   scope.ImportBuffer(_pipe.ShadowLe1,     default, "shadowLe1") };
        var connectArgs  = new[] { scope.ImportBuffer(_pipe.ConnectArgsBuffer0, default, "connectArgs0"),
                                   scope.ImportBuffer(_pipe.ConnectArgsBuffer1, default, "connectArgs1") };
        var shadowRad    = scope.ImportBuffer(_pipe.ShadowRadiance, default, "shadowRadiance");
        var counters     = scope.ImportBuffer(_pipe.Counters,     default, "counters");
        var dispatchArgs = scope.ImportBuffer(_pipe.DispatchArgsBuffer, default, "dispatchArgs");

        // ---- Import the host-owned storage images (General, preserved) + FinalColor ----
        var accum = scope.ImportImage(inp.Accumulator.Image, inp.Accumulator.ImageView, default,
            ImageLayout.General, "ptAccum");
        var outColor = scope.ImportImage(inp.OutColor.Image, inp.OutColor.ImageView, default,
            ImageLayout.General, "ptOutColor");
        var finalDesc = new ImageDesc { Format = inp.FinalColor._format, Mips = 1, Layers = 1 };
        var final = scope.ImportImage(inp.FinalColor.Image, inp.FinalColor.ImageView, in finalDesc,
            ImageLayout.Undefined, "FinalColor", ImageLayout.ShaderReadOnlyOptimal);

        // ---- Generate: dense primary rays, seed rayQueue0 + counters + extend-args[0] ----
        // Generate also writes the first bounce's indirect launch dims (the old PrepExtend(0)
        // dispatch, fused in): N is dense + known, so no last-group election is needed here.
        scope.AddPass("Generate", PassType.Compute, QueueClass.Graphics,
            bld =>
            {
                psRayOrigin  = bld.Write(psRayOrigin,  ResourceUsage.StorageWriteCompute);
                psRayDir     = bld.Write(psRayDir,     ResourceUsage.StorageWriteCompute);
                psThroughput = bld.Write(psThroughput, ResourceUsage.StorageWriteCompute);
                psRadiance   = bld.Write(psRadiance,   ResourceUsage.StorageWriteCompute);
                psRng        = bld.Write(psRng,        ResourceUsage.StorageWriteCompute);
                psSigmaA     = bld.Write(psSigmaA,     ResourceUsage.StorageWriteCompute);   // init vacuum
                shadowRad    = bld.Write(shadowRad,    ResourceUsage.StorageWriteCompute);   // zero Connect's accumulator
                rayQ0        = bld.Write(rayQ0,        ResourceUsage.StorageWriteCompute);
                counters     = bld.Write(counters,     ResourceUsage.StorageWriteCompute);
                dispatchArgs = bld.Write(dispatchArgs, ResourceUsage.StorageWriteCompute);   // extend-args[0]
            },
            (CommandBuffer cmd, PassResources res, in FrameContext f) => _pipe.RecordGenerate(cmd, f));

        // Connect (2.5, DECOUPLED): fire the occlusion ray per shadow record; add shadowLe if
        // visible. Connect(cb) is declared BETWEEN Extend(cb+1) and Shade(cb+1) so it overlaps
        // the trace -- the incoherent, low-occupancy occlusion work fills in under Extend
        // instead of bubbling between two drains. Two mechanisms, picked by device support:
        //
        //  ASYNC (dedicated compute family): the pass runs on QueueClass.AsyncCompute. The
        //  graph chunks the schedule and rides timeline semaphores -- C(cb) waits Shade(cb)'s
        //  signal, Shade(cb+1) waits C(cb)'s -- giving driver-guaranteed concurrent SM
        //  residency with Extend(cb+1), which sits between those sync points on graphics.
        //
        //  FALLBACK (single queue): same-queue overlap by barrier omission -- Extend(cb+1)
        //  carries phantom declarations that absorb Connect's barriers, so Connect's batch is
        //  empty and its dispatch pipelines into Extend's occupancy tail.
        //
        // Either way Connect reads NOTHING Extend(cb+1) writes: thread gate from connectArgs
        // (not counters, whose SHADOW_COUNT Extend's tail zeroes mid-flight), RNG hash-derived
        // (not the psRng stream), and it is the sole psRadiance writer in the window (Extend
        // stashes env misses in psThroughput). Shade(cb+1) is ordered after it by WAR on the
        // shadow records. The LAST bounce's Connect has no Extend to hide under; it stays a
        // serial leaf (async still moves it off-queue, harmlessly).
        bool useAsync = scope.AsyncComputeAvailable;
        void AddConnectPass(uint cb)
        {
            int par = (int)(cb & 1u);
            scope.AddPass($"Connect{cb}", PassType.Compute,
                useAsync ? QueueClass.AsyncCompute : QueueClass.Graphics,
                bld =>
                {
                    bld.Read(shadowPath[par],   ResourceUsage.StorageReadCompute);
                    bld.Read(shadowOrigin[par], ResourceUsage.StorageReadCompute);
                    bld.Read(shadowDir[par],    ResourceUsage.StorageReadCompute);
                    bld.Read(shadowLe[par],     ResourceUsage.StorageReadCompute);
                    bld.Read(connectArgs[par],  ResourceUsage.IndirectArgStorageRead);   // dims + count word
                    // The accumulator. ASYNC: plain RW -- the WAW barrier this derives lands
                    // in Connect's own chunk on the COMPUTE queue, ordering C(b-1)->C(b)
                    // there (same-path += across bounces) without touching graphics.
                    // FALLBACK: concurrent, so Connect's batch stays empty and it pipelines
                    // into Extend's tail; the C(b-1)->C(b) visibility is re-published by
                    // Shade's phantom RW between them (see the Shade pass).
                    shadowRad = bld.Write(shadowRad, useAsync
                        ? ResourceUsage.StorageRWCompute
                        : ResourceUsage.StorageConcurrentCompute);
                },
                (CommandBuffer cmd, PassResources res, in FrameContext f) =>
                    _pipe.RecordConnect(cmd, f.FrameIndex, cb));
        }

        // Per bounce: Extend -> Connect(prev) -> Shade. The arg-prep that used to sit before each
        // worker (PrepExtend / PrepShade / PrepConnect, one 1-thread dispatch + two barriers
        // apiece) is FUSED onto the producer's tail (the last group to finish writes the
        // downstream stage's indirect args -- see Extend/Shade .slang + wfFuseTailElect). So
        // each Extend/Shade declares dispatchArgs as BOTH an indirect Read (its own launch dims,
        // written upstream) and a storage Write (the next stage's dims). The graph processes
        // reads before writes, so the derived barriers are: <upstream write> -> indirect-read,
        // then indirect-read -> storage-write; the next worker's indirect-read barrier then
        // publishes this tail's write. counters stays RW on Extend/Shade to keep the worker
        // chain linear; Connect no longer touches it (see AddConnectPass).
        // Megakernel tail: the wavefront chain only runs bounces [0, waveBounces); a single
        // TailMegakernel dispatch loops the rest per surviving path (see WavefrontPTPipeline.
        // TailStartBounce). Sparse deep bounces stop paying per-bounce dispatch/arg-gen/barrier cost.
        uint waveBounces = WavefrontPTPipeline.TailEnabled ? WavefrontPTPipeline.TailStartBounce : MAX_BOUNCES;

        for (uint b = 0; b < waveBounces; b++)
        {
            uint bb = b;                       // capture for the deferred execute closures
            GraphBuffer src = (bb % 2 == 0) ? rayQ0 : rayQ1;   // ping-pong read queue

            // Extend: trace the live rays; miss -> env stashed in psThroughput (w = -1 sentinel,
            // folded in Finalize -- keeps this pass OFF psRadiance so Connect(bb-1) can overlap);
            // hit -> append to shadeQueue. Fused tail: writes shade-args[bb], zeroes shadowCount
            // + nextRayCount (was PrepShade).
            scope.AddPass($"Extend{bb}", PassType.Compute, QueueClass.Graphics,
                bld =>
                {
                    bld.Read(src,          ResourceUsage.StorageReadCompute);
                    bld.Read(psRayOrigin,  ResourceUsage.StorageReadCompute);
                    bld.Read(psRayDir,     ResourceUsage.StorageReadCompute);
                    bld.Read(psSigmaA,     ResourceUsage.StorageReadCompute);   // P2.6 medium absorption
                    psThroughput = bld.Write(psThroughput, ResourceUsage.StorageRWCompute);   // Beer-Lambert + env stash
                    psRng      = bld.Write(psRng,      ResourceUsage.StorageRWCompute);   // BLEND coin flip advances it
                    hitRecPrim = bld.Write(hitRecPrim, ResourceUsage.StorageWriteCompute);
                    hitT       = bld.Write(hitT,       ResourceUsage.StorageWriteCompute);
                    hitBary    = bld.Write(hitBary,    ResourceUsage.StorageWriteCompute);
                    shadeQ     = bld.Write(shadeQ,     ResourceUsage.StorageWriteCompute);
                    counters   = bld.Write(counters,   ResourceUsage.StorageRWCompute);
                    bld.Read(dispatchArgs, ResourceUsage.IndirectArg);                       // own launch dims
                    dispatchArgs = bld.Write(dispatchArgs, ResourceUsage.StorageWriteCompute); // fused: shade-args[bb]

                    // SINGLE-QUEUE DECOUPLING (fallback only): absorb Connect(bb-1)'s barriers
                    // so its batch stays empty. The phantom reads (this kernel never touches
                    // the shadow records or connectArgs) place the RAW barriers vs
                    // Shade(bb-1)'s writes HERE, with dst scopes covering Connect's real
                    // accesses (compute reads + the indirect fetch); Connect then derives
                    // read-after-read = no barrier. The concurrent shadowRadiance write
                    // absorbs the RW->concurrent transition the same way (concurrent->
                    // concurrent on Connect is barrier-free). The kernel itself never
                    // accesses shadowRadiance. In ASYNC mode these MUST be omitted: a
                    // phantom write here would version-chain Connect(bb-1) AFTER this pass,
                    // turning into a cross-queue wait that serializes the very overlap
                    // async provides.
                    if (!useAsync)
                    {
                        shadowRad = bld.Write(shadowRad, ResourceUsage.StorageConcurrentCompute);
                        if (bb > 0)
                        {
                            int cpar = (int)((bb - 1u) & 1u);   // the set Connect(bb-1) reads
                            bld.Read(shadowPath[cpar],   ResourceUsage.StorageReadCompute);
                            bld.Read(shadowOrigin[cpar], ResourceUsage.StorageReadCompute);
                            bld.Read(shadowDir[cpar],    ResourceUsage.StorageReadCompute);
                            bld.Read(shadowLe[cpar],     ResourceUsage.StorageReadCompute);
                            bld.Read(connectArgs[cpar],  ResourceUsage.IndirectArgStorageRead);
                        }
                    }
                },
                (CommandBuffer cmd, PassResources res, in FrameContext f) =>
                    _pipe.RecordExtend(cmd, f.FrameIndex, bb));

            // Previous bounce's Connect, slotted here so it overlaps Extend(bb) (see above).
            if (bb > 0) AddConnectPass(bb - 1u);

            // Shade: ONE graph pass recording all C class dispatches back-to-back with NO internal
            // barriers -- the bins are data-independent (disjoint queue slices, disjoint counter
            // slots, and disjoint path sets, since Extend routes each path into exactly one bin),
            // so the dispatches may overlap on the GPU and the near-empty bins stop costing a full
            // pipeline drain apiece. Extend's tail published every class's indirect args, so the
            // single IndirectArg barrier covers all C dispatches. The fused tail (connect-args[bb]
            // + extend-args[bb+1], was PrepConnect + PrepExtend) is elected across ALL classes'
            // workgroups via the shared COMPLETED_WG ticket (numGroups = wfShadeTotalGroups()), so
            // it runs in whichever class's group arrives last -- by then SHADOW_COUNT /
            // NEXT_RAY_COUNT are summed across all classes. Empty bins still dispatch one group
            // (shade-args use max(1,..)) so the arrival count matches the launch count.
            scope.AddPass($"Shade{bb}", PassType.Compute, QueueClass.Graphics,
                bld =>
                {
                    bld.Read(shadeQ,     ResourceUsage.StorageReadCompute);
                    bld.Read(hitRecPrim, ResourceUsage.StorageReadCompute);
                    bld.Read(hitT,       ResourceUsage.StorageReadCompute);
                    bld.Read(hitBary,    ResourceUsage.StorageReadCompute);
                    psRng        = bld.Write(psRng,        ResourceUsage.StorageRWCompute);
                    psRayOrigin  = bld.Write(psRayOrigin,  ResourceUsage.StorageRWCompute);
                    psRayDir     = bld.Write(psRayDir,     ResourceUsage.StorageRWCompute);
                    psThroughput = bld.Write(psThroughput, ResourceUsage.StorageRWCompute);
                    psRadiance   = bld.Write(psRadiance,   ResourceUsage.StorageRWCompute);
                    psSigmaA     = bld.Write(psSigmaA,     ResourceUsage.StorageWriteCompute);   // P2.6 set on transmit
                    // Re-queue survivors into the OTHER ping-pong queue (dst = !src).
                    if (bb % 2 == 0) rayQ1 = bld.Write(rayQ1, ResourceUsage.StorageWriteCompute);
                    else             rayQ0 = bld.Write(rayQ0, ResourceUsage.StorageWriteCompute);
                    // This bounce's parity record set (Connect(bb-1) is still reading the
                    // OTHER set, so no WAR edge against it -- the wide-window point).
                    int spar = (int)(bb & 1u);
                    shadowPath[spar]   = bld.Write(shadowPath[spar],   ResourceUsage.StorageWriteCompute);
                    shadowOrigin[spar] = bld.Write(shadowOrigin[spar], ResourceUsage.StorageWriteCompute);
                    shadowDir[spar]    = bld.Write(shadowDir[spar],    ResourceUsage.StorageWriteCompute);
                    shadowLe[spar]     = bld.Write(shadowLe[spar],     ResourceUsage.StorageWriteCompute);
                    counters     = bld.Write(counters,     ResourceUsage.StorageRWCompute);
                    bld.Read(dispatchArgs, ResourceUsage.IndirectArg);                            // all classes' launch dims
                    dispatchArgs = bld.Write(dispatchArgs, ResourceUsage.StorageWriteCompute);    // fused tail: extend-args[bb+1] + debug connect slot
                    connectArgs[spar] = bld.Write(connectArgs[spar], ResourceUsage.StorageWriteCompute); // fused tail: connect dims + record count
                    // FALLBACK only: phantom plain-RW touch of the accumulator. It re-
                    // publishes Connect(bb-1)'s concurrent writes (the conc->RW transition
                    // barriers HERE, where a batch already exists) so Connect(bb)'s same-path
                    // += sees them. ASYNC mode must omit it -- it would be a WAW edge
                    // C(bb-1)->S(bb) = a cross-queue wait that re-narrows the window; there
                    // the Connects order among themselves on the compute queue instead.
                    if (!useAsync)
                        shadowRad = bld.Write(shadowRad, ResourceUsage.StorageRWCompute);
                },
                (CommandBuffer cmd, PassResources res, in FrameContext f) =>
                {
                    for (uint c = 0; c < SHADE_CLASSES; c++)
                        _pipe.RecordShade(cmd, f.FrameIndex, bb, c);
                });
        }

        // The last wavefront bounce's Connect (its deferred NEE). With the megakernel tail this is
        // Connect(waveBounces-1); the tail does its own NEE inline, so no further Connect follows.
        AddConnectPass(waveBounces - 1u);

        // Megakernel tail: continue the bounce-cap survivors through bounces [waveBounces, MAX_BOUNCES)
        // in one dispatch (loops in-thread; inline NEE). Reads the survivor queue + live SoA state
        // Shade(waveBounces-1) left, folds all remaining radiance into psRadiance for Finalize.
        // Launched over RAY_COUNT via extend-args[waveBounces] (Shade's fused tail promoted both).
        if (WavefrontPTPipeline.TailEnabled)
        {
            uint start = waveBounces;
            GraphBuffer survivors = (start % 2 == 0) ? rayQ0 : rayQ1;   // rayQueue[start&1] = Shade(start-1)'s dst
            scope.AddPass("TailMegakernel", PassType.Compute, QueueClass.Graphics,
                bld =>
                {
                    bld.Read(survivors,    ResourceUsage.StorageReadCompute);
                    bld.Read(psRayOrigin,  ResourceUsage.StorageReadCompute);
                    bld.Read(psRayDir,     ResourceUsage.StorageReadCompute);
                    bld.Read(psThroughput, ResourceUsage.StorageReadCompute);
                    bld.Read(psSigmaA,     ResourceUsage.StorageReadCompute);
                    bld.Read(counters,     ResourceUsage.StorageReadCompute);   // RAY_COUNT thread gate
                    bld.Read(dispatchArgs, ResourceUsage.IndirectArg);          // extend-args[start] dims
                    psRadiance = bld.Write(psRadiance, ResourceUsage.StorageRWCompute);   // continue accumulation
                    psRng      = bld.Write(psRng,      ResourceUsage.StorageRWCompute);   // advances the stream
                },
                (CommandBuffer cmd, PassResources res, in FrameContext f) => _pipe.RecordTail(cmd, f.FrameIndex));
        }

        // ---- Finalize: accumulate radiance, normalize into out-color ----
        scope.AddPass("Finalize", PassType.Compute, QueueClass.Graphics,
            bld =>
            {
                bld.Read(psRadiance,   ResourceUsage.StorageReadCompute);
                bld.Read(psThroughput, ResourceUsage.StorageReadCompute);   // escaped-path env stash (w = -1)
                bld.Read(shadowRad,    ResourceUsage.StorageReadCompute);   // Connect's NEE accumulator
                accum    = bld.Write(accum,    ResourceUsage.StorageRWCompute);
                outColor = bld.Write(outColor, ResourceUsage.StorageWriteCompute);
            },
            (CommandBuffer cmd, PassResources res, in FrameContext f) => _pipe.RecordFinalize(cmd, f));

        // ---- Tonemap: direct pass (out-color -> FinalColor). The tonemap pipeline samples
        // PtOutColor through its own HDR-input descriptor (bound to PtOutColor in the core's
        // Activate); the graph import of outColor here is only for barrier derivation. ----
        scope.AddPass("Tonemap", PassType.Graphics, QueueClass.Graphics,
            bld =>
            {
                bld.Read(outColor, ResourceUsage.SampledFragment);
                final = bld.Write(final, ResourceUsage.ColorAttachment);
            },
            (CommandBuffer cmd, PassResources res, in FrameContext f) =>
                _tonemap.Record(cmd, f, res.View(final)));

        o = new Outputs(final);
    }
}