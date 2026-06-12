using CadThingo.VulkanEngine.Renderer.FrameGraph;
using CadThingo.VulkanEngine.Renderer.Pipelines;
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
/// The bounce body is serialized through the shared `counters` buffer (declared RW on every pass),
/// which over-orders for P1 simplicity (correctness over pass count). Tonemap is a DIRECT pass --
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
    // P3: material-sorted shading fans the shade stage out to C bins, one Shade pass per class.
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
        var shadowPath   = scope.ImportBuffer(_pipe.ShadowPath,   default, "shadowPath");
        var shadowOrigin = scope.ImportBuffer(_pipe.ShadowOrigin, default, "shadowOrigin");
        var shadowDir    = scope.ImportBuffer(_pipe.ShadowDir,    default, "shadowDir");
        var shadowLe     = scope.ImportBuffer(_pipe.ShadowLe,     default, "shadowLe");
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
                rayQ0        = bld.Write(rayQ0,        ResourceUsage.StorageWriteCompute);
                counters     = bld.Write(counters,     ResourceUsage.StorageWriteCompute);
                dispatchArgs = bld.Write(dispatchArgs, ResourceUsage.StorageWriteCompute);   // extend-args[0]
            },
            (CommandBuffer cmd, PassResources res, in FrameContext f) => _pipe.RecordGenerate(cmd, f));

        // Per bounce: Extend -> Shade -> Connect. The arg-prep that used to sit before each
        // worker (PrepExtend / PrepShade / PrepConnect, one 1-thread dispatch + two barriers
        // apiece) is now FUSED onto the producer's tail (the last group to finish writes the
        // downstream stage's indirect args -- see Extend/Shade .slang + wfFuseTailElect). So
        // each Extend/Shade declares dispatchArgs as BOTH an indirect Read (its own launch dims,
        // written upstream) and a storage Write (the next stage's dims). The graph processes
        // reads before writes, so the derived barriers are: <upstream write> -> indirect-read,
        // then indirect-read -> storage-write; the next worker's indirect-read barrier then
        // publishes this tail's write. counters stays RW on every pass to keep the chain linear.
        for (uint b = 0; b < MAX_BOUNCES; b++)
        {
            uint bb = b;                       // capture for the deferred execute closures
            GraphBuffer src = (bb % 2 == 0) ? rayQ0 : rayQ1;   // ping-pong read queue

            // Extend: trace the live rays; miss -> radiance += env; hit -> append to shadeQueue.
            // Fused tail: writes shade-args[bb], zeroes shadowCount + nextRayCount (was PrepShade).
            scope.AddPass($"Extend{bb}", PassType.Compute, QueueClass.Graphics,
                bld =>
                {
                    bld.Read(src,          ResourceUsage.StorageReadCompute);
                    bld.Read(psRayOrigin,  ResourceUsage.StorageReadCompute);
                    bld.Read(psRayDir,     ResourceUsage.StorageReadCompute);
                    bld.Read(psSigmaA,     ResourceUsage.StorageReadCompute);   // P2.6 medium absorption
                    psThroughput = bld.Write(psThroughput, ResourceUsage.StorageRWCompute);   // Beer-Lambert multiply on hit
                    psRng      = bld.Write(psRng,      ResourceUsage.StorageRWCompute);   // BLEND coin flip advances it
                    psRadiance = bld.Write(psRadiance, ResourceUsage.StorageRWCompute);
                    hitRecPrim = bld.Write(hitRecPrim, ResourceUsage.StorageWriteCompute);
                    hitT       = bld.Write(hitT,       ResourceUsage.StorageWriteCompute);
                    hitBary    = bld.Write(hitBary,    ResourceUsage.StorageWriteCompute);
                    shadeQ     = bld.Write(shadeQ,     ResourceUsage.StorageWriteCompute);
                    counters   = bld.Write(counters,   ResourceUsage.StorageRWCompute);
                    bld.Read(dispatchArgs, ResourceUsage.IndirectArg);                       // own launch dims
                    dispatchArgs = bld.Write(dispatchArgs, ResourceUsage.StorageWriteCompute); // fused: shade-args[bb]
                },
                (CommandBuffer cmd, PassResources res, in FrameContext f) =>
                    _pipe.RecordExtend(cmd, f.FrameIndex, bb));

            // Shade: one pass per material class (P3.5). Each shades its own queue bin (selected by
            // the push class) + the shared hit*, and writes nextRay*/shadow*/psRadiance through the
            // SAME handles, so the graph serializes the C passes via WAW on counters / the dst queue /
            // the shadow + persistent buffers. Empty bins still dispatch one group (shade-args use
            // max(1,..)) -- needed because the LAST class carries the fused tail (connect-args[bb] +
            // extend-args[bb+1]); by then SHADOW_COUNT / NEXT_RAY_COUNT are summed across all classes.
            // P3a binds the FULL PSO to every class (RecordShade); P3b will bake per-class PSOs.
            for (uint c = 0; c < SHADE_CLASSES; c++)
            {
                uint cc        = c;
                bool lastClass = cc == SHADE_CLASSES - 1u;
                scope.AddPass($"Shade{bb}_c{cc}", PassType.Compute, QueueClass.Graphics,
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
                        shadowPath   = bld.Write(shadowPath,   ResourceUsage.StorageWriteCompute);
                        shadowOrigin = bld.Write(shadowOrigin, ResourceUsage.StorageWriteCompute);
                        shadowDir    = bld.Write(shadowDir,    ResourceUsage.StorageWriteCompute);
                        shadowLe     = bld.Write(shadowLe,     ResourceUsage.StorageWriteCompute);
                        counters     = bld.Write(counters,     ResourceUsage.StorageRWCompute);
                        bld.Read(dispatchArgs, ResourceUsage.IndirectArg);                        // own class's launch dims
                        if (lastClass)                                                            // fused tail on the last class only
                            dispatchArgs = bld.Write(dispatchArgs, ResourceUsage.StorageWriteCompute); // connect-args[bb] + extend-args[bb+1]
                    },
                    (CommandBuffer cmd, PassResources res, in FrameContext f) =>
                        _pipe.RecordShade(cmd, f.FrameIndex, bb, cc));
            }

            // Connect (2.5): fire the occlusion ray per shadow record; add shadowLe if visible.
            // No fused tail (leaf of the bounce); reads its own connect-args, counters RW links it.
            scope.AddPass($"Connect{bb}", PassType.Compute, QueueClass.Graphics,
                bld =>
                {
                    bld.Read(shadowPath,   ResourceUsage.StorageReadCompute);
                    bld.Read(shadowOrigin, ResourceUsage.StorageReadCompute);
                    bld.Read(shadowDir,    ResourceUsage.StorageReadCompute);
                    bld.Read(shadowLe,     ResourceUsage.StorageReadCompute);
                    psRng      = bld.Write(psRng,      ResourceUsage.StorageRWCompute);   // isOccluded alpha gate
                    psRadiance = bld.Write(psRadiance, ResourceUsage.StorageRWCompute);
                    counters   = bld.Write(counters,   ResourceUsage.StorageRWCompute);   // linearize the chain
                    bld.Read(dispatchArgs, ResourceUsage.IndirectArg);
                },
                (CommandBuffer cmd, PassResources res, in FrameContext f) =>
                    _pipe.RecordConnect(cmd, f.FrameIndex, bb));
        }

        // ---- Finalize: accumulate radiance, normalize into out-color ----
        scope.AddPass("Finalize", PassType.Compute, QueueClass.Graphics,
            bld =>
            {
                bld.Read(psRadiance, ResourceUsage.StorageReadCompute);
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