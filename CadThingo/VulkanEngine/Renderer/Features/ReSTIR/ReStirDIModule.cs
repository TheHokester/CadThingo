using CadThingo.VulkanEngine.Renderer.FrameGraph;
using CadThingo.VulkanEngine.Renderer.Pipelines;
using CadThingo.VulkanEngine.Renderer.Features.Tonemapping;
using Silk.NET.Vulkan;
using FrameContext = CadThingo.VulkanEngine.Renderer.Renderer.FrameContext;

namespace CadThingo.VulkanEngine.Renderer.Features.ReSTIR;

/// <summary>
/// ReSTIR DI as a composable graph module. This is the first RAY-TRACE pass in the frame graph
/// (PassType.RayTrace -> CmdTraceRays), proving the graph drives RT-pipeline work the same way it
/// drives compute/graphics: the pass declares its image writes and the compiler derives the Sync2
/// barriers + ordering, no hand-rolled transitions.
///
///   Trace (CmdTraceRays) -> Tonemap
///
/// P0: the Trace pass runs the unmodified ReStirDI tracer (byte-identical to the megakernel RT
/// path), writing the shared accumulator + out-color. The reservoir-init / temporal / spatial /
/// final-visibility passes land here in later phases -- each a new graph node (RT for the primary
/// trace, Compute w/ inline ray query for the reuse + final shade), which is exactly why ReSTIR
/// moved onto the graph now.
///
/// "Import for barriers, bind for access" (as in WavefrontPTModule): the tracer touches the
/// accumulator / out-color through the pipeline-owned descriptor sets; the graph imports the SAME
/// handles only so it can sequence Trace -> Tonemap from the declared usages. Tonemap is the
/// composed TonemapModule (HDR format parameterized to PtOutColor's R32F): its HDR-input set is
/// graph-baked from the imported out-color, so no host-side rebind happens on core switch.
/// </summary>
internal sealed class ReStirDIModule : IGraphModule<ReStirDIModule.Inputs, ReStirDIModule.Outputs>
{
    /// <summary>Host-owned targets the module imports: the PT accumulator + out-color (General,
    /// preserved across frames so progressive accumulation survives) and FinalColor (graph output).</summary>
    public readonly record struct Inputs(ImageResource Accumulator, ImageResource OutColor, ImageResource FinalColor);

    /// <summary>The written FinalColor handle, for the host to MarkOutput.</summary>
    public readonly record struct Outputs(GraphImage Final);

    private readonly ReStirDIPipeline _pipe;
    private readonly TonemapPipeline  _tonemap;

    public ReStirDIModule(ReStirDIPipeline pipe, TonemapPipeline tonemap)
    {
        _pipe = pipe;
        _tonemap = tonemap;
    }

    public void Build(GraphScope scope, in Inputs inp, out Outputs o)
    {
        // ---- Import the host-owned storage images (General, preserved) ----
        // FinalColor is NOT imported here: the composed TonemapModule below imports it.
        var accum = scope.ImportImage(inp.Accumulator.Image, inp.Accumulator.ImageView, default,
            ImageLayout.General, "ptAccum");
        // Full desc (not default) so TonemapModule's ExpectImage port check can validate it.
        var outColorDesc = new ImageDesc
        {
            Format = inp.OutColor._format, Mips = 1, Layers = 1,
            Usage = ImageUsageFlags.StorageBit | ImageUsageFlags.SampledBit,
        };
        var outColor = scope.ImportImage(inp.OutColor.Image, inp.OutColor.ImageView, in outColorDesc,
            ImageLayout.General, "ptOutColor");

        // ---- Import the pipeline-owned ping-pong reservoir + G-buffer buffers (P2 temporal) ----
        // The tracer reads PREV / writes CUR (parity-selected) through bound descriptors; the graph
        // imports the same handles so the Trace pass's declared writes give a per-frame barrier that
        // orders this frame's access after the PREVIOUS frame's write to the same physical buffer.
        var resA     = scope.ImportBuffer(_pipe.ReservoirA,    default, "reservoirA");
        var resB     = scope.ImportBuffer(_pipe.ReservoirB,    default, "reservoirB");
        var gbufA    = scope.ImportBuffer(_pipe.GBufferA,      default, "gbufferA");
        var gbufB    = scope.ImportBuffer(_pipe.GBufferB,      default, "gbufferB");
        var sceneRad = scope.ImportBuffer(_pipe.SceneRadiance, default, "sceneRadiance");

        // The set-2 working set is graph-owned: the graph allocates + writes it from the pipeline's
        // buffer handles, and the core hands the baked set back to the pipeline after Compile.
        scope.UseGraphSharedSet(_pipe.GraphSharedSpec);

        // ---- Trace (RT): path-trace to the fat G-buffer + sceneRadiance (indirect + emissive + env --
        // everything EXCEPT primary direct light). The reservoir build moved OUT of this megakernel
        // (BuildTemporal below) to relieve its register pressure / occupancy. Does NOT touch the
        // accumulator: shading is deferred to SpatialShade. ----
        scope.AddPass("Trace", PassType.RayTrace, QueueClass.Graphics,
            bld =>
            {
                gbufA    = bld.Write(gbufA,    ResourceUsage.StorageRT);
                gbufB    = bld.Write(gbufB,    ResourceUsage.StorageRT);
                sceneRad = bld.Write(sceneRad, ResourceUsage.StorageRT);
            },
            (CommandBuffer cmd, PassResources res, in FrameContext f) => _pipe.Record(cmd, f));

        // ---- BuildTemporal (compute): reconstruct the surface from the G-buffer Trace wrote, build
        // the unified DI reservoir (RIS) + temporally reuse the reprojected prev reservoir, write the
        // cur reservoir. Reads the cur+prev G-buffer halves (RAW after Trace), reads prev + writes cur
        // reservoir half (RW). Separate from spatial reuse: all reservoirs must be finalised before
        // SpatialShade reads neighbours. ----
        scope.AddPass("BuildTemporal", PassType.Compute, QueueClass.Graphics,
            bld =>
            {
                bld.Read(gbufA, ResourceUsage.StorageReadCompute);
                bld.Read(gbufB, ResourceUsage.StorageReadCompute);
                resA = bld.Write(resA, ResourceUsage.StorageRWCompute);
                resB = bld.Write(resB, ResourceUsage.StorageRWCompute);
            },
            (CommandBuffer cmd, PassResources res, in FrameContext f) => _pipe.RecordBuildTemporal(cmd, f));

        // ---- SpatialShade (compute): spatially reuse same-frame neighbour reservoirs, shade the
        // analytic sample (opaque shadow ray), add to sceneRadiance, fold into the accumulator.
        // Reads the reservoir/G-buffer/sceneRadiance Trace wrote (RAW barrier) + writes accum/out. ----
        scope.AddPass("SpatialShade", PassType.Compute, QueueClass.Graphics,
            bld =>
            {
                bld.Read(resA,     ResourceUsage.StorageReadCompute);
                bld.Read(resB,     ResourceUsage.StorageReadCompute);
                bld.Read(gbufA,    ResourceUsage.StorageReadCompute);
                bld.Read(gbufB,    ResourceUsage.StorageReadCompute);
                bld.Read(sceneRad, ResourceUsage.StorageReadCompute);
                accum    = bld.Write(accum,    ResourceUsage.StorageRWCompute);    // += progressive
                outColor = bld.Write(outColor, ResourceUsage.StorageWriteCompute);
            },
            (CommandBuffer cmd, PassResources res, in FrameContext f) => _pipe.RecordSpatialShade(cmd, f));

        // ---- Tonemap: composed TonemapModule; HDR-input set graph-baked from out-color. ----
        var tonemapModule = new TonemapModule(_tonemap, inp.OutColor._format);
        tonemapModule.Build(scope, new TonemapModule.Input(outColor, inp.FinalColor), out var tm);

        o = new Outputs(tm.FinalColor);
    }
}