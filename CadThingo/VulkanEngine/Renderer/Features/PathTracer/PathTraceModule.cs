using CadThingo.VulkanEngine.Renderer.FrameGraph;
using CadThingo.VulkanEngine.Renderer.Features.Tonemapping;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Features.PathTracer;

/// <summary>
/// The megakernel path tracers as a composable graph module: one Trace pass (compute ray-query or
/// RT-pipeline CmdTraceRays, selected by <c>traceType</c>) writing the imported accumulator +
/// out-color, then the composed TonemapModule (HDR-input set graph-baked from out-color).
/// "Import for barriers, bind for access" as in ReStirDIModule: the tracer touches the images
/// through its pipeline-owned IO set; the imports exist only so the barriers are derived.
/// </summary>
internal sealed class PathTraceModule : IGraphModule<PathTraceModule.Inputs, PathTraceModule.Outputs>
{
    /// <summary>Host-owned targets the module imports: the PT accumulator + out-color (General,
    /// preserved across frames so progressive accumulation survives) and FinalColor (graph output).</summary>
    public readonly record struct Inputs(ImageResource Accumulator, ImageResource OutColor, ImageResource FinalColor);

    /// <summary>The written FinalColor handle, for the host to MarkOutput.</summary>
    public readonly record struct Outputs(GraphImage Final);

    private readonly TonemapPipeline _tonemap;
    private readonly PassType        _traceType;
    private readonly PassExecute     _recordTrace;

    public PathTraceModule(TonemapPipeline tonemap, PassType traceType, PassExecute recordTrace)
    {
        _tonemap     = tonemap;
        _traceType   = traceType;
        _recordTrace = recordTrace;
    }

    public void Build(GraphScope scope, in Inputs inp, out Outputs o)
    {
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

        // The usage stage must match the recorded work so the derived barriers cover it.
        bool rt = _traceType == PassType.RayTrace;
        scope.AddPass("Trace", _traceType, QueueClass.Graphics,
            bld =>
            {
                accum    = bld.Write(accum,    rt ? ResourceUsage.StorageRT : ResourceUsage.StorageRWCompute);
                outColor = bld.Write(outColor, rt ? ResourceUsage.StorageRT : ResourceUsage.StorageWriteCompute);
            },
            _recordTrace);

        var tonemapModule = new TonemapModule(_tonemap, inp.OutColor._format);
        tonemapModule.Build(scope, new TonemapModule.Input(outColor, inp.FinalColor), out var tm);

        o = new Outputs(tm.FinalColor);
    }
}