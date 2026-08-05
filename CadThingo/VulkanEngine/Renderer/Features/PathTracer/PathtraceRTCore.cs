using System.Runtime.CompilerServices;
using CadThingo.VulkanEngine.Renderer.FeatureLifecycle;
using CadThingo.VulkanEngine.Renderer.FeatureLifecycle.RenderFeatureInterfaces;
using CadThingo.VulkanEngine.Renderer.FrameGraph;
using CadThingo.VulkanEngine.Renderer.Pipelines;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Features.PathTracer;

/// <summary>
/// Progressive path tracer driven by the RT pipeline / CmdTraceRays (RenderMode.RayTrace). A thin
/// <see cref="PathTraceCoreBase"/> subclass binding the renderer-owned <see cref="RTPipeline"/>;
/// the Trace pass runs as a graph RayTrace pass. Gated on the RT pipeline being available - on a
/// device without it this core is never constructed and never appears in the mode combo.
/// </summary>
internal sealed class PathtraceRTCore : PathTraceCoreBase,
                                        ISelfRegisteringFeature<PathtraceRTCore>
{
    // Same condition the host used to build rtPipeline under, now stated once, here.
    public static FeatureDesc Desc =>
        new(Order: 30,
            Gate: gpu => gpu.Gfx.RayTracePipelineSupported,
            Make: () => new PathtraceRTCore());

    [ModuleInitializer]
    internal static void _Reg() => FeatureCatalog.Register<PathtraceRTCore>();

    private RTPipeline _pipeline = null!;

    public override string Name => "PathTrace (RT pipeline)";
    public override Renderer.RenderMode Mode => Renderer.RenderMode.RayTrace;

    protected override void CreatePipeline()
    {
        // Shares the accumulator / out-color images + scene buffers with the compute path; envCube
        // rides FeatureEnv. The gate above is what makes the unconditional construction safe.
        _pipeline = new RTPipeline(_gpu, _ibl, _pt);
        _pipeline.Initialize();
    }

    protected override void DestroyPipeline() => _pipeline?.Dispose();

    protected override PassType TracePassType => PassType.RayTrace;
    protected override void PipelineMarkAccumulatorDirty() => _pipeline.MarkAccumulatorDirty();
    protected override bool PipelineUpdatePerFrame(uint frameIndex, Camera camera, uint lightCount, Extent2D renderExtent) =>
        _pipeline.UpdatePerFrame(frameIndex, camera, lightCount, renderExtent);
    protected override void PipelineRecord(CommandBuffer cmd, in RenderView ctx) =>
        _pipeline.Record(cmd, ctx);
}
