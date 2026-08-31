using System.Runtime.CompilerServices;
using CadThingo.VulkanEngine.Renderer.FeatureLifecycle;
using CadThingo.VulkanEngine.Renderer.FeatureLifecycle.RenderFeatureInterfaces;
using CadThingo.VulkanEngine.Renderer.FrameGraph;
using CadThingo.VulkanEngine.Renderer.Pipelines;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Features.PathTracer;

/// <summary>
/// Progressive ray-query path tracer driven by a compute dispatch (RenderMode.RayCompute).
/// A thin <see cref="PathTraceCoreBase"/> subclass binding the renderer-owned
/// <see cref="PTComputePipeline"/>; the Trace pass runs as a graph compute pass.
/// </summary>
internal sealed class PathtraceComputeCore : PathTraceCoreBase,
                                             ISelfRegisteringFeature<PathtraceComputeCore>
{
    // Ungated: the ray-query compute path is built unconditionally today, matching the pipeline it
    // binds. If that pipeline ever becomes conditional, this gate is where it says so.
    public static FeatureDesc Desc =>
        new(Order: 20, Gate: _ => true, Make: () => new PathtraceComputeCore());

    [ModuleInitializer]
    internal static void _Reg() => FeatureCatalog.Register<PathtraceComputeCore>();

    private PTComputePipeline _pipeline = null!;

    public override string Name => "PathTrace (Compute)";
    public override Renderer.RenderMode Mode => Renderer.RenderMode.RayCompute;

    /// <summary>Exposed only for the settings / viewport panels, which read its sample counters.</summary>
    internal PTComputePipeline Pipeline => _pipeline;

    protected override void CreatePipeline()
    {
        // Scene buffers (TLAS / lights / shadow info / vb+ib / emissive / bindless) come from the
        // scene set; the accumulator / out-color pair from the registry-owned FeaturePTIO set.
        _pipeline = new PTComputePipeline(_gpu, _ibl, _pt);
        _pipeline.Initialize();
    }

    protected override void DestroyPipeline() => _pipeline?.Dispose();

    protected override PassType TracePassType => PassType.Compute;
    protected override void PipelineMarkAccumulatorDirty() => _pipeline.MarkAccumulatorDirty();
    protected override bool PipelineUpdatePerFrame(uint frameIndex, Camera camera, uint lightCount, Extent2D renderExtent) =>
        _pipeline.UpdatePerFrame(frameIndex, camera, lightCount, renderExtent);
    protected override void PipelineRecord(CommandBuffer cmd, in RenderView ctx) =>
        _pipeline.Record(cmd, ctx);
}
