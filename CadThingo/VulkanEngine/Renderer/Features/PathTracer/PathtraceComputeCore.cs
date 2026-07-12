using CadThingo.VulkanEngine.Renderer.FrameGraph;
using CadThingo.VulkanEngine.Renderer.Pipelines;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Features.PathTracer;

/// <summary>
/// Progressive ray-query path tracer driven by a compute dispatch (RenderMode.RayCompute).
/// A thin <see cref="PathTraceCoreBase"/> subclass binding the renderer-owned
/// <see cref="PTComputePipeline"/>; the Trace pass runs as a graph compute pass.
/// </summary>
internal sealed class PathtraceComputeCore : PathTraceCoreBase
{
    private readonly PTComputePipeline _pipeline;

    public PathtraceComputeCore(Renderer host) : base(host)
    {
        _pipeline = host.ptComputePipeline;
        BuildGraph();
    }

    public override string Name => "PathTrace (Compute)";
    public override Renderer.RenderMode Mode => Renderer.RenderMode.RayCompute;

    protected override PassType TracePassType => PassType.Compute;
    protected override void PipelineMarkAccumulatorDirty() => _pipeline.MarkAccumulatorDirty();
    protected override bool PipelineUpdatePerFrame(uint frameIndex, Camera camera, uint lightCount, Extent2D renderExtent) =>
        _pipeline.UpdatePerFrame(frameIndex, camera, lightCount, renderExtent);
    protected override void PipelineRecord(CommandBuffer cmd, in Renderer.FrameContext ctx) =>
        _pipeline.Record(cmd, ctx);
    protected override void PipelineWriteStorageImages(ImageView accumView, ImageView outColorView) =>
        _pipeline.WriteStorageImageDescriptors(accumView, outColorView);
}