using CadThingo.VulkanEngine.Renderer.Pipelines;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Features.PathTracer;

/// <summary>
/// Progressive ray-query path tracer driven by a compute dispatch (RenderMode.RayCompute).
/// A thin <see cref="PathTraceCoreBase"/> subclass binding the renderer-owned
/// <see cref="PTComputePipeline"/>; the writer stage is the compute shader.
/// </summary>
internal sealed class PathtraceComputeCore(Renderer host) : PathTraceCoreBase(host)
{
    private readonly PTComputePipeline _pipeline = host.ptComputePipeline;

    public override string Name => "PathTrace (Compute)";
    public override Renderer.RenderMode Mode => Renderer.RenderMode.RayCompute;

    protected override PipelineStageFlags WriterStage => PipelineStageFlags.ComputeShaderBit;
    protected override void PipelineMarkAccumulatorDirty() => _pipeline.MarkAccumulatorDirty();
    protected override bool PipelineUpdatePerFrame(uint frameIndex, Camera camera, uint lightCount, Extent2D renderExtent) =>
        _pipeline.UpdatePerFrame(frameIndex, camera, lightCount, renderExtent);
    protected override void PipelineRecord(CommandBuffer cmd, in Renderer.FrameContext ctx) =>
        _pipeline.Record(cmd, ctx);
    protected override void PipelineWriteStorageImages(ImageView accumView, ImageView outColorView) =>
        _pipeline.WriteStorageImageDescriptors(accumView, outColorView);
}