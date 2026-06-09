using CadThingo.VulkanEngine.Renderer.Pipelines;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Features.PathTracer;

/// <summary>
/// Progressive path tracer driven by the RT pipeline / CmdTraceRays (RenderMode.RayTrace). A thin
/// <see cref="PathTraceCoreBase"/> subclass binding the renderer-owned <see cref="RTPipeline"/>;
/// the writer stage is the ray-tracing shader. Only constructed when the device exposes the RT
/// pipeline (<c>rtPipeline != null</c>); the host falls back to the deferred core otherwise.
/// </summary>
internal sealed class PathtraceRTCore : PathTraceCoreBase
{
    private readonly RTPipeline _pipeline;

    public PathtraceRTCore(Renderer host) : base(host) => _pipeline = host.rtPipeline!;

    public override string Name => "PathTrace (RT pipeline)";
    public override Renderer.RenderMode Mode => Renderer.RenderMode.RayTrace;

    protected override PipelineStageFlags WriterStage => PipelineStageFlags.RayTracingShaderBitKhr;
    protected override void PipelineMarkAccumulatorDirty() => _pipeline.MarkAccumulatorDirty();
    protected override bool PipelineUpdatePerFrame(uint frameIndex, Camera camera, uint lightCount, Extent2D renderExtent) =>
        _pipeline.UpdatePerFrame(frameIndex, camera, lightCount, renderExtent);
    protected override void PipelineRecord(CommandBuffer cmd, in Renderer.FrameContext ctx) =>
        _pipeline.Record(cmd, ctx);
    protected override void PipelineWriteStorageImages(ImageView accumView, ImageView outColorView) =>
        _pipeline.WriteStorageImageDescriptors(accumView, outColorView);
}