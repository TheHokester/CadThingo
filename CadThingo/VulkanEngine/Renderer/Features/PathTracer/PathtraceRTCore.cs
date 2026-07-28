using CadThingo.VulkanEngine.Renderer.FrameGraph;
using CadThingo.VulkanEngine.Renderer.Pipelines;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Features.PathTracer;

/// <summary>
/// Progressive path tracer driven by the RT pipeline / CmdTraceRays (RenderMode.RayTrace). A thin
/// <see cref="PathTraceCoreBase"/> subclass binding the renderer-owned <see cref="RTPipeline"/>;
/// the Trace pass runs as a graph RayTrace pass. Only constructed when the device exposes the RT
/// pipeline (<c>rtPipeline != null</c>); the host falls back to the deferred core otherwise.
/// </summary>
internal sealed class PathtraceRTCore : PathTraceCoreBase
{
    private readonly RTPipeline _pipeline;

    public PathtraceRTCore(Renderer host) : base(host)
    {
        _pipeline = host.rtPipeline!;
        BuildGraph();
    }

    public override string Name => "PathTrace (RT pipeline)";
    public override Renderer.RenderMode Mode => Renderer.RenderMode.RayTrace;

    protected override PassType TracePassType => PassType.RayTrace;
    protected override void PipelineMarkAccumulatorDirty() => _pipeline.MarkAccumulatorDirty();
    protected override bool PipelineUpdatePerFrame(uint frameIndex, Camera camera, uint lightCount, Extent2D renderExtent) =>
        _pipeline.UpdatePerFrame(frameIndex, camera, lightCount, renderExtent);
    protected override void PipelineRecord(CommandBuffer cmd, in RenderView ctx) =>
        _pipeline.Record(cmd, ctx);
}