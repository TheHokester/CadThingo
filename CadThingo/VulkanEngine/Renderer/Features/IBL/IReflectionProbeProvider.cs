using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Features.IBL;

public interface IReflectionProbeProvider
{ 
    uint MaxProbes { get; }
    uint ProbeCount { get; }
    uint ProbeMipLevels { get; }
    
    IReadOnlyList<ReflectionProbeComponent> Probes { get; }
    
    ProbeCapturePipeline CapturePipeline { get; }
    
    ProbeClusterGrid ClusterGrid { get; }

    void BuildClusters(RenderView view, float nearZ, float farZ,
        uint tileCountX, uint tileCountY, uint zSlices = 1);
    void RecordCapture(CommandBuffer cmd, RenderView view, Scene scene);
    
    void Tick(ulong frameIndex, Scene scene);
    void WriteProbeRecords(uint currentFrame);



}