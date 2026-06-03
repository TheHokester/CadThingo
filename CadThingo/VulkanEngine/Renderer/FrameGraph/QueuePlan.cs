using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.FrameGraph;

/// <summary>
/// 1. Graphics = the universal family (graphics, compute, transfer)
/// 2. AsyncCompute = the compute family compute but not graphics
/// 3. Transfer = a family advertising transfer but not graphics or compute (a dedicated DMA queue)
/// 4. same handle guards <see cref ="HasRealAsyncCompute"/> and <see cref="HasDedicatedTransfer"/>
/// </summary>
public struct QueuePlan
{
    QueueRef? Graphics;
    QueueRef? AsyncCompute;
    QueueRef? Transfer;
    private bool HasRealAsyncCompute => AsyncCompute is not null;
    bool HasDedicatedTransfer => Transfer is not null;
}

struct QueueRef
{
    uint Family;
    uint Index;
    Queue Handle;
}

public enum QueueClass
{
    Graphics,
    AsyncCompute,
    Transfer,
}