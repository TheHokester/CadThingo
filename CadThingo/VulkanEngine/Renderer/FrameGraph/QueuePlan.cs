using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.FrameGraph;

/// <summary>
/// The graph's view of the device's queues, resolved once per graph from
/// <see cref="GraphicsDevice"/>:
/// 1. Graphics = the universal family (graphics, compute, transfer) -- always present.
/// 2. AsyncCompute = the dedicated compute family (compute but not graphics). The device's
///    FindQueueFamilies resolves computeFamily to exactly that, falling back to the graphics
///    family when none exists; the same-handle guard here turns that fallback into
///    <c>AsyncCompute = null</c> so callers degrade to single-queue + barriers instead of
///    pretending a second queue exists.
/// 3. Transfer = a family advertising transfer but not graphics or compute (the DMA engine).
///    Resolved the same way; currently unused by the graph (Phase 2 transfer uploads).
/// </summary>
public struct QueuePlan
{
    public QueueRef  Graphics;
    public QueueRef? AsyncCompute;
    public QueueRef? Transfer;

    public bool HasRealAsyncCompute  => AsyncCompute is not null;
    public bool HasDedicatedTransfer => Transfer is not null;

    public static QueuePlan Resolve(GraphicsDevice dev)
    {
        var idx = dev.QueueFamilyIndices;
        var plan = new QueuePlan
        {
            Graphics = new QueueRef(idx.graphicsFamily!.Value, 0, dev.GraphicsQueue),
        };

        if (idx.computeFamily is { } cf && cf != plan.Graphics.Family)
            plan.AsyncCompute = new QueueRef(cf, 0, dev.ComputeQueue);

        if (idx.transferFamily is { } tf && tf != plan.Graphics.Family)
            plan.Transfer = new QueueRef(tf, 0, dev.TransferQueue);

        return plan;
    }
}

public readonly struct QueueRef(uint family, uint index, Queue handle)
{
    public readonly uint  Family = family;
    public readonly uint  Index  = index;
    public readonly Queue Handle = handle;
}

public enum QueueClass
{
    Graphics,
    AsyncCompute,
    Transfer,
}