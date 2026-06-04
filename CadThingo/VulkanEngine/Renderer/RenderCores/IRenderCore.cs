using Silk.NET.Vulkan;
using FrameGraph = CadThingo.VulkanEngine.Renderer.FrameGraph.FrameGraph;

namespace CadThingo.VulkanEngine.Renderer.RenderCores;

readonly ref struct RenderFrame
{
    public readonly CommandBuffer cmd;
    public readonly RenderView View;
    public readonly ImageResource SceneColorHDR;
}

internal interface IRenderCore
{
    internal string Name { get; }
    
    FrameGraph.FrameGraph FrameGraph { get; }
    void Initialize(GraphicsDevice gfx, ImageResource output);
    void Resize(Extent2D extent);

    public void Render(in RenderFrame frame)
    {
        FrameGraph.Execute(frame.cmd, new Renderer.FrameContext());
    }
    public void BuildFrameGraph();
}