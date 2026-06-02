namespace CadThingo.VulkanEngine.Renderer.FrameGraph;

public interface IGraphModule<TInputs, TOutputs>
{
    void Build(IGraphBuilder b, in TInputs inputs, in TOutputs outputs);
}