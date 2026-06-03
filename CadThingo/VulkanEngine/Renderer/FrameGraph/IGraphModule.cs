namespace CadThingo.VulkanEngine.Renderer.FrameGraph;

public interface IGraphModule<TInputs, TOutputs>
{
    void Build(GraphBuilder b, in TInputs inputs, in TOutputs outputs);
}