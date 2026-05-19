namespace CadThingo.VulkanEngine.Renderer.Pipelines;

public class RayQueryPipeline : ComputePipeline
{
    public RayQueryPipeline(Renderer renderer) : base(renderer)
    {
    }


    protected override void CreateDescriptorSetLayouts()
    {
        throw new NotImplementedException();
    }

    protected override string ShaderPath { get; }
    
    
}