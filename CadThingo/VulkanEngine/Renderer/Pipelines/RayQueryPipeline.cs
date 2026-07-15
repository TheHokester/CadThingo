namespace CadThingo.VulkanEngine.Renderer.Pipelines;

public class RayQueryPipeline : ComputePipeline
{
    public RayQueryPipeline(in GpuContext gpu, Renderer renderer) : base(gpu, renderer)
    {
    }


    protected override void CreateDescriptorSetLayouts()
    {
        throw new NotImplementedException();
    }

    protected override string ShaderPath { get; }
    
    
}