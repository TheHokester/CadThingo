using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.RenderGraph;

public unsafe struct Pass
{
    public string _name;
    public List<string> _inputs;//names of the resources this pass reads from
    public List<string> _outputs;//names of the resources this pass writes to

    public Pass(string name, List<string> inputs, List<string> outputs)
    {
        _name = name;
        _inputs = inputs ?? new List<string>();
        _outputs = outputs ?? new List<string>();
    }
    
    public void AddInput(string input) => _inputs.Add(input);
    public void AddOutput(string output) => _outputs.Add(output);

    public Action<CommandBuffer, Renderer.FrameContext> ExecuteFunc { get; set; } = (_, _) => { };
    
    
    // Builder helpers — lets callers chain 
    public Pass ReadsFrom (string resourceName) { _inputs .Add(resourceName); return this; }
    public Pass WritesTo  (string resourceName) { _outputs.Add(resourceName); return this; }
    public Pass Executes  (Action<CommandBuffer, Renderer.FrameContext> fn) { ExecuteFunc = fn; return this; }
}