namespace CadThingo.VulkanEngine.Renderer.FrameGraph;

/// <summary>
/// A composable graph feature. <see cref="Build"/> appends this module's passes + internal
/// resources into <paramref name="scope"/> (which namespaces them), wiring the declared
/// <typeparamref name="TInputs"/> to existing handles and producing <typeparamref name="TOutputs"/>.
///
/// Modules are flattened into the parent graph at build time -- a logical subgraph, not a
/// separately compiled one -- so the compiler keeps a global view for culling/sync/aliasing
/// (render-graph.md sec 9). The host owns Compile/Execute and consumes the outputs.
/// </summary>
public interface IGraphModule<TInputs, TOutputs>
{
    void Build(GraphScope scope, in TInputs inputs, out TOutputs outputs);
}