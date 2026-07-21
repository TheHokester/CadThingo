namespace CadThingo.VulkanEngine.Renderer.Descriptors;

// Set-index convention for the unified descriptor model (docs/descriptor-system.md
// section 2). The fixed-count set kinds hold the low reserved indices; feature sets are the
// only kind whose count grows over time, so they climb upward from FirstFeature and never
// renumber anything below them. Only the feature module itself bakes its index (reflected by
// DescriptorRegistry); consumers read it via FeatureSetIndex, so a renumber is a one-line edit.
public static class ShaderSets
{
    public const uint Scene = 0;        // DescriptorRegistry-owned, bound once per cmd buffer
    public const uint Pass = 1;         // graph-baked pass-local resources
    public const uint GraphShared = 2;  // graph-owned working set, shared by every pass in a graph (PT working / IO)
    public const uint FirstFeature = 3; // ibl@3, env@4, ...
}