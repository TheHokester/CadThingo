namespace CadThingo.VulkanEngine.Renderer.Descriptors;

// Set-index convention for the unified descriptor model (docs/descriptor-system.md
// section 2): frequency-ordered so the most stable set sits lowest and rebinds never
// disturb it. Feature groups pin their own indices >= FirstFeature.
public static class ShaderSets
{
    public const uint Scene = 0;        // DescriptorRegistry-owned, bound once per cmd buffer
    public const uint Pass = 1;         // graph-baked pass-local resources
    public const uint FirstFeature = 2; // ibl@2, wavefrontPT@3, ...
}