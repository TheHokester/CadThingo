namespace CadThingo.VulkanEngine;

/// <summary>
/// Resolves a compiled-shader (.spv) path from its logical name, replacing the per-pipeline
/// hardcoded absolute paths (render-graph-implementation.md 5.1). One place owns the content
/// root, so there is no machine path scattered across the pipelines.
///
/// Resolution mirrors the Assets/Models + Assets/Textures convention used elsewhere
/// (FileBrowserPanel / RendererSettingsPanel / Renderer_Ibl): Assets/Shaders next to the
/// running executable first (deployed / content copied to output), falling back to the source
/// tree so running straight from bin without a copy step still works.
///
/// Source location and compiled-output location are separate axes: .spv stay flat under
/// Assets/Shaders even after the sources nest into feature folders (section 5.3).
/// </summary>
public static class ShaderPaths
{
    // Resolved once at startup. AppContext.BaseDirectory is the bin dir the exe runs from.
    private static readonly string Dir = Resolve();

    private static string Resolve()
    {
        string app = Path.Combine(AppContext.BaseDirectory, "Assets", "Shaders");
        return Directory.Exists(app)
            ? app
            : @"C:\Users\jamie\RiderProjects\CadThingo\CadThingo\Assets\Shaders";
    }

    /// <summary>Maps a logical shader name (no extension) to its compiled .spv path,
    /// e.g. <c>Spv("PBR")</c> -> <c>.../Assets/Shaders/PBR.spv</c>.</summary>
    public static string Spv(string name) => Path.Combine(Dir, name + ".spv");

    /// <summary>Maps a feature kernel to its compiled .spv, mirrored under the shader root by
    /// feature folder (not flat), matching the build's per-feature output layout:
    /// <c>Kernel("WavefrontPathTracer", "Shade")</c> ->
    /// <c>.../Assets/Shaders/WavefrontPathTracer/Shade.spv</c>.</summary>
    public static string Kernel(string feature, string name) => Path.Combine(Dir, feature, name + ".spv");
}