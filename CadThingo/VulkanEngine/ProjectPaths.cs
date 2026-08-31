namespace CadThingo.VulkanEngine;

/// <summary>
/// Locates the engine's content directories. Shaders and assets are not copied to the output
/// directory, so a run out of bin/ has to reach back into the source tree to find them.
/// </summary>
public static class ProjectPaths
{
    private static readonly Lazy<string> _root = new(Resolve);

    /// <summary>The directory holding VulkanEngine/ and Assets/: the executable's own directory
    /// when content sits beside it, otherwise the nearest ancestor holding both.</summary>
    /// <exception cref="DirectoryNotFoundException">Neither the executable's directory nor any
    /// ancestor holds both.</exception>
    public static string Root => _root.Value;

    public static string Engine => Path.Combine(Root, "VulkanEngine");
    public static string Assets => Path.Combine(Root, "Assets");

    private static string Resolve()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (Directory.Exists(Path.Combine(dir.FullName, "VulkanEngine")) &&
                Directory.Exists(Path.Combine(dir.FullName, "Assets")))
                return dir.FullName;
        }

        throw new DirectoryNotFoundException(
            $"Content root not found: no ancestor of '{AppContext.BaseDirectory}' holds both " +
            "VulkanEngine and Assets.");
    }
}
