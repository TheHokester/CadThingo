namespace CadThingo.VulkanEngine.Renderer.Shaders;

// Quarantine seam for the Slang interop: everything
// above this interface must not know whether SPIR-V came from slang.dll or a slangc
// subprocess fallback. Grows a reflection payload when the reflection walker lands.
public interface IShaderCompiler : IDisposable
{
    // Build tag of the backing compiler (cache-key ingredient), e.g. "2026.1-52-gc8ddf20bb".
    string CompilerVersion { get; }

    ShaderCompileResult Compile(in ShaderCompileRequest request);
}

public readonly record struct ShaderCompileRequest(
    string Module,          // module name resolvable via the session search paths
    string[] EntryPoints,   // ["main"], or the RT entry list in SBT group order
    string[] Defines,       // ["USE_SER=1"] -> name/value pairs split on '='
    string[] Capabilities); // ["spvRayQueryKHR"]

public sealed class ShaderCompileResult
{
    public required byte[][] SpirvPerEntryPoint { get; init; }
    public required string[] Dependencies { get; init; } // transitive source files, for the cache dep manifest
    public string Diagnostics { get; init; } = "";       // warnings when compilation succeeded
}