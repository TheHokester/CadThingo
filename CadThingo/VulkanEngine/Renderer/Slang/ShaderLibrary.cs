namespace CadThingo.VulkanEngine.Renderer.Slang;

// Cache-or-compile front door for runtime shaders 
// Warm path never touches slang.dll: the compiler is constructed lazily on the first disk
// miss, so a fully cached startup is pure file IO. Not thread-safe (neither is Slang's
// global session) - use from one thread.
public sealed class ShaderLibrary : IDisposable
{
    private readonly string _libDir;
    private readonly string _featuresDir;
    private readonly string[]? _extraIncludeDirs;
    private readonly ShaderCache _cache;
    private readonly Dictionary<string, ShaderProgram> _programs = new(); // memo by request key
    private IShaderCompiler? _compiler;

    public ShaderLibrary(string libDir, string featuresDir, string cacheDir, string[]? extraIncludeDirs = null)
    {
        _libDir = libDir;
        _featuresDir = featuresDir;
        _extraIncludeDirs = extraIncludeDirs;
        _cache = new ShaderCache(cacheDir);
    }

    // Two-step resolution: content next to the executable first, source tree fallback so
    // running from bin without a copy step works.
    public static ShaderLibrary CreateDefault()
    {
        string engine = Path.Combine(AppContext.BaseDirectory, "VulkanEngine");
        if (!Directory.Exists(engine))
            engine = @"C:\Users\jamie\RiderProjects\CadThingo\CadThingo\VulkanEngine";

        string assets = Path.Combine(AppContext.BaseDirectory, "Assets");
        if (!Directory.Exists(assets))
            assets = @"C:\Users\jamie\RiderProjects\CadThingo\CadThingo\Assets";

        return new ShaderLibrary(
            Path.Combine(engine, "Shaders"),
            Path.Combine(engine, "Renderer", "Features"),
            Path.Combine(assets, "ShaderCache"));
    }

    public ShaderProgram GetProgram(in ShaderCompileRequest request)
    {
        string key = _cache.RequestKey(request);
        if (_programs.TryGetValue(key, out var memoized)) return memoized;

        ShaderProgram program;
        if (_cache.TryLoad(request, key, out var cached))
        {
            program = new ShaderProgram(request, cached, fromCache: true);
        }
        else
        {
            _compiler ??= new SlangShaderCompiler(_libDir, _featuresDir, _extraIncludeDirs);
            var compiled = _compiler.Compile(request);
            _cache.Store(request, key, compiled);
            program = new ShaderProgram(request, compiled, fromCache: false);
        }

        _programs[key] = program;
        return program;
    }

    /// Reflection for a module with no entry points (SceneBindings-style shared modules).
    public ShaderReflectionData ReflectModule(string module) =>
        GetProgram(new ShaderCompileRequest(module, [], [], [])).Reflection;

    public void Dispose() => _compiler?.Dispose();
}

public sealed class ShaderProgram
{
    private readonly byte[][] _spirv;

    internal ShaderProgram(in ShaderCompileRequest desc, ShaderCompileResult result, bool fromCache)
    {
        Desc = desc;
        Reflection = result.Reflection;
        FromCache = fromCache;
        _spirv = result.SpirvPerEntryPoint;
    }

    public ShaderCompileRequest Desc { get; }
    public ShaderReflectionData Reflection { get; }
    public bool FromCache { get; }

    /// SPIR-V for the entry point at the same index in Desc.EntryPoints.
    public ReadOnlyMemory<byte> Spirv(int entryIndex) => _spirv[entryIndex];
}