using System.Runtime.InteropServices;

namespace CadThingo.VulkanEngine.Renderer.Slang;

// In-proc IShaderCompiler over slang.dll: one lazily created global session, one cached
// ISession per (defines, capabilities) combination. Slang's global session is not
// thread-safe - all calls must come from one thread.
//
// Module name resolution ("Deferred/PBR" -> Features/Deferred/Kernels/PBR.slang, "PbrUtils"
// -> the shared lib dir) mirrors the on-disk layout; imports resolve through session search
// paths covering the lib dir plus every feature Kernels dir.
public sealed unsafe class SlangShaderCompiler(string libDir, string featuresDir, string[]? extraIncludeDirs = null) : IShaderCompiler
{
    private SlangGlobalSession _global;
    private bool _globalCreated;
    private readonly Dictionary<string, SlangSession> _sessions = new();

    public string CompilerVersion
    {
        get { EnsureGlobal(); return _global.BuildTag(); }
    }

    public ShaderCompileResult Compile(in ShaderCompileRequest request)
    {
        EnsureGlobal();
        string sourceFile = ResolveModuleFile(request.Module);
        var session = GetSession(request.Defines, request.Capabilities);

        // The module is session-owned (borrowed pointer, never released here); the session
        // is cached and torn down in Dispose.
        var module = session.LoadModuleFromSourceString(
            Path.GetFileNameWithoutExtension(sourceFile), sourceFile, File.ReadAllText(sourceFile));
        string[] deps = NormalizeDependencies(module.DependencyFiles(), sourceFile);

        if (request.EntryPoints.Length == 0)
        {
            // Module-only reflection (SceneBindings-style shared modules): no codegen.
            var moduleRefl = SlangReflectionWalker.Walk(module.AsComponentType().GetLayout());
            return new ShaderCompileResult
            {
                SpirvPerEntryPoint = [],
                Reflection = moduleRefl,
                Dependencies = deps,
            };
        }

        var entryPoints = new SlangEntryPoint[request.EntryPoints.Length];
        var components = new nint[request.EntryPoints.Length + 1];
        components[0] = module.Handle;
        for (int i = 0; i < entryPoints.Length; i++)
        {
            entryPoints[i] = module.FindEntryPointByName(request.EntryPoints[i]);
            components[i + 1] = entryPoints[i].Handle;
        }

        var composite = session.CreateComposite(components);
        var linked = composite.Link();
        try
        {
            var spirv = new byte[request.EntryPoints.Length][];
            for (int i = 0; i < spirv.Length; i++)
                spirv[i] = linked.GetEntryPointCode(i);
            var reflection = SlangReflectionWalker.Walk(linked.GetLayout());

            return new ShaderCompileResult
            {
                SpirvPerEntryPoint = spirv,
                Reflection = reflection,
                Dependencies = deps,
            };
        }
        finally
        {
            linked.Release();
            composite.Release();
            foreach (var ep in entryPoints) ep.Release();
        }
    }

    private string ResolveModuleFile(string module)
    {
        string path;
        int slash = module.IndexOf('/');
        if (slash >= 0)
        {
            string feature = module[..slash];
            string name = module[(slash + 1)..];
            path = Path.Combine(featuresDir, feature, "Kernels", name + ".slang");
        }
        else
        {
            path = Path.Combine(libDir, module + ".slang");
        }
        if (!File.Exists(path))
            throw new FileNotFoundException($"Slang module '{module}' not found at '{path}'");
        return path;
    }

    private static string[] NormalizeDependencies(string[] reported, string sourceFile)
    {
        var deps = new List<string> { Path.GetFullPath(sourceFile) };
        foreach (var d in reported)
        {
            if (!File.Exists(d)) continue; // synthetic paths (in-memory core module etc.)
            string full = Path.GetFullPath(d);
            if (!deps.Contains(full, StringComparer.OrdinalIgnoreCase)) deps.Add(full);
        }
        return deps.ToArray();
    }

    private void EnsureGlobal()
    {
        if (_globalCreated) return;
        _global = SlangGlobalSession.Create();
        _globalCreated = true;
    }

    private SlangSession GetSession(string[] defines, string[] capabilities)
    {
        string key = string.Join(";", defines) + "|" + string.Join(";", capabilities);
        if (_sessions.TryGetValue(key, out var cached)) return cached;

        var session = CreateSession(defines, capabilities);
        _sessions[key] = session;
        return session;
    }

    private SlangSession CreateSession(string[] defines, string[] capabilities)
    {
        var searchPaths = new List<string> { libDir };
        if (extraIncludeDirs != null) searchPaths.AddRange(extraIncludeDirs);
        if (Directory.Exists(featuresDir))
            foreach (var featureDir in Directory.GetDirectories(featuresDir))
            {
                string kernels = Path.Combine(featureDir, "Kernels");
                if (Directory.Exists(kernels)) searchPaths.Add(kernels);
            }

        // Everything below is copied by Slang during createSession; the unmanaged strings
        // only need to survive the call.
        var nativeStrings = new List<nint>();
        byte* Utf8(string s)
        {
            nint p = Marshal.StringToCoTaskMemUTF8(s);
            nativeStrings.Add(p);
            return (byte*)p;
        }

        try
        {
            var pathPtrs = new nint[searchPaths.Count];
            for (int i = 0; i < searchPaths.Count; i++) pathPtrs[i] = (nint)Utf8(searchPaths[i]);

            var macros = new PreprocessorMacroDesc[defines.Length];
            for (int i = 0; i < defines.Length; i++)
            {
                int eq = defines[i].IndexOf('=');
                macros[i].Name = Utf8(eq < 0 ? defines[i] : defines[i][..eq]);
                macros[i].Value = Utf8(eq < 0 ? "1" : defines[i][(eq + 1)..]);
            }

            // Target-scoped options (everything at/after CompilerOptionName.Capability in slang.h).
            // -O2: release-grade codegen, and cheap here because results are disk-cached.
            var options = new CompilerOptionEntry[capabilities.Length + 2];
            for (int i = 0; i < capabilities.Length; i++)
            {
                options[i].Name = CompilerOptionName.Capability;
                options[i].Value.Kind = CompilerOptionValueKind.Int;
                options[i].Value.IntValue0 = (int)_global.FindCapability(capabilities[i]);
            }
            options[^2].Name = CompilerOptionName.Optimization;
            options[^2].Value.Kind = CompilerOptionValueKind.Int;
            options[^2].Value.IntValue0 = 2;
            // Keeps OpEntryPoint named as declared, so the names reflection reports are the names
            // VkPipelineShaderStageCreateInfo.pName can use.
            options[^1].Name = CompilerOptionName.VulkanUseEntryPointName;
            options[^1].Value.Kind = CompilerOptionValueKind.Int;
            options[^1].Value.IntValue0 = 1;

            // Session-scoped options (declared BEFORE the "// Target" marker in slang.h's
            // CompilerOptionName); a target entry for these is silently ignored.
            //
            // COLUMN_MAJOR here emits SPIR-V matrix members decorated RowMajor -- Slang's
            // source-semantics naming is the INVERSE of the SPIR-V decoration. Verified with
            // slangc + spirv-dis: no flag (its default) -> RowMajor decoration, which is what
            // every build-time .spv the engine has ever run was compiled with and what the C#
            // Matrix4x4 mirrors are laid out for. Asking for ROW_MAJOR instead yields a
            // ColMajor decoration and transposes every matrix a shader reads.
            var sessionOptions = new CompilerOptionEntry[1];
            sessionOptions[0].Name = CompilerOptionName.MatrixLayoutColumn;
            sessionOptions[0].Value.Kind = CompilerOptionValueKind.Int;
            sessionOptions[0].Value.IntValue0 = 1;

            fixed (nint* pPaths = pathPtrs)
            fixed (PreprocessorMacroDesc* pMacros = macros)
            fixed (CompilerOptionEntry* pOptions = options)
            fixed (CompilerOptionEntry* pSessionOptions = sessionOptions)
            {
                var target = TargetDesc.Create();
                target.Format = SlangCompileTarget.Spirv;
                target.Profile = _global.FindProfile("spirv_1_6");
                target.CompilerOptionEntries = pOptions;
                target.CompilerOptionEntryCount = (uint)options.Length;

                var desc = SessionDesc.Create();
                desc.Targets = &target;
                desc.TargetCount = 1;
                desc.SearchPaths = (byte**)pPaths;
                desc.SearchPathCount = pathPtrs.Length;
                desc.CompilerOptionEntries = pSessionOptions;
                desc.CompilerOptionEntryCount = (uint)sessionOptions.Length;
                if (macros.Length > 0)
                {
                    desc.PreprocessorMacros = pMacros;
                    desc.PreprocessorMacroCount = macros.Length;
                }
                return _global.CreateSession(desc);
            }
        }
        finally
        {
            foreach (var p in nativeStrings) Marshal.FreeCoTaskMem(p);
        }
    }

    public void Dispose()
    {
        foreach (var s in _sessions.Values) s.Release();
        _sessions.Clear();
        if (_globalCreated) _global.Release();
        _globalCreated = false;
    }
}