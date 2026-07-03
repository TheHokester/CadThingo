using System.Reflection;
using System.Runtime.InteropServices;

namespace CadThingo.VulkanEngine.Renderer.Shaders;

// Flat C exports from slang.dll, resolved from %VULKAN_SDK%\Bin (docs/descriptor-system.md
// section 3). COM-lite vtable wrappers live in SlangCom.cs, struct mirrors in SlangStructs.cs.
// All exports are extern "C"; on x64 the calling convention is the standard one.
internal static unsafe class SlangNative
{
    private const string Dll = "slang";

    static SlangNative()
    {
        NativeLibrary.SetDllImportResolver(Assembly.GetExecutingAssembly(), Resolve);
    }

    private static nint Resolve(string name, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (name != Dll) return nint.Zero;
        var sdk = Environment.GetEnvironmentVariable("VULKAN_SDK")
                  ?? throw new InvalidOperationException("VULKAN_SDK not set; cannot locate slang.dll");
        return NativeLibrary.Load(Path.Combine(sdk, "Bin", "slang.dll"));
    }

    // Touching any member runs the static ctor, registering the resolver before a DllImport fires.
    internal static void EnsureLoaded() { }

    // const char* spGetBuildTagString()
    // Version pinning: slang.dll ships with NO FileVersionInfo, so this string (plus file
    // size/mtime for the no-dll-load path) is the version identity for the shader cache key.
    [DllImport(Dll)]
    private static extern byte* spGetBuildTagString();

    internal static string BuildTag()
    {
        EnsureLoaded();
        return Marshal.PtrToStringUTF8((nint)spGetBuildTagString()) ?? "unknown";
    }

    // SlangResult slang_createGlobalSession(SlangInt apiVersion, slang::IGlobalSession** out)
    // SlangInt is int64 on x64. Pass apiVersion = 0 (SLANG_API_VERSION). Negative result = failure.
    [DllImport(Dll)]
    internal static extern int slang_createGlobalSession(long apiVersion, void** outGlobalSession);

    // TODO(reflection phase, after compilation works): every reflection query is a flat C export;
    // the slang.h reflection "classes" are inline wrappers over them, e.g.
    //     SLANG_API unsigned spReflection_getParameterCount(SlangReflection* reflection);
    //     SLANG_API char const* spReflectionVariableLayout_GetName(SlangReflectionVariableLayout*);
    // Pattern: opaque reflection pointers become void*, copy the signature, done. No vtables here.
}