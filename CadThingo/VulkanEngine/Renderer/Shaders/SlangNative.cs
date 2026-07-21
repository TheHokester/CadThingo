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

    // ---- reflection exports -------------------------------------------------------------
    // Flat C ABI beneath slang.h's inline reflection wrappers; no vtables. All reflection
    // pointers are opaque void*: program layout (from IComponentType::getLayout), variable
    // layout, variable, type layout, type, entry point. Signature key: SlangInt/SlangUInt =
    // long/ulong (8 bytes), size_t = nuint, enums/unsigned = uint, char* = byte* (UTF-8).
    // Casing is mixed in the ABI (Get vs get) and must match exactly.

    // program layout
    [DllImport(Dll)] internal static extern uint  spReflection_GetParameterCount(void* layout);
    [DllImport(Dll)] internal static extern void* spReflection_GetParameterByIndex(void* layout, uint index);
    [DllImport(Dll)] internal static extern ulong spReflection_getEntryPointCount(void* layout);
    [DllImport(Dll)] internal static extern void* spReflection_getEntryPointByIndex(void* layout, ulong index);

    // variable layout
    [DllImport(Dll)] internal static extern void* spReflectionVariableLayout_GetVariable(void* varLayout);
    [DllImport(Dll)] internal static extern void* spReflectionVariableLayout_GetTypeLayout(void* varLayout);
    [DllImport(Dll)] internal static extern nuint spReflectionVariableLayout_GetOffset(void* varLayout, uint category);
    [DllImport(Dll)] internal static extern nuint spReflectionVariableLayout_GetSpace(void* varLayout, uint category);

    // variable
    [DllImport(Dll)] internal static extern byte* spReflectionVariable_GetName(void* variable);

    // type layout
    [DllImport(Dll)] internal static extern void* spReflectionTypeLayout_GetType(void* typeLayout);
    [DllImport(Dll)] internal static extern uint  spReflectionTypeLayout_GetParameterCategory(void* typeLayout);
    [DllImport(Dll)] internal static extern nuint spReflectionTypeLayout_GetSize(void* typeLayout, uint category);
    [DllImport(Dll)] internal static extern void* spReflectionTypeLayout_GetElementTypeLayout(void* typeLayout);
    [DllImport(Dll)] internal static extern long  spReflectionTypeLayout_getBindingRangeCount(void* typeLayout);
    [DllImport(Dll)] internal static extern uint  spReflectionTypeLayout_getBindingRangeType(void* typeLayout, long index);
    [DllImport(Dll)] internal static extern long  spReflectionTypeLayout_getBindingRangeBindingCount(void* typeLayout, long index);
    [DllImport(Dll)] internal static extern long  spReflectionTypeLayout_getBindingRangeIndexOffset(void* typeLayout, long index);
    [DllImport(Dll)] internal static extern long  spReflectionTypeLayout_getBindingRangeSpaceOffset(void* typeLayout, long index);
    [DllImport(Dll)] internal static extern void* spReflectionTypeLayout_getBindingRangeLeafTypeLayout(void* typeLayout, long index);
    [DllImport(Dll)] internal static extern void* spReflectionTypeLayout_getBindingRangeLeafVariable(void* typeLayout, long index);

    // type
    [DllImport(Dll)] internal static extern uint  spReflectionType_GetKind(void* type);
    [DllImport(Dll)] internal static extern nuint spReflectionType_GetElementCount(void* type);

    // entry point
    [DllImport(Dll)] internal static extern byte* spReflectionEntryPoint_getName(void* entryPoint);
    [DllImport(Dll)] internal static extern uint  spReflectionEntryPoint_getStage(void* entryPoint);
    [DllImport(Dll)] internal static extern void  spReflectionEntryPoint_getComputeThreadGroupSize(void* entryPoint, ulong axisCount, ulong* outSizes);
}