using System.Runtime.InteropServices;

namespace CadThingo.VulkanEngine.Renderer.Shaders;

// COM-lite wrappers over slang.dll interface pointers. Slang objects are C++ classes whose
// vtables start IUnknown-compatible (slot 0 queryInterface, 1 addRef, 2 release). A wrapper
// is a readonly struct over the raw pointer; methods index the vtable with
// delegate* unmanaged[MemberFunction], which matches the MSVC virtual-call ABI on x64
// (instance pointer in the first register).
//
// Slot maps were hand-counted against %VULKAN_SDK%\Include\slang\slang.h (Slang 2026.1):
// slot = 3 + zero-based position of the virtual in declaration order, derived interfaces
// continue after the base's last slot. RECOUNT ON SDK UPGRADE - a wrong slot is a crash at
// best and silent garbage at worst. Verify with the smoke test before trusting anything.
//
// Lifetime: everything created from a global session is refcounted; call Release() when done.
// Blobs returned via out-params arrive with a reference you own; diagnostics blobs can be
// non-null even on success (warnings), so they are always drained via TakeDiagnostics.

// Unmanaged UTF-8 copy of a string, valid until Dispose. A stackalloc inside a helper method
// dies with that method's frame, hence heap allocation here.
internal readonly unsafe struct Utf8Str(string s) : IDisposable
{
    public readonly byte* Ptr = (byte*)Marshal.StringToCoTaskMemUTF8(s);
    public void Dispose() => Marshal.FreeCoTaskMem((nint)Ptr);
}

internal static unsafe class SlangCom
{
    // Drains an out-diagnostics blob: empty string when null, else text + release.
    public static string TakeDiagnostics(void* blobPtr)
    {
        if (blobPtr == null) return "";
        var blob = new SlangBlob(blobPtr);
        string text = blob.AsString();
        blob.Release();
        return text;
    }
}

internal readonly unsafe struct SlangBlob(void* ptr)
{
    public readonly void* Ptr = ptr;
    private void** Vtbl => *(void***)Ptr;

    public bool IsNull => Ptr == null;

    // slot 3: void const* getBufferPointer()
    public void* BufferPointer => ((delegate* unmanaged[MemberFunction]<void*, void*>)Vtbl[3])(Ptr);

    // slot 4: size_t getBufferSize()
    public nuint BufferSize => ((delegate* unmanaged[MemberFunction]<void*, nuint>)Vtbl[4])(Ptr);

    public byte[] ToArray() => new ReadOnlySpan<byte>(BufferPointer, (int)BufferSize).ToArray();

    // Diagnostics blobs are UTF-8 text (not null-terminated; length from BufferSize).
    public string AsString() => System.Text.Encoding.UTF8.GetString(
        new ReadOnlySpan<byte>(BufferPointer, (int)BufferSize));

    // slot 2: uint32_t release()
    public void Release() => ((delegate* unmanaged[MemberFunction]<void*, uint>)Vtbl[2])(Ptr);
}

// IGlobalSession slot map (slang.h struct IGlobalSession):
//   3 createSession(SessionDesc const&, ISession**)        -> SlangResult
//   4 findProfile(char const*)                             -> SlangProfileID (uint32)
//   5 setDownstreamCompilerPath   6 setDownstreamCompilerPrelude
//   7 getDownstreamCompilerPrelude
//   8 getBuildTagString()                                  -> char const*
//   9 setDefaultDownstreamCompiler  10 getDefaultDownstreamCompiler
//  11 setLanguagePrelude  12 getLanguagePrelude  13 createCompileRequest (deprecated)
//  14 addBuiltins  15 setSharedLibraryLoader  16 getSharedLibraryLoader
//  17 checkCompileTargetSupport  18 checkPassThroughSupport
//  19 compileCoreModule  20 loadCoreModule  21 saveCoreModule
//  22 findCapability(char const*)                          -> SlangCapabilityID (uint32)
//  23 setDownstreamCompilerForTransition  24 getDownstreamCompilerForTransition
//  25 getCompilerElapsedTime  26 setSPIRVCoreGrammar  27 parseCommandLineArguments
//  28 getSessionDescDigest  29 compileBuiltinModule  30 loadBuiltinModule  31 saveBuiltinModule
internal readonly unsafe struct SlangGlobalSession(void* ptr)
{
    public readonly void* Ptr = ptr;
    private void** Vtbl => *(void***)Ptr;

    public static SlangGlobalSession Create()
    {
        void* p;
        int r = SlangNative.slang_createGlobalSession(0, &p);
        if (r < 0) throw new InvalidOperationException($"slang_createGlobalSession failed: 0x{r:X8}");
        return new SlangGlobalSession(p);
    }

    /// slot 8: char const* getBuildTagString()
    public string BuildTag()
    {
        byte* s = ((delegate* unmanaged[MemberFunction]<void*, byte*>)Vtbl[8])(Ptr);
        return Marshal.PtrToStringUTF8((nint)s) ?? "unknown";
    }

    /// slot 4: SlangProfileID findProfile(char const* name) - call with "spirv_1_6".
    /// Returns SLANG_PROFILE_UNKNOWN (0) when the name is not found.
    public uint FindProfile(string name)
    {
        using var pName = new Utf8Str(name);
        uint profileId = ((delegate* unmanaged[MemberFunction]<void*, byte*, uint>)Vtbl[4])(Ptr, pName.Ptr);
        if (profileId == 0)
            throw new Exception($"Slang profile '{name}' not found");
        return profileId;
    }

    /// slot 22: SlangCapabilityID findCapability(char const* name)
    /// Names: "spvRayQueryKHR", "spvRayTracingKHR", "spvShaderInvocationReorderNV" etc.
    public uint FindCapability(string name)
    {
        using var pName = new Utf8Str(name);
        uint capabilityId = ((delegate* unmanaged[MemberFunction]<void*, byte*, uint>)Vtbl[22])(Ptr, pName.Ptr);
        if (capabilityId == 0)
            throw new Exception($"Slang capability '{name}' not found");
        return capabilityId;
    }

    /// slot 3: SlangResult createSession(SessionDesc const& desc, ISession** outSession)
    public SlangSession CreateSession(in SessionDesc desc)
    {
        fixed (SessionDesc* pDesc = &desc)
        {
            void* session;
            int r = ((delegate* unmanaged[MemberFunction]<void*, SessionDesc*, void**, int>)Vtbl[3])(Ptr, pDesc, &session);
            if (r < 0)
                throw new Exception($"SlangGlobalSession.CreateSession failed: 0x{r:X8}");
            return new SlangSession(session);
        }
    }

    public void Release() => ((delegate* unmanaged[MemberFunction]<void*, uint>)Vtbl[2])(Ptr);
}

// ISession slot map (slang.h struct ISession):
//   3 getGlobalSession
//   4 loadModule(char const* moduleName, IBlob** outDiagnostics)              -> IModule*
//   5 loadModuleFromSource
//   6 createCompositeComponentType(IComponentType* const*, SlangInt count,
//                                  IComponentType** out, ISlangBlob** outDiag) -> SlangResult
//   7 specializeType  8 getTypeLayout  9 getContainerType  10 getDynamicType
//  11 getTypeRTTIMangledName  12 getTypeConformanceWitnessMangledName
//  13 getTypeConformanceWitnessSequentialID  14 createCompileRequest
//  15 createTypeConformanceComponentType  16 loadModuleFromIRBlob
//  17 getLoadedModuleCount  18 getLoadedModule  19 isBinaryModuleUpToDate
//  20 loadModuleFromSourceString(char const* moduleName, char const* path,
//                                char const* string, ISlangBlob** outDiag)     -> IModule*
internal readonly unsafe struct SlangSession(void* ptr)
{
    public readonly void* Ptr = ptr;
    private void** Vtbl => *(void***)Ptr;

    /// slot 4: IModule* loadModule(char const* moduleName, IBlob** outDiagnostics)
    /// Resolves via the session search paths; null return = compile error (text in diagnostics).
    public SlangModule LoadModule(string moduleName)
    {
        using var pName = new Utf8Str(moduleName);
        void* diag = null;
        void* module = ((delegate* unmanaged[MemberFunction]<void*, byte*, void**, void*>)Vtbl[4])(Ptr, pName.Ptr, &diag);
        string diagText = SlangCom.TakeDiagnostics(diag);
        if (module == null)
            throw new Exception($"Slang loadModule('{moduleName}') failed:\n{diagText}");
        return new SlangModule(module);
    }

    /// slot 20: IModule* loadModuleFromSourceString(char const* moduleName, char const* path,
    ///                                              char const* string, ISlangBlob** outDiag)
    /// No search paths involved; `path` is only used for diagnostics display.
    public SlangModule LoadModuleFromSourceString(string moduleName, string path, string source)
    {
        using var pName = new Utf8Str(moduleName);
        using var pPath = new Utf8Str(path);
        using var pSource = new Utf8Str(source);
        void* diag = null;
        void* module = ((delegate* unmanaged[MemberFunction]<void*, byte*, byte*, byte*, void**, void*>)Vtbl[20])(
            Ptr, pName.Ptr, pPath.Ptr, pSource.Ptr, &diag);
        string diagText = SlangCom.TakeDiagnostics(diag);
        if (module == null)
            throw new Exception($"Slang loadModuleFromSourceString('{moduleName}') failed:\n{diagText}");
        return new SlangModule(module);
    }

    /// slot 6: SlangResult createCompositeComponentType(IComponentType* const* componentTypes,
    ///             SlangInt count, IComponentType** out, ISlangBlob** outDiag)
    /// Pass module + entry point handles (both ARE IComponentType*); order defines entry indices.
    public SlangComponentType CreateComposite(ReadOnlySpan<nint> components)
    {
        fixed (nint* pComponents = components)
        {
            void* composite;
            void* diag = null;
            int r = ((delegate* unmanaged[MemberFunction]<void*, void**, long, void**, void**, int>)Vtbl[6])(
                Ptr, (void**)pComponents, components.Length, &composite, &diag);
            string diagText = SlangCom.TakeDiagnostics(diag);
            if (r < 0)
                throw new Exception($"Slang createCompositeComponentType failed: 0x{r:X8}\n{diagText}");
            return new SlangComponentType(composite);
        }
    }

    public void Release() => ((delegate* unmanaged[MemberFunction]<void*, uint>)Vtbl[2])(Ptr);
}

// IComponentType slot map (slang.h struct IComponentType) - IModule and IEntryPoint derive
// from this, so these slots are valid on their pointers too:
//   3 getSession
//   4 getLayout(SlangInt targetIndex, IBlob** outDiagnostics)   -> ProgramLayout* (reflection root)
//   5 getSpecializationParamCount
//   6 getEntryPointCode(SlangInt entryPointIndex, SlangInt targetIndex,
//                       IBlob** outCode, IBlob** outDiagnostics) -> SlangResult
//   7 getResultAsFileSystem  8 getEntryPointHash  9 specialize
//  10 link(IComponentType** outLinked, ISlangBlob** outDiagnostics) -> SlangResult
//  11 getEntryPointHostCallable  12 renameEntryPoint  13 linkWithOptions
//  14 getTargetCode  15 getTargetMetadata  16 getEntryPointMetadata
internal readonly unsafe struct SlangComponentType(void* ptr)
{
    public readonly void* Ptr = ptr;
    private void** Vtbl => *(void***)Ptr;

    public nint Handle => (nint)Ptr;

    /// slot 4: ProgramLayout* getLayout(SlangInt targetIndex, IBlob** outDiagnostics)
    /// Opaque reflection root for the spReflection_* flat exports; owned by the component,
    /// no Release. Null = layout unavailable (text in diagnostics).
    public void* GetLayout(long targetIndex = 0)
    {
        void* diag = null;
        void* layout = ((delegate* unmanaged[MemberFunction]<void*, long, void**, void*>)Vtbl[4])(Ptr, targetIndex, &diag);
        string diagText = SlangCom.TakeDiagnostics(diag);
        if (layout == null)
            throw new Exception($"Slang getLayout failed:\n{diagText}");
        return layout;
    }

    /// slot 6: SlangResult getEntryPointCode(SlangInt entryPointIndex, SlangInt targetIndex,
    ///                                       IBlob** outCode, IBlob** outDiagnostics)
    public byte[] GetEntryPointCode(long entryPointIndex, long targetIndex = 0)
    {
        void* code = null;
        void* diag = null;
        int r = ((delegate* unmanaged[MemberFunction]<void*, long, long, void**, void**, int>)Vtbl[6])(
            Ptr, entryPointIndex, targetIndex, &code, &diag);
        string diagText = SlangCom.TakeDiagnostics(diag);
        if (r < 0 || code == null)
            throw new Exception($"Slang getEntryPointCode({entryPointIndex}) failed: 0x{r:X8}\n{diagText}");
        var blob = new SlangBlob(code);
        byte[] bytes = blob.ToArray();
        blob.Release();
        return bytes;
    }

    /// slot 10: SlangResult link(IComponentType** outLinked, ISlangBlob** outDiagnostics)
    public SlangComponentType Link()
    {
        void* linked;
        void* diag = null;
        int r = ((delegate* unmanaged[MemberFunction]<void*, void**, void**, int>)Vtbl[10])(Ptr, &linked, &diag);
        string diagText = SlangCom.TakeDiagnostics(diag);
        if (r < 0)
            throw new Exception($"Slang link failed: 0x{r:X8}\n{diagText}");
        return new SlangComponentType(linked);
    }

    // TODO(reflection phase) slot 16 getEntryPointMetadata -> IMetadata, for
    //   isParameterLocationUsed consumer tracking; needs the IMetadata slot map first.

    public void Release() => ((delegate* unmanaged[MemberFunction]<void*, uint>)Vtbl[2])(Ptr);
}

// IEntryPoint = IComponentType slots + 17 getFunctionReflection. Only exists to be handed
// back to CreateComposite, so it stays a thin handle.
internal readonly unsafe struct SlangEntryPoint(void* ptr)
{
    public readonly void* Ptr = ptr;
    private void** Vtbl => *(void***)Ptr;

    public nint Handle => (nint)Ptr;
    public SlangComponentType AsComponentType() => new(Ptr);

    public void Release() => ((delegate* unmanaged[MemberFunction]<void*, uint>)Vtbl[2])(Ptr);
}

// IModule continues after IComponentType at:
//  17 findEntryPointByName(char const* name, IEntryPoint** out)  -> SlangResult
//  18 getDefinedEntryPointCount  19 getDefinedEntryPoint  20 serialize  21 writeToFile
//  22 getName  23 getFilePath  24 getUniqueIdentity  25 findAndCheckEntryPoint
//  26 getDependencyFileCount()                                   -> SlangInt32 (int)
//  27 getDependencyFilePath(SlangInt32 index)                    -> char const*
//  28 getModuleReflection  29 disassemble
internal readonly unsafe struct SlangModule(void* ptr)
{
    public readonly void* Ptr = ptr;
    private void** Vtbl => *(void***)Ptr;

    public nint Handle => (nint)Ptr;
    public SlangComponentType AsComponentType() => new(Ptr);

    /// slot 17: SlangResult findEntryPointByName(char const* name, IEntryPoint** out)
    /// Only finds functions marked [shader("...")].
    public SlangEntryPoint FindEntryPointByName(string name)
    {
        using var pName = new Utf8Str(name);
        void* entryPoint;
        int r = ((delegate* unmanaged[MemberFunction]<void*, byte*, void**, int>)Vtbl[17])(Ptr, pName.Ptr, &entryPoint);
        if (r < 0 || entryPoint == null)
            throw new Exception($"Slang entry point '{name}' not found: 0x{r:X8}");
        return new SlangEntryPoint(entryPoint);
    }

    /// slot 22: char const* getName()
    public string Name()
    {
        byte* s = ((delegate* unmanaged[MemberFunction]<void*, byte*>)Vtbl[22])(Ptr);
        return Marshal.PtrToStringUTF8((nint)s) ?? "";
    }

    /// slots 26/27: the transitive source-file closure, feeding the shader cache dep manifest.
    public string[] DependencyFiles()
    {
        int count = ((delegate* unmanaged[MemberFunction]<void*, int>)Vtbl[26])(Ptr);
        var files = new string[count];
        for (int i = 0; i < count; i++)
        {
            byte* s = ((delegate* unmanaged[MemberFunction]<void*, int, byte*>)Vtbl[27])(Ptr, i);
            files[i] = Marshal.PtrToStringUTF8((nint)s) ?? "";
        }
        return files;
    }

    public void Release() => ((delegate* unmanaged[MemberFunction]<void*, uint>)Vtbl[2])(Ptr);
}