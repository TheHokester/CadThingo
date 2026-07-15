using System.Runtime.InteropServices;

namespace CadThingo.VulkanEngine.Renderer.Shaders;

// Blittable mirrors of slang.h structs and enums (Slang 2026.1,
// %VULKAN_SDK%\Include\slang\slang.h). Layout rules that make Sequential match MSVC:
//   SlangInt/SlangUInt = 8 bytes (long/ulong), size_t = nuint, all enums = 4 bytes,
//   C++ bool = 1 byte (C# bool in an unmanaged struct is also 1 byte), pointers = 8 bytes.
// Every desc struct starts with size_t structureSize and MUST be set to sizeof(that struct);
// Slang uses it to version-check the caller.

internal enum SlangCompileTarget : int
{
    Unknown = 0,
    Spirv = 6, // SLANG_SPIRV; values 3 and 4 are deprecated placeholders, do not renumber
}

internal enum CompilerOptionValueKind : int { Int = 0, String = 1 }

// Values are declaration order of slang.h's enum class CompilerOptionName. Hand-counted
// against the installed header; RECOUNT ON SDK UPGRADE (they are not stable across versions).
internal enum CompilerOptionName : int
{
    MacroDefine = 0,   // stringValue0 = name, stringValue1 = value
    Include = 6,       // stringValue0 = search path
    Profile = 15,      // intValue0 = SlangProfileID
    Capability = 39,   // intValue0 = SlangCapabilityID from IGlobalSession::findCapability
    Optimization = 46, // intValue0 = 0..3 (-O0..-O3)
    
    VulkanUseEntryPointName = 52,//keep slang entry point name as declared, otherwise gets simplified to "main"
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct CompilerOptionValue
{
    public CompilerOptionValueKind Kind;
    public int IntValue0;
    public int IntValue1;
    public byte* StringValue0; // UTF-8, null-terminated; must stay pinned for the call
    public byte* StringValue1;
}

[StructLayout(LayoutKind.Sequential)]
internal struct CompilerOptionEntry
{
    public CompilerOptionName Name;
    public CompilerOptionValue Value;
}

// Worked layout example. C++ original:
//   struct TargetDesc {
//       size_t structureSize = sizeof(TargetDesc);
//       SlangCompileTarget format;            // enum, 4B
//       SlangProfileID profile;               // uint32
//       SlangTargetFlags flags;               // uint32; C++ default kDefaultTargetFlags = 1 << 10
//       SlangFloatingPointMode floatingPointMode;  // uint32, 0 = default
//       SlangLineDirectiveMode lineDirectiveMode;  // uint32, 0 = default
//       bool forceGLSLScalarBufferLayout;     // 1B, then 3B pad to align next pointer... (8B here)
//       const CompilerOptionEntry* compilerOptionEntries;
//       uint32_t compilerOptionEntryCount;    // trailing 4B + 4B tail pad
//   };
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct TargetDesc
{
    public nuint StructureSize;
    public SlangCompileTarget Format;
    public uint Profile;
    // C# zero-init would drop GENERATE_SPIRV_DIRECTLY (1 << 10) and silently reroute
    // compilation through downstream glslang; always set DefaultFlags explicitly.
    public uint Flags;
    public uint FloatingPointMode;
    public uint LineDirectiveMode;
    public bool ForceGlslScalarBufferLayout;
    public CompilerOptionEntry* CompilerOptionEntries;
    public uint CompilerOptionEntryCount;

    public const uint DefaultFlags = 1u << 10; // SLANG_TARGET_FLAG_GENERATE_SPIRV_DIRECTLY

    public static TargetDesc Create() => new()
    {
        StructureSize = (nuint)sizeof(TargetDesc),
        Flags = DefaultFlags,
    };
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct PreprocessorMacroDesc
{
    public byte* Name;  // UTF-8
    public byte* Value;
}

// C++ original:
//   struct SessionDesc {
//       size_t structureSize = sizeof(SessionDesc);
//       TargetDesc const* targets;
//       SlangInt targetCount;                          // int64!
//       SessionFlags flags;                            // uint32, 0
//       SlangMatrixLayoutMode defaultMatrixLayoutMode; // uint32, 1 = ROW_MAJOR
//       char const* const* searchPaths;
//       SlangInt searchPathCount;                      // int64!
//       PreprocessorMacroDesc const* preprocessorMacros;
//       SlangInt preprocessorMacroCount;               // int64!
//       ISlangFileSystem* fileSystem;                  // null = OS filesystem
//       bool enableEffectAnnotations;                  // two adjacent 1B bools...
//       bool allowGLSLSyntax;                          // ...then 6B pad before next pointer
//       CompilerOptionEntry* compilerOptionEntries;
//       uint32_t compilerOptionEntryCount;
//       bool skipSPIRVValidation;                      // 1B after a 4B field + tail pad
//   };
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct SessionDesc
{
    public nuint StructureSize;
    public TargetDesc* Targets;
    public long TargetCount;
    public uint Flags;
    public uint DefaultMatrixLayoutMode;
    public byte** SearchPaths; // UTF-8 char*; C# char is 2-byte UTF-16, so byte**, not char**
    public long SearchPathCount;
    public PreprocessorMacroDesc* PreprocessorMacros;
    public long PreprocessorMacroCount;
    public void* FileSystem;   // ISlangFileSystem*; null = OS filesystem
    public bool EnableEffectAnnotations;
    public bool AllowGlslSyntax;
    public CompilerOptionEntry* CompilerOptionEntries;
    public uint CompilerOptionEntryCount;
    public bool SkipSpirvValidation;

    public static SessionDesc Create() => new()
    {
        StructureSize = (nuint)sizeof(SessionDesc),
        DefaultMatrixLayoutMode = 1, // SLANG_MATRIX_LAYOUT_ROW_MAJOR
    };
}