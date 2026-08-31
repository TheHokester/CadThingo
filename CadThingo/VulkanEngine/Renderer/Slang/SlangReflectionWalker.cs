using System.Runtime.InteropServices;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Slang;

// Walks an opaque ProgramLayout* (SlangComponentType.GetLayout) into ShaderReflectionData
// via the flat spReflection_* exports. The layout is owned by the component it came from -
// walk BEFORE releasing that component; the result is fully managed and safe afterwards.
//
// Enum values below are declaration-order constants from slang.h (Slang 2026.1); same
// recount-on-SDK-upgrade rule as the vtable slot maps in SlangCom.cs.

internal enum SlangParameterCategory : uint
{
    None = 0,
    Mixed = 1,
    ConstantBuffer = 2,
    Uniform = 8,
    DescriptorTableSlot = 9,
    SpecializationConstant = 10,
    PushConstantBuffer = 11,
}

internal enum SlangBindingType : uint
{
    Unknown = 0,
    Sampler = 1,
    Texture = 2,
    ConstantBuffer = 3,
    ParameterBlock = 4,
    TypedBuffer = 5,
    RawBuffer = 6,
    CombinedTextureSampler = 7,
    InputRenderTarget = 8,
    InlineUniformData = 9,
    RayTracingAccelerationStructure = 10,
    PushConstant = 14,
    MutableFlag = 0x100, // RW/storage variant of Texture / TypedBuffer / RawBuffer
    BaseMask = 0x00FF,
}

internal enum SlangTypeKind : uint
{
    Array = 2,
}

internal enum SlangStage : uint
{
    Vertex = 1,
    Hull = 2,
    Domain = 3,
    Geometry = 4,
    Fragment = 5,
    Compute = 6,
    RayGeneration = 7,
    Intersection = 8,
    AnyHit = 9,
    ClosestHit = 10,
    Miss = 11,
    Callable = 12,
    Mesh = 13,
    Amplification = 14,
}

internal static unsafe class SlangReflectionWalker
{
    public static ShaderReflectionData Walk(void* programLayout)
    {
        var entryPoints = WalkEntryPoints(programLayout);

        // Per-binding stage granularity buys nothing here (the registry design binds with
        // All anyway); module-only reflection has no entry points at all.
        var stages = ShaderStageFlags.All;
        if (entryPoints.Length > 0)
        {
            stages = 0;
            foreach (var ep in entryPoints) stages |= ep.Stage;
        }

        var bindings = new List<BindingDesc>();
        var pushRanges = new List<PushRangeDesc>();
        var specConsts = new List<SpecConstDesc>();

        uint paramCount = SlangNative.spReflection_GetParameterCount(programLayout);
        for (uint i = 0; i < paramCount; i++)
        {
            void* varLayout = SlangNative.spReflection_GetParameterByIndex(programLayout, i);
            void* variable = SlangNative.spReflectionVariableLayout_GetVariable(varLayout);
            string name = Utf8(SlangNative.spReflectionVariable_GetName(variable));
            void* typeLayout = SlangNative.spReflectionVariableLayout_GetTypeLayout(varLayout);
            var category = (SlangParameterCategory)SlangNative.spReflectionTypeLayout_GetParameterCategory(typeLayout);

            switch (category)
            {
                case SlangParameterCategory.DescriptorTableSlot:
                    bindings.Add(WalkBinding(name, varLayout, typeLayout, stages));
                    break;

                case SlangParameterCategory.PushConstantBuffer:
                {
                    // The parameter is a ConstantBuffer wrapper; the payload size is the
                    // ELEMENT type's size in the Uniform layout category.
                    void* elemLayout = SlangNative.spReflectionTypeLayout_GetElementTypeLayout(typeLayout);
                    uint size = (uint)SlangNative.spReflectionTypeLayout_GetSize(
                        elemLayout, (uint)SlangParameterCategory.Uniform);
                    pushRanges.Add(new PushRangeDesc(stages, size, name));
                    break;
                }

                case SlangParameterCategory.SpecializationConstant:
                {
                    uint id = (uint)SlangNative.spReflectionVariableLayout_GetOffset(
                        varLayout, (uint)SlangParameterCategory.SpecializationConstant);
                    specConsts.Add(new SpecConstDesc(name, id));
                    break;
                }

                default:
                    Console.WriteLine($"[slang] reflection: skipping global '{name}' with unhandled category {category}");
                    break;
            }
        }

        return new ShaderReflectionData
        {
            Bindings = bindings.ToArray(),
            PushConstants = pushRanges.ToArray(),
            SpecConstants = specConsts.ToArray(),
            EntryPoints = entryPoints,
        };
    }

    private static BindingDesc WalkBinding(string name, void* varLayout, void* typeLayout, ShaderStageFlags stages)
    {
        const uint cat = (uint)SlangParameterCategory.DescriptorTableSlot;
        uint set = (uint)SlangNative.spReflectionVariableLayout_GetSpace(varLayout, cat);
        uint binding = (uint)SlangNative.spReflectionVariableLayout_GetOffset(varLayout, cat);

        long rangeCount = SlangNative.spReflectionTypeLayout_getBindingRangeCount(typeLayout);
        if (rangeCount < 1)
            throw new Exception($"Slang reflection: global '{name}' occupies a descriptor slot but has no binding range");
        var bindingType = (SlangBindingType)SlangNative.spReflectionTypeLayout_getBindingRangeType(typeLayout, 0);

        // Arrays: descriptor count from the type's element count; 0 = unbounded (Texture2D t[]).
        uint count = 1;
        void* type = SlangNative.spReflectionTypeLayout_GetType(typeLayout);
        if ((SlangTypeKind)SlangNative.spReflectionType_GetKind(type) == SlangTypeKind.Array)
            count = (uint)SlangNative.spReflectionType_GetElementCount(type);

        return new BindingDesc(name, set, binding, ToDescriptorType(bindingType, name), count, stages);
    }

    private static DescriptorType ToDescriptorType(SlangBindingType t, string name)
    {
        bool mutable = (t & SlangBindingType.MutableFlag) != 0;
        return (SlangBindingType)(t & SlangBindingType.BaseMask) switch
        {
            SlangBindingType.Sampler => DescriptorType.Sampler,
            SlangBindingType.Texture => mutable ? DescriptorType.StorageImage : DescriptorType.SampledImage,
            SlangBindingType.CombinedTextureSampler => DescriptorType.CombinedImageSampler,
            SlangBindingType.ConstantBuffer => DescriptorType.UniformBuffer,
            SlangBindingType.RawBuffer => DescriptorType.StorageBuffer,
            SlangBindingType.TypedBuffer => mutable ? DescriptorType.StorageTexelBuffer : DescriptorType.UniformTexelBuffer,
            SlangBindingType.RayTracingAccelerationStructure => DescriptorType.AccelerationStructureKhr,
            var other => throw new Exception($"Slang reflection: global '{name}' has unmapped binding type {other} (mutable={mutable})"),
        };
    }

    private static EntryPointDesc[] WalkEntryPoints(void* programLayout)
    {
        ulong count = SlangNative.spReflection_getEntryPointCount(programLayout);
        var result = new EntryPointDesc[count];
        for (ulong i = 0; i < count; i++)
        {
            void* ep = SlangNative.spReflection_getEntryPointByIndex(programLayout, i);
            string name = Utf8(SlangNative.spReflectionEntryPoint_getName(ep));
            var stage = (SlangStage)SlangNative.spReflectionEntryPoint_getStage(ep);

            uint gx = 0, gy = 0, gz = 0;
            if (stage == SlangStage.Compute)
            {
                ulong* sizes = stackalloc ulong[3];
                SlangNative.spReflectionEntryPoint_getComputeThreadGroupSize(ep, 3, sizes);
                gx = (uint)sizes[0]; gy = (uint)sizes[1]; gz = (uint)sizes[2];
            }
            result[i] = new EntryPointDesc(name, ToStageFlags(stage, name), gx, gy, gz);
        }
        return result;
    }

    private static ShaderStageFlags ToStageFlags(SlangStage stage, string name) => stage switch
    {
        SlangStage.Vertex => ShaderStageFlags.VertexBit,
        SlangStage.Hull => ShaderStageFlags.TessellationControlBit,
        SlangStage.Domain => ShaderStageFlags.TessellationEvaluationBit,
        SlangStage.Geometry => ShaderStageFlags.GeometryBit,
        SlangStage.Fragment => ShaderStageFlags.FragmentBit,
        SlangStage.Compute => ShaderStageFlags.ComputeBit,
        SlangStage.RayGeneration => ShaderStageFlags.RaygenBitKhr,
        SlangStage.Intersection => ShaderStageFlags.IntersectionBitKhr,
        SlangStage.AnyHit => ShaderStageFlags.AnyHitBitKhr,
        SlangStage.ClosestHit => ShaderStageFlags.ClosestHitBitKhr,
        SlangStage.Miss => ShaderStageFlags.MissBitKhr,
        SlangStage.Callable => ShaderStageFlags.CallableBitKhr,
        SlangStage.Mesh => ShaderStageFlags.MeshBitExt,
        SlangStage.Amplification => ShaderStageFlags.TaskBitExt,
        _ => throw new Exception($"Slang reflection: entry point '{name}' has unmapped stage {stage}"),
    };

    private static string Utf8(byte* s) => Marshal.PtrToStringUTF8((nint)s) ?? "";
}