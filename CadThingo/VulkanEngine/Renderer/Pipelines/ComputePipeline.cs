using Silk.NET.Core.Native;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Pipelines;

public abstract unsafe class ComputePipeline : PipelineBase
{
    public override PipelineBindPoint BindPoint => PipelineBindPoint.Compute;

    protected ComputePipeline(in GpuContext gpu, Renderer renderer) : base(gpu, renderer) { }

    // Build-time .spv for the legacy route; null on the reflected route (see PipelineBase.Program).
    protected virtual string? ShaderPath => null;

    // Legacy-route entry-point symbol: slangc emits the SPIR-V OpEntryPoint as "main" regardless
    // of the source function name when compiling a single-entry kernel. The reflected route takes
    // the name from reflection instead and ignores this.
    protected virtual string EntryPoint => "main";

    protected sealed override void CreatePipeline()
    {
        ShaderModule module;
        string entryName;
        if (Reflected != null)
        {
            var ep = Reflected.Reflection.EntryPoints[0];
            module = CreateReflectedModule(0);
            entryName = ep.Name;
        }
        else
        {
            module = Gfx.CreateShaderModule(File.ReadAllBytes(
                ShaderPath ?? throw new InvalidOperationException(
                    $"{GetType().Name}: needs either a Program or a ShaderPath.")));
            entryName = EntryPoint;
        }
        var entry = SilkMarshal.StringToPtr(entryName);

        var stage = new PipelineShaderStageCreateInfo
        {
            SType  = StructureType.PipelineShaderStageCreateInfo,
            Stage  = ShaderStageFlags.ComputeBit,
            Module = module,
            PName  = (byte*)entry,
        };

        // Scratch must outlive the create call below.
        var specInfo    = new SpecializationInfo();
        var specEntries = stackalloc SpecializationMapEntry[SpecScratchEntries];
        var specData    = stackalloc byte[SpecScratchBytes];
        int filled = FillStageSpecialization(0, specEntries, specData, out uint dataSize);
        if (filled > 0)
        {
            specInfo = new SpecializationInfo
            {
                MapEntryCount = (uint)filled,
                PMapEntries   = specEntries,
                DataSize      = (UIntPtr)dataSize,
                PData         = specData,
            };
            stage.PSpecializationInfo = &specInfo;
        }

        var info = new ComputePipelineCreateInfo
        {
            SType  = StructureType.ComputePipelineCreateInfo,
            Stage  = stage,
            Layout = PipelineLayoutHandle,
        };

        if (Vk.CreateComputePipelines(Device, PipelineCacheHandle, 1, &info, null, out PipelineHandle) != Result.Success)
            throw new Exception($"Failed to create compute pipeline for {GetType().Name}");
        SilkMarshal.Free(entry);
        Vk.DestroyShaderModule(Device, module, null);
    }
}