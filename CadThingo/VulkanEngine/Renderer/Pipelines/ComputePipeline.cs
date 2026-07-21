using Silk.NET.Core.Native;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Pipelines;

public abstract unsafe class ComputePipeline : PipelineBase
{
    public override PipelineBindPoint BindPoint => PipelineBindPoint.Compute;

    protected ComputePipeline(in GpuContext gpu, Renderer renderer) : base(gpu, renderer) { }

    protected sealed override void CreatePipeline()
    {
        var reflected = Reflected ?? throw new InvalidOperationException(
            $"{GetType().Name}: needs a Program.");
        ShaderModule module = CreateReflectedModule(0);
        var entry = SilkMarshal.StringToPtr(reflected.Reflection.EntryPoints[0].Name);

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