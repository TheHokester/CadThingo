using CadThingo.VulkanEngine.Renderer.Pipelines;
using CadThingo.VulkanEngine.Renderer.Slang;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Features.IBL;

// One generic compute pipeline used by every IBL bake step. The four bake
// shaders share an almost-identical descriptor layout - set 0 binding 0 is an
// optional input sampler (cube or 2D), binding 1 is the output storage image
// (cube layer + mip, or 2D LUT). Push constants carry the bake-specific args
// (face size, roughness, etc.) in a tiny <16B block.
//
// Per-dispatch descriptor sets are allocated on demand from the global
// descriptor pool (FreeDescriptorSetBit is on) and freed when the bake is
// finished - these resources are entirely transient and don't survive past
// the single-time command submission that drove the bake.

public sealed unsafe class IblBakePipeline : ComputePipeline
{
    readonly string _module;
    readonly bool   _hasInputSampler;
    readonly uint   _pushSize;

    public IblBakePipeline(GpuContext gpu, Renderer renderer, string module, bool hasInputSampler, uint pushSize)
        : base(gpu, renderer)
    {
        _module          = module;
        _hasInputSampler = hasInputSampler;
        _pushSize        = pushSize;
    }

    protected override ShaderCompileRequest? Program => new(_module, ["main"], [], []);

    protected override void CreateDescriptorSetLayouts()
    {
        // Assigned here, not in the ctor: Initialize derives PushConstantRanges from reflection
        // before this runs, so a ctor assignment would be overwritten. The caller's size is the
        // authority because it is the size the bake dispatch actually pushes.
        PushConstantRanges = _pushSize > 0
            ? [new PushConstantRange { StageFlags = ShaderStageFlags.ComputeBit, Offset = 0, Size = _pushSize }]
            : [];

        DescriptorSetLayouts            = new DescriptorSetLayout[1];
        OwnedDescriptorSetLayoutIndices = new[] { 0 };

        // Two bindings (or one for BrdfLutGen which has no input). The shader
        // is free to use slot 0 alone — binding declarations the slang doesn't
        // reference are silently ignored by validation.
        var bindings = new DescriptorSetLayoutBinding[_hasInputSampler ? 2 : 1];
        int idx = 0;
        if (_hasInputSampler)
        {
            bindings[idx++] = new DescriptorSetLayoutBinding
            {
                Binding         = 0,
                DescriptorType  = DescriptorType.CombinedImageSampler,
                DescriptorCount = 1,
                StageFlags      = ShaderStageFlags.ComputeBit,
            };
        }
        bindings[idx] = new DescriptorSetLayoutBinding
        {
            Binding         = _hasInputSampler ? 1u : 0u,
            DescriptorType  = DescriptorType.StorageImage,
            DescriptorCount = 1,
            StageFlags      = ShaderStageFlags.ComputeBit,
        };

        fixed (DescriptorSetLayoutBinding* p = bindings)
        {
            DescriptorSetLayoutCreateInfo info = new()
            {
                SType        = StructureType.DescriptorSetLayoutCreateInfo,
                BindingCount = (uint)bindings.Length,
                PBindings    = p,
            };
            if (Vk.CreateDescriptorSetLayout(Device, &info, null, out DescriptorSetLayouts[0]) != Result.Success)
                throw new System.Exception($"Failed to create IBL bake set 0 layout for {_module}");
        }
    }

    // No per-pipeline descriptor sets — the caller allocates one set per
    // dispatch via AllocateDescriptorSet(), writes it, and frees it after
    // the command buffer that uses it has been waited on.
    protected override void CreateDescriptorSets() { }
    protected override void WriteDescriptors()     { }

    public DescriptorSet AllocateDescriptorSet()
    {
        var layout = DescriptorSetLayouts[0];
        DescriptorSetAllocateInfo info = new()
        {
            SType              = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool     = Gfx.DescriptorPool,
            DescriptorSetCount = 1,
            PSetLayouts        = &layout,
        };
        DescriptorSet set;
        if (Vk.AllocateDescriptorSets(Device, &info, &set) != Result.Success)
            throw new Exception("Failed to allocate IBL bake descriptor set");
        return set;
    }

    public void FreeDescriptorSet(DescriptorSet set)
    {
        if (set.Handle == 0) return;
        Vk.FreeDescriptorSets(Device, Gfx.DescriptorPool, 1, &set);
    }

    /// <summary>
    /// Writes (input sampler, output storage image) into a freshly-allocated set.
    /// Pass <paramref name="inputView"/> + <paramref name="inputSampler"/> as
    /// zero handles for pipelines without an input (BrdfLutGen).
    /// </summary>
    public void WriteDescriptors(DescriptorSet set, ImageView outputView,
        ImageView inputView, Sampler inputSampler)
    {
        int writeCount = _hasInputSampler ? 2 : 1;
        var writes = stackalloc WriteDescriptorSet[writeCount];
        var imageInfos = stackalloc DescriptorImageInfo[2];

        int w = 0;
        if (_hasInputSampler)
        {
            imageInfos[w] = new DescriptorImageInfo
            {
                Sampler     = inputSampler,
                ImageView   = inputView,
                ImageLayout = ImageLayout.ShaderReadOnlyOptimal,
            };
            writes[w] = new WriteDescriptorSet
            {
                SType           = StructureType.WriteDescriptorSet,
                DstSet          = set,
                DstBinding      = 0,
                DstArrayElement = 0,
                DescriptorType  = DescriptorType.CombinedImageSampler,
                DescriptorCount = 1,
                PImageInfo      = &imageInfos[w],
            };
            w++;
        }

        imageInfos[w] = new DescriptorImageInfo
        {
            Sampler     = default,
            ImageView   = outputView,
            ImageLayout = ImageLayout.General,
        };
        writes[w] = new WriteDescriptorSet
        {
            SType           = StructureType.WriteDescriptorSet,
            DstSet          = set,
            DstBinding      = _hasInputSampler ? 1u : 0u,
            DstArrayElement = 0,
            DescriptorType  = DescriptorType.StorageImage,
            DescriptorCount = 1,
            PImageInfo      = &imageInfos[w],
        };

        Vk.UpdateDescriptorSets(Device, (uint)writeCount, writes, 0, null);
    }
}
