using System.Runtime.InteropServices;
using CadThingo.VulkanEngine.Renderer.Shaders;
using Silk.NET.Core.Native;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Features.TextureCompression;

/// <summary>Which block format the encoder produces. Matches the MODE_* constants in
/// BcEncode.slang. BC1 = opaque RGB (8B/block, 8x), BC3 = RGBA (16B, 4x), BC5 = two-channel
/// (16B, 4x; normals), BC4 = single-channel (8B).</summary>
public enum BcMode : uint { Bc1 = 0, Bc3 = 1, Bc5 = 2, Bc4 = 3 }

/// <summary>
/// Load-time GPU block-compression encoder. Drives the BcEncode compute kernel: each dispatch
/// compresses one mip of a SampledImage source (read raw-UNORM) into a tightly-packed output
/// buffer, which the caller then copies into the BC image. One PSO + one reused descriptor set;
/// callers serialize through single-time submits (so the reused set is safe to re-point per
/// texture). Owned lazily by ResourceManager -- the device only enables the BC capability, the
/// resource layer owns the tooling; see Texture.CreateCompressedTexture.
/// </summary>
public sealed unsafe class BcEncoder : IDisposable
{
    // Matches BcEncode.slang::EncPush (5 uints = 20B).
    [StructLayout(LayoutKind.Sequential)]
    private struct EncPush { public uint Mode, SrcMip, MipW, MipH, BlockOffset; }

    private readonly GraphicsDevice _gfx;
    private readonly Vk             _vk;
    private readonly Device         _device;

    private DescriptorSetLayout _setLayout;
    private PipelineLayout      _pipelineLayout;
    private Pipeline            _pso;
    private DescriptorPool      _pool;
    private DescriptorSet       _set;

    public BcEncoder(in GpuContext gpu)
    {
        _gfx = gpu.Gfx; _vk = _gfx.Vk; _device = _gfx.Device;

        // Set layout: binding 0 = source (sampled image, read via Load), binding 1 = output SSBO.
        var bindings = stackalloc DescriptorSetLayoutBinding[2];
        bindings[0] = new() { Binding = 0, DescriptorType = DescriptorType.SampledImage,  DescriptorCount = 1, StageFlags = ShaderStageFlags.ComputeBit };
        bindings[1] = new() { Binding = 1, DescriptorType = DescriptorType.StorageBuffer,  DescriptorCount = 1, StageFlags = ShaderStageFlags.ComputeBit };
        var slci = new DescriptorSetLayoutCreateInfo { SType = StructureType.DescriptorSetLayoutCreateInfo, BindingCount = 2, PBindings = bindings };
        if (_vk.CreateDescriptorSetLayout(_device, &slci, null, out _setLayout) != Result.Success)
            throw new Exception("BcEncoder: failed to create descriptor set layout");

        var pcr = new PushConstantRange { StageFlags = ShaderStageFlags.ComputeBit, Offset = 0, Size = (uint)sizeof(EncPush) };
        var sl  = _setLayout;
        var plci = new PipelineLayoutCreateInfo
        {
            SType = StructureType.PipelineLayoutCreateInfo,
            SetLayoutCount = 1, PSetLayouts = &sl,
            PushConstantRangeCount = 1, PPushConstantRanges = &pcr,
        };
        if (_vk.CreatePipelineLayout(_device, &plci, null, out _pipelineLayout) != Result.Success)
            throw new Exception("BcEncoder: failed to create pipeline layout");

        // Single-entry compute kernel: slang names the SPIR-V entry point "main" regardless of the
        // source function name, so the reflected name and the PName below agree.
        var program = gpu.Shaders.GetProgram(new ShaderCompileRequest("TextureCompression/BcEncode", ["main"], [], []));
        var module = _gfx.CreateShaderModule(program.Spirv(0).ToArray());
        var entry  = SilkMarshal.StringToPtr(program.Reflection.EntryPoints[0].Name);
        var stage  = new PipelineShaderStageCreateInfo
        {
            SType = StructureType.PipelineShaderStageCreateInfo,
            Stage = ShaderStageFlags.ComputeBit, Module = module, PName = (byte*)entry,
        };
        var cpci = new ComputePipelineCreateInfo { SType = StructureType.ComputePipelineCreateInfo, Stage = stage, Layout = _pipelineLayout };
        if (_vk.CreateComputePipelines(_device, default, 1, &cpci, null, out _pso) != Result.Success)
            throw new Exception("BcEncoder: failed to create compute pipeline");
        _vk.DestroyShaderModule(_device, module, null);
        SilkMarshal.Free(entry);

        // One persistent set, re-pointed per texture. Single-time submits serialize encodes, so
        // the prior encode is done (queue-idle) before the next UpdateDescriptorSets.
        var sizes = stackalloc DescriptorPoolSize[2]
        {
            new() { Type = DescriptorType.SampledImage, DescriptorCount = 1 },
            new() { Type = DescriptorType.StorageBuffer, DescriptorCount = 1 },
        };
        var dpci = new DescriptorPoolCreateInfo { SType = StructureType.DescriptorPoolCreateInfo, MaxSets = 1, PoolSizeCount = 2, PPoolSizes = sizes };
        if (_vk.CreateDescriptorPool(_device, &dpci, null, out _pool) != Result.Success)
            throw new Exception("BcEncoder: failed to create descriptor pool");

        var lay  = _setLayout;
        var dsai = new DescriptorSetAllocateInfo { SType = StructureType.DescriptorSetAllocateInfo, DescriptorPool = _pool, DescriptorSetCount = 1, PSetLayouts = &lay };
        if (_vk.AllocateDescriptorSets(_device, &dsai, out _set) != Result.Success)
            throw new Exception("BcEncoder: failed to allocate descriptor set");
    }

    /// <summary>Bytes one block occupies for a given mode (BC1/BC4 = 8, BC3/BC5 = 16).</summary>
    public static uint BlockBytes(BcMode mode) => mode is BcMode.Bc1 or BcMode.Bc4 ? 8u : 16u;

    /// <summary>Total tightly-packed BC byte size across the full mip chain -- the size of the
    /// output buffer the caller must allocate before <see cref="RecordEncode"/>.</summary>
    public static ulong PackedSize(BcMode mode, uint baseW, uint baseH, uint mipLevels)
    {
        uint blockBytes = BlockBytes(mode);
        ulong total = 0;
        for (uint m = 0; m < mipLevels; m++)
        {
            uint mw = Math.Max(1u, baseW >> (int)m), mh = Math.Max(1u, baseH >> (int)m);
            total += (ulong)((mw + 3u) / 4u) * ((mh + 3u) / 4u) * blockBytes;
        }
        return total;
    }

    /// <summary>Byte offset of mip <paramref name="mip"/> within the packed output buffer (the
    /// same offset the caller passes to vkCmdCopyBufferToImage for that mip).</summary>
    public static ulong MipOffset(BcMode mode, uint baseW, uint baseH, uint mip)
        => PackedSize(mode, baseW, baseH, mip);   // sum of all preceding mips

    /// <summary>Records the per-mip encode dispatches into <paramref name="cmd"/>. <paramref name="srcView"/>
    /// must be a full-mip-chain SampledImage view in ShaderReadOnlyOptimal; <paramref name="dstBuffer"/>
    /// must be at least <see cref="PackedSize"/> bytes. The caller owns the compute->transfer barrier
    /// and the buffer->image copies that follow.</summary>
    public void RecordEncode(CommandBuffer cmd, ImageView srcView, Buffer dstBuffer, BcMode mode,
                             uint baseW, uint baseH, uint mipLevels)
    {
        var imgInfo = new DescriptorImageInfo { ImageView = srcView, ImageLayout = ImageLayout.ShaderReadOnlyOptimal };
        var bufInfo = new DescriptorBufferInfo { Buffer = dstBuffer, Offset = 0, Range = Vk.WholeSize };
        var writes  = stackalloc WriteDescriptorSet[2];
        writes[0] = new() { SType = StructureType.WriteDescriptorSet, DstSet = _set, DstBinding = 0, DescriptorType = DescriptorType.SampledImage,  DescriptorCount = 1, PImageInfo  = &imgInfo };
        writes[1] = new() { SType = StructureType.WriteDescriptorSet, DstSet = _set, DstBinding = 1, DescriptorType = DescriptorType.StorageBuffer, DescriptorCount = 1, PBufferInfo = &bufInfo };
        _vk.UpdateDescriptorSets(_device, 2, writes, 0, null);

        _vk.CmdBindPipeline(cmd, PipelineBindPoint.Compute, _pso);
        var set = _set;
        _vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute, _pipelineLayout, 0, 1, &set, 0, null);

        uint blockBytes = BlockBytes(mode);
        uint offset = 0;
        for (uint m = 0; m < mipLevels; m++)
        {
            uint mw = Math.Max(1u, baseW >> (int)m), mh = Math.Max(1u, baseH >> (int)m);
            uint bx = (mw + 3u) / 4u, by = (mh + 3u) / 4u;
            var push = new EncPush { Mode = (uint)mode, SrcMip = m, MipW = mw, MipH = mh, BlockOffset = offset };
            _vk.CmdPushConstants(cmd, _pipelineLayout, ShaderStageFlags.ComputeBit, 0, (uint)sizeof(EncPush), &push);
            _vk.CmdDispatch(cmd, (bx + 7u) / 8u, (by + 7u) / 8u, 1);
            offset += bx * by * blockBytes;
        }
    }

    public void Dispose()
    {
        if (_pso.Handle            != 0) _vk.DestroyPipeline(_device, _pso, null);
        if (_pipelineLayout.Handle != 0) _vk.DestroyPipelineLayout(_device, _pipelineLayout, null);
        if (_pool.Handle           != 0) _vk.DestroyDescriptorPool(_device, _pool, null);
        if (_setLayout.Handle      != 0) _vk.DestroyDescriptorSetLayout(_device, _setLayout, null);
    }
}