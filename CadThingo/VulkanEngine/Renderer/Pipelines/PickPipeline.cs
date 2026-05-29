using System.Numerics;
using System.Runtime.InteropServices;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Pipelines;

// Object-picking compute pipeline. Casts one ray-query through the clicked
// pixel against the scene TLAS; the committed hit's InstanceCustomIndex (==
// scene entity index, set in RebuildTlas) is written to a host-visible result
// buffer.
//
// Dispatched out-of-band via a single-time command submit (see
// Renderer.ProcessPickRequest), not in the per-frame command buffer — picking
// is a rare on-click event, so the single-time submit's QueueWaitIdle keeps the
// readback trivially synchronous instead of needing a ring of fenced buffers.
public sealed unsafe class PickPipeline : ComputePipeline
{
    // Sentinel written on a ray miss — matches PICK_NONE in PickCompute.slang.
    public const uint PickNone = 0xFFFFFFFFu;

    // Matches PickCompute.slang::PickParams byte-for-byte. 96B, under the 128B
    // Vulkan push-constant minimum.
    [StructLayout(LayoutKind.Sequential)]
    private struct PickPushConstants
    {
        public Matrix4x4 InvViewProj;   // 64
        public Vector4   CamPos;        // 16
        public Vector2   ScreenSize;    //  8
        public uint      PixelX;        //  4
        public uint      PixelY;        //  4
    }

    protected override string ShaderPath { get; } =
        @"C:\Users\jamie\RiderProjects\CadThingo\CadThingo\Assets\Shaders\PickCompute.spv";

    // 4B result (entity index or PickNone). Host-visible + coherent so the
    // single-time submit's QueueWaitIdle is all the synchronisation the readback
    // needs.
    private Buffer   _resultBuffer;
    private SubAlloc _resultAlloc;
    private void*    _resultMapped;

    public PickPipeline(Renderer renderer) : base(renderer)
    {
        PushConstantRanges = new[]
        {
            new PushConstantRange
            {
                StageFlags = ShaderStageFlags.ComputeBit,
                Offset     = 0,
                Size       = (uint)sizeof(PickPushConstants),
            }
        };
    }

    public override void Dispose()
    {
        _resultMapped = null;
        if (_resultBuffer.Handle != 0) Renderer.DestroyBuffer(_resultBuffer, _resultAlloc);
        base.Dispose();
    }

    protected override void CreateDescriptorSetLayouts()
    {
        // Binding 0: TLAS. Binding 1: result SSBO (single uint). Binding 2:
        // entityInfo SSBO — the hit resolves to an entity via
        // entityInfo[InstanceCustomIndex + GeometryIndex].entityIndex now that
        // instances are per-cluster (custom index is no longer the entity).
        var bindings = stackalloc DescriptorSetLayoutBinding[3];
        bindings[0] = new() { Binding = 0, DescriptorType = DescriptorType.AccelerationStructureKhr, DescriptorCount = 1, StageFlags = ShaderStageFlags.ComputeBit };
        bindings[1] = new() { Binding = 1, DescriptorType = DescriptorType.StorageBuffer,            DescriptorCount = 1, StageFlags = ShaderStageFlags.ComputeBit };
        bindings[2] = new() { Binding = 2, DescriptorType = DescriptorType.StorageBuffer,            DescriptorCount = 1, StageFlags = ShaderStageFlags.ComputeBit };

        DescriptorSetLayoutCreateInfo info = new()
        {
            SType        = StructureType.DescriptorSetLayoutCreateInfo,
            BindingCount = 3,
            PBindings    = bindings,
        };
        if (Vk.CreateDescriptorSetLayout(Device, &info, null, out var layout) != Result.Success)
            throw new Exception("Failed to create pick descriptor set layout");
        DescriptorSetLayouts            = new[] { layout };
        OwnedDescriptorSetLayoutIndices = new[] { 0 };
    }

    protected override void CreateResources()
    {
        Renderer.CreateBuffer(sizeof(uint),
            BufferUsageFlags.StorageBufferBit,
            MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit,
            out _resultBuffer, out _resultAlloc);
        _resultMapped = Renderer.memAllocator.GetMapped(_resultAlloc);
    }

    protected override void CreateDescriptorSets()
    {
        var layout = DescriptorSetLayouts[0];
        DescriptorSetAllocateInfo alloc = new()
        {
            SType              = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool     = Renderer.descriptorPool,
            DescriptorSetCount = 1,
            PSetLayouts        = &layout,
        };
        DescriptorSets    = new DescriptorSet[1][];
        DescriptorSets[0] = new DescriptorSet[1];
        fixed (DescriptorSet* p = DescriptorSets[0])
            if (Vk.AllocateDescriptorSets(Device, &alloc, p) != Result.Success)
                throw new Exception("Failed to allocate pick descriptor set");
    }

    protected override void WriteDescriptors()
    {
        // Result buffer is owned here. The TLAS lives elsewhere and changes
        // handle on every rebuild, so it's written separately via WriteTlasDescriptor.
        DescriptorBufferInfo info = new() { Buffer = _resultBuffer, Offset = 0, Range = sizeof(uint) };
        var write = new WriteDescriptorSet
        {
            SType           = StructureType.WriteDescriptorSet,
            DstSet          = DescriptorSets[0][0],
            DstBinding      = 1,
            DescriptorType  = DescriptorType.StorageBuffer,
            DescriptorCount = 1,
            PBufferInfo     = &info,
        };
        Vk.UpdateDescriptorSets(Device, 1, &write, 0, null);
    }

    /// <summary>Binding 2: the renderer-owned ShadowEntityInfo SSBO. Call after
    /// InitRayQuery and whenever the buffer reallocates (resize).</summary>
    public void WriteEntityInfoDescriptor()
    {
        var buf = Renderer.ShadowInfoBuffer;
        if (buf.Handle == 0) return;
        DescriptorBufferInfo info = new() { Buffer = buf, Offset = 0, Range = Renderer.ShadowInfoBufferSize };
        var write = new WriteDescriptorSet
        {
            SType           = StructureType.WriteDescriptorSet,
            DstSet          = DescriptorSets[0][0],
            DstBinding      = 2,
            DescriptorType  = DescriptorType.StorageBuffer,
            DescriptorCount = 1,
            PBufferInfo     = &info,
        };
        Vk.UpdateDescriptorSets(Device, 1, &write, 0, null);
    }

    /// <summary>Binding 0: TLAS. Call after InitRayQuery and on every TLAS
    /// rebuild that recreates the handle.</summary>
    public void WriteTlasDescriptor(AccelerationStructureKHR tlas)
    {
        if (tlas.Handle == 0) return;
        var tlasH = tlas;
        var asWrite = new WriteDescriptorSetAccelerationStructureKHR
        {
            SType                      = StructureType.WriteDescriptorSetAccelerationStructureKhr,
            AccelerationStructureCount = 1,
            PAccelerationStructures    = &tlasH,
        };
        var write = new WriteDescriptorSet
        {
            SType           = StructureType.WriteDescriptorSet,
            PNext           = &asWrite,
            DstSet          = DescriptorSets[0][0],
            DstBinding      = 0,
            DescriptorType  = DescriptorType.AccelerationStructureKhr,
            DescriptorCount = 1,
        };
        Vk.UpdateDescriptorSets(Device, 1, &write, 0, null);
    }

    /// <summary>Records the 1×1×1 pick dispatch into <paramref name="cmd"/>.
    /// The caller submits the command buffer and waits (QueueWaitIdle) before
    /// calling <see cref="ReadResult"/>.</summary>
    public void Record(CommandBuffer cmd, in Matrix4x4 invViewProj, Vector3 camPos,
                       Vector2 screenSize, uint pixelX, uint pixelY)
    {
        Vk.CmdBindPipeline(cmd, PipelineBindPoint.Compute, PipelineHandle);
        var dset = DescriptorSets[0][0];
        Vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute, PipelineLayoutHandle, 0, 1, &dset, 0, null);

        var push = new PickPushConstants
        {
            InvViewProj = invViewProj,
            CamPos      = new Vector4(camPos, 1f),
            ScreenSize  = screenSize,
            PixelX      = pixelX,
            PixelY      = pixelY,
        };
        Vk.CmdPushConstants(cmd, PipelineLayoutHandle, ShaderStageFlags.ComputeBit,
            0, (uint)sizeof(PickPushConstants), &push);

        Vk.CmdDispatch(cmd, 1, 1, 1);
    }

    /// <summary>Reads back the index the last dispatch wrote. Only valid once
    /// the command buffer that ran <see cref="Record"/> has completed.</summary>
    public uint ReadResult() => *(uint*)_resultMapped;
}