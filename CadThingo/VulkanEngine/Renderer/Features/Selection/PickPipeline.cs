using System.Numerics;
using System.Runtime.InteropServices;
using CadThingo.VulkanEngine.Renderer.Pipelines;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Features.Selection;

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

    protected override string ShaderPath { get; } = ShaderPaths.Kernel("Selection", "PickCompute");

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
        if (_resultBuffer.Handle != 0) Gfx.DestroyBuffer(_resultBuffer, _resultAlloc);
        base.Dispose();
    }

    // Set 0 borrowed from the registry (sceneTlas + sceneEntityInfo); set 1 owns the result SSBO.
    private const int SetScene  = 0;
    private const int SetResult = 1;

    protected override void CreateDescriptorSetLayouts()
    {
        // Set 1 binding 0: result SSBO (single uint). TLAS + entityInfo now resolve
        // through the scene set (sceneTlas / sceneEntityInfo), so the hit still maps to
        // an entity via sceneEntityInfo[InstanceCustomIndex + GeometryIndex].entityIndex.
        var binding = new DescriptorSetLayoutBinding
        {
            Binding = 0, DescriptorType = DescriptorType.StorageBuffer, DescriptorCount = 1, StageFlags = ShaderStageFlags.ComputeBit,
        };
        DescriptorSetLayoutCreateInfo info = new()
        {
            SType        = StructureType.DescriptorSetLayoutCreateInfo,
            BindingCount = 1,
            PBindings    = &binding,
        };
        if (Vk.CreateDescriptorSetLayout(Device, &info, null, out var layout) != Result.Success)
            throw new Exception("Failed to create pick descriptor set layout");

        DescriptorSetLayouts                 = new DescriptorSetLayout[2];
        DescriptorSetLayouts[SetScene]       = Renderer.descriptorRegistry.SceneSetLayout;
        DescriptorSetLayouts[SetResult]      = layout;
        OwnedDescriptorSetLayoutIndices      = new[] { SetResult };
    }

    protected override void CreateResources()
    {
        Gfx.CreateBuffer(sizeof(uint),
            BufferUsageFlags.StorageBufferBit,
            MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit,
            out _resultBuffer, out _resultAlloc);
        _resultMapped = Gfx.Allocator.GetMapped(_resultAlloc);
    }

    protected override void CreateDescriptorSets()
    {
        // Set 0 is registry-owned; Record binds descriptorRegistry.SceneSet(frame).
        var layout = DescriptorSetLayouts[SetResult];
        DescriptorSetAllocateInfo alloc = new()
        {
            SType              = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool     = Gfx.DescriptorPool,
            DescriptorSetCount = 1,
            PSetLayouts        = &layout,
        };
        DescriptorSets            = new DescriptorSet[2][];
        DescriptorSets[SetScene]  = null;
        DescriptorSets[SetResult] = new DescriptorSet[1];
        fixed (DescriptorSet* p = DescriptorSets[SetResult])
            if (Vk.AllocateDescriptorSets(Device, &alloc, p) != Result.Success)
                throw new Exception("Failed to allocate pick descriptor set");
    }

    protected override void WriteDescriptors()
    {
        // Result buffer is the only owned resource; TLAS + entityInfo ride the scene set.
        DescriptorBufferInfo info = new() { Buffer = _resultBuffer, Offset = 0, Range = sizeof(uint) };
        var write = new WriteDescriptorSet
        {
            SType           = StructureType.WriteDescriptorSet,
            DstSet          = DescriptorSets[SetResult][0],
            DstBinding      = 0,
            DescriptorType  = DescriptorType.StorageBuffer,
            DescriptorCount = 1,
            PBufferInfo     = &info,
        };
        Vk.UpdateDescriptorSets(Device, 1, &write, 0, null);
    }

    /// <summary>Records the 1×1×1 pick dispatch into <paramref name="cmd"/>.
    /// The caller submits the command buffer and waits (QueueWaitIdle) before
    /// calling <see cref="ReadResult"/>. Binds the scene set for the current frame:
    /// picking runs out-of-band before this frame's BeginFrame, so that set still
    /// holds the TLAS state that produced the image the user clicked on.</summary>
    public void Record(CommandBuffer cmd, in Matrix4x4 invViewProj, Vector3 camPos,
                       Vector2 screenSize, uint pixelX, uint pixelY)
    {
        Vk.CmdBindPipeline(cmd, PipelineBindPoint.Compute, PipelineHandle);

        // Scene set (zero dynamic offset: pick params stay push constants, the (0,0)
        // arena slot is unused here) + the owned result set.
        uint zeroOffset = 0;
        var sets = stackalloc DescriptorSet[2]
        {
            Renderer.descriptorRegistry.SceneSet(Renderer.currentFrame),
            DescriptorSets[SetResult][0],
        };
        Vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute, PipelineLayoutHandle, 0, 2, sets, 1, &zeroOffset);

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