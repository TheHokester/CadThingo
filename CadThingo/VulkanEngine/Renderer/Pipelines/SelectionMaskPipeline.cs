using System.Numerics;
using System.Runtime.InteropServices;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Pipelines;

// Selection-mask compute pipeline. Full-screen ray-query against the TLAS that
// writes a binary coverage mask of the selected entity into an R32F storage
// image. Mode-agnostic (it never touches the deferred / PT outputs), so the
// outline overlay built from this mask looks identical in every render mode.
//
// The mask image is renderer-owned (recreated on resize); this pipeline only
// borrows it via WriteMaskImageDescriptor. The TLAS is likewise external.
public sealed unsafe class SelectionMaskPipeline : ComputePipeline
{
    // Matches SelectionMask.slang::MaskParams. 96B, under the 128B minimum.
    [StructLayout(LayoutKind.Sequential)]
    private struct MaskPushConstants
    {
        public Matrix4x4 InvViewProj;    // 64
        public Vector4   CamPos;         // 16
        public Vector2   ScreenSize;     //  8
        public uint      SelectedIndex;  //  4
        public uint      _pad;           //  4
    }

    protected override string ShaderPath { get; } =
        @"C:\Users\jamie\RiderProjects\CadThingo\CadThingo\Assets\Shaders\SelectionMask.spv";

    public SelectionMaskPipeline(Renderer renderer) : base(renderer)
    {
        PushConstantRanges = new[]
        {
            new PushConstantRange
            {
                StageFlags = ShaderStageFlags.ComputeBit,
                Offset     = 0,
                Size       = (uint)sizeof(MaskPushConstants),
            }
        };
    }

    protected override void CreateDescriptorSetLayouts()
    {
        // Binding 2: entityInfo SSBO — the hit resolves to an entity via
        // entityInfo[InstanceCustomIndex + GeometryIndex].entityIndex now that
        // instances are per-cluster (custom index is no longer the entity).
        var bindings = stackalloc DescriptorSetLayoutBinding[3];
        bindings[0] = new() { Binding = 0, DescriptorType = DescriptorType.AccelerationStructureKhr, DescriptorCount = 1, StageFlags = ShaderStageFlags.ComputeBit };
        bindings[1] = new() { Binding = 1, DescriptorType = DescriptorType.StorageImage,             DescriptorCount = 1, StageFlags = ShaderStageFlags.ComputeBit };
        bindings[2] = new() { Binding = 2, DescriptorType = DescriptorType.StorageBuffer,            DescriptorCount = 1, StageFlags = ShaderStageFlags.ComputeBit };

        DescriptorSetLayoutCreateInfo info = new()
        {
            SType        = StructureType.DescriptorSetLayoutCreateInfo,
            BindingCount = 3,
            PBindings    = bindings,
        };
        if (Vk.CreateDescriptorSetLayout(Device, &info, null, out var layout) != Result.Success)
            throw new Exception("Failed to create selection-mask descriptor set layout");
        DescriptorSetLayouts            = new[] { layout };
        OwnedDescriptorSetLayoutIndices = new[] { 0 };
    }

    protected override void CreateDescriptorSets()
    {
        var layout = DescriptorSetLayouts[0];
        DescriptorSetAllocateInfo alloc = new()
        {
            SType              = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool     = Gfx.DescriptorPool,
            DescriptorSetCount = 1,
            PSetLayouts        = &layout,
        };
        DescriptorSets    = new DescriptorSet[1][];
        DescriptorSets[0] = new DescriptorSet[1];
        fixed (DescriptorSet* p = DescriptorSets[0])
            if (Vk.AllocateDescriptorSets(Device, &alloc, p) != Result.Success)
                throw new Exception("Failed to allocate selection-mask descriptor set");
    }

    // Both descriptors target external resources (TLAS / mask image) whose
    // handles change on rebuild / resize, so they're written explicitly.
    protected override void WriteDescriptors() { }

    /// <summary>Binding 0: TLAS. Call after InitRayQuery and on every rebuild.</summary>
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

    /// <summary>Binding 1: the R32F mask storage image (GENERAL layout). Call at
    /// init and on every render-target resize.</summary>
    public void WriteMaskImageDescriptor(ImageView maskView)
    {
        DescriptorImageInfo info = new() { ImageView = maskView, ImageLayout = ImageLayout.General };
        var write = new WriteDescriptorSet
        {
            SType           = StructureType.WriteDescriptorSet,
            DstSet          = DescriptorSets[0][0],
            DstBinding      = 1,
            DescriptorType  = DescriptorType.StorageImage,
            DescriptorCount = 1,
            PImageInfo      = &info,
        };
        Vk.UpdateDescriptorSets(Device, 1, &write, 0, null);
    }

    /// <summary>Records the full-screen mask dispatch (8×8 threads/group, matching
    /// [numthreads(8,8,1)]). The mask image must be in GENERAL before this call.</summary>
    public void Record(CommandBuffer cmd, in Matrix4x4 invViewProj, Vector3 camPos,
                       Extent2D extent, uint selectedIndex)
    {
        Vk.CmdBindPipeline(cmd, PipelineBindPoint.Compute, PipelineHandle);
        var dset = DescriptorSets[0][0];
        Vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute, PipelineLayoutHandle, 0, 1, &dset, 0, null);

        var push = new MaskPushConstants
        {
            InvViewProj   = invViewProj,
            CamPos        = new Vector4(camPos, 1f),
            ScreenSize    = new Vector2(extent.Width, extent.Height),
            SelectedIndex = selectedIndex,
        };
        Vk.CmdPushConstants(cmd, PipelineLayoutHandle, ShaderStageFlags.ComputeBit,
            0, (uint)sizeof(MaskPushConstants), &push);

        uint gx = (extent.Width  + 7u) / 8u;
        uint gy = (extent.Height + 7u) / 8u;
        Vk.CmdDispatch(cmd, gx, gy, 1);
    }
}