using System.Numerics;
using System.Runtime.InteropServices;
using CadThingo.VulkanEngine.Renderer.Pipelines;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Features.Selection;

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

    protected override string ShaderPath { get; } = ShaderPaths.Kernel("Selection", "SelectionMask");

    public SelectionMaskPipeline(GpuContext gpu,Renderer renderer) : base(gpu, renderer)
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

    // Set 0 borrowed from the registry (sceneTlas + sceneEntityInfo); set 1 owns the mask image.
    private const int SetScene = 0;
    private const int SetMask  = 1;

    protected override void CreateDescriptorSetLayouts()
    {
        // Set 1 binding 0: the R32F coverage mask. TLAS + entityInfo now resolve
        // through the scene set (sceneTlas / sceneEntityInfo).
        var binding = new DescriptorSetLayoutBinding
        {
            Binding = 0, DescriptorType = DescriptorType.StorageImage, DescriptorCount = 1, StageFlags = ShaderStageFlags.ComputeBit,
        };
        DescriptorSetLayoutCreateInfo info = new()
        {
            SType        = StructureType.DescriptorSetLayoutCreateInfo,
            BindingCount = 1,
            PBindings    = &binding,
        };
        if (Vk.CreateDescriptorSetLayout(Device, &info, null, out var layout) != Result.Success)
            throw new Exception("Failed to create selection-mask descriptor set layout");

        DescriptorSetLayouts            = new DescriptorSetLayout[2];
        DescriptorSetLayouts[SetScene]  = Registry.SceneSetLayout;
        DescriptorSetLayouts[SetMask]   = layout;
        OwnedDescriptorSetLayoutIndices = new[] { SetMask };
    }

    protected override void CreateDescriptorSets()
    {
        // Set 0 is registry-owned; Record binds descriptorRegistry.SceneSet(frame).
        var layout = DescriptorSetLayouts[SetMask];
        DescriptorSetAllocateInfo alloc = new()
        {
            SType              = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool     = Gfx.DescriptorPool,
            DescriptorSetCount = 1,
            PSetLayouts        = &layout,
        };
        DescriptorSets           = new DescriptorSet[2][];
        DescriptorSets[SetScene] = null;
        DescriptorSets[SetMask]  = new DescriptorSet[1];
        fixed (DescriptorSet* p = DescriptorSets[SetMask])
            if (Vk.AllocateDescriptorSets(Device, &alloc, p) != Result.Success)
                throw new Exception("Failed to allocate selection-mask descriptor set");
    }

    // The mask image is the only owned resource; its handle changes on resize, so it's
    // written explicitly. TLAS + entityInfo ride the scene set.
    protected override void WriteDescriptors() { }

    /// <summary>Set 1 binding 0: the R32F mask storage image (GENERAL layout). Call at
    /// init and on every render-target resize.</summary>
    public void WriteMaskImageDescriptor(ImageView maskView)
    {
        DescriptorImageInfo info = new() { ImageView = maskView, ImageLayout = ImageLayout.General };
        var write = new WriteDescriptorSet
        {
            SType           = StructureType.WriteDescriptorSet,
            DstSet          = DescriptorSets[SetMask][0],
            DstBinding      = 0,
            DescriptorType  = DescriptorType.StorageImage,
            DescriptorCount = 1,
            PImageInfo      = &info,
        };
        Vk.UpdateDescriptorSets(Device, 1, &write, 0, null);
    }

    /// <summary>Records the full-screen mask dispatch (8×8 threads/group, matching
    /// [numthreads(8,8,1)]). The mask image must be in GENERAL before this call. Runs in
    /// the per-frame command buffer after BeginFrame, so the current frame's scene set is
    /// fresh.</summary>
    public void Record(CommandBuffer cmd, in Matrix4x4 invViewProj, Vector3 camPos,
                       Extent2D extent, uint selectedIndex)
    {
        Vk.CmdBindPipeline(cmd, PipelineBindPoint.Compute, PipelineHandle);

        // Scene set (zero dynamic offset: mask params stay push constants) + owned mask set.
        uint zeroOffset = 0;
        var sets = stackalloc DescriptorSet[2]
        {
            Registry.SceneSet(Renderer.currentFrame),
            DescriptorSets[SetMask][0],
        };
        Vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute, PipelineLayoutHandle, 0, 2, sets, 1, &zeroOffset);

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