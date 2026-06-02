using System.Numerics;
using System.Runtime.InteropServices;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer;

/// <summary>
/// Multiview graphics pipeline that renders a 6-face cubemap capture in one
/// pass. Each invocation of vkCmdDraw fans out across the 6 layers of the
/// attached cube image via <c>viewMask = 0x3F</c>; the vertex shader picks the
/// matching per-face VP matrix using <c>SV_ViewID</c>.
///
/// Reuses the bindless materials/instances/textures/samplers layout owned by
/// <see cref="ResourceManager"/> so the per-frame descriptor bind is identical
/// to the geometry pass. Set 0 is a private per-frame UBO carrying the 6 view
/// matrices + projection.
/// </summary>
public sealed unsafe class ProbeCapturePipeline : Pipelines.GraphicsPipeline
{
    // CPU mirror of the UBO. Layout matches ProbeCapture.slang::CaptureUBO.
    [StructLayout(LayoutKind.Sequential)]
    public struct CaptureUbo
    {
        // 6 view matrices, one per cube face. Order matches Vulkan cubemap
        // layer ordering (+X -X +Y -Y +Z -Z) — see ReflectionProbeSystem.BuildCaptureMatrices.
        public Matrix4x4 View0;
        public Matrix4x4 View1;
        public Matrix4x4 View2;
        public Matrix4x4 View3;
        public Matrix4x4 View4;
        public Matrix4x4 View5;
        public Matrix4x4 Proj;
    }

    // Push constants — model matrix + materialIndex per draw. 80B total; well
    // under the 128B guaranteed minimum for VkPhysicalDeviceLimits.maxPushConstantsSize.
    [StructLayout(LayoutKind.Sequential)]
    public struct CapturePushConst
    {
        public Matrix4x4 Model;
        public uint MaterialIndex;
        public uint Pad0;
        public uint Pad1;
        public uint Pad2;
    }

    private UboBuffer[] _ubos = new UboBuffer[Renderer.MAX_CONCURRENT_FRAMES];

    public ProbeCapturePipeline(Renderer renderer) : base(renderer)
    {
        // Matches the capture depth attachment created by ReflectionProbeSystem.
        DepthAttachmentFormat = Format.D32Sfloat;
    }

    // GraphicsPipeline overrides

    protected override string ShaderPath { get; } =
        @"C:\Users\jamie\RiderProjects\CadThingo\CadThingo\Assets\Shaders\ProbeCapture.spv";

    protected override Format[] ColorAttachmentFormats { get; } = [ Format.R16G16B16A16Sfloat ];

    // Six bits set — multiview fans each draw across all 6 layers of the
    // captureCubeImage. gl_ViewIndex (SV_ViewID) in the shader returns 0..5.
    protected override uint RenderingViewMask => 0x3Fu;

    protected override VertexInputBindingDescription[] GetVertexInputBindings()
        => new[] { Vertex.GetBindingDescription() };

    protected override VertexInputAttributeDescription[] GetVertexInputAttributes()
        => Vertex.GetAttributeDescriptions();

    protected override PipelineRasterizationStateCreateInfo BuildRasterizer() => new()
    {
        SType                   = StructureType.PipelineRasterizationStateCreateInfo,
        DepthClampEnable        = false,
        RasterizerDiscardEnable = false,
        PolygonMode             = PolygonMode.Fill,
        LineWidth               = 1.0f,
        // No back-face culling for the capture pass. Cube-face view matrices
        // flip winding on alternating faces; disabling culling sidesteps having
        // to flip front-face per-view, at the cost of a fraction more rasterized
        // pixels. Acceptable for 256² captures.
        CullMode                = CullModeFlags.None,
        FrontFace               = FrontFace.CounterClockwise,
        DepthBiasEnable         = false,
    };

    // Resource lifetime

    protected override void CreateDescriptorSetLayouts()
    {
        // Set 0 — private per-frame UBO.
        DescriptorSetLayoutBinding uboBinding = new()
        {
            Binding         = 0,
            DescriptorType  = DescriptorType.UniformBuffer,
            DescriptorCount = 1,
            StageFlags      = ShaderStageFlags.VertexBit,
        };
        DescriptorSetLayoutCreateInfo set0Info = new()
        {
            SType        = StructureType.DescriptorSetLayoutCreateInfo,
            BindingCount = 1,
            PBindings    = &uboBinding,
        };
        if (Vk.CreateDescriptorSetLayout(Device, &set0Info, null, out var set0) != Result.Success)
            throw new Exception("Failed to create probe capture descriptor set layout 0");

        DescriptorSetLayouts = new[] { set0, Engine.ResourceManager.GetBindlessLayout() };
        OwnedDescriptorSetLayoutIndices = new[] { 0 };

        PushConstantRanges = new[]
        {
            new PushConstantRange
            {
                StageFlags = ShaderStageFlags.VertexBit | ShaderStageFlags.FragmentBit,
                Offset     = 0,
                Size       = (uint)sizeof(CapturePushConst),
            },
        };
    }

    protected override void CreateResources()
    {
        for (int i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
            Gfx.CreateMappedUniformBuffer(sizeof(CaptureUbo), ref _ubos[i]);
    }

    protected override void CreateDescriptorSets()
    {
        var layouts = stackalloc DescriptorSetLayout[(int)Renderer.MAX_CONCURRENT_FRAMES];
        for (int i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++) layouts[i] = DescriptorSetLayouts[0];

        DescriptorSetAllocateInfo alloc = new()
        {
            SType              = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool     = Gfx.DescriptorPool,
            DescriptorSetCount = Renderer.MAX_CONCURRENT_FRAMES,
            PSetLayouts        = layouts,
        };
        DescriptorSets = new DescriptorSet[1][];
        DescriptorSets[0] = new DescriptorSet[Renderer.MAX_CONCURRENT_FRAMES];
        fixed (DescriptorSet* p = DescriptorSets[0])
        {
            if (Vk.AllocateDescriptorSets(Device, &alloc, p) != Result.Success)
                throw new Exception("Failed to allocate probe capture descriptor sets");
        }
    }

    protected override void WriteDescriptors()
    {
        for (int i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            DescriptorBufferInfo bufInfo = new()
            {
                Buffer = _ubos[i].buffer,
                Offset = 0,
                Range  = (ulong)sizeof(CaptureUbo),
            };
            WriteDescriptorSet write = new()
            {
                SType           = StructureType.WriteDescriptorSet,
                DstSet          = DescriptorSets[0][i],
                DstBinding      = 0,
                DstArrayElement = 0,
                DescriptorType  = DescriptorType.UniformBuffer,
                DescriptorCount = 1,
                PBufferInfo     = &bufInfo,
            };
            Vk.UpdateDescriptorSets(Device, 1, &write, 0, null);
        }
    }

    public override void Dispose()
    {
        for (int i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
            Gfx.DestroyBuffer(_ubos[i].buffer, _ubos[i].alloc);
        base.Dispose();
    }


    /// <summary>
    /// Uploads <paramref name="ubo"/> into the frame slot's mapped UBO. Cheap
    /// (~448B host write, host-coherent so no flush needed).
    /// </summary>
    public void WriteUbo(uint frameIndex, in CaptureUbo ubo)
    {
        *(CaptureUbo*)_ubos[frameIndex].mapped = ubo;
    }

    public DescriptorSet GetFrameSet(uint frameIndex) => DescriptorSets[0][frameIndex];
}
