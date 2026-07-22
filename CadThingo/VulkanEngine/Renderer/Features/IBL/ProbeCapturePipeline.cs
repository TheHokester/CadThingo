using System.Numerics;
using System.Runtime.InteropServices;
using CadThingo.VulkanEngine.Renderer.Pipelines;
using CadThingo.VulkanEngine.Renderer.Shaders;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Features.IBL;

/// <summary>
/// Multiview graphics pipeline that renders a 6-face cubemap capture in one
/// pass. Each invocation of vkCmdDraw fans out across the 6 layers of the
/// attached cube image via <c>viewMask = 0x3F</c>; the vertex shader picks the
/// matching per-face VP matrix using <c>SV_ViewID</c>.
///
/// Consumes the registry's scene set for materials/instances/textures/samplers, so
/// the per-frame descriptor bind is identical to the geometry pass and this pipeline
/// owns no descriptor sets of its own. The 6 view matrices + projection ride the scene
/// set's (0,0) dynamic constant slot: ReflectionProbeSystem pushes a <see cref="CaptureUbo"/>
/// into the PassConstantArena per capture and binds with the returned dynamic offset.
/// </summary>
public sealed unsafe class ProbeCapturePipeline : GraphicsPipeline
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

    public ProbeCapturePipeline(GpuContext gpu, Renderer renderer) : base(gpu, renderer)
    {
        // Matches the capture depth attachment created by ReflectionProbeSystem.
        DepthAttachmentFormat = Format.D32Sfloat;
    }

    // GraphicsPipeline overrides

    protected override ShaderCompileRequest? Program => new("IBL/ProbeCapture", ["VSMain", "PSMain"], [], []);

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

        DescriptorSetLayouts = Registry.BuildPipelineSetLayouts(passSet: null);
        OwnedDescriptorSetLayoutIndices = [];
    
    }

    protected override void CreateResources()
    {
        var reflected = PushConstantRanges[0].Size;
        if (reflected != (uint)sizeof(CapturePushConst))
        {
            throw new Exception(
                $"CapturePushConst is {sizeof(CapturePushConst)} bytes but ProbeCapture.slang " +
                $"reflects {reflected}");
        }
    }

    
}
