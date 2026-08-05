using System.Numerics;
using System.Runtime.InteropServices;
using CadThingo.VulkanEngine.Renderer.Pipelines;
using CadThingo.VulkanEngine.Renderer.Slang;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Features.Selection;

// Selection-outline overlay. Fullscreen pass run on FinalColor (LoadOp.Load)
// after the active render mode has composited the scene; reads the R32F
// selection mask and draws an outer ring around the selected entity's
// silhouette, discarding every non-edge pixel so the rest of the image is
// untouched. Mode-agnostic because it only depends on FinalColor + the mask.
public sealed unsafe class OutlinePipeline : GraphicsPipeline
{
    // Matches Outline.slang::OutlineParams. 32B, under the 128B minimum.
    [StructLayout(LayoutKind.Sequential)]
    private struct OutlinePushConstants
    {
        public Vector4 Color;        // rgb outline color (a unused)
        public int     ScreenW;
        public int     ScreenH;
        public int     Thickness;    // outline radius in pixels
        public int     _pad;
    }

    protected override ShaderCompileRequest? Program => new("Selection/Outline", ["VSMain", "PSMain"], [], []);

    // Writes FinalColor (LDR), same format as the tonemap pass.
    protected override Format[] ColorAttachmentFormats { get; } = new[] { Format.R8G8B8A8Unorm };

    public Vector3 Color     { get; set; } = new(1.0f, 0.55f, 0.1f);   // editor orange
    public int     Thickness { get; set; } = 3;

    public OutlinePipeline(GpuContext gpu) : base(gpu) { }

    // Fullscreen triangle — no vertex inputs, no depth, no cull, no blend.
    protected override PipelineDepthStencilStateCreateInfo BuildDepthStencil() => new()
    {
        SType                 = StructureType.PipelineDepthStencilStateCreateInfo,
        DepthTestEnable       = false,
        DepthWriteEnable      = false,
        DepthCompareOp        = CompareOp.Always,
        DepthBoundsTestEnable = false,
        StencilTestEnable     = false,
    };

    protected override PipelineRasterizationStateCreateInfo BuildRasterizer() => new()
    {
        SType                   = StructureType.PipelineRasterizationStateCreateInfo,
        DepthClampEnable        = false,
        RasterizerDiscardEnable = false,
        PolygonMode             = PolygonMode.Fill,
        LineWidth               = 1.0f,
        CullMode                = CullModeFlags.None,
        FrontFace               = FrontFace.CounterClockwise,
        DepthBiasEnable         = false,
    };

    protected override void CreateDescriptorSetLayouts()
    {
        // Private set 0 (the mask, fetched via .Load with no sampler): this pass is not on the
        // scene set, so the layout is just the reflected set.
        DescriptorSetLayouts            = new[] { CreateReflectedSetLayout(0) };
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
                throw new Exception("Failed to allocate outline descriptor set");
    }

    // The C# mirror of OutlineParams is the one thing reflection cannot keep honest.
    protected override void CreateResources()
    {
        uint reflected = PushConstantRanges[0].Size;
        if (reflected != (uint)sizeof(OutlinePushConstants))
            throw new Exception(
                $"OutlinePushConstants is {sizeof(OutlinePushConstants)} bytes but Outline.slang " +
                $"reflects {reflected}");
    }

    // Mask view only exists after CreateSelectionResources, so it's written
    // explicitly by the renderer (and re-written on resize).
    protected override void WriteDescriptors() { }

    /// <summary>Binding 0: the R32F selection mask (SHADER_READ_ONLY layout).
    /// Call at init and on every render-target resize.</summary>
    public void WriteMaskDescriptor(ImageView maskView)
    {
        DescriptorImageInfo info = new() { ImageView = maskView, ImageLayout = ImageLayout.ShaderReadOnlyOptimal };
        var write = new WriteDescriptorSet
        {
            SType           = StructureType.WriteDescriptorSet,
            DstSet          = DescriptorSets[0][0],
            DstBinding      = 0,
            DescriptorType  = DescriptorType.SampledImage,
            DescriptorCount = 1,
            PImageInfo      = &info,
        };
        Vk.UpdateDescriptorSets(Device, 1, &write, 0, null);
    }

    /// <summary>Records the outline draw into <paramref name="finalColor"/>
    /// (LoadOp.Load — only edge pixels are written, everything else discards).
    /// The mask must be in SHADER_READ_ONLY before this call.</summary>
    public void Record(CommandBuffer cmd, Extent2D extent, ImageView finalColor)
    {
        BeginRendering(cmd, extent, [finalColor], colorLoad: AttachmentLoadOp.Load);

        Vk.CmdBindPipeline(cmd, PipelineBindPoint.Graphics, Handle);

        Viewport vp = new()
        {
            X = 0, Y = 0, Width = extent.Width, Height = extent.Height,
            MinDepth = 0.0f, MaxDepth = 1.0f,
        };
        Rect2D scissor = new(new Offset2D(0, 0), extent);
        Vk.CmdSetViewport(cmd, 0, 1, &vp);
        Vk.CmdSetScissor(cmd, 0, 1, &scissor);

        var set = GetDescriptorSet(0, 0);
        Vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.Graphics, Layout, 0, 1, &set, 0, null);

        var pc = new OutlinePushConstants
        {
            Color     = new Vector4(Color, 1f),
            ScreenW   = (int)extent.Width,
            ScreenH   = (int)extent.Height,
            Thickness = Thickness,
        };
        // Reflection owns the range's stage mask; vkCmdPushConstants requires the two to agree.
        Vk.CmdPushConstants(cmd, Layout, PushConstantRanges[0].StageFlags,
            0, (uint)sizeof(OutlinePushConstants), &pc);

        Vk.CmdDraw(cmd, 3, 1, 0, 0);

        EndRendering(cmd);
    }
}