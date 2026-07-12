using System.Runtime.InteropServices;
using CadThingo.VulkanEngine.Renderer.FrameGraph;
using CadThingo.VulkanEngine.Renderer.Shaders;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Features.Tonemapping;

// Matches the TONEMAP_OPERATOR spec constant in Tonemap.slang. Read once at
// pipeline build; changing the operator requires Dispose + rebuild.
public enum TonemapOperator : uint
{
    Reinhard = 0,
    Filmic   = 1,
}

// 
//  Tone-map / post pass — samples HDRColor, writes FinalColor (LDR)
// Base type is namespace-qualified: the dead VulkanTut `CadThingo.GraphicsPipeline`
// namespace would otherwise shadow the `GraphicsPipeline` base from an enclosing-namespace
// lookup here under Features.
public sealed unsafe class TonemapPipeline : Pipelines.GraphicsPipeline
{
    // Push constants — fragment-only, 8 bytes total, well under the 128B
    // guaranteed minimum. Kept off the per-frame UBO because exposure/gamma
    // change rarely and don't deserve a descriptor binding of their own.
    [StructLayout(LayoutKind.Sequential)]
    struct TonemapPushConstants
    {
        public float Exposure;
        public float Gamma;
    }

    protected override string ShaderPath { get; } = ShaderPaths.Kernel("Tonemapping", "Tonemap");

    protected override Format[] ColorAttachmentFormats { get; } = new[] { Format.R8G8B8A8Unorm };

    // Surfaced as plain properties so the renderer (or imgui later) can adjust.
    // Defaults preserve the visual output of the pre-refactor inline tone-map.
    public float Exposure { get; set; } = 4.5f;
    public float Gamma    { get; set; } = 2.0f;

    /// <summary>Selects the tone-map curve via <c>constant_id 0</c> in
    /// Tonemap.slang. Read at each pipeline build; set then call
    /// <see cref="PipelineBase.Rebuild"/> to apply a change.</summary>
    public TonemapOperator Operator { get; set; } = TonemapOperator.Filmic;

    // Graph-baked pass set (set 0): the single HDR-input sampler, filled by each core's
    // TonemapModule from its scene-colour source. Name matches the TonemapPass Read bind; the
    // sampler is immutable in the layout, so only the view is written. Every core is
    // graph-resident now, so this is the ONLY way the HDR input is bound.
    private static readonly BindingDesc[] _passBindings =
    {
        new("hdrInput", 0, 0, DescriptorType.CombinedImageSampler, 1, ShaderStageFlags.FragmentBit),
    };

    public PassSetSpec PassSet => new(SetIndex: 0, DescriptorSetLayouts[0], _passBindings);

    public TonemapPipeline(Renderer renderer) : base(renderer)
    {
        PushConstantRanges = new[]
        {
            new PushConstantRange
            {
                StageFlags = ShaderStageFlags.FragmentBit,
                Offset     = 0,
                Size       = (uint)sizeof(TonemapPushConstants),
            }
        };
    }

    // TonemapModule passes its graph-baked HDR set (see PassSet).
    internal void Record(CommandBuffer cmd, Renderer.FrameContext ctx, ImageView finalColor, DescriptorSet hdrSet)
    {
        BeginRendering(cmd,
            ctx.RenderExtent,
            [finalColor],
            colorLoad: AttachmentLoadOp.DontCare
            );

        Vk!.CmdBindPipeline(cmd, PipelineBindPoint.Graphics, Handle);

        Viewport vp = new()
        {
            X = 0, Y = 0,
            Width = ctx.RenderExtent.Width, Height = ctx.RenderExtent.Height,
            MinDepth = 0.0f, MaxDepth = 1.0f,
        };
        Rect2D scissor = new(new Offset2D(0, 0), ctx.RenderExtent);
        Vk!.CmdSetViewport(cmd, 0, 1, &vp);
        Vk!.CmdSetScissor(cmd, 0, 1, &scissor);

        Vk!.CmdBindDescriptorSets(cmd, PipelineBindPoint.Graphics,
            Layout, 0, 1, &hdrSet, 0, null);
        PushConstants(cmd);

        Vk!.CmdDraw(cmd, 3, 1, 0, 0);

        EndRendering(cmd);
    }
    
    // Fullscreen triangle — no vertex inputs, no depth, no blend, no culling.

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

    // Wire constant_id 0 (TONEMAP_OPERATOR) on the fragment stage. The slang
    // declaration is uint, so we pack the enum value into a 4-byte slot.
    protected override int FillSpecializationData(
        int stageIdx,
        SpecializationMapEntry* entries,
        byte* data,
        out uint dataSize)
    {
        // ShaderStages default: [0]=VS, [1]=FS. Spec constant lives on FS only.
        if (stageIdx == 1)
        {
            entries[0] = new SpecializationMapEntry
            {
                ConstantID = 0,
                Offset     = 0,
                Size       = sizeof(uint),
            };
            *(uint*)data = (uint)Operator;
            dataSize = sizeof(uint);
            return 1;
        }
        dataSize = 0;
        return 0;
    }

    //  Descriptor layout — set 0, binding 0 = HDR input sampler
    
    protected override void CreateDescriptorSetLayouts()
    {
        DescriptorSetLayouts = new DescriptorSetLayout[1];
        OwnedDescriptorSetLayoutIndices = new[] { 0 };

        // The HDR sampler is baked in as an IMMUTABLE sampler: the graph-baked HDR set (see
        // PassSet) then only writes the view.
        Sampler hdrSampler = Renderer.gBufferSampler;
        var binding = new DescriptorSetLayoutBinding
        {
            Binding            = 0,
            DescriptorType     = DescriptorType.CombinedImageSampler,
            DescriptorCount    = 1,
            StageFlags         = ShaderStageFlags.FragmentBit,
            PImmutableSamplers = &hdrSampler,
        };
        var layoutInfo = new DescriptorSetLayoutCreateInfo
        {
            SType        = StructureType.DescriptorSetLayoutCreateInfo,
            BindingCount = 1,
            PBindings    = &binding,
        };
        if (Vk.CreateDescriptorSetLayout(Device, &layoutInfo, null, out DescriptorSetLayouts[0]) != Result.Success)
            throw new Exception("Failed to create tonemap descriptor set layout");
    }

    // No per-frame mapped buffers — the HDR image is graph-owned, single-buffered,
    // and tunables ride in push constants.
    protected override void CreateResources() { }

    // No pipeline-owned sets: every core's TonemapModule graph-bakes the HDR-input set (PassSet).
    protected override void CreateDescriptorSets() { }

    protected override void WriteDescriptors() { }

    public void PushConstants(CommandBuffer cmd)
    {
        var pc = new TonemapPushConstants { Exposure = Exposure, Gamma = Gamma };
        Vk.CmdPushConstants(cmd, Layout,
            ShaderStageFlags.FragmentBit,
            0, (uint)sizeof(TonemapPushConstants), &pc);
    }
}