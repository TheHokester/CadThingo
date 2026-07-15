using System.Runtime.InteropServices;
using CadThingo.VulkanEngine.Renderer.FrameGraph;
using CadThingo.VulkanEngine.Renderer.Shaders;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Features.Tonemapping;

// Values must match the TONEMAP_OPERATOR branch in Tonemap.slang. Read once at pipeline build;
// changing the operator requires a rebuild.
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
    // Mirrors TonemapParams in Tonemap.slang. Kept off the per-frame UBO because exposure/gamma
    // change rarely and don't deserve a descriptor binding of their own. The reflected push range
    // drives the layout; Initialize asserts this struct still matches it.
    [StructLayout(LayoutKind.Sequential)]
    struct TonemapPushConstants
    {
        public float Exposure;
        public float Gamma;
    }

    protected override ShaderCompileRequest? Program =>
        new("Tonemapping/Tonemap", ["VSMain", "PSMain"], [], []);

    protected override Format[] ColorAttachmentFormats { get; } = new[] { Format.R8G8B8A8Unorm };

    // Surfaced as plain properties so the renderer (or imgui later) can adjust.
    // Defaults preserve the visual output of the pre-refactor inline tone-map.
    public float Exposure { get; set; } = 4.5f;
    public float Gamma    { get; set; } = 2.0f;

    /// <summary>Selects the tone-map curve via the TONEMAP_OPERATOR spec constant. Read at each
    /// pipeline build; set then call <see cref="PipelineBase.Rebuild"/> to apply a change.</summary>
    public TonemapOperator Operator { get; set; } = TonemapOperator.Filmic;

    // Graph-baked pass set: the single HDR-input sampler, filled by each core's TonemapModule from
    // its scene-colour source. Contents AND set index both come from reflection, so hdrInput's
    // declaration in Tonemap.slang is the only place either is stated. Every core is graph-resident
    // now, so this is the ONLY way the HDR input is bound.
    private uint PassSetIndex => Reflected!.Reflection.Bindings.Select(b => b.Set).Distinct().Single();

    public PassSetSpec PassSet =>
        new(PassSetIndex, DescriptorSetLayouts[PassSetIndex], ReflectedBindings(PassSetIndex));

    public TonemapPipeline(GpuContext gpu, Renderer renderer) : base( gpu, renderer) { }

    // The graph writes only the view into the pass set, so the sampler is pinned here.
    protected override Sampler? ImmutableSamplerFor(string bindingName)
        => bindingName == "hdrInput" ? Renderer.gBufferSampler : null;

    protected override SpecValues? Specialization =>
        new SpecValues().Set("TONEMAP_OPERATOR", (uint)Operator);

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

    // Only the pass set is real; any index below it is a gap this pipeline never binds.
    protected override void CreateDescriptorSetLayouts()
    {
        uint set = PassSetIndex;
        DescriptorSetLayouts = new DescriptorSetLayout[set + 1];
        for (int i = 0; i < set; i++) DescriptorSetLayouts[i] = Registry.EmptySetLayout;
        DescriptorSetLayouts[set] = CreateReflectedSetLayout(set);
        OwnedDescriptorSetLayoutIndices = new[] { (int)set };
    }

    // No per-frame mapped buffers — the HDR image is graph-owned, single-buffered,
    // and tunables ride in push constants. The size check is the one thing reflection cannot
    // enforce on its own: the C# mirror of TonemapParams has to keep matching the shader.
    protected override void CreateResources()
    {
        uint reflected = PushConstantRanges[0].Size;
        if (reflected != (uint)sizeof(TonemapPushConstants))
            throw new Exception(
                $"TonemapPushConstants is {sizeof(TonemapPushConstants)} bytes but Tonemap.slang " +
                $"reflects {reflected}");
    }

    // No pipeline-owned sets: every core's TonemapModule graph-bakes the HDR-input set (PassSet).
    protected override void CreateDescriptorSets() { }

    protected override void WriteDescriptors() { }

    // Stages come from the same reflected range the layout was built from: vkCmdPushConstants
    // requires the two to agree, so neither side is allowed to name a stage mask of its own.
    public void PushConstants(CommandBuffer cmd)
    {
        var pc = new TonemapPushConstants { Exposure = Exposure, Gamma = Gamma };
        Vk.CmdPushConstants(cmd, Layout,
            PushConstantRanges[0].StageFlags,
            0, (uint)sizeof(TonemapPushConstants), &pc);
    }
}