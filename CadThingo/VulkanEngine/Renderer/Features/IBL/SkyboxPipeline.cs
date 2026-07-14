using System.Numerics;
using System.Runtime.InteropServices;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Features.IBL;


//  Skybox pass — samples envCube along per-pixel view rays. Renders between
//  the deferred lighting pass and the transparent forward+ pass; depth test
//  EQUAL against 1.0 so it lights up only the cleared (sky) pixels of the
//  depth buffer the geometry pass left behind.

public sealed unsafe class SkyboxPipeline : Pipelines.GraphicsPipeline
{
    // Mirrors Skybox.slang::SkyboxUBO. Std140 — invViewProj is 64B, then 16B
    // camPos, then intensity + 3×4B trailing pad to round to 96B total.
    [StructLayout(LayoutKind.Sequential)]
    struct SkyboxUBO
    {
        public Matrix4x4 invViewProj;
        public Vector4   camPos;
        public float     intensity;
        public float     _pad0;
        public float     _pad1;
        public float     _pad2;
    }

    protected override string ShaderPath { get; } = ShaderPaths.Kernel("IBL", "Skybox");

    // Writes the same HDRColor as the lighting / transparent passes so it gets
    // tone-mapped with everything else.
    protected override Format[] ColorAttachmentFormats { get; } = new[] { Format.R16G16B16A16Sfloat };

    public SkyboxPipeline(Renderer renderer) : base(renderer)
    {
        DepthAttachmentFormat = Gfx.FindDepthFormat();
    }

    private UboBuffer[] FrameUniformBuffers = new UboBuffer[Renderer.MAX_CONCURRENT_FRAMES];

    public override void Dispose()
    {
        foreach (var b in FrameUniformBuffers) Gfx.DestroyBuffer(b.buffer, b.alloc);
        base.Dispose();
    }

    public readonly ref struct Attachments(ImageView hdrColor, ImageView depth)
    {
        public readonly ImageView HdrColor = hdrColor;
        public readonly ImageView Depth = depth;
    }
    
    internal void Record(CommandBuffer cmd, in Renderer.FrameContext ctx , Attachments attachments)
    {
        if (!ImGui.EditorState.SkyboxEnabled) return;

        BeginRendering(cmd,
            ctx.RenderExtent,
            [attachments.HdrColor],
            depthView: attachments.Depth,
            depthLoad: AttachmentLoadOp.Load,
            colorLoad: AttachmentLoadOp.Load,
            clearValues: []
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

        // Set 0 = private UBO; FeatureEnv (envCube) at its own reflected index (set 4). The
        // intervening slots are gaps in the layout, never bound, so this is two separate binds.
        var registry = Renderer.descriptorRegistry;
        var set0 = GetDescriptorSet(0, ctx.FrameIndex);
        Vk!.CmdBindDescriptorSets(cmd, PipelineBindPoint.Graphics, Layout, 0, 1, &set0, 0, null);

        var envSet = registry.FeatureSet(FeatureEnv, ctx.FrameIndex);
        Vk!.CmdBindDescriptorSets(cmd, PipelineBindPoint.Graphics, Layout, registry.FeatureSetIndex(FeatureEnv), 1, &envSet, 0, null);

        Vk!.CmdDraw(cmd, 3, 1, 0, 0);
        
        EndRendering(cmd);
    }
    
    //  Pipeline state 

    // Depth test EQUAL @ 1.0. The vertex shader writes gl_Position.z = 1.0
    // (post-divide depth = 1.0), so only pixels still at the cleared far value
    // — i.e. where geometry never landed — pass the test.
    protected override PipelineDepthStencilStateCreateInfo BuildDepthStencil() => new()
    {
        SType                 = StructureType.PipelineDepthStencilStateCreateInfo,
        DepthTestEnable       = true,
        DepthWriteEnable      = false,
        DepthCompareOp        = CompareOp.Equal,
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


    private const string FeatureEnv = "FeatureEnv";

    protected override void CreateDescriptorSetLayouts()
    {
        // Private set 0: the skybox's own per-frame UBO (this pass is not on the scene set).
        // envCube comes from the registry-owned FeatureEnv set (set 4), so the pipeline layout
        // carries empty placeholders in every slot between (pass / graph-shared / FeatureIBL,
        // none of which this pass uses). This is the first consumer to exercise the gap-filling path.
        var uboBinding = new DescriptorSetLayoutBinding
        {
            Binding         = 0,
            DescriptorType  = DescriptorType.UniformBuffer,
            DescriptorCount = 1,
            StageFlags      = ShaderStageFlags.VertexBit | ShaderStageFlags.FragmentBit,
        };
        DescriptorSetLayout set0;
        DescriptorSetLayoutCreateInfo info = new()
        {
            SType        = StructureType.DescriptorSetLayoutCreateInfo,
            BindingCount = 1,
            PBindings    = &uboBinding,
        };
        if (Vk.CreateDescriptorSetLayout(Device, &info, null, out set0) != Result.Success)
            throw new System.Exception("Failed to create skybox UBO descriptor set layout");

        // Array sized to FeatureEnv's reflected index: private UBO at 0, the shared empty layout
        // in every gap, FeatureEnv at its pinned index. Follows a module renumber automatically.
        var registry = Renderer.descriptorRegistry;
        uint envIndex = registry.FeatureSetIndex(FeatureEnv);
        var layouts = new DescriptorSetLayout[envIndex + 1];
        for (int i = 0; i < layouts.Length; i++) layouts[i] = registry.EmptySetLayout;
        layouts[0] = set0;
        layouts[envIndex] = registry.FeatureSetLayout(FeatureEnv);
        DescriptorSetLayouts = layouts;
        OwnedDescriptorSetLayoutIndices = new[] { 0 };   // only the private UBO layout
    }

    protected override void CreateResources()
    {
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
            Gfx.CreateMappedUniformBuffer(sizeof(SkyboxUBO), ref FrameUniformBuffers[i]);
    }

    protected override void CreateDescriptorSets()
    {
        var layouts = stackalloc DescriptorSetLayout[(int)Renderer.MAX_CONCURRENT_FRAMES];
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++) layouts[i] = DescriptorSetLayouts[0];

        DescriptorSetAllocateInfo allocInfo = new()
        {
            SType              = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool     = Gfx.DescriptorPool,
            DescriptorSetCount = Renderer.MAX_CONCURRENT_FRAMES,
            PSetLayouts        = layouts,
        };

        DescriptorSets    = new DescriptorSet[1][];
        DescriptorSets[0] = new DescriptorSet[Renderer.MAX_CONCURRENT_FRAMES];
        fixed (DescriptorSet* pSets = DescriptorSets[0])
        {
            if (Vk.AllocateDescriptorSets(Device, &allocInfo, pSets) != Result.Success)
                throw new System.Exception("Failed to allocate skybox descriptor sets");
        }
    }

    protected override void WriteDescriptors()
    {
        // Only the private UBO (set 0). envCube is registry-owned on FeatureEnv (set 3).
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            DescriptorBufferInfo frameInfo = new()
            {
                Buffer = FrameUniformBuffers[i].buffer,
                Offset = 0,
                Range  = (ulong)sizeof(SkyboxUBO),
            };
            var write = new WriteDescriptorSet
            {
                SType           = StructureType.WriteDescriptorSet,
                DstSet          = DescriptorSets[0][i],
                DstBinding      = 0,
                DstArrayElement = 0,
                DescriptorType  = DescriptorType.UniformBuffer,
                DescriptorCount = 1,
                PBufferInfo     = &frameInfo,
            };
            Vk.UpdateDescriptorSets(Device, 1, &write, 0, null);
        }
    }

    // Per-frame upload

    public void UpdatePerFrame(uint frameIndex, Camera camera, float intensity)
    {
        SkyboxUBO ubo = new();
        if (camera != null)
        {
            var proj = camera.GetProjectionMatrix(
                (float)Renderer.renderExtent.Width / Renderer.renderExtent.Height, 0.1f, 100.0f);
            proj.M22 *= -1;
            var view = camera.GetViewMatrix();
            var vp   = view * proj;
            Matrix4x4.Invert(vp, out ubo.invViewProj);
            ubo.camPos = new Vector4(camera.GetPosition(), 1.0f);
        }
        else
        {
            ubo.invViewProj = Matrix4x4.Identity;
            ubo.camPos      = Vector4.Zero;
        }
        ubo.intensity = intensity;

        void* data = FrameUniformBuffers[frameIndex].mapped;
        new System.Span<SkyboxUBO>(data, 1).Fill(ubo);
    }
}
