using System.Numerics;
using System.Runtime.InteropServices;
using CadThingo.VulkanEngine.GLTF;
using CadThingo.VulkanEngine.Renderer.Pipelines;
using CadThingo.VulkanEngine.Renderer.Slang;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Features.Deferred;
//
//  Geometry pass — writes the G-buffer
//
public sealed unsafe class GeometryPipeline : GraphicsPipeline
{
    struct GeometryUBO
    {
        public Matrix4x4 view;
        public Matrix4x4 proj;
    }

    protected override ShaderCompileRequest? Program =>
        new("Deferred/Geometry", ["VSMain", "PSMain"], [], []);

    protected override Format[] ColorAttachmentFormats { get; } =
    [
        Format.R32G32B32A32Sfloat, // Position
        Format.R32G32B32A32Sfloat, // Normal
        Format.R8G8B8A8Unorm, // Albedo
        Format.R8G8B8A8Unorm, // Material
        Format.R16G16B16A16Sfloat // Emissive
    ];


    public GeometryPipeline(GpuContext gpu, Renderer renderer) : base(gpu, renderer)
    {
        DepthAttachmentFormat = Gfx.FindDepthFormat();
    }

    public readonly ref struct Attachments(
        ImageView position,
        ImageView normal,
        ImageView albedo,
        ImageView material,
        ImageView emissive,
        ImageView depth)
    {
        public readonly ImageView Position = position, Normal = normal, Albedo = albedo, Material = material, Emissive = emissive, Depth = depth;
    }
    /// <summary>
    /// Records the pass commands for the frame.
    /// </summary>
    /// <param name="cmd"></param>
    /// <param name="ctx"></param>
    /// <param name="indirectCmd"></param>
    /// <param name="indirectCount"></param>
    /// <param name="drawCount"></param>
    /// <param name="attachments"></param>
    internal void Record(CommandBuffer cmd, in RenderView ctx, Buffer indirectCmd,
        Buffer indirectCount, uint drawCount, Attachments attachments)
    {
        BeginRendering(cmd,
            ctx.RenderExtent, 
            [
                attachments.Position,
                attachments.Normal,
                attachments.Albedo,
                attachments.Material,
                attachments.Emissive
            ], 
            attachments.Depth
            );
        
        
        // Pipeline + dynamic state
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

        // First SceneBindings consumer: the unified scene set carries the bindless
        // materials/instances/textures/samplers, and the per-pass view+proj rides the
        // (0,0) dynamic slot as an arena push. One set, one bind, zero per-draw rebinds.
        uint frameConstants = Registry.ConstantArena.Push(ctx.FrameIndex, BuildFrameUbo(ctx.Camera, ctx.RenderExtent));
        var sceneSet = Registry.SceneSet(ctx.FrameIndex);
        Vk!.CmdBindDescriptorSets(cmd, PipelineBindPoint.Graphics,
            Layout, 0, 1, &sceneSet, 1, &frameConstants);

        // Bind global VB/IB once — every mesh is packed into these
        var vb = Engine.ResourceManager.GlobalVertexBuffer;
        var ib = Engine.ResourceManager.GlobalIndexBuffer;
        ulong vbOffset = 0;
        Vk!.CmdBindVertexBuffers(cmd, 0, 1, &vb, &vbOffset);
        Vk!.CmdBindIndexBuffer(cmd, ib, 0, IndexType.Uint32);

        // The draw-cull compute pass already populated:
        //   - InstanceStorageBuffers (ResourceManager): per-visible-renderable model
        //     + materialIndex, read by the VS via SV_InstanceID.
        //   - IndirectCmdBuffers     (DrawCullPipeline): VkDrawIndexedIndirectCommand[]
        //                                                visible mesh draws.
        //   - IndirectCountBuffers   (DrawCullPipeline): single uint of valid entries.
        // A single vkCmdDrawIndexedIndirectCount consumes them all.
        if (drawCount > 0)
        {
            Vk!.CmdDrawIndexedIndirectCount(cmd,
                indirectCmd, 0,
                indirectCount, 0,
                drawCount,
                (uint)sizeof(DrawIndexedIndirectCommandGpu));
        }
        
        EndRendering(cmd);
        
    }
    protected override void CreateDescriptorSetLayouts()
    {
        // Scene set only - Geometry.slang declares nothing outside SceneBindings, so there is no
        // pass set to build. The layout is registry-owned and borrowed (never destroyed here);
        // the pipeline owns no descriptor sets, layouts, or UBO buffers.
        DescriptorSetLayouts = Registry.BuildPipelineSetLayouts(null);
        OwnedDescriptorSetLayoutIndices = [];
    }

    protected override VertexInputBindingDescription[] GetVertexInputBindings()
    {
        return [Vertex.GetBindingDescription()];
    }

    protected override VertexInputAttributeDescription[] GetVertexInputAttributes()
    {
        return Vertex.GetAttributeDescriptions();
    }


    // Per-pass view+proj for the (0,0) arena slot; pushed by Record each frame.
    // Per-draw model matrix lives in the instance SSBO.
    private GeometryUBO BuildFrameUbo(Camera? camera, Extent2D renderExtent)
    {
        GeometryUBO ubo = new();
        var aspect = (float)renderExtent.Width / renderExtent.Height;
        if (camera != null)
        {
            ubo.proj = camera.GetProjectionMatrix(aspect, 0.1f, 100.0f);
            ubo.view = camera.GetViewMatrix();
        }
        else
        {
            ubo.view = Matrix4x4.CreateLookAt(new Vector3(2, 2, 2), new Vector3(0, 0, 0), new Vector3(0, 0, 1));
            ubo.proj = Matrix4x4.CreatePerspectiveFieldOfView((float)(45 * Math.PI / 180), aspect, 0.1f, 100.0f);
        }

        ubo.proj.M22 *= -1; // Vulkan clip space has Y down
        return ubo;
    }
}