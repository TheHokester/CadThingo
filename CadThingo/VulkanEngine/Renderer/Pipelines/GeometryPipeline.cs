using System.Numerics;
using System.Runtime.InteropServices;
using CadThingo.VulkanEngine.GLTF;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Pipelines;
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
    //Per frame uniform buffers for geometry pipeline
    private UboBuffer[] GeometryUniformBuffers = new UboBuffer[Renderer.MAX_CONCURRENT_FRAMES];

    protected override string ShaderPath { get; } =
        @"C:\Users\jamie\RiderProjects\CadThingo\CadThingo\Assets\Shaders\Geometry.spv";
    protected override Format[] ColorAttachmentFormats { get; } =
    [
        Format.R32G32B32A32Sfloat, // Position
        Format.R32G32B32A32Sfloat, // Normal
        Format.R8G8B8A8Unorm, // Albedo
        Format.R8G8B8A8Unorm, // Material
        Format.R16G16B16A16Sfloat // Emissive
    ];


    public GeometryPipeline(Renderer renderer) : base(renderer)
    {
        DepthAttachmentFormat = renderer.FindDepthFormat();
    }

    public override void Dispose()
    {
        foreach (var ubo in GeometryUniformBuffers) Renderer.DestroyBuffer(ubo.buffer, ubo.alloc);
        base.Dispose();
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

    internal void Record(CommandBuffer cmd, in Renderer.FrameContext ctx, Buffer indirectCmd,
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

        // Set 0 (FrameUBO) + Set 1 (bindless: materials/instances/textures/samplers)
        // bound once. The whole geometry pass issues zero per-draw rebinds and zero
        // push constants — model + materialIndex live in the per-instance SSBO.
        var frameSet = GetDescriptorSet(0, ctx.FrameIndex);
        var bindlessSet = Engine.ResourceManager.GetBindlessSet(ctx.FrameIndex);
        var geomSets = stackalloc DescriptorSet[2] { frameSet, bindlessSet };
        Vk!.CmdBindDescriptorSets(cmd, PipelineBindPoint.Graphics,
            Layout, 0, 2, geomSets, 0, null);

        // Bind global VB/IB once — every mesh is packed into these
        var vb = Engine.ResourceManager.GlobalVertexBuffer;
        var ib = Engine.ResourceManager.GlobalIndexBuffer;
        ulong vbOffset = 0;
        Vk!.CmdBindVertexBuffers(cmd, 0, 1, &vb, &vbOffset);
        Vk!.CmdBindIndexBuffer(cmd, ib, 0, IndexType.Uint32);

        // Material SSBO upload moved to Renderer.UpdateMaterials so every
        // rendering path (deferred / forward+ / pathtracer) sees the same
        // per-frame snapshot. Inline upload here was deferred-only and PT mode
        // never saw inspector edits as a result.
        
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
        CreateFrameDescriptorSetLayout(out var frameDSL);
        // Slot 0 is OWNED here; slot 1 is borrowed from ResourceManager (don't destroy on Dispose).
        DescriptorSetLayouts = new[] { frameDSL,  Engine.ResourceManager.GetBindlessLayout()};
        OwnedDescriptorSetLayoutIndices = new[] { 0 };
    }

    protected override void CreateResources()
    {
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            Renderer.CreateMappedUniformBuffer(sizeof(GeometryUBO), ref GeometryUniformBuffers[i]);
        }
    }

    protected override void CreateDescriptorSets()
    {
        // Per-frame "frame" descriptor sets (set 0): only binding 0 = FrameUBO (view+proj).
        // Set 1 (bindless materials/instances/textures/samplers) is owned and bound by
        // ResourceManager.
        var layouts = stackalloc DescriptorSetLayout[(int)Renderer.MAX_CONCURRENT_FRAMES];
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++) layouts[i] = DescriptorSetLayouts[0];

        DescriptorSetAllocateInfo allocateInfo = new()
        {
            SType = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool = Renderer.descriptorPool,
            DescriptorSetCount = Renderer.MAX_CONCURRENT_FRAMES,
            PSetLayouts = layouts
        };
        DescriptorSets = new DescriptorSet[1][];
        DescriptorSets[0] = new DescriptorSet[Renderer.MAX_CONCURRENT_FRAMES];
        fixed (DescriptorSet* pDS = DescriptorSets[0])
        {
            if (Vk.AllocateDescriptorSets(Device, &allocateInfo, pDS) != Result.Success)
                throw new Exception("Failed to allocate descriptor sets using layout 0 for geometry pipeline");
        }
    }

    protected override void WriteDescriptors()
    {
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            DescriptorBufferInfo bufferInfo = new()
            {
                Buffer = GeometryUniformBuffers[i].buffer,
                Offset = 0,
                Range = (ulong)sizeof(GeometryUBO),
            };
            WriteDescriptorSet descriptorWrite = new()
            {
                SType = StructureType.WriteDescriptorSet,
                DstSet = DescriptorSets[0][i],
                DstBinding = 0,
                DstArrayElement = 0,
                DescriptorType = DescriptorType.UniformBuffer,
                DescriptorCount = 1,
                PBufferInfo = &bufferInfo
            };
            Vk.UpdateDescriptorSets(Device, 1, &descriptorWrite, 0, null);
        }
    }

    protected override VertexInputBindingDescription[] GetVertexInputBindings()
    {
        return [Vertex.GetBindingDescription()];
    }

    protected override VertexInputAttributeDescription[] GetVertexInputAttributes()
    {
        return Vertex.GetAttributeDescriptions();
    }


    // Set 0 of the geometry pipeline. One binding (the per-frame FrameUBO with view+proj),
    // bound once at the start of the geometry pass and reused for every draw.
    private void CreateFrameDescriptorSetLayout(out DescriptorSetLayout layout)
    {
        var binding = new DescriptorSetLayoutBinding
        {
            Binding = 0,
            DescriptorType = DescriptorType.UniformBuffer,
            DescriptorCount = 1,
            StageFlags = ShaderStageFlags.VertexBit,
            PImmutableSamplers = null,
        };

        DescriptorSetLayoutBindingFlagsCreateInfo flagsCreateInfo = new()
            { SType = StructureType.DescriptorSetLayoutBindingFlagsCreateInfo };
        var flag = DescriptorBindingFlags.UpdateAfterBindBit |
                   DescriptorBindingFlags.UpdateUnusedWhilePendingBit;

        if (Renderer.descriptorIndexEnabled)
        {
            flagsCreateInfo.BindingCount = 1;
            flagsCreateInfo.PBindingFlags = &flag;
        }

        DescriptorSetLayoutCreateInfo layoutInfo = new()
        {
            SType = StructureType.DescriptorSetLayoutCreateInfo,
            BindingCount = 1,
            PBindings = &binding,
        };
        if (Renderer.descriptorIndexEnabled)
        {
            layoutInfo.Flags |= DescriptorSetLayoutCreateFlags.UpdateAfterBindPoolBit;
            layoutInfo.PNext = &flagsCreateInfo;
        }

        if (Vk.CreateDescriptorSetLayout(Device, &layoutInfo, null, out layout) !=
            Result.Success)
            throw new Exception("Failed to create geometry frame descriptor set layout");
    }

    // Writes the per-frame view+proj into GeometryUniformBuffers[frameIndex].
    // Called once per frame in DrawFrame; per-draw model matrix lives in the instance SSBO.
    public void UpdateUbo(uint frameIndex, Camera camera)
    {
        GeometryUBO ubo = new();
        if (camera != null)
        {
            ubo.proj = camera.GetProjectionMatrix((float)Renderer.renderExtent.Width / Renderer.renderExtent.Height, 0.1f, 100.0f);
            ubo.view = camera.GetViewMatrix();
            ubo.proj.M22 *= -1; // Vulkan clip space has Y down
        }
        else
        {
            ubo.view = Matrix4x4.CreateLookAt(new Vector3(2, 2, 2), new Vector3(0, 0, 0), new Vector3(0, 0, 1));
            ubo.proj = Matrix4x4.CreatePerspectiveFieldOfView((float)(45 * Math.PI / 180),
                (float)Renderer.renderExtent.Width / Renderer.renderExtent.Height, 0.1f, 100.0f);
            ubo.proj.M22 *= -1; // flip Y for Vulkan clip space
        }

        void* data = GeometryUniformBuffers[frameIndex].mapped;
        new Span<GeometryUBO>(data, 1).Fill(ubo);
    }
}