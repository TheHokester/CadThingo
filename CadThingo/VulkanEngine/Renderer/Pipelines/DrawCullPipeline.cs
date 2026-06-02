using System.Numerics;
using System.Runtime.InteropServices;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Pipelines;
//  Draw-cull compute pass — frustum-tests scene renderables and emits
//  VkDrawIndexedIndirectCommand[] consumed by the geometry pass.
public sealed unsafe class DrawCullPipeline : ComputePipeline
{
    // Push constants pushed at every dispatch. 100 bytes — within the
    // 128B Vulkan minimum (maxPushConstantsSize).
    [StructLayout(LayoutKind.Sequential)]
    private struct CullPushConstants
    {
        public Vector4 PlaneL;
        public Vector4 PlaneR;
        public Vector4 PlaneB;
        public Vector4 PlaneT;
        public Vector4 PlaneN;
        public Vector4 PlaneF;
        public uint    RenderableCount;
        public uint    _pad0;
        public uint    _pad1;
        public uint    _pad2;
    }

    protected override string ShaderPath { get; } =
        @"C:\Users\jamie\RiderProjects\CadThingo\CadThingo\Assets\Shaders\CullDraws.spv";

    // Per-frame buffers owned by this pipeline. RenderableInput is the CPU input
    // list filled in Record(); IndirectCmd holds the post-cull
    // VkDrawIndexedIndirectCommand array; IndirectCount holds one uint the cull
    // shader InterlockedAdds into and the rasterizer reads via
    // vkCmdDrawIndexedIndirectCount.
    private UboBuffer[] RenderableInputBuffers = new UboBuffer[Renderer.MAX_CONCURRENT_FRAMES];
    private UboBuffer[] IndirectCmdBuffers     = new UboBuffer[Renderer.MAX_CONCURRENT_FRAMES];
    private UboBuffer[] IndirectCountBuffers   = new UboBuffer[Renderer.MAX_CONCURRENT_FRAMES];

    public Buffer GetIndirectCmdBuffer  (uint frame) => IndirectCmdBuffers[frame].buffer;
    public Buffer GetIndirectCountBuffer(uint frame) => IndirectCountBuffers[frame].buffer;

    /// <summary>Renderables packed in the most recent Record() call — drives
    /// maxDrawCount on vkCmdDrawIndexedIndirectCount.</summary>
    public uint LastRenderableCount { get; private set; }

    // BLEND-mode entities partitioned out of the cull input during Record.
    // Sorted back-to-front by view-space depth so the transparent pass renders
    // far-first.
    private readonly List<TransparentDraw> _transparentDraws = new();

    /// <summary>BLEND-mode draws captured this frame, sorted back-to-front by view-space Z.
    /// Consumed by the TransparentPass; empty when no scene material is BLEND-mode.</summary>
    public IReadOnlyList<TransparentDraw> LastTransparentDraws => _transparentDraws;

    public DrawCullPipeline(Renderer renderer) : base(renderer)
    {
        PushConstantRanges = new[]
        {
            new PushConstantRange
            {
                StageFlags = ShaderStageFlags.ComputeBit,
                Offset     = 0,
                Size       = (uint)sizeof(CullPushConstants),
            }
        };
    }

    public override void Dispose()
    {
        foreach (var b in RenderableInputBuffers) Gfx.DestroyBuffer(b.buffer, b.alloc);
        foreach (var b in IndirectCmdBuffers)     Gfx.DestroyBuffer(b.buffer, b.alloc);
        foreach (var b in IndirectCountBuffers)   Gfx.DestroyBuffer(b.buffer, b.alloc);
        base.Dispose();
    }

    protected override void CreateDescriptorSetLayouts()
    {
        // Four storage buffers — all writable except binding 0 (input renderables).
        // We mark them all StorageBuffer with no UpdateAfterBind — the sets are
        // written once at startup (the buffers themselves are persistently mapped).
        var bindings = stackalloc DescriptorSetLayoutBinding[4];
        for (uint b = 0; b < 4; b++)
        {
            bindings[b] = new DescriptorSetLayoutBinding
            {
                Binding         = b,
                DescriptorType  = DescriptorType.StorageBuffer,
                DescriptorCount = 1,
                StageFlags      = ShaderStageFlags.ComputeBit,
                PImmutableSamplers = null,
            };
        }
        DescriptorSetLayoutCreateInfo info = new()
        {
            SType        = StructureType.DescriptorSetLayoutCreateInfo,
            BindingCount = 4,
            PBindings    = bindings,
        };
        if (Vk.CreateDescriptorSetLayout(Device, &info, null, out var layout) != Result.Success)
            throw new Exception("Failed to create cull descriptor set layout");
        DescriptorSetLayouts = new[] { layout };
        OwnedDescriptorSetLayoutIndices = new[] { 0 };
    }

    protected override void CreateResources()
    {
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            Gfx.CreateMappedStorageBuffer(
                (ulong)(Renderer.MAX_INSTANCES * (uint)sizeof(RenderableInputGpu)),
                ref RenderableInputBuffers[i]);

            // Indirect-command buffer also needs IndirectBuffer usage so the
            // vkCmdDraw...IndirectCount call can read it without validation errors.
            Gfx.CreateMappedStorageBuffer(
                (ulong)(Renderer.MAX_INSTANCES * (uint)sizeof(DrawIndexedIndirectCommandGpu)),
                ref IndirectCmdBuffers[i],
                BufferUsageFlags.IndirectBufferBit);

            // Count buffer is one uint. Needs IndirectBuffer for the count read and
            // TransferDst so vkCmdFillBuffer can reset it to 0 every frame.
            Gfx.CreateMappedStorageBuffer(
                sizeof(uint),
                ref IndirectCountBuffers[i],
                BufferUsageFlags.IndirectBufferBit | BufferUsageFlags.TransferDstBit);
        }
    }

    protected override void CreateDescriptorSets()
    {
        var layouts = stackalloc DescriptorSetLayout[(int)Renderer.MAX_CONCURRENT_FRAMES];
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++) layouts[i] = DescriptorSetLayouts[0];

        DescriptorSetAllocateInfo alloc = new()
        {
            SType              = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool     = Gfx.DescriptorPool,
            DescriptorSetCount = Renderer.MAX_CONCURRENT_FRAMES,
            PSetLayouts        = layouts,
        };
        DescriptorSets = new DescriptorSet[1][];
        DescriptorSets[0] = new DescriptorSet[Renderer.MAX_CONCURRENT_FRAMES];
        fixed (DescriptorSet* pSets = DescriptorSets[0])
        {
            if (Vk.AllocateDescriptorSets(Device, &alloc, pSets) != Result.Success)
                throw new Exception("Failed to allocate cull descriptor sets");
        }
    }

    protected override void WriteDescriptors()
    {
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            DescriptorBufferInfo bufIn = new()
            {
                Buffer = RenderableInputBuffers[i].buffer, Offset = 0,
                Range  = (ulong)(Renderer.MAX_INSTANCES * (uint)sizeof(RenderableInputGpu)),
            };
            DescriptorBufferInfo bufCmd = new()
            {
                Buffer = IndirectCmdBuffers[i].buffer, Offset = 0,
                Range  = (ulong)(Renderer.MAX_INSTANCES * (uint)sizeof(DrawIndexedIndirectCommandGpu)),
            };
            DescriptorBufferInfo bufInst = new()
            {
                Buffer = Engine.ResourceManager.GetInstanceBuffer((uint)i), Offset = 0,
                Range  = (ulong)(Renderer.MAX_INSTANCES * (uint)sizeof(InstanceDataGPU)),
            };
            DescriptorBufferInfo bufCount = new()
            {
                Buffer = IndirectCountBuffers[i].buffer, Offset = 0,
                Range  = sizeof(uint),
            };

            var writes = stackalloc WriteDescriptorSet[4];
            for (uint b = 0; b < 4; b++)
            {
                writes[b] = new WriteDescriptorSet
                {
                    SType           = StructureType.WriteDescriptorSet,
                    DstSet          = DescriptorSets[0][i],
                    DstBinding      = b,
                    DescriptorType  = DescriptorType.StorageBuffer,
                    DescriptorCount = 1,
                };
            }
            writes[0].PBufferInfo = &bufIn;
            writes[1].PBufferInfo = &bufCmd;
            writes[2].PBufferInfo = &bufInst;
            writes[3].PBufferInfo = &bufCount;

            Vk.UpdateDescriptorSets(Device, 4, writes, 0, null);
        }
    }

    // CPU side of the cull pass. Walks scene entities, fills the input buffer,
    // then records the dispatch + barriers. Returns the renderable count packed
    // this frame (cached as LastRenderableCount for the geometry pass).
    public uint Record(CommandBuffer cmd, uint frameIndex, Camera cam, Scene scene)
    {
        
        // Pack RenderableInput rows from the scene
        // Opaque (OPAQUE + MASK) entities go through the GPU cull → indirect-draw path.
        // BLEND entities are siphoned into _transparentDraws for the forward+ pass.
        RenderableInputGpu* inputPtr = (RenderableInputGpu*)RenderableInputBuffers[frameIndex].mapped;
        uint count = 0;
        int matCount = scene.MaterialCount;
        int fallbackMatIdx = matCount; // matches the fallback slot written by the geometry pass

        _transparentDraws.Clear();
        Matrix4x4 viewMat = cam != null ? cam.GetViewMatrix() : Matrix4x4.Identity;

        for (int i = 0; i < scene.EntityCount; i++)
        {
            Entity* e = scene.GetEntity(i);
            if (e == null) continue;
            var meshComp = e->GetComponent<MeshComponent>();
            if (meshComp == null || meshComp.mesh == null) continue;
            var transform = e->GetComponent<TransformComponent>();
            if (transform == null) continue;

            int  matIdx = meshComp.materialIndex >= 0 ? meshComp.materialIndex : fallbackMatIdx;
            Mesh m      = *meshComp.mesh;
            var  world  = *transform.GetWorldMatrix();

            // Fallback material is implicitly opaque. Real materials carry their
            // mode in the Flags bitfield.
            AlphaMode mode = AlphaMode.Opaque;
            if (matIdx >= 0 && matIdx < matCount)
                mode = scene.Materials[matIdx].GetAlphaMode();

            if (mode == AlphaMode.Blend)
            {
                // System.Numerics row-vector convention: Vector.Transform(v, m) = v * m.
                // CreateLookAt produces -Z-forward view space, so farther entities
                // have more-negative Z. Sort ascending for back-to-front order.
                var worldOrigin = new Vector4(world.M41, world.M42, world.M43, 1f);
                float viewZ = Vector4.Transform(worldOrigin, viewMat).Z;

                _transparentDraws.Add(new TransparentDraw
                {
                    Model         = world,
                    MaterialIndex = (uint)matIdx,
                    IndexCount    = (uint)m.count,
                    FirstIndex    = (uint)m.offset,
                    ViewDepth     = viewZ,
                });
                continue;
            }

            if (count >= Renderer.MAX_INSTANCES)
                throw new InvalidOperationException($"Renderable count exceeds MAX_INSTANCES ({Renderer.MAX_INSTANCES}).");

            inputPtr[count] = new RenderableInputGpu
            {
                model         = world,
                sphereLocal   = m.sphereLocal,
                indexCount    = (uint)m.count,
                firstIndex    = (uint)m.offset,
                materialIndex = (uint)matIdx,
            };
            count++;
        }

        // Back-to-front: far (lowest view-space Z) first.
        _transparentDraws.Sort((a, b) => a.ViewDepth.CompareTo(b.ViewDepth));

        LastRenderableCount = count;
        if (count == 0) return 0;

        // 1. Reset the count buffer to 0 via vkCmdFillBuffer
        Vk.CmdFillBuffer(cmd, IndirectCountBuffers[frameIndex].buffer, 0, sizeof(uint), 0);

        //  2. Barrier: transfer write -> compute shader access on count buffer 
        var fillBarrier = new BufferMemoryBarrier
        {
            SType         = StructureType.BufferMemoryBarrier,
            SrcAccessMask = AccessFlags.TransferWriteBit,
            DstAccessMask = AccessFlags.ShaderReadBit | AccessFlags.ShaderWriteBit,
            SrcQueueFamilyIndex = Silk.NET.Vulkan.Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Silk.NET.Vulkan.Vk.QueueFamilyIgnored,
            Buffer = IndirectCountBuffers[frameIndex].buffer,
            Offset = 0,
            Size   = sizeof(uint),
        };
        Vk.CmdPipelineBarrier(cmd,
            PipelineStageFlags.TransferBit,
            PipelineStageFlags.ComputeShaderBit,
            0, 0, null, 1, &fillBarrier, 0, null);

        Vk.CmdBindPipeline(cmd, PipelineBindPoint.Compute, PipelineHandle);
        var dset = DescriptorSets[0][frameIndex];
        Vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute,
            PipelineLayoutHandle, 0, 1, &dset, 0, null);

        // Build frustum from the camera's view*proj. Deliberately use a non-Y-flipped
        // projection here: the visible volume is the same in both conventions, and
        // Frustum.FromViewProjection assumes standard row-major Vulkan NDC.
        Matrix4x4 view = cam.GetViewMatrix();
        Matrix4x4 proj = cam.GetProjectionMatrix(
            (float)Renderer.renderExtent.Width / Renderer.renderExtent.Height, 0.1f, 100.0f);
        Matrix4x4 vp   = view * proj;
        var frustum    = Frustum.FromViewProjection(vp, vulkanNDC: true);

        var push = new CullPushConstants
        {
            PlaneL = frustum.PlaneLeft.Data,
            PlaneR = frustum.PlaneRight.Data,
            PlaneB = frustum.PlaneBottom.Data,
            PlaneT = frustum.PlaneTop.Data,
            PlaneN = frustum.PlaneNear.Data,
            PlaneF = frustum.PlaneFar.Data,
            RenderableCount = count,
        };
        Vk.CmdPushConstants(cmd, PipelineLayoutHandle, ShaderStageFlags.ComputeBit,
            0, (uint)sizeof(CullPushConstants), &push);

        // 64 threads per group; ceil-divide so the last group covers the tail.
        uint groups = (count + 63u) / 64u;
        Vk.CmdDispatch(cmd, groups, 1, 1);

        // 4. Barrier: compute writes -> indirect/vertex stage reads
        var postBarriers = stackalloc BufferMemoryBarrier[3];
        postBarriers[0] = new BufferMemoryBarrier
        {
            SType         = StructureType.BufferMemoryBarrier,
            SrcAccessMask = AccessFlags.ShaderWriteBit,
            DstAccessMask = AccessFlags.IndirectCommandReadBit,
            SrcQueueFamilyIndex = Silk.NET.Vulkan.Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Silk.NET.Vulkan.Vk.QueueFamilyIgnored,
            Buffer = IndirectCmdBuffers[frameIndex].buffer,
            Offset = 0,
            Size   = Silk.NET.Vulkan.Vk.WholeSize,
        };
        postBarriers[1] = new BufferMemoryBarrier
        {
            SType         = StructureType.BufferMemoryBarrier,
            SrcAccessMask = AccessFlags.ShaderWriteBit,
            DstAccessMask = AccessFlags.IndirectCommandReadBit,
            SrcQueueFamilyIndex = Silk.NET.Vulkan.Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Silk.NET.Vulkan.Vk.QueueFamilyIgnored,
            Buffer = IndirectCountBuffers[frameIndex].buffer,
            Offset = 0,
            Size   = sizeof(uint),
        };
        postBarriers[2] = new BufferMemoryBarrier
        {
            SType         = StructureType.BufferMemoryBarrier,
            SrcAccessMask = AccessFlags.ShaderWriteBit,
            DstAccessMask = AccessFlags.ShaderReadBit,
            SrcQueueFamilyIndex = Silk.NET.Vulkan.Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Silk.NET.Vulkan.Vk.QueueFamilyIgnored,
            Buffer = Engine.ResourceManager.GetInstanceBuffer(frameIndex),
            Offset = 0,
            Size   = Silk.NET.Vulkan.Vk.WholeSize,
        };
        Vk.CmdPipelineBarrier(cmd,
            PipelineStageFlags.ComputeShaderBit,
            PipelineStageFlags.DrawIndirectBit | PipelineStageFlags.VertexShaderBit,
            0, 0, null, 3, postBarriers, 0, null);

        return count;
    }
}