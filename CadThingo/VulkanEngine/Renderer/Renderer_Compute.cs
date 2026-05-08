using System.Numerics;
using System.Runtime.InteropServices;
using Silk.NET.Core.Native;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer;

public unsafe partial class Renderer
{
    // Draw-cull pipeline (Phase 3) — frustum-tests scene entities against the
    // camera and emits VkDrawIndexedIndirectCommand[] for visible meshes.
    private DescriptorSetLayout cullDescriptorSetLayout;
    private PipelineLayout      cullPipelineLayout;
    private Pipeline            cullPipeline;
    private DescriptorSet[]     cullDescriptorSets = new DescriptorSet[MAX_CONCURRENT_FRAMES];

    // Light-cull pipeline — bins active lights into per-tile slot lists
    // so the deferred lighting fragment shader iterates only the few lights that
    // actually touch each 16×16 tile.
    private DescriptorSetLayout lightCullDescriptorSetLayout;
    private PipelineLayout      lightCullPipelineLayout;
    private Pipeline            lightCullPipeline;
    private DescriptorSet[]     lightCullDescriptorSets = new DescriptorSet[MAX_CONCURRENT_FRAMES];

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

    // Light-cull push constants — 64B invViewProj + 16 camPos + 8 screenSize +
    // 8 tileCounts + 4 lightCount + 12 pad = 112B, well under the 128B minimum.
    [StructLayout(LayoutKind.Sequential)]
    private struct LightCullPushConstants
    {
        public Matrix4x4 InvViewProj;
        public Vector4   CamPos;
        public Vector2   ScreenSize;
        public uint      TileCountX;
        public uint      TileCountY;
        public uint      LightCount;
        public uint      _pad0;
        public uint      _pad1;
        public uint      _pad2;
    }

    private void CreateCullDescriptorSetLayout()
    {
        // Four storage buffers; all writable except binding 0 (the input renderables).
        // We mark them all StorageBuffer with no UpdateAfterBind — the descriptor sets
        // are written once at startup (the buffers themselves are persistently mapped).
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
        if (vk!.CreateDescriptorSetLayout(device, &info, null, out cullDescriptorSetLayout) != Result.Success)
            throw new Exception("Failed to create cull descriptor set layout");
    }

    private void CreateCullPipeline()
    {
        // ── Pipeline layout: one DSL + one push-constant range ─────────────
        var pcRange = new PushConstantRange
        {
            StageFlags = ShaderStageFlags.ComputeBit,
            Offset     = 0,
            Size       = (uint)sizeof(CullPushConstants),
        };
        var dsl = cullDescriptorSetLayout;
        PipelineLayoutCreateInfo layoutInfo = new()
        {
            SType                  = StructureType.PipelineLayoutCreateInfo,
            SetLayoutCount         = 1,
            PSetLayouts            = &dsl,
            PushConstantRangeCount = 1,
            PPushConstantRanges    = &pcRange,
        };
        if (vk!.CreatePipelineLayout(device, &layoutInfo, null, out cullPipelineLayout) != Result.Success)
            throw new Exception("Failed to create cull pipeline layout");

        // ── Compute shader stage ───────────────────────────────────────────
        byte[] shaderCode = File.ReadAllBytes(
            @"C:\Users\jamie\RiderProjects\CadThingo\CadThingo\Assets\Shaders\CullDraws.spv");
        ShaderModule shader = CreateShaderModule(shaderCode);

        // slangc emits the SPIR-V OpEntryPoint name as "main" regardless of the
        // source function name — the original "Main" survives only as a debug
        // label. Using anything else yields VUID-VkPipelineShaderStageCreateInfo-pName-00707.
        var stage = new PipelineShaderStageCreateInfo
        {
            SType  = StructureType.PipelineShaderStageCreateInfo,
            Stage  = ShaderStageFlags.ComputeBit,
            Module = shader,
            PName  = (byte*)SilkMarshal.StringToPtr("main"),
        };

        // ── Compute pipeline (no vertex/raster/etc. state — just stage + layout) ─
        ComputePipelineCreateInfo pipelineInfo = new()
        {
            SType  = StructureType.ComputePipelineCreateInfo,
            Stage  = stage,
            Layout = cullPipelineLayout,
        };
        if (vk!.CreateComputePipelines(device, default, 1, &pipelineInfo, null, out cullPipeline) != Result.Success)
            throw new Exception("Failed to create cull compute pipeline");

        SilkMarshal.Free((nint)stage.PName);
        vk!.DestroyShaderModule(device, shader, null);
    }

    // Allocates one cull descriptor set per frame in flight from the main pool and
    // writes the 4 storage-buffer bindings. Each frame's set points at its own
    // private pair of (input list, indirect cmds, instances out, count).
    private void CreateCullDescriptorSets()
    {
        var layouts = stackalloc DescriptorSetLayout[(int)MAX_CONCURRENT_FRAMES];
        for (var i = 0; i < MAX_CONCURRENT_FRAMES; i++) layouts[i] = cullDescriptorSetLayout;

        DescriptorSetAllocateInfo alloc = new()
        {
            SType              = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool     = descriptorPool,
            DescriptorSetCount = MAX_CONCURRENT_FRAMES,
            PSetLayouts        = layouts,
        };
        fixed (DescriptorSet* pSets = cullDescriptorSets)
        {
            if (vk!.AllocateDescriptorSets(device, &alloc, pSets) != Result.Success)
                throw new Exception("Failed to allocate cull descriptor sets");
        }

        for (var i = 0; i < MAX_CONCURRENT_FRAMES; i++)
        {
            DescriptorBufferInfo bufIn = new()
            {
                Buffer = RenderableInputBuffers[i].buffer, Offset = 0,
                Range = (ulong)(MAX_INSTANCES * (uint)sizeof(RenderableInputGpu)),
            };
            DescriptorBufferInfo bufCmd = new()
            {
                Buffer = IndirectCmdBuffers[i].buffer, Offset = 0,
                Range = (ulong)(MAX_INSTANCES * (uint)sizeof(DrawIndexedIndirectCommandGpu)),
            };
            DescriptorBufferInfo bufInst = new()
            {
                Buffer = Engine.ResourceManager.GetInstanceBuffer((uint)i), Offset = 0,
                Range = (ulong)(MAX_INSTANCES * (uint)sizeof(InstanceDataGPU)),
            };
            DescriptorBufferInfo bufCount = new()
            {
                Buffer = IndirectCountBuffers[i].buffer, Offset = 0,
                Range = sizeof(uint),
            };

            var writes = stackalloc WriteDescriptorSet[4];
            for (uint b = 0; b < 4; b++)
            {
                writes[b] = new WriteDescriptorSet
                {
                    SType           = StructureType.WriteDescriptorSet,
                    DstSet          = cullDescriptorSets[i],
                    DstBinding      = b,
                    DescriptorType  = DescriptorType.StorageBuffer,
                    DescriptorCount = 1,
                };
            }
            writes[0].PBufferInfo = &bufIn;
            writes[1].PBufferInfo = &bufCmd;
            writes[2].PBufferInfo = &bufInst;
            writes[3].PBufferInfo = &bufCount;

            vk!.UpdateDescriptorSets(device, 4, writes, 0, null);
        }
    }

    // CPU side of the cull pass. Walks scene entities, fills RenderableInputBuffers
    // with one row per renderable, then records:
    //   1. vkCmdFillBuffer to reset the count buffer to 0.
    //   2. A buffer barrier transfer-write -> compute-read on the count buffer.
    //   3. Bind cull pipeline + descriptor set, push frustum constants, vkCmdDispatch.
    //   4. A buffer barrier compute-write -> indirect-read|shader-read on the three
    //      output buffers so the geometry pass sees the cull results.
    //
    // Returns the renderable count actually packed (so debug/stats can read it).
    private uint RecordCullPass(CommandBuffer cmd, uint frameIndex, Camera cam)
    {
        // ── Pack RenderableInput rows from the scene ─────────────────────
        RenderableInputGpu* inputPtr = (RenderableInputGpu*)RenderableInputBuffers[frameIndex].mapped;
        uint count = 0;
        int matCount = scene.MaterialCount;
        int fallbackMatIdx = matCount; // matches the fallback slot we write in the geometry pass

        for (int i = 0; i < scene.EntityCount; i++)
        {
            Entity* e = scene.GetEntity(i);
            if (e == null) continue;
            var meshComp = e->GetComponent<MeshComponent>();
            if (meshComp == null || meshComp.mesh == null) continue;
            var transform = e->GetComponent<TransformComponent>();
            if (transform == null) continue;
            if (count >= MAX_INSTANCES)
                throw new InvalidOperationException($"Renderable count exceeds MAX_INSTANCES ({MAX_INSTANCES}).");

            int matIdx = meshComp.materialIndex >= 0 ? meshComp.materialIndex : fallbackMatIdx;
            Mesh m = *meshComp.mesh;

            inputPtr[count] = new RenderableInputGpu
            {
                model         = *transform.GetWorldMatrix(),
                sphereLocal   = m.sphereLocal,
                indexCount    = (uint)m.count,
                firstIndex    = (uint)m.offset,
                materialIndex = (uint)matIdx,
            };
            count++;
        }

        if (count == 0) return 0;

        // ── 1. Reset the count buffer to 0 via vkCmdFillBuffer ───────────
        vk!.CmdFillBuffer(cmd, IndirectCountBuffers[frameIndex].buffer, 0, sizeof(uint), 0);

        // ── 2. Barrier: transfer write -> compute shader access on the count buffer ─
        var fillBarrier = new BufferMemoryBarrier
        {
            SType         = StructureType.BufferMemoryBarrier,
            SrcAccessMask = AccessFlags.TransferWriteBit,
            DstAccessMask = AccessFlags.ShaderReadBit | AccessFlags.ShaderWriteBit,
            SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
            Buffer = IndirectCountBuffers[frameIndex].buffer,
            Offset = 0,
            Size   = sizeof(uint),
        };
        vk!.CmdPipelineBarrier(cmd,
            PipelineStageFlags.TransferBit,
            PipelineStageFlags.ComputeShaderBit,
            0, 0, null, 1, &fillBarrier, 0, null);

        // ── 3. Bind cull pipeline + descriptor set + push constants, dispatch ──
        vk!.CmdBindPipeline(cmd, PipelineBindPoint.Compute, cullPipeline);
        var dset = cullDescriptorSets[frameIndex];
        vk!.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute,
            cullPipelineLayout, 0, 1, &dset, 0, null);

        // Build frustum from the camera's view*proj. We deliberately use a non-Y-flipped
        // projection here: the visible volume is the same in both conventions, and the
        // Frustum.FromViewProjection helper assumes standard row-major Vulkan NDC.
        Matrix4x4 view = cam.GetViewMatrix();
        Matrix4x4 proj = cam.GetProjectionMatrix(
            (float)swapChainExtent.Width / swapChainExtent.Height, 0.1f, 100.0f);
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
        vk!.CmdPushConstants(cmd, cullPipelineLayout, ShaderStageFlags.ComputeBit,
            0, (uint)sizeof(CullPushConstants), &push);

        // 64 threads per group; ceil-divide so the last group covers the tail.
        uint groups = (count + 63u) / 64u;
        vk!.CmdDispatch(cmd, groups, 1, 1);

        // ── 4. Barrier: compute writes -> indirect/vertex stage reads ────
        // The geometry pass consumes:
        //   - IndirectCmdBuffer  via vkCmdDrawIndexedIndirectCount      (DrawIndirectBit / IndirectCommandReadBit)
        //   - IndirectCountBuffer via the same call                     (DrawIndirectBit / IndirectCommandReadBit)
        //   - InstanceStorageBuffer via VS storage-buffer read          (VertexShaderBit / ShaderReadBit)
        var postBarriers = stackalloc BufferMemoryBarrier[3];
        postBarriers[0] = new BufferMemoryBarrier
        {
            SType         = StructureType.BufferMemoryBarrier,
            SrcAccessMask = AccessFlags.ShaderWriteBit,
            DstAccessMask = AccessFlags.IndirectCommandReadBit,
            SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
            Buffer = IndirectCmdBuffers[frameIndex].buffer,
            Offset = 0,
            Size   = Vk.WholeSize,
        };
        postBarriers[1] = new BufferMemoryBarrier
        {
            SType         = StructureType.BufferMemoryBarrier,
            SrcAccessMask = AccessFlags.ShaderWriteBit,
            DstAccessMask = AccessFlags.IndirectCommandReadBit,
            SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
            Buffer = IndirectCountBuffers[frameIndex].buffer,
            Offset = 0,
            Size   = sizeof(uint),
        };
        postBarriers[2] = new BufferMemoryBarrier
        {
            SType         = StructureType.BufferMemoryBarrier,
            SrcAccessMask = AccessFlags.ShaderWriteBit,
            DstAccessMask = AccessFlags.ShaderReadBit,
            SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
            Buffer = Engine.ResourceManager.GetInstanceBuffer(frameIndex),
            Offset = 0,
            Size   = Vk.WholeSize,
        };
        vk!.CmdPipelineBarrier(cmd,
            PipelineStageFlags.ComputeShaderBit,
            PipelineStageFlags.DrawIndirectBit | PipelineStageFlags.VertexShaderBit,
            0, 0, null, 3, postBarriers, 0, null);

        return count;
    }

    // ────────────────────────────────────────────────────────────────────────
    //  Light-cull pipeline 
    // ────────────────────────────────────────────────────────────────────────

    private void CreateLightCullDescriptorSetLayout()
    {
        // 3 storage buffers: lights (read), tileLightCount (write), tileLightIndices (write).
        var bindings = stackalloc DescriptorSetLayoutBinding[3];
        for (uint b = 0; b < 3; b++)
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
            BindingCount = 3,
            PBindings    = bindings,
        };
        if (vk!.CreateDescriptorSetLayout(device, &info, null, out lightCullDescriptorSetLayout) != Result.Success)
            throw new Exception("Failed to create light-cull descriptor set layout");
    }

    private void CreateLightCullPipeline()
    {
        var pcRange = new PushConstantRange
        {
            StageFlags = ShaderStageFlags.ComputeBit,
            Offset     = 0,
            Size       = (uint)sizeof(LightCullPushConstants),
        };
        var dsl = lightCullDescriptorSetLayout;
        PipelineLayoutCreateInfo layoutInfo = new()
        {
            SType                  = StructureType.PipelineLayoutCreateInfo,
            SetLayoutCount         = 1,
            PSetLayouts            = &dsl,
            PushConstantRangeCount = 1,
            PPushConstantRanges    = &pcRange,
        };
        if (vk!.CreatePipelineLayout(device, &layoutInfo, null, out lightCullPipelineLayout) != Result.Success)
            throw new Exception("Failed to create light-cull pipeline layout");

        byte[] shaderCode = File.ReadAllBytes(
            @"C:\Users\jamie\RiderProjects\CadThingo\CadThingo\Assets\Shaders\LightCulling.spv");
        ShaderModule shader = CreateShaderModule(shaderCode);

        // slangc emits the SPIR-V OpEntryPoint name as "main" regardless of source.
        var stage = new PipelineShaderStageCreateInfo
        {
            SType  = StructureType.PipelineShaderStageCreateInfo,
            Stage  = ShaderStageFlags.ComputeBit,
            Module = shader,
            PName  = (byte*)SilkMarshal.StringToPtr("main"),
        };

        ComputePipelineCreateInfo pipelineInfo = new()
        {
            SType  = StructureType.ComputePipelineCreateInfo,
            Stage  = stage,
            Layout = lightCullPipelineLayout,
        };
        if (vk!.CreateComputePipelines(device, default, 1, &pipelineInfo, null, out lightCullPipeline) != Result.Success)
            throw new Exception("Failed to create light-cull compute pipeline");

        SilkMarshal.Free((nint)stage.PName);
        vk!.DestroyShaderModule(device, shader, null);
    }

    private void CreateLightCullDescriptorSets()
    {
        var layouts = stackalloc DescriptorSetLayout[(int)MAX_CONCURRENT_FRAMES];
        for (var i = 0; i < MAX_CONCURRENT_FRAMES; i++) layouts[i] = lightCullDescriptorSetLayout;

        DescriptorSetAllocateInfo alloc = new()
        {
            SType              = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool     = descriptorPool,
            DescriptorSetCount = MAX_CONCURRENT_FRAMES,
            PSetLayouts        = layouts,
        };
        fixed (DescriptorSet* pSets = lightCullDescriptorSets)
        {
            if (vk!.AllocateDescriptorSets(device, &alloc, pSets) != Result.Success)
                throw new Exception("Failed to allocate light-cull descriptor sets");
        }

        for (var i = 0; i < MAX_CONCURRENT_FRAMES; i++)
        {
            DescriptorBufferInfo bufLights = new()
            {
                Buffer = LightStorageBuffers[i].buffer, Offset = 0,
                Range = (ulong)(MAX_LIGHTS * (uint)sizeof(PbrLightGpu)),
            };
            DescriptorBufferInfo bufTileCount = new()
            {
                Buffer = TileLightCountBuffers[i].buffer, Offset = 0,
                Range = (ulong)(MAX_TILE_COUNT * sizeof(uint)),
            };
            DescriptorBufferInfo bufTileIdx = new()
            {
                Buffer = TileLightIndicesBuffers[i].buffer, Offset = 0,
                Range = (ulong)(MAX_TILE_COUNT * MAX_LIGHTS_PER_TILE * sizeof(uint)),
            };

            var writes = stackalloc WriteDescriptorSet[3];
            for (uint b = 0; b < 3; b++)
            {
                writes[b] = new WriteDescriptorSet
                {
                    SType           = StructureType.WriteDescriptorSet,
                    DstSet          = lightCullDescriptorSets[i],
                    DstBinding      = b,
                    DescriptorType  = DescriptorType.StorageBuffer,
                    DescriptorCount = 1,
                };
            }
            writes[0].PBufferInfo = &bufLights;
            writes[1].PBufferInfo = &bufTileCount;
            writes[2].PBufferInfo = &bufTileIdx;

            vk!.UpdateDescriptorSets(device, 3, writes, 0, null);
        }
    }

    // CPU side of the light-cull pass. Computes invViewProj + tile counts from the
    // current camera/swapchain extent, pushes them, dispatches one group per tile,
    // and barriers compute-write -> fragment-read on the two tile buffers.
    private void RecordLightCullPass(CommandBuffer cmd, uint frameIndex, Camera cam,
                                     uint lightCount, uint tileCountX, uint tileCountY)
    {
        if (tileCountX == 0 || tileCountY == 0) return;

        // ── Bind + push constants ────────────────────────────────────────
        vk!.CmdBindPipeline(cmd, PipelineBindPoint.Compute, lightCullPipeline);
        var dset = lightCullDescriptorSets[frameIndex];
        vk!.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute,
            lightCullPipelineLayout, 0, 1, &dset, 0, null);

        Matrix4x4 view = cam.GetViewMatrix();
        Matrix4x4 proj = cam.GetProjectionMatrix(
            (float)swapChainExtent.Width / swapChainExtent.Height, 0.1f, 100.0f);
        // The lighting fragment shader sees a Y-flipped projection (renderer flips
        // proj.M22 in UpdateGeometryFrameUBO). We must build invViewProj from the
        // SAME flipped matrix so the cull frustum lines up with where pixels actually
        // sample world positions from the g-buffer.
        proj.M22 *= -1f;
        Matrix4x4 vp = view * proj;
        Matrix4x4 invVP;
        if (!Matrix4x4.Invert(vp, out invVP))
            invVP = Matrix4x4.Identity;

        var push = new LightCullPushConstants
        {
            InvViewProj = invVP,
            CamPos      = new Vector4(cam.GetPosition(), 1f),
            ScreenSize  = new Vector2(swapChainExtent.Width, swapChainExtent.Height),
            TileCountX  = tileCountX,
            TileCountY  = tileCountY,
            LightCount  = lightCount,
        };
        vk!.CmdPushConstants(cmd, lightCullPipelineLayout, ShaderStageFlags.ComputeBit,
            0, (uint)sizeof(LightCullPushConstants), &push);

        // One thread group per tile (each group is 16×16 = 256 threads cooperating).
        vk!.CmdDispatch(cmd, tileCountX, tileCountY, 1);

        // ── Barrier: compute writes -> fragment shader reads ─────────────
        // The lighting pass's fragment shader reads tileLightCount + tileLightIndices.
        var postBarriers = stackalloc BufferMemoryBarrier[2];
        postBarriers[0] = new BufferMemoryBarrier
        {
            SType         = StructureType.BufferMemoryBarrier,
            SrcAccessMask = AccessFlags.ShaderWriteBit,
            DstAccessMask = AccessFlags.ShaderReadBit,
            SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
            Buffer = TileLightCountBuffers[frameIndex].buffer,
            Offset = 0,
            Size   = Vk.WholeSize,
        };
        postBarriers[1] = new BufferMemoryBarrier
        {
            SType         = StructureType.BufferMemoryBarrier,
            SrcAccessMask = AccessFlags.ShaderWriteBit,
            DstAccessMask = AccessFlags.ShaderReadBit,
            SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
            Buffer = TileLightIndicesBuffers[frameIndex].buffer,
            Offset = 0,
            Size   = Vk.WholeSize,
        };
        vk!.CmdPipelineBarrier(cmd,
            PipelineStageFlags.ComputeShaderBit,
            PipelineStageFlags.FragmentShaderBit,
            0, 0, null, 2, postBarriers, 0, null);
    }
}