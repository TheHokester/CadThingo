using System.Numerics;
using System.Runtime.InteropServices;
using CadThingo.VulkanEngine.GLTF;
using ImGuiNET;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer;
/* How the shadows with ray query work 
 * Three parts. The flow is: deferred shading hands the lighting pass a world-space position per pixel → ray query asks "is anything between that position and the light?" → AS makes that question fast.                       
                                                                                                                                                                                                                             
  1. How "this pixel" becomes "this point in the world"                                                                                                                                                                        
   
  Your geometry pass doesn't draw colors — it draws a g-buffer. For every pixel the screen covers, the geometry shader writes the world-space position of whatever surface is there into GBuffer_Position, plus the            
  normal/albedo/material into the others. That image is just a 2D grid of world coordinates indexed by screen pixel.

  The lighting pass then runs a fullscreen triangle. There's one fragment shader invocation per pixel of the framebuffer. Inside that shader, gBufferPositionSampler.Sample(input.UV) reads back the world position that the
  geometry pass wrote for that exact pixel. So WorldPos in PSMain literally means "the 3D point that this screen pixel is showing."

  That's the bridge. Once you have WorldPos per pixel, "shadow ray from this pixel to the light" makes sense — you cast from WorldPos toward lightPos, and if anything is in the way, the pixel's lit by something other than
  that light. No connection to triangles or meshes anymore — just two world points and a yes/no question.

  2. What an acceleration structure actually is

  Naive shadow trace: for each pixel, walk every triangle in the scene and test ray-triangle intersection. Viking room is ~3800 triangles, screen is ~1M pixels, one light → ~3.8 billion intersection tests per frame. Dead.

  An acceleration structure is a spatial index. It's a tree built over your geometry where each interior node holds a bounding box and each leaf holds a triangle. To check if a ray hits anything, you start at the root: does
   the ray intersect this node's box? If no, skip the entire subtree. If yes, recurse. You touch O(log N) triangles instead of O(N).

  The hardware on your 4070 Ti has dedicated silicon for this traversal — that's what RT cores are. RayQuery.TraceRayInline doesn't loop in software; it dispatches to those cores. That's why ~1M shadow rays per frame is
  cheap.

  Tree shape, build algorithm, node packing — all driver/hardware specifics behind vkCmdBuildAccelerationStructures. You hand it triangles + a flag like PreferFastTrace, and it picks the structure for you.

  3. Why two levels (BLAS + TLAS)

  Imagine 100 instances of the viking room scattered around. If everything were one big AS, you'd:
  - Pay to retesselate the same mesh into the structure 100 times (huge build cost),
  - Have to rebuild the entire thing every time any instance moves.

  Two levels split that:

  BLAS (Bottom-Level) — built per unique mesh, in mesh-local space. Holds the actual triangles. Expensive to build (it's the spatial tree over real geometry), but you only build it once per mesh and cache it. Your blasCache
   is keyed by Mesh* for exactly this reason.

  TLAS (Top-Level) — built per scene. Doesn't hold triangles at all. Each entry is a tiny record: a 3×4 transform + a pointer to a BLAS. So 100 viking rooms = one BLAS + 100 small TLAS records. When a viking room moves, you
   only rebuild the TLAS (cheap — it's effectively a scene graph in spatial form). The BLAS is untouched.
 */
public unsafe partial class Renderer
{
    private void CreateDepthResources()
    {
        // Depth buffer matches the render target extent — sampled / depth-tested
        // by the deferred passes which all run at renderExtent.
        var width = renderExtent.Width;
        var height = renderExtent.Height;

        depthImageResource = new ImageResource(vk, device, "Depth", Format.D32Sfloat, new Extent2D(width, height),
            ImageUsageFlags.DepthStencilAttachmentBit | ImageUsageFlags.InputAttachmentBit,
            ImageLayout.Undefined, ImageLayout.DepthStencilAttachmentOptimal);
    }

    private void CreateGBufferResources()
    {
        // G-buffers + lighting all run at the render extent (== panel size in
        // editor mode), not necessarily the swapchain size.
        var width = renderExtent.Width;
        var height = renderExtent.Height;
        // SampledBit required: the lighting pass samples these via CombinedImageSampler descriptors.
        const ImageUsageFlags gBufferUsage =
            ImageUsageFlags.ColorAttachmentBit | ImageUsageFlags.InputAttachmentBit | ImageUsageFlags.SampledBit;
        gBufferPosition = new ImageResource(vk, device, "GBuffer_Position", Format.R32G32B32A32Sfloat,
            new Extent2D(width, height), gBufferUsage,
            ImageLayout.Undefined, ImageLayout.ShaderReadOnlyOptimal);
        gBufferNormal = new ImageResource(vk, device, "GBuffer_Normal", Format.R32G32B32A32Sfloat,
            new Extent2D(width, height), gBufferUsage,
            ImageLayout.Undefined, ImageLayout.ShaderReadOnlyOptimal);
        gBufferAlbedo = new ImageResource(vk, device, "GBuffer_Albedo", Format.R8G8B8A8Unorm,
            new Extent2D(width, height), gBufferUsage,
            ImageLayout.Undefined, ImageLayout.ShaderReadOnlyOptimal);
        gBufferMaterial = new ImageResource(vk, device, "GBuffer_Material", Format.R8G8B8A8Unorm,
            new Extent2D(width, height), gBufferUsage,
            ImageLayout.Undefined, ImageLayout.ShaderReadOnlyOptimal);
        // Emissive G-buffer: HDR range. Geometry pass writes emissiveSampler.rgb;
        // lighting pass adds it to the final color before tone-mapping.
        gBufferEmissive = new ImageResource(vk, device, "GBuffer_Emissive", Format.R16G16B16A16Sfloat,
            new Extent2D(width, height), gBufferUsage,
            ImageLayout.Undefined, ImageLayout.ShaderReadOnlyOptimal);
    }

    public CommandBuffer BeginSingleTimeCommands()
    {
        CommandBufferAllocateInfo allocInfo = new()
        {
            SType = StructureType.CommandBufferAllocateInfo,
            CommandPool = commandPool,
            Level = CommandBufferLevel.Primary,
            CommandBufferCount = 1,
        };
        vk!.AllocateCommandBuffers(device, &allocInfo, out CommandBuffer cmd);
        CommandBufferBeginInfo beginInfo = new()
        {
            SType = StructureType.CommandBufferBeginInfo,
            Flags = CommandBufferUsageFlags.OneTimeSubmitBit,
        };
        vk!.BeginCommandBuffer(cmd, &beginInfo);
        return cmd;
    }

    public void EndSingleTimeCommands(CommandBuffer cmd)
    {
        vk!.EndCommandBuffer(cmd);
        SubmitInfo submit = new()
        {
            SType = StructureType.SubmitInfo,
            CommandBufferCount = 1,
            PCommandBuffers = &cmd,
        };
        vk!.QueueSubmit(graphicsQueue, 1, &submit, default);
        vk!.QueueWaitIdle(graphicsQueue);
        vk!.FreeCommandBuffers(device, commandPool, 1, &cmd);
    }

    public void CreateBuffer(ulong size, BufferUsageFlags usage, MemoryPropertyFlags memProps,
        out Buffer buffer, out SubAlloc alloc)
    {
        BufferCreateInfo bufferInfo = new()
        {
            SType = StructureType.BufferCreateInfo,
            Size = size,
            Usage = usage,
            SharingMode = SharingMode.Exclusive,
        };
        if (vk!.CreateBuffer(device, &bufferInfo, null, out buffer) != Result.Success)
            throw new Exception("Failed to create buffer");

        // DeviceAddress flag is already baked into every buffer-bucket block by
        // the allocator — no per-call PNext chain needed. Caller still has to set
        // ShaderDeviceAddressBit in `usage` for the buffer itself, which they do.
        alloc = memAllocator.AllocateForBuffer(buffer, memProps);
    }

    public void CopyBuffer(Buffer src, Buffer dst, ulong size)
    {
        var cmd = BeginSingleTimeCommands();
        BufferCopy region = new()
        {
            SrcOffset = 0,
            DstOffset = 0,
            Size = size,
        };
        vk!.CmdCopyBuffer(cmd, src, dst, 1, &region);
        EndSingleTimeCommands(cmd);
    }

    public void UploadBufferData(Buffer dst, long dstOffset, void* srcData, ulong size)
    {
        CreateBuffer(size, BufferUsageFlags.TransferSrcBit,
            MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit,
            out var staging, out var stagingAlloc);

        void* mapped = memAllocator.GetMapped(stagingAlloc);
        System.Buffer.MemoryCopy(srcData, mapped, (long)size, (long)size);

        var cmd = BeginSingleTimeCommands();
        BufferCopy region = new() { SrcOffset = 0, DstOffset = (ulong)dstOffset, Size = size };
        vk!.CmdCopyBuffer(cmd, staging, dst, 1, &region);
        EndSingleTimeCommands(cmd);

        DestroyBuffer(staging, stagingAlloc);
    }

    public void DestroyBuffer(Buffer buffer, SubAlloc alloc)
    {
        if (buffer.Handle != 0) vk!.DestroyBuffer(device, buffer, null);
        memAllocator.Free(alloc);
    }

    private void CreateGBufferSampler()
    {
        SamplerCreateInfo samplerInfo = new()
        {
            SType = StructureType.SamplerCreateInfo,
            MagFilter = Filter.Nearest,
            MinFilter = Filter.Nearest,
            AddressModeU = SamplerAddressMode.ClampToEdge,
            AddressModeV = SamplerAddressMode.ClampToEdge,
            AddressModeW = SamplerAddressMode.ClampToEdge,
            AnisotropyEnable = true,
            MaxAnisotropy = 16,
            BorderColor = BorderColor.FloatOpaqueBlack,
            UnnormalizedCoordinates = false,
            CompareEnable = false,
            CompareOp = CompareOp.Always,
            MipmapMode = SamplerMipmapMode.Nearest,
            MinLod = 0.0f,
            MaxLod = 1.0f,
            MipLodBias = 0.0f,
        };
        if(vk!.CreateSampler(device, &samplerInfo, null, out gBufferSampler) != Result.Success)
        {
            throw new Exception("Failed to create gBuffer sampler");
        }
            
    }
    internal void CreateMappedUniformBuffer(int sizeBytes, ref UboBuffer ubo)
    {
        BufferCreateInfo bufferInfo = new()
        {
            SType = StructureType.BufferCreateInfo,
            Size = (ulong)sizeBytes,
            Usage = BufferUsageFlags.UniformBufferBit,
            SharingMode = SharingMode.Exclusive,
        };
        vk!.CreateBuffer(device, &bufferInfo, null, out ubo.buffer);
        ubo.alloc  = memAllocator.AllocateForBuffer(ubo.buffer,
            MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit);
        ubo.mapped = memAllocator.GetMapped(ubo.alloc);
    }
    
    public void TransitionImageLayout( CommandBuffer cmd, Image image, Format format,  ImageLayout oldLayout,
        ImageLayout newLayout, uint mipLevels = 1)
    {
        bool isDepth = format == Format.D32Sfloat || format == Format.D32SfloatS8Uint || format == Format.D24UnormS8Uint;
        ImageMemoryBarrier barrier = new()
        {
            SType = StructureType.ImageMemoryBarrier,
            OldLayout = oldLayout,
            NewLayout = newLayout,
            SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
            Image = image,
            SubresourceRange =
                new ImageSubresourceRange(isDepth ? ImageAspectFlags.DepthBit : ImageAspectFlags.ColorBit, 0, mipLevels,
                    0, 1)
        };
        
        //Initialize pipeline stage tracking for synchronization timing
        //these stages define when operations must complete and when new operations can begin
        PipelineStageFlags sourceStage;
        PipelineStageFlags destinationStage;
        
        //configure sync for undefined -> transfer layout transition
        //common when preparing images
        if (oldLayout == ImageLayout.Undefined && newLayout == ImageLayout.TransferDstOptimal)
        {
            barrier.SrcAccessMask = 0;
            barrier.DstAccessMask = AccessFlags.TransferWriteBit;
             
            sourceStage = PipelineStageFlags.TopOfPipeBit;
            destinationStage = PipelineStageFlags.TransferBit;

        } 
        //configure sync for transfer -> shader read layout transition
        //pattern prepares uploaded images for shader sampling
        else if (oldLayout == ImageLayout.TransferDstOptimal && newLayout == ImageLayout.ShaderReadOnlyOptimal)
        {
            barrier.SrcAccessMask = AccessFlags.TransferWriteBit;
            barrier.DstAccessMask = AccessFlags.ShaderReadBit;

            sourceStage = PipelineStageFlags.TransferBit;
            destinationStage = PipelineStageFlags.FragmentShaderBit;
        }
        // Swapchain image post-blit → ImGui overlay pass.
        // Blit (transfer write) must finish before color-attachment writes from the UI pipeline.
        else if (oldLayout == ImageLayout.TransferDstOptimal && newLayout == ImageLayout.ColorAttachmentOptimal)
        {
            barrier.SrcAccessMask = AccessFlags.TransferWriteBit;
            barrier.DstAccessMask = AccessFlags.ColorAttachmentWriteBit;

            sourceStage = PipelineStageFlags.TransferBit;
            destinationStage = PipelineStageFlags.ColorAttachmentOutputBit;
        }
        // ImGui overlay → present. Color writes must finish before the presentation engine reads.
        // PresentSrcKhr has no access mask (acquire/release sync handled by semaphores).
        else if (oldLayout == ImageLayout.ColorAttachmentOptimal && newLayout == ImageLayout.PresentSrcKhr)
        {
            barrier.SrcAccessMask = AccessFlags.ColorAttachmentWriteBit;
            barrier.DstAccessMask = 0;

            sourceStage = PipelineStageFlags.ColorAttachmentOutputBit;
            destinationStage = PipelineStageFlags.BottomOfPipeBit;
        }
        // FinalColor: settled in ShaderReadOnlyOptimal by the render-graph final-layout barrier,
        // briefly TransferSrcOptimal for the swapchain blit, then back. Prior shader reads must
        // finish before the blit, and the blit's transfer reads must finish before the next
        // ImGui sample.
        else if (oldLayout == ImageLayout.ShaderReadOnlyOptimal && newLayout == ImageLayout.TransferSrcOptimal)
        {
            barrier.SrcAccessMask = AccessFlags.ShaderReadBit;
            barrier.DstAccessMask = AccessFlags.TransferReadBit;

            sourceStage = PipelineStageFlags.FragmentShaderBit;
            destinationStage = PipelineStageFlags.TransferBit;
        }
        else if (oldLayout == ImageLayout.TransferSrcOptimal && newLayout == ImageLayout.ShaderReadOnlyOptimal)
        {
            barrier.SrcAccessMask = AccessFlags.TransferReadBit;
            barrier.DstAccessMask = AccessFlags.ShaderReadBit;

            sourceStage = PipelineStageFlags.TransferBit;
            destinationStage = PipelineStageFlags.FragmentShaderBit;
        }
        // Pathtracer storage-image transitions
        // Fresh storage image — one-shot init in CreatePathTracingResources.
        // No prior work to wait on; the next user is a compute shader.
        else if (oldLayout == ImageLayout.Undefined && newLayout == ImageLayout.General)
        {
            barrier.SrcAccessMask = 0;
            barrier.DstAccessMask = AccessFlags.ShaderReadBit | AccessFlags.ShaderWriteBit;

            sourceStage      = PipelineStageFlags.TopOfPipeBit;
            destinationStage = PipelineStageFlags.ComputeShaderBit;
        }
        // PT dispatch finished writing ptOutColor; tonemap is about to sample it
        // as a CombinedImageSampler.  Compute-write → fragment-read.
        else if (oldLayout == ImageLayout.General && newLayout == ImageLayout.ShaderReadOnlyOptimal)
        {
            barrier.SrcAccessMask = AccessFlags.ShaderWriteBit;
            barrier.DstAccessMask = AccessFlags.ShaderReadBit;

            sourceStage      = PipelineStageFlags.ComputeShaderBit;
            destinationStage = PipelineStageFlags.FragmentShaderBit;
        }
        // ptOutColor back to General for the next frame's dispatch.  The reader
        // was tonemap's fragment shader; the next user is the compute dispatch.
        else if (oldLayout == ImageLayout.ShaderReadOnlyOptimal && newLayout == ImageLayout.General)
        {
            barrier.SrcAccessMask = AccessFlags.ShaderReadBit;
            barrier.DstAccessMask = AccessFlags.ShaderReadBit | AccessFlags.ShaderWriteBit;

            sourceStage      = PipelineStageFlags.FragmentShaderBit;
            destinationStage = PipelineStageFlags.ComputeShaderBit;
        }
        // FinalColor was sitting in ShaderReadOnly (viewport sample / last frame's
        // graph end-state); tonemap is about to write it as a color attachment.
        else if (oldLayout == ImageLayout.ShaderReadOnlyOptimal && newLayout == ImageLayout.ColorAttachmentOptimal)
        {
            barrier.SrcAccessMask = AccessFlags.ShaderReadBit;
            barrier.DstAccessMask = AccessFlags.ColorAttachmentWriteBit;

            sourceStage      = PipelineStageFlags.FragmentShaderBit;
            destinationStage = PipelineStageFlags.ColorAttachmentOutputBit;
        }
        // Tonemap finished writing FinalColor; next consumers are the swapchain
        // blit (transfer read) and the ImGui viewport sampler (fragment read).
        // Cover both downstream stages so neither needs an extra barrier.
        else if (oldLayout == ImageLayout.ColorAttachmentOptimal && newLayout == ImageLayout.ShaderReadOnlyOptimal)
        {
            barrier.SrcAccessMask = AccessFlags.ColorAttachmentWriteBit;
            barrier.DstAccessMask = AccessFlags.ShaderReadBit | AccessFlags.TransferReadBit;

            sourceStage      = PipelineStageFlags.ColorAttachmentOutputBit;
            destinationStage = PipelineStageFlags.FragmentShaderBit | PipelineStageFlags.TransferBit;
        }
        else
        {
            throw new Exception($"Unsupported layout transition: {oldLayout} -> {newLayout}");
        }
        
        vk.CmdPipelineBarrier(cmd,
            sourceStage,
            destinationStage,
            0,
            0, null, 
            0, null,
            1, &barrier);
    }

    internal void GenerateMipMaps(CommandBuffer cmds, Image image, Format format, uint width, uint height,uint mipLevels)
    {
        ImageMemoryBarrier barrier = new()
        {
            SType = StructureType.ImageMemoryBarrier,
            Image = image,
            SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
            SubresourceRange = new()
            {
                AspectMask = ImageAspectFlags.ColorBit,
                LevelCount = 1,
                LayerCount = 1
            }
        };
        var mipWidth = width;
        var mipHeight = height;
        
        
        for(uint i =1; i < mipLevels; i++)
        {
            barrier.SubresourceRange.BaseMipLevel = i - 1;
            barrier.OldLayout = ImageLayout.TransferDstOptimal;
            barrier.NewLayout = ImageLayout.TransferSrcOptimal;
            barrier.SrcAccessMask = AccessFlags.TransferWriteBit;
            barrier.DstAccessMask = AccessFlags.TransferReadBit;
            
            vk!.CmdPipelineBarrier(cmds, PipelineStageFlags.TransferBit,
                PipelineStageFlags.TransferBit, 0, 
                0, null,
                0, null,
                1, &barrier);

            ImageBlit blit = new()
            {
                SrcOffsets =
                {
                    Element0 = new Offset3D(0, 0, 0),
                    Element1 = new Offset3D((int)mipWidth, (int)mipHeight, 1)
                },
                SrcSubresource =
                {
                    AspectMask = ImageAspectFlags.ColorBit,
                    MipLevel = i - 1,
                    BaseArrayLayer = 0,
                    LayerCount = 1
                },
                DstOffsets =
                {
                    Element0 = new Offset3D(0, 0, 0),
                    Element1 = new Offset3D((int)mipWidth > 1 ? (int)mipWidth / 2 : 1,
                        (int)mipHeight > 1 ? (int)mipHeight / 2 : 1, 1)
                },
                DstSubresource =
                {
                    AspectMask = ImageAspectFlags.ColorBit,
                    MipLevel = i,
                    BaseArrayLayer = 0,
                    LayerCount = 1
                }
            };
            
            vk!.CmdBlitImage(cmds, 
                image, ImageLayout.TransferSrcOptimal,
                image, ImageLayout.TransferDstOptimal,
                1, &blit,
                Filter.Linear);
            
            barrier.OldLayout = ImageLayout.TransferSrcOptimal;
            barrier.NewLayout = ImageLayout.ShaderReadOnlyOptimal;
            barrier.SrcAccessMask = AccessFlags.TransferReadBit;
            barrier.DstAccessMask = AccessFlags.ShaderReadBit;
            
            vk!.CmdPipelineBarrier(cmds, PipelineStageFlags.TransferBit,
                PipelineStageFlags.FragmentShaderBit, 0,
                 0, null,
                 0, null,
                 1, &barrier);
            
            if(mipWidth > 1) mipWidth /= 2;
            if(mipHeight > 1) mipHeight /= 2;
        }
        
        barrier.SubresourceRange.BaseMipLevel = mipLevels - 1;
        barrier.OldLayout = ImageLayout.TransferDstOptimal;
        barrier.NewLayout = ImageLayout.ShaderReadOnlyOptimal;
        barrier.SrcAccessMask = AccessFlags.TransferWriteBit;
        barrier.DstAccessMask = AccessFlags.ShaderReadBit;
        
        vk!.CmdPipelineBarrier(cmds, PipelineStageFlags.TransferBit,
            PipelineStageFlags.FragmentShaderBit, 0,
             0, null,
             0, null,
             1, &barrier);
        
    }
    

    private void CreateDescriptorPool()
    {
        // Sizing budget per frame-in-flight where relevant:
        //   - UniformBuffer:   GeometryFrameUBO + LightingUBO  →  2 × MAX_FRAMES + headroom
        //   - StorageBuffer:   PbrMaterial[]   + InstanceData[] →  2 × MAX_FRAMES + headroom
        //   - SampledImage:    bindless texture array          →  MAX_BINDLESS_TEXTURES × MAX_FRAMES
        //   - Sampler:         bindless samplers (default 0)   →  8 × MAX_FRAMES
        //   - CombinedImageSampler: g-buffer samplers (shared) →  5 + headroom
        //   - AccelerationStructure: TLAS                       →  MAX_FRAMES + headroom
        var poolSizes = new DescriptorPoolSize[]
        {
            new() { Type = DescriptorType.UniformBuffer,            DescriptorCount = 16 },
            // Storage buffer budget: bindless mat+instance (2 × MAX_FRAMES), light SSBO
            // (MAX_FRAMES), cull pass (renderables + cmds + instancesOut + count =
            // 4 × MAX_FRAMES), PBR shadow-alpha set (ShadowEntityInfo + global vb + ib).
            // Round up generously.
            new() { Type = DescriptorType.StorageBuffer,            DescriptorCount = 48 },
            new() { Type = DescriptorType.SampledImage,             DescriptorCount = MAX_BINDLESS_TEXTURES * MAX_CONCURRENT_FRAMES },
            new() { Type = DescriptorType.Sampler,                  DescriptorCount = 8 * MAX_CONCURRENT_FRAMES + 4 },
            // 5 g-buffer samplers (set 1) + 3 IBL samplers × MAX_FRAMES on set 0
            // for both PBR + Transparent pipelines + ImGui viewport + headroom.
            // Probe prefilter sets reuse iblCubeSampler — one CombinedImageSampler
            // each. MaxProbes (16) × MipLevels (9) = 144. Cheap to oversize.
            new() { Type = DescriptorType.CombinedImageSampler,     DescriptorCount = 32 + 200 },
            new() { Type = DescriptorType.AccelerationStructureKhr, DescriptorCount = 4 },
            // IBL bake passes need StorageImage descriptors — one per dispatch.
            // Worst case is the prefilter chain (1 set × prefilteredCubeMipLevels
            // mips) + equirect→cube + irradiance + BRDF LUT ≈ 11 sets. Reflection
            // probes pre-allocate one set per (slot, mip): MaxProbes (16) × MipLevels
            // (9) = 144. Round up.
            new() { Type = DescriptorType.StorageImage,             DescriptorCount = 24 + 200 },
        };
        fixed (DescriptorPoolSize* poolSizesPtr = poolSizes)
        {
            DescriptorPoolCreateInfo poolInfo = new()
            {
                SType = StructureType.DescriptorPoolCreateInfo,
                MaxSets = 48 + 200,
                PoolSizeCount = (uint)poolSizes.Length,
                PPoolSizes = poolSizesPtr,
                Flags = DescriptorPoolCreateFlags.UpdateAfterBindBit | DescriptorPoolCreateFlags.FreeDescriptorSetBit,
            };
            if (vk!.CreateDescriptorPool(device, &poolInfo, null, out descriptorPool) != Result.Success)
            {
                throw new Exception("Failed to create descriptor pool");
            }
        }
    }
   

    // Pipeline-wide size budgets. The actual GPU buffers these size are owned by
    // the pipelines that need them (DrawCullPipeline, LightCullPipeline,
    // PbrDeferredPipeline) or by ResourceManager (materials/instances/bindless
    // textures).
    internal const uint MAX_MATERIALS         = 256;
    internal const uint MAX_INSTANCES         = 4096;
    // Must match ResourceManager.MAX_BINDLESS_TEXTURES — both bound the same
    // descriptor array. 9 = worst-case textures per material post-extensions
    // (5 core + transmission + 3× clearcoat).
    internal const uint MAX_BINDLESS_TEXTURES = MAX_MATERIALS * 9;

    // Lighting pipeline set 0 binding 1 — per-frame StructuredBuffer<PbrLight>.
    // Cap chosen to keep the SSBO at 64KB (1024 × 64B); raise if needed.
    internal const uint MAX_LIGHTS = 1024;

    // Tiled light culling. Tile size matches the compute group
    // (16×16 threads collaborating on one tile). MAX_LIGHTS_PER_TILE bounds the
    // worst-case per-tile slot count; lights past this are dropped (the cull
    // shader saturates the count).
    internal const uint TILE_SIZE           = 16;
    internal const uint MAX_LIGHTS_PER_TILE = 64;
    // Hard cap on tile count — covers up to 3840×2160(4K) (240*135 tiles). The actual
    // per-frame tileCountX/Y depends on swapChainExtent and is uploaded via the
    // frame UBO each frame.
    internal const uint MAX_TILE_COUNT      = 240 * 135;

    // Allocates a host-visible, coherent, persistently-mapped SSBO. Optional extra
    // usage bits let callers turn the same buffer into an indirect-cmd / indirect-
    // count source on top of plain storage usage.
    internal void CreateMappedStorageBuffer(ulong sizeBytes, ref UboBuffer ubo,
        BufferUsageFlags extraUsage = 0)
    {
        BufferCreateInfo bufferInfo = new()
        {
            SType = StructureType.BufferCreateInfo,
            Size = sizeBytes,
            Usage = BufferUsageFlags.StorageBufferBit | extraUsage,
            SharingMode = SharingMode.Exclusive,
        };
        vk!.CreateBuffer(device, &bufferInfo, null, out ubo.buffer);
        ubo.alloc  = memAllocator.AllocateForBuffer(ubo.buffer,
            MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit);
        ubo.mapped = memAllocator.GetMapped(ubo.alloc);
    }

    // Lights SSBO
    // Lives on Renderer (not PbrDeferredPipeline) so the pathtracer can read
    // it without depending on the deferred lighting pipeline existing. The
    // scene owns the LightComponents; this is just the per-frame GPU mirror.

    internal UboBuffer[] lightStorageBuffers = new UboBuffer[MAX_CONCURRENT_FRAMES];
    private readonly List<LightComponent> _lightScratch = new();

    /// <summary>Buffer holding packed PbrLightGpu records for the given frame
    /// slot. Bound by every pipeline that needs scene lighting. Stable for the
    /// renderer's lifetime.</summary>
    public Buffer GetLightStorageBuffer(uint frame) => lightStorageBuffers[frame].buffer;

    /// <summary>Number of valid lights packed by the most recent UpdateLights.</summary>
    public uint LightCount { get; private set; }

    private void CreateLightBuffers()
    {
        for (int i = 0; i < MAX_CONCURRENT_FRAMES; i++)
        {
            CreateMappedStorageBuffer(
                (ulong)(MAX_LIGHTS * (uint)sizeof(PbrLightGpu)),
                ref lightStorageBuffers[i]);
        }
    }

    private void DestroyLightBuffers()
    {
        for (int i = 0; i < lightStorageBuffers.Length; i++)
        {
            if (lightStorageBuffers[i].buffer.Handle != 0)
                DestroyBuffer(lightStorageBuffers[i].buffer, lightStorageBuffers[i].alloc);
        }
    }

    /// <summary>
    /// Copy every scene material into the per-frame material SSBO, plus a
    /// fallback entry at slot [MaterialCount] for legacy / procedural geometry
    /// without an assigned material index. Every rendering path (deferred,
    /// forward+, pathtracer) calls this once per frame so live edits in the
    /// inspector show up on the next frame regardless of which renderer is
    /// active. Returns the material count actually written (excluding the
    /// fallback).
    /// </summary>
    public uint UpdateMaterials(uint frameIndex, Scene scene)
    {
        int matCount = scene.MaterialCount;
        if (matCount + 1 > (int)MAX_MATERIALS)
            throw new InvalidOperationException(
                $"Scene material count ({matCount}) exceeds MAX_MATERIALS ({MAX_MATERIALS}).");

        PbrMaterial* matPtr = (PbrMaterial*)Engine.ResourceManager.GetMaterialMapped(frameIndex);
        for (int mi = 0; mi < matCount; mi++) matPtr[mi] = scene.Materials[mi];

        matPtr[matCount] = new PbrMaterial
        {
            BaseColorFactor          = new Vector4(1, 1, 1, 1),
            EmissiveFactor           = Vector3.Zero,
            AlphaCutoff              = 0f,
            MetallicFactor           = 0.3f,
            RoughnessFactor          = 0.7f,
            Flags                    = 0,
            BaseColorTex             = GltfDefaults.BaseColorIndex,
            PhysicalDescriptorTex    = GltfDefaults.MetallicRoughnessIndex,
            NormalTex                = GltfDefaults.NormalIndex,
            OcclusionTex             = GltfDefaults.OcclusionIndex,
            EmissiveTex              = GltfDefaults.EmissiveIndex,
            TransmissionFactor       = 0f,
            Ior                      = 1.5f,
            ClearcoatFactor          = 0f,
            ClearcoatRoughnessFactor = 0f,
            TransmissionTex          = GltfDefaults.OcclusionIndex,
            ClearcoatTex             = GltfDefaults.OcclusionIndex,
            ClearcoatRoughnessTex    = GltfDefaults.OcclusionIndex,
            ClearcoatNormalTex       = GltfDefaults.NormalIndex,
        };

        return (uint)matCount;
    }

    /// <summary>Walks scene lights into the per-frame light SSBO. Returns the
    /// packed light count, also cached in <see cref="LightCount"/>. Every
    /// rendering path (deferred, forward+, pathtracer) calls this once per frame
    /// before recording its draws.</summary>
    public uint UpdateLights(uint frameIndex, Scene scene)
    {
        scene.EnumerateLights(_lightScratch);

        PbrLightGpu* lightPtr = (PbrLightGpu*)lightStorageBuffers[frameIndex].mapped;
        uint count = 0;
        foreach (var light in _lightScratch)
        {
            if (count >= MAX_LIGHTS) break;

            // World-space position from the owner transform if present.
            Vector3 worldPos = Vector3.Zero;
            if (light.Owner != null)
            {
                var t = light.Owner->GetComponent<TransformComponent>();
                if (t != null)
                {
                    var w = *t.GetWorldMatrix();
                    worldPos = new Vector3(w.M41, w.M42, w.M43);
                }
            }

            // Normalize direction — guard against zero-vector default.
            Vector3 dir = light.Direction.LengthSquared() > 1e-8f
                ? Vector3.Normalize(light.Direction)
                : new Vector3(0, -1, 0);

            // Range: -1 sentinel marks directional lights so the shader can branch
            // on attenuation without inspecting Type for the most common test.
            float range = light.Type == LightType.Directional ? -1f : light.Range;

            lightPtr[count] = new PbrLightGpu
            {
                positionRange  = new Vector4(worldPos, range),
                colorIntensity = new Vector4(light.Color, light.Intensity),
                directionType  = new Vector4(dir, (float)(uint)light.Type),
                spotCones      = new Vector4(light.InnerConeCos, light.OuterConeCos,
                    light.CastShadows ? 1f : 0f, light.Radius),
            };
            count++;
        }

        LightCount = count;
        return count;
    }

   

    

    /// <summary>
    /// Writes a bindless texture to the given descriptor set at the given binding.
    /// All descriptor sets must use the same layout.
    /// </summary>
    /// <param name="index"></param>
    /// <param name="texture"></param>
    /// <param name="sets"></param>
    /// <param name="binding"></param>
    /// <exception cref="ArgumentOutOfRangeException"></exception>
    public void WriteBindlessTexture(int index, Texture texture, DescriptorSet[] sets, uint binding)
    {
        if (index < 0 || index >= MAX_BINDLESS_TEXTURES)
            throw new ArgumentOutOfRangeException(nameof(index), $"bindless texture index {index} out of [0, {MAX_BINDLESS_TEXTURES}).");

        DescriptorImageInfo imgInfo = new()
        {
            ImageView = texture.View,
            ImageLayout = ImageLayout.ShaderReadOnlyOptimal,
        };

        var writes = stackalloc WriteDescriptorSet[sets.Length];
        for (var f = 0; f < sets.Length; f++)
        {
            writes[f] = new WriteDescriptorSet
            {
                SType = StructureType.WriteDescriptorSet,
                DstSet = sets[f],
                DstBinding = binding,
                DstArrayElement = (uint)index,
                DescriptorType = DescriptorType.SampledImage,
                DescriptorCount = 1,
                PImageInfo = &imgInfo,
            };
        }
        vk!.UpdateDescriptorSets(device, (uint)sets.Length, writes, 0, null);
    }

    private Extent2D ChooseSwapExtent(SurfaceCapabilitiesKHR capabilities)
    {
        if (capabilities.CurrentExtent.Width != uint.MaxValue)
            return capabilities.CurrentExtent;
        else
        {
            var framebufferSize = Engine.window!.FramebufferSize;
            Extent2D actualExtent = new()
            {
                Width = (uint)framebufferSize.X,
                Height = (uint)framebufferSize.Y
            };
            actualExtent.Width = Math.Clamp(actualExtent.Width, capabilities.MinImageExtent.Width, capabilities.MaxImageExtent.Width);
            actualExtent.Height = Math.Clamp(actualExtent.Height, capabilities.MinImageExtent.Height, capabilities.MaxImageExtent.Height);
            
            return actualExtent;
        }
    }

    private PresentModeKHR ChooseSwapPresentMode(PresentModeKHR[] presentModes)
    {
        foreach (var availablePresentMode in presentModes)
        {
            if(availablePresentMode == PresentModeKHR.MailboxKhr)
                return availablePresentMode;
        }
        
        return PresentModeKHR.FifoKhr;
    }

    private SurfaceFormatKHR ChooseSwapSurfaceFormat(SurfaceFormatKHR[] formats)
    {
        foreach (var availableFormat in formats)
        {
            if(availableFormat.Format == Format.B8G8R8A8Srgb && availableFormat.ColorSpace == ColorSpaceKHR.PaceSrgbNonlinearKhr)
                return availableFormat;
        }
        
        return formats[0];
    }
}

// Mirrors PbrUtils.slang::PbrLight under std430 (16B alignment, no padding needed —
// the struct is exactly 4 × float4 = 64B).
[StructLayout(LayoutKind.Sequential)]
public struct PbrLightGpu
{
    public Vector4 positionRange;   // xyz = world pos, w = range (point/spot; ignored for directional)
    public Vector4 colorIntensity;  // rgb = linear color, a = intensity
    public Vector4 directionType;   // xyz = direction (dir/spot, world-space), w = LightType as float
    public Vector4 spotCones;       // x = innerCos, y = outerCos, z = castShadows (0/1), w = lightRadius
}

// Per-instance row in the geometry pipeline's instance SSBO. Shader sees this as
// PbrUtils.slang::InstanceData (std430 layout, stride 80B). Padding keeps the C#
// struct aligned to the shader struct so a Span<InstanceDataGPU> blits cleanly.
[StructLayout(LayoutKind.Sequential)]
public struct InstanceDataGPU
{
    public Matrix4x4 model;
    public uint materialIndex;
    public uint _pad0, _pad1, _pad2;
}

// Cull-pass input record. CPU writes one per scene entity into RenderableInputBuffers
// each frame; the compute shader reads it and emits an indirect draw + InstanceData
// when the bounding sphere passes the frustum test. Std430 alignment, total 96B.
[StructLayout(LayoutKind.Sequential)]
public struct RenderableInputGpu
{
    public Matrix4x4 model;        // 64B
    public Vector4 sphereLocal;    // 16B  (xyz center, w radius — local space)
    public uint indexCount;        //  4B  ┐
    public uint firstIndex;        //  4B  │  pulled straight from Mesh
    public uint materialIndex;     //  4B  │
    public uint _pad;              //  4B  ┘  std430 16B alignment
}

// CPU-side transparent draw record. Forward+ transparent pass walks a sorted
// list of these (back-to-front by view-space Z) and issues one push-constant +
// vkCmdDrawIndexed per entity. Not a GPU SSBO type — the per-draw model + matidx
// ride in push constants since the count is typically small.
public struct TransparentDraw
{
    public Matrix4x4 Model;
    public uint      MaterialIndex;
    public uint      IndexCount;
    public uint      FirstIndex;     // into Engine.ResourceManager.GlobalIndexBuffer
    public float     ViewDepth;      // view-space Z of the entity's origin; sort key
}

// Mirrors VkDrawIndexedIndirectCommand exactly. Compute writes this struct into the
// indirect-cmd buffer per surviving renderable; vkCmdDrawIndexedIndirectCount
// consumes them.
[StructLayout(LayoutKind.Sequential)]
public struct DrawIndexedIndirectCommandGpu
{
    public uint indexCount;
    public uint instanceCount;
    public uint firstIndex;
    public int  vertexOffset;
    public uint firstInstance;
}


// Per-frame mapped UBO/SSBO bundle. Storage is owned by the renderer's
// GpuMemoryAllocator; Dispose is intentionally absent — callers free through
// Renderer.DestroyBuffer(b.buffer, b.alloc) so both the VkBuffer handle AND
// the suballocation are released together.
unsafe struct UboBuffer
{
    public Buffer    buffer;
    public SubAlloc  alloc;
    public void*     mapped;
}

// Mirrors PbrUtils.slang::PbrMaterial under std430. First 64B holds the core
// glTF metallic-roughness block (laid out so the vec3 emissive factor's
// trailing 4B slack absorbs AlphaCutoff). The next 32B carries KHR extension
// data — transmission, IOR, clearcoat. All scalar / 4B-aligned so std430
// packs them with zero padding. Fields the shaders don't yet consume sit
// here as data only; uploaded every frame regardless. Total: 96B.
[StructLayout(LayoutKind.Sequential)]
public struct PbrMaterial
{
    public Vector4 BaseColorFactor;
    public Vector3 EmissiveFactor;
    public float AlphaCutoff;
    public float MetallicFactor;
    public float RoughnessFactor;
    public uint Flags;//bit 0 alphaMask, bit 1 doubleSided, bit 2 alphaBlend, ...
    public int BaseColorTex;
    public int PhysicalDescriptorTex;
    public int NormalTex;
    public int OcclusionTex;
    public int EmissiveTex;

    // KHR_materials_transmission + KHR_materials_ior
    public float TransmissionFactor;     // 0 = opaque, 1 = full transmission
    public float Ior;                    // index of refraction; glTF default 1.5

    // KHR_materials_clearcoat
    public float ClearcoatFactor;        // 0 = no coat, 1 = full coat
    public float ClearcoatRoughnessFactor;

    public int TransmissionTex;          // R channel multiplies TransmissionFactor
    public int ClearcoatTex;             // R channel multiplies ClearcoatFactor
    public int ClearcoatRoughnessTex;    // G channel multiplies ClearcoatRoughnessFactor
    public int ClearcoatNormalTex;       // tangent-space normal for the coat layer
}

public enum AlphaMode : uint
{
    Opaque = 0,
    Mask   = 1,
    Blend  = 2,
}

public static class PbrMaterialAlphaExtensions
{
    // BLEND takes precedence over MASK if both bits are accidentally set
    // (shouldn't happen from glTF — they're mutually exclusive there).
    public static AlphaMode GetAlphaMode(this in PbrMaterial mat)
    {
        if ((mat.Flags & 0x4u) != 0u) return AlphaMode.Blend;
        if ((mat.Flags & 0x1u) != 0u) return AlphaMode.Mask;
        return AlphaMode.Opaque;
    }
}

public unsafe class ImageResource : IDisposable
{
    private readonly Vk     _vk;
    private readonly Device _device;
    private readonly GpuMemoryAllocator _allocator;
    private bool            _disposed;
    public bool IsAllocated { get; set; }

    public string _name { get; }
    public Format _format;
    Extent2D _extent;
    ImageUsageFlags _usage;
    public ImageLayout _initialLayout;
    public ImageLayout _finalLayout;


    public VkImage Image;
    public SubAlloc ImageAlloc;
    public ImageView ImageView;


    public ImageResource(
        Vk vk, Device device,
        string name, Format format, Extent2D extent,
        ImageUsageFlags usage,
        ImageLayout initialLayout = ImageLayout.Undefined,
        ImageLayout finalLayout   = ImageLayout.ShaderReadOnlyOptimal)
    {
        _vk = vk;
        _device = device;
        _allocator = Engine.renderer!.memAllocator;
        _name = name;
        _format = format;
        _extent = extent;
        _usage = usage;
        _initialLayout = initialLayout;
        _finalLayout = finalLayout;
    }

    /// <summary>
    /// Adopts already-allocated image handles. Used for resources whose creation
    /// flow doesn't fit the graph's <see cref="Allocate"/> path (e.g. textures
    /// uploaded from disk via a staging buffer). Dispose semantics match the
    /// allocate path — handles are destroyed on Dispose().
    /// </summary>
    public ImageResource(
        Vk vk, Device device,
        string name, Format format, Extent2D extent,
        ImageUsageFlags usage,
        VkImage image, SubAlloc alloc, ImageView view)
    {
        _vk = vk;
        _device = device;
        _allocator = Engine.renderer!.memAllocator;
        _name = name;
        _format = format;
        _extent = extent;
        _usage = usage;
        _initialLayout = ImageLayout.ShaderReadOnlyOptimal;
        _finalLayout   = ImageLayout.ShaderReadOnlyOptimal;
        Image = image;
        ImageAlloc = alloc;
        ImageView = view;
        IsAllocated = true;
    }
    public void Allocate(PhysicalDevice physicalDevice)
    {
        //configure image creation info based on resource properties
        ImageCreateInfo imageInfo = new()
        {
            SType = StructureType.ImageCreateInfo,
            ImageType = ImageType.Type2D,
            Format = _format,
            Extent = new Extent3D(_extent.Width, _extent.Height, 1),
            MipLevels = 1,
            ArrayLayers = 1,
            Samples = SampleCountFlags.Count1Bit,
            Tiling = ImageTiling.Optimal,
            Usage = _usage,
            SharingMode = SharingMode.Exclusive,
            InitialLayout = _initialLayout
        };
        if (_vk.CreateImage(_device, ref imageInfo, null, out Image) != Result.Success)
        {
            throw new Exception("Failed to create image for resource " + _name);
        }

        ImageAlloc = _allocator.AllocateForImage(Image, MemoryPropertyFlags.DeviceLocalBit);

        bool isDepth = _format is Format.D32Sfloat
            or Format.D24UnormS8Uint
            or Format.D16Unorm;

        var viewInfo = new ImageViewCreateInfo()
        {
            SType = StructureType.ImageViewCreateInfo,
            Image = Image,
            ViewType = ImageViewType.Type2D,
            Format = _format,
            SubresourceRange = new ImageSubresourceRange()
            {
                AspectMask = isDepth ? ImageAspectFlags.DepthBit : ImageAspectFlags.ColorBit,
                BaseMipLevel = 0,
                LevelCount = 1,
                BaseArrayLayer = 0,
                LayerCount = 1
            }
        };

        if (_vk.CreateImageView(_device, ref viewInfo, null, out ImageView) != Result.Success)
        {
            throw new Exception("Failed to create image view for resource " + _name);
        };
        IsAllocated = true;

    }



    public void Dispose()
    {
        if (_disposed) return;

        // Destroy in reverse-creation order — mirrors C++ RAII destruction
        if (ImageView.Handle   != 0) _vk.DestroyImageView(_device, ImageView,   null);
        if (Image.Handle  != 0)      _vk.DestroyImage    (_device, Image,  null);
        _allocator.Free(ImageAlloc);

        _disposed = true;
    }

}

/// <summary>
/// Composes an <see cref="ImageResource"/> + <see cref="Sampler"/> as a single
/// shader-readable texture object. Use for sampled textures (loaded from disk,
/// dummy fallbacks, baked LUTs); <see cref="ImageResource"/> alone is reserved
/// for render-graph attachments.
/// </summary>
public unsafe class Texture : IDisposable
{
    private readonly Vk _vk;
    private readonly Device _device;
    private bool _disposed;

    public ImageResource Resource { get; }
    public Sampler Sampler { get; }

    public VkImage Image     => Resource.Image;
    public ImageView View    => Resource.ImageView;

    public Texture(Vk vk, Device device, ImageResource resource, Sampler sampler)
    {
        _vk = vk;
        _device = device;
        Resource = resource;
        Sampler = sampler;
    }

    /// <summary>
    /// Loads a 2D RGBA texture from disk into a device-local image + linear sampler.
    /// </summary>
    public static Texture CreateTextureFromPath(Renderer renderer, string path, Format format)
    {
        using var img = SixLabors.ImageSharp.Image.Load<SixLabors.ImageSharp.PixelFormats.Rgba32>(path);
        var pixels = new SixLabors.ImageSharp.PixelFormats.Rgba32[img.Width * img.Height];
        img.CopyPixelDataTo(pixels);
        fixed (SixLabors.ImageSharp.PixelFormats.Rgba32* p = pixels)
        {
            return CreateTextureFromMemory(renderer, (byte*)p,
                (uint)img.Width, (uint)img.Height, format,
                new Extent3D((uint)img.Width, (uint)img.Height, 1));
        }
    }

    /// <summary>
    /// Uploads raw RGBA pixel data into a device-local image + linear sampler.
    /// <paramref name="width"/>/<paramref name="height"/> size the staging copy;
    /// <paramref name="extent"/> sizes the destination image.
    /// </summary>
    public static Texture CreateTextureFromMemory(Renderer renderer, byte* pixels,
        uint width, uint height, Format format, Extent3D extent)
    {
        var vk = Globals.vk!;
        var device = renderer.device;
        var physicalDevice = renderer.physicalDevice;

        ulong imageSize = (ulong)width * (ulong)height * 4UL;
        uint mipLevels = (uint)Math.Floor(Math.Log2(Math.Max(width, height))) + 1;

        // Stage pixels in a host-visible buffer.
        renderer.CreateBuffer(imageSize, BufferUsageFlags.TransferSrcBit,
            MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit,
            out var staging, out var stagingAlloc);

        void* mapped = renderer.memAllocator.GetMapped(stagingAlloc);
        System.Buffer.MemoryCopy(pixels, mapped, (long)imageSize, (long)imageSize);

        const ImageUsageFlags usage = ImageUsageFlags.TransferDstBit |ImageUsageFlags.TransferSrcBit | ImageUsageFlags.SampledBit;
        var extent2D = new Extent2D(extent.Width, extent.Height);

        // Device-local image.
        ImageCreateInfo imageInfo = new()
        {
            SType = StructureType.ImageCreateInfo,
            ImageType = ImageType.Type2D,
            Format = format,
            Extent = extent,
            MipLevels = mipLevels,
            ArrayLayers = 1,
            Samples = SampleCountFlags.Count1Bit,
            Tiling = ImageTiling.Optimal,
            Usage = usage,
            SharingMode = SharingMode.Exclusive,
            InitialLayout = ImageLayout.Undefined,
        };
        if (vk.CreateImage(device, &imageInfo, null, out var image) != Result.Success)
            throw new Exception("Failed to create image for texture");

        var imageAlloc = renderer.memAllocator.AllocateForImage(image, MemoryPropertyFlags.DeviceLocalBit);

        // Transition → copy → transition (single-time command).
        var cmd = renderer.BeginSingleTimeCommands();
        renderer.TransitionImageLayout(cmd, image, format, ImageLayout.Undefined, ImageLayout.TransferDstOptimal, mipLevels: mipLevels);
        BufferImageCopy region = new()
        {
            BufferOffset = 0,
            BufferRowLength = 0,
            BufferImageHeight = 0,
            ImageSubresource = new ImageSubresourceLayers
            {
                AspectMask = ImageAspectFlags.ColorBit,
                MipLevel = 0,
                BaseArrayLayer = 0,
                LayerCount = 1,
            },
            ImageOffset = new Offset3D(0, 0, 0),
            ImageExtent = extent,
        };
        vk.CmdCopyBufferToImage(cmd, staging, image, ImageLayout.TransferDstOptimal, 1, &region);
        renderer.GenerateMipMaps(cmd, image, format, width, height,mipLevels);
        renderer.EndSingleTimeCommands(cmd);

        renderer.DestroyBuffer(staging, stagingAlloc);

        ImageViewCreateInfo viewInfo = new()
        {
            SType = StructureType.ImageViewCreateInfo,
            Image = image,
            ViewType = ImageViewType.Type2D,
            Format = format,
            SubresourceRange = new ImageSubresourceRange
            {
                AspectMask = ImageAspectFlags.ColorBit,
                BaseMipLevel = 0, LevelCount = mipLevels,
                BaseArrayLayer = 0, LayerCount = 1,
            },
        };
        vk.CreateImageView(device, &viewInfo, null, out var view);

        vk.GetPhysicalDeviceProperties(physicalDevice, out var props);
        SamplerCreateInfo samplerInfo = new()
        {
            SType = StructureType.SamplerCreateInfo,
            MagFilter = Filter.Linear,
            MinFilter = Filter.Linear,
            AddressModeU = SamplerAddressMode.Repeat,
            AddressModeV = SamplerAddressMode.Repeat,
            AddressModeW = SamplerAddressMode.Repeat,
            AnisotropyEnable = true,
            MaxAnisotropy = props.Limits.MaxSamplerAnisotropy,
            BorderColor = BorderColor.FloatOpaqueBlack,
            UnnormalizedCoordinates = false,
            CompareEnable = false,
            CompareOp = CompareOp.Always,
            MipmapMode = SamplerMipmapMode.Linear,
            MinLod = 0.0f,
            MaxLod = Vk.LodClampNone,
            MipLodBias = 0.0f,
        };
        vk.CreateSampler(device, &samplerInfo, null, out var sampler);

        var resource = new ImageResource(vk, device, "FontTexture", format, extent2D, usage, image, imageAlloc, view);
        return new Texture(vk, device, resource, sampler);
    } 

    public void Dispose()
    {
        if (_disposed) return;
        if (Sampler.Handle != 0) _vk.DestroySampler(_device, Sampler, null);
        Resource?.Dispose();
        _disposed = true;
    }
}