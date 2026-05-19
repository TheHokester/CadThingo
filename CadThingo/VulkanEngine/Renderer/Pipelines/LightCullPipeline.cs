using System.Numerics;
using System.Runtime.InteropServices;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Pipelines;


public sealed unsafe class LightCullPipeline : ComputePipeline
{
    // 64B invViewProj + 16 camPos + 8 screenSize + 8 tileCounts + 4 lightCount
    // + 12 pad = 112B, well under the 128B Vulkan minimum.
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

    protected override string ShaderPath { get; } =
        @"C:\Users\jamie\RiderProjects\CadThingo\CadThingo\Assets\Shaders\LightCulling.spv";

    // Tile-cull per-frame outputs owned by this pipeline. TileLightCount[tileIdx]
    // is the number of lights overlapping each tile; TileLightIndices[tileIdx*MAX + slot]
    // is the flat index into the lights SSBO. The lighting pass reads both, keyed by
    // tileIdx = (gl_FragCoord / TILE_SIZE).
    private UboBuffer[] TileLightCountBuffers   = new UboBuffer[Renderer.MAX_CONCURRENT_FRAMES];
    private UboBuffer[] TileLightIndicesBuffers = new UboBuffer[Renderer.MAX_CONCURRENT_FRAMES];

    public Buffer GetTileLightCountBuffer  (uint frame) => TileLightCountBuffers[frame].buffer;
    public Buffer GetTileLightIndicesBuffer(uint frame) => TileLightIndicesBuffers[frame].buffer;

    public LightCullPipeline(Renderer renderer) : base(renderer)
    {
        PushConstantRanges = new[]
        {
            new PushConstantRange
            {
                StageFlags = ShaderStageFlags.ComputeBit,
                Offset     = 0,
                Size       = (uint)sizeof(LightCullPushConstants),
            }
        };
    }

    public override void Dispose()
    {
        foreach (var b in TileLightCountBuffers)   Renderer.DestroyBuffer(b.buffer, b.alloc);
        foreach (var b in TileLightIndicesBuffers) Renderer.DestroyBuffer(b.buffer, b.alloc);
        base.Dispose();
    }

    protected override void CreateDescriptorSetLayouts()
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
        if (Vk.CreateDescriptorSetLayout(Device, &info, null, out var layout) != Result.Success)
            throw new Exception("Failed to create light-cull descriptor set layout");
        DescriptorSetLayouts = new[] { layout };
        OwnedDescriptorSetLayoutIndices = new[] { 0 };
    }

    protected override void CreateResources()
    {
        // Tile-cull buffers sized for worst-case tile count (MAX_TILE_COUNT).
        // Per frame: TileLightCount = MAX × 4B, TileLightIndices = MAX × MAX_LIGHTS_PER_TILE × 4B.
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            Renderer.CreateMappedStorageBuffer(
                (ulong)(Renderer.MAX_TILE_COUNT * sizeof(uint)),
                ref TileLightCountBuffers[i]);
            Renderer.CreateMappedStorageBuffer(
                (ulong)(Renderer.MAX_TILE_COUNT * Renderer.MAX_LIGHTS_PER_TILE * sizeof(uint)),
                ref TileLightIndicesBuffers[i]);
        }
    }

    protected override void CreateDescriptorSets()
    {
        var layouts = stackalloc DescriptorSetLayout[(int)Renderer.MAX_CONCURRENT_FRAMES];
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++) layouts[i] = DescriptorSetLayouts[0];

        DescriptorSetAllocateInfo alloc = new()
        {
            SType              = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool     = Renderer.descriptorPool,
            DescriptorSetCount = Renderer.MAX_CONCURRENT_FRAMES,
            PSetLayouts        = layouts,
        };
        DescriptorSets = new DescriptorSet[1][];
        DescriptorSets[0] = new DescriptorSet[Renderer.MAX_CONCURRENT_FRAMES];
        fixed (DescriptorSet* pSets = DescriptorSets[0])
        {
            if (Vk.AllocateDescriptorSets(Device, &alloc, pSets) != Result.Success)
                throw new Exception("Failed to allocate light-cull descriptor sets");
        }
    }

    /// <summary>
    /// Rewrites just binding 0 (the lights SSBO that PbrDeferredPipeline owns).
    /// Called by RebuildPbrPipelines — when PBR is disposed + recreated the old
    /// VkBuffer this descriptor pointed at is gone, so the consumer side has to
    /// re-bind to the fresh handle.
    /// </summary>
    public void RewriteLightsBinding()
    {
        var pbr = Renderer.PbrDeferredPipeline
                  ?? throw new InvalidOperationException("PbrDeferredPipeline not initialized.");

        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            DescriptorBufferInfo bufLights = new()
            {
                Buffer = pbr.GetLightStorageBuffer((uint)i), Offset = 0,
                Range  = (ulong)(Renderer.MAX_LIGHTS * (uint)sizeof(PbrLightGpu)),
            };
            var write = new WriteDescriptorSet
            {
                SType           = StructureType.WriteDescriptorSet,
                DstSet          = DescriptorSets[0][i],
                DstBinding      = 0,
                DescriptorType  = DescriptorType.StorageBuffer,
                DescriptorCount = 1,
                PBufferInfo     = &bufLights,
            };
            Vk.UpdateDescriptorSets(Device, 1, &write, 0, null);
        }
    }

    protected override void WriteDescriptors()
    {
        // PbrDeferredPipeline owns the lights SSBO; this binding is the consumer side.
        var pbr = Renderer.PbrDeferredPipeline
                  ?? throw new InvalidOperationException(
                      "LightCullPipeline must be initialized AFTER PbrDeferredPipeline so the lights SSBO is available.");

        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            DescriptorBufferInfo bufLights = new()
            {
                Buffer = pbr.GetLightStorageBuffer((uint)i), Offset = 0,
                Range  = (ulong)(Renderer.MAX_LIGHTS * (uint)sizeof(PbrLightGpu)),
            };
            DescriptorBufferInfo bufTileCount = new()
            {
                Buffer = TileLightCountBuffers[i].buffer, Offset = 0,
                Range  = (ulong)(Renderer.MAX_TILE_COUNT * sizeof(uint)),
            };
            DescriptorBufferInfo bufTileIdx = new()
            {
                Buffer = TileLightIndicesBuffers[i].buffer, Offset = 0,
                Range  = (ulong)(Renderer.MAX_TILE_COUNT * Renderer.MAX_LIGHTS_PER_TILE * sizeof(uint)),
            };

            var writes = stackalloc WriteDescriptorSet[3];
            for (uint b = 0; b < 3; b++)
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
            writes[0].PBufferInfo = &bufLights;
            writes[1].PBufferInfo = &bufTileCount;
            writes[2].PBufferInfo = &bufTileIdx;

            Vk.UpdateDescriptorSets(Device, 3, writes, 0, null);
        }
    }

    // CPU side. Computes invViewProj + tile counts from the current
    // camera/swapchain extent, pushes them, dispatches one group per tile, and
    // barriers compute-write -> fragment-read on the two tile buffers.
    public void Record(CommandBuffer cmd, uint frameIndex, Camera cam,
                       uint lightCount, uint tileCountX, uint tileCountY)
    {
        if (tileCountX == 0 || tileCountY == 0) return;

        Vk.CmdBindPipeline(cmd, PipelineBindPoint.Compute, PipelineHandle);
        var dset = DescriptorSets[0][frameIndex];
        Vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute,
            PipelineLayoutHandle, 0, 1, &dset, 0, null);

        Matrix4x4 view = cam.GetViewMatrix();
        Matrix4x4 proj = cam.GetProjectionMatrix(
            (float)Renderer.renderExtent.Width / Renderer.renderExtent.Height, 0.1f, 100.0f);
        // The lighting fragment shader sees a Y-flipped projection (the geometry
        // pipeline flips proj.M22 in UpdateUbo). Build invViewProj from the SAME
        // flipped matrix so the cull frustum lines up with where pixels actually
        // sample world positions from the g-buffer.
        proj.M22 *= -1f;
        Matrix4x4 vp = view * proj;
        if (!Matrix4x4.Invert(vp, out Matrix4x4 invVP))
            invVP = Matrix4x4.Identity;

        var push = new LightCullPushConstants
        {
            InvViewProj = invVP,
            CamPos      = new Vector4(cam.GetPosition(), 1f),
            ScreenSize  = new Vector2(Renderer.renderExtent.Width, Renderer.renderExtent.Height),
            TileCountX  = tileCountX,
            TileCountY  = tileCountY,
            LightCount  = lightCount,
        };
        Vk.CmdPushConstants(cmd, PipelineLayoutHandle, ShaderStageFlags.ComputeBit,
            0, (uint)sizeof(LightCullPushConstants), &push);

        // One thread group per tile (each group is 16×16 = 256 threads).
        Vk.CmdDispatch(cmd, tileCountX, tileCountY, 1);

        // Barrier: compute writes -> fragment shader reads of the tile buffers.
        var postBarriers = stackalloc BufferMemoryBarrier[2];
        postBarriers[0] = new BufferMemoryBarrier
        {
            SType         = StructureType.BufferMemoryBarrier,
            SrcAccessMask = AccessFlags.ShaderWriteBit,
            DstAccessMask = AccessFlags.ShaderReadBit,
            SrcQueueFamilyIndex = Silk.NET.Vulkan.Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Silk.NET.Vulkan.Vk.QueueFamilyIgnored,
            Buffer = TileLightCountBuffers[frameIndex].buffer,
            Offset = 0,
            Size   = Silk.NET.Vulkan.Vk.WholeSize,
        };
        postBarriers[1] = new BufferMemoryBarrier
        {
            SType         = StructureType.BufferMemoryBarrier,
            SrcAccessMask = AccessFlags.ShaderWriteBit,
            DstAccessMask = AccessFlags.ShaderReadBit,
            SrcQueueFamilyIndex = Silk.NET.Vulkan.Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Silk.NET.Vulkan.Vk.QueueFamilyIgnored,
            Buffer = TileLightIndicesBuffers[frameIndex].buffer,
            Offset = 0,
            Size   = Silk.NET.Vulkan.Vk.WholeSize,
        };
        Vk.CmdPipelineBarrier(cmd,
            PipelineStageFlags.ComputeShaderBit,
            PipelineStageFlags.FragmentShaderBit,
            0, 0, null, 2, postBarriers, 0, null);
    }
}

// ────────────────────────────────────────────────────────────────────────────
//  PBR deferred lighting pass — fullscreen triangle, samples G-buffer +
//  per-tile light list, optional ray-queried shadows.
// ────────────────────────────────────────────────────────────────────────────

// ────────────────────────────────────────────────────────────────────────────
//  Tone-map / post pass — samples HDRColor, writes FinalColor (LDR)
// ────────────────────────────────────────────────────────────────────────────

// Matches the TONEMAP_OPERATOR spec constant in Tonemap.slang. Read once at
// pipeline build; changing the operator requires Dispose + rebuild.

// ────────────────────────────────────────────────────────────────────────────
//  Transparent forward+ pass — renders BLEND-mode materials into HDRColor with
//  src-alpha / one-minus-src-alpha blending, depth-tested LE against the
//  geometry pass's depth buffer (no depth write).
// ────────────────────────────────────────────────────────────────────────────