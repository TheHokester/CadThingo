using System.Numerics;
using System.Runtime.InteropServices;
using CadThingo.VulkanEngine.Renderer.Pipelines;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Features.Forward;


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

    protected override string ShaderPath { get; } = ShaderPaths.Kernel("Forward", "LightCulling");

    // Set 0 - unified scene set (sceneLights is the only read; this pass keeps
    //         its params in push constants so the (0,0) slot goes unused and the
    //         bind supplies a zero dynamic offset).
    // Set 1 - pass-local outputs owned by this pipeline. TileLightCount[tileIdx]
    //         is the number of lights overlapping each tile;
    //         TileLightIndices[tileIdx*MAX + slot] is the flat index into the
    //         lights SSBO. The lighting passes read both, keyed by
    //         tileIdx = (gl_FragCoord / TILE_SIZE).
    private const int SetScene   = 0;
    private const int SetTileOut = 1;

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
        foreach (var b in TileLightCountBuffers)   Gfx.DestroyBuffer(b.buffer, b.alloc);
        foreach (var b in TileLightIndicesBuffers) Gfx.DestroyBuffer(b.buffer, b.alloc);
        base.Dispose();
    }

    protected override void CreateDescriptorSetLayouts()
    {
        // Set 0 is borrowed from DescriptorRegistry (never destroyed here);
        // set 1 (the two tile output SSBOs) is owned by this pipeline.
        DescriptorSetLayouts = new DescriptorSetLayout[2];
        OwnedDescriptorSetLayoutIndices = new[] { SetTileOut };
        DescriptorSetLayouts[SetScene] = Renderer.descriptorRegistry.SceneSetLayout;

        var bindings = stackalloc DescriptorSetLayoutBinding[2];
        for (uint b = 0; b < 2; b++)
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
            BindingCount = 2,
            PBindings    = bindings,
        };
        if (Vk.CreateDescriptorSetLayout(Device, &info, null, out DescriptorSetLayouts[SetTileOut]) != Result.Success)
            throw new Exception("Failed to create light-cull tile-output descriptor set layout");
    }

    protected override void CreateResources()
    {
        // Tile-cull buffers sized for worst-case tile count (MAX_TILE_COUNT).
        // Per frame: TileLightCount = MAX × 4B, TileLightIndices = MAX × MAX_LIGHTS_PER_TILE × 4B.
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            Gfx.CreateMappedStorageBuffer(
                (ulong)(Renderer.MAX_TILE_COUNT * sizeof(uint)),
                ref TileLightCountBuffers[i]);
            Gfx.CreateMappedStorageBuffer(
                (ulong)(Renderer.MAX_TILE_COUNT * Renderer.MAX_LIGHTS_PER_TILE * sizeof(uint)),
                ref TileLightIndicesBuffers[i]);
        }
    }

    protected override void CreateDescriptorSets()
    {
        DescriptorSets = new DescriptorSet[2][];

        // Set 0 — scene set is owned by DescriptorRegistry; Record binds
        // Renderer.descriptorRegistry.SceneSet(frame) directly.
        DescriptorSets[SetScene] = null;

        var layouts = stackalloc DescriptorSetLayout[(int)Renderer.MAX_CONCURRENT_FRAMES];
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++) layouts[i] = DescriptorSetLayouts[SetTileOut];

        DescriptorSetAllocateInfo alloc = new()
        {
            SType              = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool     = Gfx.DescriptorPool,
            DescriptorSetCount = Renderer.MAX_CONCURRENT_FRAMES,
            PSetLayouts        = layouts,
        };
        DescriptorSets[SetTileOut] = new DescriptorSet[Renderer.MAX_CONCURRENT_FRAMES];
        fixed (DescriptorSet* pSets = DescriptorSets[SetTileOut])
        {
            if (Vk.AllocateDescriptorSets(Device, &alloc, pSets) != Result.Success)
                throw new Exception("Failed to allocate light-cull descriptor sets");
        }
    }

    protected override void WriteDescriptors()
    {
        // Only the owned tile outputs; sceneLights is registry-maintained.
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
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

            var writes = stackalloc WriteDescriptorSet[2];
            for (uint b = 0; b < 2; b++)
            {
                writes[b] = new WriteDescriptorSet
                {
                    SType           = StructureType.WriteDescriptorSet,
                    DstSet          = DescriptorSets[SetTileOut][i],
                    DstBinding      = b,
                    DescriptorType  = DescriptorType.StorageBuffer,
                    DescriptorCount = 1,
                };
            }
            writes[0].PBufferInfo = &bufTileCount;
            writes[1].PBufferInfo = &bufTileIdx;

            Vk.UpdateDescriptorSets(Device, 2, writes, 0, null);
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

        // The scene-set layout carries the (0,0) dynamic constant slot even
        // though this shader doesn't declare it (params ride push constants),
        // so the bind must still supply one dynamic offset - zero is valid.
        uint zeroOffset = 0;
        var sets = stackalloc DescriptorSet[2]
        {
            Renderer.descriptorRegistry.SceneSet(frameIndex),
            DescriptorSets[SetTileOut][frameIndex],
        };
        Vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute,
            PipelineLayoutHandle, 0, 2, sets, 1, &zeroOffset);

        Matrix4x4 view = cam.GetViewMatrix();
        Matrix4x4 proj = cam.GetProjectionMatrix(
            (float)Renderer.renderExtent.Width / Renderer.renderExtent.Height, 0.1f, 100.0f);
        // The lighting fragment shader sees a Y-flipped projection (the geometry
        // pipeline flips proj.M22 in its frame constants). Build invViewProj from
        // the SAME flipped matrix so the cull frustum lines up with where pixels
        // actually sample world positions from the g-buffer.
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


    }
}