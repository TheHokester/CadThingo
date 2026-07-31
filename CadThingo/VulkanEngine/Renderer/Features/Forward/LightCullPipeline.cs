using System.Numerics;
using System.Runtime.InteropServices;
using CadThingo.VulkanEngine.Renderer.Descriptors;
using CadThingo.VulkanEngine.Renderer.FrameGraph;
using CadThingo.VulkanEngine.Renderer.Pipelines;
using CadThingo.VulkanEngine.Renderer.Slang;
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

    protected override ShaderCompileRequest? Program =>
        new("Forward/LightCulling", ["Main"], [], []);

    // Set 0 - unified scene set (sceneLights is the only read; this pass keeps
    //         its params in push constants so the (0,0) slot goes unused and the
    //         bind supplies a zero dynamic offset).
    // Set 1 - pass-local outputs owned by this pipeline. TileLightCount[tileIdx]
    //         is the number of lights overlapping each tile;
    //         TileLightIndices[tileIdx*MAX + slot] is the flat index into the
    //         lights SSBO. The lighting passes read both, keyed by
    //         tileIdx = (gl_FragCoord / TILE_SIZE).
    private UboBuffer[] TileLightCountBuffers   = new UboBuffer[RenderConfig.MAX_CONCURRENT_FRAMES];
    private UboBuffer[] TileLightIndicesBuffers = new UboBuffer[RenderConfig.MAX_CONCURRENT_FRAMES];

    public Buffer GetTileLightCountBuffer  (uint frame) => TileLightCountBuffers[frame].buffer;
    public Buffer GetTileLightIndicesBuffer(uint frame) => TileLightIndicesBuffers[frame].buffer;

    // Graph-baked pass set: the two tile-cull outputs, filled by the deferred FrameGraph (they
    // are graph imports this pass writes). Both the layout and the names the graph matches
    // against come from LightCulling.slang's set-1 declarations, so the LightCullPass Write
    // binds name the shader globals; the pipeline owns only the layout.
    public PassSetSpec PassSet =>
        new(ShaderSets.Pass, DescriptorSetLayouts[ShaderSets.Pass], ReflectedBindings(ShaderSets.Pass));

    // Push-constant range is reflected in Initialize; CreateResources asserts the C# mirror
    // still matches the reflected size.
    public LightCullPipeline(GpuContext gpu, Renderer renderer) : base(gpu, renderer) { }

    public override void Dispose()
    {
        foreach (var b in TileLightCountBuffers)   Gfx.DestroyBuffer(b.buffer, b.alloc);
        foreach (var b in TileLightIndicesBuffers) Gfx.DestroyBuffer(b.buffer, b.alloc);
        base.Dispose();
    }

    protected override void CreateDescriptorSetLayouts()
    {
        // Assemble [scene(0), pass(1)]. Scene is registry-owned; the pipeline owns only the pass
        // layout, built from LightCulling.slang's set-1 declarations.
        var passLayout = CreateReflectedSetLayout(ShaderSets.Pass);
        DescriptorSetLayouts = Registry.BuildPipelineSetLayouts(passLayout);
        OwnedDescriptorSetLayoutIndices = new[] { (int)ShaderSets.Pass };
    }

    protected override void CreateResources()
    {
        // Reflection cannot check this on its own: the C# mirror of CullParams has to keep
        // matching the shader.
        uint reflected = PushConstantRanges[0].Size;
        if (reflected != (uint)sizeof(LightCullPushConstants))
            throw new Exception(
                $"LightCullPushConstants is {sizeof(LightCullPushConstants)} bytes but " +
                $"LightCulling.slang reflects {reflected}");

        // Tile-cull buffers sized for worst-case tile count (MAX_TILE_COUNT).
        // Per frame: TileLightCount = MAX × 4B, TileLightIndices = MAX × MAX_LIGHTS_PER_TILE × 4B.
        for (var i = 0; i < RenderConfig.MAX_CONCURRENT_FRAMES; i++)
        {
            Gfx.CreateMappedStorageBuffer(
                (ulong)(RenderConfig.MAX_TILE_COUNT * sizeof(uint)),
                ref TileLightCountBuffers[i]);
            Gfx.CreateMappedStorageBuffer(
                (ulong)(RenderConfig.MAX_TILE_COUNT * RenderConfig.MAX_LIGHTS_PER_TILE * sizeof(uint)),
                ref TileLightIndicesBuffers[i]);
        }
    }

    // Descriptor sets are graph-owned now: the deferred FrameGraph allocates set 1 from this
    // pipeline's pass-set layout and writes the two tile buffers by name each Compile. The
    // scene set (sceneLights) comes from the registry. No CreateDescriptorSets / WriteDescriptors.

    // CPU side. Computes invViewProj + tile counts from the current
    // camera/swapchain extent, pushes them, and dispatches one group per tile.
    // (The compute-write -> fragment-read barrier on the tile buffers is derived
    // by the graph from the LightCullPass Write + the lighting-pass Read.)
    public void Record(CommandBuffer cmd, uint frameIndex, Camera cam,
                       uint lightCount, uint tileCountX, uint tileCountY, DescriptorSet tileSet)
    {
        if (tileCountX == 0 || tileCountY == 0) return;

        Vk.CmdBindPipeline(cmd, PipelineBindPoint.Compute, PipelineHandle);

        // The scene-set layout carries the (0,0) dynamic constant slot even
        // though this shader doesn't declare it (params ride push constants),
        // so the bind must still supply one dynamic offset - zero is valid.
        // Set 1 is the graph-baked tile-output pass set.
        uint zeroOffset = 0;
        var sets = stackalloc DescriptorSet[2]
        {
            Registry.SceneSet(frameIndex),
            tileSet,
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
        // Stage mask comes from the reflected range the layout was built from: vkCmdPushConstants
        // requires the two to agree, so neither side names a stage mask of its own.
        Vk.CmdPushConstants(cmd, PipelineLayoutHandle, PushConstantRanges[0].StageFlags,
            0, (uint)sizeof(LightCullPushConstants), &push);

        // One thread group per tile (each group is 16×16 = 256 threads).
        Vk.CmdDispatch(cmd, tileCountX, tileCountY, 1);


    }
}