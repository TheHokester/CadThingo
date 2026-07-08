using System.IO;
using System.Numerics;
using System.Runtime.InteropServices;
using CadThingo.VulkanEngine.Renderer.Pipelines;
using CadThingo.VulkanEngine.Renderer.Features.PathTracer;
using Silk.NET.Core.Native;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Features.ReSTIR;

//
//  ReSTIR direct-illumination pipeline (VK_KHR_ray_tracing_pipeline + a compute sibling).
//
//  Subclass of RTPipeline reusing its SBT / descriptor-set layout (sets 0-3) / per-frame UBO /
//  CmdTraceRays dispatch, swapping the shader module for the forked ReStirDI tracer. On top it owns
//  SET 4 (the ReSTIR working set) + a push constant, AND a second VkPipeline -- the SpatialShade
//  COMPUTE shader -- built on the SAME pipeline layout (set 0 carries ComputeBit, so the exact
//  descriptor sets the RT Trace pass uses are bindable from compute too).
//
//  Two-pass deferred ReSTIR DI:
//    * Trace (RT): build + temporally reuse the reservoir, write cur reservoir + fat G-buffer
//      (PackedHit) + sceneRadiance (everything EXCEPT analytic direct).
//    * SpatialShade (compute): spatially reuse same-frame neighbours, shade the analytic sample
//      (opaque shadow ray), add to sceneRadiance, fold into the accumulator.
//
internal sealed unsafe class ReStirDIPipeline : RTPipeline
{
    // Appended after the base RTPipeline sets (scene 0, IO 1, IBL 2).
    private const int SetReStir = 3;

    // Per-pixel byte strides -- MUST match the Slang StructuredBuffer element STRIDE, which is
    // std430, NOT the tight field-sum: a float3 has 16-byte alignment and rounds the struct up to a
    // multiple of 16. Strides confirmed via `slangc -reflection-json` (samplePos lands at offset 16).
    //   Reservoir { uint type,index; float3 pos@16; float3 le@32; float esPdf; uint M; float wSum,W,pHat } = 64B
    //   PackedHit { float3 worldPos; uint x7 }  -> 40B of data, rounded up to 16 (the float3 align) = 48B
    //   sceneRadiance : float4 (no float3 -> no rounding)                                                 = 16B
    // Mis-sizing these undersizes the buffer: the shader strides past the end on the bottom rows, which
    // read back zero (robust access) -> the G-buffer reports "no hit" and the deferred shade is skipped.
    private const ulong ReservoirBytes = 64;
    private const ulong PackedHitBytes = 48;
    private const ulong SceneRadBytes  = 16;

    // Push constant uploaded each frame; mirrors `struct ReStirPush` in the shaders (80B).
    [StructLayout(LayoutKind.Sequential)]
    private struct ReStirPush
    {
        public Matrix4x4 PrevViewProj;   // 64
        public uint      FrameParity;    //  4 — monotonic frame & 1
        public uint      HasHistory;     //  4 — 0 on the first frame after (re)alloc
        public uint      Pad0, Pad1;     //  8
    }

    private Buffer   _resA, _resB, _gbufA, _gbufB, _sceneRad;
    private SubAlloc _resAAlloc, _resBAlloc, _gbufAAlloc, _gbufBAlloc, _sceneRadAlloc;
    private uint     _pixelCount;

    private ReStirPush _push;
    private Matrix4x4  _prevViewProj = Matrix4x4.Identity;
    private uint       _frameCounter;

    private Pipeline _buildTemporalPipeline;   // BuildTemporal compute (reservoir build + temporal)
    private Pipeline _spatialPipeline;         // SpatialShade compute (spatial reuse + shade)

    public ReStirDIPipeline(Renderer renderer) : base(renderer) { }

    // Public handles for the graph module to import (barrier ordering across passes + frames).
    public Buffer ReservoirA    => _resA;
    public Buffer ReservoirB    => _resB;
    public Buffer GBufferA      => _gbufA;
    public Buffer GBufferB      => _gbufB;
    public Buffer SceneRadiance => _sceneRad;

    protected override string ShaderPath =>
        ShaderPaths.Kernel("ReSTIR", Gfx.SerSupported ? "ReStirDI_SER" : "ReStirDI");

    // The IO set (accumulator/outColor) is shared with the SpatialShade compute pipeline (it folds
    // the analytic direct into the accumulator) -> needs ComputeBit. The scene set is already
    // All-stages via the registry, so it needs no override.
    protected override ShaderStageFlags OwnedSetExtraStages => ShaderStageFlags.ComputeBit;


    // ---- Set 3 layout + push-constant range, appended onto the base RTPipeline sets (scene/IO/IBL) -
    protected override void CreateDescriptorSetLayouts()
    {
        base.CreateDescriptorSetLayouts();   // sets 0-2 (IO set now Raygen|...|Compute)

        const ShaderStageFlags s4 = ShaderStageFlags.RaygenBitKhr | ShaderStageFlags.ComputeBit;
        System.Array.Resize(ref DescriptorSetLayouts, SetReStir + 1);
        var bindings = stackalloc DescriptorSetLayoutBinding[5];
        for (uint b = 0; b < 5; b++)
            bindings[b] = new() { Binding = b, DescriptorType = DescriptorType.StorageBuffer, DescriptorCount = 1, StageFlags = s4 };
        var info = new DescriptorSetLayoutCreateInfo
        {
            SType = StructureType.DescriptorSetLayoutCreateInfo, BindingCount = 5, PBindings = bindings,
        };
        if (Vk.CreateDescriptorSetLayout(Device, &info, null, out DescriptorSetLayouts[SetReStir]) != Result.Success)
            throw new Exception("ReStirDIPipeline: failed to create set 4 layout");

        var owned = OwnedDescriptorSetLayoutIndices;
        System.Array.Resize(ref owned, owned.Length + 1);
        owned[^1] = SetReStir;
        OwnedDescriptorSetLayoutIndices = owned;

        // Push-constant range read by the base CreatePipelineLayout (runs after this). Visible to
        // both the raygen Trace shader and the SpatialShade compute shader (shared layout).
        PushConstantRanges = new[]
        {
            new PushConstantRange
            {
                StageFlags = ShaderStageFlags.RaygenBitKhr | ShaderStageFlags.ComputeBit,
                Offset = 0, Size = (uint)sizeof(ReStirPush),
            },
        };
    }

    // Build the RT pipeline + SBT (base), then the two ReSTIR compute pipelines (BuildTemporal +
    // SpatialShade) on the SAME shared pipeline layout (set 0/1 carry ComputeBit, set 2 bindless is
    // compute-capable). Deferred order at record time: Trace (RT) -> BuildTemporal -> SpatialShade.
    protected override void CreatePipeline()
    {
        base.CreatePipeline();
        _buildTemporalPipeline = CreateComputePipeline("BuildTemporal");
        _spatialPipeline       = CreateComputePipeline("SpatialShade");
    }

    private Pipeline CreateComputePipeline(string kernel)
    {
        byte[] code   = File.ReadAllBytes(ShaderPaths.Kernel("ReSTIR", kernel));
        var    module = Gfx.CreateShaderModule(code);
        var    entry  = SilkMarshal.StringToPtr("main");
        var stage = new PipelineShaderStageCreateInfo
        {
            SType = StructureType.PipelineShaderStageCreateInfo,
            Stage = ShaderStageFlags.ComputeBit, Module = module, PName = (byte*)entry,
        };
        var info = new ComputePipelineCreateInfo
        {
            SType = StructureType.ComputePipelineCreateInfo, Stage = stage, Layout = PipelineLayoutHandle,
        };
        if (Vk.CreateComputePipelines(Device, PipelineCacheHandle, 1, &info, null, out Pipeline pipeline) != Result.Success)
            throw new Exception($"ReStirDIPipeline: failed to create {kernel} compute pipeline");
        SilkMarshal.Free(entry);
        Vk.DestroyShaderModule(Device, module, null);
        return pipeline;
    }

    protected override void CreateResources()
    {
        base.CreateResources();   // per-frame UBO
        AllocBuffers(Renderer.RenderExtent);
    }

    protected override void CreateDescriptorSets()
    {
        base.CreateDescriptorSets();   // sets 1 (IO) + 2 (IBL); set 0 (scene) is registry-owned

        System.Array.Resize(ref DescriptorSets, SetReStir + 1);
        var layout = DescriptorSetLayouts[SetReStir];
        var alloc = new DescriptorSetAllocateInfo
        {
            SType = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool = Gfx.DescriptorPool, DescriptorSetCount = 1, PSetLayouts = &layout,
        };
        DescriptorSets[SetReStir] = new DescriptorSet[1];
        fixed (DescriptorSet* p = DescriptorSets[SetReStir])
            if (Vk.AllocateDescriptorSets(Device, &alloc, p) != Result.Success)
                throw new Exception("ReStirDIPipeline: failed to allocate set 4");
    }

    protected override void WriteDescriptors()
    {
        base.WriteDescriptors();
        WriteReStirDescriptors();
    }


    // ---- ReSTIR working-set buffers (extent-sized, device-local, ping-ponged) -------------------
    private void AllocBuffers(Extent2D extent)
    {
        _pixelCount = extent.Width * extent.Height;
        ulong n = _pixelCount;
        Gfx.CreateBuffer(n * ReservoirBytes, BufferUsageFlags.StorageBufferBit, MemoryPropertyFlags.DeviceLocalBit, out _resA,     out _resAAlloc);
        Gfx.CreateBuffer(n * ReservoirBytes, BufferUsageFlags.StorageBufferBit, MemoryPropertyFlags.DeviceLocalBit, out _resB,     out _resBAlloc);
        Gfx.CreateBuffer(n * PackedHitBytes, BufferUsageFlags.StorageBufferBit, MemoryPropertyFlags.DeviceLocalBit, out _gbufA,    out _gbufAAlloc);
        Gfx.CreateBuffer(n * PackedHitBytes, BufferUsageFlags.StorageBufferBit, MemoryPropertyFlags.DeviceLocalBit, out _gbufB,    out _gbufBAlloc);
        Gfx.CreateBuffer(n * SceneRadBytes,  BufferUsageFlags.StorageBufferBit, MemoryPropertyFlags.DeviceLocalBit, out _sceneRad, out _sceneRadAlloc);
        _frameCounter = 0;   // fresh buffers hold garbage -> first frame must not reuse history
    }

    private void FreeBuffers()
    {
        if (_resA.Handle     != 0) Gfx.DestroyBuffer(_resA,     _resAAlloc);
        if (_resB.Handle     != 0) Gfx.DestroyBuffer(_resB,     _resBAlloc);
        if (_gbufA.Handle    != 0) Gfx.DestroyBuffer(_gbufA,    _gbufAAlloc);
        if (_gbufB.Handle    != 0) Gfx.DestroyBuffer(_gbufB,    _gbufBAlloc);
        if (_sceneRad.Handle != 0) Gfx.DestroyBuffer(_sceneRad, _sceneRadAlloc);
        _resA = default; _resB = default; _gbufA = default; _gbufB = default; _sceneRad = default;
    }

    private void WriteReStirDescriptors()
    {
        var set = DescriptorSets[SetReStir][0];
        Span<Buffer> bufs = stackalloc Buffer[5] { _resA, _resB, _gbufA, _gbufB, _sceneRad };
        var infos  = stackalloc DescriptorBufferInfo[5];
        var writes = stackalloc WriteDescriptorSet[5];
        for (uint b = 0; b < 5; b++)
        {
            infos[b] = new DescriptorBufferInfo { Buffer = bufs[(int)b], Offset = 0, Range = Vk.WholeSize };
            writes[b] = new WriteDescriptorSet
            {
                SType = StructureType.WriteDescriptorSet, DstSet = set, DstBinding = b,
                DescriptorType = DescriptorType.StorageBuffer, DescriptorCount = 1, PBufferInfo = &infos[b],
            };
        }
        Vk.UpdateDescriptorSets(Device, 5, writes, 0, null);
    }

    /// <summary>Resize path: reallocate the per-pixel working set to the new extent + rewrite set 4.
    /// Resets the history gate. Called by the core's Resize before the graph rebuild.</summary>
    public void ReallocReStirBuffers(Extent2D extent)
    {
        FreeBuffers();
        AllocBuffers(extent);
        WriteReStirDescriptors();
    }


    // ---- Per-frame state: temporal reprojection matrix + ping-pong parity -----------------------
    public override bool UpdatePerFrame(uint frameIndex, Camera camera, uint lightCount, Extent2D renderExtent)
    {
        bool reset = base.UpdatePerFrame(frameIndex, camera, lightCount, renderExtent);

        Matrix4x4 view = camera != null ? camera.GetViewMatrix() : Matrix4x4.Identity;
        Matrix4x4 proj = camera != null
            ? camera.GetProjectionMatrix((float)renderExtent.Width / renderExtent.Height, 0.1f, 100.0f)
            : Matrix4x4.Identity;
        proj.M22 *= -1f;
        Matrix4x4 viewProj = view * proj;

        _push = new ReStirPush
        {
            PrevViewProj = _prevViewProj,
            FrameParity  = _frameCounter & 1u,
            HasHistory   = _frameCounter > 0u ? 1u : 0u,
        };

        _prevViewProj = viewProj;
        _frameCounter++;
        return reset;
    }

    protected override void RecordPushConstants(CommandBuffer cmd)
    {
        var p = _push;
        Vk.CmdPushConstants(cmd, Layout, ShaderStageFlags.RaygenBitKhr, 0, (uint)sizeof(ReStirPush), &p);
    }

    /// <summary>Records the BuildTemporal compute pass: reconstruct the surface from the G-buffer,
    /// build the unified DI reservoir (RIS) + temporal reuse, write the cur reservoir. Runs after
    /// Trace (G-buffer RAW), before SpatialShade (reservoir RAW).</summary>
    public void RecordBuildTemporal(CommandBuffer cmd, in Renderer.FrameContext ctx)
        => RecordComputePass(cmd, ctx, _buildTemporalPipeline);

    /// <summary>Records the SpatialShade compute pass: spatial reuse + deferred shade + accumulation.</summary>
    public void RecordSpatialShade(CommandBuffer cmd, in Renderer.FrameContext ctx)
        => RecordComputePass(cmd, ctx, _spatialPipeline);

    // Binds the SAME descriptor sets (0-3) the Trace pass uses on the shared layout, pushes the
    // per-frame ReSTIR state, and dispatches a full-screen 8x8 grid. Both ReSTIR compute passes share
    // this; which buffers each actually touches is decided by its shader's declared bindings. The
    // scene set carries the frame-UBO arena slot, so its bind supplies the same dynamic offset the RT
    // Trace pass uses.
    private void RecordComputePass(CommandBuffer cmd, in Renderer.FrameContext ctx, Pipeline pipeline)
    {
        Vk.CmdBindPipeline(cmd, PipelineBindPoint.Compute, pipeline);

        uint frameConstants = PushFrameConstants(ctx.FrameIndex);
        var sets = stackalloc DescriptorSet[4];
        sets[0] = Renderer.descriptorRegistry.SceneSet(ctx.FrameIndex);  // frame UBO + lights / tlas / entityInfo / vb+ib / emissive / bindless
        sets[1] = DescriptorSets[SetIO][0];                             // accumulator / outColor
        sets[2] = DescriptorSets[SetIbl][0];                            // ibl (unused by compute)
        sets[3] = DescriptorSets[SetReStir][0];                         // reservoirs / gbuffers / sceneRadiance
        Vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute, Layout, 0, 4, sets, 1, &frameConstants);

        var p = _push;
        Vk.CmdPushConstants(cmd, Layout, ShaderStageFlags.ComputeBit, 0, (uint)sizeof(ReStirPush), &p);

        Vk.CmdDispatch(cmd, (ctx.RenderExtent.Width + 7) / 8, (ctx.RenderExtent.Height + 7) / 8, 1);
    }


    // ---- Bind set 3 after the base sets (RTPipeline.Record, RT path) -----------------------------
    protected override uint ExtraSetCount => 1u;
    protected override void WriteExtraSets(DescriptorSet* dst, uint frame)
    {
        dst[0] = DescriptorSets[SetReStir][0];
    }

    protected override void ReleaseGpuResources()
    {
        if (_buildTemporalPipeline.Handle != 0) { Vk.DestroyPipeline(Device, _buildTemporalPipeline, null); _buildTemporalPipeline = default; }
        if (_spatialPipeline.Handle != 0) { Vk.DestroyPipeline(Device, _spatialPipeline, null); _spatialPipeline = default; }
        FreeBuffers();
        base.ReleaseGpuResources();
    }
}