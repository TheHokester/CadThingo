using System.Numerics;
using System.Runtime.InteropServices;
using CadThingo.VulkanEngine.ImGui;
using Silk.NET.Core.Native;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Pipelines;

//
//  Ray-tracing-PIPELINE path tracer (VK_KHR_ray_tracing_pipeline).
//
//  Additive, opt-in sibling of PTComputePipeline. Shares the SAME acceleration
//  structure, scene buffers, IBL, bindless materials, PathFrameUBO layout, and
//  accumulator/outColor storage images as the compute path tracer — the only
//  differences are the pipeline object (RT groups + SBT) and the dispatch
//  (CmdTraceRays). Descriptor set layout / bindings mirror PTComputePipeline so
//  the renderer can drive both with the same Write* calls.
//
//  Phase 2: pipeline + SBT + dispatch wired against the minimal PathTraceRT.spml
//  (raygen/closesthit/miss/anyhit). The bounce loop / NEE / accumulation move
//  into the shader in phase 3; the descriptor layout already carries every
//  binding they will need.
//
//  Inherits RtPipeline, which owns the VK_KHR_ray_tracing_pipeline dispatch
//  table + SBT-layout properties (loaded in its constructor from the device).
public sealed unsafe class RTPipeline : RtPipeline
{
    // Matches PathTraceRT.slang / PTComputePipeline PathFrameUBO byte-for-byte.
    [StructLayout(LayoutKind.Sequential)]
    private struct PathFrameUBO
    {
        public Matrix4x4 invView;
        public Matrix4x4 invProj;
        public Vector4   camPos;
        public uint      frameIndex;
        public uint      bounceCap;
        public uint      lightCount;
        public uint      resetAccum;
        public Vector2   screenSize;
        public float     fov;
        public float     tanHalfFov;
        public float     prefilteredCubeMipLevels;
        public float     scaleIBLAmbient;
        public float     focusDistance;
        public float     aperture;
        public float     paniniDistance;
        public float     verticalCompression;
        public uint      emissiveTriCount;
        public float     totalEmissivePower;
    }

    private const int SetFrame    = 0;
    private const int SetGeom     = 1;
    private const int SetBindless = 2;
    private const int SetIbl      = 3;

    // Every RT stage; descriptor bindings declare the union (a binding may name
    // more stages than actually read it — simpler and valid).
    private const ShaderStageFlags RtAll =
        ShaderStageFlags.RaygenBitKhr | ShaderStageFlags.MissBitKhr |
        ShaderStageFlags.ClosestHitBitKhr | ShaderStageFlags.AnyHitBitKhr;

    // SER variant (HitObject + ReorderThread raygen) when the device exposes
    // VK_NV_ray_tracing_invocation_reorder; otherwise the plain TraceRay variant.
    // Both .spv expose the same entry points, so the pipeline build is identical.
    protected override string ShaderPath =>
        Renderer.SerSupported
            ? @"C:\Users\jamie\RiderProjects\CadThingo\CadThingo\Assets\Shaders\PathTraceRT_SER.spv"
            : @"C:\Users\jamie\RiderProjects\CadThingo\CadThingo\Assets\Shaders\PathTraceRT.spv";

    // SBT: one buffer holding [raygen][miss][hit] regions, each padded to the
    // device's shaderGroupBaseAlignment; CmdTraceRays reads the strided regions.
    private Buffer   _sbtBuffer;
    private SubAlloc _sbtAlloc;
    private StridedDeviceAddressRegionKHR _raygenRegion;
    private StridedDeviceAddressRegionKHR _missRegion;
    private StridedDeviceAddressRegionKHR _hitRegion;
    private StridedDeviceAddressRegionKHR _callableRegion;   // unused (no callables)

    private UboBuffer[] _frameUbos = new UboBuffer[Renderer.MAX_CONCURRENT_FRAMES];

    // Per-frame runtime state (mirrors the compute path).
    public uint  BounceCap     { get; set; } = 8;
    public float FocusDistance { get; set; } = 5.0f;
    public float Aperture      { get; set; } = 0.0f;

    private uint _accumSamples;
    private bool _accumDirty = true;

    public RTPipeline(Renderer renderer) : base(renderer) { }

    public void MarkAccumulatorDirty() => _accumDirty = true;
    public uint CurrentSampleCount => _accumSamples;


    // Descriptor set layouts — identical bindings to PTComputePipeline, RT stages.
    protected override void CreateDescriptorSetLayouts()
    {
        DescriptorSetLayouts            = new DescriptorSetLayout[4];
        OwnedDescriptorSetLayoutIndices = new[] { SetFrame, SetGeom, SetIbl };

        var set0 = stackalloc DescriptorSetLayoutBinding[8];
        set0[0] = new() { Binding = 0, DescriptorType = DescriptorType.UniformBuffer,            DescriptorCount = 1, StageFlags = RtAll };
        set0[1] = new() { Binding = 1, DescriptorType = DescriptorType.StorageBuffer,            DescriptorCount = 1, StageFlags = RtAll };
        set0[2] = new() { Binding = 2, DescriptorType = DescriptorType.AccelerationStructureKhr, DescriptorCount = 1, StageFlags = RtAll };
        set0[3] = new() { Binding = 3, DescriptorType = DescriptorType.StorageBuffer,            DescriptorCount = 1, StageFlags = RtAll };
        set0[4] = new() { Binding = 4, DescriptorType = DescriptorType.StorageImage,             DescriptorCount = 1, StageFlags = RtAll };
        set0[5] = new() { Binding = 5, DescriptorType = DescriptorType.StorageImage,             DescriptorCount = 1, StageFlags = RtAll };
        set0[6] = new() { Binding = 6, DescriptorType = DescriptorType.StorageBuffer,            DescriptorCount = 1, StageFlags = RtAll };
        set0[7] = new() { Binding = 7, DescriptorType = DescriptorType.StorageBuffer,            DescriptorCount = 1, StageFlags = RtAll };
        CreateLayout(set0, 8, out DescriptorSetLayouts[SetFrame]);

        var set1 = stackalloc DescriptorSetLayoutBinding[2];
        set1[0] = new() { Binding = 1, DescriptorType = DescriptorType.StorageBuffer, DescriptorCount = 1, StageFlags = RtAll };
        set1[1] = new() { Binding = 2, DescriptorType = DescriptorType.StorageBuffer, DescriptorCount = 1, StageFlags = RtAll };
        CreateLayout(set1, 2, out DescriptorSetLayouts[SetGeom]);

        // Borrowed bindless layout (ResourceManager adds RT stage flags when
        // RayTracePipelineSupported — see CreateBindlessDescriptorSetLayout).
        DescriptorSetLayouts[SetBindless] = Engine.ResourceManager.GetBindlessLayout();

        var set3 = stackalloc DescriptorSetLayoutBinding[4];
        for (uint b = 0; b < 4; b++)
            set3[b] = new() { Binding = b, DescriptorType = DescriptorType.CombinedImageSampler, DescriptorCount = 1, StageFlags = RtAll };
        CreateLayout(set3, 4, out DescriptorSetLayouts[SetIbl]);
    }

    private void CreateLayout(DescriptorSetLayoutBinding* bindings, uint count, out DescriptorSetLayout layout)
    {
        DescriptorSetLayoutCreateInfo info = new()
        {
            SType        = StructureType.DescriptorSetLayoutCreateInfo,
            BindingCount = count,
            PBindings    = bindings,
        };
        if (Vk.CreateDescriptorSetLayout(Device, &info, null, out layout) != Result.Success)
            throw new Exception("Failed to create RT pipeline descriptor set layout");
    }


    // Pipeline build: one RT pipeline (raygen + miss + closest/any-hit group) + SBT.
    protected override void CreatePipeline()
    {
        if (KhrRtPipeline == null)
            throw new Exception("RTPipeline: VK_KHR_ray_tracing_pipeline dispatch table not loaded");

        byte[] code   = File.ReadAllBytes(ShaderPath);
        var    module = Renderer.CreateShaderModule(code);

        // Entry-point names preserved by slangc for the multi-entry-point module
        // (confirmed in phase 1: rayGenMain / missMain / closestHitMain / anyHitMain).
        var rgName = SilkMarshal.StringToPtr("rayGenMain");
        var msName = SilkMarshal.StringToPtr("missMain");
        var chName = SilkMarshal.StringToPtr("closestHitMain");
        var ahName = SilkMarshal.StringToPtr("anyHitMain");

        var stages = stackalloc PipelineShaderStageCreateInfo[4];
        stages[0] = Stage(ShaderStageFlags.RaygenBitKhr,     module, rgName);
        stages[1] = Stage(ShaderStageFlags.MissBitKhr,       module, msName);
        stages[2] = Stage(ShaderStageFlags.ClosestHitBitKhr, module, chName);
        stages[3] = Stage(ShaderStageFlags.AnyHitBitKhr,     module, ahName);

        const uint UNUSED = uint.MaxValue;   // VK_SHADER_UNUSED_KHR
        var groups = stackalloc RayTracingShaderGroupCreateInfoKHR[3];
        // Group 0: raygen (general, stage 0)
        groups[0] = new() { SType = StructureType.RayTracingShaderGroupCreateInfoKhr, Type = RayTracingShaderGroupTypeKHR.GeneralKhr,
                            GeneralShader = 0, ClosestHitShader = UNUSED, AnyHitShader = UNUSED, IntersectionShader = UNUSED };
        // Group 1: miss (general, stage 1)
        groups[1] = new() { SType = StructureType.RayTracingShaderGroupCreateInfoKhr, Type = RayTracingShaderGroupTypeKHR.GeneralKhr,
                            GeneralShader = 1, ClosestHitShader = UNUSED, AnyHitShader = UNUSED, IntersectionShader = UNUSED };
        // Group 2: hit group (closest=stage 2, any=stage 3)
        groups[2] = new() { SType = StructureType.RayTracingShaderGroupCreateInfoKhr, Type = RayTracingShaderGroupTypeKHR.TrianglesHitGroupKhr,
                            GeneralShader = UNUSED, ClosestHitShader = 2, AnyHitShader = 3, IntersectionShader = UNUSED };

        var info = new RayTracingPipelineCreateInfoKHR
        {
            SType                        = StructureType.RayTracingPipelineCreateInfoKhr,
            StageCount                   = 4,
            PStages                      = stages,
            GroupCount                   = 3,
            PGroups                      = groups,
            // Iterative bounce loop in raygen + inline ray query for shadows keeps
            // this at 1; raising it is a hard perf cliff.
            MaxPipelineRayRecursionDepth = 1,
            Layout                       = PipelineLayoutHandle,
        };

        Pipeline pipeline;
        var res = KhrRtPipeline.CreateRayTracingPipelines(
            Device, default, PipelineCacheHandle, 1, &info, null, &pipeline);
        if (res != Result.Success)
            throw new Exception($"Failed to create ray tracing pipeline: {res}");
        PipelineHandle = pipeline;

        SilkMarshal.Free(rgName);
        SilkMarshal.Free(msName);
        SilkMarshal.Free(chName);
        SilkMarshal.Free(ahName);
        Vk.DestroyShaderModule(Device, module, null);

        BuildShaderBindingTable();
    }

    private PipelineShaderStageCreateInfo Stage(ShaderStageFlags stage, ShaderModule module, nint entry) => new()
    {
        SType  = StructureType.PipelineShaderStageCreateInfo,
        Stage  = stage,
        Module = module,
        PName  = (byte*)entry,
    };

    private static uint AlignUp(uint v, uint a) => (v + a - 1) & ~(a - 1);

    private ulong BufferDeviceAddress(Buffer buf)
    {
        var info = new BufferDeviceAddressInfo { SType = StructureType.BufferDeviceAddressInfo, Buffer = buf };
        return Vk.GetBufferDeviceAddress(Device, &info);
    }

    // Packs the 3 group handles into [raygen][miss][hit] regions per the SBT
    // alignment rules (each region base-aligned, handles handle-aligned; raygen
    // size must equal its stride).
    private void BuildShaderBindingTable()
    {
        uint handleSize    = ShaderGroupHandleSize;
        uint handleAligned = AlignUp(handleSize, ShaderGroupHandleAlignment);
        uint baseAlign     = ShaderGroupBaseAlignment;

        const uint raygenCount = 1, missCount = 1, hitCount = 1;
        const uint groupCount  = raygenCount + missCount + hitCount;   // 3

        _raygenRegion.Stride = AlignUp(handleAligned, baseAlign);
        _raygenRegion.Size   = _raygenRegion.Stride;                   // raygen: size == stride
        _missRegion.Stride   = handleAligned;
        _missRegion.Size     = AlignUp(missCount * handleAligned, baseAlign);
        _hitRegion.Stride    = handleAligned;
        _hitRegion.Size      = AlignUp(hitCount * handleAligned, baseAlign);

        ulong sbtSize = _raygenRegion.Size + _missRegion.Size + _hitRegion.Size;

        Renderer.CreateBuffer(sbtSize,
            BufferUsageFlags.ShaderBindingTableBitKhr | BufferUsageFlags.ShaderDeviceAddressBit | BufferUsageFlags.TransferSrcBit,
            MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit,
            out _sbtBuffer, out _sbtAlloc);

        byte* sbt     = (byte*)Renderer.memAllocator.GetMapped(_sbtAlloc);
        var   handles = new byte[groupCount * handleSize];
        fixed (byte* pHandles = handles)
        {
            if (KhrRtPipeline!.GetRayTracingShaderGroupHandles(
                    Device, PipelineHandle, 0, groupCount, (nuint)handles.Length, pHandles) != Result.Success)
                throw new Exception("Failed to fetch RT shader group handles");

            // raygen → 0, miss → after raygen region, hit → after miss region.
            System.Buffer.MemoryCopy(pHandles + 0u * handleSize, sbt,                                                       handleSize, handleSize);
            System.Buffer.MemoryCopy(pHandles + 1u * handleSize, sbt + _raygenRegion.Size,                                  handleSize, handleSize);
            System.Buffer.MemoryCopy(pHandles + 2u * handleSize, sbt + _raygenRegion.Size + _missRegion.Size,              handleSize, handleSize);
        }

        ulong baseAddr = BufferDeviceAddress(_sbtBuffer);
        _raygenRegion.DeviceAddress = baseAddr;
        _missRegion.DeviceAddress   = baseAddr + _raygenRegion.Size;
        _hitRegion.DeviceAddress    = baseAddr + _raygenRegion.Size + _missRegion.Size;
        _callableRegion             = default;
    }


    // Per-pipeline-owned resources (per-frame UBO).
    protected override void CreateResources()
    {
        for (int i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
            Renderer.CreateMappedUniformBuffer(sizeof(PathFrameUBO), ref _frameUbos[i]);
    }


    protected override void CreateDescriptorSets()
    {
        DescriptorSets = new DescriptorSet[4][];

        var set0Layouts = stackalloc DescriptorSetLayout[(int)Renderer.MAX_CONCURRENT_FRAMES];
        for (int i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++) set0Layouts[i] = DescriptorSetLayouts[SetFrame];
        DescriptorSetAllocateInfo alloc0 = new()
        {
            SType              = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool     = Renderer.descriptorPool,
            DescriptorSetCount = Renderer.MAX_CONCURRENT_FRAMES,
            PSetLayouts        = set0Layouts,
        };
        DescriptorSets[SetFrame] = new DescriptorSet[Renderer.MAX_CONCURRENT_FRAMES];
        fixed (DescriptorSet* p = DescriptorSets[SetFrame])
            if (Vk.AllocateDescriptorSets(Device, &alloc0, p) != Result.Success)
                throw new Exception("Failed to allocate RT pipeline set 0");

        var geomLayout = DescriptorSetLayouts[SetGeom];
        DescriptorSetAllocateInfo alloc1 = new()
        {
            SType              = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool     = Renderer.descriptorPool,
            DescriptorSetCount = 1,
            PSetLayouts        = &geomLayout,
        };
        DescriptorSets[SetGeom] = new DescriptorSet[1];
        fixed (DescriptorSet* p = DescriptorSets[SetGeom])
            if (Vk.AllocateDescriptorSets(Device, &alloc1, p) != Result.Success)
                throw new Exception("Failed to allocate RT pipeline set 1");

        DescriptorSets[SetBindless] = null;   // borrowed

        var iblLayout = DescriptorSetLayouts[SetIbl];
        DescriptorSetAllocateInfo alloc3 = new()
        {
            SType              = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool     = Renderer.descriptorPool,
            DescriptorSetCount = 1,
            PSetLayouts        = &iblLayout,
        };
        DescriptorSets[SetIbl] = new DescriptorSet[1];
        fixed (DescriptorSet* p = DescriptorSets[SetIbl])
            if (Vk.AllocateDescriptorSets(Device, &alloc3, p) != Result.Success)
                throw new Exception("Failed to allocate RT pipeline set 3");
    }


    protected override void WriteDescriptors()
    {
        WriteFrameUboDescriptors();
        WriteGeometryDescriptors();
    }

    private void WriteFrameUboDescriptors()
    {
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            DescriptorBufferInfo info = new() { Buffer = _frameUbos[i].buffer, Offset = 0, Range = (ulong)sizeof(PathFrameUBO) };
            var write = new WriteDescriptorSet
            {
                SType = StructureType.WriteDescriptorSet, DstSet = DescriptorSets[SetFrame][i],
                DstBinding = 0, DescriptorType = DescriptorType.UniformBuffer, DescriptorCount = 1, PBufferInfo = &info,
            };
            Vk.UpdateDescriptorSets(Device, 1, &write, 0, null);
        }
    }

    public void WriteGeometryDescriptors()
    {
        var rm = Engine.ResourceManager;
        DescriptorBufferInfo vbInfo = new() { Buffer = rm.GlobalVertexBuffer, Offset = 0, Range = Vk.WholeSize };
        DescriptorBufferInfo ibInfo = new() { Buffer = rm.GlobalIndexBuffer,  Offset = 0, Range = Vk.WholeSize };
        var writes = stackalloc WriteDescriptorSet[2];
        writes[0] = new() { SType = StructureType.WriteDescriptorSet, DstSet = DescriptorSets[SetGeom][0], DstBinding = 1, DescriptorType = DescriptorType.StorageBuffer, DescriptorCount = 1, PBufferInfo = &vbInfo };
        writes[1] = new() { SType = StructureType.WriteDescriptorSet, DstSet = DescriptorSets[SetGeom][0], DstBinding = 2, DescriptorType = DescriptorType.StorageBuffer, DescriptorCount = 1, PBufferInfo = &ibInfo };
        Vk.UpdateDescriptorSets(Device, 2, writes, 0, null);
    }

    public void WriteLightsDescriptor()
    {
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            DescriptorBufferInfo info = new() { Buffer = Renderer.GetLightStorageBuffer((uint)i), Offset = 0, Range = (ulong)(Renderer.MAX_LIGHTS * (uint)sizeof(PbrLightGpu)) };
            var write = new WriteDescriptorSet { SType = StructureType.WriteDescriptorSet, DstSet = DescriptorSets[SetFrame][i], DstBinding = 1, DescriptorType = DescriptorType.StorageBuffer, DescriptorCount = 1, PBufferInfo = &info };
            Vk.UpdateDescriptorSets(Device, 1, &write, 0, null);
        }
    }

    public void WriteTlasDescriptor(AccelerationStructureKHR tlas)
    {
        if (tlas.Handle == 0) return;
        var tlasH = tlas;
        var asWrite = new WriteDescriptorSetAccelerationStructureKHR
        {
            SType = StructureType.WriteDescriptorSetAccelerationStructureKhr,
            AccelerationStructureCount = 1, PAccelerationStructures = &tlasH,
        };
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            var write = new WriteDescriptorSet { SType = StructureType.WriteDescriptorSet, PNext = &asWrite, DstSet = DescriptorSets[SetFrame][i], DstBinding = 2, DescriptorType = DescriptorType.AccelerationStructureKhr, DescriptorCount = 1 };
            Vk.UpdateDescriptorSets(Device, 1, &write, 0, null);
        }
    }

    public void WriteShadowInfoDescriptor()
    {
        var buf = Renderer.ShadowInfoBuffer;
        if (buf.Handle == 0) return;
        DescriptorBufferInfo info = new() { Buffer = buf, Offset = 0, Range = Renderer.ShadowInfoBufferSize };
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            var write = new WriteDescriptorSet { SType = StructureType.WriteDescriptorSet, DstSet = DescriptorSets[SetFrame][i], DstBinding = 3, DescriptorType = DescriptorType.StorageBuffer, DescriptorCount = 1, PBufferInfo = &info };
            Vk.UpdateDescriptorSets(Device, 1, &write, 0, null);
        }
    }

    public void WriteEmissiveDescriptors()
    {
        var triBuf   = Renderer.EmissiveTriBuffer;
        var aliasBuf = Renderer.EmissiveAliasBuffer;
        if (triBuf.Handle == 0 || aliasBuf.Handle == 0) return;
        DescriptorBufferInfo triInfo   = new() { Buffer = triBuf,   Offset = 0, Range = Renderer.EmissiveTriBufferSize };
        DescriptorBufferInfo aliasInfo = new() { Buffer = aliasBuf, Offset = 0, Range = Renderer.EmissiveAliasBufferSize };
        var writes = stackalloc WriteDescriptorSet[2];
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            writes[0] = new() { SType = StructureType.WriteDescriptorSet, DstSet = DescriptorSets[SetFrame][i], DstBinding = 6, DescriptorType = DescriptorType.StorageBuffer, DescriptorCount = 1, PBufferInfo = &triInfo };
            writes[1] = new() { SType = StructureType.WriteDescriptorSet, DstSet = DescriptorSets[SetFrame][i], DstBinding = 7, DescriptorType = DescriptorType.StorageBuffer, DescriptorCount = 1, PBufferInfo = &aliasInfo };
            Vk.UpdateDescriptorSets(Device, 2, writes, 0, null);
        }
    }

    public void WriteStorageImageDescriptors(ImageView accumView, ImageView outColorView)
    {
        DescriptorImageInfo accumInfo = new() { ImageView = accumView,    ImageLayout = ImageLayout.General };
        DescriptorImageInfo outInfo   = new() { ImageView = outColorView, ImageLayout = ImageLayout.General };
        var writes = stackalloc WriteDescriptorSet[2];
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            writes[0] = new() { SType = StructureType.WriteDescriptorSet, DstSet = DescriptorSets[SetFrame][i], DstBinding = 4, DescriptorType = DescriptorType.StorageImage, DescriptorCount = 1, PImageInfo = &accumInfo };
            writes[1] = new() { SType = StructureType.WriteDescriptorSet, DstSet = DescriptorSets[SetFrame][i], DstBinding = 5, DescriptorType = DescriptorType.StorageImage, DescriptorCount = 1, PImageInfo = &outInfo };
            Vk.UpdateDescriptorSets(Device, 2, writes, 0, null);
        }
        MarkAccumulatorDirty();
    }

    public void WriteIblDescriptors()
    {
        var imageInfos = stackalloc DescriptorImageInfo[4]
        {
            new() { ImageView = Renderer.irradianceCubeView,  Sampler = Renderer.iblCubeSampler, ImageLayout = ImageLayout.ShaderReadOnlyOptimal },
            new() { ImageView = Renderer.prefilteredCubeView, Sampler = Renderer.iblCubeSampler, ImageLayout = ImageLayout.ShaderReadOnlyOptimal },
            new() { ImageView = Renderer.brdfLutView,         Sampler = Renderer.iblLutSampler,  ImageLayout = ImageLayout.ShaderReadOnlyOptimal },
            new() { ImageView = Renderer.envCubeView,         Sampler = Renderer.iblCubeSampler, ImageLayout = ImageLayout.ShaderReadOnlyOptimal },
        };
        var writes = stackalloc WriteDescriptorSet[4];
        for (uint b = 0; b < 4; b++)
            writes[b] = new WriteDescriptorSet { SType = StructureType.WriteDescriptorSet, DstSet = DescriptorSets[SetIbl][0], DstBinding = b, DescriptorType = DescriptorType.CombinedImageSampler, DescriptorCount = 1, PImageInfo = &imageInfos[b] };
        Vk.UpdateDescriptorSets(Device, 4, writes, 0, null);
    }


    public bool UpdatePerFrame(uint frameIndex, Camera camera, uint lightCount, Extent2D renderExtent)
    {
        bool reset = _accumDirty;
        if (reset) { _accumSamples = 0; _accumDirty = false; } else { _accumSamples++; }

        Matrix4x4 view = camera != null ? camera.GetViewMatrix() : Matrix4x4.Identity;
        Matrix4x4 proj = camera != null
            ? camera.GetProjectionMatrix((float)renderExtent.Width / renderExtent.Height, 0.1f, 100.0f)
            : Matrix4x4.Identity;
        proj.M22 *= -1;
        Matrix4x4.Invert(view, out var invView);
        Matrix4x4.Invert(proj, out var invProj);

        float fovDeg     = camera != null ? camera.Fov : 60.0f;
        float fovRad     = fovDeg * (float)(Math.PI / 180.0);
        float tanHalfFov = MathF.Tan(fovRad * 0.5f);

        PathFrameUBO ubo = new()
        {
            invView                  = invView,
            invProj                  = invProj,
            camPos                   = camera != null ? new Vector4(camera.GetPosition(), 1.0f) : new Vector4(2, 2, 2, 1),
            frameIndex               = _accumSamples,
            bounceCap                = BounceCap,
            lightCount               = lightCount,
            resetAccum               = reset ? 1u : 0u,
            screenSize               = new Vector2(renderExtent.Width, renderExtent.Height),
            fov                      = fovRad,
            tanHalfFov               = tanHalfFov,
            prefilteredCubeMipLevels = Renderer.prefilteredCubeMipLevels,
            scaleIBLAmbient          = EditorState.IblIntensity,
            focusDistance            = FocusDistance,
            aperture                 = Aperture,
            paniniDistance           = 1.0f,
            verticalCompression      = 0.0f,
            emissiveTriCount         = Renderer.EmissiveTriangleCount,
            totalEmissivePower       = Renderer.TotalEmissivePower,
        };

        void* data = _frameUbos[frameIndex].mapped;
        new Span<PathFrameUBO>(data, 1).Fill(ubo);
        return reset;
    }


    /// <summary>Records the CmdTraceRays dispatch. Caller must have run
    /// UpdatePerFrame, written all external descriptors, and transitioned the
    /// accumulator/outColor images to GENERAL beforehand.</summary>
    public void Record(CommandBuffer cmd, in Renderer.FrameContext ctx)
    {
        Vk.CmdBindPipeline(cmd, PipelineBindPoint.RayTracingKhr, PipelineHandle);

        var sets = stackalloc DescriptorSet[4]
        {
            DescriptorSets[SetFrame][ctx.FrameIndex],
            DescriptorSets[SetGeom][0],
            Engine.ResourceManager.GetBindlessSet(ctx.FrameIndex),
            DescriptorSets[SetIbl][0],
        };
        Vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.RayTracingKhr, Layout, 0, 4, sets, 0, null);

        var rg = _raygenRegion; var ms = _missRegion; var ht = _hitRegion; var cl = _callableRegion;
        KhrRtPipeline!.CmdTraceRays(cmd, &rg, &ms, &ht, &cl,
            ctx.RenderExtent.Width, ctx.RenderExtent.Height, 1);
    }


    public override void Dispose()
    {
        if (_sbtBuffer.Handle != 0) Renderer.DestroyBuffer(_sbtBuffer, _sbtAlloc);
        foreach (var b in _frameUbos) Renderer.DestroyBuffer(b.buffer, b.alloc);
        base.Dispose();
    }
}