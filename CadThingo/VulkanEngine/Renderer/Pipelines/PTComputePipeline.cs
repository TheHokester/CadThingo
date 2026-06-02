using System.Numerics;
using System.Runtime.InteropServices;
using CadThingo.VulkanEngine.ImGui;
using Silk.NET.Core.Native;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Pipelines;

//
//  Path-tracer compute pipeline (ray query + progressive accumulation).
//
//  Owns:    PathFrameUBO per frame; 4 PSOs (one per camera mode) baked via
//           spec constant id=3; sets 0/1/3.
//  Borrows: TLAS, lights SSBO (PbrDeferred), ShadowEntityInfo (Renderer),
//           global VB/IB (ResourceManager), IBL cubes + BRDF LUT (Renderer),
//           bindless materials/textures (ResourceManager set 2).
//  Externally supplied: accumulator + outColor storage image views — the
//           wiring lives in the renderer (storage image creation + GENERAL
//           layout transitions before/after the dispatch).
//
//  Inherits from PipelineBase (not ComputePipeline) because the base's
//  CreatePipeline is sealed and has no spec-constant hook — we need full
//  control to build one PSO per camera mode at init.
public sealed unsafe class PTComputePipeline : PipelineBase
{
    // Matches PTCompute.slang::PathFrameUBO byte-for-byte.
    [StructLayout(LayoutKind.Sequential)]
    private struct PathFrameUBO
    {
        public Matrix4x4 invView;            // 64B
        public Matrix4x4 invProj;            // 64B
        public Vector4   camPos;             // 16B
        public uint      frameIndex;         //  4B - sample index for accumulator + RNG seed
        public uint      bounceCap;          //  4B - clamped below MAX_BOUNCES spec const
        public uint      lightCount;
        public uint      resetAccum;         //  4B - 1 = overwrite accumulator this dispatch
        public Vector2   screenSize;         //  8B
        public float     fov;                //  4B - radians, used by Panini + Fisheye
        public float     tanHalfFov;         //  4B - used by Pinhole + ThinLens
        public float     prefilteredCubeMipLevels;
        public float     scaleIBLAmbient;
        public float     focusDistance;
        public float     aperture;
        public float     paniniDistance;
        public float     verticalCompression;
        public uint      emissiveTriCount;    // emissive area-light triangles in the alias table
        public float     totalEmissivePower;  // Σ area·luminance(Le) — alias-table normaliser
    }

    public enum CameraMode : uint
    {
        Pinhole      = 0,
        ThinLens     = 1,
        Panini       = 2,
        Fisheye      = 3,
        Orthographic = 4,   // not yet implemented in PTUtils.slang — falls back to Pinhole PSO
    }

    public override PipelineBindPoint BindPoint => PipelineBindPoint.Compute;

    private const int SetFrame    = 0;
    private const int SetGeom     = 1;
    private const int SetBindless = 2;
    private const int SetIbl      = 3;

    private const string ShaderPath =
        @"C:\Users\jamie\RiderProjects\CadThingo\CadThingo\Assets\Shaders\PTCompute.spv";

    // Multi-PSO pattern: one pipeline per camera mode, mode baked via spec
    // constant id=3. Mode switch at Record time = `CmdBindPipeline` only, no
    // rebuild. Indices match CameraMode values 0..3 (Orthographic re-uses 0).
    private Pipeline[] _modePipelines = new Pipeline[4];

    //  Compile-time tunables (baked into spec constants at Initialize) 
    // Change after Initialize requires Dispose + rebuild.
    public bool EnableNee         { get; init; } = true;
    public bool RussianRoulette   { get; init; } = true;
    public uint MaxBouncesHardCap { get; init; } = 8;

    //  Per-frame runtime state (uploaded via PathFrameUBO each frame) 
    public CameraMode Mode                { get; set; } = CameraMode.Pinhole;
    public uint       BounceCap           { get; set; } = 8;
    public float      Aperture            { get; set; } = 0.0f;
    public float      FocusDistance       { get; set; } = 5.0f;
    public float      PaniniDistance      { get; set; } = 1.0f;
    public float      VerticalCompression { get; set; } = 0.0f;
    public float      IblIntensity        { get; set; } = 1.0f;

    private UboBuffer[] _frameUbos = new UboBuffer[Renderer.MAX_CONCURRENT_FRAMES];

    // Progressive-accumulation bookkeeping. Renderer flips _accumDirty on
    // camera move / scene edit / extent change via MarkAccumulatorDirty();
    // UpdatePerFrame consumes it and emits resetAccum=1 on the next dispatch.
    private uint _accumSamples;
    private bool _accumDirty = true;

    public PTComputePipeline(Renderer renderer) : base(renderer) { }

    /// <summary>Force the next dispatch to overwrite (not add to) the
    /// accumulator. Call from input / scene-edit handlers.</summary>
    public void MarkAccumulatorDirty() => _accumDirty = true;

    public uint CurrentSampleCount => _accumSamples;


    // Descriptor set layouts
    protected override void CreateDescriptorSetLayouts()
    {
        DescriptorSetLayouts            = new DescriptorSetLayout[4];
        OwnedDescriptorSetLayoutIndices = new[] { SetFrame, SetGeom, SetIbl };

        // Set 0: UBO + lights + TLAS + shadow info + accumulator + outColor
        //        + emissive triangles + emissive alias table.
        var set0 = stackalloc DescriptorSetLayoutBinding[8];
        set0[0] = new() { Binding = 0, DescriptorType = DescriptorType.UniformBuffer,            DescriptorCount = 1, StageFlags = ShaderStageFlags.ComputeBit };
        set0[1] = new() { Binding = 1, DescriptorType = DescriptorType.StorageBuffer,            DescriptorCount = 1, StageFlags = ShaderStageFlags.ComputeBit };
        set0[2] = new() { Binding = 2, DescriptorType = DescriptorType.AccelerationStructureKhr, DescriptorCount = 1, StageFlags = ShaderStageFlags.ComputeBit };
        set0[3] = new() { Binding = 3, DescriptorType = DescriptorType.StorageBuffer,            DescriptorCount = 1, StageFlags = ShaderStageFlags.ComputeBit };
        set0[4] = new() { Binding = 4, DescriptorType = DescriptorType.StorageImage,             DescriptorCount = 1, StageFlags = ShaderStageFlags.ComputeBit };
        set0[5] = new() { Binding = 5, DescriptorType = DescriptorType.StorageImage,             DescriptorCount = 1, StageFlags = ShaderStageFlags.ComputeBit };
        set0[6] = new() { Binding = 6, DescriptorType = DescriptorType.StorageBuffer,            DescriptorCount = 1, StageFlags = ShaderStageFlags.ComputeBit };
        set0[7] = new() { Binding = 7, DescriptorType = DescriptorType.StorageBuffer,            DescriptorCount = 1, StageFlags = ShaderStageFlags.ComputeBit };
        CreateLayout(set0, 8, out DescriptorSetLayouts[SetFrame]);

        // Set 1: globalVertices + globalIndices. Shader uses bindings 1 and 2
        // (binding 0 is intentionally unused — match the shader exactly).
        var set1 = stackalloc DescriptorSetLayoutBinding[2];
        set1[0] = new() { Binding = 1, DescriptorType = DescriptorType.StorageBuffer, DescriptorCount = 1, StageFlags = ShaderStageFlags.ComputeBit };
        set1[1] = new() { Binding = 2, DescriptorType = DescriptorType.StorageBuffer, DescriptorCount = 1, StageFlags = ShaderStageFlags.ComputeBit };
        CreateLayout(set1, 2, out DescriptorSetLayouts[SetGeom]);

        // Set 2: borrowed bindless layout. NOTE: if ResourceManager's layout
        // doesn't declare ComputeBit on its bindings, validation will warn
        // when we bind this set in a compute pipeline — add ComputeBit to
        // ResourceManager.MaterialBindlessLayout if you see that.
        DescriptorSetLayouts[SetBindless] = Engine.ResourceManager.GetBindlessLayout();

        // Set 3: IBL cubes + BRDF LUT + full-res envCube.
        var set3 = stackalloc DescriptorSetLayoutBinding[4];
        for (uint b = 0; b < 4; b++)
            set3[b] = new() { Binding = b, DescriptorType = DescriptorType.CombinedImageSampler, DescriptorCount = 1, StageFlags = ShaderStageFlags.ComputeBit };
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
            throw new Exception("Failed to create pathtracer descriptor set layout");
    }


    // Pipeline build: one PSO per camera mode
    protected override void CreatePipeline()
    {
        byte[] code     = File.ReadAllBytes(ShaderPath);
        var    module   = Gfx.CreateShaderModule(code);
        var    entryPtr = SilkMarshal.StringToPtr("main");

        // 4 spec entries packed into a 16B blob:
        //   id 0 = ENABLE_NEE        (bool → uint)
        //   id 1 = RUSSIAN_ROULETTE  (bool → uint)
        //   id 2 = MAX_BOUNCES       (uint)
        //   id 3 = MODE              (uint) — patched per pipeline below
        var specEntries = stackalloc SpecializationMapEntry[4];
        specEntries[0] = new() { ConstantID = 0, Offset = 0,  Size = sizeof(uint) };
        specEntries[1] = new() { ConstantID = 1, Offset = 4,  Size = sizeof(uint) };
        specEntries[2] = new() { ConstantID = 2, Offset = 8,  Size = sizeof(uint) };
        specEntries[3] = new() { ConstantID = 3, Offset = 12, Size = sizeof(uint) };

        var specData = stackalloc uint[4];
        specData[0] = EnableNee       ? 1u : 0u;
        specData[1] = RussianRoulette ? 1u : 0u;
        specData[2] = MaxBouncesHardCap;
        // specData[3] patched per-mode in the loop.

        var specInfo = new SpecializationInfo
        {
            MapEntryCount = 4,
            PMapEntries   = specEntries,
            DataSize      = (UIntPtr)(4u * sizeof(uint)),
            PData         = specData,
        };

        for (uint mode = 0; mode < 4; mode++)
        {
            specData[3] = mode;

            var stage = new PipelineShaderStageCreateInfo
            {
                SType               = StructureType.PipelineShaderStageCreateInfo,
                Stage               = ShaderStageFlags.ComputeBit,
                Module              = module,
                PName               = (byte*)entryPtr,
                PSpecializationInfo = &specInfo,
            };

            var info = new ComputePipelineCreateInfo
            {
                SType  = StructureType.ComputePipelineCreateInfo,
                Stage  = stage,
                Layout = PipelineLayoutHandle,
            };

            if (Vk.CreateComputePipelines(Device, PipelineCacheHandle, 1, &info, null, out _modePipelines[mode]) != Result.Success)
                throw new Exception($"Failed to create pathtracer compute pipeline (mode {mode})");
        }

        // PipelineHandle is the "default" the base tracks for disposal — point
        // it at mode 0 so base.Dispose() destroys exactly one PSO via that path
        // and we destroy modes 1-3 ourselves.
        PipelineHandle = _modePipelines[0];

        SilkMarshal.Free(entryPtr);
        Vk.DestroyShaderModule(Device, module, null);
    }


    // Per-pipeline-owned resources (just the per-frame UBO)
    protected override void CreateResources()
    {
        for (int i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
            Gfx.CreateMappedUniformBuffer(sizeof(PathFrameUBO), ref _frameUbos[i]);
    }


    // Descriptor set allocation
    protected override void CreateDescriptorSets()
    {
        DescriptorSets = new DescriptorSet[4][];

        // Set 0 — per-frame in flight (UBO double-buffered; storage images and
        // borrowed buffers are shared but written into the same set slot).
        var set0Layouts = stackalloc DescriptorSetLayout[(int)Renderer.MAX_CONCURRENT_FRAMES];
        for (int i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++) set0Layouts[i] = DescriptorSetLayouts[SetFrame];
        DescriptorSetAllocateInfo alloc0 = new()
        {
            SType              = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool     = Gfx.DescriptorPool,
            DescriptorSetCount = Renderer.MAX_CONCURRENT_FRAMES,
            PSetLayouts        = set0Layouts,
        };
        DescriptorSets[SetFrame] = new DescriptorSet[Renderer.MAX_CONCURRENT_FRAMES];
        fixed (DescriptorSet* p = DescriptorSets[SetFrame])
            if (Vk.AllocateDescriptorSets(Device, &alloc0, p) != Result.Success)
                throw new Exception("Failed to allocate pathtracer set 0");

        // Set 1 — single shared (global VB/IB are renderer-wide singletons).
        var geomLayout = DescriptorSetLayouts[SetGeom];
        DescriptorSetAllocateInfo alloc1 = new()
        {
            SType              = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool     = Gfx.DescriptorPool,
            DescriptorSetCount = 1,
            PSetLayouts        = &geomLayout,
        };
        DescriptorSets[SetGeom] = new DescriptorSet[1];
        fixed (DescriptorSet* p = DescriptorSets[SetGeom])
            if (Vk.AllocateDescriptorSets(Device, &alloc1, p) != Result.Success)
                throw new Exception("Failed to allocate pathtracer set 1");

        // Set 2 — borrowed; no allocation here.
        DescriptorSets[SetBindless] = null;

        // Set 3 — IBL, single shared (renderer-wide images).
        var iblLayout = DescriptorSetLayouts[SetIbl];
        DescriptorSetAllocateInfo alloc3 = new()
        {
            SType              = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool     = Gfx.DescriptorPool,
            DescriptorSetCount = 1,
            PSetLayouts        = &iblLayout,
        };
        DescriptorSets[SetIbl] = new DescriptorSet[1];
        fixed (DescriptorSet* p = DescriptorSets[SetIbl])
            if (Vk.AllocateDescriptorSets(Device, &alloc3, p) != Result.Success)
                throw new Exception("Failed to allocate pathtracer set 3");
    }


    // Only the things this pipeline owns are written from WriteDescriptors.
    // The rest are external — each has a public Write* method the renderer
    // calls once the producer exists (TLAS after InitRayQuery, lights after
    // PbrDeferred, storage images after RebuildRenderTargets, etc.).
    protected override void WriteDescriptors()
    {
        WriteFrameUboDescriptors();
        WriteGeometryDescriptors();
    }

    private void WriteFrameUboDescriptors()
    {
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            DescriptorBufferInfo info = new()
            {
                Buffer = _frameUbos[i].buffer,
                Offset = 0,
                Range  = (ulong)sizeof(PathFrameUBO),
            };
            var write = new WriteDescriptorSet
            {
                SType           = StructureType.WriteDescriptorSet,
                DstSet          = DescriptorSets[SetFrame][i],
                DstBinding      = 0,
                DescriptorType  = DescriptorType.UniformBuffer,
                DescriptorCount = 1,
                PBufferInfo     = &info,
            };
            Vk.UpdateDescriptorSets(Device, 1, &write, 0, null);
        }
    }

    /// <summary>Set 1 bindings 1/2: globalVertices + globalIndices. Call once
    /// at startup; buffers are renderer-wide singletons.</summary>
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

    /// <summary>Set 0 binding 1: PbrLight SSBO. Borrowed from Renderer.
    /// Call once after CreateLightBuffers has run — that's part of Renderer
    /// init, so anytime after Renderer.Initialize completes is safe.</summary>
    public void WriteLightsDescriptor()
    {
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            DescriptorBufferInfo info = new()
            {
                Buffer = Renderer.GetLightStorageBuffer((uint)i),
                Offset = 0,
                Range  = (ulong)(Renderer.MAX_LIGHTS * (uint)sizeof(PbrLightGpu)),
            };
            var write = new WriteDescriptorSet
            {
                SType           = StructureType.WriteDescriptorSet,
                DstSet          = DescriptorSets[SetFrame][i],
                DstBinding      = 1,
                DescriptorType  = DescriptorType.StorageBuffer,
                DescriptorCount = 1,
                PBufferInfo     = &info,
            };
            Vk.UpdateDescriptorSets(Device, 1, &write, 0, null);
        }
    }

    /// <summary>Set 0 binding 2: TLAS. Call after InitRayQuery and on every
    /// TLAS rebuild that recreates the handle.</summary>
    public void WriteTlasDescriptor(AccelerationStructureKHR tlas)
    {
        if (tlas.Handle == 0) return;
        var tlasH = tlas;
        var asWrite = new WriteDescriptorSetAccelerationStructureKHR
        {
            SType                      = StructureType.WriteDescriptorSetAccelerationStructureKhr,
            AccelerationStructureCount = 1,
            PAccelerationStructures    = &tlasH,
        };
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            var write = new WriteDescriptorSet
            {
                SType           = StructureType.WriteDescriptorSet,
                PNext           = &asWrite,
                DstSet          = DescriptorSets[SetFrame][i],
                DstBinding      = 2,
                DescriptorType  = DescriptorType.AccelerationStructureKhr,
                DescriptorCount = 1,
            };
            Vk.UpdateDescriptorSets(Device, 1, &write, 0, null);
        }
    }

    /// <summary>Set 0 binding 3: ShadowEntityInfo SSBO. Re-call whenever
    /// RebuildTlas reallocates the underlying VkBuffer.</summary>
    public void WriteShadowInfoDescriptor()
    {
        var buf = Renderer.ShadowInfoBuffer;
        if (buf.Handle == 0) return;
        DescriptorBufferInfo info = new() { Buffer = buf, Offset = 0, Range = Renderer.ShadowInfoBufferSize };
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            var write = new WriteDescriptorSet
            {
                SType           = StructureType.WriteDescriptorSet,
                DstSet          = DescriptorSets[SetFrame][i],
                DstBinding      = 3,
                DescriptorType  = DescriptorType.StorageBuffer,
                DescriptorCount = 1,
                PBufferInfo     = &info,
            };
            Vk.UpdateDescriptorSets(Device, 1, &write, 0, null);
        }
    }

    /// <summary>Set 0 bindings 6/7: emissive-triangle SSBO + alias table.
    /// Borrowed from Renderer (built in RebuildTlas). Re-call whenever the
    /// emissive buffers reallocate. Buffers are always allocated (≥1 slot) once
    /// RebuildTlas has run, so this is safe to call after InitRayQuery.</summary>
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

    /// <summary>Set 0 bindings 4/5: accumulator + outColor storage images.
    /// Both VkImages must be in ImageLayout.General before Record runs. Call
    /// after the renderer creates the images and again on render-extent
    /// resize.</summary>
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

        // Image identity changed (resize), so any in-progress accumulation is
        // invalid by construction — drop the sample count.
        MarkAccumulatorDirty();
    }

    /// <summary>Set 3 bindings 0/1/2/3: irradiance + prefiltered cube + BRDF LUT
    /// + full-res envCube. Call once after CreateIblResources; underlying
    /// VkImage handles persist across IBL rebakes so content updates don't
    /// require re-writes.</summary>
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
        {
            writes[b] = new WriteDescriptorSet
            {
                SType           = StructureType.WriteDescriptorSet,
                DstSet          = DescriptorSets[SetIbl][0],
                DstBinding      = b,
                DescriptorType  = DescriptorType.CombinedImageSampler,
                DescriptorCount = 1,
                PImageInfo      = &imageInfos[b],
            };
        }
        Vk.UpdateDescriptorSets(Device, 4, writes, 0, null);
    }


    // Per-frame UBO upload
    /// <summary>Fills the PathFrameUBO. Returns true when this dispatch will
    /// reset the accumulator (camera move / scene edit / extent change since
    /// the last sample).</summary>
    public bool UpdatePerFrame(uint frameIndex, Camera camera, uint lightCount, Extent2D renderExtent)
    {
        bool reset = _accumDirty;
        if (reset)
        {
            _accumSamples = 0;
            _accumDirty   = false;
        }
        else
        {
            _accumSamples++;
        }

        Matrix4x4 view = camera != null ? camera.GetViewMatrix() : Matrix4x4.Identity;
        Matrix4x4 proj = camera != null
            ? camera.GetProjectionMatrix((float)renderExtent.Width / renderExtent.Height, 0.1f, 100.0f)
            : Matrix4x4.Identity;
        // Y-flip for Vulkan NDC, matching the convention used everywhere else
        // in the renderer. The pathtracer needs the *original* proj inverse,
        // not the flipped one, but since invProj is used to reconstruct ray
        // directions from NDC and we feed Y-flipped NDC into raygen helpers,
        // keeping the flip consistent is correct.
        proj.M22 *= -1;

        Matrix4x4.Invert(view, out var invView);
        Matrix4x4.Invert(proj, out var invProj);

        // FOV comes from the camera — the single source of truth shared with the
        // raster paths (GetProjectionMatrix), so switching into PT doesn't change
        // the framing. Every projection mode (pinhole / thin-lens / panini /
        // fisheye) derives its angle from this vertical FOV.
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
            paniniDistance           = PaniniDistance,
            verticalCompression      = VerticalCompression,
            emissiveTriCount         = Renderer.EmissiveTriangleCount,
            totalEmissivePower       = Renderer.TotalEmissivePower,
        };

        void* data = _frameUbos[frameIndex].mapped;
        new Span<PathFrameUBO>(data, 1).Fill(ubo);
        return reset;
    }


    // Record
    /// <summary>Records the dispatch. The caller is responsible for:
    ///   - Having called UpdatePerFrame this frame.
    ///   - Having written all external descriptors (TLAS, lights, shadow info,
    ///     storage images, IBL) at least once before the first Record.
    ///   - Transitioning the accumulator and outColor images to GENERAL
    ///     before this call (and to ShaderReadOnly afterwards if Tonemap
    ///     samples outColor as a CombinedImageSampler).</summary>
    public void Record(CommandBuffer cmd, in Renderer.FrameContext ctx)
    {
        // Orthographic falls back to Pinhole until the helper lands in PTUtils.
        int modeIdx = (int)Mode;
        if (modeIdx >= _modePipelines.Length) modeIdx = 0;

        Vk.CmdBindPipeline(cmd, PipelineBindPoint.Compute, _modePipelines[modeIdx]);

        var sets = stackalloc DescriptorSet[4]
        {
            DescriptorSets[SetFrame][ctx.FrameIndex],
            DescriptorSets[SetGeom][0],
            Engine.ResourceManager.GetBindlessSet(ctx.FrameIndex),
            DescriptorSets[SetIbl][0],
        };
        Vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute, Layout, 0, 4, sets, 0, null);

        // 8×8 workgroup matches [numthreads(8,8,1)] in PTCompute.slang.
        uint gx = (ctx.RenderExtent.Width  + 7u) / 8u;
        uint gy = (ctx.RenderExtent.Height + 7u) / 8u;
        Vk.CmdDispatch(cmd, gx, gy, 1);
    }


    public override void Dispose()
    {
        // Mode 0 is aliased to PipelineHandle — base.Dispose destroys it.
        // Modes 1-3 are ours to clean up.
        for (int i = 1; i < _modePipelines.Length; i++)
        {
            if (_modePipelines[i].Handle != 0)
                Vk.DestroyPipeline(Device, _modePipelines[i], null);
        }
        foreach (var b in _frameUbos) Gfx.DestroyBuffer(b.buffer, b.alloc);
        base.Dispose();
    }
}