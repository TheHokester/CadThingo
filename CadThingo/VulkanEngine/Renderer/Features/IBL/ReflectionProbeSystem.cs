using System.Numerics;
using System.Runtime.InteropServices;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Features.IBL;

// GPU-side per-probe record. std430-compatible (32B, 16B aligned). Layout must
// stay in lock-step with the PBR shader's `ReflectionProbe` struct when Phase 7
// wires this into the lighting shader.
[StructLayout(LayoutKind.Sequential)]
public struct ProbeGpuRecord
{
    public System.Numerics.Vector4 PositionRadius;  // xyz = world pos, w = influence radius
    public uint CubeArraySlot;                      // index into prefilteredArrayImage layer set (slot*6 .. slot*6+5)
    public uint _pad0;
    public uint _pad1;
    public uint _pad2;
}

// Layout-identical to the PcPrefilter struct inside IblSystem. Mirrored locally
// so the probe system doesn't have to reach across its private types.
[StructLayout(LayoutKind.Sequential)]
internal struct PcProbePrefilter
{
    public float Roughness;
    public uint  MipFaceSize;
    public uint  EnvCubeSize;
}

/// <summary>
/// Owns the GPU resources backing runtime reflection probes: one big cube-array
/// image holding every probe's prefiltered specular cubemap, plus a per-frame
/// SSBO of <see cref="ProbeGpuRecord"/>s. Maintains the CPU-side probe registry
/// and assigns each <see cref="ReflectionProbeComponent"/> a stable cube-array
/// slot.
///
/// Phase-2 status: GPU resources allocated, registry working — but the capture
/// pass, prefilter dispatch, cluster grid, and shader integration are all
/// downstream phases.
/// </summary>
public sealed unsafe class ReflectionProbeSystem : IDisposable
{
    // Hard caps. Bumping MaxProbes resizes the cubemap array (one-time cost at
    // startup); bumping ProbeFaceSize costs 4× VRAM per size step. 16 × 256² is
    // the working default for typical CAD scenes (~67 MB RGBA16F mipped).
    public const uint MaxProbes     = 16;
    public const uint ProbeFaceSize = 256;
    public static readonly uint ProbeMipLevels =
        (uint)System.Math.Floor(System.Math.Log2(ProbeFaceSize)) + 1;
    private readonly Renderer _renderer;
    private readonly GpuContext _gpu;
    private readonly GraphicsDevice _gfx;
    private readonly IblSystem _iblSystem;
    private readonly Vk       _vk;
    private readonly Device   _device;

    // GPU resources

    // Prefiltered cubemap array: 6×MaxProbes layers, mipped (split-sum specular).
    // VK_IMAGE_VIEW_TYPE_CUBE_ARRAY view — sampled by the PBR shader as
    // TextureCubeArray with float4(dir, slot).
    internal Image    prefilteredArrayImage;
    internal SubAlloc   prefilteredArrayAlloc;
    internal ImageView  prefilteredArrayView;
    internal Sampler    prefilteredArraySampler;

    // Per-probe SSBO. Host-visible mapped so the system can write fresh records
    // each frame (cheap: MaxProbes×32B = 512B). One ring per frame-in-flight
    // because the shader reads it during draw and we'd race a writer otherwise.
    internal UboBuffer[] probeRecordBuffers = new UboBuffer[RenderConfig.MAX_CONCURRENT_FRAMES];

    // Transient capture cube. The scene is rendered into this image with multiview
    // (one draw → 6 layers), then the prefilter pass samples it via captureCubeSampleView
    // and writes per-mip into a slot of prefilteredArrayImage. Single mip — the mip
    // chain only exists in the prefiltered array.
    //   layoutAttachment view: VK_IMAGE_VIEW_TYPE_2D_ARRAY (renderable, 6 layers)
    //   layoutSample view:     VK_IMAGE_VIEW_TYPE_CUBE     (samplable by prefilter)
    internal Image     captureCubeImage;
    internal SubAlloc  captureCubeAlloc;
    internal ImageView captureCubeAttachmentView;
    internal ImageView captureCubeSampleView;

    // Capture depth — 6 layers, multiview-routed via gl_ViewIndex. Not sampled
    // after the capture pass; usage = DepthStencilAttachment only.
    internal Image     captureDepthImage;
    internal SubAlloc  captureDepthAlloc;
    internal ImageView captureDepthAttachmentView;

    // Capture pipeline. Null when ProbeCapture failed to compile at startup —
    // in that state the system runs scheduler bookkeeping but skips the actual
    // draw recording. Lets the engine boot during shader iteration without a
    // hard dependency on a working probe shader.
    internal ProbeCapturePipeline? capturePipeline;

    // CPU-built per-cluster probe lists + the SSBOs the PBR shader will bind
    // to look up probes intersecting a fragment. Rebuilt each frame in BuildClusters.
    internal ProbeClusterGrid clusterGrid = null!;

    // Tracks whether the capture image is still in its post-init Undefined
    // layout. First capture transitions Undefined → ColorAttachmentOptimal;
    // subsequent captures transition ShaderReadOnly → ColorAttachmentOptimal.
    private bool _captureCubeInitialLayout = true;

    // Pre-allocated prefilter resources, indexed [slot, mip]. Each storage view
    // covers one mip level of one probe's 6 layers in prefilteredArrayImage,
    // and each descriptor set binds (captureCubeSampleView + iblCubeSampler) →
    // that storage view. Pre-allocating here means RecordCapture's prefilter
    // dispatch never touches the descriptor pool or VkImageView creation, which
    // would otherwise require fence-tracked deferred frees.
    private ImageView[,]    _prefilterStorageViews = null!;
    private DescriptorSet[,] _prefilterSets         = null!;

    // CPU-side registry

    // All probes the system knows about. Insertion-ordered; slot index is
    // assigned at Register and stays stable.
    private readonly List<ReflectionProbeComponent> _probes = new();
    // Free-list of cube-array slots. Pop on Register, push on Unregister.
    private readonly Stack<int> _freeSlots = new();
    

    public IReadOnlyList<ReflectionProbeComponent> Probes => _probes;

    public ReflectionProbeSystem(GpuContext gpu, Renderer renderer, IblSystem iblSystem)
    {
        _gpu = gpu;
        _renderer = renderer;
        _gfx = _gpu.Gfx;
        _vk = _gpu.Gfx.Vk;
        _device = _gpu.Gfx.Device;
        _iblSystem = iblSystem;

        CreatePrefilteredArray();
        CreateSampler();
        CreateProbeRecordBuffers();
        CreateCaptureAttachments();
        TryCreateCapturePipeline();
        CreatePrefilterDescriptors();
        clusterGrid = new ProbeClusterGrid(_gfx);

        for (int i = (int)MaxProbes - 1; i >= 0; i--) _freeSlots.Push(i);
    }

    /// <summary>
    /// Pre-creates one storage view + descriptor set per (probe slot, mip)
    /// pair. Bound to (captureCubeSampleView + iblCubeSampler) → that slot's
    /// 6 layers at the specified mip. The runtime prefilter path only binds +
    /// dispatches; no per-frame descriptor updates.
    /// </summary>
    private void CreatePrefilterDescriptors()
    {
        var bake = _iblSystem.prefilterEnvPipeline;
        var inputView = captureCubeSampleView;
        var inputSampler = _iblSystem.iblCubeSampler;

        _prefilterStorageViews = new ImageView[MaxProbes, ProbeMipLevels];
        _prefilterSets         = new DescriptorSet[MaxProbes, ProbeMipLevels];

        for (uint slot = 0; slot < MaxProbes; slot++)
        for (uint mip  = 0; mip  < ProbeMipLevels; mip++)
        {
            ImageViewCreateInfo info = new()
            {
                SType    = StructureType.ImageViewCreateInfo,
                Image    = prefilteredArrayImage,
                ViewType = ImageViewType.Type2DArray,
                Format   = Format.R16G16B16A16Sfloat,
                SubresourceRange = new ImageSubresourceRange
                {
                    AspectMask     = ImageAspectFlags.ColorBit,
                    BaseMipLevel   = mip,
                    LevelCount     = 1,
                    BaseArrayLayer = slot * 6,
                    LayerCount     = 6,
                },
            };
            if (_vk.CreateImageView(_device, &info, null, out var view) != Result.Success)
                throw new Exception($"Failed to create probe prefilter view (slot {slot}, mip {mip})");
            _prefilterStorageViews[slot, mip] = view;

            var set = bake.AllocateDescriptorSet();
            bake.WriteDescriptors(set, view, inputView, inputSampler);
            _prefilterSets[slot, mip] = set;
        }
    }

    private void TryCreateCapturePipeline()
    {
        // ProbeCapture.slang is compiled here, at pipeline-create time. A syntax error in it
        // used to be a missing .spv; now it is an exception out of ShaderLibrary. Either way the
        // engine still boots -- probes keep their scheduler bookkeeping and skip the draw -- so
        // shader iteration on the capture kernel does not take the whole editor down with it.
        try
        {
            capturePipeline = new ProbeCapturePipeline(_gpu, _renderer);
            capturePipeline.Initialize();
        }
        catch (Exception e)
        {
            capturePipeline = null;
            Console.Error.WriteLine($"[Probe] capture pipeline skipped: {e.Message}");
        }
    }

    // Registry

    /// <summary>
    /// Assigns <paramref name="probe"/> a cube-array slot and marks it dirty
    /// for first capture. Silently no-ops if the probe is already registered.
    /// Returns false when the slot pool is exhausted  caller should warn the
    /// user; the probe still works as an entity, it just won't render.
    /// </summary>
    public bool Register(ReflectionProbeComponent probe)
    {
        if (probe.CubeArraySlot >= 0) return true;
        if (_freeSlots.Count == 0) return false;

        probe.CubeArraySlot = _freeSlots.Pop();
        probe.Dirty = true;
        _probes.Add(probe);
        return true;
    }

    public void Unregister(ReflectionProbeComponent probe)
    {
        if (probe.CubeArraySlot < 0) return;
        _freeSlots.Push(probe.CubeArraySlot);
        _probes.Remove(probe);
        probe.CubeArraySlot = -1;
    }

    /// <summary>
    /// Marks every probe whose influence sphere contains <paramref name="worldPoint"/>
    /// as dirty. Wired by Phase 8's transform-change event hook so that moving an
    /// entity invalidates the probes that can see it.
    /// </summary>
    public void MarkDirtyAt(Vector3 worldPoint)
    {
        foreach (var p in _probes)
        {
            var t = (Entity*)null; // resolved by caller via component->entity link (Phase 8)
            // Position lookup will land alongside the dirty-propagation hook;
            // for now the system can be force-dirtied from the editor panel.
            _ = t; _ = p;
        }
    }

    // GPU resource lifetime

    private void CreatePrefilteredArray()
    {
        ImageCreateInfo info = new()
        {
            SType         = StructureType.ImageCreateInfo,
            ImageType     = ImageType.Type2D,
            Format        = Format.R16G16B16A16Sfloat,
            Extent        = new Extent3D(ProbeFaceSize, ProbeFaceSize, 1),
            MipLevels     = ProbeMipLevels,
            ArrayLayers   = 6 * MaxProbes,
            Samples       = SampleCountFlags.Count1Bit,
            Tiling        = ImageTiling.Optimal,
            // Sampled for shader reads; Storage so the prefilter compute pass can
            // write per-mip per-face per-probe (Phase 5); TransferDst for layout
            // initialization barriers.
            Usage         = ImageUsageFlags.SampledBit | ImageUsageFlags.StorageBit | ImageUsageFlags.TransferDstBit,
            SharingMode   = SharingMode.Exclusive,
            InitialLayout = ImageLayout.Undefined,
            Flags         = ImageCreateFlags.CreateCubeCompatibleBit,
        };
        if (_vk.CreateImage(_device, &info, null, out prefilteredArrayImage) != Result.Success)
            throw new Exception("Failed to create reflection probe cubemap array image");

        prefilteredArrayAlloc = _gfx.Allocator.AllocateForImage(
            prefilteredArrayImage, MemoryPropertyFlags.DeviceLocalBit);

        ImageViewCreateInfo viewInfo = new()
        {
            SType    = StructureType.ImageViewCreateInfo,
            Image    = prefilteredArrayImage,
            ViewType = ImageViewType.TypeCubeArray,
            Format   = Format.R16G16B16A16Sfloat,
            SubresourceRange = new ImageSubresourceRange
            {
                AspectMask     = ImageAspectFlags.ColorBit,
                BaseMipLevel   = 0,
                LevelCount     = ProbeMipLevels,
                BaseArrayLayer = 0,
                LayerCount     = 6 * MaxProbes,
            },
        };
        if (_vk.CreateImageView(_device, &viewInfo, null, out prefilteredArrayView) != Result.Success)
            throw new Exception("Failed to create reflection probe cubemap array view");

        // Initial transition to ShaderReadOnlyOptimal — the PBR pipeline binds
        // this descriptor unconditionally even before any probe has been baked.
        // Empty content reads as zero specular, which falls back cleanly to the
        // global IBL term in the lighting shader.
        var cmd = _gfx.BeginSingleTimeCommands();
        ImageMemoryBarrier barrier = new()
        {
            SType               = StructureType.ImageMemoryBarrier,
            Image               = prefilteredArrayImage,
            OldLayout           = ImageLayout.Undefined,
            NewLayout           = ImageLayout.ShaderReadOnlyOptimal,
            SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
            SrcAccessMask       = 0,
            DstAccessMask       = AccessFlags.ShaderReadBit,
            SubresourceRange    = new ImageSubresourceRange(ImageAspectFlags.ColorBit, 0, ProbeMipLevels, 0, 6 * MaxProbes),
        };
        _vk.CmdPipelineBarrier(cmd,
            PipelineStageFlags.TopOfPipeBit, PipelineStageFlags.FragmentShaderBit,
            0, 0, null, 0, null, 1, &barrier);
        _gfx.EndSingleTimeCommands(cmd);
    }

    private void CreateSampler()
    {
        SamplerCreateInfo info = new()
        {
            SType         = StructureType.SamplerCreateInfo,
            MagFilter     = Filter.Linear,
            MinFilter     = Filter.Linear,
            AddressModeU  = SamplerAddressMode.ClampToEdge,
            AddressModeV  = SamplerAddressMode.ClampToEdge,
            AddressModeW  = SamplerAddressMode.ClampToEdge,
            MipmapMode    = SamplerMipmapMode.Linear,
            MinLod        = 0f,
            MaxLod        = Vk.LodClampNone,
            CompareEnable = false,
        };
        if (_vk.CreateSampler(_device, &info, null, out prefilteredArraySampler) != Result.Success)
            throw new Exception("Failed to create reflection probe sampler");
    }

    private void CreateProbeRecordBuffers()
    {
        ulong sizeBytes = MaxProbes * (ulong)sizeof(ProbeGpuRecord);
        for (int i = 0; i < RenderConfig.MAX_CONCURRENT_FRAMES; i++)
            _gfx.CreateMappedStorageBuffer(sizeBytes, ref probeRecordBuffers[i]);
    }

    private void CreateCaptureAttachments()
    {
        // Color cube
        ImageCreateInfo colorInfo = new()
        {
            SType         = StructureType.ImageCreateInfo,
            ImageType     = ImageType.Type2D,
            Format        = Format.R16G16B16A16Sfloat,
            Extent        = new Extent3D(ProbeFaceSize, ProbeFaceSize, 1),
            MipLevels     = 1,
            ArrayLayers   = 6,
            Samples       = SampleCountFlags.Count1Bit,
            Tiling        = ImageTiling.Optimal,
            // ColorAttachment: rendered into by the capture pass.
            // Sampled:         consumed by the prefilter compute pass as input.
            // TransferSrc:     allowed for debug blits / inspection.
            Usage         = ImageUsageFlags.ColorAttachmentBit | ImageUsageFlags.SampledBit |
                            ImageUsageFlags.TransferSrcBit,
            SharingMode   = SharingMode.Exclusive,
            InitialLayout = ImageLayout.Undefined,
            Flags         = ImageCreateFlags.CreateCubeCompatibleBit,
        };
        if (_vk.CreateImage(_device, &colorInfo, null, out captureCubeImage) != Result.Success)
            throw new Exception("Failed to create probe capture cube image");
        captureCubeAlloc = _gfx.Allocator.AllocateForImage(captureCubeImage, MemoryPropertyFlags.DeviceLocalBit);

        // 2D_ARRAY view for use as a color attachment in the capture pass.
        // Multiview directs gl_ViewIndex-tagged writes to layers 0..5.
        ImageViewCreateInfo colorAttachViewInfo = new()
        {
            SType    = StructureType.ImageViewCreateInfo,
            Image    = captureCubeImage,
            ViewType = ImageViewType.Type2DArray,
            Format   = Format.R16G16B16A16Sfloat,
            SubresourceRange = new ImageSubresourceRange
            {
                AspectMask = ImageAspectFlags.ColorBit,
                BaseMipLevel = 0, LevelCount = 1,
                BaseArrayLayer = 0, LayerCount = 6,
            },
        };
        if (_vk.CreateImageView(_device, &colorAttachViewInfo, null, out captureCubeAttachmentView) != Result.Success)
            throw new Exception("Failed to create probe capture color attachment view");

        // CUBE view sampled by the prefilter compute pass — needs the cube
        // semantics so direction-based sampling works.
        ImageViewCreateInfo colorSampleViewInfo = colorAttachViewInfo;
        colorSampleViewInfo.ViewType = ImageViewType.TypeCube;
        if (_vk.CreateImageView(_device, &colorSampleViewInfo, null, out captureCubeSampleView) != Result.Success)
            throw new Exception("Failed to create probe capture cube-sample view");

        // Depth cube
        ImageCreateInfo depthInfo = new()
        {
            SType         = StructureType.ImageCreateInfo,
            ImageType     = ImageType.Type2D,
            Format        = Format.D32Sfloat,
            Extent        = new Extent3D(ProbeFaceSize, ProbeFaceSize, 1),
            MipLevels     = 1,
            ArrayLayers   = 6,
            Samples       = SampleCountFlags.Count1Bit,
            Tiling        = ImageTiling.Optimal,
            Usage         = ImageUsageFlags.DepthStencilAttachmentBit,
            SharingMode   = SharingMode.Exclusive,
            InitialLayout = ImageLayout.Undefined,
        };
        if (_vk.CreateImage(_device, &depthInfo, null, out captureDepthImage) != Result.Success)
            throw new Exception("Failed to create probe capture depth image");
        captureDepthAlloc = _gfx.Allocator.AllocateForImage(captureDepthImage, MemoryPropertyFlags.DeviceLocalBit);

        ImageViewCreateInfo depthViewInfo = new()
        {
            SType    = StructureType.ImageViewCreateInfo,
            Image    = captureDepthImage,
            ViewType = ImageViewType.Type2DArray,
            Format   = Format.D32Sfloat,
            SubresourceRange = new ImageSubresourceRange
            {
                AspectMask = ImageAspectFlags.DepthBit,
                BaseMipLevel = 0, LevelCount = 1,
                BaseArrayLayer = 0, LayerCount = 6,
            },
        };
        if (_vk.CreateImageView(_device, &depthViewInfo, null, out captureDepthAttachmentView) != Result.Success)
            throw new Exception("Failed to create probe capture depth view");
    }


    /// <summary>
    /// Picks the next probe to capture this frame. Pass priority: probes that
    /// have never captured come before probes that just need a refresh. Caller
    /// is expected to invoke this once per frame and skip the capture pass when
    /// the return value is null.
    /// </summary>
    public ReflectionProbeComponent? NextDirtyProbe()
    {
        // Two-pass scan keeps "never captured" probes ahead of "needs refresh"
        // without sorting the list. Cost is O(N) where N ≤ MaxProbes — trivial.
        for (int i = 0; i < _probes.Count; i++)
        {
            var p = _probes[i];
            if (p.Dirty && p.LastCaptureFrame == 0) return p;
        }
        for (int i = 0; i < _probes.Count; i++)
        {
            var p = _probes[i];
            if (p.Dirty) return p;
        }
        return null;
    }

    /// <summary>
    /// Builds the 6 view matrices + 1 perspective projection used by the capture
    /// pass for a probe centred at <paramref name="position"/>. Face ordering
    /// follows Vulkan's cube-array layer convention:
    /// <code>
    ///   layer 0 = +X   layer 1 = -X
    ///   layer 2 = +Y   layer 3 = -Y
    ///   layer 4 = +Z   layer 5 = -Z
    /// </code>
    /// View matrices use the +Y-down up vector for the X/Z faces (Vulkan cube
    /// convention) which matches the existing IBL bake — the prefilter pass can
    /// sample <see cref="captureCubeSampleView"/> directly without face remapping.
    /// </summary>
    public static void BuildCaptureMatrices(Vector3 position, float nearPlane, float farPlane,
        Span<Matrix4x4> views, out Matrix4x4 projection)
    {
        if (views.Length < 6) throw new ArgumentException("views span must hold 6 matrices");

        // 90° FOV — fills the full face. NOTE: no Y-flip here, even though the
        // screen pipelines all do `proj.M22 *= -1f` for Vulkan's y-down NDC.
        // Cube faces are written to a texture that will later be sampled with
        // Vulkan-native v=0-at-top convention; the GLM-style up vectors below
        // already place world +Y at clip -y, which lands at framebuffer top
        // (=texture v=0) under Vulkan rules. Adding an M22 flip here would
        // double-flip vertical orientation — walls would render upside down.
        projection = Matrix4x4.CreatePerspectiveFieldOfView(MathF.PI / 2f, 1.0f, nearPlane, farPlane);

        // Six face directions + their up vectors. Up = (0,-1,0) for the lateral
        // faces and ±Z for the polar ones is the canonical Vulkan cubemap setup.
        var dirs = stackalloc Vector3[6]
        {
            new( 1f,  0f,  0f),  // +X
            new(-1f,  0f,  0f),  // -X
            new( 0f,  1f,  0f),  // +Y
            new( 0f, -1f,  0f),  // -Y
            new( 0f,  0f,  1f),  // +Z
            new( 0f,  0f, -1f),  // -Z
        };
        var ups = stackalloc Vector3[6]
        {
            new( 0f, -1f,  0f),
            new( 0f, -1f,  0f),
            new( 0f,  0f,  1f),
            new( 0f,  0f, -1f),
            new( 0f, -1f,  0f),
            new( 0f, -1f,  0f),
        };

        for (int i = 0; i < 6; i++)
            views[i] = Matrix4x4.CreateLookAt(position, position + dirs[i], ups[i]);
    }

    /// <summary>
    /// Per-frame CPU bookkeeping. Refreshes cached probe world positions from
    /// their owning entities' transforms and flips Dirty for probes whose
    /// EveryNFrames interval has elapsed. Run before any GPU work in DrawFrame.
    /// </summary>
    public void Tick(ulong frameIndex, Scene scene)
    {
        // Refresh world positions. Walk scene once and match registered probes
        // by reference — O(N entities) per frame regardless of probe count.
        foreach (var ptr in scene.Entities)
        {
            var e = (Entity*)ptr;
            if (e == null || !e->IsActive) continue;
            var probe = e->GetComponent<ReflectionProbeComponent>();
            if (probe == null || probe.CubeArraySlot < 0) continue;
            var t = e->GetComponent<TransformComponent>();
            if (t != null) probe.WorldPosition = t.GetPosition();
        }

        foreach (var p in _probes)
        {
            if (p.UpdatePolicy == ProbeUpdatePolicy.EveryNFrames
                && frameIndex - p.LastCaptureFrame >= p.UpdateIntervalFrames)
                p.Dirty = true;
        }
    }

    /// <summary>
    /// Writes <see cref="ProbeGpuRecord"/>s for every registered probe into the
    /// per-frame mapped SSBO. Cheap (MaxProbes × 32B). Unregistered slots are
    /// left zeroed so the PBR shader's distance test trivially rejects them.
    /// </summary>
    public void WriteProbeRecords(uint frameIndex)
    {
        var ptr = (ProbeGpuRecord*)probeRecordBuffers[frameIndex].mapped;
        // Zero everything first — slot indices can be sparse if probes were
        // unregistered, and the cluster grid emits absolute slot indices.
        for (int i = 0; i < MaxProbes; i++) ptr[i] = default;

        foreach (var p in _probes)
        {
            if (p.CubeArraySlot < 0) continue;
            ptr[p.CubeArraySlot] = new ProbeGpuRecord
            {
                PositionRadius = new Vector4(p.WorldPosition, p.InfluenceRadius),
                CubeArraySlot  = (uint)p.CubeArraySlot,
            };
        }
    }

    /// <summary>
    /// Rebuilds the cluster grid for the next frame's PBR pass. Called from
    /// DrawFrame after the camera's per-frame matrices are settled and tile
    /// counts known. Tile-only today; pass zSlices &gt; 1 when 3D clustering lands.
    /// </summary>
    public void BuildClusters(uint frameIndex, Camera camera, float aspect, float nearZ, float farZ,
        uint tileCountX, uint tileCountY, uint zSlices = 1)
    {
        clusterGrid.Build(frameIndex, camera, aspect, nearZ, farZ,
            tileCountX, tileCountY, zSlices, _probes);
    }

    /// <summary>
    /// Records one probe capture into <paramref name="cmd"/>. Picks the next
    /// dirty probe; if none, returns without touching the command buffer.
    /// Caller is responsible for placing this call before any pass that would
    /// sample <see cref="captureCubeSampleView"/> downstream (currently only
    /// the prefilter compute pass, landing in Phase 5).
    /// </summary>
    public void RecordCapture(CommandBuffer cmd, uint frameIndex, ulong frameCounter, Scene scene)
    {
        if (capturePipeline == null) return;

        var probe = NextDirtyProbe();
        if (probe == null) return;

        // Position cached by Tick — no per-capture scene walk.
        Vector3 probePos = probe.WorldPosition;

        // Build + upload per-face VP UBO
        Span<Matrix4x4> views = stackalloc Matrix4x4[6];
        BuildCaptureMatrices(probePos, 0.1f, 200f, views, out var proj);
        var ubo = new ProbeCapturePipeline.CaptureUbo
        {
            View0 = views[0], View1 = views[1], View2 = views[2],
            View3 = views[3], View4 = views[4], View5 = views[5],
            Proj  = proj,
        };
        capturePipeline.WriteUbo(frameIndex, in ubo);

        // Layout transitions: cube + depth to attachment layouts
        ImageMemoryBarrier* preBarriers = stackalloc ImageMemoryBarrier[2];
        preBarriers[0] = new()
        {
            SType               = StructureType.ImageMemoryBarrier,
            Image               = captureCubeImage,
            OldLayout           = _captureCubeInitialLayout ? ImageLayout.Undefined : ImageLayout.ShaderReadOnlyOptimal,
            NewLayout           = ImageLayout.ColorAttachmentOptimal,
            SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
            SrcAccessMask       = _captureCubeInitialLayout ? 0 : AccessFlags.ShaderReadBit,
            DstAccessMask       = AccessFlags.ColorAttachmentWriteBit,
            SubresourceRange    = new ImageSubresourceRange(ImageAspectFlags.ColorBit, 0, 1, 0, 6),
        };
        preBarriers[1] = new()
        {
            SType               = StructureType.ImageMemoryBarrier,
            Image               = captureDepthImage,
            OldLayout           = ImageLayout.Undefined,  // depth is cleared at LoadOp so previous content doesn't matter
            NewLayout           = ImageLayout.DepthStencilAttachmentOptimal,
            SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
            SrcAccessMask       = 0,
            DstAccessMask       = AccessFlags.DepthStencilAttachmentWriteBit | AccessFlags.DepthStencilAttachmentReadBit,
            SubresourceRange    = new ImageSubresourceRange(ImageAspectFlags.DepthBit, 0, 1, 0, 6),
        };
        _vk.CmdPipelineBarrier(cmd,
            PipelineStageFlags.FragmentShaderBit | PipelineStageFlags.TopOfPipeBit,
            PipelineStageFlags.ColorAttachmentOutputBit | PipelineStageFlags.EarlyFragmentTestsBit,
            0, 0, null, 0, null, 2, preBarriers);

        // Begin multiview rendering
        var colorAttach = new RenderingAttachmentInfo
        {
            SType       = StructureType.RenderingAttachmentInfo,
            ImageView   = captureCubeAttachmentView,
            ImageLayout = ImageLayout.ColorAttachmentOptimal,
            LoadOp      = AttachmentLoadOp.Clear,
            StoreOp     = AttachmentStoreOp.Store,
            ClearValue  = new ClearValue { Color = new ClearColorValue(0f, 0f, 0f, 1f) },
        };
        var depthAttach = new RenderingAttachmentInfo
        {
            SType       = StructureType.RenderingAttachmentInfo,
            ImageView   = captureDepthAttachmentView,
            ImageLayout = ImageLayout.DepthStencilAttachmentOptimal,
            LoadOp      = AttachmentLoadOp.Clear,
            StoreOp     = AttachmentStoreOp.DontCare,
            ClearValue  = new ClearValue { DepthStencil = new ClearDepthStencilValue(1f, 0) },
        };
        var renderInfo = new RenderingInfo
        {
            SType                = StructureType.RenderingInfo,
            RenderArea           = new Rect2D(new Offset2D(0, 0), new Extent2D(ProbeFaceSize, ProbeFaceSize)),
            LayerCount           = 1,           // multiview supplies the layer fan-out
            ViewMask             = 0x3Fu,       // 6 bits = 6 cube faces
            ColorAttachmentCount = 1,
            PColorAttachments    = &colorAttach,
            PDepthAttachment     = &depthAttach,
        };
        _vk.CmdBeginRendering(cmd, &renderInfo);

        _vk.CmdBindPipeline(cmd, PipelineBindPoint.Graphics, capturePipeline.Handle);

        Viewport vp = new() { X = 0, Y = 0, Width = ProbeFaceSize, Height = ProbeFaceSize, MinDepth = 0f, MaxDepth = 1f };
        Rect2D scissor = new(new Offset2D(0, 0), new Extent2D(ProbeFaceSize, ProbeFaceSize));
        _vk.CmdSetViewport(cmd, 0, 1, &vp);
        _vk.CmdSetScissor (cmd, 0, 1, &scissor);

        var frameSet    = capturePipeline.GetFrameSet(frameIndex);
        var bindlessSet = Engine.ResourceManager.GetBindlessSet(frameIndex);
        var sets        = stackalloc DescriptorSet[2] { frameSet, bindlessSet };
        _vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.Graphics,
            capturePipeline.Layout, 0, 2, sets, 0, null);

        var vb = Engine.ResourceManager.GlobalVertexBuffer;
        var ib = Engine.ResourceManager.GlobalIndexBuffer;
        ulong vbOffset = 0;
        _vk.CmdBindVertexBuffers(cmd, 0, 1, &vb, &vbOffset);
        _vk.CmdBindIndexBuffer(cmd, ib, 0, IndexType.Uint32);

        // Per-entity direct draws
        // No GPU cull for probe captures yet — every renderable goes through.
        // For typical CAD scenes (≤ ~1k draws) this is well under the 1ms budget
        // and avoids standing up a separate cull pass per probe.
        var pc = default(ProbeCapturePipeline.CapturePushConst);
        foreach (var ptr in scene.Entities)
        {
            var e = (Entity*)ptr;
            if (e == null || !e->IsActive) continue;
            var mc = e->GetComponent<MeshComponent>();
            if (mc == null || mc.mesh == null) continue;
            var tc = e->GetComponent<TransformComponent>();
            if (tc == null) continue;

            pc.Model         = *tc.GetWorldMatrix();
            pc.MaterialIndex = (uint)(mc.materialIndex >= 0 ? mc.materialIndex : scene.MaterialCount);
            _vk.CmdPushConstants(cmd, capturePipeline.Layout,
                ShaderStageFlags.VertexBit | ShaderStageFlags.FragmentBit,
                0, (uint)sizeof(ProbeCapturePipeline.CapturePushConst), &pc);

            _vk.CmdDrawIndexed(cmd, (uint)mc.mesh->count, 1, (uint)mc.mesh->offset, 0, 0);
        }

        _vk.CmdEndRendering(cmd);

        // Transition cube → ShaderReadOnly so prefilter compute can sample it 
        // Also transition the destination probe slot's layers in the prefiltered
        // array from ShaderReadOnly → General for storage writes. Combined into
        // one CmdPipelineBarrier so we pay the sync cost once.
        ImageMemoryBarrier* postBarriers = stackalloc ImageMemoryBarrier[2];
        postBarriers[0] = new()
        {
            SType               = StructureType.ImageMemoryBarrier,
            Image               = captureCubeImage,
            OldLayout           = ImageLayout.ColorAttachmentOptimal,
            NewLayout           = ImageLayout.ShaderReadOnlyOptimal,
            SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
            SrcAccessMask       = AccessFlags.ColorAttachmentWriteBit,
            DstAccessMask       = AccessFlags.ShaderReadBit,
            SubresourceRange    = new ImageSubresourceRange(ImageAspectFlags.ColorBit, 0, 1, 0, 6),
        };
        postBarriers[1] = new()
        {
            SType               = StructureType.ImageMemoryBarrier,
            Image               = prefilteredArrayImage,
            OldLayout           = ImageLayout.ShaderReadOnlyOptimal,
            NewLayout           = ImageLayout.General,
            SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
            SrcAccessMask       = AccessFlags.ShaderReadBit,
            DstAccessMask       = AccessFlags.ShaderWriteBit,
            SubresourceRange    = new ImageSubresourceRange(
                ImageAspectFlags.ColorBit, 0, ProbeMipLevels, (uint)probe.CubeArraySlot * 6, 6),
        };
        _vk.CmdPipelineBarrier(cmd,
            PipelineStageFlags.ColorAttachmentOutputBit | PipelineStageFlags.FragmentShaderBit,
            PipelineStageFlags.ComputeShaderBit,
            0, 0, null, 0, null, 2, postBarriers);

        // Prefilter dispatch: per-mip GGX importance sample into the slot
        var prefilter = _iblSystem.prefilterEnvPipeline;
        _vk.CmdBindPipeline(cmd, PipelineBindPoint.Compute, prefilter.Handle);
        uint slot = (uint)probe.CubeArraySlot;
        for (uint m = 0; m < ProbeMipLevels; m++)
        {
            uint mipSize = Math.Max(1u, ProbeFaceSize >> (int)m);
            float roughness = ProbeMipLevels == 1 ? 0f : m / (float)(ProbeMipLevels - 1);

            var set = _prefilterSets[slot, m];
            _vk.CmdBindDescriptorSets(cmd, PipelineBindPoint.Compute, prefilter.Layout,
                0, 1, &set, 0, null);

            var prefPc = new PcProbePrefilter
            {
                Roughness   = roughness,
                MipFaceSize = mipSize,
                EnvCubeSize = ProbeFaceSize,
            };
            _vk.CmdPushConstants(cmd, prefilter.Layout, ShaderStageFlags.ComputeBit,
                0, (uint)sizeof(PcProbePrefilter), &prefPc);

            uint groups = (mipSize + 15) / 16;
            _vk.CmdDispatch(cmd, groups, groups, 6);
        }

        // Transition slot's layers back to ShaderReadOnly
        ImageMemoryBarrier slotReadBarrier = new()
        {
            SType               = StructureType.ImageMemoryBarrier,
            Image               = prefilteredArrayImage,
            OldLayout           = ImageLayout.General,
            NewLayout           = ImageLayout.ShaderReadOnlyOptimal,
            SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
            SrcAccessMask       = AccessFlags.ShaderWriteBit,
            DstAccessMask       = AccessFlags.ShaderReadBit,
            SubresourceRange    = new ImageSubresourceRange(
                ImageAspectFlags.ColorBit, 0, ProbeMipLevels, slot * 6, 6),
        };
        _vk.CmdPipelineBarrier(cmd,
            PipelineStageFlags.ComputeShaderBit,
            PipelineStageFlags.FragmentShaderBit,
            0, 0, null, 0, null, 1, &slotReadBarrier);

        _captureCubeInitialLayout = false;
        probe.LastCaptureFrame    = frameCounter;
        probe.Dirty               = probe.UpdatePolicy switch
        {
            ProbeUpdatePolicy.Once         => false,
            ProbeUpdatePolicy.OnDirty      => false,
            ProbeUpdatePolicy.EveryNFrames => false,  // Tick re-flips when interval elapses
            _ => false,
        };
    }

    public void Dispose()
    {
        clusterGrid?.Dispose();

        // Prefilter resources first — descriptor sets live in the renderer's
        // pool which is destroyed downstream; views must be torn down explicitly.
        if (_prefilterStorageViews != null)
        {
            for (int slot = 0; slot < MaxProbes; slot++)
            for (int mip  = 0; mip  < ProbeMipLevels; mip++)
            {
                var v = _prefilterStorageViews[slot, mip];
                if (v.Handle != 0) _vk.DestroyImageView(_device, v, null);
            }
        }

        capturePipeline?.Dispose();

        for (int i = 0; i < RenderConfig.MAX_CONCURRENT_FRAMES; i++)
            _gfx.DestroyBuffer(probeRecordBuffers[i].buffer, probeRecordBuffers[i].alloc);

        if (captureDepthAttachmentView.Handle != 0) _vk.DestroyImageView(_device, captureDepthAttachmentView, null);
        if (captureDepthImage.Handle          != 0) _vk.DestroyImage(_device, captureDepthImage, null);
        _gfx.Allocator.Free(captureDepthAlloc);

        if (captureCubeSampleView.Handle     != 0) _vk.DestroyImageView(_device, captureCubeSampleView, null);
        if (captureCubeAttachmentView.Handle != 0) _vk.DestroyImageView(_device, captureCubeAttachmentView, null);
        if (captureCubeImage.Handle          != 0) _vk.DestroyImage(_device, captureCubeImage, null);
        _gfx.Allocator.Free(captureCubeAlloc);

        if (prefilteredArraySampler.Handle != 0) _vk.DestroySampler(_device, prefilteredArraySampler, null);
        if (prefilteredArrayView.Handle    != 0) _vk.DestroyImageView(_device, prefilteredArrayView, null);
        if (prefilteredArrayImage.Handle   != 0) _vk.DestroyImage(_device, prefilteredArrayImage, null);
        _gfx.Allocator.Free(prefilteredArrayAlloc);

        _probes.Clear();
        _freeSlots.Clear();
    }
}
