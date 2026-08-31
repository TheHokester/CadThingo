using System.Numerics;
using System.Reflection;
using CadThingo.VulkanEngine.ImGui;
using CadThingo.VulkanEngine.Renderer.Descriptors;
using CadThingo.VulkanEngine.Renderer.FeatureLifecycle;
using CadThingo.VulkanEngine.Renderer.FeatureLifecycle.RenderFeatureInterfaces;
using CadThingo.VulkanEngine.Renderer.Features.Deferred;
using CadThingo.VulkanEngine.Renderer.Features.IBL;
using CadThingo.VulkanEngine.Renderer.Features.PathTracer;
using CadThingo.VulkanEngine.Renderer.Features.SceneAcceleration;
using CadThingo.VulkanEngine.Renderer.Features.Shared;
using CadThingo.VulkanEngine.Renderer.Features.Tonemapping;
using CadThingo.VulkanEngine.Renderer.Features.WavefrontPathTracer;
using CadThingo.VulkanEngine.Renderer.FrameGraph;
using CadThingo.VulkanEngine.Renderer.Pipelines;
using CadThingo.VulkanEngine.Renderer.Slang;
using Silk.NET.Vulkan;
using Silk.NET.Windowing;
// ReSharper disable InconsistentNaming

namespace CadThingo.VulkanEngine.Renderer;

/// <summary>
/// Composition root for the render path. Builds the device, the presentation chain, the shared
/// scene mirror and the feature set, then runs one frame skeleton: extract, pump the features,
/// dispatch the active core, blit, present.
///
/// This class names no technique. Every pipeline and subsystem belongs to a feature the
/// <see cref="FeatureHost"/> constructs from the catalog, so a new technique is a new file with a
/// descriptor. What is left here that names a feature type is bridge state for callers that still
/// hold a Renderer, marked as such where it appears.
/// </summary>
public unsafe class Renderer
{
    public Renderer(IWindow window)
    {
        this.window = window;
        camera = new Camera();
    }

    //=========================================================================
    //====== Device And Presentation ==========================================
    //=========================================================================

    internal GraphicsDevice gfx = null!;

    private bool initialized;

    // Instance validation layers. Sits next to the renderer so the toggle is where the frame
    // skeleton is; handed to the GraphicsDevice constructor.
    private const bool enableValidationLayers = false;
    private readonly IWindow? window;

    // Delegating accessors onto GraphicsDevice. Callers outside the renderer read these under
    // their old field names; each one deletes itself when its last caller takes a GpuContext.
    internal Vk?                vk             => gfx?.Vk;
    internal Device             device         => gfx.Device;
    internal PhysicalDevice     physicalDevice => gfx.PhysicalDevice;
    internal GpuMemoryAllocator memAllocator   => gfx.Allocator;
    internal Queue              graphicsQueue  => gfx.GraphicsQueue;
    internal Queue              presentQueue   => gfx.PresentQueue;

    /// <summary>True when both KHR_acceleration_structure and KHR_ray_query are enabled. Gates the
    /// ray-traced shadow path.</summary>
    public bool RayShadowsSupported => gfx.RayShadowsSupported;

    /// <summary>True when KHR_ray_tracing_pipeline is enabled alongside
    /// KHR_acceleration_structure. Gates the opt-in RT-pipeline path tracer.</summary>
    public bool RayTracePipelineSupported => gfx.RayTracePipelineSupported;

    /// <summary>True when VK_NV_ray_tracing_invocation_reorder is enabled on top of the RT
    /// pipeline. Selects the HitObject/ReorderThread raygen variant.</summary>
    public bool SerSupported => gfx.SerSupported;

    internal Swapchain   swapchain            = null!;
    internal Format      swapChainImageFormat => swapchain.ImageFormat;
    internal Extent2D    swapChainExtent      => swapchain.Extent;
    internal Image[]     swapChainImages      => swapchain.Images;
    internal ImageView[] swapChainImageViews  => swapchain.ImageViews;

    // Depth, g-buffers, sampler and the PT/selection images, sized to the render extent.
    internal RenderTargets renderTargets = null!;

    // Size of the deferred chain and the lighting tile grid. Tracks the swapchain only until the
    // editor's viewport panel drives it smaller than the OS window.
    internal Extent2D renderExtent => renderTargets.RenderExtent;
    public   Extent2D RenderExtent => renderExtent;

    // Per-frame command buffers plus the sync ring. FrameRing owns the acquire/submit/present
    // cadence; currentFrame and frameCounter advance through frameRing.Advance().
    internal FrameRing frameRing = null!;

    /// <summary>UI overlay drawn after the FinalColor blit and before the present transition. Null
    /// when the editor UI is disabled; every use is guarded.</summary>
    public ImGuiVulkanUtils? imGuiUtils;

    // The scene-to-GPU mirror and the single Extract path.
    internal GpuScene gpuScene = null!;

    private Scene scene;
    public Scene Scene => scene;
    private Entity* testEntity;

    private Camera camera;
    public Camera Camera => camera;

    // Runtime shader compile plus cache, and the unified scene set plus constant arena. Pipelines
    // migrate onto the registry shader by shader.
    internal ShaderLibrary      shaderLibrary      = null!;
    internal DescriptorRegistry descriptorRegistry = null!;

    /// <summary>Device services bundled for injection. Valid once Initialize has built the device,
    /// shader library and registry; construct pipelines after that point.</summary>
    internal GpuContext Gpu;

    //=========================================================================
    //====== Features And Core Selection ======================================
    //=========================================================================

    public enum RenderMode : uint
    {
        Deferred    = 0,
        ForwardPlus = 1,
        RayCompute  = 2,
        RayTrace    = 3,
        RayWavefront = 4,
        ReStirDI    = 5
    }

    // Every feature this device can run, constructed from the catalog, wired, initialized,
    // phase-pumped and disposed without this class naming any of them.
    private FeatureHost _features = null!;

    // Exactly one core produces the frame. The ImGui combo lists cores by index and hands one back
    // through RequestCoreIndex; DrawFrame swaps and Activates at the top of the next frame.
    private IRenderCore _activeCore = null!;
    private int         _desiredCoreIndex;

    /// <summary>Technique identity of the active core. Read by the call sites that branch on the
    /// kind of renderer running: the viewport fullscreen gate and the path-tracer settings. Not a
    /// selection knob - a device that gates a core out never registers it, so it cannot be picked.
    /// </summary>
    public RenderMode renderMode => _activeCore?.Mode ?? RenderMode.Deferred;

    /// <summary>The built render cores, in descriptor Order. Drives the ImGui mode combo.</summary>
    internal IReadOnlyList<IRenderCore> RenderCores => _features.Cores;

    /// <summary>List index of the active core, for the mode combo's current selection.</summary>
    internal int ActiveCoreIndex
    {
        get
        {
            var cores = _features.Cores;
            for (int i = 0; i < cores.Count; i++)
                if (ReferenceEquals(cores[i], _activeCore)) return i;
            return -1;
        }
    }

    /// <summary>Requests the core at list index <paramref name="i"/> become active. The swap and
    /// Activate happen after the next DeviceWaitIdle, so this is safe to call mid-frame from the
    /// UI.</summary>
    internal void RequestCoreIndex(int i)
    {
        if (i >= 0 && i < RenderCores.Count) _desiredCoreIndex = i;
    }

    /// <summary>Re-runs the active core's Activate. Hook for a shared-pipeline rebuild: the
    /// pipeline object survives, but the PT cores restart progressive accumulation in Activate and
    /// a new tone curve is a different image.</summary>
    internal void ReactivateCore() => _activeCore.Activate();

    // Bridges onto two features the FeatureHost owns, kept so callers that still name the subsystem
    // work: the settings panel's HDR picker and probe list, and SpawnTestProbe. Assigned once after
    // BuildAll. Each deletes itself when its last caller stops naming the feature.
    internal IblSystem             Ibl                   = null!;
    internal ReflectionProbeSystem reflectionProbeSystem = null!;

    // Null until BuildAll. The scene-AS bridge below can be reached during it, so uses stay guarded.
    private IPathTracingProvider? _ptResources;

    /// <summary>The one IBL scalar the lighting and tracing shaders need as a uniform (roughness to
    /// mip mapping). Resolved through <see cref="IIblProvider"/> at record time, so a rebake cannot
    /// leave a stale copy.</summary>
    internal uint PrefilteredCubeMipLevels => ((IIblProvider)Ibl).PrefilteredCubeMipLevels;

    // Read paths into feature-owned pipelines, for the settings panel. Every one resolves through
    // the FeatureHost and tolerates null, because the owning core may be gated out on this device.
    internal TonemapPipeline?     tonemapPipeline   => _features.Get<SharedPipelines>()?.Tonemap;
    internal PTComputePipeline?   ptComputePipeline => _features.Get<PathtraceComputeCore>()?.Pipeline;
    internal WavefrontPTPipeline? wavefrontPipeline => _features.Get<WavefrontPTCore>()?.Pipeline;

    /// <summary>Current tone-map curve, for the settings panel's combo. Owned by SharedPipelines as
    /// a spec constant on the pipeline it builds.</summary>
    public TonemapOperator tonemapOperator => _features.Get<SharedPipelines>()?.Operator ?? TonemapOperator.Filmic;

    /// <summary>Current soft-shadow state, for the settings panel's checkbox. Owned by DeferredCore
    /// as a spec constant on that core's pipelines.</summary>
    public bool softShadowsEnabled => _features.Get<DeferredCore>()?.SoftShadowsEnabled ?? true;

    //=========================================================================
    //====== Initialization ===================================================
    //=========================================================================

    /// <summary>Brings up the device, the presentation chain, the scene mirror and every feature,
    /// then activates the boot core. Call once, before the first <see cref="Update"/>.</summary>
    public void Initialize()
    {
        // Vulkan RHI context: instance, debug, surface, physical device, logical device, allocator,
        // command pool. Everything below allocates through it.
        gfx = new GraphicsDevice(window!, enableValidationLayers);
        gfx.Initialize();

        swapchain = new Swapchain(gfx, window!);
        swapchain.Create();
        // Boot render extent tracks the swapchain. ViewportPanel shrinks it later through
        // EditorState.RequestedRenderExtent.
        renderTargets = new RenderTargets(gfx);
        renderTargets.SetExtent(swapChainExtent);
        swapchain.CreateImageViews();

        CreateDescriptorPool();

        // Reflects SceneBindings.slang into the canonical scene set layout and allocates the
        // per-frame set instances. Providers register at the end of this method.
        shaderLibrary = ShaderLibrary.CreateDefault();
        descriptorRegistry = new DescriptorRegistry(gfx, shaderLibrary, RenderConfig.MAX_CONCURRENT_FRAMES);

        Gpu = new GpuContext(gfx, descriptorRegistry, shaderLibrary);

        Engine.ResourceManager.Initialize(Gpu);

        renderTargets.AllocateAll();

        scene = new Scene();
        gpuScene = new GpuScene(gfx, scene);
        gpuScene.CreateLightBuffers();
        // Cull-input SSBO. Must exist before DrawCullPipeline binds it at binding 0.
        gpuScene.CreateRenderableBuffers();

        imGuiUtils = new ImGuiVulkanUtils(this, (uint)gfx.QueueFamilyIndices.graphicsFamily!);
        imGuiUtils?.init(swapChainExtent.Width, swapChainExtent.Height);

        // Binds FinalColor into the ImGui viewport descriptor so the Viewport panel can draw the
        // scene as an ImGui.Image. Re-bound on swapchain recreate, which rebuilds the ImageView.
        imGuiUtils?.WriteViewportDescriptor(renderTargets.FinalColor.ImageView);

        frameRing = new FrameRing(gfx, RenderConfig.MAX_CONCURRENT_FRAMES);
        frameRing.CreateCommandBuffers(swapChainImages.Length);
        frameRing.CreateSyncObjects();

        // Stands up every feature the device can run: construct (gated), wire, Initialize, in
        // descriptor Order. Runs ahead of the scene setup and the registry cross-check because IBL
        // and the probes publish their own bindings during Initialize, and the probe registry has
        // to exist before an entity carrying a probe component is spawned. The dump reports what
        // the gates excluded on this device, which no source-order list can.
        _features = new FeatureHost(Gpu, gpuScene, this);
        _features.BuildAll(renderTargets.Snapshot);
        Console.WriteLine(_features.Dump());
        Ibl                   = _features.Get<IblSystem>()!;
        reflectionProbeSystem = _features.Get<ReflectionProbeSystem>()!;
        _ptResources          = _features.Get<IPathTracingProvider>();
        _sceneAs              = _features.Get<SceneAS>();   // null when the device gated it out

        CreateTestEntity();

        // Scene-set registrations for every provider that exists at init, matched by SceneBindings
        // parameter name. Runtime handle changes re-register at their rebuild sites. IBL, the
        // probes and SceneAS are absent here because they registered during BuildAll.
        RegisterSceneBindings();
        Console.WriteLine(descriptorRegistry.DumpBindings());

        // Cross-checks every migrated pipeline's reflected bindings against what the registry owns
        // and was handed. Runs here because it needs both sides complete. Throws on a mismatch.
        Console.WriteLine(descriptorRegistry.Validate(ReflectedPrograms()));
        var lc = gfx.LayoutCache;
        Console.WriteLine($"[layout-cache] set layouts {lc.SetLayoutCount}/{lc.SetLayoutRequests} distinct, " +
                          $"pipeline layouts {lc.PipelineLayoutCount}/{lc.PipelineLayoutRequests} distinct");

        // Boot core is the lowest Order, unless RequestCoreIndex ran before init.
        _activeCore = RenderCores[_desiredCoreIndex];
        _activeCore.Activate();

        initialized = true;
    }

    private void CreateDescriptorPool()
    {
        // Per-frame budget, oversized where the count is hard to pin:
        //   UniformBuffer         GeometryFrameUBO + LightingUBO
        //   StorageBuffer         bindless mat+instance, light SSBO, the four cull buffers, the PBR
        //                         shadow-alpha set, and the wavefront tracer's 25-binding set 4
        //   SampledImage          bindless texture array
        //   Sampler               bindless samplers
        //   CombinedImageSampler  5 g-buffer + 3 IBL samplers, ImGui viewport, and one per probe
        //                         (slot, mip) prefilter set: MaxProbes 16 x MipLevels 9 = 144
        //   AccelerationStructure TLAS for PBR, Transparent, PT, wavefront, pick, selection mask
        //   StorageImage          IBL bake dispatches plus the same 144 probe prefilter sets
        var poolSizes = new DescriptorPoolSize[]
        {
            new() { Type = DescriptorType.UniformBuffer,            DescriptorCount = 24 },
            new() { Type = DescriptorType.StorageBuffer,            DescriptorCount = 128 },
            new() { Type = DescriptorType.SampledImage,             DescriptorCount = RenderConfig.MAX_BINDLESS_TEXTURES * RenderConfig.MAX_CONCURRENT_FRAMES },
            new() { Type = DescriptorType.Sampler,                  DescriptorCount = 8 * RenderConfig.MAX_CONCURRENT_FRAMES + 4 },
            new() { Type = DescriptorType.CombinedImageSampler,     DescriptorCount = 32 + 200 },
            new() { Type = DescriptorType.AccelerationStructureKhr, DescriptorCount = 16 },
            new() { Type = DescriptorType.StorageImage,             DescriptorCount = 24 + 200 },
        };
        // Sizing is an app-level budget; GraphicsDevice owns the pool handle.
        gfx.CreateDescriptorPool(poolSizes, maxSets: 48 + 200 + 16,
            DescriptorPoolCreateFlags.UpdateAfterBindBit | DescriptorPoolCreateFlags.FreeDescriptorSetBit);
    }

    //=========================================================================
    //====== Scene Setup ======================================================
    //=========================================================================

    private void CreateTestEntity()
    {
        SpawnTestLights();
    }

    private void SpawnTestLights()
    {
        // Directional key light, sun-like, pointing down and forward. Position is irrelevant for
        // directional, so the transform stays identity.
        Entity* dirLight = Entity.Create("KeyLight");
        dirLight->AddComponent(new TransformComponent());
        dirLight->AddComponent(new LightComponent
        {
            Type        = LightType.Directional,
            Color       = new Vector3(255f/255, 223f/255, 155f/255),
            Intensity   = 2.0f,
            Direction   = new Vector3(0.45f, -0.9f, 0.25f),
            Radius      = 0.01f,
            CastShadows = true,
        });
        dirLight->Initialize();
        scene.AddEntity(dirLight);

        // Point fill light, warm, right of and above the helmet.
        Entity* pointLight = Entity.Create("PointFill");
        pointLight->AddComponent(new TransformComponent());
        pointLight->GetComponent<TransformComponent>()?.SetPosition(new Vector3(10f, 20f, 0f));
        pointLight->AddComponent(new LightComponent
        {
            Type      = LightType.Point,
            Color     = new Vector3(1.0f, 1.0f, 0.4f),
            Intensity = 5000.0f,
            Range     = 100.0f,
            CastShadows = true,
            Radius = 0.03f,
        });
        pointLight->Initialize();
        // scene.AddEntity(pointLight);

        // Spot rim light, tight cone aimed at the helmet from below-left.
        Entity* spotLight1 = Entity.Create("SpotRim");
        spotLight1->AddComponent(new TransformComponent());
        spotLight1->GetComponent<TransformComponent>()?.SetPosition(new Vector3(0.75f, 0.82f, 2.2f));
        spotLight1->AddComponent(new LightComponent
        {
            Type         = LightType.Point,
            Color        = new Vector3(0.8f, 0.8f, 1.0f),
            Intensity    = 25.0f,
            Range        = 10.0f,
            Direction    = new Vector3(0f, 0f, 1f),
            InnerConeCos = MathF.Cos(MathF.PI / 4f),
            OuterConeCos = MathF.Cos(MathF.PI / 2f),
        });
        spotLight1->Initialize();
        // scene.AddEntity(spotLight1);

        Entity* spotLight2 = Entity.Create("SpotRim2");
        spotLight2->AddComponent(new TransformComponent());
        spotLight2->GetComponent<TransformComponent>()?.SetPosition(new Vector3(-0.75f, 0.82f, 2.2f));
        spotLight2->AddComponent(new LightComponent
        {
            Type = LightType.Point,
            Color = new Vector3(0.8f, 0.8f, 1.0f),
            Intensity = 25.0f,
            Range = 10.0f,
            CastShadows = true,
        });
        spotLight2->Initialize();
        // scene.AddEntity(spotLight2);
    }

    //=========================================================================
    //====== Descriptor Registration ==========================================
    //=========================================================================

    // Registers the current owners' handles into the unified scene set by SceneBindings parameter
    // name. Idempotent: registrations are queued as fence-safe rewrites.
    private void RegisterSceneBindings()
    {
        var rm = Engine.ResourceManager;
        var materials = new Buffer[RenderConfig.MAX_CONCURRENT_FRAMES];
        var instances = new Buffer[RenderConfig.MAX_CONCURRENT_FRAMES];
        var lights = new Buffer[RenderConfig.MAX_CONCURRENT_FRAMES];
        for (uint i = 0; i < RenderConfig.MAX_CONCURRENT_FRAMES; i++)
        {
            materials[i] = rm.GetMaterialBuffer((int)i);
            instances[i] = rm.GetInstanceBuffer(i);
            lights[i] = gpuScene.GetLightStorageBuffer(i);
        }
        descriptorRegistry.RegisterBufferPerFrame("sceneMaterials", materials);
        descriptorRegistry.RegisterBufferPerFrame("sceneInstances", instances);
        descriptorRegistry.RegisterBufferPerFrame("sceneLights", lights);
        descriptorRegistry.RegisterBuffer("sceneVertices", rm.GlobalVertexBuffer);
        descriptorRegistry.RegisterBuffer("sceneIndices", rm.GlobalIndexBuffer);
        // Engine shaders only index samplers[0]; fill the whole array with the default sampler so
        // no element of the non-PartiallyBound binding is left invalid.
        for (int s = 0; s < 16; s++)
            descriptorRegistry.RegisterSampler("sceneSamplers", rm.DefaultSampler, s);
    }

    // Every reflected program the pipelines resolved, for the registry cross-check. Walks
    // PipelineBase-typed fields on this object and on every built feature, so a new pipeline joins
    // the check by existing and migrating one into its core cannot drop it out.
    private IEnumerable<ProgramUse> ReflectedPrograms()
        => new object[] { this }.Concat(_features.All)
            .SelectMany(PipelinesOn)
            .Where(p => p?.ReflectedProgram != null)
            .Select(p => new ProgramUse(p!.ReflectedProgram!, p.PrivateSetIndices))
            .DistinctBy(u => u.Program);

    // The PipelineBase-typed instance fields of one object, at any access level. Fields only: a
    // property could resolve through the feature host and re-enter the walk.
    private static IEnumerable<PipelineBase?> PipelinesOn(object owner)
        => owner.GetType()
            .GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)
            .Where(f => typeof(PipelineBase).IsAssignableFrom(f.FieldType))
            .Select(f => f.GetValue(owner) as PipelineBase);

    //=========================================================================
    //====== Frame ============================================================
    //=========================================================================

    /// <summary>Engine update hook. Draws exactly one frame.</summary>
    /// <param name="d">Delta time in seconds. Unused: the frame skeleton is not time-stepped.</param>
    public void Update(double d)
    {
        DrawFrame();
    }

    /// <summary>Records and submits one frame: extract, pump the features, dispatch the active
    /// core, blit FinalColor to the swapchain, overlay the UI, present.</summary>
    public void DrawFrame()
    {
        var currentFrame = frameRing.CurrentFrame;
        var imageAvailableSemaphore = frameRing.ImageAvailableSemaphores[currentFrame];
        var renderFinishedSemaphore = frameRing.RenderFinishedSemaphores[currentFrame];
        var graphicsCmds = frameRing.CommandBuffers[currentFrame];
        var inFlightFence = frameRing.InFlightFences[currentFrame];

        // 0. Apply a pending render-extent request from the ViewportPanel. Requests are one frame
        //    stale, since the panel sizes itself during the previous frame's DrawFrame.
        //    ResizeRenderTargets no-ops when the request matches the current extent.
        if (ImGui.EditorState.RequestedRenderExtent is var req && req.HasValue)
        {
            ResizeRenderTargets(req.Value.w, req.Value.h);
            ImGui.EditorState.RequestedRenderExtent = null;
        }

        // 0b. Service any feature whose baked content went stale: the scene acceleration structure
        //     after an editor mutation, the shared tonemap after a curve change, the deferred PBR
        //     pipelines after a soft-shadow toggle. The only place a pipeline is torn down and
        //     rebuilt, and it runs before recording starts, because mid-frame disposal of a bound
        //     pipeline corrupts the in-flight command buffer's references. A feature with nothing
        //     pending costs one bool test, and however many edits landed last frame collapse into
        //     a single rebuild.
        _features.ServiceBakes();

        // 0c. Render-mode change. Tonemap's HDR input is graph-baked per core, so Activate only
        //     restarts the PT cores' progressive accumulation. DeviceWaitIdle keeps an in-flight
        //     frame from straddling the switch; the hitch is acceptable on a user-driven flip.
        var desiredCore = RenderCores[_desiredCoreIndex];
        if (!ReferenceEquals(desiredCore, _activeCore))
        {
            vk!.DeviceWaitIdle(device);
            _activeCore = desiredCore;
            _activeCore.Activate();
        }

        // 1. CPU/GPU sync for this slot.
        vk!.WaitForFences(device, 1, ref inFlightFence, true, ulong.MaxValue);

        // Apply queued registry rewrites.
        descriptorRegistry.BeginFrame(currentFrame);

        // 2. Acquire a swapchain image.
        var acquireResult = swapchain.AcquireNextImage(imageAvailableSemaphore, out var imageIndex);
        if (acquireResult == Result.ErrorOutOfDateKhr) { RecreateSwapChain(); return; }

        // 3. Reset the fence: the submission below signals it.
        vk!.ResetFences(device, 1, ref inFlightFence);

        // 4. Reset and begin the command buffer.
        vk!.ResetCommandBuffer(graphicsCmds, 0);
        var beginInfo = new CommandBufferBeginInfo
        {
            SType = StructureType.CommandBufferBeginInfo,
            Flags = CommandBufferUsageFlags.OneTimeSubmitBit,
        };
        if (vk!.BeginCommandBuffer(graphicsCmds, &beginInfo) != Result.Success)
            throw new Exception("Failed to begin command buffer");

        // 5. Scene extraction, once per frame. Cores consume the result and never re-trigger it.
        gpuScene.BeginTransforms();
        uint materialCount   = gpuScene.UpdateMaterials(currentFrame);
        uint lightCount      = gpuScene.UpdateLights(currentFrame);
        uint renderableCount = gpuScene.ExtractRenderables(currentFrame);

        var view = new RenderView
        {
            FrameIndex            = currentFrame,
            FrameCounter          = frameRing.FrameCounter,
            Camera                = camera,
            Scene                 = scene,
            RenderExtent          = renderExtent,
            RenderableCount       = renderableCount,
            LightCount            = lightCount,
            MaterialCount         = materialCount,
            TransparentCandidates = gpuScene.TransparentCandidates,
        };

        // 6. Fixed pump points around the technique. A feature slots into the frame by implementing
        //    the matching interface; this sequence never grows an entry per feature.
        _features.PreDraw(view);

        // The active core records its technique and leaves FinalColor in ShaderReadOnlyOptimal.
        _activeCore.Render(new RenderFrame { Cmd = graphicsCmds, View = view });

        _features.PostDraw(graphicsCmds, view);

        // 7. Blit FinalColor onto the swapchain image.
        var swapImage = swapChainImages[imageIndex];

        // 7a. Swapchain Undefined to TransferDstOptimal.
        var toTransferDst = new ImageMemoryBarrier
        {
            SType = StructureType.ImageMemoryBarrier,
            OldLayout = ImageLayout.Undefined,
            NewLayout = ImageLayout.TransferDstOptimal,
            SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
            Image = swapImage,
            SrcAccessMask = 0,
            DstAccessMask = AccessFlags.TransferWriteBit,
            SubresourceRange = new ImageSubresourceRange(ImageAspectFlags.ColorBit, 0, 1, 0, 1),
        };
        vk!.CmdPipelineBarrier(graphicsCmds,
            PipelineStageFlags.TopOfPipeBit,
            PipelineStageFlags.TransferBit,
            0, 0, null, 0, null, 1, &toTransferDst);

        // 7b. FinalColor's graph-declared finalLayout is ShaderReadOnlyOptimal so the viewport panel
        //     can sample it, so the blit walks it through TransferSrcOptimal and back. Ending where
        //     the graph expects keeps its layout tracker consistent.
        var finalColor = renderTargets.FinalColor;
        gfx.TransitionImageLayout(graphicsCmds, finalColor.Image, finalColor._format,
            ImageLayout.ShaderReadOnlyOptimal, ImageLayout.TransferSrcOptimal);
        var blit = new ImageBlit
        {
            SrcSubresource = new ImageSubresourceLayers
            {
                AspectMask = ImageAspectFlags.ColorBit,
                MipLevel = 0,
                BaseArrayLayer = 0,
                LayerCount = 1,
            },
            DstSubresource = new ImageSubresourceLayers
            {
                AspectMask = ImageAspectFlags.ColorBit,
                MipLevel = 0,
                BaseArrayLayer = 0,
                LayerCount = 1,
            },
        };
        // Source is FinalColor at renderExtent, destination the swapchain image at swapChainExtent.
        // CmdBlitImage scales when they differ.
        blit.SrcOffsets[0] = new Offset3D(0, 0, 0);
        blit.SrcOffsets[1] = new Offset3D((int)renderExtent.Width, (int)renderExtent.Height, 1);
        blit.DstOffsets[0] = new Offset3D(0, 0, 0);
        blit.DstOffsets[1] = new Offset3D((int)swapChainExtent.Width, (int)swapChainExtent.Height, 1);

        vk!.CmdBlitImage(graphicsCmds,
            finalColor.Image, ImageLayout.TransferSrcOptimal,
            swapImage, ImageLayout.TransferDstOptimal,
            1, &blit, Filter.Linear);

        gfx.TransitionImageLayout(graphicsCmds, finalColor.Image, finalColor._format,
            ImageLayout.TransferSrcOptimal, ImageLayout.ShaderReadOnlyOptimal);

        // 7c. Swapchain TransferDstOptimal to ColorAttachmentOptimal so the UI overlay can draw on
        //     top of the blitted scene.
        gfx.TransitionImageLayout(graphicsCmds, swapImage, swapChainImageFormat,
            ImageLayout.TransferDstOptimal, ImageLayout.ColorAttachmentOptimal);

        // 7d. UI overlay. No-op when imGuiUtils is unwired.
        if (imGuiUtils != null)
        {
            imGuiUtils.newFrame();
            imGuiUtils.updateBuffers(currentFrame);
        }
        imGuiUtils?.DrawFrame(graphicsCmds, swapChainImageViews[imageIndex], currentFrame);

        // 7e. Swapchain ColorAttachmentOptimal to PresentSrcKhr.
        gfx.TransitionImageLayout(graphicsCmds, swapImage, swapChainImageFormat,
            ImageLayout.ColorAttachmentOptimal, ImageLayout.PresentSrcKhr);

        // 8. End the command buffer.
        if (vk!.EndCommandBuffer(graphicsCmds) != Result.Success)
            throw new Exception("Failed to end command buffer");

        // 9. Submit through vkQueueSubmit2. When the active core has deferred graphics chunks (the
        //    wavefront async path), SubmitGfxChunks merges them with the host command buffer so all
        //    graphics work lands in one submission and every graph pass shows in Nsight's timeline.
        var imgAvailWait = new SemaphoreSubmitInfo
        {
            SType     = StructureType.SemaphoreSubmitInfo,
            Semaphore = imageAvailableSemaphore,
            Value     = 0,
            StageMask = PipelineStageFlags2.TransferBit,
        };
        var renderDoneSignal = new SemaphoreSubmitInfo
        {
            SType     = StructureType.SemaphoreSubmitInfo,
            Semaphore = renderFinishedSemaphore,
            Value     = 0,
            StageMask = PipelineStageFlags2.AllCommandsBit,
        };

        if (ActiveGraphCore is { HasPendingGfxChunks: true } graphCore)
        {
            graphCore.SubmitGfxChunks(gfx.GraphicsQueue, imgAvailWait, renderDoneSignal,
                graphicsCmds, inFlightFence);
        }
        else
        {
            var hostCmdInfo = new CommandBufferSubmitInfo
            {
                SType = StructureType.CommandBufferSubmitInfo,
                CommandBuffer = graphicsCmds,
            };
            var submitInfo2 = new SubmitInfo2
            {
                SType                    = StructureType.SubmitInfo2,
                WaitSemaphoreInfoCount   = 1,
                PWaitSemaphoreInfos      = &imgAvailWait,
                CommandBufferInfoCount   = 1,
                PCommandBufferInfos      = &hostCmdInfo,
                SignalSemaphoreInfoCount = 1,
                PSignalSemaphoreInfos    = &renderDoneSignal,
            };
            if (vk!.QueueSubmit2(gfx.GraphicsQueue, 1, &submitInfo2, inFlightFence) != Result.Success)
                throw new Exception("Queue submit failed");
        }

        // 10. Present.
        swapchain.Present(gfx.PresentQueue, renderFinishedSemaphore, imageIndex);

        frameRing.Advance();
    }

    //=========================================================================
    //====== Resize ===========================================================
    //=========================================================================

    /// <summary>Rebuilds everything attached to the surface extent. Blocks while the window is
    /// minimized, since a zero framebuffer cannot size a swapchain.</summary>
    public void RecreateSwapChain()
    {
        // DoEvents pumps the message loop so a restore still gets through.
        var fb = window!.FramebufferSize;
        while (fb.X == 0 || fb.Y == 0)
        {
            window.DoEvents();
            fb = window.FramebufferSize;
        }
        vk!.DeviceWaitIdle(device);

        swapchain.Recreate();

        // Bring the viewport panel back to the swapchain extent. The panel drives it back down on
        // its next frame.
        RebuildRenderTargets(swapChainExtent.Width, swapChainExtent.Height);
        imGuiUtils?.UpdateScreenSize(swapChainExtent.Width, swapChainExtent.Height);
    }

    /// <summary>Blocks on vkDeviceWaitIdle and reallocates the render targets at the requested
    /// extent. Safe to call from outside the renderer, which is how ViewportPanel resizes.</summary>
    public void ResizeRenderTargets(uint width, uint height)
    {
        if (width == 0 || height == 0) return;
        if (width == renderExtent.Width && height == renderExtent.Height) return;

        vk!.DeviceWaitIdle(device);
        RebuildRenderTargets(width, height);
    }

    // Reallocates the size-dependent targets, then pumps every resize feature. The caller must have
    // idled the device: none of the disposed resources can be in flight.
    private void RebuildRenderTargets(uint width, uint height)
    {
        // Counted so the Stats panel can correlate spp/s degradation with resize churn.
        _renderTargetRebuilds++;

        // Targets first, since every core imports the fresh handles from the snapshot below.
        renderTargets.ReallocateSizeDependent(new Extent2D(width, height));

        // Each resize feature rebuilds its size-dependent state, in Order: PathTracingSystem
        // reallocates the accumulator and out-color pair and re-registers the FeaturePTIO bindings,
        // then each core recompiles its graph against the fresh handles, then Selection reallocates
        // its coverage mask.
        _features.Resize(renderTargets.Snapshot);

        // Tonemap's HDR input is graph-baked and was re-pointed by the core rebuilds above.
        // Activate restarts the active PT core's accumulation on the fresh targets.
        _activeCore.Activate();

        imGuiUtils?.WriteViewportDescriptor(renderTargets.FinalColor.ImageView);

        // Samples allocator occupancy now that old targets are freed and new ones allocated, which
        // is the steady post-resize state the history plot wants.
        RecordMemorySample();
    }

    //=========================================================================
    //====== Scene Acceleration Bridge ========================================
    //=========================================================================

    // Forwarders onto the gated SceneAS feature, which owns clustering, rebuild cadence, and the
    // instance / shadow-info / emissive packing. The editor panels and the path-trace pipelines
    // still name the Renderer, so their calls land here.
    //
    // Every member tolerates a null feature: a device without ray query gates SceneAS out, which
    // leaves a scene with no ray infrastructure at all.
    private SceneAS? _sceneAs;

    /// <summary>Synchronous counterpart of <c>SceneDirtyEvent</c>. The asset paths (scene load,
    /// mesh destroy) want the new geometry traceable before they return, so this fans out to the
    /// three subscribers directly instead of publishing and waiting for the next drain.</summary>
    public void OnSceneEntitiesChanged()
    {
        if (!initialized) return;

        gpuScene.MarkSceneDirty();
        _ptResources?.MarkAccumulatorDirty();
        // Null on a device with no ray infrastructure, where the two marks above are the whole of
        // the invalidation.
        _sceneAs?.Rebuild();
    }

    /// <summary>Pairs with file destroy in the editor. Cluster BLASes are world-space and rebuilt
    /// wholesale, so there is nothing mesh-keyed to free here.</summary>
    public void DestroyBlasFor(IEnumerable<nint> meshPtrs)
    {
        _ = meshPtrs;
        Engine.EventBus.PublishEvent(new SceneDirtyEvent());
    }

    //=========================================================================
    //====== Buffers =========================================================
    //=========================================================================

    /// <summary>Allocates a buffer through the device allocator. Forwards to
    /// <see cref="GraphicsDevice"/>.</summary>
    public void CreateBuffer(ulong size, BufferUsageFlags usage, MemoryPropertyFlags memProps,
        out Buffer buffer, out SubAlloc alloc, float priority = GpuMemoryAllocator.PriorityDefault,
        bool preferDeviceLocal = false)
        => gfx.CreateBuffer(size, usage, memProps, out buffer, out alloc, priority, preferDeviceLocal);

    /// <summary>Destroys the handle and releases its suballocation together.</summary>
    public void DestroyBuffer(Buffer buffer, SubAlloc alloc) => gfx.DestroyBuffer(buffer, alloc);

    //=========================================================================
    //====== Diagnostics ======================================================
    //=========================================================================

    // The frame graph belongs to whichever core is active, so these forward through IGraphCore.
    // Cores with no graph (megakernel PT, forward+) are not IGraphCore, and the panel shows nothing.
    private IGraphCore? ActiveGraphCore => _activeCore as IGraphCore;

    /// <summary>Last-frame per-pass GPU/CPU timings and counts, or null when the active core has no
    /// graph.</summary>
    public GraphStats? ActiveGraphStats => ActiveGraphCore?.GraphStats;

    /// <summary>The active core's display name, for the panel header.</summary>
    public string ActiveCoreName => _activeCore?.Name ?? "";

    /// <summary>Graphviz dump of the active core's compiled graph.</summary>
    public string ActiveGraphDot() => ActiveGraphCore?.ToDot() ?? "(active core has no frame graph)";

    /// <summary>Runtime toggle for the active core's graph pipeline-statistics collection.</summary>
    public bool ActiveGraphPipelineStats
    {
        get => ActiveGraphCore?.CollectPipelineStats ?? false;
        set { if (ActiveGraphCore is { } g) g.CollectPipelineStats = value; }
    }

    private int _renderTargetRebuilds;
    /// <summary>Full render-target rebuilds since launch, one per resize.</summary>
    public int RenderTargetRebuilds => _renderTargetRebuilds;

    /// <summary>Live allocator occupancy: reserved (held from the driver) against actually used.
    /// </summary>
    public AllocatorStats GpuMemoryStats => gfx.Allocator.GetStats();

    /// <summary>Driver-reported WDDM budget and usage for the device-local heap, or
    /// Available=false when VK_EXT_memory_budget is not enabled. This is the number the hand-rolled
    /// reserved/used counters approximate, and it includes the OS budget the committed allocations
    /// are racing against.</summary>
    public MemoryBudgetInfo GpuMemoryBudget => gfx.QueryMemoryBudget();

    // Per-rebuild MB history, sampled at the end of RebuildRenderTargets. The shape discriminates
    // the diagnosis: a monotonic climb with rebuild count is a per-resize leak, a step-then-plateau
    // is a one-time high-water mark from a retained empty block.
    private const int MemHistoryLen = 256;
    private readonly float[] _usedMbHistory     = new float[MemHistoryLen];
    private readonly float[] _reservedMbHistory = new float[MemHistoryLen];
    private int _memHistoryHead;
    public float[] UsedMbHistory     => _usedMbHistory;
    public float[] ReservedMbHistory => _reservedMbHistory;
    public int     MemHistoryHead    => _memHistoryHead;
    public int     MemHistoryLength  => MemHistoryLen;

    private void RecordMemorySample()
    {
        var s = gfx.Allocator.GetStats();
        const float MB = 1024f * 1024f;
        _usedMbHistory[_memHistoryHead]     = s.UsedBytes     / MB;
        _reservedMbHistory[_memHistoryHead] = s.ReservedBytes / MB;
        _memHistoryHead = (_memHistoryHead + 1) % MemHistoryLen;
    }

    /// <summary>Per-bounce [extend, shade, connect] indirect workgroup counts for the wavefront
    /// tracer, or null when wavefront is not the active core. Roughly a frame stale, best-effort.
    /// </summary>
    // TODO: remove alongside the pipeline's _argsReadback readback path.
    public uint[]? WavefrontDispatchCounts =>
        _activeCore is WavefrontPTCore && wavefrontPipeline != null
            ? wavefrontPipeline.ReadDispatchArgs() : null;

    //=========================================================================
    //====== Teardown =========================================================
    //=========================================================================

    /// <summary>Drains the GPU, then destroys every Vulkan handle and unmanaged resource the
    /// renderer owns, in reverse creation order.</summary>
    public void Cleanup()
    {
        if (!initialized) return;

        // Drain GPU work so nothing references resources about to be destroyed.
        vk!.DeviceWaitIdle(device);

        frameRing.Dispose();

        // Scene set and constant arena, then the shader library, which drops slang.dll if loaded.
        descriptorRegistry.Dispose();
        shaderLibrary.Dispose();

        if (testEntity != null)
        {
            Entity.Destroy(testEntity);
            testEntity = null;
        }

        // Mesh pool: the global vertex and index buffers.
        Engine.ResourceManager.Dispose();

        imGuiUtils?.Dispose();

        // Every feature, in reverse Order.
        _features.Dispose();
        renderTargets.Dispose();
        gpuScene.Dispose();
        swapchain.Dispose();

        // RHI context, strictly last.
        gfx.Dispose();

        initialized = false;
    }
}