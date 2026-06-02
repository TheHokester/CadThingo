using System.Numerics;
using CadThingo.Graphics.Rendering;
using CadThingo.VulkanEngine.ImGui;
using CadThingo.VulkanEngine.Renderer.Pipelines;
using Silk.NET.Core;
using Silk.NET.Core.Native;
using Silk.NET.Vulkan;
using Silk.NET.Vulkan.Extensions.EXT;
using Silk.NET.Vulkan.Extensions.KHR;
using Silk.NET.Windowing;
// ReSharper disable InconsistentNaming

namespace CadThingo.VulkanEngine.Renderer;

public unsafe partial class Renderer
{
    public Renderer(IWindow window)
    {
        this.window = window;
        camera = new Camera();
    }
    /// <summary>
    /// Renderer fields
    /// </summary>
    // The Vulkan RHI context — instance, device, queues, allocator, descriptor /
    // command pools, capability flags, and the device-service helpers. Owned for the
    // app's lifetime; constructed first in Initialize() and disposed strictly last in
    // Cleanup(). Renderer exposes the same-named delegating accessors below so the rest
    // of the Renderer partials and every pipeline / ImageResource / ResourceManager
    // call site keep reaching device services unchanged while the deeper L1/L2/L3
    // repoints land (L1 of the renderer refactor).
    internal GraphicsDevice gfx = null!;

    public bool initialized = false;

    // Config input for the GraphicsDevice (instance validation layers). Lives here so
    // the toggle stays next to the renderer; passed to the GraphicsDevice constructor.
    private bool enableValidationLayers = false;
    private IWindow? window;

    // ---- Delegating accessors onto GraphicsDevice (transitional) -----------
    internal Vk?               vk             => gfx?.Vk;
    internal Device            device         => gfx.Device;
    internal PhysicalDevice    physicalDevice => gfx.PhysicalDevice;
    internal GpuMemoryAllocator memAllocator  => gfx.Allocator;
    internal DescriptorPool    descriptorPool => gfx.DescriptorPool;
    internal CommandPool       commandPool    => gfx.CommandPool;
    internal Queue             graphicsQueue  => gfx.GraphicsQueue;
    internal Queue             presentQueue   => gfx.PresentQueue;
    internal Queue             computeQueue   => gfx.ComputeQueue;
    internal Queue             transferQueue  => gfx.TransferQueue;
    private  Instance          instance       => gfx.Instance;
    private  SurfaceKHR        surface        => gfx.Surface;
    private  QueueFamilyIndices queueFamilyIndices => gfx.QueueFamilyIndices;

    internal bool descriptorIndexEnabled => gfx.DescriptorIndexingEnabled;
    internal bool multiviewEnabled       => gfx.MultiviewEnabled;

    // Read-only handle accessor — pipelines that load their own device-extension
    // dispatch tables (e.g. RtPipeline → KhrRayTracingPipeline) need the instance
    // for vk.TryGetDeviceExtension. Generic, not tied to any one extension.
    internal Instance GetVkInstance() => gfx.GetVkInstance();

    public enum RenderMode : uint
    {
        Deferred = 0,
        ForwardPlus = 1,
        RayCompute = 2,
        RayTrace =3
    }
    public RenderMode renderMode = RenderMode.Deferred;
    // Tracks the last mode DrawFrame actually rendered. When this differs from
    // `renderMode` at the top of a frame, tonemap's HDR-input descriptor is
    // rebound to the appropriate source (HDRColor vs ptOutColor) under
    // DeviceWaitIdle. Initialized to the same default so the first frame
    // doesn't trigger a spurious rebind.
    private RenderMode _lastRenderMode = RenderMode.Deferred;
    
    
    // Runtime reflection probes — GPU resources + CPU registry. Allocated after
    // the IBL cubemaps because it reuses the same prefilter pipeline. Phase 2
    // stops at resource allocation; capture / shader integration come later.
    internal ReflectionProbeSystem reflectionProbeSystem = null!;

    //world scene
    private Scene scene;
    public Scene Scene => scene;
    private Entity* testEntity;
    
    /// <summary>True iff both KHR_acceleration_structure and KHR_ray_query are enabled.
    /// Gates the ray-traced shadow path. Forwards to <see cref="GraphicsDevice"/>.</summary>
    public bool RayShadowsSupported => gfx.RayShadowsSupported;

    /// <summary>True iff KHR_ray_tracing_pipeline is enabled alongside
    /// KHR_acceleration_structure. Gates the opt-in RT-pipeline path tracer.</summary>
    public bool RayTracePipelineSupported => gfx.RayTracePipelineSupported;

    /// <summary>True iff VK_NV_ray_tracing_invocation_reorder (SER) is enabled on top of
    /// the RT pipeline. Selects the HitObject/ReorderThread raygen variant.</summary>
    public bool SerSupported => gfx.SerSupported;


    //swapchain fields
    internal const uint MAX_CONCURRENT_FRAMES = 2;
    
    private KhrSwapchain swapChainKhr;
    private SwapchainKHR swapChain;
    private Image[] swapChainImages;
    internal Format swapChainImageFormat;
    internal Extent2D swapChainExtent;

    // Render target extent — the size at which the deferred chain (gbuffers,
    // HDRColor, FinalColor) and the lighting tile grid are sized. Distinct from
    // swapChainExtent because the editor's viewport panel can be smaller than
    // the OS window. Initialized to swapChainExtent and tracks it until the
    // viewport panel posts a different requested size via EditorState.
    internal Extent2D renderExtent;
    public Extent2D RenderExtent => renderExtent;
    internal ImageView[] swapChainImageViews;
    private ImageLayout[] swapChainImageLayouts;

    // UI overlay drawn after the FinalColor blit, before the present transition.
    // Lifetime/instantiation owned externally; null when UI is disabled.
    public ImGuiVulkanUtils? imGuiUtils;
    
    //dynamic rendering fields
    RenderingInfo renderingInfo;
    List<RenderingAttachmentInfo> colorAttachments;
    RenderingAttachmentInfo depthAttachment;
    
    
    //pipelines — each owns its own VkPipeline, layouts, descriptor sets, and per-pipeline buffers
    internal GeometryPipeline     geometryPipeline;
    internal DrawCullPipeline     drawCullPipeline;
    internal LightCullPipeline    lightCullPipeline;
    internal PbrDeferredPipeline  PbrDeferredPipeline;   // accessor used by LightCullPipeline (consumer of the lights SSBO)
    internal TonemapPipeline      tonemapPipeline;       // post-process: HDRColor → FinalColor
    internal TransparentPipeline  transparentPipeline;   // forward+ BLEND-mode pass between lighting and tonemap
    internal SkyboxPipeline       skyboxPipeline;        // background env-cube draw, between lighting and transparent
    internal PTComputePipeline    ptComputePipeline;
    internal RTPipeline?          rtPipeline;            // opt-in RT-pipeline path tracer (null when unsupported)
    internal PickPipeline         pickPipeline;          // ray-query object picking (TLAS InstanceCustomIndex → entity)
    internal SelectionMaskPipeline selectionMaskPipeline; // ray-query coverage mask of the selected entity
    internal OutlinePipeline       outlinePipeline;       // composites the selection outline into FinalColor

    // Specialization-constant gate for soft (PCSS-style) ray-queried shadows.
    // Threaded into PbrDeferredPipeline.SoftShadowsEnabled at construction time.
    public bool softShadowsEnabled = true;

    // Pending rebuild flags posted by the Renderer Settings panel. Consumed at
    // the top of DrawFrame so the rebuild never races a command buffer that
    // already bound the old pipeline.
    internal bool pendingPbrRebuild     = false;
    internal bool pendingTonemapRebuild = false;

    // Tone-map curve selector — threaded into TonemapPipeline.Operator at
    // construction time as a specialization constant. Toggling requires a
    // pipeline rebuild.
    public TonemapOperator tonemapOperator = TonemapOperator.Filmic;
    
    //Command buffers (the pool lives on GraphicsDevice)
    CommandBuffer[] commandBuffers;
    
    //Camera
    Camera camera;
    public Camera Camera => camera;

    //Sync objects
    Semaphore[] imageAvailableSemaphores;
    Semaphore[] renderFinishedSemaphores;
    Fence[] inFlightFences;
    uint currentFrame;
    // Monotonic frame counter — incremented once per DrawFrame. Distinct from
    // currentFrame which cycles 0..MAX_CONCURRENT_FRAMES-1. Used by the probe
    // scheduler for EveryNFrames bookkeeping and capture timing.
    internal ulong frameCounter;
    
    //upload timeline semaphore
    Semaphore uploadsTimeline;
    //tracks last timeline value that was submitted
    volatile uint lastTimelineValue;
    
    //Depth buffer + Images — shared across passes via the render graph and the
    //PBR pipeline's g-buffer sampler set. Internal so PbrDeferredPipeline.WriteGBufferDescriptors
    //can rebind ImageViews after swapchain recreate.
    ImageResource depthImageResource;
    internal ImageResource gBufferPosition;
    internal ImageResource gBufferNormal;
    internal ImageResource gBufferAlbedo;
    internal ImageResource gBufferMaterial;
    internal ImageResource gBufferEmissive;

    internal Sampler gBufferSampler;
    
    //ImageResources for Path tracing
    bool accumulatorDirty = false;
    public void MarkAccumulatorDirty() => accumulatorDirty = true;
    private ImageResource ptAccumulator;
    private ImageResource ptOutColor;

    // R32F coverage mask of the selected entity, written by the ray-query
    // SelectionMaskPipeline and read by the OutlinePipeline. Renderer-owned,
    // recreated on resize. Sits in ShaderReadOnly between frames.
    private ImageResource selectionMask;

    // Previous-frame camera snapshot. DrawPathtraced compares against this and
    // flips accumulatorDirty when either differs, so any camera move restarts
    // progressive integration. Identity / zero defaults mean the first PT
    // frame after switching modes always restarts (which is what we want).
    private Matrix4x4 _ptLastCameraView = Matrix4x4.Identity;
    private Vector3   _ptLastCameraPos  = Vector3.Zero;
    // FOV doesn't move the view matrix, so it needs its own snapshot — otherwise
    // a FOV edit (camera panel or PT settings) wouldn't restart accumulation.
    private float     _ptLastCameraFov  = 0f;
    
    
    
    public void Initialize()
    {
        // Bring up the Vulkan RHI context (instance → debug → surface → physical →
        // logical device → memory allocator → command pool). Everything below
        // allocates through it.
        gfx = new GraphicsDevice(window!, enableValidationLayers);
        gfx.Initialize();

        CreateSwapChain();
        // Initial render extent tracks swapchain extent. ViewportPanel can later
        // shrink it via EditorState.RequestedRenderExtent → ResizeRenderTargets.
        renderExtent = swapChainExtent;
        CreateImageViews();
        SetupDynamicRendering();

        CreateDescriptorPool();
        Engine.ResourceManager.Initialize(this);

        // ImageResource objects only — actual VkImage + ImageView allocation happens
        // inside graph.Compile() called from SetupDeferredRenderer below.
        CreateDepthResources();
        CreateGBufferResources();
        CreateGBufferSampler();
        CreatePathTracingResources(renderExtent.Width, renderExtent.Height);
        CreateSelectionResources(renderExtent.Width, renderExtent.Height);
        // Lights SSBO lives on Renderer so every rendering path (deferred,
        // forward+, pathtracer) reads from the same buffer. Must exist before
        // any pipeline that wants to bind it.
        CreateLightBuffers();

        // IBL images allocated up-front, cleared to black. The PBR lighting set
        // binds them unconditionally; the compute bake passes fill the content
        // when an HDR is loaded via LoadEnvironmentHdr.
        CreateIblResources();
        CreateIblBakePipelines();
        // BRDF LUT is view-independent — bake once at init and reuse for every
        // environment that gets loaded later.
        BakeBrdfLut();

        // Reflection-probe GPU resources (cubemap array + per-probe SSBO).
        // Reuses the IBL prefilter pipeline at capture time, so it has to be
        // constructed after CreateIblBakePipelines.
        reflectionProbeSystem = new ReflectionProbeSystem(this);

        // Pipelines that don't depend on allocated g-buffer image views 
        // Render-graph pass closures (registered in SetupDeferredRenderer) read
        // `geometryPipeline` / `drawCullPipeline` / `PbrDeferredPipeline` through
        // `this`, so they must exist (be non-null) by the time the closures *run*
        // (per frame in DrawFrame), not when they're declared.
        geometryPipeline = new GeometryPipeline(this);
        geometryPipeline.Initialize();

        drawCullPipeline = new DrawCullPipeline(this);
        drawCullPipeline.Initialize();

        scene = new Scene(vk, device, physicalDevice);//initialise scene
        // graph.Compile() inside SetupDeferredRenderer allocates the g-buffer
        // images. The PBR pipeline's set 1 binds those views, so PbrDeferredPipeline
        // must initialize AFTER this call.
        SetupDeferredRenderer(scene.renderGraph, renderExtent.Width, renderExtent.Height);
        imGuiUtils = new ImGuiVulkanUtils(this, (uint)queueFamilyIndices.graphicsFamily! );
        imGuiUtils?.init(swapChainExtent.Width, swapChainExtent.Height);

        // Bind FinalColor into the ImGui viewport descriptor so the Viewport panel
        // can render the scene as an ImGui.Image. Re-bound on swapchain recreate
        // because the underlying ImageView is rebuilt.
        imGuiUtils?.WriteViewportDescriptor(scene.renderGraph.GetResource("FinalColor").ImageView);

        //  Lighting + light-cull pipelines (depend on allocated g-buffer / lights SSBO) 
        PbrDeferredPipeline = new PbrDeferredPipeline(this) { SoftShadowsEnabled = softShadowsEnabled };
        PbrDeferredPipeline.Initialize();

        lightCullPipeline = new LightCullPipeline(this);
        lightCullPipeline.Initialize();
        
        // Wire light-cull tile buffers into the PBR lighting set now that both exist.
        PbrDeferredPipeline.WriteTileBufferDescriptors(lightCullPipeline);
        // Same idea for the reflection-probe bindings — the cube array, probe
        // records SSBO, cluster range / index list SSBOs are all stable for the
        // renderer's lifetime so we write once at init.
        PbrDeferredPipeline.WriteProbeDescriptors();
        
        
        ptComputePipeline = new PTComputePipeline(this);
        ptComputePipeline.Initialize();
        ptComputePipeline.WriteStorageImageDescriptors(ptAccumulator.ImageView, ptOutColor.ImageView);
        ptComputePipeline.WriteGeometryDescriptors();
        ptComputePipeline.WriteLightsDescriptor();
        // Same as the transparent / deferred pipelines: IBL images are Renderer-
        // owned and stable across rebakes, so set 3 only needs writing once.
        ptComputePipeline.WriteIblDescriptors();

        // Opt-in RT-pipeline path tracer (RenderMode.RayTrace). Shares the same
        // accumulator/outColor images + scene buffers as the compute path; only
        // built when the device exposes the feature. TLAS/shadow/emissive sets
        // are bound below after InitRayQuery, mirroring the compute pipeline.
        if (RayTracePipelineSupported)
        {
            rtPipeline = new RTPipeline(this);
            rtPipeline.Initialize();
            rtPipeline.WriteStorageImageDescriptors(ptAccumulator.ImageView, ptOutColor.ImageView);
            rtPipeline.WriteGeometryDescriptors();
            rtPipeline.WriteLightsDescriptor();
            rtPipeline.WriteIblDescriptors();
        }

        // Object picking — owns only a tiny result SSBO; the TLAS is bound below
        // after InitRayQuery. No-op at runtime when ray queries aren't supported
        // (ProcessPickRequest gates on RayShadowsSupported + a valid TLAS).
        pickPipeline = new PickPipeline(this);
        pickPipeline.Initialize();

        // Selection outline (mode-agnostic, ray-query driven). Mask pipeline
        // borrows the TLAS (bound below) + the renderer-owned mask image; the
        // outline pass reads that mask and draws into FinalColor.
        selectionMaskPipeline = new SelectionMaskPipeline(this);
        selectionMaskPipeline.Initialize();
        selectionMaskPipeline.WriteMaskImageDescriptor(selectionMask.ImageView);

        outlinePipeline = new OutlinePipeline(this);
        outlinePipeline.Initialize();
        outlinePipeline.WriteMaskDescriptor(selectionMask.ImageView);

        // Tone-map / post pass — reads HDRColor produced by the lighting pass, writes
        // the LDR FinalColor that the swapchain blit sources. Initialized after
        // SetupDeferredRenderer so the HDRColor ImageView exists for the descriptor write.
        tonemapPipeline = new TonemapPipeline(this) { Operator = tonemapOperator };
        tonemapPipeline.Initialize();
        tonemapPipeline.WriteHdrInputDescriptor(
            scene.renderGraph.GetResource("HDRColor").ImageView, gBufferSampler);

        // Transparent forward+ pass — renders BLEND-mode materials between the lighting
        // pass and the tonemap pass. Shares the lights SSBO + TLAS + tile cull buffers
        // with PbrDeferredPipeline; shares the bindless set with GeometryPipeline.
        transparentPipeline = new TransparentPipeline(this) { SoftShadowsEnabled = softShadowsEnabled };
        transparentPipeline.Initialize();
        transparentPipeline.WriteSharedLightingDescriptors(lightCullPipeline);
        // IBL bindings live on Renderer-owned VkImages that exist before any
        // pipeline initializes; write them straight after Initialize.
        transparentPipeline.WriteIblDescriptors();
        // Probe bindings — same stable-handle story as IBL. Wired once at init.
        transparentPipeline.WriteProbeDescriptors();

        // Skybox renders the envCube into HDRColor between lighting and transparent.
        // EditorState.SkyboxEnabled gates the draw without re-recording the graph.
        skyboxPipeline = new SkyboxPipeline(this);
        skyboxPipeline.Initialize();

        //Create command buffers
        CreateCommandBuffers();
        //Create sync objects
        CreateSyncObjects();

        CreateTestEntity();

        // Build BLAS / TLAS for ray-traced shadows. Gated on RayShadowsSupported
        // inside InitRayQuery — safe to call even when ray queries aren't available.
        InitRayQuery();
        // Bind the TLAS into the lighting descriptor sets — both the deferred lighting
        // pass and the forward+ transparent pass walk it for ray-traced shadows.
        if (tlas.Handle != 0)
        {
            PbrDeferredPipeline.WriteTlasDescriptor(tlas);
            transparentPipeline.WriteTlasDescriptor(tlas);
            ptComputePipeline.WriteTlasDescriptor(tlas);
            rtPipeline?.WriteTlasDescriptor(tlas);
            pickPipeline.WriteTlasDescriptor(tlas);
            selectionMaskPipeline.WriteTlasDescriptor(tlas);
            // Bind the ShadowEntityInfo SSBO + global vb/ib for the alpha-test
            // path in the PBR lighting shadow rays. Has to happen after
            // InitRayQuery because the SSBO is allocated inside RebuildTlas.
            PbrDeferredPipeline.WriteShadowAlphaDescriptors();
            ptComputePipeline.WriteShadowInfoDescriptor();
            rtPipeline?.WriteShadowInfoDescriptor();
            // Pick + selection resolve the hit entity through the same flat
            // ShadowEntityInfo table (per-cluster instances → entity via
            // entityInfo[InstanceCustomIndex + GeometryIndex].entityIndex).
            pickPipeline.WriteEntityInfoDescriptor();
            selectionMaskPipeline.WriteEntityInfoDescriptor();
            // Emissive area-light buffers (built inside RebuildTlas, always
            // allocated with ≥1 slot so the binding is valid even with no emitters).
            ptComputePipeline.WriteEmissiveDescriptors();
            rtPipeline?.WriteEmissiveDescriptors();
        }

        initialized = true;
    }

    private void CreateTestEntity()
    {
        // First-launch convenience: bake IBL from whatever .hdr the user has
        // already placed in Assets/Textures. The Renderer Settings panel can
        // load a different file at any time. No HDR present → cubes stay black
        // and the scene runs with direct lighting only.
        TryAutoLoadEnvironment();

        // Scene loads are driven by FileBrowserPanel (File → Open). Lights and
        // the test probe stay here because they aren't tied to a glTF import.
        SpawnTestLights();
        SpawnTestProbe();
    }

    // Phase 3 smoke test — a single reflection probe at the scene origin. Once
    // Phase 4 lands its capture shader, this probe will start producing actual
    // prefiltered cubemap content. Until then the registration / scheduler path
    // is what gets exercised.
    private void SpawnTestProbe()
    {
        Entity* probeEntity = Entity.Create("TestProbe");
        probeEntity->AddComponent(new TransformComponent());
        probeEntity->GetComponent<TransformComponent>()?.SetPosition(new Vector3(-7f, 2f, 4f));
        probeEntity->AddComponent(new ReflectionProbeComponent
        {
            InfluenceRadius = 12f,
            UpdatePolicy    = ProbeUpdatePolicy.OnDirty,
        });
        probeEntity->Initialize();
        scene.AddEntity(probeEntity);

        var probe = probeEntity->GetComponent<ReflectionProbeComponent>();
        if (probe != null && !reflectionProbeSystem.Register(probe))
            Console.WriteLine("[Probe] Out of cube-array slots — test probe not registered");
    }

    private void SpawnTestLights()
    {
        // Directional key light — sun-like, pointing slightly down and forward.
        // Position irrelevant for directional; we keep an identity transform.
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

        // Point fill light — warm, sits to the right of and above the helmet.
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

        // Spot rim light — tight cone aimed at the helmet from below-left.
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
            InnerConeCos = MathF.Cos(MathF.PI / 4f),  //  ~90°
            OuterConeCos = MathF.Cos(MathF.PI / 2f),  // ~120°
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

    /// <summary>
    /// Rebuild the deferred + transparent PBR pipelines. Use this after toggling
    /// softShadowsEnabled — that flag is a fragment-stage specialization constant
    /// so changes don't apply to a live pipeline. Cross-pipeline descriptor
    /// writes (TLAS, tile buffers, shadow-alpha, IBL) are re-issued because the
    /// new VkPipeline owns brand-new descriptor sets.
    /// </summary>
    public void RebuildPbrPipelines()
    {
        if (!initialized) return;
        vk!.DeviceWaitIdle(device);

        transparentPipeline?.Dispose();
        PbrDeferredPipeline?.Dispose();

        PbrDeferredPipeline = new PbrDeferredPipeline(this) { SoftShadowsEnabled = softShadowsEnabled };
        PbrDeferredPipeline.Initialize();

        transparentPipeline = new TransparentPipeline(this) { SoftShadowsEnabled = softShadowsEnabled };
        transparentPipeline.Initialize();

        // Re-wire cross-pipeline + Renderer-owned bindings on the fresh descriptor sets.
        PbrDeferredPipeline.WriteTileBufferDescriptors(lightCullPipeline);
        PbrDeferredPipeline.WriteProbeDescriptors();
        transparentPipeline.WriteSharedLightingDescriptors(lightCullPipeline);
        transparentPipeline.WriteIblDescriptors();
        transparentPipeline.WriteProbeDescriptors();
        // LightCullPipeline survives the rebuild but its set 0 binding 0 still
        // points at the freed PBR light SSBO — fix it up.
        lightCullPipeline.RewriteLightsBinding();
        if (tlas.Handle != 0)
        {
            PbrDeferredPipeline.WriteTlasDescriptor(tlas);
            transparentPipeline.WriteTlasDescriptor(tlas);
            PbrDeferredPipeline.WriteShadowAlphaDescriptors();
        }
    }

    /// <summary>
    /// Rebuild the tonemap pipeline. Use this after changing tonemapOperator —
    /// the operator selection is a fragment-stage spec constant. HDRColor view
    /// is re-bound because Initialize() allocated a fresh descriptor set.
    /// </summary>
    public void RebuildTonemapPipeline()
    {
        if (!initialized) return;
        vk!.DeviceWaitIdle(device);

        tonemapPipeline?.Dispose();
        tonemapPipeline = new TonemapPipeline(this) { Operator = tonemapOperator };
        tonemapPipeline.Initialize();
        tonemapPipeline.WriteHdrInputDescriptor(
            scene.renderGraph.GetResource("HDRColor").ImageView, gBufferSampler);
    }

    public void Update(double d)
    {

        DrawFrame();
    }

    // Full teardown in reverse-creation order. Safe to call once after Initialize.
    // Globals.vk is a process-wide singleton — never Dispose it here. The window
    // is owned by Engine and disposed by Engine.Shutdown.
    public void Cleanup()
    {
        if (!initialized) return;

        // Drain GPU work so nothing references resources we're about to destroy.
        vk!.DeviceWaitIdle(device);

        //  Frame sync
        for (var i = 0; i < MAX_CONCURRENT_FRAMES; i++)
        {
            if (renderFinishedSemaphores[i].Handle != 0)
                vk.DestroySemaphore(device, renderFinishedSemaphores[i], null);
            if (imageAvailableSemaphores[i].Handle != 0)
                vk.DestroySemaphore(device, imageAvailableSemaphores[i], null);
            if (inFlightFences[i].Handle != 0)
                vk.DestroyFence(device, inFlightFences[i], null);
        }
        
        if (testEntity != null)
        {
            Entity.Destroy(testEntity);
            testEntity = null;
        }

        //  Ray-query AS handles + scratch + instance buffer
        // Must come before ResourceManager.Dispose so the BLAS storage destroys cleanly
        // (BLAS doesn't reference VB/IB after build, but order is least-surprising this way).
        CleanupRayQuery();

        //  Mesh pool (global VB/IB)
        Engine.ResourceManager.Dispose();
        
        // ImGUI dispose
        imGuiUtils?.Dispose();
        
        
        // Render graph + size-dependent attachments
        // RenderGraph.Dispose disposes resources it owns; ImageResource.Dispose
        // is idempotent so the explicit calls below are safe redundant cleanup.
        scene?.renderGraph?.Dispose();
        depthImageResource?.Dispose();
        gBufferPosition?.Dispose();
        gBufferNormal?.Dispose();
        gBufferAlbedo?.Dispose();
        gBufferMaterial?.Dispose();
        gBufferEmissive?.Dispose();

        // Pathtracer storage images — same size class as the g-buffers, separate
        // owner because they aren't render-graph resources.
        ptAccumulator?.Dispose();
        ptOutColor?.Dispose();
        selectionMask?.Dispose();

        // Lights SSBO — renderer-owned mirror of Scene's LightComponents.
        DestroyLightBuffers();

        // G-buffer sampler
        if (gBufferSampler.Handle != 0) vk.DestroySampler(device, gBufferSampler, null);

        // Reflection probes
        reflectionProbeSystem?.Dispose();

        // IBL images + samplers
        CleanupIblResources();

        // Pipelines (each pipeline disposes its own buffers, sets, layouts)
        outlinePipeline      ?.Dispose();
        selectionMaskPipeline?.Dispose();
        pickPipeline       ?.Dispose();
        rtPipeline         ?.Dispose();
        ptComputePipeline  ?.Dispose();
        skyboxPipeline     ?.Dispose();
        transparentPipeline?.Dispose();
        tonemapPipeline    ?.Dispose();
        PbrDeferredPipeline?.Dispose();
        lightCullPipeline  ?.Dispose();
        drawCullPipeline   ?.Dispose();
        geometryPipeline   ?.Dispose();

        // Swap chain + image views
        CleanupSwapChain();

        // RHI context — strictly last. GraphicsDevice.Dispose() destroys the
        // descriptor pool, command pool, frees every VkDeviceMemory block, then
        // tears down device → debug → surface → instance in order. Every resource
        // freed above allocated through it, so it has to outlive them all.
        gfx.Dispose();

        initialized = false;
    }
}
