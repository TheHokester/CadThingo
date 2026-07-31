using System.Linq;
using System.Numerics;
using System.Reflection;
using CadThingo.Graphics.Rendering;
using CadThingo.VulkanEngine.ImGui;
using CadThingo.VulkanEngine.Renderer.Descriptors;
using CadThingo.VulkanEngine.Renderer.FeatureLifecycle;
using CadThingo.VulkanEngine.Renderer.FeatureLifecycle.RenderFeatureInterfaces;
using CadThingo.VulkanEngine.Renderer.Pipelines;
using CadThingo.VulkanEngine.Renderer.Features.Deferred;
using CadThingo.VulkanEngine.Renderer.Features.PathTracer;
using CadThingo.VulkanEngine.Renderer.Features.Forward;
using CadThingo.VulkanEngine.Renderer.Features.WavefrontPathTracer;
using CadThingo.VulkanEngine.Renderer.Features.ReSTIR;
using CadThingo.VulkanEngine.Renderer.Features.Tonemapping;
using CadThingo.VulkanEngine.Renderer.Features.IBL;
using CadThingo.VulkanEngine.Renderer.Features.Selection;
using CadThingo.VulkanEngine.Renderer.Slang;
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
    // repoints land (L1 of the renderer refactor).a
    internal GraphicsDevice gfx = null!;

    private bool initialized = false;

    // Config input for the GraphicsDevice (instance validation layers). Lives here so
    // the toggle stays next to the renderer; passed to the GraphicsDevice constructor.
    private readonly bool enableValidationLayers = false;
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

    // Read-only handle accessor - pipelines that load their own device-extension
    // dispatch tables (e.g. RtPipeline -> KhrRayTracingPipeline) need the instance
    // for vk.TryGetDeviceExtension. Generic, not tied to any one extension.
    internal Instance GetVkInstance() => gfx.GetVkInstance();

    public enum RenderMode : uint
    {
        Deferred = 0,
        ForwardPlus = 1,
        RayCompute = 2,
        RayTrace =3,
        RayWavefront = 4,
        ReStirDI = 5
    }
    // Active technique identity, DERIVED from the active core. The few call sites that branch on
    // the kind of renderer running (ViewportPanel fullscreen gate, the pathtracer settings panel)
    // read this. It is no longer a selection knob -- selection is by list index (RequestCoreIndex);
    // a device without the RT pipeline simply never registers those cores, so they can't be picked.
    public RenderMode renderMode => _activeCore?.Mode ?? RenderMode.Deferred;

    // Every feature the device can run, owned by the FeatureHost: constructed from the catalog,
    // wired, initialized, phase-pumped and disposed without this class naming any of them. Adding a
    // technique or subsystem is a new file with a descriptor -- no switch, no field here, no ImGui
    // label array to keep in sync.
    private FeatureHost _features = null!;

    // Exactly one core produces the frame. The ImGui mode combo lists the cores by index and hands
    // an index back (RequestCoreIndex); DrawFrame swaps _activeCore to the requested core and
    // Activates it (rebinding tonemap's HDR input to the new core's scene-colour source).
    private IRenderCore _activeCore = null!;
    private int         _desiredCoreIndex;

    /// <summary>The built render cores, in descriptor Order. Drives the ImGui mode combo, so a new
    /// core needs no host edit at all.</summary>
    internal IReadOnlyList<IRenderCore> RenderCores => _features.Cores;

    /// <summary>List index of the currently-active core (for the ImGui combo's current selection).</summary>
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

    /// <summary>ImGui mode-combo entry point: request the core at list index <paramref name="i"/>
    /// become active. The swap + Activate happen at the start of the next frame (DrawFrame step 0d),
    /// after DeviceWaitIdle, so it is safe to call mid-frame from the UI.</summary>
    internal void RequestCoreIndex(int i)
    {
        if (i >= 0 && i < RenderCores.Count) _desiredCoreIndex = i;
    }
    
    
    // Image-based lighting — env/irradiance/prefiltered cubes + BRDF LUT + bake
    // pipelines. Host-owned; pipelines and cores read its views/samplers.
    internal IblSystem Ibl = null!;

    // Runtime reflection probes - GPU resources + CPU registry. Allocated after
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


    

    // Presentation resources (L1.3). Renderer keeps the former field names as
    // delegating accessors so the frame body / ImGui / command-buffer sizing read
    // them unchanged.
    internal Swapchain swapchain = null!;
    internal Format    swapChainImageFormat => swapchain.ImageFormat;
    internal Extent2D  swapChainExtent      => swapchain.Extent;
    internal Image[]   swapChainImages      => swapchain.Images;
    internal ImageView[] swapChainImageViews => swapchain.ImageViews;

    // Render targets (L1.5): depth + g-buffers + sampler + PT/selection images, sized
    // to the render extent. Renderer keeps the former field names as delegating
    // accessors so pipelines / the frame body read them unchanged.
    internal RenderTargets renderTargets = null!;

    // GpuScene : the scene→GPU mirror + the single Extract path. Owns
    // the light SSBO and hosts the light/material extractors today; renderable /
    // AS buffers + the scene descriptor set fold in over the later L2 steps.
    internal GpuScene gpuScene = null!;

    // Descriptor-system track (docs/descriptor-system.md): runtime shader compile+cache
    // (Phase A) and the unified scene set + constant arena (Phase B). Pipelines migrate
    // onto these shader-by-shader; until then the registry's sets are written but unbound.
    internal ShaderLibrary shaderLibrary = null!;
    internal DescriptorRegistry descriptorRegistry = null!;

    /// The device-services handles, bundled for injection. Valid once Initialize has built the
    /// device, shader library, and registry - construct pipelines after that point.
    internal GpuContext Gpu; 

    // Render target extent — the size at which the deferred chain (gbuffers,
    // HDRColor, FinalColor) and the lighting tile grid are sized. Distinct from
    // swapChainExtent because the editor's viewport panel can be smaller than
    // the OS window. Owned by RenderTargets; viewport panel drives resize.
    internal Extent2D renderExtent => renderTargets.RenderExtent;
    public Extent2D RenderExtent => renderExtent;

    // UI overlay drawn after the FinalColor blit, before the present transition.
    // Lifetime/instantiation owned externally; null when UI is disabled.
    public ImGuiVulkanUtils? imGuiUtils;
    
    //dynamic rendering fields
    RenderingInfo renderingInfo;
    List<RenderingAttachmentInfo> colorAttachments;
    RenderingAttachmentInfo depthAttachment;
    
    
    //pipelines - each owns its own VkPipeline, layouts, descriptor sets, and per-pipeline buffers
    internal GeometryPipeline     geometryPipeline;
    internal DrawCullPipeline     drawCullPipeline;
    internal LightCullPipeline    lightCullPipeline;
    internal PbrDeferredPipeline  PbrDeferredPipeline;   // accessor used by LightCullPipeline (consumer of the lights SSBO)
    internal TonemapPipeline      tonemapPipeline;       // post-process: HDRColor → FinalColor
    internal TransparentPipeline  transparentPipeline;   // forward+ BLEND-mode pass between lighting and tonemap
    internal SkyboxPipeline       skyboxPipeline;        // background env-cube draw, between lighting and transparent
    internal PTComputePipeline    ptComputePipeline;
    internal WavefrontPTPipeline  wavefrontPipeline;     // graph-resident wavefront path tracer (RenderMode.RayWavefront)
    internal RTPipeline?          rtPipeline;            // opt-in RT-pipeline path tracer (null when unsupported)
    internal ReStirDIPipeline?    reStirPipeline;        // opt-in ReSTIR DI tracer, RT-pipeline (null when unsupported)
    internal SelectionSystem      selection = null!;     // host-owned editor selection: pick + coverage mask + outline

    

    // Specialization-constant gate for soft (PCSS-style) ray-queried shadows.
    // Threaded into PbrDeferredPipeline.SoftShadowsEnabled at construction time.
    public bool softShadowsEnabled = true;

    // Pending rebuild flags, set by the event handlers below. Consumed at the top
    // of DrawFrame so the rebuild never races a command buffer that already bound
    // the old pipeline. These are the single owner of "a rebuild is due" - the
    // settings panel used to keep its own copies and write these directly.
    internal bool pendingPbrRebuild     = false;
    internal bool pendingTonemapRebuild = false;

    // Bus subscriptions, disposed in Cleanup so a torn-down renderer stops being
    // handed events (the bus outlives it - Engine owns the bus).
    private readonly List<IDisposable> eventSubscriptions = new();

    // Tone-map curve selector — threaded into TonemapPipeline.Operator at
    // construction time as a specialization constant. Toggling requires a
    // pipeline rebuild.
    public TonemapOperator tonemapOperator = TonemapOperator.Filmic;
    
    // Per-frame command buffers + sync ring (L1.4). FrameRing owns the state and the
    // acquire/submit/present cadence; Renderer keeps the former field names as
    // delegating accessors so the frame body reads them unchanged. currentFrame /
    // frameCounter advance via frameRing.Advance() at end-of-frame.
    internal FrameRing frameRing = null!;
    internal CommandBuffer[] commandBuffers           => frameRing.CommandBuffers;
    internal Semaphore[]     imageAvailableSemaphores => frameRing.ImageAvailableSemaphores;
    internal Semaphore[]     renderFinishedSemaphores => frameRing.RenderFinishedSemaphores;
    internal Fence[]         inFlightFences           => frameRing.InFlightFences;
    internal uint            currentFrame             => frameRing.CurrentFrame;
    internal ulong           frameCounter             => frameRing.FrameCounter;

    //Camera
    Camera camera;
    public Camera Camera => camera;
    
    //Sampler for gbuffers and pathtracing ImageResources (the selection mask is
    //owned by RenderTargets and reached through SelectionSystem)
    internal Sampler       gBufferSampler     => renderTargets.GBufferSampler;
    private  ImageResource ptAccumulator      => renderTargets.PtAccumulator;
    private  ImageResource ptOutColor         => renderTargets.PtOutColor;

    //Path-tracing accumulation state (the images live on RenderTargets; this is the
    //CPU-side dirty/restart bookkeeping that drives progressive integration).
    bool accumulatorDirty = false;
    public void MarkAccumulatorDirty()
    {
        accumulatorDirty = true;
        // Dirty-driven scene extract (L2 step 7) piggybacks on this near-universal
        // "scene visually changed" signal — every material/light/transform edit calls
        // it, so the GPU mirror re-packs. Over-triggering (e.g. a PT camera move, which
        // doesn't change scene data) is harmless: it only costs a re-pack.
        gpuScene?.MarkSceneDirty();
    }
    // Read/clear hooks for the path-trace cores (L3): the dirty flag lives here because
    // MarkAccumulatorDirty is called from all over the editor, but the per-frame
    // consume-and-clear is the PT core's job (bridged into its pipeline's accumulator reset).
    internal bool AccumulatorDirty => accumulatorDirty;
    internal void ClearAccumulatorDirty() => accumulatorDirty = false;

    /// <summary>
    /// Subscribes the renderer to the editor's intents. Everything here is Renderer-category, so
    /// the bus queues it and drains in Engine's Update pass - ahead of DrawFrame, which is where
    /// the flags these handlers set get consumed. Publishers therefore never touch renderer state
    /// directly and never need to know about the frame boundary.
    /// </summary>
    private void SubscribeToEvents()
    {
        var bus = Engine.EventBus;

        // Scene edits invalidate both the acceleration structure and any in-progress
        // integration; a pure view change invalidates only the latter.
        eventSubscriptions.Add(bus.Subscribe<SceneDirtyEvent>(_ =>
        {
            MarkTlasDirty();
            MarkAccumulatorDirty();
        }));
        eventSubscriptions.Add(bus.Subscribe<PathTracingAccumulatorInvalidatedEvent>(
            _ => MarkAccumulatorDirty()));

        // Both of these are specialization constants baked at pipeline build, so the handler
        // stores the value and flags the rebuild rather than applying it live.
        eventSubscriptions.Add(bus.Subscribe<TonemapFilterChangedEvent>(e =>
        {
            tonemapOperator       = e.GetOperator;
            pendingTonemapRebuild = true;
        }));
        eventSubscriptions.Add(bus.Subscribe<PbrSoftShadowingChangedEvent>(e =>
        {
            softShadowsEnabled = e.GetEnabled;
            pendingPbrRebuild  = true;
        }));
    }

    private void UnsubscribeFromEvents()
    {
        foreach (var sub in eventSubscriptions) sub.Dispose();
        eventSubscriptions.Clear();
    }

    // (The previous-frame camera snapshot that drove PT accumulator restarts now lives in
    // PathTraceCoreBase, the owner of the PT render technique.)


    public void Initialize()
    {
        // Bring up the Vulkan RHI context (instance → debug → surface → physical →
        // logical device → memory allocator → command pool). Everything below
        // allocates through it.
        gfx = new GraphicsDevice(window!, enableValidationLayers);
        gfx.Initialize();

        swapchain = new Swapchain(gfx, window!);
        swapchain.Create();
        // Initial render extent tracks swapchain extent. ViewportPanel can later
        // shrink it via EditorState.RequestedRenderExtent → ResizeRenderTargets.
        renderTargets = new RenderTargets(gfx);
        renderTargets.SetExtent(swapChainExtent);
        swapchain.CreateImageViews();
        SetupDynamicRendering();

        CreateDescriptorPool();

        // Reflects SceneBindings.slang into the canonical scene set layout and allocates
        // the per-frame set instances. Providers register at the end of Initialize
        // (RegisterSceneBindings) once they exist.
        shaderLibrary = ShaderLibrary.CreateDefault();
        descriptorRegistry = new DescriptorRegistry(gfx, shaderLibrary, RenderConfig.MAX_CONCURRENT_FRAMES);
        
        Gpu = new(gfx, descriptorRegistry, shaderLibrary);
        
        Engine.ResourceManager.Initialize(Gpu);

        // Depth + g-buffers are ImageResource objects only — the render graph allocates
        // their VkImages inside Compile() (SetupDeferredRenderer below). The PT /
        // selection images are allocated + laid out here.
        renderTargets.AllocateAll();
        // GpuScene owns the light SSBO (and, as L2 progresses, the rest of the
        // scene→GPU mirror). Built on gfx; must exist before any pipeline that
        // binds the lights buffer, and before the first per-frame Extract.
        gpuScene = new GpuScene(gfx);
        gpuScene.CreateLightBuffers();
        // Cull-input SSBO - must exist before DrawCullPipeline binds it (binding 0).
        gpuScene.CreateRenderableBuffers();

        // IBL images allocated up-front, cleared to black. The PBR lighting set
        // binds them unconditionally; the compute bake passes fill the content
        // when an HDR is loaded via Ibl.LoadEnvironmentHdr. The ctor also bakes
        // the view-independent BRDF LUT once.
        Ibl = new IblSystem(Gpu, this);

        // Reflection-probe GPU resources (cubemap array + per-probe SSBO).
        // Reuses the IBL prefilter pipeline at capture time, so it has to be
        // constructed after the IblSystem.
        reflectionProbeSystem = new ReflectionProbeSystem(Gpu, this, Ibl);

        // Pipelines that don't depend on allocated g-buffer image views 
        // Render-graph pass closures (registered in SetupDeferredRenderer) read
        // `geometryPipeline` / `drawCullPipeline` / `PbrDeferredPipeline` through
        // `this`, so they must exist (be non-null) by the time the closures *run*
        // (per frame in DrawFrame), not when they're declared.
        geometryPipeline = new GeometryPipeline(Gpu, this);
        geometryPipeline.Initialize();

        drawCullPipeline = new DrawCullPipeline(Gpu, this);
        drawCullPipeline.Initialize();

        scene = new Scene(vk, device, physicalDevice);//initialise scene
        // The deferred chain's g-buffers / depth / HDR are now FrameGraph transients,
        // allocated when DeferredCore (constructed at the end of Initialize) compiles its graph.
        // PbrDeferredPipeline's g-buffer set is (re)bound there from the freshly-allocated views,
        // so it can initialize in any order below.
        imGuiUtils = new ImGuiVulkanUtils(this, (uint)queueFamilyIndices.graphicsFamily! );
        imGuiUtils?.init(swapChainExtent.Width, swapChainExtent.Height);

        // Bind FinalColor into the ImGui viewport descriptor so the Viewport panel
        // can render the scene as an ImGui.Image. Re-bound on swapchain recreate
        // because the underlying ImageView is rebuilt.
        imGuiUtils?.WriteViewportDescriptor(renderTargets.FinalColor.ImageView);

        //  Lighting + light-cull pipelines (depend on allocated g-buffer / lights SSBO) 
        PbrDeferredPipeline = new PbrDeferredPipeline(Gpu, this) { SoftShadowsEnabled = softShadowsEnabled };
        PbrDeferredPipeline.Initialize();

        lightCullPipeline = new LightCullPipeline(Gpu, this);
        lightCullPipeline.Initialize();

        // Scene buffers (TLAS / lights / shadow info / vb+ib / emissive / bindless) come from the
        // scene set; the accumulator / out-color pair from the registry-owned FeaturePTIO set.
        ptComputePipeline = new PTComputePipeline(Gpu, this);
        ptComputePipeline.Initialize();

        // Wavefront path tracer (RenderMode.RayWavefront). Shares the same accumulator /
        // out-color images as the megakernel via FeaturePTIO; scene buffers come from the scene
        // set. The SoA working set is pipeline-owned; envCube rides FeatureEnv.
        wavefrontPipeline = new WavefrontPTPipeline(Gpu, this);
        wavefrontPipeline.Initialize();

        // Opt-in RT-pipeline path tracer (RenderMode.RayTrace). Shares the same
        // accumulator/outColor images + scene buffers as the compute path; only
        // built when the device exposes the feature. TLAS/shadow/emissive sets
        // are bound below after InitRayQuery, mirroring the compute pipeline.
        if (RayTracePipelineSupported)
        {
            // Scene buffers (TLAS / lights / shadow info / vb+ib / emissive / bindless) come from
            // the scene set; envCube rides FeatureEnv, accumulator / out-color ride FeaturePTIO.
            rtPipeline = new RTPipeline(Gpu, this);
            rtPipeline.Initialize();

            // ReSTIR DI tracer (RenderMode.ReStirDI). Same RT-pipeline machinery as rtPipeline
            // (it subclasses RTPipeline), forked only at the shader; shares the same accumulator /
            // outColor + scene set.
            reStirPipeline = new ReStirDIPipeline(Gpu, this);
            reStirPipeline.Initialize();
        }

        // Editor selection - object picking + ray-query coverage mask + outline
        // composite. Host-owned; the TLAS / entity-info descriptors are bound
        // below after InitRayQuery. No-op at runtime when ray queries aren't
        // supported (ProcessPickRequest / RecordOutline gate on a valid TLAS).
        selection = new SelectionSystem(Gpu, this);

        // Tone-map / post pass - reads the FrameGraph's HDRColor transient, writes the LDR
        // FinalColor that the swapchain blit sources. Its HDR-input descriptor is bound from the
        // graph's freshly-allocated HDR view when DeferredCore compiles its graph (and re-pointed
        // per active core via IRenderCore.Activate), so no descriptor write here.
        tonemapPipeline = new TonemapPipeline(Gpu, this) { Operator = tonemapOperator };
        tonemapPipeline.Initialize();

        // Transparent forward+ pass — renders BLEND-mode materials between the lighting
        // pass and the tonemap pass. Lights / TLAS / bindless come from the scene set;
        // the tile cull buffers are wired from LightCullPipeline below.
        transparentPipeline = new TransparentPipeline(Gpu, this) { SoftShadowsEnabled = softShadowsEnabled };
        transparentPipeline.Initialize();

        // Skybox renders the envCube into HDRColor between lighting and transparent.
        // EditorState.SkyboxEnabled gates the draw without re-recording the graph.
        skyboxPipeline = new SkyboxPipeline(Gpu, this);
        skyboxPipeline.Initialize();

        // Per-frame command buffers + sync ring.
        frameRing = new FrameRing(gfx, RenderConfig.MAX_CONCURRENT_FRAMES);
        frameRing.CreateCommandBuffers(swapChainImages.Length);
        frameRing.CreateSyncObjects();

        CreateTestEntity();

        // Build BLAS / TLAS for ray-traced shadows. Gated on RayShadowsSupported
        // inside InitRayQuery — safe to call even when ray queries aren't available.
        InitRayQuery();
        

        // Scene-set registrations: every provider that exists
        // at init, matched by SceneBindings parameter name. Runtime handle changes
        // re-register at their rebuild sites; the dump shows any remaining holes.
        RegisterSceneBindings();
        RegisterFeatureBindings();
        RegisterPathTraceIoBindings();
        Console.WriteLine(descriptorRegistry.DumpBindings());

        // Cross-check every migrated pipeline's reflected bindings against what the registry owns
        // and was handed. Runs here because it needs both sides complete: pipelines built above,
        // providers registered on the two lines before. Throws on a real mismatch.
        Console.WriteLine(descriptorRegistry.Validate(ReflectedPrograms()));
        var lc = gfx.LayoutCache;
        Console.WriteLine($"[layout-cache] set layouts {lc.SetLayoutCount}/{lc.SetLayoutRequests} distinct, " +
                          $"pipeline layouts {lc.PipelineLayoutCount}/{lc.PipelineLayoutRequests} distinct");

        // Stand up every feature the device can run: construct (gated) -> wire -> Initialize, in
        // descriptor Order. This method names none of them; a new technique is a new file whose
        // module initializer put a descriptor in the catalog before Main ran. The manifest is the
        // replacement for the boot-order list that used to live right here, and it shows what the
        // gates excluded on THIS device, which a source list never could.
        _features = new FeatureHost(Gpu, this);
        _features.BuildAll();
        Console.WriteLine(_features.Dump());

        // Activate the boot core (lowest Order = Deferred, unless RequestCoreIndex ran before init).
        _activeCore = RenderCores[_desiredCoreIndex];
        _activeCore.Activate();

        // Last, so no handler can fire against half-built pipelines.
        SubscribeToEvents();

        initialized = true;
    }

    private void CreateTestEntity()
    {
        // First-launch convenience: bake IBL from whatever .hdr the user has
        // already placed in Assets/Textures. The Renderer Settings panel can
        // load a different file at any time. No HDR present → cubes stay black
        // and the scene runs with direct lighting only.
        Ibl.TryAutoLoadEnvironment();

        // Scene loads are driven by FileBrowserPanel (File → Open). Lights and
        // the test probe stay here because they aren't tied to a glTF import.
        SpawnTestLights();
        SpawnTestProbe();
    }

    // Reflection probe test - a single reflection probe at the scene origin. Once
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
    /// writes (tile buffers, IBL, probes, transparent's TLAS) are re-issued
    /// because the new VkPipeline owns brand-new descriptor sets; scene-set
    /// bindings live on the registry and need no rewire.
    /// </summary>
    public void RebuildPbrPipelines()
    {
        if (!initialized) return;
        vk!.DeviceWaitIdle(device);

        // In-place rebuild: same pipeline objects, fresh GPU handles. Pipeline-ref holders stay
        // valid, but Rebuild recreates the set-1 layout handle, so the deferred graph's baked
        // g-buffer set must be re-baked (OnPbrPipelineRebuilt below rebuilds the graph).
        // SoftShadowsEnabled is a spec constant, read by Rebuild's Initialize.
        PbrDeferredPipeline.SoftShadowsEnabled = softShadowsEnabled;
        PbrDeferredPipeline.Rebuild();

        transparentPipeline.SoftShadowsEnabled = softShadowsEnabled;
        transparentPipeline.Rebuild();

        // Re-wire cross-pipeline + Renderer-owned bindings on the fresh descriptor sets. The
        // g-buffer set (set 1) is graph-owned now, so DeferredCore re-bakes it by rebuilding
        // the deferred graph against PBR's new layout.
        _features.Get<DeferredCore>()?.OnPbrPipelineRebuilt();
        
    }

    /// <summary>
    /// Rebuild the tonemap pipeline. Use this after changing tonemapOperator —
    /// the operator selection is a fragment-stage spec constant.
    /// </summary>
    public void RebuildTonemapPipeline()
    {
        if (!initialized) return;
        vk!.DeviceWaitIdle(device);

        // In-place rebuild (stable object identity, fresh GPU handles) so each core module's
        // tonemap ref stays valid. Operator is a spec constant, read by Rebuild's Initialize.
        tonemapPipeline.Operator = tonemapOperator;
        tonemapPipeline.Rebuild();
        // Every core's graph-baked HDR set survives the in-place rebuild -- a set outlives its
        // source layout and binds by pipeline-layout compatibility (identical definition), and
        // the HDR view is unchanged -- so no graph rebuild is needed. Activate restarts the PT
        // cores' progressive accumulation.
        _activeCore.Activate();
    }

    public void Update(double d)
    {

        DrawFrame();
    }

    // Registers the current owners' handles into the unified scene set by SceneBindings
    // parameter name. Idempotent (registrations are queued, fence-safe rewrites); runtime
    // handle changes re-register just the changed name at their rebuild sites.
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
        // Engine shaders only ever index samplers[0]; fill the whole array with the default
        // sampler anyway so no element of the (non-PartiallyBound) binding is left invalid.
        for (int s = 0; s < 16; s++)
            descriptorRegistry.RegisterSampler("sceneSamplers", rm.DefaultSampler, s);
        if (tlas.Handle != 0)
        {
            descriptorRegistry.RegisterTlas("sceneTlas", tlas);
            descriptorRegistry.RegisterBuffer("sceneEntityInfo", ShadowInfoBuffer);
            descriptorRegistry.RegisterBuffer("sceneEmissiveTris", EmissiveTriBuffer);
            descriptorRegistry.RegisterBuffer("sceneEmissiveAlias", EmissiveAliasBuffer);
        }
    }

    // Every reflected program the renderer's pipelines resolved, for the registry cross-check.
    // Found by walking the renderer's own PipelineBase-typed fields rather than a hand-kept list:
    // a new pipeline joins the check by existing, not by someone remembering to add it.
    private IEnumerable<ProgramUse> ReflectedPrograms()
        => GetType()
            .GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)
            .Where(f => typeof(PipelineBase).IsAssignableFrom(f.FieldType))
            .Select(f => f.GetValue(this) as PipelineBase)
            .Where(p => p?.ReflectedProgram != null)
            .Select(p => new ProgramUse(p!.ReflectedProgram!, p.PrivateSetIndices))
            .DistinctBy(u => u.Program);

    // The progressive-accumulation IO pair (FeaturePTIO, set 5), shared by all four tracers -- they
    // all write the SAME two renderer-owned images, so this is one registry set rather than the
    // identical pass set each pipeline used to build and write for itself. Called after every
    // (re)allocation of the size-dependent targets, since resize replaces both views; the registry
    // queue applies the rewrite when each frame slot is provably idle.
    internal void RegisterPathTraceIoBindings()
    {
        descriptorRegistry.RegisterImage("accumulator", renderTargets.PtAccumulator.ImageView, ImageLayout.General);
        descriptorRegistry.RegisterImage("outColor",    renderTargets.PtOutColor.ImageView,    ImageLayout.General);
    }

    // Registers the FeatureIBL set (set 2): the global IBL split-sum + reflection-probe resources
    // consumed by the raster lighting shaders. Owned by IblSystem / ReflectionProbeSystem; their
    // views and (max-sized) buffers are stable for the renderer's lifetime and rebakes overwrite
    // content not handles, so registering once at init is enough. FeatureEnv's envCube registers
    // from IblSystem in a later step.
    private void RegisterFeatureBindings()
    {
        var r = descriptorRegistry;
        r.RegisterImage("irradianceCube",  Ibl.irradianceCubeView,  ImageLayout.ShaderReadOnlyOptimal, Ibl.iblCubeSampler);
        r.RegisterImage("prefilteredCube", Ibl.prefilteredCubeView, ImageLayout.ShaderReadOnlyOptimal, Ibl.iblCubeSampler);
        r.RegisterImage("brdfLut",         Ibl.brdfLutView,         ImageLayout.ShaderReadOnlyOptimal, Ibl.iblLutSampler);

        var probeSys = reflectionProbeSystem;
        r.RegisterImage("probeCubeArray", probeSys.prefilteredArrayView, ImageLayout.ShaderReadOnlyOptimal, probeSys.prefilteredArraySampler);

        var probes       = new Buffer[RenderConfig.MAX_CONCURRENT_FRAMES];
        var clusterRange = new Buffer[RenderConfig.MAX_CONCURRENT_FRAMES];
        var indexList    = new Buffer[RenderConfig.MAX_CONCURRENT_FRAMES];
        for (int i = 0; i < RenderConfig.MAX_CONCURRENT_FRAMES; i++)
        {
            probes[i]       = probeSys.probeRecordBuffers[i].buffer;
            clusterRange[i] = probeSys.clusterGrid.GetClusterRangeBuffer((uint)i);
            indexList[i]    = probeSys.clusterGrid.GetProbeIndexBuffer((uint)i);
        }
        r.RegisterBufferPerFrame("probes", probes);
        r.RegisterBufferPerFrame("probeClusterRange", clusterRange);
        r.RegisterBufferPerFrame("probeIndexList", indexList);

        // FeatureEnv (set 3): the raw environment cube, owned by IblSystem and consumed by the
        // skybox + path tracers. Stable handle (rebakes overwrite content), registered once.
        r.RegisterImage("envCube", Ibl.envCubeView, ImageLayout.ShaderReadOnlyOptimal, Ibl.iblCubeSampler);
    }

    // Centralized teardown. Each lifetime-scoped owner (FrameRing, RenderTargets,
    // Swapchain, GraphicsDevice) frees only what it owns; the orchestrator owns the
    // *order*. The one hard rule: GraphicsDevice.Dispose() runs strictly last - it frees
    // every VkDeviceMemory block + the device itself, so everything that allocated
    // through it must already be gone. Globals.vk is a process-wide singleton - never
    // disposed here. The window is owned by Engine and disposed by Engine.Shutdown.
    public void Cleanup()
    {
        if (!initialized) return;

        // Off the bus first: the bus outlives the renderer, and a queued intent drained after
        // this point would run its handler against destroyed pipelines.
        UnsubscribeFromEvents();

        // Drain GPU work so nothing references resources we're about to destroy.
        vk!.DeviceWaitIdle(device);

        //  Frame sync + per-frame command buffers
        frameRing.Dispose();

        //  Scene set + constant arena, then the shader library (drops slang.dll if loaded)
        descriptorRegistry.Dispose();
        shaderLibrary.Dispose();

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
        
        
        // Every feature, in reverse Order. DeferredCore owns the deferred FrameGraph (g-buffer /
        // depth / HDR transients), so features go before RenderTargets, which owns FinalColor + the
        // PT / selection images those graphs import.
        _features.Dispose();
        renderTargets.Dispose();

        // GpuScene buffers (light SSBO today) — the GPU mirror of Scene's render data.
        gpuScene.Dispose();

        // Reflection probes
        reflectionProbeSystem?.Dispose();

        // IBL images + samplers + bake pipelines
        Ibl?.Dispose();

        // Editor selection pipelines (pick + coverage mask + outline)
        selection?.Dispose();

        // Pipelines (each pipeline disposes its own buffers, sets, layouts)
        reStirPipeline     ?.Dispose();
        rtPipeline         ?.Dispose();
        wavefrontPipeline  ?.Dispose();
        ptComputePipeline  ?.Dispose();
        skyboxPipeline     ?.Dispose();
        transparentPipeline?.Dispose();
        tonemapPipeline    ?.Dispose();
        PbrDeferredPipeline?.Dispose();
        lightCullPipeline  ?.Dispose();
        drawCullPipeline   ?.Dispose();
        geometryPipeline   ?.Dispose();

        // Swap chain + image views
        swapchain.Dispose();

        // RHI context - strictly last. GraphicsDevice.Dispose() destroys the
        // descriptor pool, command pool, frees every VkDeviceMemory block, then
        // tears down device → debug → surface → instance in order. Every resource
        // freed above allocated through it, so it has to outlive them all.
        gfx.Dispose();

        initialized = false;
    }
}
