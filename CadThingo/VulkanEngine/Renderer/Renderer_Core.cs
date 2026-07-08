using System.Linq;
using System.Numerics;
using CadThingo.Graphics.Rendering;
using CadThingo.VulkanEngine.ImGui;
using CadThingo.VulkanEngine.Renderer.Pipelines;
using CadThingo.VulkanEngine.Renderer.RenderCores;
using CadThingo.VulkanEngine.Renderer.Features.Deferred;
using CadThingo.VulkanEngine.Renderer.Features.PathTracer;
using CadThingo.VulkanEngine.Renderer.Features.Forward;
using CadThingo.VulkanEngine.Renderer.Features.WavefrontPathTracer;
using CadThingo.VulkanEngine.Renderer.Features.ReSTIR;
using CadThingo.VulkanEngine.Renderer.Features.Tonemapping;
using CadThingo.VulkanEngine.Renderer.Features.IBL;
using CadThingo.VulkanEngine.Renderer.Features.Selection;
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

    // L3 render cores (renderer-refactor.md): one pluggable technique each, built eagerly (all
    // their pipelines already exist up front). Cores register THEMSELVES into this list from their
    // ctor (IRenderCore via RegisterCore), so adding a technique is just "construct it" -- no
    // switch, no per-core field, no ImGui label array to keep in sync. The ImGui mode combo lists
    // the cores by index and hands an index back (RequestCoreIndex); DrawFrame swaps _activeCore to
    // the requested core + Activates it (rebinds tonemap's HDR input to the new core's scene-colour
    // source). See IRenderCore. Index 0 (first registered = Deferred) is the boot default.
    private readonly List<IRenderCore> _renderCores = new();
    private IRenderCore                _activeCore  = null!;
    private int                        _desiredCoreIndex;

    /// <summary>The registered render cores, in construction order. Drives the ImGui mode combo so
    /// a new core needs no host edit beyond constructing it.</summary>
    internal IReadOnlyList<IRenderCore> RenderCores => _renderCores;

    /// <summary>List index of the currently-active core (for the ImGui combo's current selection).</summary>
    internal int ActiveCoreIndex => _renderCores.IndexOf(_activeCore);

    /// <summary>Called by an <see cref="IRenderCore"/> ctor to add itself to the registry.</summary>
    internal void RegisterCore(IRenderCore core) => _renderCores.Add(core);

    /// <summary>ImGui mode-combo entry point: request the core at list index <paramref name="i"/>
    /// become active. The swap + Activate happen at the start of the next frame (DrawFrame step 0d),
    /// after DeviceWaitIdle, so it is safe to call mid-frame from the UI.</summary>
    internal void RequestCoreIndex(int i)
    {
        if (i >= 0 && i < _renderCores.Count) _desiredCoreIndex = i;
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


    //frames-in-flight count. Referenced pervasively as Renderer.MAX_CONCURRENT_FRAMES;
    //FrameRing takes it as a ctor arg (L1.4) rather than depending back on Renderer.
    internal const uint MAX_CONCURRENT_FRAMES = 2;

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
    internal Shaders.ShaderLibrary shaderLibrary = null!;
    internal Descriptors.DescriptorRegistry descriptorRegistry = null!;

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
    
    
    //pipelines — each owns its own VkPipeline, layouts, descriptor sets, and per-pipeline buffers
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

    // GPU block-compression encoder for material textures (lazy: created on the first compressed
    // texture load, so it costs nothing for runs that never load a BC-formatted asset).
    private Features.TextureCompression.BcEncoder? _bcEncoder;
    internal Features.TextureCompression.BcEncoder BcEncoder => _bcEncoder ??= new Features.TextureCompression.BcEncoder(gfx);

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
        shaderLibrary = Shaders.ShaderLibrary.CreateDefault();
        descriptorRegistry = new Descriptors.DescriptorRegistry(gfx, shaderLibrary, MAX_CONCURRENT_FRAMES);

        Engine.ResourceManager.Initialize(this);

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
        Ibl = new IblSystem(this);

        // Reflection-probe GPU resources (cubemap array + per-probe SSBO).
        // Reuses the IBL prefilter pipeline at capture time, so it has to be
        // constructed after the IblSystem.
        reflectionProbeSystem = new ReflectionProbeSystem(this, Ibl);

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
        
        
        // Scene buffers (TLAS / lights / shadow info / vb+ib / emissive / bindless)
        // come from the scene set; only the storage-image IO + IBL sets are wired here.
        ptComputePipeline = new PTComputePipeline(this);
        ptComputePipeline.Initialize();
        ptComputePipeline.WriteStorageImageDescriptors(ptAccumulator.ImageView, ptOutColor.ImageView);
        // Same as the transparent / deferred pipelines: IBL images are Renderer-
        // owned and stable across rebakes, so the IBL set only needs writing once.
        ptComputePipeline.WriteIblDescriptors();

        // Wavefront path tracer (RenderMode.RayWavefront). Shares the same accumulator /
        // out-color images as the megakernel; scene buffers (TLAS / lights / shadow info /
        // vb+ib / emissive / bindless) come from the scene set. The SoA working set is
        // pipeline-owned. Only the storage-image IO + IBL sets are wired here.
        wavefrontPipeline = new WavefrontPTPipeline(this);
        wavefrontPipeline.Initialize();
        wavefrontPipeline.WriteStorageImageDescriptors(ptAccumulator.ImageView, ptOutColor.ImageView);
        wavefrontPipeline.WriteIblDescriptors();

        // Opt-in RT-pipeline path tracer (RenderMode.RayTrace). Shares the same
        // accumulator/outColor images + scene buffers as the compute path; only
        // built when the device exposes the feature. TLAS/shadow/emissive sets
        // are bound below after InitRayQuery, mirroring the compute pipeline.
        if (RayTracePipelineSupported)
        {
            // Scene buffers (TLAS / lights / shadow info / vb+ib / emissive / bindless) come
            // from the scene set; only the storage-image IO + IBL sets are wired here.
            rtPipeline = new RTPipeline(this);
            rtPipeline.Initialize();
            rtPipeline.WriteStorageImageDescriptors(ptAccumulator.ImageView, ptOutColor.ImageView);
            rtPipeline.WriteIblDescriptors();

            // ReSTIR DI tracer (RenderMode.ReStirDI). Same RT-pipeline machinery as rtPipeline
            // (it subclasses RTPipeline), forked only at the shader; shares the same accumulator /
            // outColor + scene set.
            reStirPipeline = new ReStirDIPipeline(this);
            reStirPipeline.Initialize();
            reStirPipeline.WriteStorageImageDescriptors(ptAccumulator.ImageView, ptOutColor.ImageView);
            reStirPipeline.WriteIblDescriptors();
        }

        // Editor selection - object picking + ray-query coverage mask + outline
        // composite. Host-owned; the TLAS / entity-info descriptors are bound
        // below after InitRayQuery. No-op at runtime when ray queries aren't
        // supported (ProcessPickRequest / RecordOutline gate on a valid TLAS).
        selection = new SelectionSystem(this);

        // Tone-map / post pass - reads the FrameGraph's HDRColor transient, writes the LDR
        // FinalColor that the swapchain blit sources. Its HDR-input descriptor is bound from the
        // graph's freshly-allocated HDR view when DeferredCore compiles its graph (and re-pointed
        // per active core via IRenderCore.Activate), so no descriptor write here.
        tonemapPipeline = new TonemapPipeline(this) { Operator = tonemapOperator };
        tonemapPipeline.Initialize();

        // Transparent forward+ pass — renders BLEND-mode materials between the lighting
        // pass and the tonemap pass. Lights / TLAS / bindless come from the scene set;
        // the tile cull buffers are wired from LightCullPipeline below.
        transparentPipeline = new TransparentPipeline(this) { SoftShadowsEnabled = softShadowsEnabled };
        transparentPipeline.Initialize();
        transparentPipeline.WriteTileBufferDescriptors(lightCullPipeline);
        // IBL bindings live on Renderer-owned VkImages that exist before any
        // pipeline initializes; write them straight after Initialize.
        transparentPipeline.WriteIblDescriptors();
        // Probe bindings — same stable-handle story as IBL. Wired once at init.
        transparentPipeline.WriteProbeDescriptors();

        // Skybox renders the envCube into HDRColor between lighting and transparent.
        // EditorState.SkyboxEnabled gates the draw without re-recording the graph.
        skyboxPipeline = new SkyboxPipeline(this);
        skyboxPipeline.Initialize();

        // Per-frame command buffers + sync ring.
        frameRing = new FrameRing(gfx, MAX_CONCURRENT_FRAMES);
        frameRing.CreateCommandBuffers(swapChainImages.Length);
        frameRing.CreateSyncObjects();

        CreateTestEntity();

        // Build BLAS / TLAS for ray-traced shadows. Gated on RayShadowsSupported
        // inside InitRayQuery — safe to call even when ray queries aren't available.
        InitRayQuery();
        // Bind the TLAS into the tracers' descriptor sets — they walk it for
        // ray-traced work. The deferred lighting + transparent passes read it
        // from the scene set (registry).
        if (tlas.Handle != 0)
        {
            selection.WriteTlasDescriptor(tlas);
            // Pick + selection resolve the hit entity through the same flat
            // ShadowEntityInfo table (per-cluster instances → entity via
            // entityInfo[InstanceCustomIndex + GeometryIndex].entityIndex).
            selection.WriteEntityInfoDescriptor();
            // wavefront / rtPipeline / reStirPipeline read TLAS / shadow-info / emissive
            // from the registry-owned scene set — no per-pipeline fan-out needed.
        }

        // Scene-set registrations: every provider that exists
        // at init, matched by SceneBindings parameter name. Runtime handle changes
        // re-register at their rebuild sites; the dump shows any remaining holes.
        RegisterSceneBindings();
        Console.WriteLine(descriptorRegistry.DumpBindings());

        // Stand up the L3 render cores (eager). Each ctor registers the core into _renderCores
        // (RegisterCore) -- construction order IS the list/combo order, so Deferred (index 0) is
        // the boot default + fallback. DeferredCore's ctor builds + compiles the deferred FrameGraph
        // (which OWNS the g-buffers / depth / HDR transients and imports FinalColor) and rebinds the
        // lighting g-buffer + tonemap-HDR descriptors from the fresh transient views. The PT/RT cores
        // wrap the renderer-owned PT pipelines; the RT-pipeline-only cores (RayTrace, ReSTIR DI) are
        // built only when the device exposed the feature, so they self-exclude from the combo.
        new DeferredCore(this);
        new PathtraceComputeCore(this);
        if (rtPipeline != null)     new PathtraceRTCore(this);
        new ForwardPlusCore(this);
        // Wavefront core builds + compiles its own FrameGraph in the ctor (like DeferredCore),
        // importing the pipeline-owned set-4 buffers + the PT/Final targets.
        new WavefrontPTCore(this);
        if (reStirPipeline != null) new ReStirDICore(this);

        // Activate the boot core (index 0 = Deferred unless RequestCoreIndex ran before init).
        _activeCore = _renderCores[_desiredCoreIndex];
        _activeCore.Activate();

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
    /// writes (tile buffers, IBL, probes, transparent's TLAS) are re-issued
    /// because the new VkPipeline owns brand-new descriptor sets; scene-set
    /// bindings live on the registry and need no rewire.
    /// </summary>
    public void RebuildPbrPipelines()
    {
        if (!initialized) return;
        vk!.DeviceWaitIdle(device);

        // In-place rebuild: same pipeline objects, fresh GPU handles -- so the DeferredModule
        // (and any other holder of these refs) stays valid without rebuilding the graph.
        // SoftShadowsEnabled is a spec constant, read by Rebuild's Initialize.
        PbrDeferredPipeline.SoftShadowsEnabled = softShadowsEnabled;
        PbrDeferredPipeline.Rebuild();

        transparentPipeline.SoftShadowsEnabled = softShadowsEnabled;
        transparentPipeline.Rebuild();

        // Re-wire cross-pipeline + Renderer-owned bindings on the fresh descriptor sets.
        // Set 1 (g-buffer samplers) is no longer written by Initialize — the views live on the
        // deferred FrameGraph — so let DeferredCore rebind it from its cached transient views.
        _renderCores.OfType<DeferredCore>().FirstOrDefault()?.OnPbrPipelineRebuilt();
        PbrDeferredPipeline.WriteTileBufferDescriptors(lightCullPipeline);
        PbrDeferredPipeline.WriteProbeDescriptors();
        transparentPipeline.WriteTileBufferDescriptors(lightCullPipeline);
        transparentPipeline.WriteIblDescriptors();
        transparentPipeline.WriteProbeDescriptors();
        // TLAS / shadow-alpha / lights bindings for both PBR pipelines live on
        // the registry-owned scene set, which survives the rebuild untouched.
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

        // In-place rebuild (stable object identity, fresh GPU handles) so the DeferredModule's
        // tonemap ref stays valid. Operator is a spec constant, read by Rebuild's Initialize.
        tonemapPipeline.Operator = tonemapOperator;
        tonemapPipeline.Rebuild();
        // Rebind the fresh tonemap descriptor set to the ACTIVE core's HDR source (deferred HDR vs
        // PT ptOutColor) — fixes the old always-rebind-to-deferred bug in PT mode.
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
        var materials = new Buffer[MAX_CONCURRENT_FRAMES];
        var instances = new Buffer[MAX_CONCURRENT_FRAMES];
        var lights = new Buffer[MAX_CONCURRENT_FRAMES];
        for (uint i = 0; i < MAX_CONCURRENT_FRAMES; i++)
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

    // Centralized teardown (L1.6). Each lifetime-scoped owner (FrameRing, RenderTargets,
    // Swapchain, GraphicsDevice) frees only what it owns; the orchestrator owns the
    // *order*. The one hard rule: GraphicsDevice.Dispose() runs strictly last — it frees
    // every VkDeviceMemory block + the device itself, so everything that allocated
    // through it must already be gone. Globals.vk is a process-wide singleton — never
    // disposed here. The window is owned by Engine and disposed by Engine.Shutdown.
    public void Cleanup()
    {
        if (!initialized) return;

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
        
        
        // Render cores: DeferredCore owns the deferred FrameGraph (g-buffer / depth / HDR
        // transients) — dispose the cores before RenderTargets, which owns FinalColor + the PT /
        // selection images the graph imports. The PT / forward cores own no GPU resources.
        foreach (var core in _renderCores) core.Dispose();
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
        _bcEncoder           ?.Dispose();
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

        // RHI context — strictly last. GraphicsDevice.Dispose() destroys the
        // descriptor pool, command pool, frees every VkDeviceMemory block, then
        // tears down device → debug → surface → instance in order. Every resource
        // freed above allocated through it, so it has to outlive them all.
        gfx.Dispose();

        initialized = false;
    }
}
