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
using CadThingo.VulkanEngine.Renderer.Features.SceneAcceleration;
using CadThingo.VulkanEngine.Renderer.Features.Shared;
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
    
    //RHI object
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
    
    internal Queue             graphicsQueue  => gfx.GraphicsQueue;
    internal Queue             presentQueue   => gfx.PresentQueue;
    
    
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
    
    
    // Transitional bridges onto two features the FeatureHost now owns and builds. They exist only
    // so callers that still name the subsystem keep working: the settings panel's HDR picker, the
    // probe registration in SpawnTestProbe, and the PBR / path-trace pipelines reading
    // prefilteredCubeMipLevels + the probe cluster grid at record time. Assigned once, right after
    // BuildAll, rather than resolved per access - the pipeline reads are per-frame. Each field
    // deletes itself when its last caller stops naming the feature.
    internal IblSystem             Ibl                   = null!;
    internal ReflectionProbeSystem reflectionProbeSystem = null!;

    /// <summary>The one non-bindable IBL scalar the lighting and tracing shaders need as a uniform
    /// (roughness -> mip mapping). Read through <see cref="IIblProvider"/> at record time, so a
    /// rebake can never leave a stale copy, and so the five pipelines that need it stop reaching
    /// into IblSystem's internals for one number.</summary>
    internal uint PrefilteredCubeMipLevels => ((IIblProvider)Ibl).PrefilteredCubeMipLevels;

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
    
    
    
    // No pipelines here any more. Each one is owned by the feature that records with it: private
    // pipelines by their core, the shared tonemap + skybox by the SharedPipelines feature. What is
    // left below is only the settings panel's transitional read path into that ownership - every
    // accessor resolves through the FeatureHost and is null-tolerant, because the core that owns a
    // pipeline may be gated out on this device.
    internal TonemapPipeline?     tonemapPipeline   => _features.Get<SharedPipelines>()?.Tonemap;
    internal PTComputePipeline?   ptComputePipeline => _features.Get<PathtraceComputeCore>()?.Pipeline;
    internal WavefrontPTPipeline? wavefrontPipeline => _features.Get<WavefrontPTCore>()?.Pipeline;

    /// <summary>Current tone-map curve, for the settings panel's combo. Owned by SharedPipelines
    /// (it is a spec constant on the pipeline it builds).</summary>
    public TonemapOperator tonemapOperator => _features.Get<SharedPipelines>()?.Operator ?? TonemapOperator.Filmic;

    /// <summary>Current soft-shadow state, for the settings panel's checkbox. Owned by DeferredCore
    /// (it is a spec constant on that core's pipelines).</summary>
    public bool softShadowsEnabled => _features.Get<DeferredCore>()?.SoftShadowsEnabled ?? true;

    /// <summary>Re-runs the active core's Activate. Narrow hook for a shared-pipeline rebuild: the
    /// pipeline object survives in place, but the PT cores restart progressive accumulation in
    /// Activate, and a new tone curve is a different image.</summary>
    internal void ReactivateCore() => _activeCore.Activate();

    // Bus subscriptions, disposed in Cleanup so a torn-down renderer stops being
    // handed events (the bus outlives it - Engine owns the bus).
    private readonly List<IDisposable> eventSubscriptions = new();

    // Per-frame command buffers + sync ring (L1.4). FrameRing owns the state and the
    // acquire/submit/present cadence; Renderer keeps the former field names as
    // delegating accessors so the frame body reads them unchanged. currentFrame /
    // frameCounter advance via frameRing.Advance() at end-of-frame.
    internal FrameRing frameRing = null!;
    

    //Camera
    Camera camera;
    public Camera Camera => camera;
    
    //Sampler for gbuffers and pathtracing ImageResources (the selection mask is
    //owned by RenderTargets and reached through SelectionSystem)
    internal Sampler       gBufferSampler     => renderTargets.GBufferSampler;
    

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

        // The two spec-constant intents - tone curve and soft shadows - are handled by the features
        // that own the pipelines they rebuild (SharedPipelines and DeferredCore), each as a bake.
        // Nothing about them passes through here.
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

        scene = new Scene(vk, device, physicalDevice);//initialise scene

        imGuiUtils = new ImGuiVulkanUtils(this, (uint)gfx.QueueFamilyIndices.graphicsFamily! );
        imGuiUtils?.init(swapChainExtent.Width, swapChainExtent.Height);

        // Bind FinalColor into the ImGui viewport descriptor so the Viewport panel
        // can render the scene as an ImGui.Image. Re-bound on swapchain recreate
        // because the underlying ImageView is rebuilt.
        imGuiUtils?.WriteViewportDescriptor(renderTargets.FinalColor.ImageView);

        // Per-frame command buffers + sync ring.
        frameRing = new FrameRing(gfx, RenderConfig.MAX_CONCURRENT_FRAMES);
        frameRing.CreateCommandBuffers(swapChainImages.Length);
        frameRing.CreateSyncObjects();

        // Stand up every feature the device can run: construct (gated) -> wire -> Initialize, in
        // descriptor Order. This method names none of them; a new technique is a new file whose
        // module initializer put a descriptor in the catalog before Main ran. The manifest is the
        // replacement for the boot-order list that used to live right here, and it shows what the
        // gates excluded on THIS device, which a source list never could.
        //
        // Ahead of the scene setup and the registry cross-check below, because features are now
        // providers: IBL and the probes publish their own bindings during Initialize, and the probe
        // registry has to exist before an entity carrying a probe component is spawned.
        _features = new FeatureHost(Gpu, this);
        _features.BuildAll(renderTargets.Snapshot);
        Console.WriteLine(_features.Dump());
        Ibl                   = _features.Get<IblSystem>()!;
        reflectionProbeSystem = _features.Get<ReflectionProbeSystem>()!;
        _sceneAs              = _features.Get<SceneAS>();   // null when the device gated it out

        CreateTestEntity();

        // Scene-set registrations: every provider that exists
        // at init, matched by SceneBindings parameter name. Runtime handle changes
        // re-register at their rebuild sites; the dump shows any remaining holes.
        // (The FeatureIBL / FeatureEnv sets and the AS side tables are absent here on purpose -
        // IBL, the probes and SceneAS register their own during BuildAll above.)
        RegisterSceneBindings();
        RegisterPathTraceIoBindings();
        Console.WriteLine(descriptorRegistry.DumpBindings());

        // Cross-check every migrated pipeline's reflected bindings against what the registry owns
        // and was handed. Runs here because it needs both sides complete: pipelines built above,
        // providers registered on the two lines before. Throws on a real mismatch.
        Console.WriteLine(descriptorRegistry.Validate(ReflectedPrograms()));
        var lc = gfx.LayoutCache;
        Console.WriteLine($"[layout-cache] set layouts {lc.SetLayoutCount}/{lc.SetLayoutRequests} distinct, " +
                          $"pipeline layouts {lc.PipelineLayoutCount}/{lc.PipelineLayoutRequests} distinct");

        // Activate the boot core (lowest Order = Deferred, unless RequestCoreIndex ran before init).
        _activeCore = RenderCores[_desiredCoreIndex];
        _activeCore.Activate();

        // Last, so no handler can fire against half-built pipelines.
        SubscribeToEvents();

        initialized = true;
    }

    private void CreateTestEntity()
    {
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
    }

    // Every reflected program the renderer's pipelines resolved, for the registry cross-check.
    // Found by walking PipelineBase-typed fields rather than a hand-kept list: a new pipeline joins
    // the check by existing, not by someone remembering to add it. Now that most pipelines are
    // owned by the feature that records them, the walk covers this object AND every built feature -
    // otherwise migrating a pipeline into its core would quietly drop it out of the check.
    private IEnumerable<ProgramUse> ReflectedPrograms()
        => new object[] { this }.Concat(_features.All)
            .SelectMany(PipelinesOn)
            .Where(p => p?.ReflectedProgram != null)
            .Select(p => new ProgramUse(p!.ReflectedProgram!, p.PrivateSetIndices))
            .DistinctBy(u => u.Program);

    // The PipelineBase-typed instance fields of one object, whatever their access level. Fields
    // only, deliberately: a property could resolve through the feature host and re-enter the walk.
    private static IEnumerable<PipelineBase?> PipelinesOn(object owner)
        => owner.GetType()
            .GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)
            .Where(f => typeof(PipelineBase).IsAssignableFrom(f.FieldType))
            .Select(f => f.GetValue(owner) as PipelineBase);

    // The progressive-accumulation IO pair (FeaturePTIO, set 5), shared by all tracers 
    internal void RegisterPathTraceIoBindings()
    {
        descriptorRegistry.RegisterImage("accumulator", renderTargets.PtAccumulator.ImageView, ImageLayout.General);
        descriptorRegistry.RegisterImage("outColor",    renderTargets.PtOutColor.ImageView,    ImageLayout.General);
    }

    /// Centralized teardown. scoped renderer elements destroy respect vk handles and other unmanaged resources 
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

        //  Mesh pool (global VB/IB)
        Engine.ResourceManager.Dispose();
        
        // Dear ImGUI dispose
        imGuiUtils?.Dispose();
        // Every feature, in reverse Order. 
        _features.Dispose();
        renderTargets.Dispose();

        // GpuScene buffers (light SSBO today) - the GPU mirror of Scene's render data.
        gpuScene.Dispose();

        // Swap chain + image views
        swapchain.Dispose();

        // RHI context - strictly last. 
        gfx.Dispose();

        initialized = false;
    }
}
