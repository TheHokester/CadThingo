using System.Numerics;
using CadThingo.Graphics.Rendering;
using CadThingo.VulkanEngine.GLTF;
using CadThingo.VulkanEngine.ImGui;
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
    internal Vk? vk = Globals.vk;
    public bool initialized = false;
    
    private bool enableValidationLayers = true;
    private readonly string[] ValidationLayers =
    [
        "VK_LAYER_KHRONOS_validation"
    ];
    private ExtDebugUtils? debugUtils;
    private DebugUtilsMessengerEXT debugMessenger;
    private IWindow? window;
    private Instance instance;
    
    private KhrSurface? khrSurface;
    private SurfaceKHR surface;
    
    internal PhysicalDevice physicalDevice;
    // List (not array) so AddSupportedOptionalExtensions can actually mutate it.
    // Required swapchain seeds the list; optional extensions append after device pick.
    private readonly List<string> deviceExtensions = new() { KhrSwapchain.ExtensionName };
    internal Device device;

    // Silk.NET 2.23 ships no KhrRayQuery wrapper (ray-query has zero host functions —
    // it's a pure SPIR-V capability), so we hardcode the extension string ourselves.
    private const string KhrRayQueryExtensionName = "VK_KHR_ray_query";
    
    //world scene
    private Scene scene;
    public Scene Scene => scene;
    private Entity* testEntity;
    
    private QueueFamilyIndices queueFamilyIndices;
    internal Queue graphicsQueue;
    private Queue presentQueue;
    private Queue computeQueue;
    private Queue transferQueue;
    internal bool descriptorIndexEnabled = false;
    private bool robustness2Enabled = false;
    private bool accelerationStructureEnabled = false;
    private bool rayQueryEnabled = false;

    /// <summary>
    /// True iff both KHR_acceleration_structure and KHR_ray_query are enabled with their
    /// device features active. Gates the ray-traced shadow path; callers should fall back
    /// to a non-shadowed code path when false.
    /// </summary>
    public bool RayShadowsSupported => rayQueryEnabled && accelerationStructureEnabled;

    
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

    // Specialization-constant gate for soft (PCSS-style) ray-queried shadows.
    // Threaded into PbrDeferredPipeline.SoftShadowsEnabled at construction time.
    public bool softShadowsEnabled = true;

    // Tone-map curve selector — threaded into TonemapPipeline.Operator at
    // construction time as a specialization constant. Toggling requires a
    // pipeline rebuild.
    public TonemapOperator tonemapOperator = TonemapOperator.Filmic;
    
    //Command pool and buffers
    CommandPool commandPool;
    CommandBuffer[] commandBuffers;
    
    CullingSystem cullingSystem;
    
    //Camera
    Camera camera;
    public Camera Camera => camera;

    //Sync objects
    Semaphore[] imageAvailableSemaphores;
    Semaphore[] renderFinishedSemaphores;
    Fence[] inFlightFences;
    uint currentFrame;
    
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

    internal DescriptorPool descriptorPool;
     
    
    private static unsafe uint DebugCallBack(
        DebugUtilsMessageSeverityFlagsEXT severity,
        DebugUtilsMessageTypeFlagsEXT types,
        DebugUtilsMessengerCallbackDataEXT* data,
        void* userData)
    {
        var message = SilkMarshal.PtrToString((nint)data->PMessage);
        Console.WriteLine($"[VALIDATION LAYER:] {message}");
        return Vk.False;
        
    }
    
    
    
    public void Initialize()
    {
        CreateInstance();
        SetupDebugMessenger(enableValidationLayers);
        CreateSurface();
        PickPhysicalDevice();
        CreateLogicalDevice();
        CreateSwapChain();
        // Initial render extent tracks swapchain extent. ViewportPanel can later
        // shrink it via EditorState.RequestedRenderExtent → ResizeRenderTargets.
        renderExtent = swapChainExtent;
        CreateImageViews();
        SetupDynamicRendering();

        CreateDescriptorPool();
        Engine.ResourceManager.Initialize(this);

        CreateCommandPool();
        // ImageResource objects only — actual VkImage + ImageView allocation happens
        // inside graph.Compile() called from SetupDeferredRenderer below.
        CreateDepthResources();
        CreateGBufferResources();
        CreateGBufferSampler();

        // IBL images allocated up-front, cleared to black. The PBR lighting set
        // binds them unconditionally; the compute bake passes fill the content
        // when an HDR is loaded via LoadEnvironmentHdr.
        CreateIblResources();
        CreateIblBakePipelines();
        // BRDF LUT is view-independent — bake once at init and reuse for every
        // environment that gets loaded later.
        BakeBrdfLut();

        // ── Pipelines that don't depend on allocated g-buffer image views ──
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

        // ── Lighting + light-cull pipelines (depend on allocated g-buffer / lights SSBO) ──
        PbrDeferredPipeline = new PbrDeferredPipeline(this) { SoftShadowsEnabled = softShadowsEnabled };
        PbrDeferredPipeline.Initialize();

        lightCullPipeline = new LightCullPipeline(this);
        lightCullPipeline.Initialize();

        // Wire light-cull tile buffers into the PBR lighting set now that both exist.
        PbrDeferredPipeline.WriteTileBufferDescriptors(lightCullPipeline);

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
        transparentPipeline.WriteSharedLightingDescriptors(PbrDeferredPipeline, lightCullPipeline);
        // IBL bindings live on Renderer-owned VkImages that exist before any
        // pipeline initializes; write them straight after Initialize.
        transparentPipeline.WriteIblDescriptors();

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
            // Bind the ShadowEntityInfo SSBO + global vb/ib for the alpha-test
            // path in the PBR lighting shadow rays. Has to happen after
            // InitRayQuery because the SSBO is allocated inside RebuildTlas.
            PbrDeferredPipeline.WriteShadowAlphaDescriptors();
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

        // Wire ResourceManager → upload viking_room mesh → create entity with transform + mesh.



        // glTF demo: DamagedHelmet to the side of viking_room so both render together.
        // Entity* blue_agent = GltfLoader.Load(
        //     @"C:\Users\jamie\RiderProjects\CadThingo\CadThingo\Assets\Models\props\sas_blue.glb",
        //     "BlueAgent", Engine.ResourceManager, this, scene);
        // blue_agent->GetComponent<TransformComponent>()?.SetScale(new Vector3(25f));
        // blue_agent->GetComponent<TransformComponent>()?.SetRotation(Quaternion.CreateFromYawPitchRoll(0, 0, MathF.PI / 2));
        // blue_agent->GetComponent<TransformComponent>()?.SetPosition(new Vector3(6f, 0f, 0f));
        //
        
        // Entity* helmetRoot = GltfLoader.Load(
        //     @"C:\Users\jamie\RiderProjects\CadThingo\CadThingo\Assets\Models\DamagedHelmet.glb",
        //     "DamagedHelmet", Engine.ResourceManager, this, scene);
        // var helmetTransform = helmetRoot->GetComponent<TransformComponent>();
        // helmetTransform?.SetPosition(new Vector3(0f, 1f, 0f));
        //
        // Phase 2 smoke test — three LightComponents (directional / point / spot)
        // confirming the per-frame StructuredBuffer<PbrLight> path lights the helmet.
        
        //Sponza scene
        // Entity* sponza = GltfLoader.Load(
        //     @"C:\Users\jamie\RiderProjects\CadThingo\CadThingo\Assets\Models\Environment\sponza.glb",
        //     "Sponza", Engine.ResourceManager, this, scene);
        // var sponzaTransform = sponza->GetComponent<TransformComponent>();
        // sponzaTransform?.SetPosition(new Vector3(0f, 0f, 0f));
        //
        
        //Chess scene 
        // Entity* chessScene = GltfLoader.Load(@"C:\Users\jamie\RiderProjects\CadThingo\CadThingo\Assets\Models\Environment\ABeautifulGame.glb",
        //     "ChessScene", Engine.ResourceManager, this, scene);
        
        //930 turbo
        Entity* turbo930 = GltfLoader.Load(
            @"C:\Users\jamie\RiderProjects\CadThingo\CadThingo\Assets\Models\Props\free_1975_porsche_911_930_turbo.glb", 
            "Turbo930", Engine.ResourceManager, this, scene);
        turbo930->GetComponent<TransformComponent>()?.SetPosition(new Vector3(-1f, 0.05f, 7.7f));
        turbo930->GetComponent<TransformComponent>()?.SetRotation(Quaternion.CreateFromYawPitchRoll(4.25f, 0, 0));
        
        Entity* Bistro = GltfLoader.Load(@"C:\Users\jamie\RiderProjects\CadThingo\CadThingo\Assets\Models\Environment\Bistro_Godot.glb",
            "Bistro", Engine.ResourceManager, this, scene);
        
        SpawnTestLights();
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
        transparentPipeline.WriteSharedLightingDescriptors(PbrDeferredPipeline, lightCullPipeline);
        transparentPipeline.WriteIblDescriptors();
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

        // ── Frame sync ──────────────────────────────────────────
        for (var i = 0; i < MAX_CONCURRENT_FRAMES; i++)
        {
            if (renderFinishedSemaphores[i].Handle != 0)
                vk.DestroySemaphore(device, renderFinishedSemaphores[i], null);
            if (imageAvailableSemaphores[i].Handle != 0)
                vk.DestroySemaphore(device, imageAvailableSemaphores[i], null);
            if (inFlightFences[i].Handle != 0)
                vk.DestroyFence(device, inFlightFences[i], null);
        }
        
        // ── ECS-owned native memory ─────────────────────────────
        if (testEntity != null)
        {
            Entity.Destroy(testEntity);
            testEntity = null;
        }

        // ── Ray-query AS handles + scratch + instance buffer ────
        // Must come before ResourceManager.Dispose so the BLAS storage destroys cleanly
        // (BLAS doesn't reference VB/IB after build, but order is least-surprising this way).
        CleanupRayQuery();

        // ── Mesh pool (global VB/IB) ────────────────────────────
        Engine.ResourceManager.Dispose();
        
        //── ImGUI dispose ────────────────────────────────────────
        imGuiUtils?.Dispose();
        
        
        // ── Render graph + size-dependent attachments ───────────
        // RenderGraph.Dispose disposes resources it owns; ImageResource.Dispose
        // is idempotent so the explicit calls below are safe redundant cleanup.
        scene?.renderGraph?.Dispose();
        depthImageResource?.Dispose();
        gBufferPosition?.Dispose();
        gBufferNormal?.Dispose();
        gBufferAlbedo?.Dispose();
        gBufferMaterial?.Dispose();
        gBufferEmissive?.Dispose();

        // ── G-buffer sampler ────────────────────────────────────
        if (gBufferSampler.Handle != 0) vk.DestroySampler(device, gBufferSampler, null);

        // ── IBL images + samplers ───────────────────────────────
        CleanupIblResources();

        // ── Pipelines (each pipeline disposes its own buffers, sets, layouts) ─
        skyboxPipeline     ?.Dispose();
        transparentPipeline?.Dispose();
        tonemapPipeline    ?.Dispose();
        PbrDeferredPipeline?.Dispose();
        lightCullPipeline  ?.Dispose();
        drawCullPipeline   ?.Dispose();
        geometryPipeline   ?.Dispose();

        // ── Descriptor pool (frees the descriptor sets owned by the pipelines) ─
        if (descriptorPool.Handle != 0) vk.DestroyDescriptorPool(device, descriptorPool, null);

        // ── Command pool (frees buffers) ────────────────────────
        if (commandPool.Handle != 0) vk.DestroyCommandPool(device, commandPool, null);

        // ── Swap chain + image views ────────────────────────────
        CleanupSwapChain();

        // ── Device, debug, surface, instance ────────────────────
        vk.DestroyDevice(device, null);
        if (enableValidationLayers && debugUtils != null)
            debugUtils.DestroyDebugUtilsMessenger(instance, debugMessenger, null);
        khrSurface?.DestroySurface(instance, surface, null);
        vk.DestroyInstance(instance, null);

        initialized = false;
    }
    
    private void CreateInstance()
    {
        
        
        var appNamePtr = SilkMarshal.StringToPtr("App");
        var engineNamePtr = SilkMarshal.StringToPtr("Engine");

        var appInfo = new ApplicationInfo()
        {
            SType = StructureType.ApplicationInfo,
            PApplicationName = (byte*)appNamePtr,
            ApplicationVersion = new Version32(1, 0, 0),
            ApiVersion = Vk.Version13,
            EngineVersion = new Version32(1, 0, 0),
            PEngineName = (byte*)engineNamePtr
        };


        var createInfo = new InstanceCreateInfo()
        {
            SType = StructureType.InstanceCreateInfo,
            PApplicationInfo = &appInfo,
        };
        var extensions = GetRequiredExtensions();
        createInfo.EnabledExtensionCount = (uint)extensions.Length;
        createInfo.PpEnabledExtensionNames = (byte**)SilkMarshal.StringArrayToPtr(extensions);
        
        //enable validation layers if requested
        ValidationFeaturesEXT validationFeatures = new(){SType = StructureType.ValidationFeaturesExt}; 
        ValidationFeatureEnableEXT[] enabledValidationFeatures = [
        ];
        
        if (enableValidationLayers)
        {
            if (!CheckValidationLayerSupport())
            {
                throw new Exception("Validation layers requested, but not available!");
            }
            
            createInfo.EnabledLayerCount = (uint)ValidationLayers.Length;
            byte** layerNames = (byte**)SilkMarshal.StringArrayToPtr(ValidationLayers);
            createInfo.PpEnabledLayerNames = layerNames;
            
            //Keep validation output quiet by default (no DebugPrint feature)
            fixed(ValidationFeatureEnableEXT* featurePtr = enabledValidationFeatures)
            {
                validationFeatures.EnabledValidationFeatureCount = (uint)enabledValidationFeatures.Length;
                validationFeatures.PEnabledValidationFeatures = featurePtr;
            }
            
            createInfo.PNext = &validationFeatures;
        }
        
        //create instance 
        if (vk!.CreateInstance(&createInfo, null, out instance) != Result.Success)
        {
            throw new Exception("Failed to create Vulkan instance");
        }
        SilkMarshal.Free(appNamePtr);
        SilkMarshal.Free(engineNamePtr);
        SilkMarshal.Free((nint)createInfo.PpEnabledExtensionNames);
        if(enableValidationLayers)
            SilkMarshal.Free((nint)createInfo.PpEnabledLayerNames);
    }

    private void SetupDebugMessenger(bool enableValidation)
    {
        if (!enableValidation) return;

        if (!vk!.TryGetInstanceExtension(instance, out debugUtils)) return;
        //create messenger here
        var createInfo = new DebugUtilsMessengerCreateInfoEXT()
        {
            SType = StructureType.DebugUtilsMessengerCreateInfoExt,
            MessageSeverity = DebugUtilsMessageSeverityFlagsEXT.VerboseBitExt |
                              DebugUtilsMessageSeverityFlagsEXT.WarningBitExt |
                              DebugUtilsMessageSeverityFlagsEXT.ErrorBitExt |
                              DebugUtilsMessageSeverityFlagsEXT.InfoBitExt,
            MessageType = DebugUtilsMessageTypeFlagsEXT.GeneralBitExt |
                          DebugUtilsMessageTypeFlagsEXT.ValidationBitExt |
                          DebugUtilsMessageTypeFlagsEXT.PerformanceBitExt,
            PfnUserCallback = (PfnDebugUtilsMessengerCallbackEXT)DebugCallBack
        };

        if (debugUtils!.CreateDebugUtilsMessenger(instance, &createInfo, null, out debugMessenger) != Result.Success)
        {
            throw new Exception("Failed to create debug messenger");
        }
    }
    
    private void CreateSurface()
    {
        if(!vk!.TryGetInstanceExtension<KhrSurface>(instance, out khrSurface))
            throw new Exception("KHR Surface ext not found");
        surface = window!.VkSurface!.Create<AllocationCallbacks>(instance.ToHandle(), null).ToSurface();
    }

    private void PickPhysicalDevice()
    {
        uint deviceCount = 0;
        vk!.EnumeratePhysicalDevices(instance, &deviceCount, null);

        if (deviceCount == 0)
            throw new Exception("No Vulkan devices found");

        var devices = stackalloc PhysicalDevice[(int)deviceCount];
        vk!.EnumeratePhysicalDevices(instance, &deviceCount, devices); 
        
        //prioritise discrete GPUs, over integrated GPUs 
        //first collect all suitable devices with their suitability scores
        Dictionary<PhysicalDevice, int> deviceSuitability = new();
        
        
        for (var i = 0; i < deviceCount; i++)
        {
            var device = devices[i];
            var deviceProperites = vk!.GetPhysicalDeviceProperties(device);
            Console.WriteLine("Checking Device: " + SilkMarshal.PtrToString((nint)deviceProperites.DeviceName) + " (Type: " + deviceProperites.DeviceType + " )");
            
            //check for vulkan 1.3 support
            bool supportsVulkan1_3 = deviceProperites.ApiVersion >= Vk.Version13;
            if (!supportsVulkan1_3)
            {
                Console.WriteLine("----> Device does not support Vulkan 1.3");
                continue;
            }
            
            //Check queue families
            QueueFamilyIndices indices = FindQueueFamilies(device);
            bool supportsGraphics = indices.IsComplete();
            if (!supportsGraphics)
            {
                Console.WriteLine("----> Device Missing required queue families");
                continue;
            }
            
            //check device extensions
            bool supportsAllRequiredExtensions = CheckDeviceExtensionSupport(device);
            if (!supportsAllRequiredExtensions)
            {
                Console.WriteLine("----> Device Missing required extensions");
                continue;
            }
            
            //Check swapchain support
            SwapChainSupportDetails swapChainSupport = QuerySwapChainSupport(device);
            bool swapChainAdequate = swapChainSupport.Formats.Length != 0 && swapChainSupport.PresentModes.Length != 0;
            if (!swapChainAdequate)
            {
                Console.WriteLine("----> Inadequate swapchain support");
                continue;
            }
            
            
            //Check for required features
            var features13 = new PhysicalDeviceVulkan13Features()
            {
                SType = StructureType.PhysicalDeviceVulkan13Features,
            }; 
            var features2 = new PhysicalDeviceFeatures2(StructureType.PhysicalDeviceFeatures2, &features13);
            vk!.GetPhysicalDeviceFeatures2(device, &features2);
            if (!features13.DynamicRendering)
            {
                Console.WriteLine("----> Device does not support dynamic rendering");
                continue;
            }
            
            
            //Calculate suitability score
            int score = 0;
            if (deviceProperites.DeviceType == PhysicalDeviceType.DiscreteGpu)
            {
                score += 1000;
                Console.WriteLine("----> Discrete GPU + 1000 points");
            } else if (deviceProperites.DeviceType == PhysicalDeviceType.IntegratedGpu)
            {
                score += 100;
                Console.WriteLine("----> Integrated GPU + 100 points");
            }
            //Add points for memory size (more VRAM = more points)
            vk!.GetPhysicalDeviceMemoryProperties(device, out var memProps);
            for(var m = 0; m < memProps.MemoryHeapCount; m++)
                if ((memProps.MemoryHeaps[m].Flags & MemoryHeapFlags.DeviceLocalBit) != 0)
                {
                    score += (int)memProps.MemoryHeaps[m].Size / (1024 * 1024 * 1024);
                    Console.WriteLine("----> Device has " + (int)memProps.MemoryHeaps[m].Size / (1024 * 1024 * 1024) + "GB VRAM");
                    break;
                }
            
            Console.WriteLine("----> Device Suitability Score: " + score);
            deviceSuitability.Add(device, score);

        } 
        if (!deviceSuitability.Count.Equals(0))
        {
            //select the device with the highest score
           physicalDevice = deviceSuitability.OrderByDescending(x => x.Value).First().Key;
           vk!.GetPhysicalDeviceProperties(physicalDevice, out var deviceProperties);
           Console.WriteLine("Selected Device: " + *deviceProperties.DeviceName + 
                             " (Type: " + deviceProperties.DeviceType + " Score: " + deviceSuitability.First().Value + ")");
        }
        //Store queue family indices for selected device
        queueFamilyIndices = FindQueueFamilies(physicalDevice);
        
        //add supported optional extensions
        AddSupportedOptionalExtensions();

        return;
    }


    // Optional device extensions tried at device-create time. Each entry is the *real*
    // VK extension string (lowercase, no _EXTENSION_NAME macro suffix). Anything Silk.NET
    // wraps as a class exposes ExtensionName; ray-query has no wrapper so we use the const.
    private readonly string[] optionalDeviceExtensions =
    {
        KhrDynamicRendering.ExtensionName,
        KhrGetPhysicalDeviceProperties2.ExtensionName,
        KhrDynamicRenderingLocalRead.ExtensionName,
        KhrDeferredHostOperations.ExtensionName,           // required by acceleration_structure
        KhrAccelerationStructure.ExtensionName,
        KhrRayTracingPipeline.ExtensionName,
        KhrRayQueryExtensionName,                          // pure SPIR-V cap, no Silk wrapper
        "VK_KHR_depth_stencil_resolve",
        "VK_EXT_descriptor_indexing",
        "VK_EXT_robustness2",
        "VK_EXT_shader_tile_image",
    };
    
    private void AddSupportedOptionalExtensions()
    {
        // Two-call pattern: first call returns the count, second populates the array.
        uint extensionCount = 0;
        vk!.EnumerateDeviceExtensionProperties(physicalDevice, (byte*)null, &extensionCount, null);
        var available = new ExtensionProperties[extensionCount];
        fixed (ExtensionProperties* pAvailable = available)
            vk!.EnumerateDeviceExtensionProperties(physicalDevice, (byte*)null, &extensionCount, pAvailable);

        var avail = new HashSet<string>();
        foreach (var ext in available)
            avail.Add(SilkMarshal.PtrToString((nint)ext.ExtensionName));

        foreach (var ext in optionalDeviceExtensions)
        {
            if (avail.Contains(ext))
            {
                deviceExtensions.Add(ext);
                Console.WriteLine("----> Added optional extension: " + ext);
            }
            else
            {
                Console.WriteLine("----> Optional extension not supported: " + ext);
            }
        }
    }
    
    private void CreateLogicalDevice()
    {
        //create queue create info for each queue family
        List<DeviceQueueCreateInfo> queueCreateInfos = new();
        HashSet<uint> uniqueQueueFamilies = new HashSet<uint>()
        {
            queueFamilyIndices.graphicsFamily.Value,
            queueFamilyIndices.presentFamily.Value,
            queueFamilyIndices.computeFamily.Value,
            queueFamilyIndices.transferFamily.Value
        };
        
        float queuePriority = 1.0f;
        foreach (var qf in uniqueQueueFamilies)
        {
            DeviceQueueCreateInfo queueCreateInfo = new()
            {
                SType = StructureType.DeviceQueueCreateInfo,
                QueueFamilyIndex = qf,
                PQueuePriorities = &queuePriority,
                QueueCount = 1
            };
            queueCreateInfos.Add(queueCreateInfo);
        }
        
        //Query supported features. We use the 1.2 / 1.3 omnibus structs in place of
        //the individual Timeline / MemoryModel / BufferAddress / 8BitStorage /
        //DescriptorIndexing feature structs because Vulkan forbids mixing them with
        //the 1.2 features struct in the same pNext chain (VUID-VkDeviceCreateInfo-
        //pNext-02830) and the omnibus already exposes every individual field we need.
        var coreSupported = new PhysicalDeviceFeatures2
        {
            SType = StructureType.PhysicalDeviceFeatures2
        };
        var vulkan11Supported = new PhysicalDeviceVulkan11Features
        {
            SType = StructureType.PhysicalDeviceVulkan11Features
        };
        var vulkan12Supported = new PhysicalDeviceVulkan12Features
        {
            SType = StructureType.PhysicalDeviceVulkan12Features
        };
        var vulkan13Supported = new PhysicalDeviceVulkan13Features
        {
            SType = StructureType.PhysicalDeviceVulkan13Features
        };
        var robust2Supported = new PhysicalDeviceRobustness2FeaturesEXT
        {
            SType = StructureType.PhysicalDeviceRobustness2FeaturesExt
        };
        var accelerationStructureFeaturesSupported = new PhysicalDeviceAccelerationStructureFeaturesKHR
        {
            SType = StructureType.PhysicalDeviceAccelerationStructureFeaturesKhr
        };
        var rayQueryFeaturesSupported = new PhysicalDeviceRayQueryFeaturesKHR
        {
            SType = StructureType.PhysicalDeviceRayQueryFeaturesKhr
        };

        coreSupported.PNext             = &vulkan11Supported;
        vulkan11Supported.PNext         = &vulkan12Supported;
        vulkan12Supported.PNext         = &vulkan13Supported;
        vulkan13Supported.PNext         = &robust2Supported;
        robust2Supported.PNext          = &accelerationStructureFeaturesSupported;
        accelerationStructureFeaturesSupported.PNext = &rayQueryFeaturesSupported;
        rayQueryFeaturesSupported.PNext = null;

        vk!.GetPhysicalDeviceFeatures2(physicalDevice, &coreSupported);

        bool supported = (
            coreSupported.Features.SamplerAnisotropy &&
            coreSupported.Features.MultiDrawIndirect &&
            coreSupported.Features.DrawIndirectFirstInstance &&
            vulkan11Supported.ShaderDrawParameters &&
            vulkan12Supported.TimelineSemaphore &&
            vulkan12Supported.VulkanMemoryModel &&
            vulkan12Supported.BufferDeviceAddress &&
            vulkan12Supported.DrawIndirectCount &&
            vulkan13Supported.DynamicRendering &&
            vulkan13Supported.Synchronization2);
        if(!supported) throw new Exception("Device does not support required features");


        //enable required features (verified to be supported)
        vk!.GetPhysicalDeviceFeatures2(physicalDevice, out var features);
        features.SType = StructureType.PhysicalDeviceFeatures2;
        features.Features.SamplerAnisotropy = true;
        features.Features.DepthBiasClamp = coreSupported.Features.DepthBiasClamp ? true : false;
        // Required by the GPU-driven cull pass: indirect draws emit one
        // VkDrawIndexedIndirectCommand per visible mesh, all consumed by a single
        // vkCmdDrawIndexedIndirectCount.
        features.Features.MultiDrawIndirect = true;
        // The cull shader writes a non-zero firstInstance into each indirect command
        // so the geometry VS can resolve SV_InstanceID -> instances[slot]. Without
        // this feature the spec requires firstInstance=0 and drivers silently clamp
        // it, making every primitive read instances[0].
        features.Features.DrawIndirectFirstInstance = true;
        //rayQuery shader uses indexing into a large sampled-image array.
        if (coreSupported.Features.ShaderSampledImageArrayDynamicIndexing)
            features.Features.ShaderSampledImageArrayDynamicIndexing = true;

        //vulkan 1.1
        PhysicalDeviceVulkan11Features vulkan11Features = new(){SType = StructureType.PhysicalDeviceVulkan11Features};
        vulkan11Features.ShaderDrawParameters = true;

        //vulkan 1.2 — rolls in Timeline, MemoryModel, BufferDeviceAddress, 8-bit storage,
        //DescriptorIndexing, and DrawIndirectCount (used by vkCmdDrawIndexedIndirectCount).
        PhysicalDeviceVulkan12Features vulkan12Features = new(){SType = StructureType.PhysicalDeviceVulkan12Features};
        vulkan12Features.TimelineSemaphore                = true;
        vulkan12Features.VulkanMemoryModel                = true;
        vulkan12Features.VulkanMemoryModelDeviceScope     = vulkan12Supported.VulkanMemoryModelDeviceScope;
        vulkan12Features.BufferDeviceAddress              = true;
        vulkan12Features.StorageBuffer8BitAccess          = vulkan12Supported.StorageBuffer8BitAccess;
        vulkan12Features.DrawIndirectCount                = true;

        //Descriptor-indexing fields live on the 1.2 features struct. The DescriptorIndexing
        //master toggle is required when VK_EXT_descriptor_indexing is in the enabled
        //extension list (VUID-VkDeviceCreateInfo-ppEnabledExtensionNames-02833).
        var descriptorIndexingEnabled = false;
        if (vulkan12Supported.ShaderSampledImageArrayNonUniformIndexing)
        {
            vulkan12Features.ShaderSampledImageArrayNonUniformIndexing = true;
            descriptorIndexingEnabled = true;
        }
        if (vulkan12Supported.RuntimeDescriptorArray)
            vulkan12Features.RuntimeDescriptorArray = true;
        if (descriptorIndexingEnabled)
        {
            if (vulkan12Supported.DescriptorBindingPartiallyBound)
                vulkan12Features.DescriptorBindingPartiallyBound = true;
            if (vulkan12Supported.DescriptorBindingUpdateUnusedWhilePending)
                vulkan12Features.DescriptorBindingUpdateUnusedWhilePending = true;
        }
        if (vulkan12Supported.DescriptorBindingSampledImageUpdateAfterBind)
            vulkan12Features.DescriptorBindingSampledImageUpdateAfterBind = true;
        if (vulkan12Supported.DescriptorBindingUniformBufferUpdateAfterBind)
            vulkan12Features.DescriptorBindingUniformBufferUpdateAfterBind = true;
        if (vulkan12Supported.DescriptorBindingUpdateUnusedWhilePending)
            vulkan12Features.DescriptorBindingUpdateUnusedWhilePending = true;
        // The descriptor-indexing master toggle gates the whole feature family and
        // is required when the legacy VK_EXT_descriptor_indexing extension is enabled.
        if (descriptorIndexingEnabled)
            vulkan12Features.DescriptorIndexing = true;

        //vulkan 1.3
        PhysicalDeviceVulkan13Features vulkan13Features = new(){SType = StructureType.PhysicalDeviceVulkan13Features};
        vulkan13Features.DynamicRendering = true;
        vulkan13Features.Synchronization2 = true;
        
        //helper to verify that an extension is enabled
        bool hasExtension(string name)
        {
            return deviceExtensions.Contains(name);
        }
        
        //prepare robustness2 featureset if the extension is enabled
        var hasRobust2 = hasExtension("VK_EXT_robustness2");
        PhysicalDeviceRobustness2FeaturesEXT robust2Enable = new() {SType = StructureType.PhysicalDeviceRobustness2FeaturesExt};
        if (hasRobust2)
        {
            if (robust2Supported.RobustBufferAccess2)
                robust2Enable.RobustBufferAccess2 = true;
            if (robust2Supported.RobustImageAccess2)
                robust2Enable.RobustImageAccess2 = true;
            if(robust2Supported.NullDescriptor)
                robust2Enable.NullDescriptor = true;
        }
        
        //prepare acceleration structure features if extension is enabled and supported
        var hasAccelerationStructure = hasExtension(KhrAccelerationStructure.ExtensionName);
        PhysicalDeviceAccelerationStructureFeaturesKHR accelerationstructureEnable = new(){SType = StructureType.PhysicalDeviceAccelerationStructureFeaturesKhr};
        if (hasAccelerationStructure)
        {
            accelerationstructureEnable.AccelerationStructure = true;
        }
        
        //prepare rayQuery features if extension is enabled and supported
        var hasRayQuery = hasExtension(KhrRayQueryExtensionName);
        PhysicalDeviceRayQueryFeaturesKHR rayQueryEnable = new(){SType = StructureType.PhysicalDeviceRayQueryFeaturesKhr};
        if (hasRayQuery)
        {
            rayQueryEnable.RayQuery = true;
        }
        
        //chain all features together: 1.1 -> 1.2 -> 1.3 -> optional ext structs
        features.PNext = &vulkan11Features;
        vulkan11Features.PNext = &vulkan12Features;
        vulkan12Features.PNext = &vulkan13Features;
        void** tailNext = (void**)&vulkan13Features.PNext;

        if (hasRobust2)
        {
            *tailNext = &robust2Enable;
            tailNext = (void**)&robust2Enable.PNext;
        }

        if (hasAccelerationStructure)
        {
            *tailNext = &accelerationstructureEnable;
            tailNext = (void**)&accelerationstructureEnable.PNext;
        }

        if (hasRayQuery)
        {
            *tailNext = &rayQueryEnable;
            tailNext = (void**)&rayQueryEnable.PNext;
        }

        //record which features ended up enabled
        descriptorIndexEnabled = descriptorIndexingEnabled && (vulkan12Features.DescriptorBindingPartiallyBound && vulkan12Features.DescriptorBindingUpdateUnusedWhilePending);
        robustness2Enabled = hasRobust2 && (robust2Enable.RobustBufferAccess2 || robust2Enable.RobustImageAccess2 || robust2Enable.NullDescriptor);
        accelerationStructureEnabled = hasAccelerationStructure && accelerationstructureEnable.AccelerationStructure;
        rayQueryEnabled = hasRayQuery && rayQueryEnable.RayQuery;

        Console.WriteLine($"----> RayShadowsSupported: {RayShadowsSupported} " +
                          $"(rayQuery={rayQueryEnabled}, accelStruct={accelerationStructureEnabled})");

        bool printFeatures = false;
        if (printFeatures)
        {
            Console.WriteLine("----> Device Features:");
            Console.WriteLine("----> Sampler Anisotropy: " + features.Features.SamplerAnisotropy);
            Console.WriteLine("----> Depth Bias Clamp: " + features.Features.DepthBiasClamp);
            Console.WriteLine("----> MultiDrawIndirect: " + features.Features.MultiDrawIndirect);
            Console.WriteLine("----> Vulkan 1.1 Features: " + vulkan11Features.ShaderDrawParameters);
            Console.WriteLine("----> Vulkan 1.2 Features: timeline=" + vulkan12Features.TimelineSemaphore +
                              " memoryModel=" + vulkan12Features.VulkanMemoryModel +
                              " bufferDeviceAddress=" + vulkan12Features.BufferDeviceAddress +
                              " drawIndirectCount=" + vulkan12Features.DrawIndirectCount +
                              " descriptorIndexing=" + vulkan12Features.DescriptorIndexing);
            Console.WriteLine("----> Vulkan 1.3 Features: " + vulkan13Features.DynamicRendering + " " + vulkan13Features.Synchronization2);
            Console.WriteLine("----> Robustness 2 Features: " + robust2Enable.RobustBufferAccess2 + "\n " + robust2Enable.RobustImageAccess2 + "\n "
                              + robust2Enable.NullDescriptor);
            Console.WriteLine("----> Acceleration Structure Features: " + accelerationstructureEnable.AccelerationStructure);
            Console.WriteLine("----> Ray Query Features: " + rayQueryEnable.RayQuery);
        }
        
        
        
        
        //Create logical device
        //only configure extensions here
        //validation enabled on instance layers
        fixed (DeviceQueueCreateInfo* queueInfoPtr = &queueCreateInfos.ToArray()[0])
        {
            DeviceCreateInfo deviceCreateInfo = new()
            {
                SType = StructureType.DeviceCreateInfo,
                PNext = &features,
                QueueCreateInfoCount = (uint)queueCreateInfos.Count,
                PQueueCreateInfos = queueInfoPtr,
                EnabledExtensionCount = (uint)deviceExtensions.Count,
                PpEnabledExtensionNames = (byte**)SilkMarshal.StringArrayToPtr(deviceExtensions.ToArray()),
                PEnabledFeatures = null //using pNext for features
            };
            if (vk!.CreateDevice(physicalDevice, &deviceCreateInfo, null, out device) != Result.Success)
            {
                throw new Exception("Failed to create logical device");   
            }
            //create queues
            vk!.GetDeviceQueue(device, queueFamilyIndices.graphicsFamily.Value, 0, out graphicsQueue);
            vk!.GetDeviceQueue(device, queueFamilyIndices.presentFamily.Value, 0, out presentQueue);
            vk!.GetDeviceQueue(device, queueFamilyIndices.computeFamily.Value, 0, out computeQueue);
            vk!.GetDeviceQueue(device, queueFamilyIndices.transferFamily.Value, 0, out transferQueue);
        }
        
        //
    }
    
    private bool CheckValidationLayerSupport()
    {
        uint layerCount = 0;
        vk!.EnumerateInstanceLayerProperties(&layerCount, null);

        var layers = stackalloc LayerProperties[(int)layerCount];
        vk!.EnumerateInstanceLayerProperties(&layerCount, layers);

        for (int i = 0; i < layerCount; i++)
        {
            var name = SilkMarshal.PtrToString((nint)layers[i].LayerName);
            if(name.Equals("VK_LAYER_KHRONOS_validation"))
                return true;
        }
        return false;
    }
    
    
}