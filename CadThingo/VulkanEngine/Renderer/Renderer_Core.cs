using CadThingo.Graphics.Rendering;
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
    Vk? vk = Globals.vk;
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
    
    private PhysicalDevice physicalDevice;
    private string[]? deviceExtensions = [
    KhrSwapchain.ExtensionName];
    private Device device; 
    
    //world scene
    private Scene scene;
    private Entity* testEntity;
    
    private QueueFamilyIndices queueFamilyIndices;
    private Queue graphicsQueue;
    private Queue presentQueue;
    private Queue computeQueue;
    private Queue transferQueue;
    private bool descriptorIndexEnabled = false;
    private bool robustness2Enabled = false;
    private bool accelerationStructureEnabled = false;
    private bool rayQueryEnabled = false;

    
    //swapchain fields
    private const uint MAX_CONCURRENT_FRAMES = 2;
    
    private KhrSwapchain swapChainKhr;
    private SwapchainKHR swapChain;
    private Image[] swapChainImages;
    private Format swapChainImageFormat;
    private Extent2D swapChainExtent;
    private ImageView[] swapChainImageViews;
    private ImageLayout[] swapChainImageLayouts;
    
    //dynamic rendering fields
    RenderingInfo renderingInfo;
    List<RenderingAttachmentInfo> colorAttachments;
    RenderingAttachmentInfo depthAttachment;
    
    
    //pipelines 
    PipelineLayout pipelineLayout;
    Pipeline graphicsPipeline;
    PipelineLayout geometryPipelineLayout;
    Pipeline geometryPipeline;
    PipelineLayout pbrPipelineLayout;
    Pipeline pbrLightingPipeline;
    Pipeline pbrBlendGraphicsPipeline;
    //transparent pbr pipeline for premultiplied alpha
    Pipeline pbrPremulPipeline;
    //opaque pbr pipeline varient used for mirrored offscreen pass(cull to avoid winding issues)
    Pipeline pbrPrepassGraphicsPipeline;
    //reflection pbr pipeline used for mirroring offscreen pass
    Pipeline pbrReflectionGraphicsPipeline;
    //specialized pipeline for architectural glass (windows, lamps etc ect...)
    //shared descriptors and vertex input with pbr pipelines but uses a dedicated 
    //fragment shader for the glass surface
    Pipeline glassGraphicsPipeline;
    PipelineLayout lightingPipelineLayout;
    Pipeline lightingPipeline;
    
    //fullscreen composite pipeline to draw the opaque offscreen color to the swapchain
    PipelineLayout compositePipelineLayout;
    Pipeline compositePipeline;
    DescriptorSetLayout compositeDSL;
    DescriptorSet compositeDS;
    
    //PipelineRenderingCreateInfos for lifetime management
    PipelineRenderingCreateInfo mainPipelineRenderingCreateInfo;
    PipelineRenderingCreateInfo geometryPipelineRenderingCreateInfo;
    PipelineRenderingCreateInfo pbrPipelineRenderingCreateInfo; 
    PipelineRenderingCreateInfo lightingPipelineRenderingCreateInfo; 
    PipelineRenderingCreateInfo compositePipelineRenderingCreateInfo;  
    
    //Compute pipeline
    PipelineLayout computePipelineLayout;
    Pipeline computePipeline;
    DescriptorSetLayout computeDSL;
    DescriptorPool computeDescriptorPool;
    DescriptorSet[] computeDSets;
    CommandPool computeCommandPool;
    
    //Command pool and buffers
    CommandPool commandPool;
    CommandBuffer[] commandBuffers;
    
    CullingSystem cullingSystem;
    
    //Camera
    Camera camera;
    //Uniform buffers — split per pass (Geometry pass writes model/view/proj; Lighting pass writes lights + camPos + tone-mapping params)
    private UboBuffer[] GeometryUniformBuffers = new UboBuffer[2];
    private UboBuffer[] LightingUniformBuffers = new UboBuffer[2];
    
    //Sync objects
    Semaphore[] imageAvailableSemaphores;
    Semaphore[] renderFinishedSemaphores;
    Fence[] inFlightFences;
    uint currentFrame;
    
    //upload timeline semaphore
    Semaphore uploadsTimeline;
    //tracks last timeline value that was submitted
    volatile uint lastTimelineValue;
    
    //Depth buffer + Images
    ImageResource depthImageResource;
    private ImageResource gBufferPosition;
    private ImageResource gBufferNormal;
    private ImageResource gBufferAlbedo;
    private ImageResource gBufferMaterial;
    
    private Sampler gBufferSampler;

    // Dummy 1x1 white texture for unbound PBR sampler slots (bindings 1-5 of geometry set)
    private Image dummyWhiteImage;
    private DeviceMemory dummyWhiteImageMemory;
    private ImageView dummyWhiteImageView;
    private Sampler dummyWhiteSampler;
    //store for lifetime management
    DescriptorSetLayout descriptorSetLayout;
    DescriptorSetLayout geometryDescriptorSetLayout;
    DescriptorSetLayout PBRDescriptorSetLayout; //Set 0 - Lighting UBO
    DescriptorSetLayout PBRGBufferDescriptorSetLayout;//Set 1 - G buffer descriptors
    DescriptorPool descriptorPool;
    private DescriptorSet[] geometryDescriptorSets;
    private DescriptorSet[] lightingDescriptorSets;   // per-frame, binds LightingUBO (set 0)
    private DescriptorSet gBufferDescriptorSet;       // shared, binds G-buffer samplers (set 1)
     
    
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
        CreateImageViews();
        SetupDynamicRendering();
        CreateDescriptorSetLayout();
        CreateGeometryDescriptorSetLayout();
        CreatePBRDescriptorSetLayout();
        CreateGraphicsPipeline();
        CreateGeometryPipeline();
        CreatePBRPipeline();
        //create command pool
        CreateCommandPool();
        
        //Creates image resources, no gpu alloc yet
        CreateDepthResources();
        //same
        CreateGBufferResources();
        //Create both geo and lighting uniform buffers
        CreateUniformBuffers();
        
        scene = new Scene(vk, device, physicalDevice);//initialise scene
        SetupDeferredRenderer(scene.renderGraph, swapChainExtent.Width, swapChainExtent.Height);//adds resources to render graph & compiles
        CreateGBufferSampler();
        CreateDummyWhiteTexture();
        //Create descriptor pool
        CreateDescriptorPool();
        CreateDescriptorSets();
        CreateLightingDescriptorSets();

        //Create command buffers
        CreateCommandBuffers();
        //Create sync objects
        CreateSyncObjects();
        //setup deffered rendering

        CreateTestEntity();

        initialized = true;
    }

    private void CreateTestEntity()
    {
        // Wire ResourceManager → upload cube → create entity with transform + mesh.
        Engine.ResourceManager.Initialize(this);

        Engine.ResourceManager.Load<MeshResource>(
            "cube", id => new ProceduralCubeResource(id, Engine.ResourceManager));

        Mesh* meshPtr = Engine.ResourceManager.GetMesh("cube");

        testEntity = Entity.Create("TestCube");
        testEntity->AddComponent(new TransformComponent());
        testEntity->AddComponent(new MeshComponent(meshPtr, -1));
        scene.AddEntity(testEntity);
    }

    public void Update(double d)
    {
        
        DrawFrame();
    }

    public void Cleanup()
    {
        
        vk!.DestroyDevice(device, null);
        if(enableValidationLayers)
            debugUtils.DestroyDebugUtilsMessenger(instance, debugMessenger, null);
        
        khrSurface.DestroySurface(instance, surface, null);
        vk!.DestroyInstance(instance, null);
        vk!.Dispose();
        window.Dispose();
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


    private string[] optionalDeviceExtensions =
    {
        KhrDynamicRendering.ExtensionName,
        KhrGetPhysicalDeviceProperties2.ExtensionName,
        KhrDynamicRenderingLocalRead.ExtensionName,
        KhrDeferredHostOperations.ExtensionName,
        KhrAccelerationStructure.ExtensionName,
        
        
        "VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME",
        "VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME",
        // Robustness and safety
        "VK_EXT_ROBUSTNESS_2_EXTENSION_NAME",
        // Tile/local memory friendly dynamic rendering readback
        // Shader tile image for fast tile access
        "VK_EXT_SHADER_TILE_IMAGE_EXTENSION_NAME",
        // Ray query support for ray-traced rendering
        "VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME",
        "VK_KHR_RAY_QUERY_EXTENSION_NAME"
    };
    
    private void AddSupportedOptionalExtensions()
    {
        
        uint extensionCount = 0;
        vk!.EnumerateDeviceExtensionProperties(physicalDevice, (byte*)null, &extensionCount, null);
        ExtensionProperties[] available = [];
        vk!.EnumerateDeviceExtensionProperties(physicalDevice, (byte*)null, &extensionCount, available);
        
        HashSet<string> avail = new HashSet<string>();
        foreach (var ext in available)
        {
            avail.Add(SilkMarshal.PtrToString((nint)ext.ExtensionName));
        }

        foreach (var ext in optionalDeviceExtensions)
        {
            if (avail.Contains(ext))
            {
                deviceExtensions!.Append(ext);
                Console.WriteLine("----> Added optional extension: " + ext);
            }
            else
                Console.WriteLine("----> Failed to add optional extension: " + ext);
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
        
        //Query supported features then enable them
        var coreSupported = new PhysicalDeviceFeatures2
        {
            SType = StructureType.PhysicalDeviceFeatures2
        };
        var timelineSupported = new PhysicalDeviceTimelineSemaphoreFeatures
        {
            SType = StructureType.PhysicalDeviceTimelineSemaphoreFeatures
        };
        var memoryModelSupported = new PhysicalDeviceVulkanMemoryModelFeatures
        {
            SType = StructureType.PhysicalDeviceVulkanMemoryModelFeatures
        };
        var bufferAddressSupported = new PhysicalDeviceBufferDeviceAddressFeatures
        {
            SType = StructureType.PhysicalDeviceBufferDeviceAddressFeatures
        };
        var storage8BitSupported = new PhysicalDevice8BitStorageFeatures
        {
            SType = StructureType.PhysicalDevice8BitStorageFeatures
        };
        var vulkan11Supported = new PhysicalDeviceVulkan11Features
        {
            SType = StructureType.PhysicalDeviceVulkan11Features
        };
        var vulkan13Supported = new PhysicalDeviceVulkan13Features
        {
            SType = StructureType.PhysicalDeviceVulkan13Features
        };
        
        coreSupported.PNext = &timelineSupported;
        timelineSupported.PNext = &memoryModelSupported;
        memoryModelSupported.PNext = &bufferAddressSupported;
        bufferAddressSupported.PNext = &storage8BitSupported;
        storage8BitSupported.PNext = &vulkan11Supported;
        vulkan11Supported.PNext = &vulkan13Supported;
        vulkan13Supported.PNext = null;

        vk!.GetPhysicalDeviceFeatures2(physicalDevice, &coreSupported);

        bool supported = (
            coreSupported.Features.SamplerAnisotropy &&
            timelineSupported.TimelineSemaphore &&
            memoryModelSupported.VulkanMemoryModel &&
            bufferAddressSupported.BufferDeviceAddress &&
            vulkan11Supported.ShaderDrawParameters &&
            vulkan13Supported.DynamicRendering &&
            vulkan13Supported.Synchronization2);
        if(!supported) throw new Exception("Device does not support required features");
        
        
        //enable required features (verified to be supported)
        vk!.GetPhysicalDeviceFeatures2(physicalDevice, out var features);
        features.SType = StructureType.PhysicalDeviceFeatures2;
        features.Features.SamplerAnisotropy = true;
        features.Features.DepthBiasClamp = coreSupported.Features.DepthBiasClamp ? true : false;
        
        //Timeline semaphore features (required for synch2)
        PhysicalDeviceTimelineSemaphoreFeatures timelineFeatures = new(){SType = StructureType.PhysicalDeviceTimelineSemaphoreFeatures};
        timelineFeatures.TimelineSemaphore = true;
        //Vulkan memory model
        PhysicalDeviceVulkanMemoryModelFeatures memoryModelFeatures = new(){SType = StructureType.PhysicalDeviceVulkanMemoryModelFeatures};
        memoryModelFeatures.VulkanMemoryModel = true;
        memoryModelFeatures.VulkanMemoryModelDeviceScope = memoryModelSupported.VulkanMemoryModelDeviceScope;
        //Buffer device address features (required for some buffer operations)
        PhysicalDeviceBufferDeviceAddressFeatures bufferAddressFeatures = new(){SType = StructureType.PhysicalDeviceBufferDeviceAddressFeatures};
        bufferAddressFeatures.BufferDeviceAddress = true;
        //8 bit storage features (required for some shader storage operations)
        PhysicalDevice8BitStorageFeatures storage8BitFeatures = new(){SType = StructureType.PhysicalDevice8BitStorageFeatures};
        storage8BitFeatures.StorageBuffer8BitAccess = storage8BitSupported.StorageBuffer8BitAccess;
        //vulkan 1.3 features
        PhysicalDeviceVulkan13Features vulkan13Features = new(){SType = StructureType.PhysicalDeviceVulkan13Features};
        vulkan13Features.DynamicRendering = true;
        vulkan13Features.Synchronization2 = true;
        //vulkan 1.1 features
        PhysicalDeviceVulkan11Features vulkan11Features = new(){SType = StructureType.PhysicalDeviceVulkan11Features};
        vulkan11Features.ShaderDrawParameters = true;
        
        
        
        //----------------------------------------------------------------
        //------------------query extended features support---------------
        //----------------------------------------------------------------
        PhysicalDeviceFeatures2 extendedFeaturesSupported = new()
        {
            SType = StructureType.PhysicalDeviceFeatures2,
        };
        //descriptor indexing features
        PhysicalDeviceDescriptorIndexingFeatures descriptorIndexingFeaturesSupported = new()
        {
            SType = StructureType.PhysicalDeviceDescriptorIndexingFeatures,
        };
        //robust2 supported
        PhysicalDeviceRobustness2FeaturesEXT robust2Supported = new()
        {
            SType = StructureType.PhysicalDeviceRobustness2FeaturesExt
        };
        //AccelerationStructure features
        PhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeaturesSupported = new()
        {
            SType = StructureType.PhysicalDeviceAccelerationStructureFeaturesKhr
        };
        //RayQuery features
        PhysicalDeviceRayQueryFeaturesKHR rayQueryFeaturesSupported = new()
        {
            SType = StructureType.PhysicalDeviceRayQueryFeaturesKhr
        };
        extendedFeaturesSupported.PNext = &descriptorIndexingFeaturesSupported;
        descriptorIndexingFeaturesSupported.PNext = &robust2Supported;
        robust2Supported.PNext = &accelerationStructureFeaturesSupported;
        accelerationStructureFeaturesSupported.PNext = &rayQueryFeaturesSupported;
        rayQueryFeaturesSupported.PNext = null;
        
        vk!.GetPhysicalDeviceFeatures2(physicalDevice, &extendedFeaturesSupported);
        
        //rayQuery shader uses indexing into a large sampled-image array.
        //some drivers require this core feature to be explicitly enabled
        if (extendedFeaturesSupported.Features.ShaderSampledImageArrayDynamicIndexing)
        {
            features.Features.ShaderSampledImageArrayDynamicIndexing = true;
        }
        //Prepare descriptor indexing features to enable if supported
        PhysicalDeviceDescriptorIndexingFeatures indexingFeaturesEnable = new(){SType = StructureType.PhysicalDeviceDescriptorIndexingFeatures};
        var descriptorIndexingEnabled = false;
        //enable non uniform indexing
        //some drivers require this core feature to be explicitly enabled for rayQuery
        if (descriptorIndexingFeaturesSupported.ShaderSampledImageArrayNonUniformIndexing)
        {
            indexingFeaturesEnable.ShaderSampledImageArrayNonUniformIndexing = true;
            descriptorIndexingEnabled = true;
        }

        if (descriptorIndexingEnabled)
        {
            if (descriptorIndexingFeaturesSupported.DescriptorBindingPartiallyBound)
            {
                indexingFeaturesEnable.DescriptorBindingPartiallyBound = true;
            }

            if (descriptorIndexingFeaturesSupported.DescriptorBindingUpdateUnusedWhilePending)
            {
                indexingFeaturesEnable.DescriptorBindingUpdateUnusedWhilePending = true;
            }
        }
        //optionally enable updateAfterBind flags when supported, not necessarily required for rayQuery tex's
        if (descriptorIndexingFeaturesSupported.DescriptorBindingSampledImageUpdateAfterBind)
            indexingFeaturesEnable.DescriptorBindingSampledImageUpdateAfterBind = true;
        if (descriptorIndexingFeaturesSupported.DescriptorBindingUniformBufferUpdateAfterBind)
            indexingFeaturesEnable.DescriptorBindingUniformBufferUpdateAfterBind = true;
        if (descriptorIndexingFeaturesSupported.DescriptorBindingUpdateUnusedWhilePending)
            indexingFeaturesEnable.DescriptorBindingUpdateUnusedWhilePending = true;
        
        //helper to verify that an extension is enabled
        bool hasExtension(string name)
        {
            return deviceExtensions.Contains(name);
        }
        
        //prepare robustness2 featureset if the extension is enabled
        var hasRobust2 = hasExtension("VK_EXT_ROBUSTNESS_2_EXTENSION_NAME");
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
        var hasAccelerationStructure = hasExtension("VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME");
        PhysicalDeviceAccelerationStructureFeaturesKHR accelerationstructureEnable = new(){SType = StructureType.PhysicalDeviceAccelerationStructureFeaturesKhr};
        if (hasAccelerationStructure)
        {
            accelerationstructureEnable.AccelerationStructure = true;
        }
        
        //prepare rayQuery features if extension is enabled and supported
        var hasRayQuery = hasExtension("VK_KHR_RAY_QUERY_EXTENSION_NAME");
        PhysicalDeviceRayQueryFeaturesKHR rayQueryEnable = new(){SType = StructureType.PhysicalDeviceRayQueryFeaturesKhr};
        if (hasRayQuery)
        {
            rayQueryEnable.RayQuery = true;
        }
        
        //chain all features together
        features.PNext = &timelineFeatures;
        timelineFeatures.PNext = &memoryModelFeatures;
        memoryModelFeatures.PNext = &bufferAddressFeatures;
        bufferAddressFeatures.PNext = &storage8BitFeatures;
        storage8BitFeatures.PNext = &vulkan11Features;
        vulkan11Features.PNext = &vulkan13Features;
        //build explicitly for enabled features
        void** tailNext = (void**)&vulkan13Features.PNext;
        if (descriptorIndexingEnabled)
        {
            *tailNext = &indexingFeaturesEnable;
            tailNext = (void**)&indexingFeaturesEnable.PNext;
        }

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
        descriptorIndexEnabled = descriptorIndexingEnabled && (indexingFeaturesEnable.DescriptorBindingPartiallyBound && indexingFeaturesEnable.DescriptorBindingUpdateUnusedWhilePending);
        robustness2Enabled = hasRobust2 && (robust2Enable.RobustBufferAccess2 || robust2Enable.RobustImageAccess2 || robust2Enable.NullDescriptor);
        accelerationStructureEnabled = hasAccelerationStructure && accelerationstructureEnable.AccelerationStructure;
        rayQueryEnabled = hasRayQuery && rayQueryEnable.RayQuery;
        
        bool printFeatures = false;
        if (printFeatures)
        {
            Console.WriteLine("----> Device Features:");
            Console.WriteLine("----> Sampler Anisotropy: " + features.Features.SamplerAnisotropy);
            Console.WriteLine("----> Depth Bias Clamp: " + features.Features.DepthBiasClamp);
            Console.WriteLine("----> Timeline Semaphore: " + timelineFeatures.TimelineSemaphore);
            Console.WriteLine("----> Vulkan Memory Model: " + memoryModelFeatures.VulkanMemoryModel);
            Console.WriteLine("----> Buffer Device Address: " + bufferAddressFeatures.BufferDeviceAddress);
            Console.WriteLine("----> 8 bit Storage: " + storage8BitFeatures.StorageBuffer8BitAccess);
            Console.WriteLine("----> Vulkan 1.1 Features: " + vulkan11Features.ShaderDrawParameters);
            Console.WriteLine("----> Vulkan 1.3 Features: " + vulkan13Features.DynamicRendering + " " + vulkan13Features.Synchronization2);
            Console.WriteLine("----> Descriptor Indexing Features: " + indexingFeaturesEnable.ShaderSampledImageArrayNonUniformIndexing + "\n "
                              + indexingFeaturesEnable.DescriptorBindingPartiallyBound + "\n "
                              + indexingFeaturesEnable.DescriptorBindingUpdateUnusedWhilePending + "\n "
                              + indexingFeaturesEnable.DescriptorBindingSampledImageUpdateAfterBind + "\n "
                              + indexingFeaturesEnable.DescriptorBindingUniformBufferUpdateAfterBind);
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
                EnabledExtensionCount = (uint)deviceExtensions.Length,
                PpEnabledExtensionNames = (byte**)SilkMarshal.StringArrayToPtr(deviceExtensions),
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