using Silk.NET.Core;
using Silk.NET.Core.Native;
using Silk.NET.Vulkan;
using Silk.NET.Vulkan.Extensions.EXT;
using Silk.NET.Vulkan.Extensions.KHR;
using Silk.NET.Windowing;
// ReSharper disable InconsistentNaming

namespace CadThingo.VulkanEngine.Renderer;

/// <summary>
/// The Vulkan RHI context — instance, debug messenger, surface, physical/logical
/// device, queues, the suballocating memory allocator, descriptor pool, and command
/// pool — plus the device-level service helpers every pipeline and resource owner
/// builds on (buffer/image creation, layout transitions, single-time commands,
/// shader-module creation, format/memory queries).
///
/// L1 of the renderer refactor: this is the narrow "device services" surface that
/// the rest of the engine depends on, peeled off the former Renderer god object.
/// Dependencies point one way — GraphicsDevice knows nothing about the Renderer,
/// the scene, pipelines, or render techniques. Renderer owns a GraphicsDevice and
/// (for now) exposes same-named delegating accessors so existing call sites keep
/// working while the rest of L1 lands.
/// </summary>
public sealed unsafe class GraphicsDevice : IDisposable
{
    public GraphicsDevice(IWindow window, bool enableValidationLayers)
    {
        _window = window;
        _enableValidationLayers = enableValidationLayers;
    }

    private readonly IWindow _window;
    private readonly bool _enableValidationLayers;
    private readonly string[] ValidationLayers =
    [
        "VK_LAYER_KHRONOS_validation"
    ];

    private readonly Vk vk = Globals.vk!;
    private Instance instance;
    private ExtDebugUtils? debugUtils;
    private DebugUtilsMessengerEXT debugMessenger;

    private KhrSurface? khrSurface;
    private SurfaceKHR surface;

    private PhysicalDevice physicalDevice;
    // List (not array) so AddSupportedOptionalExtensions can actually mutate it.
    // Required swapchain seeds the list; optional extensions append after device pick.
    private readonly List<string> deviceExtensions = new() { KhrSwapchain.ExtensionName };
    private Device device;

    // Silk.NET 2.23 ships no KhrRayQuery wrapper (ray-query has zero host functions —
    // it's a pure SPIR-V capability), so we hardcode the extension string ourselves.
    private const string KhrRayQueryExtensionName = "VK_KHR_ray_query";

    private QueueFamilyIndices queueFamilyIndices;
    private Queue graphicsQueue;
    private Queue presentQueue;
    private Queue computeQueue;
    private Queue transferQueue;

    // Suballocates VkDeviceMemory blocks so we don't burn one slot of
    // maxMemoryAllocationCount per resource. Owned for the lifetime of the device;
    // disposed in Dispose() before vkDestroyDevice so every block frees cleanly.
    private GpuMemoryAllocator memAllocator = null!;
    private DescriptorPool descriptorPool;
    private CommandPool commandPool;

    private bool descriptorIndexEnabled = false;
    private bool robustness2Enabled = false;
    private bool accelerationStructureEnabled = false;
    private bool rayQueryEnabled = false;
    private bool rayTracePipelineEnabled = false;
    private bool serEnabled = false;   // VK_NV_ray_tracing_invocation_reorder
    // Vulkan 1.1 multiview. Required for the reflection-probe capture pass that
    // renders all 6 faces of a cubemap in one draw via gl_ViewIndex.
    private bool multiviewEnabled = false;

    // Debug/profiling capability cache (filled in CreateLogicalDevice). timestampPeriod
    // is ns-per-tick from the device limits; graphicsTimestampValidBits masks the high
    // bits of a raw timestamp the graphics queue doesn't write. pipelineStatisticsEnabled
    // reflects whether the optional pipelineStatisticsQuery feature was turned on.
    private float timestampPeriod;
    private uint  graphicsTimestampValidBits;
    private bool  pipelineStatisticsEnabled;

    // ---- Public device-services surface ------------------------------------

    public Vk                 Vk             => vk;
    
    public ExtDebugUtils? DebugUtils => debugUtils;
    public Device             Device         => device;
    public PhysicalDevice     PhysicalDevice => physicalDevice;
    public GpuMemoryAllocator Allocator      => memAllocator;
    public DescriptorPool     DescriptorPool => descriptorPool;
    public CommandPool        CommandPool    => commandPool;
    
    public Queue GraphicsQueue => graphicsQueue;
    public Queue PresentQueue  => presentQueue;
    public Queue ComputeQueue  => computeQueue;
    public Queue TransferQueue => transferQueue;

    public Instance           Instance           => instance;
    public SurfaceKHR         Surface            => surface;
    public KhrSurface?        KhrSurface         => khrSurface;
    public QueueFamilyIndices QueueFamilyIndices => queueFamilyIndices;

    // ---- Debug / profiling capabilities ------------------------------------
    /// <summary>Nanoseconds represented by one timestamp-query tick (device limit).</summary>
    public float TimestampPeriod => timestampPeriod;
    /// <summary>Valid low-bit count of a graphics-queue timestamp; 0 ⇒ the queue can't timestamp.</summary>
    public uint  TimestampValidBits => graphicsTimestampValidBits;
    /// <summary>True iff the graphics queue can write timestamps with a non-zero period.</summary>
    public bool  TimestampsSupported => timestampPeriod != 0f && graphicsTimestampValidBits != 0;
    /// <summary>True iff the optional pipelineStatisticsQuery feature was enabled at device creation.</summary>
    public bool  PipelineStatisticsSupported => pipelineStatisticsEnabled;

    // Read-only handle accessor — pipelines that load their own device-extension
    // dispatch tables (e.g. RtPipeline → KhrRayTracingPipeline) need the instance
    // for vk.TryGetDeviceExtension. Generic, not tied to any one extension.
    public Instance GetVkInstance() => instance;

    public bool DescriptorIndexingEnabled => descriptorIndexEnabled;
    public bool MultiviewEnabled           => multiviewEnabled;

    /// <summary>
    /// True iff both KHR_acceleration_structure and KHR_ray_query are enabled with their
    /// device features active. Gates the ray-traced shadow path; callers should fall back
    /// to a non-shadowed code path when false.
    /// </summary>
    public bool RayShadowsSupported => rayQueryEnabled && accelerationStructureEnabled;

    /// <summary>
    /// True iff KHR_ray_tracing_pipeline is enabled alongside KHR_acceleration_structure.
    /// Gates the (additive, opt-in) RT-pipeline path tracer; the inline-ray-query compute
    /// path tracer stays available regardless. Falls back to compute when false.
    /// </summary>
    public bool RayTracePipelineSupported => rayTracePipelineEnabled && accelerationStructureEnabled;

    /// <summary>
    /// True iff VK_NV_ray_tracing_invocation_reorder (SER) is enabled on top of the
    /// RT pipeline. When true the RT path tracer loads the SER raygen variant
    /// (HitObject + ReorderThread); otherwise the plain TraceRay variant.
    /// </summary>
    public bool SerSupported => serEnabled && RayTracePipelineSupported;

    // ---- Lifecycle ---------------------------------------------------------

    /// <summary>
    /// Brings up the full device context in dependency order: instance → debug →
    /// surface → physical device → logical device → memory allocator → command pool.
    /// The descriptor pool is created separately via <see cref="CreateDescriptorPool"/>
    /// because its sizing depends on app-level budgets owned by the Renderer.
    /// </summary>
    public void Initialize()
    {
        CreateInstance();
        SetupDebugMessenger(_enableValidationLayers);
        CreateSurface();
        PickPhysicalDevice();
        CreateLogicalDevice();
        // Must exist before ANY buffer/image allocation downstream.
        memAllocator = new GpuMemoryAllocator(vk, device, physicalDevice);
        CreateCommandPool();
    }

    // Full teardown in reverse-creation order. The orchestrator (Renderer) calls
    // this strictly last, after every resource that allocated through this device
    // has been freed. Globals.vk is a process-wide singleton — never disposed here.
    public void Dispose()
    {
        // Descriptor pool (frees the descriptor sets owned by the pipelines)
        if (descriptorPool.Handle != 0) vk.DestroyDescriptorPool(device, descriptorPool, null);

        // Command pool (frees buffers)
        if (commandPool.Handle != 0) vk.DestroyCommandPool(device, commandPool, null);

        // Memory allocator (frees every VkDeviceMemory block)
        // Must run after every other Vk*Destroy — those don't free memory, they only
        // release the buffer/image handle. The allocator owns the underlying blocks.
        memAllocator?.Dispose();

        // Device, debug, surface, instance
        vk.DestroyDevice(device, null);
        if (_enableValidationLayers && debugUtils != null)
            debugUtils.DestroyDebugUtilsMessenger(instance, debugMessenger, null);
        khrSurface?.DestroySurface(instance, surface, null);
        vk.DestroyInstance(instance, null);
    }

    // ---- Instance / debug / surface ----------------------------------------

    private static uint DebugCallBack(
        DebugUtilsMessageSeverityFlagsEXT severity,
        DebugUtilsMessageTypeFlagsEXT types,
        DebugUtilsMessengerCallbackDataEXT* data,
        void* userData)
    {
        var message = SilkMarshal.PtrToString((nint)data->PMessage);
        Console.WriteLine($"[VALIDATION LAYER:] {message}");
        return Vk.False;
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

        if (_enableValidationLayers)
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
        if (vk.CreateInstance(&createInfo, null, out instance) != Result.Success)
        {
            throw new Exception("Failed to create Vulkan instance");
        }
        SilkMarshal.Free(appNamePtr);
        SilkMarshal.Free(engineNamePtr);
        SilkMarshal.Free((nint)createInfo.PpEnabledExtensionNames);
        if(_enableValidationLayers)
            SilkMarshal.Free((nint)createInfo.PpEnabledLayerNames);
    }

    private void SetupDebugMessenger(bool enableValidation)
    {
        if (!enableValidation) return;

        if (!vk.TryGetInstanceExtension(instance, out debugUtils)) return;
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
        if(!vk.TryGetInstanceExtension<KhrSurface>(instance, out khrSurface))
            throw new Exception("KHR Surface ext not found");
        surface = _window!.VkSurface!.Create<AllocationCallbacks>(instance.ToHandle(), null).ToSurface();
    }

    private void PickPhysicalDevice()
    {
        uint deviceCount = 0;
        vk.EnumeratePhysicalDevices(instance, &deviceCount, null);

        if (deviceCount == 0)
            throw new Exception("No Vulkan devices found");

        var devices = stackalloc PhysicalDevice[(int)deviceCount];
        vk.EnumeratePhysicalDevices(instance, &deviceCount, devices);

        //prioritise discrete GPUs, over integrated GPUs
        //first collect all suitable devices with their suitability scores
        Dictionary<PhysicalDevice, int> deviceSuitability = new();


        for (var i = 0; i < deviceCount; i++)
        {
            var device = devices[i];
            var deviceProperites = vk.GetPhysicalDeviceProperties(device);
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
            vk.GetPhysicalDeviceFeatures2(device, &features2);
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
            vk.GetPhysicalDeviceMemoryProperties(device, out var memProps);
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
           vk.GetPhysicalDeviceProperties(physicalDevice, out var deviceProperties);
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
        "VK_NV_ray_tracing_invocation_reorder",            // SER (Ada-class); opt-in, no Silk wrapper
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
        vk.EnumerateDeviceExtensionProperties(physicalDevice, (byte*)null, &extensionCount, null);
        var available = new ExtensionProperties[extensionCount];
        fixed (ExtensionProperties* pAvailable = available)
            vk.EnumerateDeviceExtensionProperties(physicalDevice, (byte*)null, &extensionCount, pAvailable);

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
        var rayTracingPipelineFeaturesSupported = new PhysicalDeviceRayTracingPipelineFeaturesKHR
        {
            SType = StructureType.PhysicalDeviceRayTracingPipelineFeaturesKhr
        };
        var serFeaturesSupported = new PhysicalDeviceRayTracingInvocationReorderFeaturesNV
        {
            SType = StructureType.PhysicalDeviceRayTracingInvocationReorderFeaturesNV
        };

        coreSupported.PNext             = &vulkan11Supported;
        vulkan11Supported.PNext         = &vulkan12Supported;
        vulkan12Supported.PNext         = &vulkan13Supported;
        vulkan13Supported.PNext         = &robust2Supported;
        robust2Supported.PNext          = &accelerationStructureFeaturesSupported;
        accelerationStructureFeaturesSupported.PNext = &rayQueryFeaturesSupported;
        rayQueryFeaturesSupported.PNext = &rayTracingPipelineFeaturesSupported;
        rayTracingPipelineFeaturesSupported.PNext = &serFeaturesSupported;
        serFeaturesSupported.PNext = null;

        vk.GetPhysicalDeviceFeatures2(physicalDevice, &coreSupported);

        bool supported = (
            coreSupported.Features.SamplerAnisotropy &&
            coreSupported.Features.MultiDrawIndirect &&
            coreSupported.Features.DrawIndirectFirstInstance &&
            coreSupported.Features.ImageCubeArray &&
            vulkan11Supported.ShaderDrawParameters &&
            vulkan11Supported.Multiview &&
            vulkan12Supported.TimelineSemaphore &&
            vulkan12Supported.VulkanMemoryModel &&
            vulkan12Supported.BufferDeviceAddress &&
            vulkan12Supported.DrawIndirectCount &&
            vulkan13Supported.DynamicRendering &&
            vulkan13Supported.Synchronization2);
        if(!supported) throw new Exception("Device does not support required features");


        //enable required features (verified to be supported)
        vk.GetPhysicalDeviceFeatures2(physicalDevice, out var features);
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
        // Cube-array views (VK_IMAGE_VIEW_TYPE_CUBE_ARRAY) — required by the
        // ReflectionProbeSystem to sample all probes from a single descriptor.
        features.Features.ImageCubeArray = true;
        //rayQuery shader uses indexing into a large sampled-image array.
        if (coreSupported.Features.ShaderSampledImageArrayDynamicIndexing)
            features.Features.ShaderSampledImageArrayDynamicIndexing = true;

        // Optional: per-pass pipeline-statistics queries for the render-graph debug
        // overlay (VS/FS/compute invocations, primitives, …). Enable only if supported;
        // GraphDebug gates its pool on PipelineStatisticsSupported.
        if (coreSupported.Features.PipelineStatisticsQuery)
        {
            features.Features.PipelineStatisticsQuery = true;
            pipelineStatisticsEnabled = true;
        }

        // Cache timestamp capabilities for the render-graph GPU timer. Period is ns/tick
        // from the device limits; valid-bits comes from the graphics queue family (0 ⇒
        // the queue can't timestamp, so GraphDebug skips GPU timing).
        vk.GetPhysicalDeviceProperties(physicalDevice, out var tsProps);
        timestampPeriod = tsProps.Limits.TimestampPeriod;
        uint qfCount = 0;
        vk.GetPhysicalDeviceQueueFamilyProperties(physicalDevice, &qfCount, null);
        var qfProps = new QueueFamilyProperties[(int)qfCount];
        fixed (QueueFamilyProperties* pQf = qfProps)
            vk.GetPhysicalDeviceQueueFamilyProperties(physicalDevice, &qfCount, pQf);
        graphicsTimestampValidBits = queueFamilyIndices.graphicsFamily.HasValue
            ? qfProps[(int)queueFamilyIndices.graphicsFamily.Value].TimestampValidBits
            : 0;

        //vulkan 1.1
        PhysicalDeviceVulkan11Features vulkan11Features = new(){SType = StructureType.PhysicalDeviceVulkan11Features};
        vulkan11Features.ShaderDrawParameters = true;
        // Multiview — gl_ViewIndex in vertex shaders so cubemap-face capture can
        // render all 6 layers in one draw call. Required for reflection probes.
        vulkan11Features.Multiview = true;

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

        //prepare ray-tracing-pipeline features if extension is enabled and supported.
        //Additive: only gates the opt-in RT-pipeline path tracer; the compute path
        //tracer is unaffected whether or not this ends up enabled.
        var hasRayTracingPipeline = hasExtension(KhrRayTracingPipeline.ExtensionName);
        PhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipelineEnable = new(){SType = StructureType.PhysicalDeviceRayTracingPipelineFeaturesKhr};
        if (hasRayTracingPipeline && rayTracingPipelineFeaturesSupported.RayTracingPipeline)
        {
            rayTracingPipelineEnable.RayTracingPipeline = true;

        }

        //prepare SER (VK_NV_ray_tracing_invocation_reorder) — built on the RT
        //pipeline; gates the HitObject/ReorderThread raygen variant. Additive.
        var hasSer = hasExtension("VK_NV_ray_tracing_invocation_reorder");
        PhysicalDeviceRayTracingInvocationReorderFeaturesNV serEnable = new(){SType = StructureType.PhysicalDeviceRayTracingInvocationReorderFeaturesNV};
        if (hasSer && serFeaturesSupported.RayTracingInvocationReorder)
        {
            serEnable.RayTracingInvocationReorder = true;
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

        if (hasRayTracingPipeline && rayTracingPipelineEnable.RayTracingPipeline)
        {
            *tailNext = &rayTracingPipelineEnable;
            tailNext = (void**)&rayTracingPipelineEnable.PNext;
        }

        if (hasSer && serEnable.RayTracingInvocationReorder)
        {
            *tailNext = &serEnable;
            tailNext = (void**)&serEnable.PNext;
        }

        //record which features ended up enabled
        descriptorIndexEnabled = descriptorIndexingEnabled && (vulkan12Features.DescriptorBindingPartiallyBound && vulkan12Features.DescriptorBindingUpdateUnusedWhilePending);
        robustness2Enabled = hasRobust2 && (robust2Enable.RobustBufferAccess2 || robust2Enable.RobustImageAccess2 || robust2Enable.NullDescriptor);
        accelerationStructureEnabled = hasAccelerationStructure && accelerationstructureEnable.AccelerationStructure;
        rayQueryEnabled = hasRayQuery && rayQueryEnable.RayQuery;
        rayTracePipelineEnabled = hasRayTracingPipeline && rayTracingPipelineEnable.RayTracingPipeline;
        serEnabled = hasSer && serEnable.RayTracingInvocationReorder;
        multiviewEnabled = vulkan11Features.Multiview;

        Console.WriteLine($"----> RayShadowsSupported: {RayShadowsSupported} " +
                          $"(rayQuery={rayQueryEnabled}, accelStruct={accelerationStructureEnabled})");
        Console.WriteLine($"----> RayTracePipelineSupported: {RayTracePipelineSupported} " +
                          $"(rayTracingPipeline={rayTracePipelineEnabled})");
        Console.WriteLine($"----> SerSupported: {SerSupported} (invocationReorder={serEnabled})");

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
            if (vk.CreateDevice(physicalDevice, &deviceCreateInfo, null, out device) != Result.Success)
            {
                throw new Exception("Failed to create logical device");
            }
            //create queues
            vk.GetDeviceQueue(device, queueFamilyIndices.graphicsFamily.Value, 0, out graphicsQueue);
            vk.GetDeviceQueue(device, queueFamilyIndices.presentFamily.Value, 0, out presentQueue);
            vk.GetDeviceQueue(device, queueFamilyIndices.computeFamily.Value, 0, out computeQueue);
            vk.GetDeviceQueue(device, queueFamilyIndices.transferFamily.Value, 0, out transferQueue);
        }
    }

    private bool CheckValidationLayerSupport()
    {
        uint layerCount = 0;
        vk.EnumerateInstanceLayerProperties(&layerCount, null);

        var layers = stackalloc LayerProperties[(int)layerCount];
        vk.EnumerateInstanceLayerProperties(&layerCount, layers);

        for (int i = 0; i < layerCount; i++)
        {
            var name = SilkMarshal.PtrToString((nint)layers[i].LayerName);
            if(name.Equals("VK_LAYER_KHRONOS_validation"))
                return true;
        }
        return false;
    }

    private string[] GetRequiredExtensions()
    {
        var glfwExtensions = _window!.VkSurface!.GetRequiredExtensions(out var glfwExtensionCount);
        var extensions = SilkMarshal.PtrToStringArray((nint)glfwExtensions, (int)glfwExtensionCount);

        if (_enableValidationLayers)
            return extensions.Append(ExtDebugUtils.ExtensionName).ToArray();

        return extensions;
    }

    private QueueFamilyIndices FindQueueFamilies(PhysicalDevice device)
    {
        QueueFamilyIndices indices = new();

        //Get queue families props

        uint count = 0;
        vk.GetPhysicalDeviceQueueFamilyProperties(device, &count, null);
        var queueFamilies = new QueueFamilyProperties[(int)count];
        fixed (QueueFamilyProperties* pQueueFamilies = queueFamilies)
        {
            vk.GetPhysicalDeviceQueueFamilyProperties(device, &count, pQueueFamilies);
        }

        for (uint i = 0; i < count; i++)
        {
            var qf = queueFamilies[i];
            //check for graphics support
            if (qf.QueueFlags.HasFlag(QueueFlags.GraphicsBit) && !indices.graphicsFamily.HasValue)
            {
                indices.graphicsFamily = i;
            }

            //check for present support
            khrSurface!.GetPhysicalDeviceSurfaceSupport(device, i, surface, out var presentSupport);
            if (presentSupport && !indices.presentFamily.HasValue)
                indices.presentFamily = i;

            //check for compute support
            if (qf.QueueFlags.HasFlag(QueueFlags.ComputeBit) && !indices.computeFamily.HasValue)
            {
                indices.computeFamily = i;
            }

            //find dedicated transfer queue
            if (qf.QueueFlags.HasFlag(QueueFlags.TransferBit) && !qf.QueueFlags.HasFlag(QueueFlags.GraphicsBit))
            {
                if (!indices.transferFamily.HasValue)
                    indices.transferFamily = i;
            }

            //if we have found all the required queue families
            if (indices.IsComplete() && indices.transferFamily.HasValue)
                break;
        }

        //fallback if no transfer family is found
        if (!indices.transferFamily.HasValue && indices.graphicsFamily.HasValue)
        {
            indices.transferFamily = indices.graphicsFamily;
        }
        return indices;
    }

    public SwapChainSupportDetails QuerySwapChainSupport(PhysicalDevice physicalDevice)
    {
        var details = new SwapChainSupportDetails();
        khrSurface!.GetPhysicalDeviceSurfaceCapabilities(physicalDevice, surface, out details.Capabilities);

        uint formatCount = 0;
        khrSurface!.GetPhysicalDeviceSurfaceFormats(physicalDevice, surface, &formatCount, null);

        if (formatCount != 0)
        {
            details.Formats = new SurfaceFormatKHR[formatCount];
            fixed (SurfaceFormatKHR* formatsPtr = details.Formats)
            {
                khrSurface!.GetPhysicalDeviceSurfaceFormats(physicalDevice, surface, &formatCount, formatsPtr);
            }

        }
        else
        {
            details.Formats = Array.Empty<SurfaceFormatKHR>();
        }

        uint presentModeCount = 0;
        khrSurface!.GetPhysicalDeviceSurfacePresentModes(physicalDevice, surface, &presentModeCount, null);
        if (presentModeCount != 0)
        {
            details.PresentModes = new PresentModeKHR[presentModeCount];
            fixed (PresentModeKHR* formatsPtr = details.PresentModes)
            {
                khrSurface!.GetPhysicalDeviceSurfacePresentModes(physicalDevice, surface, &presentModeCount,
                    formatsPtr);
            }
        }
        else
        {
            details.PresentModes = Array.Empty<PresentModeKHR>();
        }

        return details;
    }

    private bool CheckDeviceExtensionSupport(PhysicalDevice device)
    {
        uint extensionCount = 0;
         vk.EnumerateDeviceExtensionProperties(device, (byte*)null, &extensionCount, null);
        var availableExtensions = stackalloc ExtensionProperties[(int)extensionCount];
        vk.EnumerateDeviceExtensionProperties(device, (byte*)null, &extensionCount, availableExtensions);

        HashSet<string> requiredExtensions = new(deviceExtensions);
        for (var i = 0; i < extensionCount; i++)
        {
            requiredExtensions.Remove(SilkMarshal.PtrToString((nint)availableExtensions[i].ExtensionName));
        }
        return requiredExtensions.Count == 0;
    }

    private void CreateCommandPool()
    {
        //Create command pool info
        CommandPoolCreateInfo poolCreateInfo = new()
        {
            SType = StructureType.CommandPoolCreateInfo,
            Flags = CommandPoolCreateFlags.ResetCommandBufferBit,
            QueueFamilyIndex = queueFamilyIndices.graphicsFamily!.Value
        };

        if (vk.CreateCommandPool(device, &poolCreateInfo, null, out commandPool) != Result.Success)
        {
            throw new Exception("Failed to create command pool");
        }
    }

    /// <summary>
    /// Creates the renderer's descriptor pool. Sizing (pool sizes / max sets) is an
    /// app-level budget decision owned by the Renderer and passed in here — the device
    /// just owns the handle and its lifetime.
    /// </summary>
    public void CreateDescriptorPool(DescriptorPoolSize[] poolSizes, uint maxSets, DescriptorPoolCreateFlags flags)
    {
        fixed (DescriptorPoolSize* poolSizesPtr = poolSizes)
        {
            DescriptorPoolCreateInfo poolInfo = new()
            {
                SType = StructureType.DescriptorPoolCreateInfo,
                MaxSets = maxSets,
                PoolSizeCount = (uint)poolSizes.Length,
                PPoolSizes = poolSizesPtr,
                Flags = flags,
            };
            if (vk.CreateDescriptorPool(device, &poolInfo, null, out descriptorPool) != Result.Success)
            {
                throw new Exception("Failed to create descriptor pool");
            }
        }
    }

    // ---- Device-service helpers --------------------------------------------

    /// <summary>
    /// Finds a memory type that matches the given properties.
    /// </summary>
    public static uint FindMemoryType(Vk _vk,
        PhysicalDevice physDev, uint typeFilter, MemoryPropertyFlags props)
    {
        _vk.GetPhysicalDeviceMemoryProperties(physDev, out var memProps);
        for (uint i = 0; i < memProps.MemoryTypeCount; i++)
            if ((typeFilter & (1u << (int)i)) != 0 &&
                (memProps.MemoryTypes[(int)i].PropertyFlags & props) == props)
                return i;
        throw new Exception("No suitable memory type found.");
    }

    public Format FindDepthFormat()
    {
        Format depthFormat =
            FindSupportedFormat(new[] { Format.D32Sfloat, Format.D32SfloatS8Uint, Format.D24UnormS8Uint },
                ImageTiling.Optimal, FormatFeatureFlags.DepthStencilAttachmentBit);

        if (depthFormat == Format.Undefined)
        {
            Console.Error.WriteLine("failed to find suitable depth format, falling back to d32sFloat");
            return Format.D32Sfloat;
        }
        return depthFormat;
    }

    /// <summary>
    /// Finds a format that supports the given features.
    /// </summary>
    private Format FindSupportedFormat(Format[] formats, ImageTiling tiling, FormatFeatureFlags features = FormatFeatureFlags.None)
    {
        foreach (var format in formats)
        {
            vk.GetPhysicalDeviceFormatProperties(physicalDevice, format, out var props);
            if(tiling == ImageTiling.Linear && (props.LinearTilingFeatures & features) != 0)
                return format;
            else if(tiling == ImageTiling.Optimal &&(props.OptimalTilingFeatures & features) != 0)
                return format;
        }
        Console.Error.WriteLine("failed to find suitable format!");
        return Format.Undefined;
    }

    public ShaderModule CreateShaderModule(byte[] shaderCode)
    {
        //Create shader module
        ShaderModuleCreateInfo createInfo = new()
        {
            SType = StructureType.ShaderModuleCreateInfo,
            CodeSize = (nuint)shaderCode.Length,
        };
        fixed (byte* pCode = shaderCode)
        {
            createInfo.PCode = (uint*)pCode;
        }

        if (vk.CreateShaderModule(device, &createInfo, null, out var shaderModule) != Result.Success)
        {
            throw new Exception("Failed to create shader module");
        }
        return shaderModule;
    }

    public CommandBuffer BeginSingleTimeCommands()
    {
        CommandBufferAllocateInfo allocInfo = new()
        {
            SType = StructureType.CommandBufferAllocateInfo,
            CommandPool = commandPool,
            Level = CommandBufferLevel.Primary,
            CommandBufferCount = 1,
        };
        vk.AllocateCommandBuffers(device, &allocInfo, out CommandBuffer cmd);
        CommandBufferBeginInfo beginInfo = new()
        {
            SType = StructureType.CommandBufferBeginInfo,
            Flags = CommandBufferUsageFlags.OneTimeSubmitBit,
        };
        vk.BeginCommandBuffer(cmd, &beginInfo);
        return cmd;
    }

    public void EndSingleTimeCommands(CommandBuffer cmd)
    {
        vk.EndCommandBuffer(cmd);
        SubmitInfo submit = new()
        {
            SType = StructureType.SubmitInfo,
            CommandBufferCount = 1,
            PCommandBuffers = &cmd,
        };
        vk.QueueSubmit(graphicsQueue, 1, &submit, default);
        vk.QueueWaitIdle(graphicsQueue);
        vk.FreeCommandBuffers(device, commandPool, 1, &cmd);
    }

    public void CreateBuffer(ulong size, BufferUsageFlags usage, MemoryPropertyFlags memProps,
        out Buffer buffer, out SubAlloc alloc)
    {
        BufferCreateInfo bufferInfo = new()
        {
            SType = StructureType.BufferCreateInfo,
            Size = size,
            Usage = usage,
            SharingMode = SharingMode.Exclusive,
        };
        if (vk.CreateBuffer(device, &bufferInfo, null, out buffer) != Result.Success)
            throw new Exception("Failed to create buffer");

        // DeviceAddress flag is already baked into every buffer-bucket block by
        // the allocator — no per-call PNext chain needed. Caller still has to set
        // ShaderDeviceAddressBit in `usage` for the buffer itself, which they do.
        alloc = memAllocator.AllocateForBuffer(buffer, memProps);
    }

    public void CopyBuffer(Buffer src, Buffer dst, ulong size)
    {
        var cmd = BeginSingleTimeCommands();
        BufferCopy region = new()
        {
            SrcOffset = 0,
            DstOffset = 0,
            Size = size,
        };
        vk.CmdCopyBuffer(cmd, src, dst, 1, &region);
        EndSingleTimeCommands(cmd);
    }

    public void UploadBufferData(Buffer dst, long dstOffset, void* srcData, ulong size)
    {
        CreateBuffer(size, BufferUsageFlags.TransferSrcBit,
            MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit,
            out var staging, out var stagingAlloc);

        void* mapped = memAllocator.GetMapped(stagingAlloc);
        System.Buffer.MemoryCopy(srcData, mapped, (long)size, (long)size);

        var cmd = BeginSingleTimeCommands();
        BufferCopy region = new() { SrcOffset = 0, DstOffset = (ulong)dstOffset, Size = size };
        vk.CmdCopyBuffer(cmd, staging, dst, 1, &region);
        EndSingleTimeCommands(cmd);

        DestroyBuffer(staging, stagingAlloc);
    }

    public void DestroyBuffer(Buffer buffer, SubAlloc alloc)
    {
        if (buffer.Handle != 0) vk.DestroyBuffer(device, buffer, null);
        memAllocator.Free(alloc);
    }

    internal void CreateMappedUniformBuffer(int sizeBytes, ref UboBuffer ubo)
    {
        BufferCreateInfo bufferInfo = new()
        {
            SType = StructureType.BufferCreateInfo,
            Size = (ulong)sizeBytes,
            Usage = BufferUsageFlags.UniformBufferBit,
            SharingMode = SharingMode.Exclusive,
        };
        vk.CreateBuffer(device, &bufferInfo, null, out ubo.buffer);
        ubo.alloc  = memAllocator.AllocateForBuffer(ubo.buffer,
            MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit);
        ubo.mapped = memAllocator.GetMapped(ubo.alloc);
    }

    // Allocates a host-visible, coherent, persistently-mapped SSBO. Optional extra
    // usage bits let callers turn the same buffer into an indirect-cmd / indirect-
    // count source on top of plain storage usage.
    internal void CreateMappedStorageBuffer(ulong sizeBytes, ref UboBuffer ubo,
        BufferUsageFlags extraUsage = 0)
    {
        BufferCreateInfo bufferInfo = new()
        {
            SType = StructureType.BufferCreateInfo,
            Size = sizeBytes,
            Usage = BufferUsageFlags.StorageBufferBit | extraUsage,
            SharingMode = SharingMode.Exclusive,
        };
        vk.CreateBuffer(device, &bufferInfo, null, out ubo.buffer);
        ubo.alloc  = memAllocator.AllocateForBuffer(ubo.buffer,
            MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit);
        ubo.mapped = memAllocator.GetMapped(ubo.alloc);
    }

    public void TransitionImageLayout( CommandBuffer cmd, Image image, Format format,  ImageLayout oldLayout,
        ImageLayout newLayout, uint mipLevels = 1)
    {
        bool isDepth = format == Format.D32Sfloat || format == Format.D32SfloatS8Uint || format == Format.D24UnormS8Uint;
        ImageMemoryBarrier barrier = new()
        {
            SType = StructureType.ImageMemoryBarrier,
            OldLayout = oldLayout,
            NewLayout = newLayout,
            SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
            Image = image,
            SubresourceRange =
                new ImageSubresourceRange(isDepth ? ImageAspectFlags.DepthBit : ImageAspectFlags.ColorBit, 0, mipLevels,
                    0, 1)
        };

        //Initialize pipeline stage tracking for synchronization timing
        //these stages define when operations must complete and when new operations can begin
        PipelineStageFlags sourceStage;
        PipelineStageFlags destinationStage;

        //configure sync for undefined -> transfer layout transition
        //common when preparing images
        if (oldLayout == ImageLayout.Undefined && newLayout == ImageLayout.TransferDstOptimal)
        {
            barrier.SrcAccessMask = 0;
            barrier.DstAccessMask = AccessFlags.TransferWriteBit;

            sourceStage = PipelineStageFlags.TopOfPipeBit;
            destinationStage = PipelineStageFlags.TransferBit;

        }
        //configure sync for transfer -> shader read layout transition
        //pattern prepares uploaded images for shader sampling
        else if (oldLayout == ImageLayout.TransferDstOptimal && newLayout == ImageLayout.ShaderReadOnlyOptimal)
        {
            barrier.SrcAccessMask = AccessFlags.TransferWriteBit;
            barrier.DstAccessMask = AccessFlags.ShaderReadBit;

            sourceStage = PipelineStageFlags.TransferBit;
            destinationStage = PipelineStageFlags.FragmentShaderBit;
        }
        // Swapchain image post-blit → ImGui overlay pass.
        // Blit (transfer write) must finish before color-attachment writes from the UI pipeline.
        else if (oldLayout == ImageLayout.TransferDstOptimal && newLayout == ImageLayout.ColorAttachmentOptimal)
        {
            barrier.SrcAccessMask = AccessFlags.TransferWriteBit;
            barrier.DstAccessMask = AccessFlags.ColorAttachmentWriteBit;

            sourceStage = PipelineStageFlags.TransferBit;
            destinationStage = PipelineStageFlags.ColorAttachmentOutputBit;
        }
        // ImGui overlay → present. Color writes must finish before the presentation engine reads.
        // PresentSrcKhr has no access mask (acquire/release sync handled by semaphores).
        else if (oldLayout == ImageLayout.ColorAttachmentOptimal && newLayout == ImageLayout.PresentSrcKhr)
        {
            barrier.SrcAccessMask = AccessFlags.ColorAttachmentWriteBit;
            barrier.DstAccessMask = 0;

            sourceStage = PipelineStageFlags.ColorAttachmentOutputBit;
            destinationStage = PipelineStageFlags.BottomOfPipeBit;
        }
        // FinalColor: settled in ShaderReadOnlyOptimal by the render-graph final-layout barrier,
        // briefly TransferSrcOptimal for the swapchain blit, then back. Prior shader reads must
        // finish before the blit, and the blit's transfer reads must finish before the next
        // ImGui sample.
        else if (oldLayout == ImageLayout.ShaderReadOnlyOptimal && newLayout == ImageLayout.TransferSrcOptimal)
        {
            barrier.SrcAccessMask = AccessFlags.ShaderReadBit;
            barrier.DstAccessMask = AccessFlags.TransferReadBit;

            sourceStage = PipelineStageFlags.FragmentShaderBit;
            destinationStage = PipelineStageFlags.TransferBit;
        }
        else if (oldLayout == ImageLayout.TransferSrcOptimal && newLayout == ImageLayout.ShaderReadOnlyOptimal)
        {
            barrier.SrcAccessMask = AccessFlags.TransferReadBit;
            barrier.DstAccessMask = AccessFlags.ShaderReadBit;

            sourceStage = PipelineStageFlags.TransferBit;
            destinationStage = PipelineStageFlags.FragmentShaderBit;
        }
        // Pathtracer storage-image transitions
        // Fresh storage image — one-shot init in CreatePathTracingResources.
        // No prior work to wait on; the next user is a compute shader.
        else if (oldLayout == ImageLayout.Undefined && newLayout == ImageLayout.General)
        {
            barrier.SrcAccessMask = 0;
            barrier.DstAccessMask = AccessFlags.ShaderReadBit | AccessFlags.ShaderWriteBit;

            sourceStage      = PipelineStageFlags.TopOfPipeBit;
            destinationStage = PipelineStageFlags.ComputeShaderBit;
        }
        // PT dispatch finished writing ptOutColor; tonemap is about to sample it
        // as a CombinedImageSampler.  Compute-write → fragment-read.
        else if (oldLayout == ImageLayout.General && newLayout == ImageLayout.ShaderReadOnlyOptimal)
        {
            barrier.SrcAccessMask = AccessFlags.ShaderWriteBit;
            barrier.DstAccessMask = AccessFlags.ShaderReadBit;

            sourceStage      = PipelineStageFlags.ComputeShaderBit;
            destinationStage = PipelineStageFlags.FragmentShaderBit;
        }
        // ptOutColor back to General for the next frame's dispatch.  The reader
        // was tonemap's fragment shader; the next user is the compute dispatch.
        else if (oldLayout == ImageLayout.ShaderReadOnlyOptimal && newLayout == ImageLayout.General)
        {
            barrier.SrcAccessMask = AccessFlags.ShaderReadBit;
            barrier.DstAccessMask = AccessFlags.ShaderReadBit | AccessFlags.ShaderWriteBit;

            sourceStage      = PipelineStageFlags.FragmentShaderBit;
            destinationStage = PipelineStageFlags.ComputeShaderBit;
        }
        // FinalColor was sitting in ShaderReadOnly (viewport sample / last frame's
        // graph end-state); tonemap is about to write it as a color attachment.
        else if (oldLayout == ImageLayout.ShaderReadOnlyOptimal && newLayout == ImageLayout.ColorAttachmentOptimal)
        {
            barrier.SrcAccessMask = AccessFlags.ShaderReadBit;
            barrier.DstAccessMask = AccessFlags.ColorAttachmentWriteBit;

            sourceStage      = PipelineStageFlags.FragmentShaderBit;
            destinationStage = PipelineStageFlags.ColorAttachmentOutputBit;
        }
        // Tonemap finished writing FinalColor; next consumers are the swapchain
        // blit (transfer read) and the ImGui viewport sampler (fragment read).
        // Cover both downstream stages so neither needs an extra barrier.
        else if (oldLayout == ImageLayout.ColorAttachmentOptimal && newLayout == ImageLayout.ShaderReadOnlyOptimal)
        {
            barrier.SrcAccessMask = AccessFlags.ColorAttachmentWriteBit;
            barrier.DstAccessMask = AccessFlags.ShaderReadBit | AccessFlags.TransferReadBit;

            sourceStage      = PipelineStageFlags.ColorAttachmentOutputBit;
            destinationStage = PipelineStageFlags.FragmentShaderBit | PipelineStageFlags.TransferBit;
        }
        else
        {
            throw new Exception($"Unsupported layout transition: {oldLayout} -> {newLayout}");
        }

        vk.CmdPipelineBarrier(cmd,
            sourceStage,
            destinationStage,
            0,
            0, null,
            0, null,
            1, &barrier);
    }

    public void GenerateMipMaps(CommandBuffer cmds, Image image, Format format, uint width, uint height,uint mipLevels)
    {
        ImageMemoryBarrier barrier = new()
        {
            SType = StructureType.ImageMemoryBarrier,
            Image = image,
            SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
            SubresourceRange = new()
            {
                AspectMask = ImageAspectFlags.ColorBit,
                LevelCount = 1,
                LayerCount = 1
            }
        };
        var mipWidth = width;
        var mipHeight = height;


        for(uint i =1; i < mipLevels; i++)
        {
            barrier.SubresourceRange.BaseMipLevel = i - 1;
            barrier.OldLayout = ImageLayout.TransferDstOptimal;
            barrier.NewLayout = ImageLayout.TransferSrcOptimal;
            barrier.SrcAccessMask = AccessFlags.TransferWriteBit;
            barrier.DstAccessMask = AccessFlags.TransferReadBit;

            vk.CmdPipelineBarrier(cmds, PipelineStageFlags.TransferBit,
                PipelineStageFlags.TransferBit, 0,
                0, null,
                0, null,
                1, &barrier);

            ImageBlit blit = new()
            {
                SrcOffsets =
                {
                    Element0 = new Offset3D(0, 0, 0),
                    Element1 = new Offset3D((int)mipWidth, (int)mipHeight, 1)
                },
                SrcSubresource =
                {
                    AspectMask = ImageAspectFlags.ColorBit,
                    MipLevel = i - 1,
                    BaseArrayLayer = 0,
                    LayerCount = 1
                },
                DstOffsets =
                {
                    Element0 = new Offset3D(0, 0, 0),
                    Element1 = new Offset3D((int)mipWidth > 1 ? (int)mipWidth / 2 : 1,
                        (int)mipHeight > 1 ? (int)mipHeight / 2 : 1, 1)
                },
                DstSubresource =
                {
                    AspectMask = ImageAspectFlags.ColorBit,
                    MipLevel = i,
                    BaseArrayLayer = 0,
                    LayerCount = 1
                }
            };

            vk.CmdBlitImage(cmds,
                image, ImageLayout.TransferSrcOptimal,
                image, ImageLayout.TransferDstOptimal,
                1, &blit,
                Filter.Linear);

            barrier.OldLayout = ImageLayout.TransferSrcOptimal;
            barrier.NewLayout = ImageLayout.ShaderReadOnlyOptimal;
            barrier.SrcAccessMask = AccessFlags.TransferReadBit;
            barrier.DstAccessMask = AccessFlags.ShaderReadBit;

            vk.CmdPipelineBarrier(cmds, PipelineStageFlags.TransferBit,
                PipelineStageFlags.FragmentShaderBit, 0,
                 0, null,
                 0, null,
                 1, &barrier);

            if(mipWidth > 1) mipWidth /= 2;
            if(mipHeight > 1) mipHeight /= 2;
        }

        barrier.SubresourceRange.BaseMipLevel = mipLevels - 1;
        barrier.OldLayout = ImageLayout.TransferDstOptimal;
        barrier.NewLayout = ImageLayout.ShaderReadOnlyOptimal;
        barrier.SrcAccessMask = AccessFlags.TransferWriteBit;
        barrier.DstAccessMask = AccessFlags.ShaderReadBit;

        vk.CmdPipelineBarrier(cmds, PipelineStageFlags.TransferBit,
            PipelineStageFlags.FragmentShaderBit, 0,
             0, null,
             0, null,
             1, &barrier);
    }
}