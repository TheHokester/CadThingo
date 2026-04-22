using System.ComponentModel;
using System.Runtime.CompilerServices;
using CadThingo.GraphicsPipeline;
using CadThingo.VulkanEngine;
using Silk.NET.Vulkan;
using Silk.NET.Vulkan.Extensions.KHR;


namespace CadThingo.VulkanEngine;

public unsafe class ResourceHandle<T> where T : Resource
{
    private string _ID;
    private ResourceManager _manager;


    public ResourceHandle()
    {
        _manager = null;
    }

    public ResourceHandle(ResourceManager manager, string id)
    {
        _manager = manager;
        _ID = id;
    }
    

    private T? Get<T>() where T : Resource
    {
        if (_manager == null) return default;
        return *_manager.GetResource<T>(_ID);
    }
    
    bool IsValid() => _manager != null && _manager.HasResource<T>(_ID);
    
    String GetId() => _ID;
    
    //convenience operators
    
    
    public static implicit operator bool(ResourceHandle<T> handle)
    {
        return handle.IsValid();
    }
    
}

public unsafe class Resource(string id)
{
    private string resourceID = id; 
    bool loaded = false;
    
    public string GetId() => resourceID;
    public Resource() : this(default)
    {
    }

    ~Resource()
    {
        Unload();
    }
    
    protected bool IsLoaded() => loaded;
    
    public virtual bool Load()
    {
        loaded = doLoad();
        return loaded;
    }

    public virtual void Unload()
    {
        doUnload();
        loaded = false;
    }
    protected virtual bool doLoad() => true;
    protected virtual void doUnload() { }
}


public unsafe class ResourceManager
{
    //2 level storage system, organise by type then unique identifier
    Dictionary<Type, Dictionary<string, Resource>> _resources = new();
    
    // Two-level reference counting system for automatic resource lifecycle management
    // First level maps resource type, second level maps resource IDs to their data
    public struct ResourceData
    {
        public Resource* Resource; // Pointer to the actual resource
        public int refCount;// Reference count for this resource
    }
    Dictionary<Type, Dictionary<string, ResourceData>> _refCounts = new();
    
    
    
    
    ///<summary>load a resource of type T with the given ID</summary>
    /// <typeparam name="T">Resource type</typeparam>
    /// <param name="resourceID">Unique identifier for the resource</param>
    /// <param name="factory">Factory function to create the resource</param>
    public ResourceHandle<T> Load<T>(string resourceID, Func<string, T> factory) where T : Resource
    {
        var typeResources = _resources[typeof(T)];
        
        
        //Check the existing resource cache to avoid redundant loading
        if (typeResources.TryGetValue(resourceID, out var existing))
        {
            return new ResourceHandle<T>(this, resourceID);
        }
        //Create a new resource instance and load it
        var resource = factory(resourceID);
        if (!resource.Load())
        {
            return new ResourceHandle<T>();
        }
        //Cache successful resource and initialize tracking
        typeResources[resourceID] = resource;
        
        return new ResourceHandle<T>(this, resourceID);
    }
    /// <summary>
    /// Gets a resource of type T with the given ID inside the resource manager
    /// </summary>
    /// <param name="resourceID"></param>
    /// <typeparam name="T"></typeparam>
    /// <returns>Pointer to the resource requested by resourceID</returns>
    public T* GetResource<T>(string resourceID) where T : Resource
    {
        var typeResources = _resources[typeof(T)];

        if (typeResources.TryGetValue(resourceID, out var resource))
        {//resource found for this type
            return (T*)&resource;
        }
        //resource not found
        return null;
    }

    /// <summary>
    /// does the resource manager have a resource of type T with the given ID
    /// </summary>
    /// <param name="resourceID"></param>
    /// <typeparam name="T"></typeparam>
    /// <returns>true if it has a resource, false if it doesn't have the resource</returns>
    public bool HasResource<T>(string resourceID) where T : Resource
    {
        return _resources.ContainsKey(typeof(T)) && _resources[typeof(T)].ContainsKey(resourceID);
    }
    
    /// <summary>
    /// Releases a resource of type T with the given ID
    /// </summary>
    /// <param name="resourceID"></param>
    /// <typeparam name="T"></typeparam>
    public void ReleaseResource<T>(string resourceID) where T : Resource
    {
        var typeResources = _resources[typeof(T)];
        if (typeResources.TryGetValue(resourceID, out var resource))
        {
            resource.Unload();
            typeResources.Remove(resourceID);
        }
    }
    
    
    /// <summary>
    /// Releases all resources in the resource manager
    /// </summary>
    public void ReleaseAll()
    {
        foreach (var typeResources in _resources.Values)
        {
            foreach (var resource in typeResources.Values)
            {
                resource.Unload();
                
            }
            typeResources.Clear();
        }
        _resources.Clear();
    }
}

/// <summary>
/// Texture resource
/// </summary>
public unsafe class TextureResource(string id) : Resource(id)
{
    //Core Vulkan GPU resources for the texture
    VkImage image;
    DeviceMemory imageMemory;
    ImageView imageView;
    Sampler sampler;
    
    //texture metadata
    uint width = 0;
    uint height = 0;
    uint channels =0;

    ~TextureResource()
    {
        Unload();
    }

    public override bool Load()
    {
        //construct file path
        string filePath = "textures/" + GetId() + ".ktx";
        //load image data from disk with file
        var data = LoadImageData(filePath, width, height, channels);
        //Transform raw pixel data into Vulkan GPU resources
        CreateVulkanImage(data, width, height, channels);
        //Clean up temporary cpu memory
        FreeImageData(data);
        
        return base.Load(); //mark resource loaded
    }

    public override void Unload()
    {
        if (IsLoaded())
        {
            //destroy Vulkan GPU resources
            //temporary implementation
            Vk? vk = Vk.GetApi();
            //obtain device handle for resource destruction
            Device device = GetDevice();
            
            vk.DestroySampler(device, sampler, null);
            vk.DestroyImageView(device, imageView, null);
            vk.DestroyImage(device, image, null);
            vk.FreeMemory(device, imageMemory, null);
        }
        base.Unload();
    }
    
    public VkImage GetImage() => image;
    public ImageView GetImageView() => imageView;
    public Sampler GetSampler => sampler;
    
    
    private char* LoadImageData(string path, uint width, uint height, uint channels)
    {
        //temporary implementation
        //load image data from disk
        //return pointer to raw pixel data
        return null;
    }

    private void FreeImageData(char* data)
    {
        //temporary implementation
        //free memory allocated for raw pixel data
    }
    private void CreateVulkanImage(char* data, uint width, uint height, uint channels)
    {
        //temporary implementation
        //transform raw pixel data into Vulkan GPU resources involving complex vulkan operations
        // - Format selection based on channels
        // - malloc with appropriate usage flags
        // - create image with optimal tiling and memory properties
        //- Data upload via staging buffer
        // - create image view
        // - create sampler for appropriate filtering and wrap modes
    }

    private Device GetDevice()
    {
        
        return default;
    }
}


public unsafe class MeshResource : Resource
{
    int _offset = 0;
    int _stride = 0;

    public MeshResource(string id, int offset, int stride) : base(id)
    {
        _offset = offset;
        _stride = stride;
    }
    ~MeshResource()
    {
        Unload();
    }


    public override bool Load()
    {
        //construct file path
        string filePath = "models/" + GetId() + ".gltf";
        //Parse GLTF file and extract vertex and index buffers
        
     
        
        //transform cpu data into GPU resources
        
        return base.Load(); 
    }

    public override void Unload()
    {
        //only proceed if loaded
        if (IsLoaded())
        {
            Vk? vk = Vk.GetApi();
            Device device = GetDevice();
            
            //Destroy Buffers and free gpu resources
            //index buffers destroyed first to maintain dependency order
            
        }
        base.Unload();
    }
    
    
    
    private Device GetDevice()
    {
        return default;
    }

    private bool LoadMeshData(string path, out List<Vertex> vertices, out List<uint> indices)
    {
        //Implementation using bla bla bla
        // This method handles the complex task of:
        // - Opening and validating the mesh file format
        // - Parsing vertex attributes (positions, normals, UVs, etc.)
        // - Extracting index data that defines triangle connectivity
        // - Converting from file format to engine-specific vertex structures
        // - Performing validation to ensure data integrity
        // ...
        vertices = [];
        indices = [];
        return true;//placeholder
    }

    private void CreateVertexBuffer(List<VkVertex> vertices)
    {
        // Implementation to create Vulkan buffer, allocate memory, and upload data
        // This involves several complex Vulkan operations:
        // - Calculating buffer size requirements based on vertex count and structure
        // - Creating buffer with appropriate usage flags (vertex buffer usage)
        // - Allocating GPU memory with optimal memory type selection
        // - Uploading data via staging buffer for efficient transfer
        // - Setting up memory barriers to ensure data availability
        // ...
    }

    private void CreateIndexBuffer(List<uint> indices)
    {
        // Implementation to create Vulkan buffer, allocate memory, and upload data
        // Similar to vertex buffer creation but optimized for index data:
        // - Buffer creation with index buffer specific usage flags
        // - Memory allocation optimized for read-heavy access patterns
        // - Efficient data transfer using appropriate staging mechanisms
        // - Index format validation (16-bit vs 32-bit indices)
        // ...
    }
}

public unsafe class MaterialResource(string id) : Resource(id)
{
    Material material;
    ~MaterialResource()
    {
        Unload();
    }
    
}

public unsafe class ShaderResource(string id) : Resource(id)
{
    ShaderModule shaderModule;
    ShaderStageFlags stage;
    ~ShaderResource()
    {
        Unload();
    }

    public override bool Load()
    {
        //determine file ext based on shader
        string ext;
        switch (stage)
        {
            case ShaderStageFlags.VertexBit: ext = ".vert"; break;
            case ShaderStageFlags.FragmentBit: ext = ".frag"; break;
            case ShaderStageFlags.ComputeBit: ext = ".comp"; break;
            default: return false;
        }
        
        //load shader from file
        string filePath = "shaders/" + GetId() + ext + ".spv";
        
        //read shader code
        var shaderCode = File.ReadAllBytes(filePath);
        
        //create shader module
        CreateShaderModule(shaderCode);
        
        return base.Load();
    }

    public override void Unload()
    {
        if (IsLoaded())
        {
            Vk? vk = Vk.GetApi();
            //Get device from somewhere
            Device device = GetDevice();
            vk!.DestroyShaderModule(device, shaderModule, null);
            
        }
    }
    //getters for vulkan resources
    public ShaderModule GetShaderModule() => shaderModule;
    public ShaderStageFlags GetStage() => stage;
    private Device GetDevice()
    {
        return default;
    }

    private void CreateShaderModule(byte[] shaderCode)
    {
        //implementation to create vulkan shader module
        //...
    }
}

public class AsyncResourceManager : IDisposable
{
    ResourceManager _manager;
    Queue<Action> _workQueue = new();
    readonly object _queueLock = new();
    volatile bool _running = false;
    private bool _disposed = false;
    
    
    Thread? _workerThread;
    public AsyncResourceManager() => Start();

    ~AsyncResourceManager() => Stop();
    

    public void Start()
    {
        if (_running) return;
        
        _running = true;

        _workerThread = new Thread(WorkerThread)
        {
            IsBackground = true,
            Name = "AsyncResourceManager"
        };
        
        _workerThread.Start();
    }
    public void Stop()
    {
        lock (_queueLock)
        {
            _running = false;
            
            Monitor.Pulse(_queueLock);
        }
            
        if( _workerThread is {IsAlive: true})
            _workerThread.Join();
    }
    /// <summary>
    /// Enques a load operation. The callback fires on the worker thread
    ///once loading completes.
    ///
    /// <paramref name="factory"/> receives the resourceId and must return a
    /// new, uninitialised T — e.g. <c>id => new Texture(id)</c>.
    /// Load() is called on it internally by ResourceManager.
    /// </summary>
    /// <param name="resourceID"></param>
    /// <param name="factory"></param>
    /// <param name="callBack"></param>
    /// <typeparam name="T"></typeparam>
    public void LoadAsync<T>(string resourceID, Func<string, T> factory,Action<ResourceHandle<T>> callBack) where T : Resource
    {
        lock (_queueLock)
        {
            //capture resourceID and callback into the closure
            _workQueue.Enqueue(() =>
            {
                var handle = _manager.Load<T>(resourceID, factory);
                callBack(handle);
            });
            Monitor.Pulse(_queueLock);
        }
    }
    
    
    public void Dispose()
    {
        if (_disposed) return;
        Stop();
        _disposed = true;
        GC.SuppressFinalize(this);
    }

    private void WorkerThread()
    {
        while (true)
        {
            Action? task;
            //Hold the lock while inspecting the queue and _running, 
            //and while calling Monitor.Wait
            lock (_queueLock)
            {
                while (_workQueue.Count == 0 && _running)
                {
                    Monitor.Wait(_queueLock);
                }
                
                if(!_running && _workQueue.Count == 0)
                {
                    return;
                }
                
                task = _workQueue.Dequeue();
            }

            try
            {
                task();
            }
            catch (Exception e)
            {
                Console.Error.WriteLine($"[ASyncResourceManager] Task Threw: {e}");
            }
        }
    }
}

public class HotReloadResourceManager : ResourceManager
{
    //TODO: implement hot reload
    // var watcher = new FileSystemWatcher();
}