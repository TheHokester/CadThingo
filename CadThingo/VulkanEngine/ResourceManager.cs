using System.ComponentModel;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
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

    // Global vertex/index buffers — all meshes are packed into these shared device-local buffers.
    // Indices are rebased by the current vertex-write offset on upload so draws use vertexOffset=0.
    private Renderer.Renderer _renderer;

    private Buffer globalVertexBuffer;
    private DeviceMemory globalVertexBufferMemory;
    private int vertexWriteOffset;   // in vertices

    private Buffer globalIndexBuffer;
    private DeviceMemory globalIndexBufferMemory;
    private int indexWriteOffset;    // in indices

    private const int MAX_VERTICES = 1 << 20;   // 1M vertices
    private const int MAX_INDICES  = 1 << 22;   // 4M indices

    public Buffer GlobalVertexBuffer => globalVertexBuffer;
    public Buffer GlobalIndexBuffer  => globalIndexBuffer;

    public DeviceMemory GlobalVertexBufferMemory => globalVertexBufferMemory;
    public DeviceMemory GlobalIndexBufferMemory  => globalIndexBufferMemory;

    // Total vertices uploaded so far. Used as a conservative MaxVertex for AS builds —
    // safe because every mesh's index range is rebased into [0, VertexHighWater).
    public int VertexHighWater => vertexWriteOffset;
    public void Initialize(Renderer.Renderer renderer)
    {
        _renderer = renderer;

        ulong vbSize = (ulong)(MAX_VERTICES * sizeof(Vertex));
        ulong ibSize = (ulong)(MAX_INDICES  * sizeof(uint));

        renderer.CreateBuffer(vbSize,
            BufferUsageFlags.VertexBufferBit | BufferUsageFlags.TransferDstBit | BufferUsageFlags.ShaderDeviceAddressBit |
            BufferUsageFlags.AccelerationStructureBuildInputReadOnlyBitKhr,
            MemoryPropertyFlags.DeviceLocalBit,
            out globalVertexBuffer, out globalVertexBufferMemory);

        renderer.CreateBuffer(ibSize,
            BufferUsageFlags.IndexBufferBit | BufferUsageFlags.TransferDstBit | BufferUsageFlags.ShaderDeviceAddressBit |
            BufferUsageFlags.AccelerationStructureBuildInputReadOnlyBitKhr,
            MemoryPropertyFlags.DeviceLocalBit,
            out globalIndexBuffer, out globalIndexBufferMemory);
    }

    public Mesh UploadMesh(Vertex[] vertices, uint[] indices)
    {
        if (_renderer == null)
            throw new InvalidOperationException("ResourceManager.Initialize(renderer) not called");
        if (vertexWriteOffset + vertices.Length > MAX_VERTICES)
            throw new Exception($"Global vertex buffer full: {vertexWriteOffset + vertices.Length} > {MAX_VERTICES}");
        if (indexWriteOffset + indices.Length > MAX_INDICES)
            throw new Exception($"Global index buffer full: {indexWriteOffset + indices.Length} > {MAX_INDICES}");

        uint baseVertex = (uint)vertexWriteOffset;
        var rebased = new uint[indices.Length];
        for (int i = 0; i < indices.Length; i++) rebased[i] = indices[i] + baseVertex;

        ulong vbBytes     = (ulong)(vertices.Length * sizeof(Vertex));
        ulong ibBytes     = (ulong)(indices.Length  * sizeof(uint));
        ulong vbDstOffset = (ulong)(vertexWriteOffset * sizeof(Vertex));
        ulong ibDstOffset = (ulong)(indexWriteOffset  * sizeof(uint));

        fixed (Vertex* vPtr = vertices)
            _renderer.UploadBufferData(globalVertexBuffer, (long)vbDstOffset, vPtr, vbBytes);
        fixed (uint* iPtr = rebased)
            _renderer.UploadBufferData(globalIndexBuffer, (long)ibDstOffset, iPtr, ibBytes);

        var mesh = new Mesh { offset = indexWriteOffset, count = indices.Length };
        vertexWriteOffset += vertices.Length;
        indexWriteOffset  += indices.Length;
        return mesh;
    }

    public void Dispose()
    {
        ReleaseAll();
        if (_renderer != null)
        {
            _renderer.DestroyBuffer(globalVertexBuffer, globalVertexBufferMemory);
            _renderer.DestroyBuffer(globalIndexBuffer,  globalIndexBufferMemory);
        }
    }
    
    
    
    
    ///<summary>load a resource of type T with the given ID</summary>
    /// <typeparam name="T">Resource type</typeparam>
    /// <param name="resourceID">Unique identifier for the resource</param>
    /// <param name="factory">Factory function to create the resource</param>
    public ResourceHandle<T> Load<T>(string resourceID, Func<string, T> factory) where T : Resource
    {
        // Lazy-init type bucket so first-time Load of a type doesn't throw KeyNotFoundException.
        if (!_resources.TryGetValue(typeof(T), out var typeResources))
        {
            typeResources = new Dictionary<string, Resource>();
            _resources[typeof(T)] = typeResources;
        }

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

    public Mesh* GetMesh(string id)
    {
        if (!_resources.TryGetValue(typeof(MeshResource), out var typeDict))
            throw new InvalidOperationException($"No MeshResource bucket — nothing loaded yet");
        if (!typeDict.TryGetValue(id, out var res))
            throw new KeyNotFoundException($"MeshResource '{id}' not found");
        return ((MeshResource)res).GetMesh();
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
    protected Mesh* mesh;  // unmanaged so MeshComponent can hold a stable Mesh*
    protected ResourceManager manager;

    public MeshResource(string id, ResourceManager manager) : base(id)
    {
        this.manager = manager;
        mesh = (Mesh*)NativeMemory.AllocZeroed((nuint)sizeof(Mesh));
    }
    ~MeshResource()
    {
        Unload();
    }

    public Mesh* GetMesh() => mesh;

    public override bool Load()
    {
        if (!LoadMeshData(out var vertices, out var indices))
            return false;
        *mesh = manager.UploadMesh(vertices, indices);
        return base.Load();
    }

    public override void Unload()
    {
        // Per-mesh destruction (GPU data) is not yet implemented — global VB/IB is grow-only
        // until compaction/free-list is added. Refcounts still drive logical lifetime.
        if (mesh != null)
        {
            NativeMemory.Free(mesh);
            mesh = null;
        }
        base.Unload();
    }

    // Subclasses override to supply vertex/index data (file, procedural, etc.).
    protected virtual bool LoadMeshData(out Vertex[] vertices, out uint[] indices)
    {
        vertices = Array.Empty<Vertex>();
        indices  = Array.Empty<uint>();
        return false;
    }
}

public static class CubeMesh
{
    /// Returns 24 vertices + 36 indices for a unit cube centered at origin.
    /// Each face has its own 4 vertices so per-face normals/UVs are correct.
    public static (Vertex[] vertices, uint[] indices) Generate()
    {
        const float h = 0.5f; // half-extent
        var vertices = new Vertex[24];
        var indices  = new uint[36];

        // +X (right)
        AddFace(vertices, indices, 0,
            new Vector3( h,-h, h), new Vector3( h,-h,-h), new Vector3( h, h,-h), new Vector3( h, h, h),
            new Vector3( 1, 0, 0), new Vector4( 0, 0,-1, 1));
        // -X (left)
        AddFace(vertices, indices, 1,
            new Vector3(-h,-h,-h), new Vector3(-h,-h, h), new Vector3(-h, h, h), new Vector3(-h, h,-h),
            new Vector3(-1, 0, 0), new Vector4( 0, 0, 1, 1));
        // +Y (up)
        AddFace(vertices, indices, 2,
            new Vector3(-h, h, h), new Vector3( h, h, h), new Vector3( h, h,-h), new Vector3(-h, h,-h),
            new Vector3( 0, 1, 0), new Vector4( 1, 0, 0, 1));
        // -Y (down)
        AddFace(vertices, indices, 3,
            new Vector3(-h,-h,-h), new Vector3( h,-h,-h), new Vector3( h,-h, h), new Vector3(-h,-h, h),
            new Vector3( 0,-1, 0), new Vector4( 1, 0, 0, 1));
        // +Z (front)
        AddFace(vertices, indices, 4,
            new Vector3(-h,-h, h), new Vector3( h,-h, h), new Vector3( h, h, h), new Vector3(-h, h, h),
            new Vector3( 0, 0, 1), new Vector4( 1, 0, 0, 1));
        // -Z (back)
        AddFace(vertices, indices, 5,
            new Vector3( h,-h,-h), new Vector3(-h,-h,-h), new Vector3(-h, h,-h), new Vector3( h, h,-h),
            new Vector3( 0, 0,-1), new Vector4(-1, 0, 0, 1));

        return (vertices, indices);
    }

    private static void AddFace(Vertex[] vertices, uint[] indices, int faceIndex,
        Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3,
        Vector3 normal, Vector4 tangent)
    {
        int v = faceIndex * 4;
        vertices[v + 0] = new Vertex { Position = p0, Normal = normal, TexCoord = new Vector2(0, 0), Tangent = tangent };
        vertices[v + 1] = new Vertex { Position = p1, Normal = normal, TexCoord = new Vector2(1, 0), Tangent = tangent };
        vertices[v + 2] = new Vertex { Position = p2, Normal = normal, TexCoord = new Vector2(1, 1), Tangent = tangent };
        vertices[v + 3] = new Vertex { Position = p3, Normal = normal, TexCoord = new Vector2(0, 1), Tangent = tangent };

        int i = faceIndex * 6;
        indices[i + 0] = (uint)(v + 0);
        indices[i + 1] = (uint)(v + 1);
        indices[i + 2] = (uint)(v + 2);
        indices[i + 3] = (uint)(v + 0);
        indices[i + 4] = (uint)(v + 2);
        indices[i + 5] = (uint)(v + 3);
    }
}

public unsafe class ProceduralCubeResource : MeshResource
{
    public ProceduralCubeResource(string id, ResourceManager manager) : base(id, manager) { }

    protected override bool LoadMeshData(out Vertex[] vertices, out uint[] indices)
    {
        (vertices, indices) = CubeMesh.Generate();
        return true;
    }
}

// Loads a mesh from any Assimp-supported format (.obj, .fbx, ...) into the engine's
// Vertex layout. Generates smooth normals + tangents when missing, joins identical
// verts, flips V so Vulkan-style top-left UV origin works.
public unsafe class ObjMeshResource : MeshResource
{
    private readonly string _filePath;

    public ObjMeshResource(string id, ResourceManager manager, string filePath) : base(id, manager)
    {
        _filePath = filePath;
    }

    protected override bool LoadMeshData(out Vertex[] vertices, out uint[] indices)
    {
        using var assimp = Silk.NET.Assimp.Assimp.GetApi();

        var flags = (uint)(Silk.NET.Assimp.PostProcessSteps.Triangulate
                           | Silk.NET.Assimp.PostProcessSteps.JoinIdenticalVertices
                           | Silk.NET.Assimp.PostProcessSteps.GenerateSmoothNormals
                           | Silk.NET.Assimp.PostProcessSteps.CalculateTangentSpace
                           | Silk.NET.Assimp.PostProcessSteps.FlipUVs);

        var scene = assimp.ImportFile(_filePath, flags);
        if (scene == null || scene->MRootNode == null)
        {
            Console.Error.WriteLine($"[ObjMeshResource] Assimp failed to load '{_filePath}'");
            vertices = Array.Empty<Vertex>();
            indices  = Array.Empty<uint>();
            return false;
        }

        var vertexMap = new Dictionary<Vertex, uint>();
        var vertexList = new List<Vertex>();
        var indexList  = new List<uint>();

        VisitNode(scene->MRootNode);

        assimp.ReleaseImport(scene);

        vertices = vertexList.ToArray();
        indices  = indexList.ToArray();
        Console.WriteLine($"[ObjMeshResource] Loaded '{_filePath}': {vertices.Length} verts, {indices.Length} indices");
        return true;

        void VisitNode(Silk.NET.Assimp.Node* node)
        {
            for (var m = 0; m < node->MNumMeshes; m++)
            {
                var aMesh = scene->MMeshes[node->MMeshes[m]];
                bool hasNormals  = aMesh->MNormals  != null;
                bool hasTangents = aMesh->MTangents != null;
                bool hasUv       = aMesh->MTextureCoords[0] != null;

                for (var f = 0; f < aMesh->MNumFaces; f++)
                {
                    var face = aMesh->MFaces[f];
                    for (var i = 0; i < face.MNumIndices; i++)
                    {
                        var idx = face.MIndices[i];
                        var p   = aMesh->MVertices[idx];
                        var n   = hasNormals  ? aMesh->MNormals[idx]          : default;
                        var uv  = hasUv       ? aMesh->MTextureCoords[0][idx] : default;
                        var t   = hasTangents ? aMesh->MTangents[idx]         : default;

                        Vertex v = new()
                        {
                            Position = new Vector3(p.X, p.Y, p.Z),
                            Normal   = hasNormals  ? new Vector3(n.X, n.Y, n.Z) : new Vector3(0, 1, 0),
                            TexCoord = hasUv       ? new Vector2(uv.X, uv.Y)    : new Vector2(0, 0),
                            Tangent  = hasTangents ? new Vector4(t.X, t.Y, t.Z, 1.0f) : new Vector4(1, 0, 0, 1),
                        };

                        if (vertexMap.TryGetValue(v, out var existing))
                        {
                            indexList.Add(existing);
                        }
                        else
                        {
                            uint newIdx = (uint)vertexList.Count;
                            indexList.Add(newIdx);
                            vertexMap[v] = newIdx;
                            vertexList.Add(v);
                        }
                    }
                }
            }

            for (var c = 0; c < node->MNumChildren; c++)
                VisitNode(node->MChildren[c]);
        }
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