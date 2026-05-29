using System.ComponentModel;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using CadThingo.GraphicsPipeline;
using CadThingo.VulkanEngine;
using CadThingo.VulkanEngine.Renderer;
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
    

    public T? Get()
    {
        if (_manager == null) return default;
        return _manager.GetResource<T>(_ID);
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


/// <summary>
/// Per-file capture of every resource ResourceManager produced during a load.
/// Filled when an active <see cref="ResourceManager.BeginManifestCapture"/>
/// scope is open. Lets the editor walk every mesh / texture / bindless slot a
/// file owns when it wants to free them.
///
/// Materials, entities, and BLAS keys are filled by GltfLoader / the panel —
/// ResourceManager doesn't see those.
/// </summary>
public sealed class LoadManifest
{
    public readonly Dictionary<Type, List<string>> ResourceIdsByType = new();
    public readonly List<int> BindlessIndices = new();
    // Filled by GltfLoader / panel.
    public readonly List<int>  MaterialIndices = new();
    public readonly List<nint> Entities        = new();
    public readonly List<nint> MeshPtrs        = new();
}

public unsafe class ResourceManager
{
    //2 level storage system, organise by type then unique identifier
    Dictionary<Type, Dictionary<string, Resource>> _resources = new();

    // Currently-active manifest capture, set by BeginManifestCapture and read
    // by Load<T> and RegisterBindless. Null when no load is in progress, in
    // which case both paths skip the tracking branch (cheap nullable check).
    private LoadManifest? _activeManifest;

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
    private SubAlloc globalVertexBufferAlloc;
    private int vertexWriteOffset;   // in vertices

    private Buffer globalIndexBuffer;
    private SubAlloc globalIndexBufferAlloc;
    private int indexWriteOffset;    // in indices

    // Free-list of returned ranges keyed by (offset, count), sorted ascending by
    // offset and coalesced on insert. UploadMesh first-fits the free-list before
    // bumping the watermark. FreeMesh inserts back. The two lists stay
    // independent because the VB and IB ranges for a single mesh land in
    // separate-sized slots most of the time.
    private readonly List<(int offset, int count)> _vbFreeList = new();
    private readonly List<(int offset, int count)> _ibFreeList = new();

    private const int MAX_VERTICES = 1 << 23;   // 4M vertices
    private const int MAX_INDICES  = 1 << 25;   // 16M indices

    // Bindless texture table — every Texture registered here gets a stable int index that
    // PbrMaterial entries reference. The renderer's bindless descriptor set sees this slot
    // through the StructuredBuffer<Texture2D> array binding. Indices never shift mid-run;
    // freed slots go onto _bindlessFree for reuse.
    private readonly List<Texture>          _bindlessTable        = new();
    private readonly Stack<int>             _bindlessFree         = new();
    // Reverse index — same Texture handle should ever only consume one slot.
    // Without this, materials referencing the same glTF image (very common with
    // KHR_materials_clearcoat where coat / roughness / normal often share an
    // ORM-style atlas) would burn through bindless slots redundantly.
    private readonly Dictionary<Texture, int> _bindlessIndexByTexture = new();

    public Buffer GlobalVertexBuffer => globalVertexBuffer;
    public Buffer GlobalIndexBuffer  => globalIndexBuffer;
    
    // Total vertices uploaded so far. Used as a conservative MaxVertex for AS builds —
    // safe because every mesh's index range is rebased into [0, VertexHighWater).
    public int VertexHighWater => vertexWriteOffset;

    // CPU-side mesh geometry (object-space positions + local 0-based indices),
    // keyed by the mesh's index-buffer offset (Mesh.offset, unique per live
    // mesh). Retained at upload so the pathtracer can extract emissive triangles
    // in world space — the global VB/IB are device-local and can't be read back.
    // Released by FreeMesh.
    private readonly Dictionary<int, (Vector3[] positions, uint[] indices)> _meshCpuGeometry = new();

    /// <summary>Object-space positions + local (0-based) indices for a mesh,
    /// keyed by <see cref="Mesh.offset"/>. Used to build emissive area lights.
    /// Returns false for meshes not uploaded through <see cref="UploadMesh"/>.</summary>
    public bool TryGetMeshGeometry(int meshOffset, out Vector3[] positions, out uint[] indices)
    {
        if (_meshCpuGeometry.TryGetValue(meshOffset, out var geo))
        {
            positions = geo.positions;
            indices   = geo.indices;
            return true;
        }
        positions = null!;
        indices   = null!;
        return false;
    }


    private Vk vk;
    private Device device;
    private DescriptorPool descriptorPool;
    private DescriptorSet[] bindlessDescriptorSets = new DescriptorSet[(int)Renderer.Renderer.MAX_CONCURRENT_FRAMES];
    private Sampler defaultBindlessSampler;
    
    internal const uint MAX_MATERIALS         = 256;
    internal const uint MAX_INSTANCES         = 4096;
    // 9 = the worst-case fields-with-textures count on PbrMaterial after the
    // KHR extensions (5 core + transmission + 3× clearcoat). Sized for the
    // worst case so a clearcoated/transmissive asset can't exhaust the table.
    internal const uint MAX_BINDLESS_TEXTURES = MAX_MATERIALS * 9;
    
    DescriptorSetLayout MaterialBindlessLayout;
    
    private UboBuffer[] MaterialStorageBuffers = new UboBuffer[Renderer.Renderer.MAX_CONCURRENT_FRAMES];
    private UboBuffer[] InstanceStorageBuffers = new UboBuffer[Renderer.Renderer.MAX_CONCURRENT_FRAMES];
     
    public void Initialize(Renderer.Renderer renderer)
    {
        _renderer = renderer;
        vk = renderer.vk;
        device = renderer.device;
        ulong vbSize = (ulong)(MAX_VERTICES * sizeof(Vertex));
        ulong ibSize = (ulong)(MAX_INDICES  * sizeof(uint));

        // StorageBufferBit so the PBR lighting shadow-ray alpha-test path can bind
        // these as ByteAddressBuffers and pull UVs / indices at hit time.
        renderer.CreateBuffer(vbSize,
            BufferUsageFlags.VertexBufferBit | BufferUsageFlags.TransferDstBit | BufferUsageFlags.ShaderDeviceAddressBit |
            BufferUsageFlags.StorageBufferBit |
            BufferUsageFlags.AccelerationStructureBuildInputReadOnlyBitKhr,
            MemoryPropertyFlags.DeviceLocalBit,
            out globalVertexBuffer, out globalVertexBufferAlloc);

        renderer.CreateBuffer(ibSize,
            BufferUsageFlags.IndexBufferBit | BufferUsageFlags.TransferDstBit | BufferUsageFlags.ShaderDeviceAddressBit |
            BufferUsageFlags.StorageBufferBit |
            BufferUsageFlags.AccelerationStructureBuildInputReadOnlyBitKhr,
            MemoryPropertyFlags.DeviceLocalBit,
            out globalIndexBuffer, out globalIndexBufferAlloc);
        
        CreateBindlessDescriptorSetLayout();
        for (int i = 0; i < Renderer.Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            renderer.CreateMappedStorageBuffer((ulong)(Renderer.Renderer.MAX_MATERIALS * (uint)sizeof(PbrMaterial)),     ref MaterialStorageBuffers[i]);
            renderer.CreateMappedStorageBuffer((ulong)(Renderer.Renderer.MAX_INSTANCES * (uint)sizeof(InstanceDataGPU)), ref InstanceStorageBuffers[i]);
        }
        CreateDefaultBindlessSampler();
        CreateBindlessDescriptorSets();
    }

    public Mesh UploadMesh(Vertex[] vertices, uint[] indices)
    {
        if (_renderer == null)
            throw new InvalidOperationException("ResourceManager.Initialize(renderer) not called");

        int vbOffset = AllocateRange(_vbFreeList, vertices.Length, ref vertexWriteOffset, MAX_VERTICES, "vertex");
        int ibOffset = AllocateRange(_ibFreeList, indices.Length,  ref indexWriteOffset,  MAX_INDICES,  "index");

        // Rebase against the actual allocated VB offset (may be a freed slot
        // below vertexWriteOffset, not the watermark) so the indices point at
        // the new vertices wherever they landed.
        uint baseVertex = (uint)vbOffset;
        var rebased = new uint[indices.Length];
        for (int i = 0; i < indices.Length; i++) rebased[i] = indices[i] + baseVertex;

        ulong vbBytes     = (ulong)(vertices.Length * sizeof(Vertex));
        ulong ibBytes     = (ulong)(indices.Length  * sizeof(uint));
        ulong vbDstOffset = (ulong)(vbOffset * sizeof(Vertex));
        ulong ibDstOffset = (ulong)(ibOffset * sizeof(uint));

        fixed (Vertex* vPtr = vertices)
            _renderer.UploadBufferData(globalVertexBuffer, (long)vbDstOffset, vPtr, vbBytes);
        fixed (uint* iPtr = rebased)
            _renderer.UploadBufferData(globalIndexBuffer, (long)ibDstOffset, iPtr, ibBytes);

        // Bounding sphere in mesh-local space — center = AABB center, radius =
        // max distance from center to any vertex. Looser than Welzl's but O(n)
        // and zero-alloc; the GPU cull only needs an over-approximation.
        Vector3 mn = new(float.PositiveInfinity), mx = new(float.NegativeInfinity);
        for (int i = 0; i < vertices.Length; i++)
        {
            mn = Vector3.Min(mn, vertices[i].Position);
            mx = Vector3.Max(mx, vertices[i].Position);
        }
        Vector3 center = (mn + mx) * 0.5f;
        float r2 = 0f;
        for (int i = 0; i < vertices.Length; i++)
        {
            float d2 = (vertices[i].Position - center).LengthSquared();
            if (d2 > r2) r2 = d2;
        }
        float radius = MathF.Sqrt(r2);

        var mesh = new Mesh
        {
            offset       = ibOffset,
            count        = indices.Length,
            sphereLocal  = new Vector4(center, radius),
            vertexOffset = vbOffset,
            vertexCount  = vertices.Length,
        };

        // Retain object-space positions + local indices for emissive-light
        // extraction. Keyed by ibOffset (== mesh.offset). Cloned so later mutation
        // of the caller's arrays can't corrupt the cache. Overwrites any stale
        // entry left at a reused free-list offset.
        var positions = new Vector3[vertices.Length];
        for (int i = 0; i < vertices.Length; i++) positions[i] = vertices[i].Position;
        _meshCpuGeometry[ibOffset] = (positions, (uint[])indices.Clone());

        return mesh;
    }

    /// <summary>
    /// First-fit allocator across the free-list with a watermark fallback. When
    /// no free-list entry is large enough we append at the high-water mark and
    /// bump it. Hard-fails when neither path can satisfy the request.
    /// </summary>
    private static int AllocateRange(List<(int offset, int count)> freeList,
                                     int needed,
                                     ref int watermark,
                                     int capacity,
                                     string label)
    {
        for (int i = 0; i < freeList.Count; i++)
        {
            var (off, cnt) = freeList[i];
            if (cnt < needed) continue;

            if (cnt == needed) freeList.RemoveAt(i);
            else               freeList[i] = (off + needed, cnt - needed);
            return off;
        }

        if (watermark + needed > capacity)
            throw new Exception($"Global {label} buffer full: {watermark + needed} > {capacity}");

        int allocated = watermark;
        watermark += needed;
        return allocated;
    }

    /// <summary>Returns a range to the free-list and coalesces with adjacent neighbours.</summary>
    private static void ReleaseRange(List<(int offset, int count)> freeList, int offset, int count)
    {
        if (count <= 0) return;

        // Sorted insert by offset.
        int i = 0;
        while (i < freeList.Count && freeList[i].offset < offset) i++;
        freeList.Insert(i, (offset, count));

        // Coalesce with right neighbour first (cheaper bookkeeping when we
        // remove a later index), then with left.
        if (i + 1 < freeList.Count &&
            freeList[i].offset + freeList[i].count == freeList[i + 1].offset)
        {
            freeList[i] = (freeList[i].offset, freeList[i].count + freeList[i + 1].count);
            freeList.RemoveAt(i + 1);
        }
        if (i > 0 &&
            freeList[i - 1].offset + freeList[i - 1].count == freeList[i].offset)
        {
            freeList[i - 1] = (freeList[i - 1].offset, freeList[i - 1].count + freeList[i].count);
            freeList.RemoveAt(i);
        }
    }

    /// <summary>
    /// Returns the VB and IB ranges owned by <paramref name="mesh"/> to the
    /// free-lists so a future UploadMesh can reuse them. Caller is responsible
    /// for ensuring no in-flight GPU work still references the ranges
    /// (DeviceWaitIdle before calling, typically driven by the editor).
    /// </summary>
    public void FreeMesh(Mesh mesh)
    {
        ReleaseRange(_vbFreeList, mesh.vertexOffset, mesh.vertexCount);
        ReleaseRange(_ibFreeList, mesh.offset,       mesh.count);
        _meshCpuGeometry.Remove(mesh.offset);
    }

    /// <summary>Sum of free-list bytes for the global VB and IB. Editor stat.</summary>
    public (long vbFreeBytes, long ibFreeBytes) GetMeshFreeStats()
    {
        long vb = 0; foreach (var e in _vbFreeList) vb += (long)e.count * sizeof(Vertex);
        long ib = 0; foreach (var e in _ibFreeList) ib += (long)e.count * sizeof(uint);
        return (vb, ib);
    }

    public void Dispose()
    {
        ReleaseAll();
        if (_renderer != null)
        {
            _renderer.DestroyBuffer(globalVertexBuffer, globalVertexBufferAlloc);
            _renderer.DestroyBuffer(globalIndexBuffer,  globalIndexBufferAlloc);

            vk!.DestroySampler(device, defaultBindlessSampler, null);
            for (int i = 0; i < Renderer.Renderer.MAX_CONCURRENT_FRAMES; i++)
            {
                _renderer.DestroyBuffer(MaterialStorageBuffers[i].buffer, MaterialStorageBuffers[i].alloc);
                _renderer.DestroyBuffer(InstanceStorageBuffers[i].buffer, InstanceStorageBuffers[i].alloc);
            }

            // Bindless descriptor set layout is created in CreateBindlessDescriptorSetLayout
            // and owned here — GeometryPipeline borrows it via GetBindlessLayout.
            if (MaterialBindlessLayout.Handle != 0)
                vk!.DestroyDescriptorSetLayout(device, MaterialBindlessLayout, null);
        }
    }
    public DescriptorSetLayout GetBindlessLayout() => MaterialBindlessLayout;
    public DescriptorSet GetBindlessSet(uint frameIndex) => bindlessDescriptorSets[frameIndex];
    public Buffer GetMaterialBuffer(int frameIndex) => MaterialStorageBuffers[frameIndex].buffer;
    public Buffer GetInstanceBuffer(uint frameIndex) => InstanceStorageBuffers[frameIndex].buffer;
    public void* GetMaterialMapped(uint frameIndex) => MaterialStorageBuffers[frameIndex].mapped;
    public void* GetInstanceMapped(int frameIndex) => InstanceStorageBuffers[frameIndex].mapped;
    
    
    /// <summary>
    /// Adds a Texture to the global bindless table and writes its image view into the renderer's
    /// bindless descriptor set at the returned index. The PbrMaterial fields baseColorTex/normalTex/...
    /// store these indices and the geometry shader samples textures[index].
    /// </summary>
    public int RegisterBindless(Texture tex)
    {
        if (_renderer == null)
            throw new InvalidOperationException("ResourceManager.Initialize(renderer) not called");

        // Same Texture handle → same bindless index. Reference equality is what
        // we want here — the resource manager already dedups Texture instances
        // by id at the load layer, so identical content shares the same handle.
        if (_bindlessIndexByTexture.TryGetValue(tex, out int existing))
            return existing;

        int index;
        if (_bindlessFree.Count > 0)
        {
            index = _bindlessFree.Pop();
            _bindlessTable[index] = tex;
        }
        else
        {
            index = _bindlessTable.Count;
            _bindlessTable.Add(tex);
        }

        _bindlessIndexByTexture[tex] = index;
        _renderer.WriteBindlessTexture(index, tex, bindlessDescriptorSets, 2);
        // Only newly-allocated slots land on the manifest; dedup hits returned
        // above without writing a fresh descriptor, so the caller-file doesn't
        // own those slots and shouldn't unregister them later.
        _activeManifest?.BindlessIndices.Add(index);
        return index;
    }

    /// <summary>
    /// Releases a bindless slot. The slot's descriptor is rewritten to point at
    /// the white BaseColor default (a stable always-loaded texture) so any stale
    /// material entry still referencing the slot reads safe data instead of a
    /// freed VkImage. Future <see cref="RegisterBindless"/> can reuse the slot.
    /// No-op if the slot wasn't currently allocated.
    /// </summary>
    public void UnregisterBindless(int index, Texture fallback)
    {
        if (_renderer == null) return;
        if (index < 0 || index >= _bindlessTable.Count) return;

        var existing = _bindlessTable[index];
        if (existing == null) return; // already freed

        _bindlessTable[index] = null!;
        _bindlessIndexByTexture.Remove(existing);
        _bindlessFree.Push(index);

        // Point the descriptor at a known-good default. Validation would yell
        // if a shader sampled a slot whose underlying VkImage we destroyed
        // before this rewrite landed — callers must DeviceWaitIdle first.
        _renderer.WriteBindlessTexture(index, fallback, bindlessDescriptorSets, 2);
    }

    /// <summary>
    /// Opens a manifest-capture scope. While the returned IDisposable is alive,
    /// every Load&lt;T&gt; and every newly-allocated bindless slot is appended
    /// to <paramref name="manifest"/>. Nesting / concurrency are not supported
    /// — only one capture may be active at a time.
    /// </summary>
    public IDisposable BeginManifestCapture(LoadManifest manifest)
    {
        if (_activeManifest != null)
            throw new InvalidOperationException("A manifest capture is already active.");
        _activeManifest = manifest;
        return new ManifestScope(this);
    }

    private sealed class ManifestScope : IDisposable
    {
        private readonly ResourceManager _rm;
        public ManifestScope(ResourceManager rm) => _rm = rm;
        public void Dispose() => _rm._activeManifest = null;
    }
    
    private void CreateBindlessDescriptorSetLayout()
    {
        var bindings = new DescriptorSetLayoutBinding[]
        {
            new()
            {
                Binding = 0,
                DescriptorType = DescriptorType.StorageBuffer,
                DescriptorCount = 1,
                StageFlags = ShaderStageFlags.FragmentBit | ShaderStageFlags.ComputeBit,
            },
            new()
            {
                Binding = 1,
                DescriptorType = DescriptorType.StorageBuffer,
                DescriptorCount = 1,
                StageFlags = ShaderStageFlags.VertexBit|ShaderStageFlags.FragmentBit | ShaderStageFlags.ComputeBit,
            },
            new()
            {
                Binding = 2,
                DescriptorType = DescriptorType.SampledImage,
                DescriptorCount = MAX_MATERIALS * 5,
                StageFlags = ShaderStageFlags.FragmentBit | ShaderStageFlags.ComputeBit,
            }, 
            new()
            {
                Binding = 3,
                DescriptorType = DescriptorType.Sampler,
                DescriptorCount = 8,
                StageFlags = ShaderStageFlags.FragmentBit | ShaderStageFlags.ComputeBit,
            }
        };

        // Make the shared bindless set bindable from the RT-pipeline path tracer's
        // hit/miss/raygen stages too — but only when ray tracing pipelines are
        // actually supported, since declaring RT stage bits on a device without
        // the extension is invalid.
        if (_renderer.RayTracePipelineSupported)
        {
            var rt = ShaderStageFlags.RaygenBitKhr | ShaderStageFlags.MissBitKhr |
                     ShaderStageFlags.ClosestHitBitKhr | ShaderStageFlags.AnyHitBitKhr;
            for (int i = 0; i < bindings.Length; i++) bindings[i].StageFlags |= rt;
        }



        DescriptorSetLayoutBindingFlagsCreateInfo flagsCreateInfo = new()
            { SType = StructureType.DescriptorSetLayoutBindingFlagsCreateInfo };
        var flags = stackalloc DescriptorBindingFlags[bindings.Length];

        fixed (DescriptorSetLayoutBinding* pBindings = bindings)
        {
            if (_renderer.descriptorIndexEnabled)
            {
                // Only the bindless texture array (binding 2) needs UpdateAfterBind +
                // PartiallyBound — RegisterBindless writes new texture slots while the set
                // is live. The two storage buffers (bindings 0,1) and sampler (3) are
                // written once at setup, so they don't need UpdateAfterBind (which would
                // require descriptorBindingStorageBufferUpdateAfterBind / SamplerUpdateAfterBind
                // features that we don't request).
                flags[0] = DescriptorBindingFlags.UpdateUnusedWhilePendingBit;
                flags[1] = DescriptorBindingFlags.UpdateUnusedWhilePendingBit;
                flags[2] = DescriptorBindingFlags.UpdateAfterBindBit |
                           DescriptorBindingFlags.UpdateUnusedWhilePendingBit |
                           DescriptorBindingFlags.PartiallyBoundBit;
                flags[3] = DescriptorBindingFlags.UpdateUnusedWhilePendingBit;

                flagsCreateInfo.BindingCount = (uint)bindings.Length;
                flagsCreateInfo.PBindingFlags = flags;
            }

            DescriptorSetLayoutCreateInfo layoutInfo = new()
            {
                SType = StructureType.DescriptorSetLayoutCreateInfo,
                BindingCount = (uint)bindings.Length,
                PBindings = pBindings,
            };
            if (_renderer.descriptorIndexEnabled)
            {
                layoutInfo.Flags |= DescriptorSetLayoutCreateFlags.UpdateAfterBindPoolBit;
                layoutInfo.PNext = &flagsCreateInfo;
            }

            if (vk!.CreateDescriptorSetLayout(device, &layoutInfo, null, out MaterialBindlessLayout) !=
                Result.Success)
                throw new Exception("Failed to create geometry material descriptor set layout");
        }
    }
    
    // Allocates one bindless descriptor set per frame in flight from the main pool. Writes the
    // per-frame StorageBuffer<PbrMaterial> + StorageBuffer<InstanceData> bindings, plus the shared
    // sampler at samplers[0]. Texture array (binding 2) is populated lazily by RegisterBindless.
    private void CreateBindlessDescriptorSets()
    {
        var layouts = stackalloc DescriptorSetLayout[(int)Renderer.Renderer.MAX_CONCURRENT_FRAMES];
        for (var i = 0; i < Renderer.Renderer.MAX_CONCURRENT_FRAMES; i++) layouts[i] = MaterialBindlessLayout;

        DescriptorSetAllocateInfo alloc = new()
        {
            SType = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool = _renderer.descriptorPool,
            DescriptorSetCount = Renderer.Renderer.MAX_CONCURRENT_FRAMES,
            PSetLayouts = layouts,
        };
        fixed (DescriptorSet* pSets = bindlessDescriptorSets)
        {
            if (vk!.AllocateDescriptorSets(device, &alloc, pSets) != Result.Success)
                throw new Exception("Failed to allocate bindless descriptor sets");
        }

        for (var i = 0; i < Renderer.Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            DescriptorBufferInfo matInfo = new()
            {
                Buffer = MaterialStorageBuffers[i].buffer, Offset = 0,
                Range = (ulong)(MAX_MATERIALS * (uint)sizeof(PbrMaterial)),
            };
            DescriptorBufferInfo instInfo = new()
            {
                Buffer = InstanceStorageBuffers[i].buffer, Offset = 0,
                Range = (ulong)(MAX_INSTANCES * (uint)sizeof(InstanceDataGPU)),
            };
            DescriptorImageInfo samplerInfo = new()
            {
                Sampler = defaultBindlessSampler,
            };

            var writes = stackalloc WriteDescriptorSet[3];
            writes[0] = new WriteDescriptorSet
            {
                SType = StructureType.WriteDescriptorSet,
                DstSet = bindlessDescriptorSets[i],
                DstBinding = 0,
                DescriptorType = DescriptorType.StorageBuffer,
                DescriptorCount = 1,
                PBufferInfo = &matInfo,
            };
            writes[1] = new WriteDescriptorSet
            {
                SType = StructureType.WriteDescriptorSet,
                DstSet = bindlessDescriptorSets[i],
                DstBinding = 1,
                DescriptorType = DescriptorType.StorageBuffer,
                DescriptorCount = 1,
                PBufferInfo = &instInfo,
            };
            writes[2] = new WriteDescriptorSet
            {
                SType = StructureType.WriteDescriptorSet,
                DstSet = bindlessDescriptorSets[i],
                DstBinding = 3,
                DstArrayElement = 0,
                DescriptorType = DescriptorType.Sampler,
                DescriptorCount = 1,
                PImageInfo = &samplerInfo,
            };
            vk!.UpdateDescriptorSets(device, 3, writes, 0, null);
        }
    }
    private void CreateDefaultBindlessSampler()
    {
        SamplerCreateInfo samplerInfo = new()
        {
            SType = StructureType.SamplerCreateInfo,
            MagFilter = Filter.Linear,
            MinFilter = Filter.Linear,
            AddressModeU = SamplerAddressMode.Repeat,
            AddressModeV = SamplerAddressMode.Repeat,
            AddressModeW = SamplerAddressMode.Repeat,
            MipmapMode = SamplerMipmapMode.Linear,
            AnisotropyEnable = true,
            MaxAnisotropy = 16,
            BorderColor = BorderColor.FloatOpaqueBlack,
            UnnormalizedCoordinates = false,
            CompareEnable = false,
            CompareOp = CompareOp.Always,
            MinLod = 0.0f, 
            MaxLod = Vk.LodClampNone,
        };
        if (vk!.CreateSampler(device, &samplerInfo, null, out defaultBindlessSampler) != Result.Success)
            throw new Exception("Failed to create default bindless sampler");
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

        // Only newly-created resources land on the manifest. Cache hits don't
        // — the file that gets the dedup'd handle doesn't own the resource.
        if (_activeManifest != null)
        {
            if (!_activeManifest.ResourceIdsByType.TryGetValue(typeof(T), out var ids))
            {
                ids = new List<string>();
                _activeManifest.ResourceIdsByType[typeof(T)] = ids;
            }
            ids.Add(resourceID);
        }

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
    public T? GetResource<T>(string resourceID) where T : Resource
    {
        if (!_resources.TryGetValue(typeof(T), out var typeResources)) return null;
        return typeResources.TryGetValue(resourceID, out var resource) ? (T)resource : null;
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
    //Core Vulkan GPU resources container for the texture
    private Texture _texture;

    public Texture Texture => _texture;
    public TextureResource(string id, Texture texture) : this(id)
    {
        this._texture = texture;
    }
    //texture metadata
    private uint width;
    private uint height;
    ~TextureResource()
    {
        Unload();
    }

    public override bool Load()
    {
        return base.Load();
    }

    public override void Unload()
    {
        _texture.Dispose();
        base.Unload();
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
        // Return VB + IB ranges to the global free-list so a future upload can
        // reuse them. Caller (typically FileBrowserPanel.Destroy) must have
        // drained in-flight GPU work first — ranges go onto the free stack
        // immediately and nothing here checks fence state.
        if (mesh != null)
        {
            manager.FreeMesh(*mesh);
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
    private ResourceManager manager;
    public PbrMaterial Material { get; }
    public ResourceHandle<TextureResource> baseTex { get; }
    public ResourceHandle<TextureResource> metallicRoughnessTex { get; }
    public ResourceHandle<TextureResource> normalTex { get; }
    public ResourceHandle<TextureResource> occlusionTex { get; }
    public ResourceHandle<TextureResource> emissiveTex { get; }

    public MaterialResource(string id, ResourceManager rm, PbrMaterial mat) : this(id)
    {
        this.manager = rm;
        Material = mat;
        baseTex = new ResourceHandle<TextureResource>(manager, id + "_baseTex");
        metallicRoughnessTex = new ResourceHandle<TextureResource>(manager, id + "_metallicRoughnessTex");
        normalTex = new ResourceHandle<TextureResource>(manager, id + "_normalTex");
        occlusionTex = new ResourceHandle<TextureResource>(manager, id + "_occlusionTex");
        emissiveTex = new ResourceHandle<TextureResource>(manager, id + "_emissiveTex");
    }
    public override bool Load()
    {
        return base.Load();
    }

    public Texture[] GetMaterialTextures()
    {
        return [
            baseTex.Get()!.Texture,
            metallicRoughnessTex.Get()!.Texture,
            normalTex.Get()!.Texture,
            occlusionTex.Get()!.Texture,
            emissiveTex.Get()!.Texture
        ];
    }

    ~MaterialResource()
    {
        Unload();
    }

    public override void Unload()
    {
        base.Unload();
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