using System.Numerics;
using System.Runtime.InteropServices;
using Silk.NET.Vulkan;
using Silk.NET.Vulkan.Extensions.KHR;

namespace CadThingo.VulkanEngine.Renderer;

// Per-entity record consumed by the PBR lighting shader's shadow-ray alpha-test
// path. One slot per TLAS instance, keyed by AccelerationStructureInstanceKHR.
// InstanceCustomIndex. Layout matches PbrShader.slang::ShadowEntityInfo.

// One world-space emissive triangle, treated as an area light by the
// pathtracer's NEE + MIS. Layout matches PTUtils.slang::EmissiveTri (80B,
// std430). Positions/normal are world-space (baked from the entity transform at
// TLAS-rebuild time); Le is the material emissive factor. IndexOffset/PrimIndex
// point back into the global index buffer so the shader can refetch the UVs and
// modulate Le by the emissive texture at the sampled point.

// Vose alias-table entry for O(1) power-proportional triangle selection.
// Matches PTCompute.slang::AliasEntry (8B).
[StructLayout(LayoutKind.Sequential)]
public struct AliasEntryGpu
{
    public float Prob;       // probability of keeping bucket i (else jump to Alias)
    public uint  Alias;      // fallback triangle index
}

public unsafe partial class Renderer
{
    
    //  State

    // Silk.NET dispatch table for VK_KHR_acceleration_structure. Loaded once via
    // TryGetDeviceExtension after the logical device is created. Null when the
    // extension wasn't enabled — every method below should early-out on null.
    private KhrAccelerationStructure? khrAccelStruct;
    
    // Pulled from PhysicalDeviceAccelerationStructurePropertiesKHR at startup.
    // Every scratch buffer offset passed to Cmd*BuildAccelerationStructures must be
    // a multiple of this value, otherwise validation errors with "scratchData not
    // properly aligned." Default 1 keeps math safe before LoadRayQueryExtensions runs.
    private uint asScratchAlignment = 1;

    // Cluster BLASes. One world-space BLAS per spatial cluster of primitives,
    // rebuilt wholesale every RebuildTlas (the geometry is baked to world space
    // via per-geometry transforms, so a moved transform invalidates the BLAS).
    // Replaces the old per-mesh BLAS cache: merging co-located primitives into a
    // shared BLAS is what kills the instance-AABB overlap that made the TLAS
    // un-prunable (wall + window in the same volume were separate BLASes for example).
    private readonly List<BlasEntry> clusterBlases = new();

    // Max primitives per cluster. int.MaxValue == one merged BLAS over the whole
    // scene: a single SAH BVH spans all triangles, so instance-AABB overlap goes
    // to zero and the builder partitions overlapping geometry at the triangle
    // level (which is the only thing that actually helps when primitives have
    // real extent — whole-prim clustering just preserves the overlap). Lower this
    // to split into multiple BLASes only if one-BLAS build time on a massive scene
    // becomes a problem; the right split then is SAH-by-AABB, not centroid-median.
    private const int CLUSTER_MAX_PRIMS = int.MaxValue;

    // Per-geometry world transforms consumed by the cluster BLAS builds. One
    // TransformMatrixKHR per primitive, keyed by the same flat geometry slot that
    // keys shadowInfo. Host-visible + device address; the AS build reads it.
    private Buffer    clusterTransformBuffer;
    private SubAlloc  clusterTransformAlloc;
    private void*     clusterTransformMapped;
    private uint      clusterTransformCapacity;   // slots (TransformMatrixKHR), not bytes

    private struct BlasEntry
    {
        public AccelerationStructureKHR Handle;
        public Buffer        Storage;          // usage = AccelerationStructureStorageBitKhr | ShaderDeviceAddressBit
        public SubAlloc      StorageAlloc;
        public ulong         DeviceAddress;    // from GetAccelerationStructureDeviceAddress (NOT GetBufferDeviceAddress)
    }

    // Primitive gathered for clustering: one per renderable entity. World matrix
    // is baked into the BLAS via per-geometry transform; Centroid (world-space
    // mesh sphere center) drives the spatial median split.
    private struct ClusterPrim
    {
        public int        EntityIndex;
        public Matrix4x4  World;
        public uint       IndexOffset;    // mesh->offset (elements into globalIndices)
        public uint       TriCount;       // mesh->count / 3
        public uint       MaterialIndex;
        public uint       Flags;          // PbrMaterial.Flags
        public bool       NonOpaque;      // MASK / BLEND / transmissive → no per-geometry OpaqueBit
        public Vector3    Centroid;
    }

    // Single scene-wide TLAS. Rebuild on entity-set / transform changes; flag
    // tlasDirty so DrawFrame can pick it up at the top of a frame.
    private AccelerationStructureKHR tlas;

    // True when the full ray-query stack is usable for this frame: the device
    // supports it, the extension loaded, and a TLAS has been built. SelectionSystem
    // (pick + outline) gates on this before touching the acceleration structure.
    internal bool RayInfraReady => RayShadowsSupported && khrAccelStruct != null && tlas.Handle != 0;
    private Buffer    tlasStorage;
    private SubAlloc  tlasStorageAlloc;

    // Instance buffer feeds Cmd*BuildAccelerationStructures with the per-instance
    // AccelerationStructureInstanceKHR records. Host-visible + coherent so we can
    // memcpy each frame; usage must include AccelerationStructureBuildInputReadOnlyBitKhr
    // and ShaderDeviceAddressBit (the build reads it via device address).
    private Buffer    tlasInstanceBuffer;
    private SubAlloc  tlasInstanceAlloc;
    private void*     tlasInstanceMapped;
    private uint      tlasInstanceCapacity;     // number of slots allocated, not bytes

    // Persistent scratch buffer reused across builds. Sized to the largest
    // BuildScratchSize seen so far; reallocated if a bigger build comes along.
    private Buffer    asScratchBuffer;
    private SubAlloc  asScratchAlloc;
    private ulong     asScratchSize;

    // Per-entity shadow-alpha info. One ShadowEntityInfo per TLAS instance,
    // indexed by InstanceCustomIndex. Host-visible + coherent so RebuildTlas
    // writes them inline with the instance buffer. Grows alongside the instance
    // buffer; capacity is tracked separately because zero-entity scenes still
    // need a valid binding.
    private Buffer    shadowInfoBuffer;
    private SubAlloc  shadowInfoAlloc;
    private void*     shadowInfoMapped;
    private uint      shadowInfoCapacity;     // number of slots allocated, not bytes

    /// <summary>Buffer holding ShadowEntityInfo records, indexed by InstanceCustomIndex.
    /// PbrDeferredPipeline binds this on its shadow-alpha descriptor set. Returns
    /// a zero handle until InitRayQuery has run.</summary>
    public Buffer ShadowInfoBuffer => shadowInfoBuffer;
    public ulong ShadowInfoBufferSize =>
        (ulong)shadowInfoCapacity * (ulong)sizeof(ShadowEntityInfo);

    /// <summary>Set by RebuildTlas whenever EnsureShadowInfoCapacity reallocates
    /// the underlying VkBuffer — the renderer reads this after RebuildTlas and
    /// re-writes the PBR pipeline's shadow-alpha descriptor when true.</summary>
    private bool shadowInfoBufferResized;

    // Emissive area-light buffers, rebuilt alongside the TLAS (world-space data
    // depends on entity transforms). Host-visible so RebuildTlas writes them
    // inline. Always allocated with capacity >= 1 so the PT descriptor stays
    // valid even in scenes with no emissive geometry (emissiveTriCount == 0
    // makes the shader skip them).
    private Buffer   emissiveTriBuffer;
    private SubAlloc emissiveTriAlloc;
    private void*    emissiveTriMapped;
    private Buffer   emissiveAliasBuffer;
    private SubAlloc emissiveAliasAlloc;
    private void*    emissiveAliasMapped;
    private uint     emissiveCapacity;       // slots allocated (shared by both buffers), not bytes
    private uint     emissiveTriCount;       // live emissive triangles this build
    private float    totalEmissivePower;     // Σ area·luminance(Le) — the alias-table normaliser
    private bool     emissiveBuffersResized; // true when EnsureEmissiveCapacity reallocated

    /// <summary>Emissive-triangle SSBO + its size. PT pipeline binds these on
    /// set 0 (bindings 6/7). Zero handle until the first RebuildTlas.</summary>
    public Buffer EmissiveTriBuffer    => emissiveTriBuffer;
    public ulong  EmissiveTriBufferSize   => (ulong)emissiveCapacity * (ulong)sizeof(EmissiveTriGpu);
    public Buffer EmissiveAliasBuffer  => emissiveAliasBuffer;
    public ulong  EmissiveAliasBufferSize => (ulong)emissiveCapacity * (ulong)sizeof(AliasEntryGpu);
    public uint   EmissiveTriangleCount => emissiveTriCount;
    public float  TotalEmissivePower    => totalEmissivePower;

    private bool tlasDirty = true;

    /// <summary>
    /// Flags the TLAS as stale. Consumed at the top of DrawFrame which runs a
    /// single <see cref="OnSceneEntitiesChanged"/> per frame regardless of how
    /// many edits accumulated the previous frame. Use this from the editor side
    /// (InspectorPanel transforms, FileBrowserPanel visibility) — direct
    /// per-mutation RebuildTlas calls would stall the device on every slider tick.
    /// </summary>
    public void MarkTlasDirty()
    {
        tlasDirty = true;
        // Structural / transform / alpha-mode edits also re-pack the GPU mirror (L2 step 7).
        gpuScene?.MarkSceneDirty();
    }
    public bool IsTlasDirty => tlasDirty;


    //  Helpers — finished

    private ulong GetBufferDeviceAddress(Buffer buffer)
    {
        BufferDeviceAddressInfo deviceAddressInfo = new()
        {
            SType = StructureType.BufferDeviceAddressInfo,
            Buffer = buffer,
        };
        return vk!.GetBufferDeviceAddress(device, &deviceAddressInfo);
    }


    // 
    //  Helpers — TODO
    // 

    /// <summary>
    /// Same shape as the existing CreateBuffer helper but chains
    /// MemoryAllocateFlagsInfo { flags = AddressBitKhr } into MemoryAllocateInfo.PNext.
    /// REQUIRED for any buffer you'll call vkGetBufferDeviceAddress on
    /// (BLAS storage, TLAS storage, scratch, instance). Without it the address
    /// returned is undefined and validation will yell on first use.
    ///
    /// Caller still passes ShaderDeviceAddressBit in `usage` — both the buffer
    /// usage bit AND the alloc flag are needed.
    /// </summary>
    private void CreateBufferWithDeviceAddress(
        ulong size, BufferUsageFlags usage, MemoryPropertyFlags memProps,
        out Buffer buffer, out SubAlloc alloc, float priority = GpuMemoryAllocator.PriorityDefault)
    {
        // Allocator buffer blocks unconditionally carry MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
        // so this is now just a buffer create + bind. Caller still owns the
        // ShaderDeviceAddressBit usage flag — the buffer needs both halves.
        BufferCreateInfo bufferInfo = new()
        {
            SType = StructureType.BufferCreateInfo,
            Size = size,
            Usage = usage,
            SharingMode = SharingMode.Exclusive,
        };
        if (vk!.CreateBuffer(device, &bufferInfo, null, out buffer) != Result.Success)
            throw new Exception("Failed to create device-address buffer");
        alloc = memAllocator.AllocateForBuffer(buffer, memProps, priority);
    }

    /// <summary>
    /// Packs System.Numerics.Matrix4x4 into Vulkan's TransformMatrixKHR (row-major 3×4,
    /// 12 floats; 4th row implicit [0,0,0,1]).
    ///
    /// Convention mismatch: System.Numerics is row-vector (translation in M41/M42/M43);
    /// Vulkan TransformMatrixKHR is column-vector (translation at slots 3/7/11). So we
    /// TRANSPOSE while packing — writing System.Numerics columns as Vulkan rows.
    ///
    /// The geometry shader gets away with no explicit transpose only because HLSL/Slang
    /// defaults to column-major matrix layout in constant buffers, which silently
    /// re-interprets the row-major upload. The AS builder doesn't do that — it reads
    /// the 12 floats verbatim per the spec.
    /// </summary>
    private static TransformMatrixKHR ToTransformMatrixKHR(in System.Numerics.Matrix4x4 m)
    {
        TransformMatrixKHR t = default;
        t.Matrix[0]  = m.M11; t.Matrix[1]  = m.M21; t.Matrix[2]  = m.M31; t.Matrix[3]  = m.M41;
        t.Matrix[4]  = m.M12; t.Matrix[5]  = m.M22; t.Matrix[6]  = m.M32; t.Matrix[7]  = m.M42;
        t.Matrix[8]  = m.M13; t.Matrix[9]  = m.M23; t.Matrix[10] = m.M33; t.Matrix[11] = m.M43;
        return t;
    }


    //
    //  Allocator helpers (used by both BuildBlas and RebuildTlas)
    //

    /// <summary>
    /// Grows the persistent scratch buffer if `required` exceeds current size.
    /// Padded up to asScratchAlignment so any offset into the buffer satisfies
    /// the scratchData alignment rule.
    /// </summary>
    private void EnsureScratchCapacity(ulong required)
    {
        ulong padded = ((required + asScratchAlignment - 1) / asScratchAlignment) * asScratchAlignment;
        if (asScratchBuffer.Handle != 0 && asScratchSize >= padded) return;

        if (asScratchBuffer.Handle != 0) DestroyBuffer(asScratchBuffer, asScratchAlloc);

        CreateBufferWithDeviceAddress(padded,
            BufferUsageFlags.StorageBufferBit | BufferUsageFlags.ShaderDeviceAddressBit,
            MemoryPropertyFlags.DeviceLocalBit,
            out asScratchBuffer, out asScratchAlloc);
        asScratchSize = padded;
    }

    /// <summary>
    /// Grows the persistently-mapped instance buffer to hold at least
    /// `requiredInstances` AccelerationStructureInstanceKHR records. Doubles
    /// capacity (min 8) so frequent small scenes don't re-allocate every frame.
    /// </summary>
    private void EnsureInstanceCapacity(uint requiredInstances)
    {
        if (tlasInstanceCapacity >= requiredInstances) return;

        if (tlasInstanceAlloc.IsValid)
        {
            DestroyBuffer(tlasInstanceBuffer, tlasInstanceAlloc);
            tlasInstanceMapped = null;
        }

        uint capacity = 8;
        while (capacity < requiredInstances) capacity <<= 1;

        ulong sizeBytes = (ulong)capacity * (ulong)sizeof(AccelerationStructureInstanceKHR);
        CreateBufferWithDeviceAddress(sizeBytes,
            BufferUsageFlags.AccelerationStructureBuildInputReadOnlyBitKhr | BufferUsageFlags.ShaderDeviceAddressBit,
            MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit,
            out tlasInstanceBuffer, out tlasInstanceAlloc);

        tlasInstanceMapped = memAllocator.GetMapped(tlasInstanceAlloc);
        tlasInstanceCapacity = capacity;
    }

    /// <summary>
    /// Mirror of EnsureInstanceCapacity for the ShadowEntityInfo SSBO. Returns
    /// true iff the underlying VkBuffer was (re-)allocated — the caller must
    /// re-write the PBR pipeline's shadow-alpha descriptor set in that case.
    /// </summary>
    private bool EnsureShadowInfoCapacity(uint requiredInstances)
    {
        if (shadowInfoCapacity >= requiredInstances && shadowInfoBuffer.Handle != 0)
            return false;

        if (shadowInfoAlloc.IsValid)
        {
            DestroyBuffer(shadowInfoBuffer, shadowInfoAlloc);
            shadowInfoMapped = null;
        }

        uint capacity = 8;
        while (capacity < requiredInstances) capacity <<= 1;

        ulong sizeBytes = (ulong)capacity * (ulong)sizeof(ShadowEntityInfo);
        CreateBuffer(sizeBytes, BufferUsageFlags.StorageBufferBit,
            MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit,
            out shadowInfoBuffer, out shadowInfoAlloc, preferDeviceLocal: true);

        shadowInfoMapped = memAllocator.GetMapped(shadowInfoAlloc);
        shadowInfoCapacity = capacity;
        return true;
    }

    /// <summary>
    /// Grows the emissive-triangle + alias-table buffers to hold at least
    /// <paramref name="requiredTris"/> entries (floored at 1 so the descriptor
    /// is always valid). Both buffers share one capacity. Returns true iff a
    /// reallocation happened — caller must re-write the PT descriptor then.
    /// </summary>
    private bool EnsureEmissiveCapacity(uint requiredTris)
    {
        uint required = Math.Max(1u, requiredTris);
        if (emissiveCapacity >= required && emissiveTriBuffer.Handle != 0)
            return false;

        if (emissiveTriAlloc.IsValid)   { DestroyBuffer(emissiveTriBuffer,   emissiveTriAlloc);   emissiveTriMapped   = null; }
        if (emissiveAliasAlloc.IsValid) { DestroyBuffer(emissiveAliasBuffer, emissiveAliasAlloc); emissiveAliasMapped = null; }

        uint capacity = 8;
        while (capacity < required) capacity <<= 1;

        CreateBuffer((ulong)capacity * (ulong)sizeof(EmissiveTriGpu),
            BufferUsageFlags.StorageBufferBit,
            MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit,
            out emissiveTriBuffer, out emissiveTriAlloc, preferDeviceLocal: true);
        CreateBuffer((ulong)capacity * (ulong)sizeof(AliasEntryGpu),
            BufferUsageFlags.StorageBufferBit,
            MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit,
            out emissiveAliasBuffer, out emissiveAliasAlloc, preferDeviceLocal: true);

        emissiveTriMapped   = memAllocator.GetMapped(emissiveTriAlloc);
        emissiveAliasMapped = memAllocator.GetMapped(emissiveAliasAlloc);
        emissiveCapacity    = capacity;
        return true;
    }

    private static float Luminance(Vector3 c) => 0.2126f * c.X + 0.7152f * c.Y + 0.0722f * c.Z;

    /// <summary>
    /// Appends every emissive triangle of <paramref name="mesh"/> (transformed
    /// to world space by <paramref name="world"/>) to <paramref name="tris"/>,
    /// accumulating per-triangle power (area·luminance(Le)) into
    /// <paramref name="weights"/> and the running total. Degenerate or zero-power
    /// triangles are skipped. No-op if the mesh has no retained CPU geometry.
    /// </summary>
    private void CollectEmissiveTriangles(Mesh* mesh, in System.Numerics.Matrix4x4 world,
        Vector3 emissive, int emissiveTex, List<EmissiveTriGpu> tris, List<float> weights, ref float totalPower)
    {
        if (!Engine.ResourceManager.TryGetMeshGeometry(mesh->offset, out var pos, out var idx))
            return;

        float lum = Luminance(emissive);
        if (lum <= 0.0f) return;

        // Element offset of this mesh's triangle list inside the global index
        // buffer — the shader pairs it with the primitive index to refetch UVs.
        int indexOffset = mesh->offset;

        for (int t = 0; t + 2 < idx.Length; t += 3)
        {
            Vector3 p0 = Vector3.Transform(pos[idx[t]],     world);
            Vector3 p1 = Vector3.Transform(pos[idx[t + 1]], world);
            Vector3 p2 = Vector3.Transform(pos[idx[t + 2]], world);

            Vector3 e1 = p1 - p0;
            Vector3 e2 = p2 - p0;
            Vector3 cr = Vector3.Cross(e1, e2);
            float   len = cr.Length();
            float   area = 0.5f * len;
            if (area < 1e-9f) continue;             // degenerate

            float power = area * lum;
            if (power <= 0.0f) continue;

            Vector3 n = cr / len;
            tris.Add(new EmissiveTriGpu
            {
                P0Area = new Vector4(p0, area),
                E1LeR  = new Vector4(e1, emissive.X),
                E2LeG  = new Vector4(e2, emissive.Y),
                NLeB   = new Vector4(n,  emissive.Z),
                // primIdx comes from the loop counter, NOT tris.Count — skipped
                // degenerate triangles must not shift the index-buffer mapping.
                IndexOffset = indexOffset,
                PrimIndex   = t / 3,
                EmissiveTex = emissiveTex,
                _pad        = 0,
            });
            weights.Add(power);
            totalPower += power;
        }
    }

    /// <summary>
    /// Uploads the collected emissive triangles and builds a Vose alias table
    /// over their power weights, so the shader can pick a triangle ∝ power in
    /// O(1). Handles the empty case (allocates a 1-slot dummy, count 0). Sets
    /// emissiveTriCount / totalEmissivePower / emissiveBuffersResized.
    /// </summary>
    private void BuildEmissiveBuffers(List<EmissiveTriGpu> tris, List<float> weights, float totalPower)
    {
        int n = tris.Count;
        emissiveBuffersResized = EnsureEmissiveCapacity((uint)n);
        emissiveTriCount   = (uint)n;
        totalEmissivePower = totalPower;

        if (n == 0) return;   // dummy buffers already valid; shader skips on count 0

        // Upload triangle records.
        var triDst = (EmissiveTriGpu*)emissiveTriMapped;
        for (int i = 0; i < n; i++) triDst[i] = tris[i];

        // Vose's alias method. scaled[i] = w_i * n / Σw  (mean 1).
        var scaled = new double[n];
        for (int i = 0; i < n; i++) scaled[i] = weights[i] * n / totalPower;

        var small = new Stack<int>();
        var large = new Stack<int>();
        for (int i = 0; i < n; i++) (scaled[i] < 1.0 ? small : large).Push(i);

        var aliasDst = (AliasEntryGpu*)emissiveAliasMapped;
        while (small.Count > 0 && large.Count > 0)
        {
            int l = small.Pop();
            int g = large.Pop();
            aliasDst[l] = new AliasEntryGpu { Prob = (float)scaled[l], Alias = (uint)g };
            scaled[g] = (scaled[g] + scaled[l]) - 1.0;
            (scaled[g] < 1.0 ? small : large).Push(g);
        }
        // Leftovers settle at prob 1 (self-alias). Floating-point drift can leave
        // a few in either stack.
        while (large.Count > 0) { int g = large.Pop(); aliasDst[g] = new AliasEntryGpu { Prob = 1.0f, Alias = (uint)g }; }
        while (small.Count > 0) { int s = small.Pop(); aliasDst[s] = new AliasEntryGpu { Prob = 1.0f, Alias = (uint)s }; }
    }


    //
    //  Building blocks
    //

    // Recursive spatial median split over primitive centroids. Emits leaf ranges
    // [lo,hi) into outClusters, each with <= CLUSTER_MAX_PRIMS primitives.
    // Co-located primitives (similar centroids) land in the same leaf → one BLAS,
    // where SAH packs the overlapping geometry into a tight hierarchy; spatially
    // separated primitives split into different leaves so the TLAS can prune.
    private static void ClusterPrims(List<ClusterPrim> prims, int[] idx, int lo, int hi,
        List<(int lo, int hi)> outClusters)
    {
        int count = hi - lo;
        if (count <= CLUSTER_MAX_PRIMS)
        {
            outClusters.Add((lo, hi));
            return;
        }

        // Centroid bounds over this range → split along the longest axis.
        Vector3 mn = new(float.MaxValue), mx = new(float.MinValue);
        for (int i = lo; i < hi; i++)
        {
            Vector3 c = prims[idx[i]].Centroid;
            mn = Vector3.Min(mn, c);
            mx = Vector3.Max(mx, c);
        }
        Vector3 ext = mx - mn;
        int axis = (ext.X >= ext.Y && ext.X >= ext.Z) ? 0 : (ext.Y >= ext.Z ? 1 : 2);

        Comparison<int> cmp = axis switch
        {
            0 => (a, b) => prims[a].Centroid.X.CompareTo(prims[b].Centroid.X),
            1 => (a, b) => prims[a].Centroid.Y.CompareTo(prims[b].Centroid.Y),
            _ => (a, b) => prims[a].Centroid.Z.CompareTo(prims[b].Centroid.Z),
        };
        Array.Sort(idx, lo, count, Comparer<int>.Create(cmp));

        // Median split by count. Coincident centroids (the wall/window case) still
        // halve, so the recursion always terminates and cluster size stays bounded.
        int mid = lo + count / 2;
        ClusterPrims(prims, idx, lo, mid, outClusters);
        ClusterPrims(prims, idx, mid, hi, outClusters);
    }

    /// <summary>
    /// Grows the per-geometry transform buffer (one TransformMatrixKHR per
    /// primitive). Build input + device address; host-visible so RebuildTlas
    /// writes it inline before the BLAS builds read it.
    /// </summary>
    private void EnsureClusterTransformCapacity(uint requiredSlots)
    {
        uint required = Math.Max(1u, requiredSlots);
        if (clusterTransformCapacity >= required && clusterTransformBuffer.Handle != 0)
            return;

        if (clusterTransformAlloc.IsValid)
        {
            DestroyBuffer(clusterTransformBuffer, clusterTransformAlloc);
            clusterTransformMapped = null;
        }

        uint capacity = 8;
        while (capacity < required) capacity <<= 1;

        CreateBufferWithDeviceAddress((ulong)capacity * (ulong)sizeof(TransformMatrixKHR),
            BufferUsageFlags.AccelerationStructureBuildInputReadOnlyBitKhr | BufferUsageFlags.ShaderDeviceAddressBit,
            MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit,
            out clusterTransformBuffer, out clusterTransformAlloc);

        clusterTransformMapped   = memAllocator.GetMapped(clusterTransformAlloc);
        clusterTransformCapacity = capacity;
    }

    /// <summary>Destroys all cluster BLAS handles + storage. Caller must have
    /// drained the device (RebuildTlas runs under the editor's DeviceWaitIdle, or
    /// at init before any frame).</summary>
    private void DestroyClusterBlases()
    {
        if (khrAccelStruct == null) return;
        foreach (var b in clusterBlases)
        {
            khrAccelStruct.DestroyAccelerationStructure(device, b.Handle, null);
            DestroyBuffer(b.Storage, b.StorageAlloc);
        }
        clusterBlases.Clear();
    }

    // Builds one world-space BLAS for primitives idx[lo..hi). One geometry per
    // primitive — so CommittedGeometryIndex() maps straight to the flat shadowInfo
    // slot (baseSlot + g) — each baked to world space by its per-geometry transform
    // at the same slot in clusterTransformBuffer. Reuses the object-space global
    // VB/IB (no vertex duplication); the builder applies the transform.
    private BlasEntry BuildClusterBlas(List<ClusterPrim> prims, int[] idx, int lo, int hi, uint baseSlot)
    {
        int geomCount = hi - lo;

        ulong vbAddr    = GetBufferDeviceAddress(Engine.ResourceManager.GlobalVertexBuffer);
        ulong ibBase    = GetBufferDeviceAddress(Engine.ResourceManager.GlobalIndexBuffer);
        ulong xfBase    = GetBufferDeviceAddress(clusterTransformBuffer);
        uint  maxVertex = (uint)Engine.ResourceManager.VertexHighWater;
        uint  xfStride  = (uint)sizeof(TransformMatrixKHR);   // 48 — a multiple of 16 (transformOffset rule)

        var geos     = new AccelerationStructureGeometryKHR[geomCount];
        var ranges   = new AccelerationStructureBuildRangeInfoKHR[geomCount];
        var maxPrims = new uint[geomCount];

        for (int g = 0; g < geomCount; g++)
        {
            ClusterPrim p = prims[idx[lo + g]];

            var geo = new AccelerationStructureGeometryKHR
            {
                SType        = StructureType.AccelerationStructureGeometryKhr,
                GeometryType = GeometryTypeKHR.TrianglesKhr,
                // Per-geometry opacity: an opaque prim keeps the fast traversal
                // path even inside a BLAS that also holds an alpha-tested / glass
                // prim (replaces the old blanket instance-level ForceNoOpaque).
                Flags        = p.NonOpaque ? 0 : GeometryFlagsKHR.OpaqueBitKhr,
            };
            geo.Geometry.Triangles.SType        = StructureType.AccelerationStructureGeometryTrianglesDataKhr;
            geo.Geometry.Triangles.VertexFormat = Format.R32G32B32Sfloat;
            geo.Geometry.Triangles.VertexStride = (ulong)sizeof(Vertex);
            geo.Geometry.Triangles.VertexData.DeviceAddress = vbAddr;
            geo.Geometry.Triangles.MaxVertex    = maxVertex;
            geo.Geometry.Triangles.IndexType    = IndexType.Uint32;
            geo.Geometry.Triangles.IndexData.DeviceAddress = ibBase + (ulong)(4 * p.IndexOffset);
            // Per-geometry world transform: the build bakes object-space verts to
            // world space, so the BLAS needs no instance transform (identity).
            geo.Geometry.Triangles.TransformData.DeviceAddress = xfBase;
            geos[g] = geo;

            ranges[g] = new AccelerationStructureBuildRangeInfoKHR
            {
                PrimitiveCount  = p.TriCount,
                PrimitiveOffset = 0,
                FirstVertex     = 0,
                TransformOffset = (baseSlot + (uint)g) * xfStride,
            };
            maxPrims[g] = p.TriCount;
        }

        fixed (AccelerationStructureGeometryKHR* pGeos = geos)
        fixed (uint* pMaxPrims = maxPrims)
        {
            var buildInfo = new AccelerationStructureBuildGeometryInfoKHR
            {
                SType         = StructureType.AccelerationStructureBuildGeometryInfoKhr,
                Type          = AccelerationStructureTypeKHR.BottomLevelKhr,
                Mode          = BuildAccelerationStructureModeKHR.BuildKhr,
                Flags         = BuildAccelerationStructureFlagsKHR.PreferFastTraceBitKhr
                              | BuildAccelerationStructureFlagsKHR.AllowCompactionBitKhr,
                GeometryCount = (uint)geomCount,
                PGeometries   = pGeos,
            };

            var sizes = new AccelerationStructureBuildSizesInfoKHR { SType = StructureType.AccelerationStructureBuildSizesInfoKhr };
            khrAccelStruct!.GetAccelerationStructureBuildSizes(
                device, AccelerationStructureBuildTypeKHR.DeviceKhr,
                &buildInfo, pMaxPrims, &sizes);

            // High priority: BLAS storage is the persistent geometry BVH the path tracer
            // traverses every frame — must stay resident under WDDM budget pressure.
            CreateBufferWithDeviceAddress(sizes.AccelerationStructureSize,
                BufferUsageFlags.AccelerationStructureStorageBitKhr | BufferUsageFlags.ShaderDeviceAddressBit,
                MemoryPropertyFlags.DeviceLocalBit, out var storage, out var storageAlloc,
                GpuMemoryAllocator.PriorityHigh);

            EnsureScratchCapacity(sizes.BuildScratchSize);

            var createInfo = new AccelerationStructureCreateInfoKHR
            {
                SType  = StructureType.AccelerationStructureCreateInfoKhr,
                Buffer = storage,
                Size   = sizes.AccelerationStructureSize,
                Type   = AccelerationStructureTypeKHR.BottomLevelKhr,
            };
            khrAccelStruct.CreateAccelerationStructure(device, &createInfo, null, out var handle);
            buildInfo.DstAccelerationStructure  = handle;
            buildInfo.ScratchData.DeviceAddress = GetBufferDeviceAddress(asScratchBuffer);

            fixed (AccelerationStructureBuildRangeInfoKHR* pRanges = ranges)
            {
                var pr  = pRanges;
                var cmd = BeginSingleTimeCommands();
                khrAccelStruct.CmdBuildAccelerationStructures(cmd, 1, &buildInfo, &pr);
                EndSingleTimeCommands(cmd);
            }

            var addrInfo = new AccelerationStructureDeviceAddressInfoKHR
            {
                SType = StructureType.AccelerationStructureDeviceAddressInfoKhr,
                AccelerationStructure = handle,
            };
            ulong devAddr = khrAccelStruct.GetAccelerationStructureDeviceAddress(device, &addrInfo);

            var built = new BlasEntry { Handle = handle, Storage = storage, StorageAlloc = storageAlloc, DeviceAddress = devAddr };
            return CompactBlas(built, sizes.AccelerationStructureSize);
        }
    }

    // Compacts a freshly-built BLAS (must have been built with AllowCompaction):
    // queries the post-build compacted size, copies into a right-sized AS, and frees
    // the original. Compaction typically ~halves the BVH footprint, so more of it
    // stays L2-resident -> shorter BVH-node fetches, which dominate RT-core traversal
    // stalls. Returns the source unchanged if the driver reports no useful shrink.
    // Uses discrete single-time submits (one extra size query + copy per BLAS) -
    // fine at edit/load time; batching belongs in the planned static/dynamic AS wrapper.
    private BlasEntry CompactBlas(BlasEntry src, ulong originalSize)
    {
        var qpInfo = new QueryPoolCreateInfo
        {
            SType      = StructureType.QueryPoolCreateInfo,
            QueryType  = QueryType.AccelerationStructureCompactedSizeKhr,
            QueryCount = 1,
        };
        vk!.CreateQueryPool(device, &qpInfo, null, out var pool);

        var srcHandle = src.Handle;
        var qcmd = BeginSingleTimeCommands();
        vk.CmdResetQueryPool(qcmd, pool, 0, 1);
        khrAccelStruct!.CmdWriteAccelerationStructuresProperties(qcmd, 1, &srcHandle,
            QueryType.AccelerationStructureCompactedSizeKhr, pool, 0);
        EndSingleTimeCommands(qcmd);

        ulong compactedSize = 0;
        vk.GetQueryPoolResults(device, pool, 0, 1, (nuint)sizeof(ulong), &compactedSize,
            (ulong)sizeof(ulong), QueryResultFlags.Result64Bit | QueryResultFlags.ResultWaitBit);
        vk.DestroyQueryPool(device, pool, null);

        // No useful shrink (or a driver reporting 0) — keep the original.
        if (compactedSize == 0 || compactedSize >= originalSize)
            return src;

        CreateBufferWithDeviceAddress(compactedSize,
            BufferUsageFlags.AccelerationStructureStorageBitKhr | BufferUsageFlags.ShaderDeviceAddressBit,
            MemoryPropertyFlags.DeviceLocalBit, out var cStorage, out var cAlloc,
            GpuMemoryAllocator.PriorityHigh);

        var createInfo = new AccelerationStructureCreateInfoKHR
        {
            SType  = StructureType.AccelerationStructureCreateInfoKhr,
            Buffer = cStorage,
            Size   = compactedSize,
            Type   = AccelerationStructureTypeKHR.BottomLevelKhr,
        };
        khrAccelStruct.CreateAccelerationStructure(device, &createInfo, null, out var cHandle);

        var copyInfo = new CopyAccelerationStructureInfoKHR
        {
            SType = StructureType.CopyAccelerationStructureInfoKhr,
            Src   = srcHandle,
            Dst   = cHandle,
            Mode  = CopyAccelerationStructureModeKHR.CompactKhr,
        };
        var ccmd = BeginSingleTimeCommands();
        khrAccelStruct.CmdCopyAccelerationStructure(ccmd, &copyInfo);
        EndSingleTimeCommands(ccmd);

        // Free the uncompacted source now that the compacted copy owns the geometry.
        khrAccelStruct.DestroyAccelerationStructure(device, src.Handle, null);
        DestroyBuffer(src.Storage, src.StorageAlloc);

        var addrInfo = new AccelerationStructureDeviceAddressInfoKHR
        {
            SType                 = StructureType.AccelerationStructureDeviceAddressInfoKhr,
            AccelerationStructure = cHandle,
        };
        ulong cAddr = khrAccelStruct.GetAccelerationStructureDeviceAddress(device, &addrInfo);

        return new BlasEntry { Handle = cHandle, Storage = cStorage, StorageAlloc = cAlloc, DeviceAddress = cAddr };
    }
    //tlas previous entry count(ensures that if all are removed old tlas isnt used)
    private static uint PreviousCount = 0;
    private void RebuildTlas()
    {
        // Reconcile renderable identity (L2 step 2) on the same cadence as the AS:
        // every renderable gathered below gets a stable RenderableHandle (freed when
        // its entity leaves the scene), written into ShadowEntityInfo.EntityIndex
        // below and resolved back by pick / selection (L2 step 6).
        gpuScene.SyncRenderables(scene);

        // Open a fresh world-transform cache window for this rebuild (L2 step 5).
        // RebuildTlas runs out-of-band (edit-driven, before the per-frame reset), so
        // it owns its own BeginTransforms — otherwise the gather below would read the
        // previous cycle's stale matrices.
        gpuScene.BeginTransforms();

        // World-space cluster BLASes depend on the current transforms, so free
        // last build's set before regathering.
        DestroyClusterBlases();

        // 1. Gather renderable primitives (one per active entity with a mesh) and,
        //    in the same walk, collect emissive area-light triangles (we already
        //    have each entity's world transform + material here).
        var prims     = new List<ClusterPrim>();
        var emTris    = new List<EmissiveTriGpu>();
        var emWeights = new List<float>();
        float emPower = 0f;

        for (int i = 0; i < scene.EntityCount; i++)
        {
            Entity* e = scene.GetEntity(i);
            if (e == null) continue;
            if (!e->IsActive) continue;
            var transform = e->GetComponent<TransformComponent>();
            var meshComp  = e->GetComponent<MeshComponent>();
            if (transform == null || meshComp == null || meshComp.mesh == null) continue;

            Matrix4x4 world = gpuScene.WorldOf(e); // cached (L2 step 5)

            uint  matFlags        = 0u;
            float matTransmission = 0f;
            int   matIdx          = meshComp.materialIndex;
            if (matIdx >= 0)
            {
                var mat = scene.GetMaterial(matIdx);
                matFlags        = mat.Flags;
                matTransmission = mat.TransmissionFactor;

                // Emissive triangles as area lights, baked to world space.
                if (mat.EmissiveFactor != Vector3.Zero)
                    CollectEmissiveTriangles(meshComp.mesh, world,
                                             mat.EmissiveFactor, mat.EmissiveTex, emTris, emWeights, ref emPower);
            }

            // World-space centroid from the mesh bounding-sphere center drives the
            // spatial split; co-located prims (wall/window) share a cluster.
            Vector4 sphere   = meshComp.mesh->sphereLocal;
            Vector3 centroid = Vector3.Transform(new Vector3(sphere.X, sphere.Y, sphere.Z), world);

            prims.Add(new ClusterPrim
            {
                // Stable RenderableHandle slot — not the flat list index `i` — so a
                // later reorder/removal can't alias identity (L2 step 6). Register is
                // idempotent: SyncRenderables already allocated this entity's slot at
                // the top of RebuildTlas, so this just reads it back.
                EntityIndex   = (int)gpuScene.Register(e).Index,
                World         = world,
                IndexOffset   = (uint)meshComp.mesh->offset,
                TriCount      = (uint)(meshComp.mesh->count / 3),
                MaterialIndex = matIdx >= 0 ? (uint)matIdx : 0u,
                Flags         = matFlags,
                // MASK (0x1) / BLEND (0x4) / transmission > 0 → non-opaque, so the
                // Proceed loop gets a chance to alpha- / fresnel-test the candidate.
                NonOpaque     = (matFlags & 5u) != 0u || matTransmission > 0f,
                Centroid      = centroid,
            });
        }

        // Build the emissive area-light buffers + alias table. Before the empty
        // early-out so the PT descriptor always points at a valid buffer.
        BuildEmissiveBuffers(emTris, emWeights, emPower);

        int n = prims.Count;

        // 2. Spatial median-split clustering — scene-graph-agnostic, so it holds
        //    for flat scenes and deep hierarchies alike.
        var clusters = new List<(int lo, int hi)>();
        int[] idx = new int[n];
        for (int k = 0; k < n; k++) idx[k] = k;
        if (n > 0) ClusterPrims(prims, idx, 0, n, clusters);

        // 3. Capacities: one TLAS instance per cluster; one shadowInfo + transform
        //    slot per primitive, laid out flat and cluster-contiguous.
        EnsureInstanceCapacity((uint)Math.Max(1, clusters.Count));
        shadowInfoBufferResized = EnsureShadowInfoCapacity((uint)Math.Max(1, n));
        EnsureClusterTransformCapacity((uint)Math.Max(1, n));

        var dst  = (AccelerationStructureInstanceKHR*)tlasInstanceMapped;
        var sDst = (ShadowEntityInfo*)shadowInfoMapped;
        var xDst = (TransformMatrixKHR*)clusterTransformMapped;

        // 4. Per cluster: write its prims' transforms + shadowInfo into a
        //    contiguous flat block, build the cluster BLAS over them, and emit one
        //    identity instance whose InstanceCustomIndex is the block base. The
        //    shaders resolve a hit via entityInfo[InstanceCustomIndex + GeometryIndex].
        uint instCount = 0;
        uint gslot     = 0;
        foreach (var (lo, hi) in clusters)
        {
            uint baseSlot = gslot;
            for (int k = lo; k < hi; k++)
            {
                ClusterPrim p = prims[idx[k]];
                xDst[gslot] = ToTransformMatrixKHR(p.World);
                // Column-vector 3x4 rows of the world matrix (same transpose
                // ToTransformMatrixKHR does) so the shader can rotate object-space
                // normals into world space — the identity instance transform can't.
                Matrix4x4 w = p.World;
                sDst[gslot] = new ShadowEntityInfo
                {
                    IndexOffset   = p.IndexOffset,
                    MaterialIndex = p.MaterialIndex,
                    Flags         = p.Flags,
                    EntityIndex   = (uint)p.EntityIndex,
                    Xform0        = new Vector4(w.M11, w.M21, w.M31, w.M41),
                    Xform1        = new Vector4(w.M12, w.M22, w.M32, w.M42),
                    Xform2        = new Vector4(w.M13, w.M23, w.M33, w.M43),
                };
                gslot++;
            }

            BlasEntry blas = BuildClusterBlas(prims, idx, lo, hi, baseSlot);
            clusterBlases.Add(blas);

            dst[instCount++] = new AccelerationStructureInstanceKHR
            {
                // Identity: the cluster BLAS is already world-space.
                Transform                              = ToTransformMatrixKHR(Matrix4x4.Identity),
                InstanceCustomIndex                    = baseSlot,   // 24-bit flat geometry-block base
                Mask                                   = 0xFF,
                InstanceShaderBindingTableRecordOffset = 0,
                // No force-opaque flags — per-geometry OpaqueBit governs, so opaque
                // prims keep the fast path inside a mixed cluster.
                Flags                                  = GeometryInstanceFlagsKHR.TriangleFacingCullDisableBitKhr,
                AccelerationStructureReference         = blas.DeviceAddress,
            };
        }

        if (instCount == 0 && PreviousCount == 0)
        {
            tlasDirty = false;
            return;
        }
        PreviousCount = instCount;


        // 3. Geometry — instance data lives at tlasInstanceBuffer's device address.
        uint instanceCount = instCount;
        var geo = new AccelerationStructureGeometryKHR
        {
            SType        = StructureType.AccelerationStructureGeometryKhr,
            GeometryType = GeometryTypeKHR.InstancesKhr,
            Flags        = GeometryFlagsKHR.OpaqueBitKhr,
        };
        geo.Geometry.Instances.SType              = StructureType.AccelerationStructureGeometryInstancesDataKhr;
        geo.Geometry.Instances.ArrayOfPointers    = false;
        geo.Geometry.Instances.Data.DeviceAddress = GetBufferDeviceAddress(tlasInstanceBuffer);

        // 4. Build info — full rebuild for now. AllowUpdateBitKhr is set so a future
        //    transform-only path can use Mode = UpdateKhr + SrcAccelerationStructure = tlas.
        var buildInfo = new AccelerationStructureBuildGeometryInfoKHR
        {
            SType         = StructureType.AccelerationStructureBuildGeometryInfoKhr,
            Type          = AccelerationStructureTypeKHR.TopLevelKhr,
            Mode          = BuildAccelerationStructureModeKHR.BuildKhr,
            Flags         = BuildAccelerationStructureFlagsKHR.PreferFastTraceBitKhr
                          | BuildAccelerationStructureFlagsKHR.AllowUpdateBitKhr,
            GeometryCount = 1,
            PGeometries   = &geo,
        };

        // 5. Size query. For TLAS, the "primitive count" is the instance count.
        var sizes = new AccelerationStructureBuildSizesInfoKHR
        {
            SType = StructureType.AccelerationStructureBuildSizesInfoKhr,
        };
        khrAccelStruct!.GetAccelerationStructureBuildSizes(
            device, AccelerationStructureBuildTypeKHR.DeviceKhr,
            &buildInfo, &instanceCount, &sizes);

        // 6. (Re)allocate TLAS storage. Free + reallocate on every rebuild until
        //    the update-mode path lands.
        if (tlas.Handle != 0)
        {
            khrAccelStruct.DestroyAccelerationStructure(device, tlas, null);
            DestroyBuffer(tlasStorage, tlasStorageAlloc);
        }
        // High priority: TLAS storage is the top-level BVH traversed every path-trace frame.
        CreateBufferWithDeviceAddress(sizes.AccelerationStructureSize,
            BufferUsageFlags.AccelerationStructureStorageBitKhr | BufferUsageFlags.ShaderDeviceAddressBit,
            MemoryPropertyFlags.DeviceLocalBit,
            out tlasStorage, out tlasStorageAlloc,
            GpuMemoryAllocator.PriorityHigh);

        var createInfo = new AccelerationStructureCreateInfoKHR
        {
            SType  = StructureType.AccelerationStructureCreateInfoKhr,
            Buffer = tlasStorage,
            Size   = sizes.AccelerationStructureSize,
            Type   = AccelerationStructureTypeKHR.TopLevelKhr,
        };
        khrAccelStruct.CreateAccelerationStructure(device, &createInfo, null, out tlas);

        // 7. Wire scratch + dst into buildInfo, record + submit.
        EnsureScratchCapacity(sizes.BuildScratchSize);
        buildInfo.DstAccelerationStructure  = tlas;
        buildInfo.ScratchData.DeviceAddress = GetBufferDeviceAddress(asScratchBuffer);

        var range = new AccelerationStructureBuildRangeInfoKHR
        {
            PrimitiveCount  = instanceCount,
            PrimitiveOffset = 0,
            FirstVertex     = 0,
            TransformOffset = 0,
        };
        var pRange = &range;

        var cmd = BeginSingleTimeCommands();
        khrAccelStruct.CmdBuildAccelerationStructures(cmd, 1, &buildInfo, &pRange);
        EndSingleTimeCommands(cmd);

        tlasDirty = false;
    }


    //  Orchestrators

    private void InitRayQuery()
    {
        if (!RayShadowsSupported) return;

        if (!vk!.TryGetDeviceExtension(instance, device, out khrAccelStruct))
        {
            Console.Error.WriteLine("[RayQuery] KhrAccelerationStructure dispatch table failed to load");
            khrAccelStruct = null;
            return;
        }

        // Pull MinAccelerationStructureScratchOffsetAlignment via the properties2 chain.
        var asProps = new PhysicalDeviceAccelerationStructurePropertiesKHR
        {
            SType = StructureType.PhysicalDeviceAccelerationStructurePropertiesKhr,
        };
        var props2 = new PhysicalDeviceProperties2
        {
            SType = StructureType.PhysicalDeviceProperties2,
            PNext = &asProps,
        };
        vk!.GetPhysicalDeviceProperties2(physicalDevice, &props2);
        asScratchAlignment = Math.Max(1, asProps.MinAccelerationStructureScratchOffsetAlignment);

        // RebuildTlas now clusters primitives and builds the world-space cluster
        // BLASes itself — no per-mesh BLAS prepass.
        RebuildTlas();
    }

    /// <summary>
    /// Pairs with file destroy in the editor. Cluster BLASes are world-space and
    /// rebuilt wholesale on the next RebuildTlas, so there is nothing mesh-keyed
    /// to free here — just mark the AS stale so the freed geometry is dropped on
    /// the next rebuild (which the editor triggers via OnSceneEntitiesChanged).
    /// </summary>
    public void DestroyBlasFor(IEnumerable<nint> meshPtrs)
    {
        _ = meshPtrs;
        MarkTlasDirty();
    }

    /// <summary>
    /// Rebuilds BLAS for any newly-seen meshes and re-runs RebuildTlas. Re-writes
    /// the TLAS and (if its underlying VkBuffer reallocated) the ShadowEntityInfo
    /// descriptor on every consumer pipeline. Safe to call after a scene-edit
    /// from the editor; performs a DeviceWaitIdle internally so it doesn't race
    /// in-flight command buffers.
    /// </summary>
    public void OnSceneEntitiesChanged()
    {
        if (!initialized) return;
        // Pathtracer always cares because changing the scene invalidates the
        // accumulator regardless of whether ray shadows are supported.
        MarkAccumulatorDirty();

        if (!RayShadowsSupported || khrAccelStruct == null) return;

        vk!.DeviceWaitIdle(device);

        // RebuildTlas reclusters from scratch and rebuilds the world-space cluster
        // BLASes, so newly-joined meshes are picked up automatically.
        RebuildTlas();

        // TLAS handle changes on every rebuild — re-bind it on every consumer
        // descriptor set even when the shadow-info buffer didn't grow.
        if (tlas.Handle != 0)
        {
            ptComputePipeline?.WriteTlasDescriptor(tlas);
            wavefrontPipeline?.WriteTlasDescriptor(tlas);
            rtPipeline?.WriteTlasDescriptor(tlas);
            reStirPipeline?.WriteTlasDescriptor(tlas);
            selection?.WriteTlasDescriptor(tlas);
            // Scene set: same handle change, one call (replaces the fan-out above once
            // consumers migrate to SceneBindings).
            descriptorRegistry.RegisterTlas("sceneTlas", tlas);
        }

        if (shadowInfoBufferResized)
        {
            ptComputePipeline?.WriteShadowInfoDescriptor();
            wavefrontPipeline?.WriteShadowInfoDescriptor();
            rtPipeline?.WriteShadowInfoDescriptor();
            reStirPipeline?.WriteShadowInfoDescriptor();
            selection?.WriteEntityInfoDescriptor();
            descriptorRegistry.RegisterBuffer("sceneEntityInfo", shadowInfoBuffer);
            shadowInfoBufferResized = false;
        }

        // Emissive buffers reallocate only when the triangle count outgrows
        // capacity; content otherwise updates in place (host-coherent), so a
        // re-write is only needed on resize.
        if (emissiveBuffersResized)
        {
            ptComputePipeline?.WriteEmissiveDescriptors();
            wavefrontPipeline?.WriteEmissiveDescriptors();
            rtPipeline?.WriteEmissiveDescriptors();
            reStirPipeline?.WriteEmissiveDescriptors();
            descriptorRegistry.RegisterBuffer("sceneEmissiveTris", emissiveTriBuffer);
            descriptorRegistry.RegisterBuffer("sceneEmissiveAlias", emissiveAliasBuffer);
            emissiveBuffersResized = false;
        }
    }

    private void CleanupRayQuery()
    {
        if (khrAccelStruct == null) return;

        // Host-visible mappings live for the lifetime of the parent block (the
        // allocator owns the map/unmap). Just null the pointers — Free below
        // releases the suballocation; the block stays mapped until allocator dispose.
        tlasInstanceMapped     = null;
        shadowInfoMapped       = null;
        emissiveTriMapped      = null;
        emissiveAliasMapped    = null;
        clusterTransformMapped = null;

        if (shadowInfoBuffer.Handle    != 0) DestroyBuffer(shadowInfoBuffer,    shadowInfoAlloc);
        if (emissiveTriBuffer.Handle   != 0) DestroyBuffer(emissiveTriBuffer,   emissiveTriAlloc);
        if (emissiveAliasBuffer.Handle != 0) DestroyBuffer(emissiveAliasBuffer, emissiveAliasAlloc);
        if (clusterTransformBuffer.Handle != 0) DestroyBuffer(clusterTransformBuffer, clusterTransformAlloc);

        if (tlas.Handle != 0)
        {
            khrAccelStruct.DestroyAccelerationStructure(device, tlas, null);
            tlas = default;
        }
        if (tlasStorage.Handle != 0)        DestroyBuffer(tlasStorage,        tlasStorageAlloc);
        if (tlasInstanceBuffer.Handle != 0) DestroyBuffer(tlasInstanceBuffer, tlasInstanceAlloc);
        if (asScratchBuffer.Handle != 0)    DestroyBuffer(asScratchBuffer,    asScratchAlloc);

        DestroyClusterBlases();

        khrAccelStruct.Dispose();
        khrAccelStruct = null;
    }
}