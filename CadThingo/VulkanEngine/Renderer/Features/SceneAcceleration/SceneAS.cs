using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using CadThingo.VulkanEngine.Renderer.FeatureLifecycle;
using CadThingo.VulkanEngine.Renderer.FeatureLifecycle.RenderFeatureInterfaces;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Features.SceneAcceleration;

// Vose alias-table entry for O(1) power-proportional triangle selection.
// Matches PTCompute.slang::AliasEntry (8B).
[StructLayout(LayoutKind.Sequential)]
public struct AliasEntryGpu
{
    public float Prob;       // probability of keeping bucket i (else jump to Alias)
    public uint  Alias;      // fallback triangle index
}

/// <summary>
/// The scene's ray-tracing acceleration structure: cluster BLASes, the scene TLAS, and the
/// side tables a hit resolves through (ShadowEntityInfo, the emissive-triangle list and its alias
/// table). This is the AS <i>policy</i> half - what to cluster, when to rebuild, how the instance
/// and shadow-info records are packed. The verbs it drives (create / build / compact / destroy an
/// AS) belong to the device and live behind <c>gfx.As</c>; nothing in this file issues a Vulkan AS
/// call directly.
///
/// Gated: on a device without ray query + acceleration structure it is never constructed, so none
/// of these buffers are allocated. Consumers cope with absence by gating the same way - the host's
/// <c>RayInfraReady</c> bridge reports false, and the scene set simply has no <c>sceneTlas</c>.
///
/// Rebuilds are a <see cref="IBakeFeature"/>: editor mutations set a dirty flag and the bake pump
/// services at most one rebuild per frame, at a fixed point before any command buffer is recorded,
/// however many slider ticks landed in between.
/// </summary>
internal sealed unsafe class SceneAS
    : IBakeFeature, ISelfRegisteringFeature<SceneAS>, INeedsGpu, INeedsHost
{
    // Order 3: after IBL / probes, before every core. Nothing here depends on those, but the TLAS
    // and its side tables have to be registered into the scene set before the boot-time registry
    // cross-check runs, and building ahead of the cores keeps that true without a second pass.
    public static FeatureDesc Desc => new(
        Order: 3,
        // Exactly the gate the old InitRayQuery applied as an early-out, hoisted to construction:
        // if we can never build anything, do not allocate the buffers to build it into.
        Gate: gpu => gpu.Gfx.RayShadowsSupported && gpu.Gfx.As.Available,
        Make: () => new SceneAS());

    [ModuleInitializer]
    internal static void _Reg() => FeatureCatalog.Register<SceneAS>();

    public string Name => "Scene AS (BLAS/TLAS)";

    private GpuContext _gpu;
    private Renderer   _host = null!;
    GpuContext INeedsGpu.Gpu   { set => _gpu  = value; }
    Renderer   INeedsHost.Host { set => _host = value; }

    private GraphicsDevice Gfx => _gpu.Gfx;
    private AsDevice       As  => _gpu.Gfx.As;
    private GpuScene       GpuScene => _host.gpuScene;
    private Scene          Scene    => _host.Scene;

    //  State

    // Cluster BLASes. One world-space BLAS per spatial cluster of primitives,
    // rebuilt wholesale every Rebuild (the geometry is baked to world space
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

    // Single scene-wide TLAS. Rebuilt on entity-set / transform changes.
    private AccelerationStructureKHR tlas;
    private Buffer    tlasStorage;
    private SubAlloc  tlasStorageAlloc;

    /// <summary>True when the full ray-query stack is usable for this frame: the feature exists
    /// (so the device supports it) and a TLAS has been built. Pick + outline gate on this before
    /// touching the acceleration structure.</summary>
    public bool Ready => tlas.Handle != 0;

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
    // State plus a growth policy, which is why it is here rather than on the device facet.
    private Buffer    asScratchBuffer;
    private SubAlloc  asScratchAlloc;
    private ulong     asScratchSize;

    // Per-entity shadow-alpha info. One ShadowEntityInfo per TLAS instance,
    // indexed by InstanceCustomIndex. Host-visible + coherent so the rebuild
    // writes them inline with the instance buffer. Grows alongside the instance
    // buffer; capacity is tracked separately because zero-entity scenes still
    // need a valid binding.
    private Buffer    shadowInfoBuffer;
    private SubAlloc  shadowInfoAlloc;
    private void*     shadowInfoMapped;
    private uint      shadowInfoCapacity;     // number of slots allocated, not bytes

    // Set when EnsureShadowInfoCapacity reallocates the underlying VkBuffer - the re-register
    // below consumes it. Sticky until consumed: a later no-resize rebuild must not erase a
    // pending notice.
    private bool shadowInfoBufferResized;

    // Emissive area-light buffers, rebuilt alongside the TLAS (world-space data
    // depends on entity transforms). Host-visible so the rebuild writes them
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

    /// <summary>Emissive-triangle scalars the path tracers need as uniforms. The buffers
    /// themselves are bindable and reach consumers through the scene set, so only these two
    /// non-bindable numbers are exposed.</summary>
    public uint  EmissiveTriangleCount => emissiveTriCount;
    public float TotalEmissivePower    => totalEmissivePower;

    // Rebuild request. Starts clean because Initialize builds once itself.
    private bool _dirty;

    /// <summary>
    /// Flags the AS as stale. Consumed by the bake pump at the top of DrawFrame, which runs a
    /// single rebuild per frame regardless of how many edits accumulated the previous frame. Use
    /// this from the editor side (InspectorPanel transforms, FileBrowserPanel visibility) - direct
    /// per-mutation rebuilds would stall the device on every slider tick.
    /// </summary>
    public void MarkDirty()
    {
        _dirty = true;
        // Structural / transform / alpha-mode edits also re-pack the GPU mirror.
        GpuScene?.MarkSceneDirty();
    }

    // ---- Bake phase --------------------------------------------------------

    public bool BakePending => _dirty;

    /// <summary>Services one pending rebuild. The pump calls this before the frame's command
    /// buffer is opened, so the DeviceWaitIdle inside <see cref="Rebuild"/> never straddles a
    /// half-recorded frame.</summary>
    public void Bake()
    {
        Rebuild();
        // Cleared unconditionally, not just on a successful build: on hardware where the rebuild
        // early-outs there would otherwise be nothing to stop the next frame re-entering.
        _dirty = false;
    }

    public void Initialize()
    {
        // Build once up front so the scene set holds a valid sceneTlas from boot. An empty scene
        // still produces a real zero-instance TLAS - an RT core has to be selectable before any
        // geometry is loaded, and rays simply miss.
        Rebuild();
    }


    //  Helpers

    // The AS buffers below (scratch / instance / cluster transforms / AS storage) are created
    // through the ordinary CreateBuffer with ShaderDeviceAddressBit in their usage - that bit is
    // the whole contract, since every allocator buffer block already carries the matching
    // MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT.

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
    //  Allocator helpers (used by both BuildClusterBlas and Rebuild)
    //

    /// <summary>
    /// Grows the persistent scratch buffer if `required` exceeds current size.
    /// Padded up to the device's scratch alignment so any offset into the buffer
    /// satisfies the scratchData alignment rule.
    /// </summary>
    private void EnsureScratchCapacity(ulong required)
    {
        uint align = As.ScratchAlignment;
        ulong padded = ((required + align - 1) / align) * align;
        if (asScratchBuffer.Handle != 0 && asScratchSize >= padded) return;

        if (asScratchBuffer.Handle != 0) Gfx.DestroyBuffer(asScratchBuffer, asScratchAlloc);

        Gfx.CreateBuffer(padded,
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
            Gfx.DestroyBuffer(tlasInstanceBuffer, tlasInstanceAlloc);
            tlasInstanceMapped = null;
        }

        uint capacity = 8;
        while (capacity < requiredInstances) capacity <<= 1;

        ulong sizeBytes = (ulong)capacity * (ulong)sizeof(AccelerationStructureInstanceKHR);
        Gfx.CreateBuffer(sizeBytes,
            BufferUsageFlags.AccelerationStructureBuildInputReadOnlyBitKhr | BufferUsageFlags.ShaderDeviceAddressBit,
            MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit,
            out tlasInstanceBuffer, out tlasInstanceAlloc);

        tlasInstanceMapped = Gfx.Allocator.GetMapped(tlasInstanceAlloc);
        tlasInstanceCapacity = capacity;
    }

    /// <summary>
    /// Mirror of EnsureInstanceCapacity for the ShadowEntityInfo SSBO. Returns
    /// true iff the underlying VkBuffer was (re-)allocated — the caller must
    /// re-register the scene-set binding in that case.
    /// </summary>
    private bool EnsureShadowInfoCapacity(uint requiredInstances)
    {
        if (shadowInfoCapacity >= requiredInstances && shadowInfoBuffer.Handle != 0)
            return false;

        if (shadowInfoAlloc.IsValid)
        {
            Gfx.DestroyBuffer(shadowInfoBuffer, shadowInfoAlloc);
            shadowInfoMapped = null;
        }

        uint capacity = 8;
        while (capacity < requiredInstances) capacity <<= 1;

        ulong sizeBytes = (ulong)capacity * (ulong)sizeof(ShadowEntityInfo);
        Gfx.CreateBuffer(sizeBytes, BufferUsageFlags.StorageBufferBit,
            MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit,
            out shadowInfoBuffer, out shadowInfoAlloc, preferDeviceLocal: true);

        shadowInfoMapped = Gfx.Allocator.GetMapped(shadowInfoAlloc);
        shadowInfoCapacity = capacity;
        return true;
    }

    /// <summary>
    /// Grows the emissive-triangle + alias-table buffers to hold at least
    /// <paramref name="requiredTris"/> entries (floored at 1 so the descriptor
    /// is always valid). Both buffers share one capacity. Returns true iff a
    /// reallocation happened — caller must re-register then.
    /// </summary>
    private bool EnsureEmissiveCapacity(uint requiredTris)
    {
        uint required = Math.Max(1u, requiredTris);
        if (emissiveCapacity >= required && emissiveTriBuffer.Handle != 0)
            return false;

        if (emissiveTriAlloc.IsValid)   { Gfx.DestroyBuffer(emissiveTriBuffer,   emissiveTriAlloc);   emissiveTriMapped   = null; }
        if (emissiveAliasAlloc.IsValid) { Gfx.DestroyBuffer(emissiveAliasBuffer, emissiveAliasAlloc); emissiveAliasMapped = null; }

        uint capacity = 8;
        while (capacity < required) capacity <<= 1;

        Gfx.CreateBuffer((ulong)capacity * (ulong)sizeof(EmissiveTriGpu),
            BufferUsageFlags.StorageBufferBit,
            MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit,
            out emissiveTriBuffer, out emissiveTriAlloc, preferDeviceLocal: true);
        Gfx.CreateBuffer((ulong)capacity * (ulong)sizeof(AliasEntryGpu),
            BufferUsageFlags.StorageBufferBit,
            MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit,
            out emissiveAliasBuffer, out emissiveAliasAlloc, preferDeviceLocal: true);

        emissiveTriMapped   = Gfx.Allocator.GetMapped(emissiveTriAlloc);
        emissiveAliasMapped = Gfx.Allocator.GetMapped(emissiveAliasAlloc);
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
        // Sticky until consumed at the re-register site: a later no-resize rebuild
        // must not erase a pending reallocation notice.
        emissiveBuffersResized |= EnsureEmissiveCapacity((uint)n);
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
    /// primitive). Build input + device address; host-visible so the rebuild
    /// writes it inline before the BLAS builds read it.
    /// </summary>
    private void EnsureClusterTransformCapacity(uint requiredSlots)
    {
        uint required = Math.Max(1u, requiredSlots);
        if (clusterTransformCapacity >= required && clusterTransformBuffer.Handle != 0)
            return;

        if (clusterTransformAlloc.IsValid)
        {
            Gfx.DestroyBuffer(clusterTransformBuffer, clusterTransformAlloc);
            clusterTransformMapped = null;
        }

        uint capacity = 8;
        while (capacity < required) capacity <<= 1;

        Gfx.CreateBuffer((ulong)capacity * (ulong)sizeof(TransformMatrixKHR),
            BufferUsageFlags.AccelerationStructureBuildInputReadOnlyBitKhr | BufferUsageFlags.ShaderDeviceAddressBit,
            MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit,
            out clusterTransformBuffer, out clusterTransformAlloc);

        clusterTransformMapped   = Gfx.Allocator.GetMapped(clusterTransformAlloc);
        clusterTransformCapacity = capacity;
    }

    /// <summary>Destroys all cluster BLAS handles + storage. Caller must have
    /// drained the device (Rebuild runs under its own DeviceWaitIdle, or at
    /// Initialize before any frame).</summary>
    private void DestroyClusterBlases()
    {
        foreach (var b in clusterBlases)
        {
            As.Destroy(b.Handle);
            Gfx.DestroyBuffer(b.Storage, b.StorageAlloc);
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

        ulong vbAddr    = Gfx.GetBufferDeviceAddress(Engine.ResourceManager.GlobalVertexBuffer);
        ulong ibBase    = Gfx.GetBufferDeviceAddress(Engine.ResourceManager.GlobalIndexBuffer);
        ulong xfBase    = Gfx.GetBufferDeviceAddress(clusterTransformBuffer);
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

            var sizes = As.GetBuildSizes(ref buildInfo, pMaxPrims);

            // High priority: BLAS storage is the persistent geometry BVH the path tracer
            // traverses every frame — must stay resident under WDDM budget pressure.
            Gfx.CreateBuffer(sizes.AccelerationStructureSize,
                BufferUsageFlags.AccelerationStructureStorageBitKhr | BufferUsageFlags.ShaderDeviceAddressBit,
                MemoryPropertyFlags.DeviceLocalBit, out var storage, out var storageAlloc,
                GpuMemoryAllocator.PriorityHigh);

            EnsureScratchCapacity(sizes.BuildScratchSize);

            var handle = As.Create(storage, sizes.AccelerationStructureSize,
                AccelerationStructureTypeKHR.BottomLevelKhr);
            buildInfo.DstAccelerationStructure  = handle;
            buildInfo.ScratchData.DeviceAddress = Gfx.GetBufferDeviceAddress(asScratchBuffer);

            fixed (AccelerationStructureBuildRangeInfoKHR* pRanges = ranges)
                As.Build(ref buildInfo, pRanges);

            var built = new BlasEntry
            {
                Handle        = handle,
                Storage       = storage,
                StorageAlloc  = storageAlloc,
                DeviceAddress = As.DeviceAddress(handle),
            };
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
        ulong compactedSize = As.QueryCompactedSize(src.Handle);

        // No useful shrink (or a driver reporting 0) — keep the original.
        if (compactedSize == 0 || compactedSize >= originalSize)
            return src;

        Gfx.CreateBuffer(compactedSize,
            BufferUsageFlags.AccelerationStructureStorageBitKhr | BufferUsageFlags.ShaderDeviceAddressBit,
            MemoryPropertyFlags.DeviceLocalBit, out var cStorage, out var cAlloc,
            GpuMemoryAllocator.PriorityHigh);

        var cHandle = As.Create(cStorage, compactedSize, AccelerationStructureTypeKHR.BottomLevelKhr);
        As.CopyCompact(src.Handle, cHandle);

        // Free the uncompacted source now that the compacted copy owns the geometry.
        As.Destroy(src.Handle);
        Gfx.DestroyBuffer(src.Storage, src.StorageAlloc);

        return new BlasEntry
        {
            Handle        = cHandle,
            Storage       = cStorage,
            StorageAlloc  = cAlloc,
            DeviceAddress = As.DeviceAddress(cHandle),
        };
    }

    //tlas previous entry count(ensures that if all are removed old tlas isnt used)
    private uint PreviousCount = 0;

    private void RebuildTlas()
    {
        // Reconcile renderable identity on the same cadence as the AS: every renderable gathered
        // below gets a stable RenderableHandle (freed when its entity leaves the scene), written
        // into ShadowEntityInfo.EntityIndex below and resolved back by pick / selection.
        GpuScene.SyncRenderables(Scene);

        // Open a fresh world-transform cache window for this rebuild. Unlike the per-frame
        // extraction, this is edit-driven and reachable from outside DrawFrame (the asset /
        // visibility paths rebuild synchronously), so it cannot assume the draw loop opened a
        // window for it - without this the gather below would read whatever cycle ran last.
        GpuScene.BeginTransforms();

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

        var scene = Scene;
        for (int i = 0; i < scene.EntityCount; i++)
        {
            Entity* e = scene.GetEntity(i);
            if (e == null) continue;
            if (!e->IsActive) continue;
            var transform = e->GetComponent<TransformComponent>();
            var meshComp  = e->GetComponent<MeshComponent>();
            if (transform == null || meshComp == null || meshComp.mesh == null) continue;

            Matrix4x4 world = GpuScene.WorldOf(e); // cached

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
                // later reorder/removal can't alias identity. Register is idempotent:
                // SyncRenderables already allocated this entity's slot at the top of
                // the rebuild, so this just reads it back.
                EntityIndex   = (int)GpuScene.Register(e).Index,
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
        shadowInfoBufferResized |= EnsureShadowInfoCapacity((uint)Math.Max(1, n));
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

        // An empty scene still builds a real zero-instance TLAS on the first pass:
        // the scene set's sceneTlas binding must hold a valid handle before an RT
        // core can be selected (rays just miss). Skip only re-building an already
        // existing TLAS with nothing.
        if (instCount == 0 && PreviousCount == 0 && tlas.Handle != 0)
            return;
        PreviousCount = instCount;


        // 5. Geometry — instance data lives at tlasInstanceBuffer's device address.
        uint instanceCount = instCount;
        var geo = new AccelerationStructureGeometryKHR
        {
            SType        = StructureType.AccelerationStructureGeometryKhr,
            GeometryType = GeometryTypeKHR.InstancesKhr,
            Flags        = GeometryFlagsKHR.OpaqueBitKhr,
        };
        geo.Geometry.Instances.SType              = StructureType.AccelerationStructureGeometryInstancesDataKhr;
        geo.Geometry.Instances.ArrayOfPointers    = false;
        geo.Geometry.Instances.Data.DeviceAddress = Gfx.GetBufferDeviceAddress(tlasInstanceBuffer);

        // 6. Build info — full rebuild for now. AllowUpdateBitKhr is set so a future
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

        // 7. Size query. For TLAS, the "primitive count" is the instance count.
        var sizes = As.GetBuildSizes(ref buildInfo, &instanceCount);

        // 8. (Re)allocate TLAS storage. Free + reallocate on every rebuild until
        //    the update-mode path lands.
        if (tlas.Handle != 0)
        {
            As.Destroy(tlas);
            Gfx.DestroyBuffer(tlasStorage, tlasStorageAlloc);
        }
        // High priority: TLAS storage is the top-level BVH traversed every path-trace frame.
        Gfx.CreateBuffer(sizes.AccelerationStructureSize,
            BufferUsageFlags.AccelerationStructureStorageBitKhr | BufferUsageFlags.ShaderDeviceAddressBit,
            MemoryPropertyFlags.DeviceLocalBit,
            out tlasStorage, out tlasStorageAlloc,
            GpuMemoryAllocator.PriorityHigh);

        tlas = As.Create(tlasStorage, sizes.AccelerationStructureSize,
            AccelerationStructureTypeKHR.TopLevelKhr);

        // 9. Wire scratch + dst into buildInfo, record + submit.
        EnsureScratchCapacity(sizes.BuildScratchSize);
        buildInfo.DstAccelerationStructure  = tlas;
        buildInfo.ScratchData.DeviceAddress = Gfx.GetBufferDeviceAddress(asScratchBuffer);

        var range = new AccelerationStructureBuildRangeInfoKHR
        {
            PrimitiveCount  = instanceCount,
            PrimitiveOffset = 0,
            FirstVertex     = 0,
            TransformOffset = 0,
        };
        As.Build(ref buildInfo, &range);
    }


    //  Orchestration

    /// <summary>
    /// Reclusters from scratch, rebuilds the world-space cluster BLASes and the TLAS, and
    /// republishes whatever the rebuild invalidated. Newly-joined meshes are picked up
    /// automatically because the gather walks the live scene. Performs a DeviceWaitIdle
    /// internally, so it must not be called with a command buffer open - the bake pump and the
    /// editor's synchronous asset paths both satisfy that.
    /// </summary>
    public void Rebuild()
    {
        // Changing the scene invalidates any in-progress path-trace integration regardless of
        // what happens to the AS below.
        _host.MarkAccumulatorDirty();

        Gfx.Vk!.DeviceWaitIdle(Gfx.Device);

        RebuildTlas();
        PublishBindings();
    }

    /// <summary>
    /// (Re)registers everything the rebuild may have replaced. The TLAS handle changes on every
    /// rebuild, so it always re-registers; the side-table buffers only change identity when their
    /// capacity is outgrown, and their contents update in place through the host-coherent mapping,
    /// so those re-register on the resize flag alone. One registry write per name covers every
    /// consuming shader - none of them names this feature.
    /// </summary>
    private void PublishBindings()
    {
        var r = _gpu.Registry;

        if (tlas.Handle != 0)
            r.RegisterTlas("sceneTlas", tlas);

        if (shadowInfoBufferResized)
        {
            r.RegisterBuffer("sceneEntityInfo", shadowInfoBuffer);
            shadowInfoBufferResized = false;
        }

        if (emissiveBuffersResized)
        {
            r.RegisterBuffer("sceneEmissiveTris",  emissiveTriBuffer);
            r.RegisterBuffer("sceneEmissiveAlias", emissiveAliasBuffer);
            emissiveBuffersResized = false;
        }
    }

    public void Dispose()
    {
        // Host-visible mappings live for the lifetime of the parent block (the
        // allocator owns the map/unmap). Just null the pointers — the frees below
        // release the suballocations; the block stays mapped until allocator dispose.
        tlasInstanceMapped     = null;
        shadowInfoMapped       = null;
        emissiveTriMapped      = null;
        emissiveAliasMapped    = null;
        clusterTransformMapped = null;

        if (shadowInfoBuffer.Handle       != 0) Gfx.DestroyBuffer(shadowInfoBuffer,       shadowInfoAlloc);
        if (emissiveTriBuffer.Handle      != 0) Gfx.DestroyBuffer(emissiveTriBuffer,      emissiveTriAlloc);
        if (emissiveAliasBuffer.Handle    != 0) Gfx.DestroyBuffer(emissiveAliasBuffer,    emissiveAliasAlloc);
        if (clusterTransformBuffer.Handle != 0) Gfx.DestroyBuffer(clusterTransformBuffer, clusterTransformAlloc);

        if (tlas.Handle != 0)
        {
            As.Destroy(tlas);
            tlas = default;
        }
        if (tlasStorage.Handle != 0)        Gfx.DestroyBuffer(tlasStorage,        tlasStorageAlloc);
        if (tlasInstanceBuffer.Handle != 0) Gfx.DestroyBuffer(tlasInstanceBuffer, tlasInstanceAlloc);
        if (asScratchBuffer.Handle != 0)    Gfx.DestroyBuffer(asScratchBuffer,    asScratchAlloc);

        DestroyClusterBlases();
    }
}
