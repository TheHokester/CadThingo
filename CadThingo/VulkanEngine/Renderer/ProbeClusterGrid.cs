using System.Numerics;
using System.Runtime.InteropServices;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer;

[StructLayout(LayoutKind.Sequential)]
public struct ClusterRange
{
    public uint Offset;   // start index into ProbeIndexList
    public uint Count;    // number of probe indices in this cluster
}

/// <summary>
/// CPU-built grid that bins reflection probes into per-cluster lists. Tile-only
/// (z = 1 slice) for now; the shader-side cluster index math
/// <c>cx + cy*dimsX + cz*dimsX*dimsY</c> stays identical when we later split
/// view-space depth into multiple z slices, so the PBR shader's lookup never
/// has to change shape.
///
/// Build runs on the CPU each frame:
///   1. Unproject each tile's 8 corners (4 near, 4 far) into world space →
///      per-tile world-AABB.
///   2. For each probe, sphere-AABB-test against every cluster. Append the
///      probe's slot index to that cluster's list (capped at MaxProbesPerCluster).
///   3. Write packed <see cref="ClusterRange"/>[] + uint[] into host-visible
///      SSBOs that the PBR shader binds.
///
/// Cost: dims.x × dims.y × dims.z × probeCount intersection tests, ~130k for a
/// 1080p × 16-probe scene → well under 0.5ms on a single core.
/// </summary>
public unsafe sealed class ProbeClusterGrid : IDisposable
{
    // Hard caps drive SSBO sizing. MaxClusters covers 4K tile-only (240 × 135);
    // when z-clustering lands the cap will need to multiply by zSlices.
    public const uint MaxClusters          = 240 * 135;
    public const uint MaxProbesPerCluster  = 8;
    public const uint MaxLinks             = MaxClusters * MaxProbesPerCluster;

    private readonly Renderer _renderer;

    // Per-frame-in-flight rings so the CPU can overwrite this frame's data
    // while the GPU still consumes the previous frame's. Host-visible mapped.
    internal UboBuffer[] clusterRangeBuffers = new UboBuffer[Renderer.MAX_CONCURRENT_FRAMES];
    internal UboBuffer[] probeIndexBuffers   = new UboBuffer[Renderer.MAX_CONCURRENT_FRAMES];

    // Snapshot of last Build's dimensionality. Phase 7 reads these to compute
    // the cluster index in the lighting shader.
    public uint DimsX { get; private set; }
    public uint DimsY { get; private set; }
    public uint DimsZ { get; private set; } = 1;

    // Scratch — kept as fields so Build doesn't reallocate every frame.
    private readonly List<int>[] _scratchPerCluster;
    private readonly List<(Vector3 Min, Vector3 Max)> _scratchBounds = new();

    public ProbeClusterGrid(Renderer renderer)
    {
        _renderer = renderer;
        _scratchPerCluster = new List<int>[MaxClusters];
        for (int i = 0; i < MaxClusters; i++) _scratchPerCluster[i] = new List<int>((int)MaxProbesPerCluster);

        for (int i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            _renderer.CreateMappedStorageBuffer(MaxClusters * (ulong)sizeof(ClusterRange), ref clusterRangeBuffers[i]);
            _renderer.CreateMappedStorageBuffer(MaxLinks    * sizeof(uint),                 ref probeIndexBuffers[i]);
        }
    }

    public Silk.NET.Vulkan.Buffer GetClusterRangeBuffer(uint frame) => clusterRangeBuffers[frame].buffer;
    public Silk.NET.Vulkan.Buffer GetProbeIndexBuffer  (uint frame) => probeIndexBuffers  [frame].buffer;
    public ulong ClusterRangeBufferSize => MaxClusters * (ulong)sizeof(ClusterRange);
    public ulong ProbeIndexBufferSize   => MaxLinks    * sizeof(uint);

    /// <summary>
    /// Rebuilds the cluster grid for the current frame. <paramref name="probes"/>
    /// is read snapshot-style — each probe's <c>WorldPosition</c> and
    /// <c>InfluenceRadius</c> are sampled once.
    /// </summary>
    public void Build(uint frameIndex, Camera camera, float aspect, float nearZ, float farZ,
        uint dimsX, uint dimsY, uint dimsZ,
        IReadOnlyList<ReflectionProbeComponent> probes)
    {
        DimsX = dimsX; DimsY = dimsY; DimsZ = dimsZ;
        uint clusterCount = dimsX * dimsY * dimsZ;
        if (clusterCount > MaxClusters)
            throw new InvalidOperationException(
                $"ProbeClusterGrid: cluster count {clusterCount} exceeds MaxClusters ({MaxClusters})");

        // Reset scratch slots up to clusterCount only — leaves prior frames'
        // tail slots dirty but Build doesn't read them.
        for (uint i = 0; i < clusterCount; i++) _scratchPerCluster[i].Clear();
        _scratchBounds.Clear();

        if (probes.Count > 0 && clusterCount > 0)
        {
            BuildClusterBounds(camera, aspect, nearZ, farZ, dimsX, dimsY, dimsZ);
            for (int pi = 0; pi < probes.Count; pi++)
            {
                var p = probes[pi];
                if (p.CubeArraySlot < 0) continue;
                BinProbe((uint)p.CubeArraySlot, p.WorldPosition, p.InfluenceRadius, clusterCount);
            }
        }

        Pack(frameIndex, clusterCount);
    }

    // ── World-space cluster bounds ───────────────────────────────

    private void BuildClusterBounds(Camera camera, float aspect, float nearZ, float farZ,
        uint dimsX, uint dimsY, uint dimsZ)
    {
        // Inverse view-proj turns NDC corners into world points. Vulkan NDC is
        // x,y in [-1,1] and z in [0,1].
        Matrix4x4 view = camera.GetViewMatrix();
        Matrix4x4 proj = camera.GetProjectionMatrix(aspect, nearZ, farZ);
        proj.M22 *= -1f;  // Vulkan Y flip (matches the rest of the engine)

        if (!Matrix4x4.Invert(view * proj, out var invVP))
            throw new InvalidOperationException("ProbeClusterGrid: camera view*proj is non-invertible");

        // For each cluster, sample the 4 near + 4 far NDC corners, unproject,
        // build a world-space AABB. With Z slices > 1, the near/far z values
        // get sliced — for tile-only the slice spans [0, 1].
        Span<Vector3> corners = stackalloc Vector3[8];
        for (uint cz = 0; cz < dimsZ; cz++)
        for (uint cy = 0; cy < dimsY; cy++)
        for (uint cx = 0; cx < dimsX; cx++)
        {
            float x0 = -1f + 2f * (float)cx       / dimsX;
            float x1 = -1f + 2f * (float)(cx + 1) / dimsX;
            float y0 = -1f + 2f * (float)cy       / dimsY;
            float y1 = -1f + 2f * (float)(cy + 1) / dimsY;
            float z0 = (float)cz       / dimsZ;   // [0,1] Vulkan NDC z slice
            float z1 = (float)(cz + 1) / dimsZ;

            corners[0] = UnprojectNdc(new Vector3(x0, y0, z0), invVP);
            corners[1] = UnprojectNdc(new Vector3(x1, y0, z0), invVP);
            corners[2] = UnprojectNdc(new Vector3(x0, y1, z0), invVP);
            corners[3] = UnprojectNdc(new Vector3(x1, y1, z0), invVP);
            corners[4] = UnprojectNdc(new Vector3(x0, y0, z1), invVP);
            corners[5] = UnprojectNdc(new Vector3(x1, y0, z1), invVP);
            corners[6] = UnprojectNdc(new Vector3(x0, y1, z1), invVP);
            corners[7] = UnprojectNdc(new Vector3(x1, y1, z1), invVP);

            Vector3 mn = corners[0], mx = corners[0];
            for (int i = 1; i < 8; i++)
            {
                mn = Vector3.Min(mn, corners[i]);
                mx = Vector3.Max(mx, corners[i]);
            }
            _scratchBounds.Add((mn, mx));
        }
    }

    private static Vector3 UnprojectNdc(Vector3 ndc, Matrix4x4 invVP)
    {
        Vector4 p = Vector4.Transform(new Vector4(ndc, 1f), invVP);
        return new Vector3(p.X, p.Y, p.Z) / p.W;
    }

    // ── Sphere-AABB binning ───────────────────────────────────────

    private void BinProbe(uint probeSlot, Vector3 center, float radius, uint clusterCount)
    {
        float r2 = radius * radius;
        for (int ci = 0; ci < clusterCount && ci < _scratchBounds.Count; ci++)
        {
            var (mn, mx) = _scratchBounds[ci];
            // Closest point on AABB to sphere center.
            Vector3 clamped = Vector3.Clamp(center, mn, mx);
            Vector3 diff = clamped - center;
            if (Vector3.Dot(diff, diff) > r2) continue;

            var list = _scratchPerCluster[ci];
            if (list.Count < MaxProbesPerCluster) list.Add((int)probeSlot);
            // Past the cap, additional probes are dropped — the lighting
            // shader sees only the first MaxProbesPerCluster contributors per
            // pixel. With 16-probe scenes this cap is never hit.
        }
    }

    // ── GPU upload ────────────────────────────────────────────────

    private void Pack(uint frameIndex, uint clusterCount)
    {
        var rangePtr = (ClusterRange*)clusterRangeBuffers[frameIndex].mapped;
        var indexPtr = (uint*)        probeIndexBuffers  [frameIndex].mapped;

        uint linkCursor = 0;
        for (uint ci = 0; ci < clusterCount; ci++)
        {
            var list = _scratchPerCluster[ci];
            uint count = (uint)list.Count;
            rangePtr[ci] = new ClusterRange { Offset = linkCursor, Count = count };
            for (int k = 0; k < count; k++) indexPtr[linkCursor + k] = (uint)list[k];
            linkCursor += count;
        }
    }

    public void Dispose()
    {
        for (int i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            _renderer.DestroyBuffer(clusterRangeBuffers[i].buffer, clusterRangeBuffers[i].alloc);
            _renderer.DestroyBuffer(probeIndexBuffers  [i].buffer, probeIndexBuffers  [i].alloc);
        }
    }
}
