using System.Numerics;

namespace CadThingo.VulkanEngine;

public enum ProbeUpdatePolicy
{
    // Capture exactly once, on first activation. Useful for static stage props
    // where the IBL doesn't need to track scene changes.
    Once,
    // Re-capture whenever the system flips Dirty=true on this probe. Default for
    // CAD scenes where parts can move and surrounding probes need to track them.
    OnDirty,
    // Forced periodic re-capture. Use for probes with dynamic content that doesn't
    // trigger transform events (animated material, environment swap, etc.). N
    // is in frames; system distributes captures across frames to stay in budget.
    EveryNFrames,
}

/// <summary>
/// Marks an entity as a reflection-probe sample point. Probe position is read
/// from the attached <see cref="TransformComponent"/> at capture time. The
/// influence radius determines which fragments blend this probe's contribution
/// at render time (see ReflectionProbeSystem.BuildClusterGrid).
///
/// Phase-2 status: data model only — no capture, no shader integration yet.
/// </summary>
public sealed unsafe class ReflectionProbeComponent : Component
{
    // Cubemap face resolution. 256 is the working default — large enough to look
    // sharp on glossy parts, small enough that capturing one face per frame at
    // ~16 probes stays well inside a 1ms GPU budget.
    public uint FaceSize = 256;

    // Influence volume: world-space sphere centred on the probe's transform.
    // Fragments inside multiple spheres get a weighted blend of probes.
    // Box / OBB volumes can land later without changing the public surface.
    public float InfluenceRadius = 5.0f;

    public ProbeUpdatePolicy UpdatePolicy = ProbeUpdatePolicy.OnDirty;

    // For EveryNFrames — interval in frames. Ignored otherwise.
    public uint UpdateIntervalFrames = 60;

    // Cube-array slot assigned by ReflectionProbeSystem at registration. -1
    // means the probe hasn't been registered yet (or the system ran out of
    // slots and skipped it).
    public int CubeArraySlot = -1;

    // System-side dirty flag. Set by the system itself in response to:
    //   - First registration (so the probe captures at least once)
    //   - Transform events on entities whose AABB overlaps InfluenceRadius
    //   - Manual user re-capture from the editor panel (Phase 8)
    public bool Dirty = true;

    // For EveryNFrames bookkeeping. Frame counter at last capture.
    internal ulong LastCaptureFrame;

    // Cached world-space position, refreshed once per frame by
    // ReflectionProbeSystem.Tick walking the scene. Kept on the component
    // (rather than recomputed from the transform per cluster test) so the
    // O(clusters × probes) cull loop stays branch-free of unsafe Entity*.
    internal System.Numerics.Vector3 WorldPosition;
}
