//set 4 - wavefront working set
module WavefrontBindings;

__exported import PTUtils;
__exported import PbrUtils;
__exported import CommonTypes;


public static const uint RAY_COUNT = 0, SHADE_COUNT_0 = 1, SHADOW_COUNT = 5, NEXT_RAY_COUNT = 6;
// Workgroup-arrival ticket for the fused-tail "last group out" election (wfFuseTailElect).
// Slot 7 sits past SHADOW_COUNT/NEXT_RAY_COUNT and clear of the shadeCount[C] fan-out region
// (1..4) reserved for P3. Reset to 0 before any dispatch whose tail fuses arg generation.
public static const uint COMPLETED_WG = 7;


public struct WavefrontPush {
public     uint bounce;
public     uint srcParity;
public     uint argsClass;     // P3 shading class (unused in P1/P2)
public     uint maxBounces;    // graph unroll depth; the fused tail guards extend-args[bounce+1]
}; // bounce parity + P3 class
public [[vk::push_constant]] WavefrontPush pc;

public uint rdRay(uint i){
    return pc.srcParity==0 ? rayQueue0[i] : rayQueue1[i];
}
public void wrRay(uint i, uint v){ 
    if (((pc.bounce + 1) & 1) == 0) rayQueue0[i] = v;
    else rayQueue1[i]=v; 
}

// Set 0 — per-frame IO (binding indices match PTCompute set 0).
public [[vk::binding(0, 0)]] ConstantBuffer<PathFrameUBO>       frameUBO;
public [[vk::binding(1, 0)]] StructuredBuffer<PbrLight>         lights;
public [[vk::binding(2, 0)]] RaytracingAccelerationStructure    tlas;
public [[vk::binding(3, 0)]] StructuredBuffer<ShadowEntityInfo> entityInfo;
public [[vk::binding(4, 0)]] RWTexture2D<float4>                accumulator;
public [[vk::binding(5, 0)]] RWTexture2D<float4>                outColor;
public [[vk::binding(6, 0)]] StructuredBuffer<EmissiveTri>      emissiveTris;
public [[vk::binding(7, 0)]] StructuredBuffer<AliasEntry>       emissiveAlias;

// Set 1 — geometry streams.
public [[vk::binding(1, 1)]] ByteAddressBuffer globalVertices;
public [[vk::binding(2, 1)]] ByteAddressBuffer globalIndices;

// Set 2 — bindless materials + textures.
public [[vk::binding(0, 2)]] StructuredBuffer<PbrMaterial> materials;
public [[vk::binding(2, 2)]] Texture2D                     textures[];
public [[vk::binding(3, 2)]] SamplerState                  samplers[];

// Set 3 — full-res environment cube (binding 3 matches PTCompute).
public [[vk::binding(3, 3)]] SamplerCube envCube;



// Persistent path state — survives every bounce. Indexed by pathIdx (= pixel index).
public [[vk::binding(0,4)]]  RWStructuredBuffer<float4> psRayOrigin;  // xyz = ray O          (w free)
public [[vk::binding(1,4)]]  RWStructuredBuffer<float4> psRayDir;     // xyz = ray D          (w = prevBsdfPdf, P2 MIS)
public [[vk::binding(2,4)]]  RWStructuredBuffer<float4> psThroughput; // xyz = throughput β   (w = packed flags: prevWasDelta|inMedium, P2)
public [[vk::binding(3,4)]]  RWStructuredBuffer<float4> psRadiance;   // xyz = radiance L     (w free)
public [[vk::binding(4,4)]]  RWStructuredBuffer<uint>   psRng;        // PRNG state (own array: frequent RMW, perfect coalescing)
public [[vk::binding(5,4)]]  RWStructuredBuffer<float4> psSigmaA;     // xyz = medium σ_a     (P2 only; omit binding in P1)

// Compact hit — transient Extend→Shade. Indexed by pathIdx (NOT by queue slot): a material
// sort/bin reorders the index list, so the hit must stay reachable from pathIdx, not from a
// position that the sort destroys (see "Material-sorted shading"). Everything else recomputed.
public [[vk::binding(6,4)]]  RWStructuredBuffer<uint2>  hitRecPrim;   // x = recordIndex (instanceID+geometryIndex), y = primIdx
public [[vk::binding(7,4)]]  RWStructuredBuffer<float>  hitT;         // committed ray t
public [[vk::binding(8,4)]]  RWStructuredBuffer<float2> hitBary;      // committed barycentrics

// Work queues — index lists (this is the ONLY thing a material sort reshuffles; state stays put).
// rayQueue ping-pongs by bounce parity (src/dst via push constant). shadeQueue is a single bin in
// P1; P3 replaces it with one bin per shading class (below).
public [[vk::binding(9,4)]]  RWStructuredBuffer<uint>   rayQueue0;
public [[vk::binding(10,4)]] RWStructuredBuffer<uint>   rayQueue1;
public [[vk::binding(11,4)]] RWStructuredBuffer<uint>   shadeQueue;   // pathIdx that hit this bounce

// Shadow records — Shade→Connect. Indexed by SHADOW SLOT (dense [0,shadowCount)). ≤1 record
// per path per bounce ⇒ shadowPath values are unique ⇒ Connect's radiance add needs no atomics.
public [[vk::binding(12,4)]] RWStructuredBuffer<uint>   shadowPath;   // target pathIdx
public [[vk::binding(13,4)]] RWStructuredBuffer<float4> shadowOrigin; // xyz = O, w = tMax
public [[vk::binding(14,4)]] RWStructuredBuffer<float4> shadowDir;    // xyz = D              (w free)
public [[vk::binding(15,4)]] RWStructuredBuffer<float4> shadowLe;     // xyz = precomputed β·f·W·Le·cosθ contribution

// Counters + indirect args. Both array-shaped so the shade stage can fan out to one entry
// per shading class in P3 (shadeCount[c] / shade-args[c]) with sizing-only changes.
public [[vk::binding(16,4)]] RWStructuredBuffer<uint>   counters;     // rayCount, shadeCount[C], shadowCount, nextRayCount  (C=1 in P1/P2)
public [[vk::binding(17,4)]] RWStructuredBuffer<uint>   dispatchArgs; // VkDispatchIndirectCommand[]: extend, shade[C], connect


// ---- Fused dispatch-arg generation (replaces the standalone PrepareArgs dispatches) --------
// Each worker kernel writes the NEXT stage's VkDispatchIndirectCommand from its own tail,
// removing the per-stage 1-thread Prep dispatch + its two pipeline barriers (less CPU record
// overhead, fewer compute-pipe drains). Because the launch size only exists once every group
// of the producer has finished its atomic counting, the tail runs in the *last* group to
// arrive (wfFuseTailElect) and reads the final counts with device-scope atomics.
//
// dispatchArgs layout MUST mirror WavefrontPTPipeline.{Extend,Shade,Connect}ArgsOffset:
// per bounce b a [extend | shade[0..C-1] | connect] block of 12-byte commands (P3: the shade
// stage fans out to one indirect command per material class). counters mirror this: shadeCount
// lives at SHADE_COUNT_0 + c for c in [0, WF_SHADE_CLASSES).
public static const uint WF_SHADE_CLASSES     = 4u;                       // C routing classes (P3.0)
public static const uint WF_STAGES_PER_BOUNCE = 2u + WF_SHADE_CLASSES;    // extend + C shade + connect
public static const uint WF_ARG_STRIDE        = 12u;   // sizeof(VkDispatchIndirectCommand), bytes
public uint wfExtendArgsOffset (uint b)         { return (b * WF_STAGES_PER_BOUNCE + 0u)            * WF_ARG_STRIDE; }
public uint wfShadeArgsOffset  (uint b, uint c) { return (b * WF_STAGES_PER_BOUNCE + 1u + c)        * WF_ARG_STRIDE; }
public uint wfConnectArgsOffset(uint b)         { return (b * WF_STAGES_PER_BOUNCE + 1u + WF_SHADE_CLASSES) * WF_ARG_STRIDE; }

// Group count a producer fans out over, for `count` items at 64/group. Clamped to >=1 so an
// EMPTY stage still launches exactly one group whose tail refreshes the downstream args -- a
// 0-group dispatch would never run, leaving the next stage's args stale. The launcher and the
// arriving kernel MUST compute this identically so the arrival ticket matches the launch size.
public uint wfGroupCount(uint count){ return max(1u, (count + 63u) / 64u); }

// Write one VkDispatchIndirectCommand (groups, 1, 1) at a byte offset into dispatchArgs.
public void wfWriteArgs(uint byteOffset, uint groups){
    uint w = byteOffset / 4u;
    dispatchArgs[w + 0u] = groups;
    dispatchArgs[w + 1u] = 1u;
    dispatchArgs[w + 2u] = 1u;
}

// Device-coherent read of a counter the other groups built with InterlockedAdd (an RMW of +0
// observes the committed value, unlike a plain load that may sit in L1).
public uint wfCounterLoad(uint idx){ uint v; InterlockedAdd(counters[idx], 0u, v); return v; }

// "Last group out" election for the fused tail. ALL threads of the group MUST call this from
// uniform control flow (it issues a group sync). `activeCount` is the value this dispatch was
// fanned out over (RAY_COUNT for Extend, SHADE_COUNT_0 for Shade); numGroups == wfGroupCount of
// it == the launch size. Returns true in thread 0 of exactly the last group to arrive -- that
// thread alone may then atomically read the final counters and write the next stage's args.
public bool wfFuseTailElect(uint groupThreadId, uint activeCount){
    GroupMemoryBarrierWithGroupSync();          // this group's global writes issued; all threads here
    if(groupThreadId != 0u) return false;
    DeviceMemoryBarrier();                       // order them device-wide before the arrival ticket
    uint numGroups = wfGroupCount(activeCount);
    uint ticket;
    InterlockedAdd(counters[COMPLETED_WG], 1u, ticket);
    if(ticket + 1u != numGroups) return false;
    counters[COMPLETED_WG] = 0u;                 // re-arm for the next fused dispatch (barrier-separated)
    return true;
}
