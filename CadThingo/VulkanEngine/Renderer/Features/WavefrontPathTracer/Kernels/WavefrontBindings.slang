//set 2 - wavefront working set (graph-shared)
module WavefrontBindings;

__exported import PTUtils;
__exported import PbrUtils;
__exported import CommonTypes;
__exported import SceneBindings;   // scene set 0: materials/lights/tlas/geometry/emissive/textures/samplers
__exported import FeatureEnv;      // feature set 4: envCube


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

// Set 0 - frameUBO on the reserved (0,0) constant-arena slot; every other scene
// resource (lights/tlas/entityInfo/emissive/geometry/materials/textures/samplers)
// comes from the imported SceneBindings module.
public [[vk::binding(0, 0)]] ConstantBuffer<PathFrameUBO> frameUBO;

// Set 1 - pass-local IO storage images (binding indices match PTCompute set 1).
public [[vk::binding(0, 1)]] RWTexture2D<float4> accumulator;
public [[vk::binding(1, 1)]] RWTexture2D<float4> outColor;

// envCube comes from the imported FeatureEnv module (feature set 4).



// ---- fp16 packing ------------------------------------------------------------------------
// The wide per-path payloads are stored as packed halves (uint2 = half4, uint = half2):
// ~44B/path off the working set AND half the load/store traffic in the hot kernels. Values
// stay fp32 in registers (f32tof16 at store, f16tof32 at load), so no 16-bit storage device
// features are needed. Flag lanes survive exactly (-1/0/1 are exact halves); the firefly
// clamp (256) bounds radiance/contribution well inside half range (65504). KEPT fp32: ray
// origins + dirs (trace precision; psRayDir.w carries the MIS pdf, which can exceed half
// range at low roughness), shadowOrigin (w = tMax), hitT.
public uint2 wfPackHalf4(float4 v) {
    return uint2(f32tof16(v.x) | (f32tof16(v.y) << 16),
                 f32tof16(v.z) | (f32tof16(v.w) << 16));
}
public float4 wfUnpackHalf4(uint2 p) {
    return float4(f16tof32(p.x & 0xFFFFu), f16tof32(p.x >> 16),
                  f16tof32(p.y & 0xFFFFu), f16tof32(p.y >> 16));
}
public uint  wfPackHalf2(float2 v) { return f32tof16(v.x) | (f32tof16(v.y) << 16); }
public float2 wfUnpackHalf2(uint p) { return float2(f16tof32(p & 0xFFFFu), f16tof32(p >> 16)); }

// Ray-cone (coneWidth, coneSpread) <-> the float bit pattern parked in the otherwise-unused
// psRayOrigin.w lane. Lets ray-cone texture LOD ride along the persistent ray state with no
// added working-set bytes (see WavefrontShading::wfTexConeLod).
public float  wfPackCone(float width, float spread) { return asfloat(wfPackHalf2(float2(width, spread))); }
public float2 wfUnpackCone(float packed)            { return wfUnpackHalf2(asuint(packed)); }

// Persistent path state — survives every bounce. Indexed by pathIdx (= pixel index).
public [[vk::binding(0,2)]]  RWStructuredBuffer<float4> psRayOrigin;  // xyz = ray O          (w free)
public [[vk::binding(1,2)]]  RWStructuredBuffer<float4> psRayDir;     // xyz = ray D          (w = prevBsdfPdf, P2 MIS)
public [[vk::binding(2,2)]]  RWStructuredBuffer<uint2>  psThroughput; // PACKED half4: xyz = β, w = flags (1 = prevWasDelta, -1 = escaped w/ env stash)
public [[vk::binding(3,2)]]  RWStructuredBuffer<uint2>  psRadiance;   // PACKED half4: xyz = radiance L
public [[vk::binding(4,2)]]  RWStructuredBuffer<uint>   psRng;        // PRNG state (own array: frequent RMW, perfect coalescing)
public [[vk::binding(5,2)]]  RWStructuredBuffer<uint2>  psSigmaA;     // PACKED half4: xyz = medium σ_a

// Compact hit — transient Extend→Shade. Indexed by pathIdx (NOT by queue slot): a material
// sort/bin reorders the index list, so the hit must stay reachable from pathIdx, not from a
// position that the sort destroys (see "Material-sorted shading"). Everything else recomputed.
public [[vk::binding(6,2)]]  RWStructuredBuffer<uint2>  hitRecPrim;   // x = recordIndex (instanceID+geometryIndex), y = primIdx
public [[vk::binding(7,2)]]  RWStructuredBuffer<float>  hitT;         // committed ray t
public [[vk::binding(8,2)]]  RWStructuredBuffer<uint>   hitBary;      // PACKED half2: committed barycentrics

// Work queues — index lists (this is the ONLY thing a material sort reshuffles; state stays put).
// rayQueue ping-pongs by bounce parity (src/dst via push constant). shadeQueue is a single bin in
// P1; P3 replaces it with one bin per shading class (below).
public [[vk::binding(9,2)]]  RWStructuredBuffer<uint>   rayQueue0;
public [[vk::binding(10,2)]] RWStructuredBuffer<uint>   rayQueue1;
public [[vk::binding(11,2)]] RWStructuredBuffer<uint>   shadeQueue;   // pathIdx that hit this bounce

// Shadow records — Shade→Connect. Indexed by SHADOW SLOT (dense [0,shadowCount)). ≤1 record
// per path per bounce ⇒ shadowPath values are unique ⇒ Connect's radiance add needs no atomics.
// PING-PONGED BY BOUNCE PARITY: bounce b's records live in set b&1 (Shade(b) writes it,
// Connect(b) reads it) while Shade(b+1) fills the OTHER set -- so Connect(b) has no WAR
// against Shade(b+1) and its async overlap window stretches to Extend(b+1)+Shade(b+1)+
// Extend(b+2) (its real downstream consumer is Shade(b+2), which reuses set b&1).
// Access through the rdShadow*/wrShadowRec helpers below, never the raw arrays.
public [[vk::binding(12,2)]] RWStructuredBuffer<uint>   shadowPath0;   // target pathIdx
public [[vk::binding(13,2)]] RWStructuredBuffer<float4> shadowOrigin0; // xyz = O, w = tMax (fp32: positions + distances)
public [[vk::binding(14,2)]] RWStructuredBuffer<uint2>  shadowDir0;    // PACKED half4: xyz = D, w = castShadows
public [[vk::binding(15,2)]] RWStructuredBuffer<uint2>  shadowLe0;     // PACKED half4: xyz = precomputed β·f·W·Le·cosθ contribution
public [[vk::binding(19,2)]] RWStructuredBuffer<uint>   shadowPath1;
public [[vk::binding(20,2)]] RWStructuredBuffer<float4> shadowOrigin1;
public [[vk::binding(21,2)]] RWStructuredBuffer<uint2>  shadowDir1;
public [[vk::binding(22,2)]] RWStructuredBuffer<uint2>  shadowLe1;

// Connect's NEE contributions accumulate HERE, not in psRadiance: Connect is then the only
// writer of this buffer all frame (Generate zeroes it, Finalize folds it into the path
// radiance), so it conflicts with no graphics pass at all -- the other half of the wide
// async window. PACKED half4: xyz = summed visible NEE contribution for this path.
public [[vk::binding(24,2)]] RWStructuredBuffer<uint2>  shadowRadiance;

public uint   rdShadowPath  (uint par, uint i) { return par == 0u ? shadowPath0[i]   : shadowPath1[i]; }
public float4 rdShadowOrigin(uint par, uint i) { return par == 0u ? shadowOrigin0[i] : shadowOrigin1[i]; }
public float4 rdShadowDir   (uint par, uint i) { return wfUnpackHalf4(par == 0u ? shadowDir0[i] : shadowDir1[i]); }
public float3 rdShadowLe    (uint par, uint i) { return wfUnpackHalf4(par == 0u ? shadowLe0[i]  : shadowLe1[i]).xyz; }
public void wrShadowRec(uint par, uint s, uint path, float4 origin, float4 dir, float3 le) {
    if (par == 0u) {
        shadowPath0[s] = path; shadowOrigin0[s] = origin;
        shadowDir0[s] = wfPackHalf4(dir); shadowLe0[s] = wfPackHalf4(float4(le, 0.0));
    } else {
        shadowPath1[s] = path; shadowOrigin1[s] = origin;
        shadowDir1[s] = wfPackHalf4(dir); shadowLe1[s] = wfPackHalf4(float4(le, 0.0));
    }
}

// Counters + indirect args. Both array-shaped so the shade stage can fan out to one entry
// per shading class in P3 (shadeCount[c] / shade-args[c]) with sizing-only changes.
public [[vk::binding(16,2)]] RWStructuredBuffer<uint>   counters;     // rayCount, shadeCount[C], shadowCount, nextRayCount  (C=1 in P1/P2)
public [[vk::binding(17,2)]] RWStructuredBuffer<uint>   dispatchArgs; // VkDispatchIndirectCommand[]: extend, shade[C], connect(DEBUG-only, see below)

// Connect's OWN indirect-args buffers (decoupling): Connect(b) overlaps the next bounce's
// workers, and Extend's fused tail writes dispatchArgs (shade-args[b+1]) -- if Connect's
// launch dims lived there too, that write would force a barrier between them. So Shade's
// tail writes Connect's dims HERE instead, 16B per bounce: [groups, 1, 1, shadowCount]. The
// 4th word is the exact record count, read by the Connect kernel as its thread gate (it must
// NOT read counters[SHADOW_COUNT]: Extend(b+1)'s tail zeroes that while Connect(b) is in
// flight). PING-PONGED BY BOUNCE PARITY like the shadow records (Shade(b+1)'s tail writes
// the other buffer, so Connect(b)'s indirect read has no WAR against it). The dispatchArgs
// connect slot is still written, but ONLY for the Stats-panel readback.
public [[vk::binding(18,2)]] RWStructuredBuffer<uint>   connectArgs0;
public [[vk::binding(23,2)]] RWStructuredBuffer<uint>   connectArgs1;


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

// Write Connect(b)'s launch dims + exact record count into its parity buffer (16B stride:
// the indirect fetch reads words 0-2 at byte offset b*16, the kernel reads word 3).
public void wfWriteConnectArgs(uint bounce, uint groups, uint count){
    uint w = bounce * 4u;
    if ((bounce & 1u) == 0u) {
        connectArgs0[w + 0u] = groups; connectArgs0[w + 1u] = 1u;
        connectArgs0[w + 2u] = 1u;     connectArgs0[w + 3u] = count;
    } else {
        connectArgs1[w + 0u] = groups; connectArgs1[w + 1u] = 1u;
        connectArgs1[w + 2u] = 1u;     connectArgs1[w + 3u] = count;
    }
}
public uint wfConnectCount(uint bounce){
    return (bounce & 1u) == 0u ? connectArgs0[bounce * 4u + 3u] : connectArgs1[bounce * 4u + 3u];
}

// Device-coherent read of a counter the other groups built with InterlockedAdd (an RMW of +0
// observes the committed value, unlike a plain load that may sit in L1).
public uint wfCounterLoad(uint idx){ uint v; InterlockedAdd(counters[idx], 0u, v); return v; }

// Total launch size of the merged shade stage: the sum over the C class bins, each clamped
// with the same max(1,..) the launcher (Extend's tail) used per class. The counts were written
// by Extend and published by the pass barrier, and nothing rewrites them until the elected
// tail zeroes them (which happens-after every group has arrived), so plain loads are safe.
public uint wfShadeTotalGroups(){
    uint total = 0u;
    for(uint c = 0u; c < WF_SHADE_CLASSES; c++)
        total += wfGroupCount(counters[SHADE_COUNT_0 + c]);
    return total;
}

// "Last group out" election for the fused tail. ALL threads of the group MUST call this from
// uniform control flow (it issues a group sync). `numGroups` is the TOTAL number of workgroups
// ticketing this election -- wfGroupCount(RAY_COUNT) for Extend, wfShadeTotalGroups() for the
// merged shade stage (all C class dispatches share the ticket, so the tail runs in whichever
// class's group arrives last). Returns true in thread 0 of exactly that group -- that thread
// alone may then atomically read the final counters and write the next stage's args.
// The device fence's ONLY job is to order this group's counter ATOMICS (SHADE_COUNT[c], device-
// coherent, committed at L2 on completion) before the COMPLETED_WG bump, so the elected last group
// reads final counts. But it also drains the group's 64 scattered shadeQueue PLAIN stores -- which
// the election never reads (their visibility to the next dispatch comes from the graph's pipeline
// barrier, not this fence) -- so it over-fences, and that drain is the membar stall (~22% of Extend
// stall samples). The GroupMemoryBarrierWithGroupSync above already orders all 64 threads' counter
// adds before thread 0; counters are coherent atomics; so on NVIDIA this fence is believed redundant.
// Toggle to drop it: MEASURE the membar-stall win AND soak-test the per-bounce dispatch counts
// (StatsPanel) for consistency -- a stale count = wrong launch dims = intermittently missed shading.
public static const bool WF_FUSE_TAIL_DEVICE_FENCE = false;

public bool wfFuseTailElect(uint groupThreadId, uint numGroups){
    GroupMemoryBarrierWithGroupSync();          // this group's counter atomics issued; all threads here
    if(groupThreadId != 0u) return false;
    if(WF_FUSE_TAIL_DEVICE_FENCE) DeviceMemoryBarrier();   // order them device-wide before the arrival ticket
    uint ticket;
    InterlockedAdd(counters[COMPLETED_WG], 1u, ticket);
    if(ticket + 1u != numGroups) return false;
    counters[COMPLETED_WG] = 0u;                 // re-arm for the next fused dispatch (barrier-separated)
    return true;
}
