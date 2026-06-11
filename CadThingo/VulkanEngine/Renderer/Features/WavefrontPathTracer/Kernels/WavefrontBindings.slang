//set 4 - wavefront working set
module WavefrontBindings;

__exported import PTUtils;
__exported import PbrUtils;
__exported import CommonTypes;


public static const uint RAY_COUNT = 0, SHADE_COUNT_0 = 1, SHADOW_COUNT = 5, NEXT_RAY_COUNT = 6;


public struct WavefrontPush {
public     uint bounce;
public     uint srcParity;
public     uint argsClass;
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
