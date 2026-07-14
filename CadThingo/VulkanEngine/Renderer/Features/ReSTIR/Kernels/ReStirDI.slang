// ReStirDI — ReSTIR direct-illumination tracer (VK_KHR_ray_tracing_pipeline).
//
// P0 SCAFFOLD: a verbatim fork of PathTraceRT (the SER RT-pipeline path tracer)
// so the ReSTIR feature has its own shader + pipeline + core to diverge into
// without touching the proven tracer, which stays as a ground-truth fallback.
// At this phase the integrator is byte-identical to PathTraceRT (raygen owns the
// bounce loop: NEE + emission MIS + BSDF scatter + Beer-Lambert + RR + progressive
// accumulation); shadow rays stay INLINE ray query so maxPipelineRayRecursionDepth
// stays 1. The reservoir/G-buffer/temporal/spatial reuse stages land in later
// phases, replacing the primary-vertex NEE here.
//
// Pinhole camera only for now (other modes are a follow-up). Builds to
// Assets/Shaders/ReSTIR/ReStirDI.spv (+ _SER variant); see Shaders.targets.

import PTUtils;       // BSDF/sampling/RNG/camera + re-exported PbrUtils/CommonTypes
import SceneBindings; // scene set 0: sceneLights/sceneTlas/sceneEntityInfo/
                      // sceneVertices+sceneIndices/sceneEmissiveTris+Alias/
                      // sceneMaterials/sceneTextures/sceneSamplers
import FeatureEnv;    // feature set 4: envCube (miss-ray background)

// Per-pass constants in the reserved (0,0) dynamic slot.
[[vk::binding(0, 0)]] ConstantBuffer<PathFrameUBO> frameUBO;

// Set 1 — pass-local IO.
[[vk::binding(0, 1)]] RWTexture2D<float4> accumulator;
[[vk::binding(1, 1)]] RWTexture2D<float4> outColor;

// Set 2 (graph-shared) - ReSTIR working set (ReStirDIPipeline-owned, sized to the render extent, indexed by
// pixel = y*width + x). P1 writes both at the primary hit; the temporal/spatial reuse passes
// consume them in later phases. Scalar-only structs (4B fields, tight packing) so the Slang
// stride matches the C# byte strides in ReStirDIPipeline (ReservoirBytes/GBufferBytes).

// Per-pixel DI reservoir over the UNIFIED light set (analytic punctual/dir lights + emissive
// triangles). The selected sample is re-evaluable at any pixel: analytic = a light index
// (deterministic L from index + worldPos); emissive = the sampled world POINT + its (textured)
// radiance Le, so reuse + shading need NO emissive-triangle/texture fetch (keeps the shade pass
// lean). pHat = target fn of the selected sample; W = wSum/(M*pHat) the unbiased contribution
// weight. Scalar/tight Slang packing (float3 stays 12B, like PackedHit) -> C# stride 48B.
struct Reservoir {
    uint   sampleType;   // SAMPLE_NONE / SAMPLE_ANALYTIC / SAMPLE_EMISSIVE
    uint   index;        // analytic: light index (emissive: unused)
    float3 samplePos;    // emissive: sampled world point Q (analytic: unused)
    float3 sampleLe;     // emissive: radiance Le at Q (analytic: unused)
    float  sampleEsPdf;  // emissive: NEE solid-angle pdf at build (for shade-time NEE<->BSDF MIS)
    uint   M;            // candidates streamed
    float  wSum;         // running Σ RIS weights
    float  W;            // unbiased contribution weight
    float  pHat;         // target fn of the selected sample
};
static const uint SAMPLE_NONE     = 0u;
static const uint SAMPLE_ANALYTIC = 1u;
static const uint SAMPLE_EMISSIVE = 2u;

// Fat G-buffer = the compact hit record (40B) the closesthit already packs. Storing it per pixel
// lets the deferred SpatialShade pass reconstruct the full BRDF (unpackHit) without re-tracing.
// (Defined here, before the set-4 bindings that use it; packHit/unpackHit are below.)
struct PackedHit {
    float3 worldPos;    // full precision (shadow-ray origins / NEE)
    uint   nOct;        // shading normal, oct 2x16
    uint   ngOct;       // geometric normal, oct 2x16
    uint   baseIor;     // baseColor.rgb unorm8 | ior unorm8 over [1,3]
    uint   emisRG;      // emissive.rg as f16x2
    uint   scal0;       // metallic | roughness | transmission | clearcoat (unorm8x4)
    uint   scal1;       // clearcoatRoughness unorm8 | emissive.b f16 in bits 16..31
    uint   matFlags;    // matidx:24 | hit:bit24 | frontFace:bit25
};

// Reservoir + G-buffer PING-PONG (P2 temporal reuse): two physical buffers per role. frameParity
// (monotonic frame counter & 1, from the push constant) picks which is CURRENT (written this
// frame) vs PREVIOUS (last frame's result, read for reprojected reuse). The host binds all four;
// the graph imports them so consecutive frames' accesses are ordered (cross-frame WAW barrier).
// sceneRadiance (binding 4) is the deferred-shading hand-off: Trace writes everything EXCEPT the
// analytic direct light (indirect + emissive + env); SpatialShade adds the analytic direct +
// accumulates. P3 deferred split (the reservoir analytic shade moved out of this megakernel).
[[vk::binding(0, 2)]] RWStructuredBuffer<Reservoir> reservoirA;
[[vk::binding(1, 2)]] RWStructuredBuffer<Reservoir> reservoirB;
[[vk::binding(2, 2)]] RWStructuredBuffer<PackedHit> gbufferA;
[[vk::binding(3, 2)]] RWStructuredBuffer<PackedHit> gbufferB;
[[vk::binding(4, 2)]] RWStructuredBuffer<float4>    sceneRadiance;

// Per-frame ReSTIR state (push constant). prevViewProj reprojects a current-frame world position
// into LAST frame's screen (same view*proj(M22-flipped) construction the engine's raster/pick path
// uses, so the mul(M,v) idiom + ndc*0.5+0.5 lands on the matching pixel). Shared with SpatialShade.
struct ReStirPush {
    float4x4 prevViewProj;
    uint     frameParity;   // monotonic frame & 1: 0 -> cur=A/prev=B, 1 -> cur=B/prev=A
    uint     hasHistory;    // 0 on the first frame after (re)alloc -> skip temporal reuse
    uint     pad0, pad1;
};
[[vk::push_constant]] ReStirPush rpc;

// History clamp: bounds how much weight a single reused reservoir contributes, limiting lag /
// ghosting when the camera or lighting changes (Bitterli 2020 M-cap).
static const uint TEMPORAL_M_CAP = 20u;

// Parity-keyed cur/prev accessors (Slang can't index a resource by variable, so branch).
// G-buffer ping-pong write: cur = the half this frame writes (frameParity), read by BuildTemporal.
void writeCurGBuffer(uint i, PackedHit g) { if (rpc.frameParity == 0u) gbufferA[i] = g; else gbufferB[i] = g; }

// Compile-time config (the RT pipeline doesn't bake spec constants; hardcoded to
// the compute path's defaults).
static const bool  ENABLE_NEE          = true;
static const bool  RUSSIAN_ROULETTE    = true;
static const uint  MAX_BOUNCES         = 8u;
static const uint  MATERIAL_LOD_BOUNCE = 2u;
static const float FIREFLY_CLAMP       = 256.0;
static const uint  RIS_LIGHT_SAMPLE_COUNT = 8u;
static const float CONE_SPREAD_K       = 1.0;   // per-bounce ray-cone widening (rad), scaled by surface roughness
static const bool  CONE_LOD            = true;  // false -> always mip 0 (pre-cone behaviour), for A/B quality compare
static const bool  STOCH_MIP           = true;  // secondary bounces collapse trilinear -> one stochastic mip (half the texels)


// Interpolated UV / alpha helpers (RT stages → SampleLevel, no derivatives).
float2 computeUV(uint indexOffset, uint primIdx, float2 bary) {
    uint i0 = fetchTriangleIndex(sceneIndices, indexOffset, primIdx, 0u);
    uint i1 = fetchTriangleIndex(sceneIndices, indexOffset, primIdx, 1u);
    uint i2 = fetchTriangleIndex(sceneIndices, indexOffset, primIdx, 2u);
    float  w0 = 1.0 - bary.x - bary.y;
    return fetchVertexUV(sceneVertices, i0) * w0
         + fetchVertexUV(sceneVertices, i1) * bary.x
         + fetchVertexUV(sceneVertices, i2) * bary.y;
}

// Alpha-test mip policy: visibility/shadow alpha doesn't need full-res texels, so
// sample a coarser, cache-resident mip to cut L1TEX traffic on alpha-tested foliage
// (the hot any-hit path) for near-invisible change. Shadow rays tolerate the most
// blur; primary visibility stays gentle so silhouettes don't thin. SampleLevel
// clamps past the coarsest mip, so small textures just average toward their mean.
static const float ALPHA_LOD_ANYHIT = 1.0;   // primary/bounce visibility (any-hit)
static const float ALPHA_LOD_SHADOW = 3.0;   // NEE shadow-ray visibility

float computeAlpha(PbrMaterial mat, uint indexOffset, uint primIdx, float2 bary, float lod) {
    if (mat.baseColorTex < 0 || mat.flags == 0) return 1.0;
    float2 uv = computeUV(indexOffset, primIdx, bary);
    return sceneTextures[NonUniformResourceIndex(mat.baseColorTex)].SampleLevel(sceneSamplers[0], uv, lod).a * mat.baseColorFactor.a;
}

// Shadow ray (NEE visibility) — inline ray query, valid inside raygen.
bool isOccluded(float3 origin, float3 dir, float tMax, inout uint rngState) {
    RayDesc r;
    r.Origin = origin; r.Direction = dir; r.TMin = 0.0; r.TMax = tMax;

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    q.TraceRayInline(sceneTlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, r);
    while (q.Proceed()) {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
            ShadowEntityInfo cei = sceneEntityInfo[q.CandidateInstanceID() + q.CandidateGeometryIndex()];
            PbrMaterial      cmat = sceneMaterials[cei.materialIndex];
            float alpha = computeAlpha(cmat, cei.indexOffset,
                                       q.CandidatePrimitiveIndex(),
                                       q.CandidateTriangleBarycentrics(),
                                       ALPHA_LOD_SHADOW);
            float transmittance = cmat.transmissionFactor;
            if (cmat.transmissionTex >= 0) {
                float2 uv = computeUV(cei.indexOffset, q.CandidatePrimitiveIndex(),
                                      q.CandidateTriangleBarycentrics());
                transmittance *= sceneTextures[NonUniformResourceIndex(cmat.transmissionTex)].SampleLevel(sceneSamplers[0], uv, ALPHA_LOD_SHADOW).r;
            }
            float passAlpha = 1.0;
            if ((cmat.flags & 1u) != 0u) passAlpha = (alpha >= cmat.alphaCutoff) ? 1.0 : 0.0;
            else if ((cmat.flags & 4u) != 0u) passAlpha = alpha;
            float blockProb = passAlpha * (1.0 - transmittance);
            if (rand(rngState) < blockProb) q.CommitNonOpaqueTriangleHit();
        }
    }
    return q.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
}

// RIS over analytic lights.
struct LightPick { LightSample sample; float weight; bool valid; };

LightPick sampleLightRIS(inout uint rngState, float3 worldPos, float3 N) {
    LightPick best;
    best.sample = (LightSample)0; best.weight = 0.0; best.valid = false;
    if (frameUBO.lightCount == 0u) return best;

    uint M = (frameUBO.lightCount == 1u) ? 1u : RIS_LIGHT_SAMPLE_COUNT;
    float wSum = 0.0, bestPTarget = 0.0;
    LightSample bestLs = (LightSample)0;

    for (uint i = 0u; i < M; ++i) {
        uint idx = min(frameUBO.lightCount - 1u, uint(rand(rngState) * float(frameUBO.lightCount)));
        PbrLight cand = sceneLights[idx];
        LightSample ls;
        if (!evaluateLight(cand, worldPos, ls)) continue;
        float NdotL = max(dot(N, ls.L), 0.0);
        if (NdotL <= 0.0) continue;
        float pTarget = luminance(ls.radiance) * NdotL;
        float w = pTarget * float(frameUBO.lightCount);
        wSum += w;
        if (rand(rngState) * wSum < w) { bestLs = ls; bestPTarget = pTarget; }
    }
    if (bestPTarget <= 0.0 || wSum <= 0.0) return best;
    best.sample = bestLs;
    best.weight = wSum / (float(M) * bestPTarget);
    best.valid  = true;
    return best;
}

// NOTE: the reservoir build (unified RIS), temporal reuse, and their helpers (reservoirInit /
// setSample / diTarget / targetAtSurface / combineReservoir / reproject) live in BuildTemporal.slang
// now -- this megakernel only writes the fat G-buffer + sceneRadiance. SpatialShade owns spatial
// reuse + shading. The Reservoir struct + reservoirA/B bindings above remain only to document the
// shared set-4 layout this RT pipeline and the two compute passes are built on.

// An "invalid" G-buffer = a PackedHit with the hit bit (matFlags bit 24) clear; unpackHit returns
// h.hit = false. Written for primary-ray misses so reuse rejects the pixel.
PackedHit invalidGBuffer() { PackedHit g = (PackedHit)0; g.matFlags = 0u; return g; }

float3 mediumSigmaA(PbrMaterial m) {
    if (m.attenuationDistance <= 0.0 || m.attenuationDistance >= 1e6) return float3(0.0);
    return -log(max(m.attenuationColor, float3(1e-4))) / m.attenuationDistance;
}

// Emissive area-light sample (NEE), textured radiance, factor-based pdf.
struct EmissiveLightSample { float3 L; float3 radiance; float dist; float pdf; bool valid; };

EmissiveLightSample sampleEmissive(inout uint rng, float3 P) {
    EmissiveLightSample es;
    es.L = float3(0.0); es.radiance = float3(0.0); es.dist = 0.0; es.pdf = 0.0; es.valid = false;

    uint n = frameUBO.emissiveTriCount;
    if (n == 0u || frameUBO.totalEmissivePower <= 0.0) return es;

    uint i = min(n - 1u, uint(rand(rng) * float(n)));
    AliasEntry a = sceneEmissiveAlias[i];
    uint idx = (rand(rng) < a.prob) ? i : a.alias;

    EmissiveTri t = sceneEmissiveTris[idx];
    float3 p0 = t.p0Area.xyz;
    float3 e1 = t.e1LeR.xyz;
    float3 e2 = t.e2LeG.xyz;
    float3 Le = float3(t.e1LeR.w, t.e2LeG.w, t.nLeB.w);
    float3 nL = t.nLeB.xyz;

    float u1 = rand(rng), u2 = rand(rng);
    float su = sqrt(u1);
    float b1 = su * (1.0 - u2);
    float b2 = su * u2;
    float3 Q = p0 + e1 * b1 + e2 * b2;

    float3 toL = Q - P;
    float  dist = length(toL);
    if (dist < 1e-4) return es;
    float3 L = toL / dist;
    float cosL = abs(dot(nL, L));
    if (cosL < 1e-4) return es;

    float3 radiance = Le;
    if (t.meta.z >= 0) {
        uint io = (uint)t.meta.x, pi = (uint)t.meta.y;
        uint vi0 = fetchTriangleIndex(sceneIndices, io, pi, 0u);
        uint vi1 = fetchTriangleIndex(sceneIndices, io, pi, 1u);
        uint vi2 = fetchTriangleIndex(sceneIndices, io, pi, 2u);
        float2 uv = fetchVertexUV(sceneVertices, vi0) * (1.0 - b1 - b2)
                  + fetchVertexUV(sceneVertices, vi1) * b1
                  + fetchVertexUV(sceneVertices, vi2) * b2;
        radiance *= sceneTextures[NonUniformResourceIndex(t.meta.z)].SampleLevel(sceneSamplers[0], uv, 0.0).rgb;
    }

    es.L = L; es.radiance = radiance; es.dist = dist;
    es.pdf = luminance(Le) * dist * dist / (frameUBO.totalEmissivePower * cosL);
    es.valid = es.pdf > 0.0;
    return es;
}

// PackedHit (the 40B compact hit carried through the payload AND stored as the fat G-buffer) is
// declared up by the set-4 bindings. Its pack/unpack helpers follow.
float2 octWrap(float2 v) {
    return (1.0 - abs(v.yx)) * float2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0);
}
uint packNormalOct(float3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    float2 e = saturate(((n.z >= 0.0) ? n.xy : octWrap(n.xy)) * 0.5 + 0.5);
    return uint(round(e.x * 65535.0)) | (uint(round(e.y * 65535.0)) << 16);
}
float3 unpackNormalOct(uint p) {
    float2 e = float2(float(p & 0xFFFFu), float(p >> 16)) / 65535.0 * 2.0 - 1.0;
    float3 n = float3(e, 1.0 - abs(e.x) - abs(e.y));
    float t  = saturate(-n.z);
    n.xy += float2(n.x >= 0.0 ? -t : t, n.y >= 0.0 ? -t : t);
    return normalize(n);
}
uint packU8x4(float a, float b, float c, float d) {
    return uint(round(saturate(a) * 255.0))
        | (uint(round(saturate(b) * 255.0)) << 8)
        | (uint(round(saturate(c) * 255.0)) << 16)
        | (uint(round(saturate(d) * 255.0)) << 24);
}
float unpackU8(uint p, uint shift) { return float((p >> shift) & 0xFFu) / 255.0; }

PackedHit packHit(HitInfo h) {
    PackedHit p;
    p.worldPos = h.worldPos;
    p.nOct     = packNormalOct(h.normal);
    p.ngOct    = packNormalOct(h.geometricNormal);
    uint iorU  = uint(round(saturate((h.ior - 1.0) * 0.5) * 255.0));
    p.baseIor  = (packU8x4(h.baseColor.r, h.baseColor.g, h.baseColor.b, 0.0) & 0x00FFFFFFu) | (iorU << 24);
    p.emisRG   = f32tof16(h.emissive.r) | (f32tof16(h.emissive.g) << 16);
    p.scal0    = packU8x4(h.metallic, h.roughness, h.transmission, h.clearcoat);
    p.scal1    = uint(round(saturate(h.clearcoatRoughness) * 255.0)) | (f32tof16(h.emissive.b) << 16);
    p.matFlags = (h.matidx & 0x00FFFFFFu) | (1u << 24) | (h.frontFace ? (1u << 25) : 0u);
    return p;
}

HitInfo unpackHit(PackedHit p, float3 rayOrigin) {
    HitInfo h = (HitInfo)0;
    h.hit = (p.matFlags & (1u << 24)) != 0u;
    if (!h.hit) return h;
    h.matidx          = p.matFlags & 0x00FFFFFFu;
    h.frontFace       = (p.matFlags & (1u << 25)) != 0u;
    h.worldPos        = p.worldPos;
    h.hitT            = length(p.worldPos - rayOrigin);
    h.normal          = unpackNormalOct(p.nOct);
    h.geometricNormal = unpackNormalOct(p.ngOct);
    h.clearcoatNormal = h.normal;
    h.baseColor       = float4(unpackU8(p.baseIor, 0), unpackU8(p.baseIor, 8), unpackU8(p.baseIor, 16), 1.0);
    float ior         = 1.0 + unpackU8(p.baseIor, 24) * 2.0;
    h.ior             = ior;
    float f0e         = (ior - 1.0) / (ior + 1.0);
    h.F0              = f0e * f0e;
    h.metallic        = unpackU8(p.scal0, 0);
    h.roughness       = max(unpackU8(p.scal0, 8), 0.045);
    h.transmission    = unpackU8(p.scal0, 16);
    h.clearcoat       = unpackU8(p.scal0, 24);
    h.clearcoatRoughness = max(unpackU8(p.scal1, 0), 0.045);
    h.emissive        = float3(f16tof32(p.emisRG & 0xFFFFu), f16tof32(p.emisRG >> 16), f16tof32(p.scal1 >> 16));
    return h;
}

// Payload: packed hit (closesthit fills / miss clears), the rng channel any-hit
// advances for stochastic alpha, the bounce index closesthit reads for material
// LOD, and the ray-cone footprint (width at the ray origin + angular spread) that
// closesthit completes with the hit distance for texture LOD.
struct RayPayload {
    PackedHit ph;
    uint      rng;
    uint      bounce;
    float     coneWidth;
    float     coneSpread;
};


[shader("anyhit")]
void anyHitMain(inout RayPayload pay, BuiltInTriangleIntersectionAttributes attr) {
    ShadowEntityInfo info = sceneEntityInfo[InstanceID() + GeometryIndex()];
    PbrMaterial      mat  = sceneMaterials[info.materialIndex];

    // Opaque / transmissive-opaque (flags==0) → accept; closesthit decides refract.
    bool isMask  = (mat.flags & 1u) != 0u;
    bool isBlend = (mat.flags & 4u) != 0u;
    if (!isMask && !isBlend) return;

    float alpha = computeAlpha(mat, info.indexOffset, PrimitiveIndex(), attr.barycentrics, ALPHA_LOD_ANYHIT);
    bool accept = true;
    if (isMask)  accept = alpha >= mat.alphaCutoff;          // MASK
    if (isBlend) accept = rand(pay.rng) < alpha;             // BLEND stochastic (advances payload rng)
    if (!accept) IgnoreHit();
}

[shader("miss")]
void missMain(inout RayPayload pay) {
    pay.ph.matFlags = 0u;   // hit bit clear; raygen samples env from its ray direction
}

// Object-space point → world via the full per-geometry 3x4 (rows carry the
// translation in .w). Used to derive the hit position in the SER path, where
// RayTCurrent isn't available in raygen.
float3 xformPoint(ShadowEntityInfo info, float3 p) {
    return float3(dot(info.xform0.xyz, p) + info.xform0.w,
                  dot(info.xform1.xyz, p) + info.xform1.w,
                  dot(info.xform2.xyz, p) + info.xform2.w);
}

// Ray-cone mip for a bindless texture: adds the texture-resolution term to the
// per-triangle/footprint base computed in fillHit. Clamped so degenerate UVs or
// tiny cones never request a negative level. When stoch is set (secondary
// bounces), the trilinear pair collapses to one stochastically-chosen mip - half
// the texels, unbiased, with variance that vanishes at coarse mips.
float texConeLod(int texIdx, float lodBase, float xi, bool stoch) {
    if (!CONE_LOD) return 0.0;
    uint tw, th;
    sceneTextures[NonUniformResourceIndex(texIdx)].GetDimensions(tw, th);
    float lod = max(lodBase + 0.5 * log2(float(tw) * float(th)), 0.0);
    if (STOCH_MIP && stoch) {
        float fl = floor(lod);
        lod = fl + (xi < (lod - fl) ? 1.0 : 0.0);
    }
    return lod;
}

// Fetches geometry + material at the hit and returns a fully-populated HitInfo.
// Runs in closesthit for both paths. World hit position comes from the world-
// transformed triangle (not ray.t), so it is stage-independent. coneWidth/
// coneSpread carry the ray-cone footprint for texture LOD.
HitInfo fillHit(ShadowEntityInfo info, uint primIdx, float2 bary,
                float3 rayOrigin, float3 rayDir, bool lowDetail,
                float coneWidth, float coneSpread, float lodJitter, bool stochMip) {
    HitInfo h = (HitInfo)0;
    h.hit    = true;
    h.matidx = info.materialIndex;

    float3 baryW = float3(1.0 - bary.x - bary.y, bary.x, bary.y);
    uint i0 = fetchTriangleIndex(sceneIndices, info.indexOffset, primIdx, 0u);
    uint i1 = fetchTriangleIndex(sceneIndices, info.indexOffset, primIdx, 1u);
    uint i2 = fetchTriangleIndex(sceneIndices, info.indexOffset, primIdx, 2u);

    float3 op0 = fetchVertexPos(sceneVertices, i0);
    float3 op1 = fetchVertexPos(sceneVertices, i1);
    float3 op2 = fetchVertexPos(sceneVertices, i2);

    // World hit position from the transformed triangle; hitT along the unit ray.
    float3 wp0 = xformPoint(info, op0);
    float3 wp1 = xformPoint(info, op1);
    float3 wp2 = xformPoint(info, op2);
    h.worldPos = wp0 * baryW.x + wp1 * baryW.y + wp2 * baryW.z;
    h.hitT     = length(h.worldPos - rayOrigin);

    float2 uv0 = fetchVertexUV(sceneVertices, i0);
    float2 uv1 = fetchVertexUV(sceneVertices, i1);
    float2 uv2 = fetchVertexUV(sceneVertices, i2);
    float2 uv  = uv0 * baryW.x + uv1 * baryW.y + uv2 * baryW.z;

    float3 nObj = fetchVertexNormal(sceneVertices, i0) * baryW.x
                + fetchVertexNormal(sceneVertices, i1) * baryW.y
                + fetchVertexNormal(sceneVertices, i2) * baryW.z;
    float3 ngObj = normalize(cross(op1 - op0, op2 - op0));

    // Object → world for directions: 3x3 of the per-geometry transform (identity
    // instance transform can't do it — the cluster BLAS baked positions).
    float3x3 obj2worldR = entityNormalMatrix(info);
    h.normal          = normalize(mul(obj2worldR, nObj));
    h.geometricNormal = normalize(mul(obj2worldR, ngObj));

    h.frontFace = dot(h.geometricNormal, rayDir) < 0.0;
    if (!h.frontFace) { h.normal = -h.normal; h.geometricNormal = -h.geometricNormal; }

    // Ray-cone texture LOD base: footprint width at this hit, the triangle's
    // world/UV area ratio (texel density), and the grazing-angle stretch. The
    // per-texture resolution term is added in texConeLod.
    float coneW   = coneWidth + coneSpread * h.hitT;
    float uvCross = abs((uv1.x - uv0.x) * (uv2.y - uv0.y) - (uv1.y - uv0.y) * (uv2.x - uv0.x));
    float wCross  = length(cross(wp1 - wp0, wp2 - wp0));
    float cosT    = max(abs(dot(h.geometricNormal, rayDir)), 1e-4);
    float lodBase = 0.5 * log2(max(uvCross, 1e-12) / max(wCross, 1e-12))
                  + log2(max(coneW, 1e-8) / cosT);

    PbrMaterial mat = sceneMaterials[h.matidx];
    SamplerState s  = sceneSamplers[0];

    h.baseColor = mat.baseColorFactor;
    if (mat.baseColorTex >= 0)
        h.baseColor *= sceneTextures[NonUniformResourceIndex(mat.baseColorTex)].SampleLevel(s, uv, texConeLod(mat.baseColorTex, lodBase, lodJitter, stochMip));

    h.metallic  = mat.metallicFactor;
    h.roughness = mat.roughnessFactor;
    if (mat.physicalDescriptorTex >= 0 && !lowDetail) {
        float2 mr = sceneTextures[NonUniformResourceIndex(mat.physicalDescriptorTex)].SampleLevel(s, uv, texConeLod(mat.physicalDescriptorTex, lodBase, lodJitter, stochMip)).rg;
        h.metallic  *= mr.x;
        h.roughness *= mr.y;
    }

    h.emissive = mat.emissiveFactor;
    if (mat.emissiveTex >= 0)
        h.emissive *= sceneTextures[NonUniformResourceIndex(mat.emissiveTex)].SampleLevel(s, uv, texConeLod(mat.emissiveTex, lodBase, lodJitter, stochMip)).rgb;

    if (mat.normalTex >= 0 && !lowDetail) {
        float3 nm  = decodeTangentNormal(sceneTextures[NonUniformResourceIndex(mat.normalTex)].SampleLevel(s, uv, texConeLod(mat.normalTex, lodBase, lodJitter, stochMip)).rg);
        // Tangent from UV gradients instead of a per-vertex tangent fetch: reuses
        // the already-loaded positions/UVs, trading 48B/hit of fetch for ALU.
        float2 d1  = uv1 - uv0, d2 = uv2 - uv0;
        float  det = d1.x * d2.y - d1.y * d2.x;
        if (abs(det) > 1e-12) {
            float  r    = 1.0 / det;
            float3 tObj = ((op1 - op0) * d2.y - (op2 - op0) * d1.y) * r;
            float3 bObj = ((op2 - op0) * d1.x - (op1 - op0) * d2.x) * r;
            float3 T    = normalize(mul(obj2worldR, tObj));
            T           = normalize(T - h.normal * dot(h.normal, T));   // orthogonalize vs shading N
            float  sgn  = dot(cross(h.normal, T), normalize(mul(obj2worldR, bObj))) < 0.0 ? -1.0 : 1.0;
            float3 B    = cross(h.normal, T) * sgn;
            h.normal    = normalize(nm.x * T + nm.y * B + nm.z * h.normal);
        }
    }

    h.transmission = mat.transmissionFactor;
    if (mat.transmissionTex >= 0 && !lowDetail)
        h.transmission *= sceneTextures[NonUniformResourceIndex(mat.transmissionTex)].SampleLevel(s, uv, texConeLod(mat.transmissionTex, lodBase, lodJitter, stochMip)).r;
    h.ior = mat.ior;

    float f0_eta = (mat.ior - 1.0) / (mat.ior + 1.0);
    h.F0 = f0_eta * f0_eta;

    h.clearcoat = mat.clearcoatFactor;
    if (mat.clearcoatTex >= 0 && !lowDetail)
        h.clearcoat *= sceneTextures[NonUniformResourceIndex(mat.clearcoatTex)].SampleLevel(s, uv, texConeLod(mat.clearcoatTex, lodBase, lodJitter, stochMip)).r;
    h.clearcoatRoughness = mat.clearcoatRoughnessFactor;
    if (mat.clearcoatRoughnessTex >= 0 && !lowDetail)
        h.clearcoatRoughness *= sceneTextures[NonUniformResourceIndex(mat.clearcoatRoughnessTex)].SampleLevel(s, uv, texConeLod(mat.clearcoatRoughnessTex, lodBase, lodJitter, stochMip)).g;
    h.clearcoatRoughness = max(h.clearcoatRoughness, 0.045);
    h.roughness          = max(h.roughness,          0.045);
    h.clearcoatNormal    = h.normal;

    return h;
}

[shader("closesthit")]
void closestHitMain(inout RayPayload pay, BuiltInTriangleIntersectionAttributes attr) {
    ShadowEntityInfo info = sceneEntityInfo[InstanceID() + GeometryIndex()];
    // Decorrelated mip jitter from a copy of the rng (stream left untouched so the
    // light-transport noise is unchanged). Stochastic mip only past the primary.
    uint jr = pay.rng;
    float lodJitter = rand(jr);
    HitInfo h = fillHit(info, PrimitiveIndex(), attr.barycentrics,
                        WorldRayOrigin(), WorldRayDirection(),
                        pay.bounce >= MATERIAL_LOD_BOUNCE,
                        pay.coneWidth, pay.coneSpread,
                        lodJitter, pay.bounce >= 1u);
    pay.ph = packHit(h);
}

[shader("raygeneration")]
void rayGenMain() {
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 dim   = DispatchRaysDimensions().xy;

    uint rngState = initRNG(pixel, dim, frameUBO.frameIndex);

    // Tent-filtered sub-pixel jitter (matches the compute path).
    float2 jitter = float2(tentWarp(rand(rngState)), tentWarp(rand(rngState)));
    float2 px     = float2(pixel) + 0.5 + jitter;
    float2 ndc    = (px / frameUBO.screenSize) * 2.0 - 1.0;
    ndc.x        *= frameUBO.screenSize.x / frameUBO.screenSize.y;

    // Pinhole only for now.
    RayDesc ray = transformRayToWorld(generatePinholeRay(ndc, frameUBO.tanHalfFov), frameUBO.invView);

    float3 radiance   = float3(0.0);
    float3 throughput = float3(1.0);
    float3 sigmaA     = float3(0.0);
    float  prevBsdfPdf  = 0.0;
    bool   prevWasDelta = true;

    // Ray-cone footprint for texture LOD: starts at the camera per-pixel spread.
    float coneWidth  = 0.0;
    float coneSpread = 2.0 * frameUBO.tanHalfFov / frameUBO.screenSize.y;

    RayPayload pay;
    uint cap = min(MAX_BOUNCES, frameUBO.bounceCap);
    for (uint b = 0u; b < cap; ++b) {
        pay.bounce      = b;
        pay.rng         = rngState;
        pay.ph.matFlags = 0u;
        pay.coneWidth   = coneWidth;
        pay.coneSpread  = coneSpread;
#ifdef USE_SER
        // SER: HitObject.TraceRay records the hit (traversal + any-hit, defers
        // closesthit/miss), ReorderThread regroups warps by material, then Invoke
        // runs the now-coherent closesthit/miss. fillHit stays in closesthit so its
        // transient register peak is off the raygen megakernel. Needs the explicit
        // spvShaderInvocationReorderNV cap (Shaders.targets) for consistent NV ops.
        HitObject ho = HitObject.TraceRay(sceneTlas, RAY_FLAG_NONE, 0xFF, 0u, 0u, 0u, ray, pay);
        uint matHint = ho.IsHit()
            ? sceneEntityInfo[ho.GetInstanceID() + ho.GetGeometryIndex()].materialIndex
            : 0u;
        ReorderThread(ho, matHint, 8u);
        HitObject.Invoke(sceneTlas, ho, pay);
#else
        TraceRay(sceneTlas, RAY_FLAG_NONE, 0xFF, 0u, 0u, 0u, ray, pay);
#endif
        rngState = pay.rng;            // any-hit advanced it (BLEND)
        HitInfo hit = unpackHit(pay.ph, ray.Origin);

        if (!hit.hit) {
            // Primary-ray miss: write an INVALID G-buffer so BuildTemporal/SpatialShade reject this
            // pixel (the buffers ping-pong, so an unwritten slot would keep an old frame's hit).
            if (b == 0u)
                writeCurGBuffer(pixel.y * dim.x + pixel.x, invalidGBuffer());
            float3 env  = envCube.SampleLevel(ray.Direction, 0.0).rgb;
            float3 envC = throughput * env * frameUBO.scaleIBLAmbient;
            float  m    = max(max(envC.r, envC.g), envC.b);
            if (m > FIREFLY_CLAMP) envC *= FIREFLY_CLAMP / m;
            radiance += envC;
            break;
        }

        // ReSTIR DI (deferred): at the PRIMARY hit write ONLY the fat G-buffer. The reservoir build +
        // temporal reuse moved to the BuildTemporal compute pass; SpatialShade then reuses + shades.
        // Store the payload's PackedHit DIRECTLY (it is already the closesthit's pack) -- re-packing
        // unpackHit(pay.ph) would just burn raygen registers/ALU on a lossy round-trip the consumers
        // re-clamp on read anyway. The bounce loop below accumulates the rest into sceneRadiance.
        if (b == 0u)
            writeCurGBuffer(pixel.y * dim.x + pixel.x, pay.ph);

        // Footprint at this hit becomes the next ray's origin cone width.
        coneWidth += coneSpread * hit.hitT;

        if (any(sigmaA > 0.0)) throughput *= exp(-sigmaA * hit.hitT);

        // Emission pickup, MIS-weighted (power heuristic) against the emissive NEE pdf -- the
        // counterpart to the reservoir's emissive sample being MIS-weighted in SpatialShade. This
        // gives the BSDF strategy its share for glossy/specular emitter reflections (where NEE is
        // poor) while NEE handles the diffuse case, with no double-count.
        float misEm = 1.0;
        if (!prevWasDelta && frameUBO.emissiveTriCount > 0u && frameUBO.totalEmissivePower > 0.0) {
            float lumLe = luminance(sceneMaterials[hit.matidx].emissiveFactor);
            float cosL  = abs(dot(hit.geometricNormal, ray.Direction));
            if (lumLe > 0.0 && cosL > 1e-4) {
                float pLight = lumLe * hit.hitT * hit.hitT / (frameUBO.totalEmissivePower * cosL);
                misEm = powerHeuristic(prevBsdfPdf, pLight);
            }
        }
        float3 emC = throughput * hit.emissive * misEm;
        float  me  = max(max(emC.r, emC.g), emC.b);
        if (me > FIREFLY_CLAMP) emC *= FIREFLY_CLAMP / me;
        radiance += emC;

        float3 V = -ray.Direction;

        // NEE — single shadow ray per strategy (see PTCompute).
        if (ENABLE_NEE) {
            bool haveAnalytic = frameUBO.lightCount > 0u;
            bool haveEmissive = frameUBO.emissiveTriCount > 0u;

            if (b == 0u) {
                // ReSTIR DI primary vertex: ALL direct lighting (analytic + emissive) is in the
                // unified reservoir, shaded (once) in SpatialShade. Nothing inline here.
            } else if (haveAnalytic || haveEmissive) {
                // Bounces >= 1: unchanged stochastic analytic-or-emissive NEE (one shadow ray).
                float pAnalytic = (haveAnalytic && haveEmissive) ? 0.5
                                : (haveAnalytic ? 1.0 : 0.0);

                if (rand(rngState) < pAnalytic) {
                    LightPick pick = sampleLightRIS(rngState, hit.worldPos, hit.normal);
                    if (pick.valid) {
                        float NdotL = max(dot(hit.normal, pick.sample.L), 0.0);
                        if (NdotL > 0.0) {
                            float3 shadowOrigin = hit.worldPos + hit.geometricNormal * 1e-3;
                            float  tMax         = max(pick.sample.distance - 2e-3, 0.0);
                            bool   shadowed     = pick.sample.castShadows &&
                                                  isOccluded(shadowOrigin, pick.sample.L, tMax, rngState);
                            if (!shadowed) {
                                float3 brdf = evaluateFullBrdf(hit, V, pick.sample.L);
                                float3 neeC = throughput * brdf * pick.sample.radiance
                                                         * NdotL * pick.weight / pAnalytic;
                                float  mn   = max(max(neeC.r, neeC.g), neeC.b);
                                if (mn > FIREFLY_CLAMP) neeC *= FIREFLY_CLAMP / mn;
                                radiance += neeC;
                            }
                        }
                    }
                } else {
                    EmissiveLightSample es = sampleEmissive(rngState, hit.worldPos);
                    if (es.valid) {
                        float NdotL = max(dot(hit.normal, es.L), 0.0);
                        if (NdotL > 0.0) {
                            float3 shadowOrigin = hit.worldPos + hit.geometricNormal * 1e-3;
                            float  tMax         = max(es.dist - 2e-3, 0.0);
                            if (!isOccluded(shadowOrigin, es.L, tMax, rngState)) {
                                float  pB    = bsdfPdfForDir(hit, V, es.L);
                                float  w     = powerHeuristic(es.pdf, pB);
                                float3 brdf  = evaluateFullBrdf(hit, V, es.L);
                                float3 emNee = throughput * brdf * es.radiance * NdotL * w
                                                         / (es.pdf * (1.0 - pAnalytic));
                                float  mn    = max(max(emNee.r, emNee.g), emNee.b);
                                if (mn > FIREFLY_CLAMP) emNee *= FIREFLY_CLAMP / mn;
                                radiance += emNee;
                            }
                        }
                    }
                }
            }
        }

        BsdfSample bs = sampleBsdf(hit, V, rngState);
        if (!bs.valid) break;
        throughput *= bs.weight;
        prevBsdfPdf  = bs.pdf;
        prevWasDelta = bs.isDelta;

        // Non-delta scatter widens the cone (diffuse/glossy spread) so deeper,
        // incoherent bounces sample coarse mips. Delta lobes keep it tight.
        if (!bs.isDelta) coneSpread += CONE_SPREAD_K * max(hit.roughness, 0.3);

        if (any(isnan(throughput)) || any(isinf(throughput))) break;

        if (b >= MIN_BOUNCES && RUSSIAN_ROULETTE) {
            float rr = clamp(luminance(throughput), 0.05, 0.95);
            if (rand(rngState) > rr) break;
            throughput /= rr;
        }

        if (bs.transmit) {
            sigmaA = hit.frontFace ? mediumSigmaA(sceneMaterials[hit.matidx]) : float3(0.0);
        }

        float3 offsetN = bs.transmit ? -hit.geometricNormal : hit.geometricNormal;
        ray.Origin    = hit.worldPos + offsetN * 1e-3;
        ray.Direction = bs.wi;
        ray.TMin      = 0.0;
        ray.TMax      = INFINITY;
    }

    if (any(isnan(radiance)) || any(isinf(radiance))) radiance = float3(0.0);

    // Deferred-shading hand-off: write everything EXCEPT the analytic direct light (indirect +
    // emissive + env) to sceneRadiance. SpatialShade reads it, adds the spatially-reused analytic
    // direct, and owns the progressive accumulation into accumulator/outColor.
    sceneRadiance[pixel.y * dim.x + pixel.x] = float4(radiance, 0.0);
}