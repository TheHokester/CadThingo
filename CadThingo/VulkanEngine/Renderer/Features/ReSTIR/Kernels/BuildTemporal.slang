// BuildTemporal -- ReSTIR DI reservoir build + temporal reuse (compute).
//
// The FIRST of the two deferred ReSTIR compute passes. The RT megakernel (ReStirDI.slang) now only
// writes the fat G-buffer (PackedHit) + sceneRadiance; ALL reservoir work moved here to get the
// BRDF-weighted RIS candidate evaluation (the register hog) off the occupancy-critical SER raygen.
//
// Per pixel: reconstruct the primary surface from this frame's G-buffer, build the unified
// (analytic + emissive) DI reservoir via streaming RIS, then temporally reuse the reprojected
// previous-frame reservoir when the surface matches. Writes the cur-parity reservoir; SpatialShade
// (the second pass) then does spatial reuse + shading. Build does NO visibility (no rays) -- shadow
// rays live only in the shade pass -- so this pass needs no TLAS.
//
// Helpers mirror ReStirDI.slang / SpatialShade.slang (hand-synced; small + stable).

import PTUtils;       // BRDF/light eval, emissive structs, HitInfo, rand/initRNG, fetchVertex*
import SceneBindings; // scene set 0: sceneLights/sceneEmissiveTris+Alias/sceneVertices+sceneIndices/sceneTextures+sceneSamplers

// ---- Shared types (mirror ReStirDI.slang) --------------------------------------------------------
struct Reservoir {
    uint   sampleType;  // SAMPLE_NONE / SAMPLE_ANALYTIC / SAMPLE_EMISSIVE
    uint   index;       // analytic: light index
    float3 samplePos;   // emissive: sampled world point Q
    float3 sampleLe;    // emissive: radiance Le at Q
    float  sampleEsPdf; // emissive: NEE solid-angle pdf at build (for NEE<->BSDF MIS)
    uint   M; float wSum; float W; float pHat;
};
struct PackedHit {
    float3 worldPos; uint nOct; uint ngOct; uint baseIor; uint emisRG; uint scal0; uint scal1; uint matFlags;
};

static const uint  SAMPLE_NONE            = 0u;
static const uint  SAMPLE_ANALYTIC        = 1u;
static const uint  SAMPLE_EMISSIVE        = 2u;
static const uint  TEMPORAL_M_CAP         = 20u;
static const uint  RIS_LIGHT_SAMPLE_COUNT = 5u;
// Decorrelate the build RNG from the megakernel's stream (which reused the same pixel/frame seed for
// jitter + primary traversal) so reservoir candidates aren't correlated with the sub-pixel jitter.
static const uint  BUILD_RNG_SALT         = 0x9E3779B9u;

float3 unpackNormalOct(uint p) {
    float2 e = float2(float(p & 0xFFFFu), float(p >> 16)) / 65535.0 * 2.0 - 1.0;
    float3 n = float3(e, 1.0 - abs(e.x) - abs(e.y));
    float t  = saturate(-n.z);
    n.xy += float2(n.x >= 0.0 ? -t : t, n.y >= 0.0 ? -t : t);
    return normalize(n);
}
float unpackU8(uint p, uint shift) { return float((p >> shift) & 0xFFu) / 255.0; }

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

// ---- Bindings -------------------------------------------------------------------------------------
// Scene set 0 (sceneLights / sceneEmissiveTris+Alias / sceneVertices+sceneIndices /
// sceneTextures+sceneSamplers) comes from the imported SceneBindings module; only the
// per-pass constants and the set-2 (graph-shared) ReSTIR working buffers are declared here.
[[vk::binding(0, 0)]] ConstantBuffer<PathFrameUBO>  frameUBO;

[[vk::binding(0, 2)]] RWStructuredBuffer<Reservoir> reservoirA;
[[vk::binding(1, 2)]] RWStructuredBuffer<Reservoir> reservoirB;
[[vk::binding(2, 2)]] RWStructuredBuffer<PackedHit> gbufferA;
[[vk::binding(3, 2)]] RWStructuredBuffer<PackedHit> gbufferB;

struct ReStirPush { float4x4 prevViewProj; uint frameParity; uint hasHistory; uint pad0, pad1; };
[[vk::push_constant]] ReStirPush rpc;

// Parity-keyed cur/prev accessors (cur = the half the megakernel wrote / we write this frame).
void      writeCurReservoir(uint i, Reservoir r) { if (rpc.frameParity == 0u) reservoirA[i] = r; else reservoirB[i] = r; }
Reservoir readPrevReservoir(uint i)              { return rpc.frameParity == 0u ? reservoirB[i] : reservoirA[i]; }
PackedHit readCurGBuffer(uint i)                 { return rpc.frameParity == 0u ? gbufferA[i]  : gbufferB[i];  }
PackedHit readPrevGBuffer(uint i)                { return rpc.frameParity == 0u ? gbufferB[i]  : gbufferA[i];  }

// ---- Reservoir helpers (mirror ReStirDI.slang) ---------------------------------------------------
void reservoirInit(out Reservoir r) {
    r.sampleType = SAMPLE_NONE; r.index = 0u; r.samplePos = float3(0.0); r.sampleLe = float3(0.0);
    r.sampleEsPdf = 0.0; r.M = 0u; r.wSum = 0.0; r.W = 0.0; r.pHat = 0.0;
}
void setSample(inout Reservoir r, uint type, uint index, float3 pos, float3 le, float esPdf, float pHat) {
    r.sampleType = type; r.index = index; r.samplePos = pos; r.sampleLe = le; r.sampleEsPdf = esPdf; r.pHat = pHat;
}

// Unshadowed DI target p_hat = luminance(f_r * Le) * NdotL (BRDF-weighted; visibility re-tested in shade).
float diTarget(HitInfo hit, float3 V, float3 L, float3 radiance) {
    float NdotL = max(dot(hit.normal, L), 0.0);
    if (NdotL <= 0.0) return 0.0;
    return luminance(evaluateFullBrdf(hit, V, L) * radiance) * NdotL;
}
float targetAtSurface(Reservoir r, HitInfo hit, float3 V) {
    if (r.sampleType == SAMPLE_ANALYTIC) {
        LightSample ls;
        if (!evaluateLight(sceneLights[r.index], hit.worldPos, ls)) return 0.0;
        return diTarget(hit, V, ls.L, ls.radiance);
    }
    if (r.sampleType == SAMPLE_EMISSIVE) {
        float3 L = normalize(r.samplePos - hit.worldPos);
        return diTarget(hit, V, L, r.sampleLe);
    }
    return 0.0;
}

// Merge `other` (reprojected prev reservoir) into `cur`, target RE-EVALUATED at this surface, history
// M-capped (biased temporal combine -- one heavily-overlapping neighbour; spatial reuse is the 1/Z form).
void combineReservoir(inout Reservoir cur, Reservoir other, HitInfo hit, float3 V, inout uint rng) {
    if (other.sampleType == SAMPLE_NONE || other.M == 0u || other.W <= 0.0) return;
    Reservoir s; reservoirInit(s);
    float wCur = cur.pHat * cur.W * float(cur.M);
    s.wSum += wCur;
    if (wCur > 0.0) setSample(s, cur.sampleType, cur.index, cur.samplePos, cur.sampleLe, cur.sampleEsPdf, cur.pHat);
    uint  oM    = min(other.M, TEMPORAL_M_CAP);
    float pHatO = targetAtSurface(other, hit, V);
    float wO    = pHatO * other.W * float(oM);
    s.wSum += wO;
    if (s.wSum > 0.0 && rand(rng) * s.wSum < wO)
        setSample(s, other.sampleType, other.index, other.samplePos, other.sampleLe, other.sampleEsPdf, pHatO);
    s.M = cur.M + oM;
    s.W = (s.pHat > 0.0 && s.wSum > 0.0) ? s.wSum / (float(s.M) * s.pHat) : 0.0;
    cur = s;
}

bool gbufHit(PackedHit g) { return (g.matFlags & (1u << 24)) != 0u; }

bool surfaceMatches(HitInfo h, PackedHit pg) {
    if ((pg.matFlags & 0x00FFFFFFu) != h.matidx) return false;
    if (dot(h.normal, unpackNormalOct(pg.nOct)) < 0.9) return false;
    if (length(pg.worldPos - h.worldPos) > 0.02 * h.hitT + 1e-3) return false;
    return true;
}

// Reproject a current-frame world position into last frame's pixel (mul(M,v) idiom + ndc*0.5+0.5).
bool reproject(float3 worldPos, float2 screenSize, out int2 q) {
    q = int2(0, 0);
    float4 clip = mul(rpc.prevViewProj, float4(worldPos, 1.0));
    if (clip.w <= 1e-6) return false;
    float2 uv = (clip.xy / clip.w) * 0.5 + 0.5;
    if (any(uv < 0.0) || any(uv > 1.0)) return false;
    q = clamp(int2(uv * screenSize), int2(0, 0), int2(screenSize) - 1);
    return true;
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

// Streaming RIS over the UNIFIED light set (analytic uniform-index OR emissive alias-table), picked
// with prob pAnalytic; RIS weight = pHat/sourcePdf (sourcePdf folds the category prob). pHat =
// diTarget (BRDF-weighted). Emissive stores the world POINT + Le so reuse/shade need no re-fetch.
Reservoir buildUnifiedReservoir(inout uint rngState, HitInfo hit, float3 V) {
    Reservoir r; reservoirInit(r);
    float3 worldPos = hit.worldPos;
    bool haveA = frameUBO.lightCount > 0u;
    bool haveE = frameUBO.emissiveTriCount > 0u && frameUBO.totalEmissivePower > 0.0;
    if (!haveA && !haveE) return r;
    float pAnalytic = (haveA && haveE) ? 0.5 : (haveA ? 1.0 : 0.0);

    uint M = RIS_LIGHT_SAMPLE_COUNT;
    for (uint i = 0u; i < M; ++i) {
        if (rand(rngState) < pAnalytic) {
            uint idx = min(frameUBO.lightCount - 1u, uint(rand(rngState) * float(frameUBO.lightCount)));
            LightSample ls;
            if (!evaluateLight(sceneLights[idx], worldPos, ls)) continue;
            float pHat   = diTarget(hit, V, ls.L, ls.radiance);
            if (pHat <= 0.0) continue;
            float srcPdf = pAnalytic / float(frameUBO.lightCount);
            float risW   = (srcPdf > 0.0) ? pHat / srcPdf : 0.0;
            r.wSum += risW;
            if (risW > 0.0 && rand(rngState) * r.wSum < risW)
                setSample(r, SAMPLE_ANALYTIC, idx, float3(0.0), float3(0.0), 0.0, pHat);
        } else {
            EmissiveLightSample es = sampleEmissive(rngState, worldPos);
            if (!es.valid) continue;
            float pHat   = diTarget(hit, V, es.L, es.radiance);
            if (pHat <= 0.0) continue;
            float srcPdf = (1.0 - pAnalytic) * es.pdf;
            float risW   = (srcPdf > 0.0) ? pHat / srcPdf : 0.0;
            r.wSum += risW;
            if (risW > 0.0 && rand(rngState) * r.wSum < risW)
                setSample(r, SAMPLE_EMISSIVE, 0u, worldPos + es.L * es.dist, es.radiance, es.pdf, pHat);
        }
    }
    r.M = M;
    r.W = (r.pHat > 0.0 && r.wSum > 0.0) ? r.wSum / (float(M) * r.pHat) : 0.0;
    return r;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint2 dim = uint2(frameUBO.screenSize);
    if (any(gid.xy >= dim)) return;
    uint pidx = gid.y * dim.x + gid.x;

    Reservoir r; reservoirInit(r);   // primary-miss pixels store an empty reservoir (reuse rejects them)
    PackedHit pg = readCurGBuffer(pidx);
    if (gbufHit(pg)) {
        float3 camPos = frameUBO.camPos.xyz;
        HitInfo hit = unpackHit(pg, camPos);
        float3 V    = normalize(camPos - hit.worldPos);
        uint   rng  = initRNG(gid.xy, dim, frameUBO.frameIndex + BUILD_RNG_SALT);

        r = buildUnifiedReservoir(rng, hit, V);

        // Temporal reuse: reproject into last frame, merge the prev reservoir if the surface matches.
        if (rpc.hasHistory != 0u) {
            int2 q;
            if (reproject(hit.worldPos, frameUBO.screenSize, q)) {
                uint qidx = uint(q.y) * dim.x + uint(q.x);
                PackedHit pgPrev = readPrevGBuffer(qidx);
                if (gbufHit(pgPrev) && surfaceMatches(hit, pgPrev))
                    combineReservoir(r, readPrevReservoir(qidx), hit, V, rng);
            }
        }
    }
    writeCurReservoir(pidx, r);
}