// SpatialShade — ReSTIR DI spatial reuse + deferred analytic shading (compute).
//
// The second ReSTIR pass: after the RT Trace pass has written every pixel's post-temporal reservoir
// + fat G-buffer (PackedHit) + sceneRadiance (indirect + emissive + env, i.e. everything EXCEPT the
// analytic direct light), this full-screen compute dispatch:
//   1. reconstructs the primary surface from the G-buffer,
//   2. SPATIALLY reuses a few same-frame neighbour reservoirs (geometric-rejected + biased-combined),
//   3. shades the surviving sample with one opaque shadow ray,
//   4. sums it with sceneRadiance and folds the result into the progressive accumulator.
//
// Runs on the RT pipeline's SHARED layout (set 0 = scene, set 1 = accum/outColor IO, set 2 = the
// ReSTIR working set), binding the exact descriptor sets the Trace pass uses -- the scene-set layout
// is All-stages and the IO/working sets carry ComputeBit. The shadow ray is ALPHA-TESTED (masked /
// blended / transmissive geometry occludes exactly as in the other tracers, for parity).
//
// The shared types/helpers mirror ReStirDI.slang (kept in sync by hand; small + stable). Same-frame
// spatial reuse is the coherence-friendly design: a uniform dispatch with a tiny register footprint
// + one coherent shadow ray, vs burying the work in the divergent megakernel.

import PTUtils;       // BRDF/light eval (evaluateFullBrdf, evaluateLight, LightSample), HitInfo, rand
import SceneBindings; // scene set 0: sceneLights/sceneTlas/sceneEntityInfo/sceneVertices+sceneIndices/sceneMaterials+sceneTextures+sceneSamplers

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
// Resolved sample ready to shade (unified across analytic + emissive).
struct ShadeSample { float3 L; float3 radiance; float dist; bool castShadows; float weight; bool valid; };

static const uint  SAMPLE_NONE            = 0u;
static const uint  SAMPLE_ANALYTIC        = 1u;
static const uint  SAMPLE_EMISSIVE        = 2u;
static const uint  TEMPORAL_M_CAP         = 30u;
static const uint  RESTIR_SPATIAL_SAMPLES = 4u;
static const float RESTIR_SPATIAL_RADIUS  = 16.0;
static const float FIREFLY_CLAMP          = 256.0;

// DIAGNOSTIC: set true to colour each pixel by what the direct-shade decided, instead of shading.
//   red    = primary miss / invalid G-buffer        blue  = G-buffer hit but reservoir has NO sample
//   yellow = sample valid but NdotL <= 0            black = shadow ray reports occluded
//   green  = lit (direct light added)
// Use it to identify what the dark bottom band is (e.g. all-black band => shadow ray over-occluding).
static const bool DEBUG_BAND = false;

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
// Scene set 0 (sceneLights / sceneTlas / sceneEntityInfo / sceneVertices+sceneIndices /
// sceneMaterials+sceneTextures+sceneSamplers) comes from the imported SceneBindings module; the
// alpha-tested shadow ray reads them exactly like the other tracers (parity). The scene-set layout
// carries All stages so it binds from this compute pass. Only the per-pass constants, pass-local IO,
// and the set-3 ReSTIR working buffers are declared here.
[[vk::binding(0, 0)]] ConstantBuffer<PathFrameUBO> frameUBO;

[[vk::binding(0, 1)]] RWTexture2D<float4> accumulator;
[[vk::binding(1, 1)]] RWTexture2D<float4> outColor;

[[vk::binding(0, 2)]] RWStructuredBuffer<Reservoir> reservoirA;
[[vk::binding(1, 2)]] RWStructuredBuffer<Reservoir> reservoirB;
[[vk::binding(2, 2)]] RWStructuredBuffer<PackedHit> gbufferA;
[[vk::binding(3, 2)]] RWStructuredBuffer<PackedHit> gbufferB;
[[vk::binding(4, 2)]] RWStructuredBuffer<float4>    sceneRadiance;   // Trace's indirect+emissive+env

struct ReStirPush { float4x4 prevViewProj; uint frameParity; uint hasHistory; uint pad0, pad1; };
[[vk::push_constant]] ReStirPush rpc;

// frameParity picks which ping-pong half is CURRENT this frame (the half Trace just wrote).
Reservoir readCurReservoir(uint i) { return rpc.frameParity == 0u ? reservoirA[i] : reservoirB[i]; }
PackedHit readCurGBuffer(uint i)   { return rpc.frameParity == 0u ? gbufferA[i]  : gbufferB[i];  }

// ---- Reservoir helpers (mirror ReStirDI.slang) ---------------------------------------------------
void reservoirInit(out Reservoir r) {
    r.sampleType = SAMPLE_NONE; r.index = 0u; r.samplePos = float3(0.0); r.sampleLe = float3(0.0);
    r.sampleEsPdf = 0.0; r.M = 0u; r.wSum = 0.0; r.W = 0.0; r.pHat = 0.0;
}
void setSample(inout Reservoir r, uint type, uint index, float3 pos, float3 le, float esPdf, float pHat) {
    r.sampleType = type; r.index = index; r.samplePos = pos; r.sampleLe = le; r.sampleEsPdf = esPdf; r.pHat = pHat;
}

// Unshadowed DI target = luminance(f_r * Le) * NdotL (BRDF-weighted; mirrors ReStirDI.slang). The
// BRDF term is what aligns spatial-reuse selection with the integrand, cutting glossy/shaded
// emissive variance. Visibility stays OUT (the shade-time shadow ray re-tests it).
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

// Lightweight domain test for the unbiased 1/Z normalisation: does candidate surface (worldPos,N)
// lie in the support of the sample's distribution? That support is {pHat>0}, and since pHat=diTarget
// has the BRDF factor and a GGX+Lambert BRDF is >0 across the hemisphere, {pHat>0} == {NdotL>0}. So
// this needs ONLY the neighbour's normal/worldPos (NOT a full unpackHit + BRDF eval) -- big register
// + ALU save in the Z loop, exact for these BRDFs.
bool sampleInDomain(Reservoir r, float3 worldPos, float3 N) {
    if (r.sampleType == SAMPLE_ANALYTIC) {
        LightSample ls;
        if (!evaluateLight(sceneLights[r.index], worldPos, ls)) return false;
        return dot(N, ls.L) > 0.0;
    }
    if (r.sampleType == SAMPLE_EMISSIVE)
        return dot(N, normalize(r.samplePos - worldPos)) > 0.0;
    return false;
}

// Resolve the reservoir's selected sample into a shadeable form: analytic re-evaluates the light;
// emissive uses the stored point + radiance (no triangle/texture fetch needed). weight = W.
ShadeSample reservoirToShade(Reservoir r, float3 worldPos, float3 N) {
    ShadeSample p = (ShadeSample)0; p.valid = false;
    if (r.sampleType == SAMPLE_NONE || r.W <= 0.0) return p;
    if (r.sampleType == SAMPLE_ANALYTIC) {
        LightSample ls;
        if (!evaluateLight(sceneLights[r.index], worldPos, ls)) return p;
        p.L = ls.L; p.radiance = ls.radiance; p.dist = ls.distance; p.castShadows = ls.castShadows;
    } else {
        float3 d = r.samplePos - worldPos; float dist = length(d);
        if (dist < 1e-4) return p;
        p.L = d / dist; p.radiance = r.sampleLe; p.dist = dist; p.castShadows = true;
    }
    p.weight = r.W; p.valid = true;
    return p;
}

// Spatial reuse is UNBIASED 1/Z combination (Bitterli 2020 Alg.6), inlined in main(): stream the
// canonical + neighbour reservoirs, then normalise by the confidence Z of only the candidates whose
// domain contains the selected sample. Kept inline so the candidate pixel indices + a single
// reconstructed neighbour surface stay in registers (this pass is deliberately register-light).

bool gbufHit(PackedHit g)    { return (g.matFlags & (1u << 24)) != 0u; }
uint gbufMatIdx(PackedHit g) { return g.matFlags & 0x00FFFFFFu; }

bool surfaceMatches(HitInfo h, PackedHit pg) {
    if (gbufMatIdx(pg) != h.matidx) return false;
    if (dot(h.normal, unpackNormalOct(pg.nOct)) < 0.9) return false;
    if (length(pg.worldPos - h.worldPos) > 0.02 * h.hitT + 1e-3) return false;
    return true;
}

// Interpolated UV + alpha at a candidate triangle (mirrors ReStirDI.slang -- no derivatives in this
// inline-ray-query path, so a fixed coarse mip). Shadow alpha tolerates heavy blur.
static const float ALPHA_LOD_SHADOW = 3.0;

float2 computeUV(uint indexOffset, uint primIdx, float2 bary) {
    uint i0 = fetchTriangleIndex(sceneIndices, indexOffset, primIdx, 0u);
    uint i1 = fetchTriangleIndex(sceneIndices, indexOffset, primIdx, 1u);
    uint i2 = fetchTriangleIndex(sceneIndices, indexOffset, primIdx, 2u);
    float w0 = 1.0 - bary.x - bary.y;
    return fetchVertexUV(sceneVertices, i0) * w0
         + fetchVertexUV(sceneVertices, i1) * bary.x
         + fetchVertexUV(sceneVertices, i2) * bary.y;
}
float computeAlpha(PbrMaterial mat, uint indexOffset, uint primIdx, float2 bary, float lod) {
    if (mat.baseColorTex < 0 || mat.flags == 0) return 1.0;
    float2 uv = computeUV(indexOffset, primIdx, bary);
    return sceneTextures[NonUniformResourceIndex(mat.baseColorTex)].SampleLevel(sceneSamplers[0], uv, lod).a * mat.baseColorFactor.a;
}

// Alpha-tested occlusion query -- mirrors ReStirDI.slang's isOccluded so masked/blended geometry
// (foliage, fences, transmissive glass) casts the SAME shadows as the other tracers (parity). Inline
// ray query, no cull flag: non-opaque candidates fetch the material + stochastically commit by
// alpha-cutoff (MASK) / alpha (BLEND) weighted by (1 - transmission).
bool isOccluded(float3 origin, float3 dir, float tMax, inout uint rngState) {
    RayDesc r; r.Origin = origin; r.Direction = dir; r.TMin = 0.0; r.TMax = tMax;
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    q.TraceRayInline(sceneTlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, r);
    while (q.Proceed()) {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
            ShadowEntityInfo cei = sceneEntityInfo[q.CandidateInstanceID() + q.CandidateGeometryIndex()];
            PbrMaterial      cmat = sceneMaterials[cei.materialIndex];
            float alpha = computeAlpha(cmat, cei.indexOffset, q.CandidatePrimitiveIndex(),
                                       q.CandidateTriangleBarycentrics(), ALPHA_LOD_SHADOW);
            float transmittance = cmat.transmissionFactor;
            if (cmat.transmissionTex >= 0) {
                float2 uv = computeUV(cei.indexOffset, q.CandidatePrimitiveIndex(), q.CandidateTriangleBarycentrics());
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


[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint2 dim = uint2(frameUBO.screenSize);
    if (any(gid.xy >= dim)) return;
    uint pidx = gid.y * dim.x + gid.x;

    float3 L = sceneRadiance[pidx].xyz;   // indirect + emissive + env from Trace
    PackedHit pg = readCurGBuffer(pidx);

    if (gbufHit(pg)) {
        // Reconstruct the primary surface; camPos as the "ray origin" makes hitT = view distance.
        float3 camPos = frameUBO.camPos.xyz;
        HitInfo hit = unpackHit(pg, camPos);
        float3 V    = normalize(camPos - hit.worldPos);

        uint rng = initRNG(gid.xy, dim, frameUBO.frameIndex);

        // Spatial reuse (UNBIASED, Bitterli 2020 Alg.6 / 1-over-Z MIS). Stream the canonical (center)
        // + a few same-frame neighbours that pass the geometric match, each resampled to THIS surface
        // (target = diTarget at center). Then normalise by Z = the summed confidence of only the
        // candidates whose domain actually contains the selected sample -- NOT the naive M-sum, which
        // over-counts disjoint domains and is what darkened disocclusions/edges. Domain test = target>0
        // at that candidate's own surface (no shadow ray -> only residual visibility bias remains, as
        // everywhere in this pass). Candidate pixel indices are recorded for the Z pass.
        const uint MAX_CAND = 1u + RESTIR_SPATIAL_SAMPLES;
        uint candIdx[MAX_CAND];
        uint candM[MAX_CAND];
        uint candN = 1u;

        Reservoir rc = readCurReservoir(pidx);
        candIdx[0] = pidx;
        candM[0]   = min(rc.M, TEMPORAL_M_CAP);

        Reservoir r; reservoirInit(r);
        float pHat0 = targetAtSurface(rc, hit, V);
        float w0    = pHat0 * rc.W * float(candM[0]);
        r.wSum += w0;
        if (w0 > 0.0) setSample(r, rc.sampleType, rc.index, rc.samplePos, rc.sampleLe, rc.sampleEsPdf, pHat0);

        for (uint i = 0u; i < RESTIR_SPATIAL_SAMPLES; ++i) {
            float2 off = (float2(rand(rng), rand(rng)) * 2.0 - 1.0) * RESTIR_SPATIAL_RADIUS;
            int2 qn = clamp(int2(gid.xy) + int2(off), int2(0, 0), int2(dim) - 1);
            uint qIdx = uint(qn.y) * dim.x + uint(qn.x);
            PackedHit ng = readCurGBuffer(qIdx);
            if (!(gbufHit(ng) && surfaceMatches(hit, ng))) continue;
            Reservoir rn = readCurReservoir(qIdx);
            if (rn.sampleType == SAMPLE_NONE || rn.M == 0u || rn.W <= 0.0) continue;
            uint  mN    = min(rn.M, TEMPORAL_M_CAP);
            float pHatN = targetAtSurface(rn, hit, V);
            float wN    = pHatN * rn.W * float(mN);
            r.wSum += wN;
            if (wN > 0.0 && rand(rng) * r.wSum < wN)
                setSample(r, rn.sampleType, rn.index, rn.samplePos, rn.sampleLe, rn.sampleEsPdf, pHatN);
            candIdx[candN] = qIdx;
            candM[candN]   = mN;
            candN++;
        }

        // Unbiased normalisation: W = wSum / (pHat * Z), Z = sum of candidate confidences whose
        // domain contains the selected sample. The canonical always counts (it produced pHat>0).
        float Zc = 0.0;
        if (r.sampleType != SAMPLE_NONE && r.pHat > 0.0) {
            for (uint j = 0u; j < candN; ++j) {
                float3 wj, nj;
                if (j == 0u) { wj = hit.worldPos; nj = hit.normal; }
                else {
                    PackedHit pgj = readCurGBuffer(candIdx[j]);   // normal/pos only -- no full unpackHit
                    wj = pgj.worldPos; nj = unpackNormalOct(pgj.nOct);
                }
                if (sampleInDomain(r, wj, nj)) Zc += float(candM[j]);
            }
        }
        r.M = candN;
        r.W = (r.pHat > 0.0 && Zc > 0.0) ? r.wSum / (r.pHat * Zc) : 0.0;

        // Shade the surviving sample (analytic OR emissive) with one opaque shadow ray.
        ShadeSample s = reservoirToShade(r, hit.worldPos, hit.normal);
        if (s.valid) {
            float NdotL = max(dot(hit.normal, s.L), 0.0);
            if (NdotL > 0.0) {
                float3 sOrigin = hit.worldPos + hit.geometricNormal * 1e-3;
                float  tMax    = max(s.dist - 2e-3, 0.0);
                bool shadowed = s.castShadows && isOccluded(sOrigin, s.L, tMax, rng);
                if (!shadowed) {
                    float3 direct = evaluateFullBrdf(hit, V, s.L) * s.radiance * NdotL * s.weight;
                    // Emissive samples MIS (power heuristic) vs the BSDF strategy (Trace's emission
                    // pickup), so glossy/specular emitter reflections get their BSDF share. Analytic
                    // lights can't be BSDF-hit -> full weight. Uses the sample's stored NEE pdf
                    // (exact for this pixel's own sample, approximate for a reused neighbour's).
                    if (r.sampleType == SAMPLE_EMISSIVE && r.sampleEsPdf > 0.0)
                        direct *= powerHeuristic(r.sampleEsPdf, bsdfPdfForDir(hit, V, s.L));
                    float  mn = max(max(direct.r, direct.g), direct.b);
                    if (mn > FIREFLY_CLAMP) direct *= FIREFLY_CLAMP / mn;
                    
                    
                    L += direct;
                    if (DEBUG_BAND) {
                        outColor[gid.xy] = float4(0, 100, 0,1);
                        return;
                    }//direct light added debug
                }
                //shadow ray report occluded
                if(DEBUG_BAND) {
                    outColor[gid.xy] = float4(0,0,0, 1); 
                    return;
                }
            } else if(DEBUG_BAND) {
                outColor[gid.xy] = float4(100, 50, 0,1);
                return;
            } // ndotl <= 0 debug
        } else if (DEBUG_BAND) {
            outColor[gid.xy] = float4(0, 0, 100, 1);   // BLUE: G-buffer hit but reservoir has NO sample
            return;
        }
    } else if (DEBUG_BAND) {
        outColor[gid.xy] = float4(100, 0, 0, 1);       // RED: primary miss / invalid G-buffer
        return;
    }

    if (any(isnan(L)) || any(isinf(L))) L = float3(0.0);
    if (frameUBO.resetAccum != 0u) accumulator[gid.xy] = float4(L, 1.0);
    else                           accumulator[gid.xy] += float4(L, 1.0);
    float4 a = accumulator[gid.xy];
    outColor[gid.xy] = float4(a.rgb /max(a.a, 1.0), 1.0);
}