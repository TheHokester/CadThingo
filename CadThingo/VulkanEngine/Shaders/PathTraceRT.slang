// PathTraceRT — ray-tracing-PIPELINE path tracer (VK_KHR_ray_tracing_pipeline).
//
// Phase 3: full integrator. Raygen owns the bounce loop (NEE + emission MIS +
// BSDF scatter + Beer-Lambert + RR + accumulation), the closest-hit fills a
// HitInfo payload, the any-hit does MASK/BLEND alpha, the miss flags a miss.
// Shadow rays stay INLINE ray query inside raygen so maxPipelineRayRecursionDepth
// stays 1. The BSDF/sampling/NEE/RNG math is reused from PTUtils; the binding-
// dependent helpers (computeUV/computeAlpha/isOccluded/sampleLightRIS/
// sampleEmissive/mediumSigmaA) are duplicated from PTCompute so the compute path
// tracer stays byte-for-byte untouched as the fallback.
//
// Pinhole camera only for now (other modes are a follow-up). Compile:
//   slangc PathTraceRT.slang -target spirv -capability spvRayTracingKHR -o ...

import PTUtils;   // BSDF/sampling/RNG/camera + re-exported PbrUtils/CommonTypes



// Set 0 — per-frame IO (binding indices match PTCompute set 0).
[[vk::binding(0, 0)]] ConstantBuffer<PathFrameUBO>       frameUBO;
[[vk::binding(1, 0)]] StructuredBuffer<PbrLight>         lights;
[[vk::binding(2, 0)]] RaytracingAccelerationStructure    tlas;
[[vk::binding(3, 0)]] StructuredBuffer<ShadowEntityInfo> entityInfo;
[[vk::binding(4, 0)]] RWTexture2D<float4>                accumulator;
[[vk::binding(5, 0)]] RWTexture2D<float4>                outColor;
[[vk::binding(6, 0)]] StructuredBuffer<EmissiveTri>      emissiveTris;
[[vk::binding(7, 0)]] StructuredBuffer<AliasEntry>       emissiveAlias;

// Set 1 — geometry streams.
[[vk::binding(1, 1)]] ByteAddressBuffer globalVertices;
[[vk::binding(2, 1)]] ByteAddressBuffer globalIndices;

// Set 2 — bindless materials + textures.
[[vk::binding(0, 2)]] StructuredBuffer<PbrMaterial> materials;
[[vk::binding(2, 2)]] Texture2D                     textures[];
[[vk::binding(3, 2)]] SamplerState                  samplers[];

// Set 3 — full-res environment cube (binding 3 matches PTCompute).
[[vk::binding(3, 3)]] SamplerCube envCube;

// Compile-time config (the RT pipeline doesn't bake spec constants; hardcoded to
// the compute path's defaults).
static const bool  ENABLE_NEE          = true;
static const bool  RUSSIAN_ROULETTE    = true;
static const uint  MAX_BOUNCES         = 8u;
static const uint  MATERIAL_LOD_BOUNCE = 2u;
static const float FIREFLY_CLAMP       = 256.0;
static const uint  RIS_LIGHT_SAMPLE_COUNT = 8u;


// Interpolated UV / alpha helpers (RT stages → SampleLevel, no derivatives).
float2 computeUV(uint indexOffset, uint primIdx, float2 bary) {
    uint i0 = fetchTriangleIndex(globalIndices, indexOffset, primIdx, 0u);
    uint i1 = fetchTriangleIndex(globalIndices, indexOffset, primIdx, 1u);
    uint i2 = fetchTriangleIndex(globalIndices, indexOffset, primIdx, 2u);
    float  w0 = 1.0 - bary.x - bary.y;
    return fetchVertexUV(globalVertices, i0) * w0
         + fetchVertexUV(globalVertices, i1) * bary.x
         + fetchVertexUV(globalVertices, i2) * bary.y;
}

float computeAlpha(PbrMaterial mat, uint indexOffset, uint primIdx, float2 bary) {
    if (mat.baseColorTex < 0 || mat.flags == 0) return 1.0;
    float2 uv = computeUV(indexOffset, primIdx, bary);
    return textures[NonUniformResourceIndex(mat.baseColorTex)].SampleLevel(samplers[0], uv, 0).a * mat.baseColorFactor.a;
}

// Shadow ray (NEE visibility) — inline ray query, valid inside raygen.
bool isOccluded(float3 origin, float3 dir, float tMax, inout uint rngState) {
    RayDesc r;
    r.Origin = origin; r.Direction = dir; r.TMin = 0.0; r.TMax = tMax;

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    q.TraceRayInline(tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, r);
    while (q.Proceed()) {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
            ShadowEntityInfo cei = entityInfo[q.CandidateInstanceID() + q.CandidateGeometryIndex()];
            PbrMaterial      cmat = materials[cei.materialIndex];
            float alpha = computeAlpha(cmat, cei.indexOffset,
                                       q.CandidatePrimitiveIndex(),
                                       q.CandidateTriangleBarycentrics());
            float transmittance = cmat.transmissionFactor;
            if (cmat.transmissionTex >= 0) {
                float2 uv = computeUV(cei.indexOffset, q.CandidatePrimitiveIndex(),
                                      q.CandidateTriangleBarycentrics());
                transmittance *= textures[NonUniformResourceIndex(cmat.transmissionTex)].SampleLevel(samplers[0], uv, 0).r;
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
        PbrLight cand = lights[idx];
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
    AliasEntry a = emissiveAlias[i];
    uint idx = (rand(rng) < a.prob) ? i : a.alias;

    EmissiveTri t = emissiveTris[idx];
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
        uint vi0 = fetchTriangleIndex(globalIndices, io, pi, 0u);
        uint vi1 = fetchTriangleIndex(globalIndices, io, pi, 1u);
        uint vi2 = fetchTriangleIndex(globalIndices, io, pi, 2u);
        float2 uv = fetchVertexUV(globalVertices, vi0) * (1.0 - b1 - b2)
                  + fetchVertexUV(globalVertices, vi1) * b1
                  + fetchVertexUV(globalVertices, vi2) * b2;
        radiance *= textures[NonUniformResourceIndex(t.meta.z)].SampleLevel(samplers[0], uv, 0.0).rgb;
    }

    es.L = L; es.radiance = radiance; es.dist = dist;
    es.pdf = luminance(Le) * dist * dist / (frameUBO.totalEmissivePower * cosL);
    es.valid = es.pdf > 0.0;
    return es;
}


// Ray payload: the slimmed HitInfo (filled by closesthit / hit=false on miss),
// plus the rng channel any-hit advances for stochastic-alpha, and the bounce
// index closesthit reads for material LOD.
struct RayPayload {
    HitInfo hit;
    uint    rng;
    uint    bounce;
};


[shader("anyhit")]
void anyHitMain(inout RayPayload pay, BuiltInTriangleIntersectionAttributes attr) {
    ShadowEntityInfo info = entityInfo[InstanceID() + GeometryIndex()];
    PbrMaterial      mat  = materials[info.materialIndex];

    // Opaque / transmissive-opaque (flags==0) → accept; closesthit decides refract.
    bool isMask  = (mat.flags & 1u) != 0u;
    bool isBlend = (mat.flags & 4u) != 0u;
    if (!isMask && !isBlend) return;

    float alpha = computeAlpha(mat, info.indexOffset, PrimitiveIndex(), attr.barycentrics);
    bool accept = true;
    if (isMask)  accept = alpha >= mat.alphaCutoff;          // MASK
    if (isBlend) accept = rand(pay.rng) < alpha;             // BLEND stochastic (advances payload rng)
    if (!accept) IgnoreHit();
}

[shader("miss")]
void missMain(inout RayPayload pay) {
    pay.hit.hit = false;   // raygen samples the env using its tracked ray direction
}

// Object-space point → world via the full per-geometry 3x4 (rows carry the
// translation in .w). Used to derive the hit position in the SER path, where
// RayTCurrent isn't available in raygen.
float3 xformPoint(ShadowEntityInfo info, float3 p) {
    return float3(dot(info.xform0.xyz, p) + info.xform0.w,
                  dot(info.xform1.xyz, p) + info.xform1.w,
                  dot(info.xform2.xyz, p) + info.xform2.w);
}

// Shared hit-shading prep: fetches geometry + material at the hit and returns a
// fully-populated HitInfo. Used by the closest-hit shader (non-SER path) and,
// under USE_SER, inline in raygen after a HitObject query — the Slang 2026.1
// HitObject.Invoke lowering is broken, and GetRayTCurrent isn't available in
// raygen, so the SER path queries the hit identity and shades inline here. The
// world hit position is computed from the world-transformed triangle (not
// ray·t), so it's identical in both paths and stage-independent.
HitInfo fillHit(ShadowEntityInfo info, uint primIdx, float2 bary,
                float3 rayOrigin, float3 rayDir, bool lowDetail) {
    HitInfo h = (HitInfo)0;
    h.hit    = true;
    h.matidx = info.materialIndex;

    float3 baryW = float3(1.0 - bary.x - bary.y, bary.x, bary.y);
    uint i0 = fetchTriangleIndex(globalIndices, info.indexOffset, primIdx, 0u);
    uint i1 = fetchTriangleIndex(globalIndices, info.indexOffset, primIdx, 1u);
    uint i2 = fetchTriangleIndex(globalIndices, info.indexOffset, primIdx, 2u);

    float3 op0 = fetchVertexPos(globalVertices, i0);
    float3 op1 = fetchVertexPos(globalVertices, i1);
    float3 op2 = fetchVertexPos(globalVertices, i2);

    // World hit position from the transformed triangle; hitT along the unit ray.
    float3 wp0 = xformPoint(info, op0);
    float3 wp1 = xformPoint(info, op1);
    float3 wp2 = xformPoint(info, op2);
    h.worldPos = wp0 * baryW.x + wp1 * baryW.y + wp2 * baryW.z;
    h.hitT     = length(h.worldPos - rayOrigin);

    float2 uv = fetchVertexUV(globalVertices, i0) * baryW.x
              + fetchVertexUV(globalVertices, i1) * baryW.y
              + fetchVertexUV(globalVertices, i2) * baryW.z;

    float3 nObj = fetchVertexNormal(globalVertices, i0) * baryW.x
                + fetchVertexNormal(globalVertices, i1) * baryW.y
                + fetchVertexNormal(globalVertices, i2) * baryW.z;
    float3 ngObj = normalize(cross(op1 - op0, op2 - op0));

    float4 t0 = fetchVertexTangent(globalVertices, i0);
    float4 t1 = fetchVertexTangent(globalVertices, i1);
    float4 t2 = fetchVertexTangent(globalVertices, i2);
    float3 tObj = t0.xyz * baryW.x + t1.xyz * baryW.y + t2.xyz * baryW.z;
    float  bitangentSign = t0.w;

    // Object → world for directions: 3x3 of the per-geometry transform (identity
    // instance transform can't do it — the cluster BLAS baked positions).
    float3x3 obj2worldR = entityNormalMatrix(info);
    h.normal          = normalize(mul(obj2worldR, nObj));
    h.geometricNormal = normalize(mul(obj2worldR, ngObj));
    float3 tangent    = normalize(mul(obj2worldR, tObj));

    h.frontFace = dot(h.geometricNormal, rayDir) < 0.0;
    if (!h.frontFace) { h.normal = -h.normal; h.geometricNormal = -h.geometricNormal; }

    PbrMaterial mat = materials[h.matidx];
    SamplerState s  = samplers[0];

    h.baseColor = mat.baseColorFactor;
    if (mat.baseColorTex >= 0)
        h.baseColor *= textures[NonUniformResourceIndex(mat.baseColorTex)].SampleLevel(s, uv, 0);

    h.metallic  = mat.metallicFactor;
    h.roughness = mat.roughnessFactor;
    if (mat.physicalDescriptorTex >= 0 && !lowDetail) {
        float2 mr = textures[NonUniformResourceIndex(mat.physicalDescriptorTex)].SampleLevel(s, uv, 0).bg;
        h.metallic  *= mr.x;
        h.roughness *= mr.y;
    }

    h.emissive = mat.emissiveFactor;
    if (mat.emissiveTex >= 0)
        h.emissive *= textures[NonUniformResourceIndex(mat.emissiveTex)].SampleLevel(s, uv, 0).rgb;

    if (mat.normalTex >= 0 && !lowDetail) {
        float3 nm = textures[NonUniformResourceIndex(mat.normalTex)].SampleLevel(s, uv, 0).xyz * 2.0 - 1.0;
        float3 T  = normalize(tangent);
        float3 B  = cross(h.normal, T) * bitangentSign;
        h.normal  = normalize(nm.x * T + nm.y * B + nm.z * h.normal);
    }

    h.transmission = mat.transmissionFactor;
    if (mat.transmissionTex >= 0 && !lowDetail)
        h.transmission *= textures[NonUniformResourceIndex(mat.transmissionTex)].SampleLevel(s, uv, 0).r;
    h.ior = mat.ior;

    float f0_eta = (mat.ior - 1.0) / (mat.ior + 1.0);
    h.F0 = f0_eta * f0_eta;

    h.clearcoat = mat.clearcoatFactor;
    if (mat.clearcoatTex >= 0 && !lowDetail)
        h.clearcoat *= textures[NonUniformResourceIndex(mat.clearcoatTex)].SampleLevel(s, uv, 0).r;
    h.clearcoatRoughness = mat.clearcoatRoughnessFactor;
    if (mat.clearcoatRoughnessTex >= 0 && !lowDetail)
        h.clearcoatRoughness *= textures[NonUniformResourceIndex(mat.clearcoatRoughnessTex)].SampleLevel(s, uv, 0).g;
    h.clearcoatRoughness = max(h.clearcoatRoughness, 0.045);
    h.roughness          = max(h.roughness,          0.045);
    h.clearcoatNormal    = h.normal;

    return h;
}

[shader("closesthit")]
void closestHitMain(inout RayPayload pay, BuiltInTriangleIntersectionAttributes attr) {
    ShadowEntityInfo info = entityInfo[InstanceID() + GeometryIndex()];
    pay.hit = fillHit(info, PrimitiveIndex(), attr.barycentrics,
                      WorldRayOrigin(), WorldRayDirection(),
                      pay.bounce >= MATERIAL_LOD_BOUNCE);
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

    RayPayload pay;
    uint cap = min(MAX_BOUNCES, frameUBO.bounceCap);
    for (uint b = 0u; b < cap; ++b) {
        pay.bounce  = b;
        pay.rng     = rngState;
        pay.hit.hit = false;
#ifdef USE_SER
        // Shader Execution Reordering: record the hit (runs traversal + any-hit),
        // regroup warps by the hit so the shading lanes that follow are coherent,
        // then query the hit identity and shade INLINE via fillHit. This is the
        // getter-based SER pattern (HitObject.Invoke is broken in Slang 2026.1 and
        // GetRayTCurrent isn't available in raygen). The reorder collapses the
        // path-/material-divergence that strands ~2/3 of each warp's lanes.
        // NV-only (Ada-class); the non-SER TraceRay variant is compiled otherwise.
        HitObject ho = HitObject.TraceRay(tlas, RAY_FLAG_NONE, 0xFF, 0u, 0u, 0u, ray, pay);
        // Material-coherence hint: bucket the reorder by material in addition to
        // hit, so warps regroup by what they're about to fetch. 8 bits = 256
        // buckets, plenty for typical scenes. Miss → bucket 0 (harmless).
        uint matHint = ho.IsHit()
            ? entityInfo[ho.GetInstanceID() + ho.GetGeometryIndex()].materialIndex
            : 0u;
        ReorderThread(ho, matHint, 8u);
        if (ho.IsHit()) {
            ShadowEntityInfo info = entityInfo[ho.GetInstanceID() + ho.GetGeometryIndex()];
            float2 bary = ho.GetAttributes<BuiltInTriangleIntersectionAttributes>().barycentrics;
            pay.hit = fillHit(info, ho.GetPrimitiveIndex(), bary,
                              ray.Origin, ray.Direction, b >= MATERIAL_LOD_BOUNCE);
        } else {
            pay.hit.hit = false;
        }
#else
        TraceRay(tlas, RAY_FLAG_NONE, 0xFF, 0u, 0u, 0u, ray, pay);
#endif
        rngState = pay.rng;            // any-hit advanced it (BLEND)
        HitInfo hit = pay.hit;

        if (!hit.hit) {
            float3 env  = envCube.SampleLevel(ray.Direction, 0.0).rgb;
            float3 envC = throughput * env * frameUBO.scaleIBLAmbient;
            float  m    = max(max(envC.r, envC.g), envC.b);
            if (m > FIREFLY_CLAMP) envC *= FIREFLY_CLAMP / m;
            radiance += envC;
            break;
        }

        if (any(sigmaA > 0.0)) throughput *= exp(-sigmaA * hit.hitT);

        // Emission pickup, MIS-weighted against emissive NEE.
        float misEm = 1.0;
        if (!prevWasDelta && frameUBO.emissiveTriCount > 0u && frameUBO.totalEmissivePower > 0.0) {
            float lumLe = luminance(materials[hit.matidx].emissiveFactor);
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

        // NEE — combined analytic + emissive, single shadow ray (see PTCompute).
        if (ENABLE_NEE) {
            bool haveAnalytic = frameUBO.lightCount > 0u;
            bool haveEmissive = frameUBO.emissiveTriCount > 0u;
            if (haveAnalytic || haveEmissive) {
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

        if (any(isnan(throughput)) || any(isinf(throughput))) break;

        if (b >= MIN_BOUNCES && RUSSIAN_ROULETTE) {
            float rr = clamp(luminance(throughput), 0.05, 0.95);
            if (rand(rngState) > rr) break;
            throughput /= rr;
        }

        if (bs.transmit) {
            sigmaA = hit.frontFace ? mediumSigmaA(materials[hit.matidx]) : float3(0.0);
        }

        float3 offsetN = bs.transmit ? -hit.geometricNormal : hit.geometricNormal;
        ray.Origin    = hit.worldPos + offsetN * 1e-3;
        ray.Direction = bs.wi;
        ray.TMin      = 0.0;
        ray.TMax      = INFINITY;
    }

    if (any(isnan(radiance)) || any(isinf(radiance))) radiance = float3(0.0);

    if (frameUBO.resetAccum != 0u) accumulator[pixel] = float4(radiance, 1.0);
    else                           accumulator[pixel] += float4(radiance, 1.0);

    float4 a = accumulator[pixel];
    outColor[pixel] = float4(a.rgb / max(a.a, 1.0), 1.0);
}