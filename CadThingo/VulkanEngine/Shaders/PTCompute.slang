#include "PbrUtils.slang"
#include "PTUtils.slang"

//frame uniform
struct PathFrameUBO {
    float4x4 invView;
    float4x4 invProj;
    float4 camPos;
    uint frameIndex;
    uint bounceCap;
    uint lightCount;
    uint resetAccum;//0 - keep, 1 = overwrite
    float2 screenSize;
    float fov;
    float tanHalfFov;
    float prefilteredCubeMipLevels;
    float scaleIBLAmbient;
    float focusDistance;
    float aperture;
    float paniniDistance = 1.0f;
    float verticalCompression = 0.0f;
};

//set 0 per-frame IO
//RGBA32F intentionally, R16F starves dark regions after ~64 samples
[[vk::binding(0, 0)]] ConstantBuffer<PathFrameUBO> frameUBO;
[[vk::binding(1, 0)]] StructuredBuffer<PbrLight>         lights;
[[vk::binding(2, 0)]] RaytracingAccelerationStructure    tlas;
[[vk::binding(3, 0)]] StructuredBuffer<ShadowEntityInfo> entityInfo;//indexes into buffer stored in tlas
[[vk::binding(4, 0)]] RWTexture2D<float4> accumulator;
[[vk::binding(5, 0)]] RWTexture2D<float4> outColor;

//Set 1 scene
struct ShadowEntityInfo {
    uint indexOffset;
    uint materialIndex;
    uint flags;
    uint _pad0;
};
//primitives
[[vk::binding(1, 1)]] ByteAddressBuffer                  globalVertices;
[[vk::binding(2, 1)]] ByteAddressBuffer                  globalIndices;

//Set 2 bindless textures + materials matches resource material dsl in resource manager
[[vk::binding(0, 2)]] StructuredBuffer<PbrMaterial>      materials;
[[vk::binding(1, 2)]] StructuredBuffer<InstanceData>     instances; //unused 
[[vk::binding(2, 2)]] Texture2D textures[];
[[vk::binding(3, 2)]] SamplerState samplers[];

//Set 3 IBL + Lights 
//Sampled on miss  (sky contribution) and clamped fallback 
[[vk::binding(0, 3)]] SamplerCube irradianceCube;
[[vk::binding(1, 3)]] SamplerCube prefilteredCube;
[[vk::binding(2, 3)]] Sampler2D brdfLUT;

//camera mode enum
enum CameraMode : uint{
    PINHOLE = 0u,
    THIN_LENS = 1u,
    PANINI = 2u,
    FISHEYE = 3u,
    ORTHOGRAPHIC = 4u,
}
//Specialization constants
[[vk::constant_id(0)]] const bool ENABLE_NEE       = true;   // next event estimation
[[vk::constant_id(1)]] const bool RUSSIAN_ROULETTE = true;   // path termination
[[vk::constant_id(2)]] const uint MAX_BOUNCES      = 8u;     // hard cap; ubo.bounceCap clamps below
[[vk::constant_id(3)]] const uint MODE             = 0u;     // 0 = PINHOLE; CameraMode enum values

// Geometry fetch helpers (mirrors PbrShader.slang) 
//Vertex layout Position(12) Normal(12) + TexCoord(8) + Tangent(16) = 48
static const uint VERTEX_STRIDE = 48u;
static const uint VERTEX_POS_OFFSET = 0u;
static const uint VERTEX_NORMAL_OFFSET = 12u;
static const uint VERTEX_UV_OFFSET = 24u;
static const uint VERTEX_TANGENT_OFFSET = 32u;

float3 fetchVertexPos (uint vi) {return asfloat(globalVertices.Load3(vi * VERTEX_STRIDE + VERTEX_POS_OFFSET));}
float3 fetchVertexNormal (uint vi) {return asfloat(globalVertices.Load3(vi * VERTEX_STRIDE + VERTEX_NORMAL_OFFSET));}
float2 fetchVertexUV (uint vi) {return asfloat(globalVertices.Load2(vi * VERTEX_STRIDE + VERTEX_UV_OFFSET));}
float4 fetchVertexTangent (uint vi) {return asfloat(globalVertices.Load4(vi * VERTEX_STRIDE + VERTEX_TANGENT_OFFSET));}

uint fetchTriangleIndex(uint indexOffset, uint primIdx, uint corner) {
    return globalIndices.Load((indexOffset + primIdx * 3u + corner) * 4u);
}

RayDesc generatePrimaryRay(float2 ndc, inout uint rngState) {
    if (MODE == (uint)CameraMode.PINHOLE) {
        return generatePinholeRay(ndc, frameUBO.tanHalfFov);
    } else if (MODE == (uint)CameraMode.THIN_LENS) {
        // Thin lens needs two uniform [0,1] samples for the disk; advance the
        // rng twice so different lanes don't end up correlated.
        float2 lensUV = float2(rand(rngState), rand(rngState));
        return generateThinLensRay(ndc, frameUBO.tanHalfFov, lensUV,
                                   frameUBO.focusDistance, frameUBO.aperture);
    } else if (MODE == (uint)CameraMode.PANINI) {
        return generatePaniniRay(ndc, frameUBO.fov, frameUBO.paniniDistance,
                                 frameUBO.verticalCompression);
    } else if (MODE == (uint)CameraMode.FISHEYE) {
        return generateFisheyeRay(ndc, frameUBO.fov);
    }
    // ORTHOGRAPHIC + unknown — sentinel ray (TMax < 0); main() skips it.
    return RayDesc(float3(0.0), 0.0, float3(0.0), RAY_INVALID_TMAX);
}

float2 computeUV(uint indexOffset, uint primIdx, float2 bary) {
    uint i0 = fetchTriangleIndex(indexOffset, primIdx, 0u);
    uint i1 = fetchTriangleIndex(indexOffset, primIdx, 1u);
    uint i2 = fetchTriangleIndex(indexOffset, primIdx, 2u);
    float2 uv0 = fetchVertexUV(i0);
    float2 uv1 = fetchVertexUV(i1);
    float2 uv2 = fetchVertexUV(i2);
    float  w0  = 1.0 - bary.x - bary.y;
    float2 uv  = uv0 * w0 + uv1 * bary.x + uv2 * bary.y;
    return uv;
}

float computeAlpha(PbrMaterial mat, uint indexOffset, uint primIdx, float2 barycentrics) {
    int baseColorIdx = mat.baseColorTex;
    if (baseColorIdx < 0 || mat.flags == 0) { return 1.0f; }
    float2 uv = computeUV(indexOffset, primIdx, barycentrics);
    return textures[NonUniformResourceIndex(baseColorIdx)].Sample(samplers[0], uv).a * mat.baseColorFactor.a;
}


// ─── Trace + shade resolve ────────────────────────────────────────────────
// One call per bounce.  Walks the alpha-test Proceed loop for non-opaque
// candidates, then on a committed hit fetches geometry + material and
// returns a fully-populated HitInfo ready for BSDF math.  On miss, returns
// HitInfo with `hit == false` and the rest zero-initialized.
// `rngState` is plumbed in for the BLEND stochastic-transparency coin flip
// in the Proceed loop — pure-opaque scenes leave it untouched.
HitInfo traceRay(RayDesc ray, inout uint rngState) {
    HitInfo h        = (HitInfo)0;
    h.ray            = ray;
    h.hit            = false;

    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(tlas, RAY_FLAG_NONE, 0xFF, ray);

    // Alpha-test loop.  TLAS instances flagged ForceNoOpaque (MASK / BLEND /
    // KHR_materials_transmission > 0, set on the CPU side in RebuildTlas)
    // ride through Proceed; pure-opaque hits skip it entirely. Transmissive-
    // OPAQUE candidates fall through to `accept = true` so sampleBsdf can
    // do the actual refract-vs-reflect decision based on Fresnel + IOR.
    while (q.Proceed()) {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
            uint candInst        = q.CandidateInstanceID();
            ShadowEntityInfo cei = entityInfo[candInst];
            PbrMaterial      cmat = materials[cei.materialIndex];
            float alpha = computeAlpha(cmat, cei.indexOffset,
                                       q.CandidatePrimitiveIndex(),
                                       q.CandidateTriangleBarycentrics());
            bool accept = true;
            if ((cmat.flags & 1u) != 0u) accept = alpha >= cmat.alphaCutoff;   // MASK
            // BLEND: stochastic transparency. Each ray samples the alpha as a
            // Bernoulli — commit with P=alpha, pass through with P=1-alpha.
            // Across the accumulator this converges to the correct expected
            // coverage and gives bushes / decals proper soft edges.
            if ((cmat.flags & 4u) != 0u) accept = rand(rngState) < alpha;
            if (accept) q.CommitNonOpaqueTriangleHit();
        }
    }

    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
        return h;   // miss — h.hit stays false, caller samples env / breaks
    }

    // ── Hit was committed.  Resolve geometry first, material second. ──────
    h.hit      = true;
    h.instID   = q.CommittedInstanceID();
    h.hitT     = q.CommittedRayT();
    h.worldPos = ray.Origin + ray.Direction * h.hitT;

    uint   primIdx = q.CommittedPrimitiveIndex();
    float2 bary    = q.CommittedTriangleBarycentrics();
    float3 baryW   = float3(1.0 - bary.x - bary.y, bary.x, bary.y);

    ShadowEntityInfo info = entityInfo[h.instID];
    h.matidx              = info.materialIndex;

    // Triangle vertex indices into globalVertices.
    uint i0 = fetchTriangleIndex(info.indexOffset, primIdx, 0u);
    uint i1 = fetchTriangleIndex(info.indexOffset, primIdx, 1u);
    uint i2 = fetchTriangleIndex(info.indexOffset, primIdx, 2u);

    // UV is needed for every material sample below, so do it first.
    h.uv = fetchVertexUV(i0) * baryW.x
         + fetchVertexUV(i1) * baryW.y
         + fetchVertexUV(i2) * baryW.z;

    // Object-space normals (smooth) + positions (for the flat / geometric N).
    float3 nObj  = fetchVertexNormal(i0) * baryW.x
                 + fetchVertexNormal(i1) * baryW.y
                 + fetchVertexNormal(i2) * baryW.z;

    float3 p0    = fetchVertexPos(i0);
    float3 p1    = fetchVertexPos(i1);
    float3 p2    = fetchVertexPos(i2);
    float3 ngObj = normalize(cross(p1 - p0, p2 - p0));

    // glTF vertex tangent: xyz = tangent, w = bitangent sign (±1).  Sign is
    // constant across the triangle in practice; take from corner 0.
    float4 t0   = fetchVertexTangent(i0);
    float4 t1   = fetchVertexTangent(i1);
    float4 t2   = fetchVertexTangent(i2);
    float3 tObj = t0.xyz * baryW.x + t1.xyz * baryW.y + t2.xyz * baryW.z;
    h.bitangentSign = t0.w;

    // Object → world via the TLAS instance transform.  Upper-3x3 is enough
    // for directions; assumes uniform / near-uniform scale.  For genuinely
    // non-uniform scales you'd want the inverse-transpose of the 3x3 for the
    // normal — add when you actually see skew.
    float3x4 obj2world  = q.CommittedObjectToWorld3x4();
    float3x3 obj2worldR = (float3x3)obj2world;

    h.normal          = normalize(mul(obj2worldR, nObj));
    h.geometricNormal = normalize(mul(obj2worldR, ngObj));
    h.tangent         = normalize(mul(obj2worldR, tObj));

    // Front-face test against the GEOMETRIC normal (shading normal can lie
    // due to interpolation / normal maps).  Flip both so downstream BSDF code
    // can treat dot(N, V) > 0 as invariant.
    h.frontFace = dot(h.geometricNormal, ray.Direction) < 0.0;
    if (!h.frontFace) {
        h.normal          = -h.normal;
        h.geometricNormal = -h.geometricNormal;
    }

    // ── Material sample. ──────────────────────────────────────────────────
    PbrMaterial  mat = materials[h.matidx];
    SamplerState s   = samplers[0];

    h.baseColor = mat.baseColorFactor;
    if (mat.baseColorTex >= 0)
        h.baseColor *= textures[NonUniformResourceIndex(mat.baseColorTex)].Sample(s, h.uv);

    h.metallic  = mat.metallicFactor;
    h.roughness = mat.roughnessFactor;
    if (mat.physicalDescriptorTex >= 0) {
        float2 mr   = textures[NonUniformResourceIndex(mat.physicalDescriptorTex)].Sample(s, h.uv).bg;
        h.metallic  *= mr.x;
        h.roughness *= mr.y;
    }

    h.emissive = mat.emissiveFactor;
    if (mat.emissiveTex >= 0)
        h.emissive *= textures[NonUniformResourceIndex(mat.emissiveTex)].Sample(s, h.uv).rgb;

    // Normal-map perturbation.  Done AFTER the front-face flip so the TBN
    // frame matches the (possibly-flipped) shading normal.
    if (mat.normalTex >= 0) {
        float3 nm = textures[NonUniformResourceIndex(mat.normalTex)].Sample(s, h.uv).xyz * 2.0 - 1.0;
        float3 T  = normalize(h.tangent);
        float3 B  = cross(h.normal, T) * h.bitangentSign;
        h.normal  = normalize(nm.x * T + nm.y * B + nm.z * h.normal);
    }

    // ── Classification + transmission / IOR from KHR_materials_transmission
    //    + KHR_materials_ior. transmissionTex .r multiplies the factor; full
    //    glass = factor 1, opaque PBR = factor 0 (the default).
    h.alphaMode    = (int)(mat.flags & 0x5u);   // bit 0 = MASK, bit 2 = BLEND
    h.transmission = mat.transmissionFactor;
    if (mat.transmissionTex >= 0) {
        h.transmission *= textures[NonUniformResourceIndex(mat.transmissionTex)].Sample(s, h.uv).r;
    }
    h.ior     = mat.ior;
    h.isGlass = h.transmission > 0.001;
    h.isLiquid = false;

    // F0 from Schlick's IOR formula: ((η - 1) / (η + 1))². Reproduces the
    // historical 0.04 dielectric default at IOR=1.5 (glass) and gives correct
    // values everywhere else — water 1.33 → ~0.02, diamond 2.4 → ~0.17.
    float f0_eta = (mat.ior - 1.0) / (mat.ior + 1.0);
    h.F0      = f0_eta * f0_eta;
    h.ambient = float3(0.0);

    return h;
}
// ─── Shadow ray (NEE visibility) ──────────────────────────────────────────
// ACCEPT_FIRST_HIT_AND_END_SEARCH: traversal stops on the first committed
// hit, which is all NEE needs. MASK passes through alpha-test; BLEND
// (glass-tagged) geometry is treated as fully transparent for shadow
// purposes — matches the deferred shadow-ray semantics.
// Same stochastic-transparency contract as traceRay: a leaf card with
// alpha=0.5 blocks the shadow ray ~50% of the time, so bushes cast a
// correctly-graded soft shadow as samples accumulate.
bool isOccluded(float3 origin, float3 dir, float tMax, inout uint rngState) {
    RayDesc r;
    r.Origin    = origin;
    r.Direction = dir;
    r.TMin      = 0.0;
    r.TMax      = tMax;

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    q.TraceRayInline(tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, r);
    while (q.Proceed()) {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
            uint candInst        = q.CandidateInstanceID();
            ShadowEntityInfo cei = entityInfo[candInst];
            PbrMaterial      cmat = materials[cei.materialIndex];
            float alpha = computeAlpha(cmat, cei.indexOffset,
                                       q.CandidatePrimitiveIndex(),
                                       q.CandidateTriangleBarycentrics());
            // Combined alpha + transmission gate. Reduce the candidate to an
            // effective opacity, then sample as a Bernoulli:
            //   passAlpha   = alpha test (MASK / BLEND / OPAQUE)
            //   blockProb   = passAlpha * (1 - transmission)
            // OPAQUE-tagged glass (alpha=1, transmission=1) → block prob 0,
            // so light passes through. Tinted glass (transmission=0.7) blocks
            // 30 % of shadow rays. Bushes (BLEND alpha=0.5, transmission=0)
            // reduce to the leaf-card behaviour: block with P=alpha.
            float transmittance = cmat.transmissionFactor;
            if (cmat.transmissionTex >= 0) {
                float2 uv = computeUV(cei.indexOffset,
                                      q.CandidatePrimitiveIndex(),
                                      q.CandidateTriangleBarycentrics());
                transmittance *= textures[NonUniformResourceIndex(cmat.transmissionTex)].Sample(samplers[0], uv).r;
            }

            float passAlpha = 1.0;
            if ((cmat.flags & 1u) != 0u) passAlpha = (alpha >= cmat.alphaCutoff) ? 1.0 : 0.0;  // MASK
            else if ((cmat.flags & 4u) != 0u) passAlpha = alpha;                                // BLEND
            // OPAQUE branch leaves passAlpha = 1 — transmission alone gates it.

            float blockProb = passAlpha * (1.0 - transmittance);
            if (rand(rngState) < blockProb) q.CommitNonOpaqueTriangleHit();
        }
    }
    return q.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
}

// ─── RIS light pick ────────────────────────────────────────────────────────
// Reservoir-style RIS over M=RIS_LIGHT_SAMPLE_COUNT uniform candidates.
//   p_source = 1 / lightCount     (uniform pick)
//   p_target = luminance(unshadowed_radiance) * NdotL
//   w_i      = p_target / p_source = p_target * lightCount
//   pick s with prob w_s / sum(w)
//   estimator weight W = sum(w) / (M * p_target(s))
// Caller multiplies the BRDF·radiance·cosθ contribution by W. Shadow ray is
// run ONCE on the chosen light (not on every candidate) — that trades a bit
// of variance for ~M× fewer rays, which is the right tradeoff at M=8.
struct LightPick {
    LightSample sample;
    float       weight;   // W; multiply into the direct-light estimator
    bool        valid;
};

static const uint RIS_LIGHT_SAMPLE_COUNT = 8u;

LightPick sampleLightRIS(inout uint rngState, float3 worldPos, float3 N) {
    LightPick best;
    best.sample = (LightSample)0;
    best.weight = 0.0;
    best.valid  = false;

    if (frameUBO.lightCount == 0u) return best;

    float wSum         = 0.0;
    float bestPTarget  = 0.0;
    LightSample bestLs = (LightSample)0;

    for (uint i = 0u; i < RIS_LIGHT_SAMPLE_COUNT; ++i) {
        uint idx = min(frameUBO.lightCount - 1u,
                       uint(rand(rngState) * float(frameUBO.lightCount)));
        PbrLight cand = lights[idx];

        LightSample ls;
        if (!evaluateLight(cand, worldPos, ls)) continue;

        float NdotL = max(dot(N, ls.L), 0.0);
        if (NdotL <= 0.0) continue;

        float pTarget = luminance(ls.radiance) * NdotL;
        float w       = pTarget * float(frameUBO.lightCount);
        wSum         += w;

        // Reservoir replacement: P(adopt) = w_i / wSum guarantees uniform
        // selection proportional to w across the M samples.
        if (rand(rngState) * wSum < w) {
            bestLs       = ls;
            bestPTarget  = pTarget;
        }
    }

    if (bestPTarget <= 0.0 || wSum <= 0.0) return best;

    best.sample = bestLs;
    best.weight = wSum / (float(RIS_LIGHT_SAMPLE_COUNT) * bestPTarget);
    best.valid  = true;
    return best;
}


//Entry point
//8x8 tile keeps subgroup occupancy high on both nvidia and amd
[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID, uint3 dim : SV_GroupThreadID){
    // pixel reject + RNG seed
    uint rngState = initRNG(gid.xy, dim.xy, frameUBO.frameIndex);

    // Sub-pixel jitter inside [0,1)² each sample — converges to a box-filter
    // AA over the accumulator. Without this every sample fires through the
    // pixel center and aliasing is baked in no matter how many samples land.
    // Box is the simplest filter; swap to tent / gaussian (e.g. Mitchell)
    // here if you want softer edges at the cost of slightly blurrier output.
    float2 jitter = float2(rand(rngState), rand(rngState));
    float2 px     = float2(gid.xy) + jitter;
    float2 ndc    = (px / frameUBO.screenSize) * 2.0f - 1.0f;
    ndc.x        *= frameUBO.screenSize.x / frameUBO.screenSize.y;

    // primary ray: camera-space helpers in PTUtils, then lift to world with
    // invView so origin lands at the actual camera and forward points along
    // -view. transformRayToWorld preserves the fisheye TMax<0 sentinel.
    RayDesc ray = transformRayToWorld(generatePrimaryRay(ndc, rngState),
                                      frameUBO.invView);

    float3 radiance   = float3(0.0);
    float3 throughput = float3(1.0);

    // Fisheye corner cull (sentinel from generateFisheyeRay) — emit zero.
    if (ray.TMax < 0.0) {
        if (frameUBO.resetAccum != 0u) accumulator[gid.xy] = float4(0.0, 0.0, 0.0, 1.0);
        else                           accumulator[gid.xy] += float4(0.0, 0.0, 0.0, 1.0);
        float4 a = accumulator[gid.xy];
        outColor[gid.xy] = float4(a.rgb / max(a.a, 1.0), 1.0);
        return;
    }

    // Per-bounce contribution cap. Tiny BRDF / pdf ratios (sharp-specular
    // grazing hits, near-degenerate refractions) produce one-sample radiances
    // in the thousands — those "fireflies" dominate the accumulator and at
    // worst overflow to Inf, which then NaN-poisons the pixel for every
    // subsequent sample. Clamping the per-bounce contribution by max channel
    // is the standard fix; biased downward at very low sample counts, but
    // by the time you've got a few hundred samples it converges back.
    const float FIREFLY_CLAMP = 16.0;

    uint cap = min(MAX_BOUNCES, frameUBO.bounceCap);
    for (uint b = 0u; b < cap; ++b) {
        HitInfo hit = traceRay(ray, rngState);

        // ── Miss: sample environment (mip 0 = sharpest) and terminate. ──
        if (!hit.hit) {
            float3 env  = prefilteredCube.SampleLevel(ray.Direction, 0.0).rgb;
            float3 envC = throughput * env * frameUBO.scaleIBLAmbient;
            float  m    = max(max(envC.r, envC.g), envC.b);
            if (m > FIREFLY_CLAMP) envC *= FIREFLY_CLAMP / m;
            radiance += envC;
            break;
        }

        // Always pick up emissive at the hit. With NEE this is technically
        // double-counted vs. the (not-yet-implemented) area-light sampling
        // for emissive geometry — fine until emissive meshes are sampled as
        // lights themselves.
        float3 emC = throughput * hit.emissive;
        float  me  = max(max(emC.r, emC.g), emC.b);
        if (me > FIREFLY_CLAMP) emC *= FIREFLY_CLAMP / me;
        radiance += emC;

        float3 V = -ray.Direction;   // toward camera/incoming ray's origin

        // ── NEE: pick a light, shadow-test, add f * Le * cosθ * W. ──
        if (ENABLE_NEE && frameUBO.lightCount > 0u) {
            LightPick pick = sampleLightRIS(rngState, hit.worldPos, hit.normal);
            if (pick.valid) {
                float NdotL = max(dot(hit.normal, pick.sample.L), 0.0);
                if (NdotL > 0.0) {
                    // Offset along the GEOMETRIC normal — shading normal can
                    // disagree with the surface and seed self-intersections.
                    float3 shadowOrigin = hit.worldPos + hit.geometricNormal * 1e-3;
                    float  tMax         = max(pick.sample.distance - 2e-3, 0.0);
                    bool   shadowed     = pick.sample.castShadows &&
                                          isOccluded(shadowOrigin, pick.sample.L, tMax, rngState);
                    if (!shadowed) {
                        float3 brdf = evaluatePbrBrdf(hit, V, pick.sample.L);
                        float3 neeC = throughput * brdf * pick.sample.radiance
                                                 * NdotL * pick.weight;
                        float  mn   = max(max(neeC.r, neeC.g), neeC.b);
                        if (mn > FIREFLY_CLAMP) neeC *= FIREFLY_CLAMP / mn;
                        radiance += neeC;
                    }
                }
            }
        }

        // ── BSDF sample for indirect. Split diffuse / GGX-VNDF sampler —
        //    Fresnel-weighted lobe pick keeps glossy surfaces actually finding
        //    their highlights instead of relying on the diffuse lobe to ever
        //    hit them by accident.
        BsdfSample bs = sampleBsdf(hit, V, rngState);
        if (!bs.valid) break;
        throughput *= bs.weight;

        // Belt-and-braces: throughput should never go non-finite, but if a
        // numerical edge case in refract / pdf math produces NaN/Inf here we
        // terminate the path. Without this the accumulator gets infected and
        // the pixel renders black for the rest of the session.
        if (any(isnan(throughput)) || any(isinf(throughput))) break;

        // ── Russian roulette after the first MIN_BOUNCES bounces. Floor at
        //    0.05 stops near-black throughput from killing the path before
        //    it can pick up an emissive / NEE contribution downstream.
        if (b >= MIN_BOUNCES && RUSSIAN_ROULETTE) {
            float rr = clamp(luminance(throughput), 0.05, 0.95);
            if (rand(rngState) > rr) break;
            throughput /= rr;
        }

        // Set up the next bounce. Refraction needs the origin on the *other*
        // side of the surface — otherwise the next traceRay would immediately
        // re-hit the same triangle from inside.
        float3 offsetN = bs.transmit ? -hit.geometricNormal : hit.geometricNormal;
        ray.Origin    = hit.worldPos + offsetN * 1e-3;
        ray.Direction = bs.wi;
        ray.TMin      = 0.0;
        ray.TMax      = INFINITY;
    }

    // Final NaN/Inf scrub. The per-bounce checks above should catch most cases,
    // but anything that slipped through gets zeroed here. Critical because the
    // accumulator is RWTexture<float4>: a single bad sample contaminates the
    // pixel for the rest of the session (NaN + anything = NaN, /n = NaN).
    if (any(isnan(radiance)) || any(isinf(radiance))) radiance = float3(0.0);

    // Progressive accumulation: .a is sample count. Reset path overwrites
    // (camera moved, mode flipped); steady-state path adds one sample.
    if (frameUBO.resetAccum != 0u) accumulator[gid.xy] = float4(radiance, 1.0);
    else                           accumulator[gid.xy] += float4(radiance, 1.0);

    float4 a = accumulator[gid.xy];
    outColor[gid.xy] = float4(a.rgb / max(a.a, 1.0), 1.0);
}