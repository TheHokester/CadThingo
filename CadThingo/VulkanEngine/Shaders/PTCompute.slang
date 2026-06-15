import PTUtils;   // pathtracer math; re-exports PbrUtils + CommonTypes



//set 0 per-frame IO
//RGBA32F intentionally, R16F starves dark regions after ~64 samples
[[vk::binding(0, 0)]] ConstantBuffer<PathFrameUBO> frameUBO;
[[vk::binding(1, 0)]] StructuredBuffer<PbrLight>         lights;
[[vk::binding(2, 0)]] RaytracingAccelerationStructure    tlas;
[[vk::binding(3, 0)]] StructuredBuffer<ShadowEntityInfo> entityInfo;//indexes into buffer stored in tlas
[[vk::binding(4, 0)]] RWTexture2D<float4> accumulator;
[[vk::binding(5, 0)]] RWTexture2D<float4> outColor;

// Emissive area lights. One record per world-space emissive triangle, built on
// the CPU in RebuildTlas. `emissiveAlias` is a Vose alias table over the same
// indices, weighted by area·luminance(Le), for O(1) power-proportional picks.
// EmissiveTri / AliasEntry layouts live in PTUtils (mirror EmissiveTriGpu /
// AliasEntryGpu in Renderer_Ray_Query.cs).
[[vk::binding(6, 0)]] StructuredBuffer<EmissiveTri> emissiveTris;
[[vk::binding(7, 0)]] StructuredBuffer<AliasEntry>  emissiveAlias;

//Set 1 scene — ShadowEntityInfo layout lives in CommonTypes.
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
// Full-res environment (512²); prefilteredCube is 128² GGX-convolved, so use
// envCube for sharp miss-ray background sampling.
[[vk::binding(3, 3)]] SamplerCube envCube;


//Specialization constants
[[vk::constant_id(0)]] const bool ENABLE_NEE       = true;   // next event estimation
[[vk::constant_id(1)]] const bool RUSSIAN_ROULETTE = true;   // path termination
[[vk::constant_id(2)]] const uint MAX_BOUNCES      = 8u;     // hard cap; ubo.bounceCap clamps below
[[vk::constant_id(3)]] const uint MODE             = (uint)CameraMode.PINHOLE;     // 0 = PINHOLE; CameraMode enum values

// Material LOD. On bounces >= this, traceRay skips the normal / metallic-
// roughness / clearcoat / transmission texture fetches and falls back to the
// material factors. Those fetches are the bulk of per-hit memory traffic (the
// dominant long-scoreboard stall), and indirect bounces carry little high-
// frequency material detail, so the quality cost is ~invisible. baseColor +
// emissive are always sampled (albedo tint + light contribution matter). Set to
// MAX_BOUNCES to disable; lower toward 1 to trade more quality for speed.
static const uint MATERIAL_LOD_BOUNCE = 2u;

// Vertex layout + fetch helpers (VERTEX_STRIDE, fetchVertexPos/Normal/UV/Tangent,
// fetchTriangleIndex) live in CommonTypes and take the ByteAddressBuffer as a
// parameter — call sites below pass globalVertices / globalIndices.



float2 computeUV(uint indexOffset, uint primIdx, float2 bary) {
    uint i0 = fetchTriangleIndex(globalIndices, indexOffset, primIdx, 0u);
    uint i1 = fetchTriangleIndex(globalIndices, indexOffset, primIdx, 1u);
    uint i2 = fetchTriangleIndex(globalIndices, indexOffset, primIdx, 2u);
    float2 uv0 = fetchVertexUV(globalVertices, i0);
    float2 uv1 = fetchVertexUV(globalVertices, i1);
    float2 uv2 = fetchVertexUV(globalVertices, i2);
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


// Trace + shade resolve
// One call per bounce.  Walks the alpha-test Proceed loop for non-opaque
// candidates, then on a committed hit fetches geometry + material and
// returns a fully-populated HitInfo ready for BSDF math.  On miss, returns
// HitInfo with `hit == false` and the rest zero-initialized.
// `rngState` is plumbed in for the BLEND stochastic-transparency coin flip
// in the Proceed loop — pure-opaque scenes leave it untouched.
HitInfo traceRay(RayDesc ray, inout uint rngState, uint bounce) {
    HitInfo h        = (HitInfo)0;
    h.hit            = false;
    // On deep bounces, skip the detail-texture fetches (see MATERIAL_LOD_BOUNCE).
    bool lowDetail = bounce >= MATERIAL_LOD_BOUNCE;

    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(tlas, RAY_FLAG_NONE, 0xFF, ray);

    // Alpha-test loop.  TLAS instances flagged ForceNoOpaque (MASK / BLEND /
    // KHR_materials_transmission > 0, set on the CPU side in RebuildTlas)
    // ride through Proceed; pure-opaque hits skip it entirely. Transmissive-
    // OPAQUE candidates fall through to `accept = true` so sampleBsdf can
    // do the actual refract-vs-reflect decision based on Fresnel + IOR.
    while (q.Proceed()) {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
            // Flat per-geometry record: instance custom index is the cluster's
            // geometry-block base, geometry index selects the primitive within it.
            ShadowEntityInfo cei = entityInfo[q.CandidateInstanceID() + q.CandidateGeometryIndex()];
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

    // Hit was committed.  Resolve geometry first, material second.
    h.hit      = true;
    h.hitT     = q.CommittedRayT();
    h.worldPos = ray.Origin + ray.Direction * h.hitT;

    uint   primIdx = q.CommittedPrimitiveIndex();
    float2 bary    = q.CommittedTriangleBarycentrics();
    float3 baryW   = float3(1.0 - bary.x - bary.y, bary.x, bary.y);

    // Flat per-geometry record index (cluster block base + geometry within it).
    // Local — only needed to fetch material/geometry below, not after traceRay.
    uint recordIndex      = q.CommittedInstanceID() + q.CommittedGeometryIndex();
    ShadowEntityInfo info = entityInfo[recordIndex];
    h.matidx              = info.materialIndex;

    // Triangle vertex indices into globalVertices.
    uint i0 = fetchTriangleIndex(globalIndices, info.indexOffset, primIdx, 0u);
    uint i1 = fetchTriangleIndex(globalIndices, info.indexOffset, primIdx, 1u);
    uint i2 = fetchTriangleIndex(globalIndices, info.indexOffset, primIdx, 2u);

    // UV is needed for every material sample below, so do it first. Local — the
    // bounce loop never reads it (sampleEmissive recomputes its own UV).
    float2 uv = fetchVertexUV(globalVertices, i0) * baryW.x
              + fetchVertexUV(globalVertices, i1) * baryW.y
              + fetchVertexUV(globalVertices, i2) * baryW.z;

    // Object-space normals (smooth) + positions (for the flat / geometric N).
    float3 nObj  = fetchVertexNormal(globalVertices, i0) * baryW.x
                 + fetchVertexNormal(globalVertices, i1) * baryW.y
                 + fetchVertexNormal(globalVertices, i2) * baryW.z;

    float3 p0    = fetchVertexPos(globalVertices, i0);
    float3 p1    = fetchVertexPos(globalVertices, i1);
    float3 p2    = fetchVertexPos(globalVertices, i2);
    float3 ngObj = normalize(cross(p1 - p0, p2 - p0));

    // Object → world. The cluster BLAS bakes positions to world space with an
    // IDENTITY instance transform, so CommittedObjectToWorld3x4 would NOT rotate
    // these object-space normals — use the per-geometry world transform carried
    // in the ShadowEntityInfo record instead. Upper-3x3 only; assumes uniform /
    // near-uniform scale (inverse-transpose would be needed for genuine skew).
    float3x3 obj2worldR = entityNormalMatrix(info);

    h.normal          = normalize(mul(obj2worldR, nObj));
    h.geometricNormal = normalize(mul(obj2worldR, ngObj));

    // Front-face test against the GEOMETRIC normal (shading normal can lie
    // due to interpolation / normal maps).  Flip both so downstream BSDF code
    // can treat dot(N, V) > 0 as invariant.
    h.frontFace = dot(h.geometricNormal, ray.Direction) < 0.0;
    if (!h.frontFace) {
        h.normal          = -h.normal;
        h.geometricNormal = -h.geometricNormal;
    }

    // Material sample.
    PbrMaterial  mat = materials[h.matidx];
    SamplerState s   = samplers[0];

    h.baseColor = mat.baseColorFactor;
    if (mat.baseColorTex >= 0)
        h.baseColor *= textures[NonUniformResourceIndex(mat.baseColorTex)].Sample(s, uv);

    h.metallic  = mat.metallicFactor;
    h.roughness = mat.roughnessFactor;
    if (mat.physicalDescriptorTex >= 0 && !lowDetail) {
        float2 mr   = textures[NonUniformResourceIndex(mat.physicalDescriptorTex)].Sample(s, uv).rg;
        h.metallic  *= mr.x;
        h.roughness *= mr.y;
    }

    h.emissive = mat.emissiveFactor;
    if (mat.emissiveTex >= 0)
        h.emissive *= textures[NonUniformResourceIndex(mat.emissiveTex)].Sample(s, uv).rgb;

    // Normal-map perturbation.  Done AFTER the front-face flip so the TBN
    // frame matches the (possibly-flipped) shading normal. Skipped on deep
    // bounces — indirect lighting barely registers the perturbation.
    if (mat.normalTex >= 0 && !lowDetail) {
        float3 nm = decodeTangentNormal(textures[NonUniformResourceIndex(mat.normalTex)].Sample(s, uv).rg);
        // Tangent from UV gradients instead of a per-vertex tangent fetch: reuses
        // the already-loaded object-space positions, trading vertex bandwidth for ALU.
        float2 uv0 = fetchVertexUV(globalVertices, i0);
        float2 uv1 = fetchVertexUV(globalVertices, i1);
        float2 uv2 = fetchVertexUV(globalVertices, i2);
        float2 d1  = uv1 - uv0, d2 = uv2 - uv0;
        float  det = d1.x * d2.y - d1.y * d2.x;
        if (abs(det) > 1e-12) {
            float  r    = 1.0 / det;
            float3 tObj = ((p1 - p0) * d2.y - (p2 - p0) * d1.y) * r;
            float3 bObj = ((p2 - p0) * d1.x - (p1 - p0) * d2.x) * r;
            float3 T    = normalize(mul(obj2worldR, tObj));
            T           = normalize(T - h.normal * dot(h.normal, T));   // orthogonalize vs shading N
            float  sgn  = dot(cross(h.normal, T), normalize(mul(obj2worldR, bObj))) < 0.0 ? -1.0 : 1.0;
            float3 B    = cross(h.normal, T) * sgn;
            h.normal    = normalize(nm.x * T + nm.y * B + nm.z * h.normal);
        }
    }

    // Transmission / IOR from KHR_materials_transmission + KHR_materials_ior.
    // transmissionTex .r multiplies the factor; full glass = factor 1, opaque
    // PBR = factor 0 (the default).
    h.transmission = mat.transmissionFactor;
    if (mat.transmissionTex >= 0 && !lowDetail) {
        h.transmission *= textures[NonUniformResourceIndex(mat.transmissionTex)].Sample(s, uv).r;
    }
    h.ior     = mat.ior;

    // F0 from Schlick's IOR formula: ((η - 1) / (η + 1))². Reproduces the
    // historical 0.04 dielectric default at IOR=1.5 (glass) and gives correct
    // values everywhere else — water 1.33 → ~0.02, diamond 2.4 → ~0.17.
    float f0_eta = (mat.ior - 1.0) / (mat.ior + 1.0);
    h.F0 = f0_eta * f0_eta;

    // KHR_materials_clearcoat
    // Clearcoat is a thin dielectric coat over the base — fixed IOR 1.5 per
    // spec, so its F0 is the canonical 0.04 (constant in the eval below).
    // glTF: clearcoatTexture uses R channel, clearcoatRoughnessTexture uses G.
    h.clearcoat = mat.clearcoatFactor;
    if (mat.clearcoatTex >= 0 && !lowDetail) {
        h.clearcoat *= textures[NonUniformResourceIndex(mat.clearcoatTex)].Sample(s, uv).r;
    }
    h.clearcoatRoughness = mat.clearcoatRoughnessFactor;
    if (mat.clearcoatRoughnessTex >= 0 && !lowDetail) {
        h.clearcoatRoughness *= textures[NonUniformResourceIndex(mat.clearcoatRoughnessTex)].Sample(s, uv).g;
    }
    // Clamp roughness to a small floor — GGX D(roughness=0) is 0 everywhere
    // except the mirror direction where it's 0/0 = NaN, so a literal zero
    // produces a coat that contributes nothing despite "full clearcoat".
    // 0.045 maps to GGX alpha ≈ 0.002, sharp-but-evaluable. Same floor on
    // the base lobe so a manually-zeroed roughness slider doesn't kill it.
    h.clearcoatRoughness = max(h.clearcoatRoughness, 0.045);
    h.roughness          = max(h.roughness,          0.045);
    // Clearcoat normal — for now reuse the shading normal. Real KHR clearcoat
    // supports a *separate* normal map for the coat (orange peel effect); add
    // when you have a test asset with one. Sampled separately because the
    // coat's TBN and the base's TBN differ.
    h.clearcoatNormal = h.normal;

    return h;
}
// Shadow ray (NEE visibility)
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
            // Flat per-geometry record: instance custom index is the cluster's
            // geometry-block base, geometry index selects the primitive within it.
            ShadowEntityInfo cei = entityInfo[q.CandidateInstanceID() + q.CandidateGeometryIndex()];
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

// RIS light pick
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

    // With a single light every candidate collapses to it, so M>1 just repeats
    // the same evaluateLight for an identical result — clamp to 1. Multi-light
    // scenes keep the full M: extra candidates improve target-proportional
    // coverage (lower variance), so we don't shrink M down to lightCount there.
    uint M = (frameUBO.lightCount == 1u) ? 1u : RIS_LIGHT_SAMPLE_COUNT;

    float wSum         = 0.0;
    float bestPTarget  = 0.0;
    LightSample bestLs = (LightSample)0;

    for (uint i = 0u; i < M; ++i) {
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
    best.weight = wSum / (float(M) * bestPTarget);
    best.valid  = true;
    return best;
}


// KHR_materials_volume absorption coefficient (Beer-Lambert).
//   σ_a = -ln(attenuationColor) / attenuationDistance
// attenuationColor is the color light becomes after one attenuationDistance of
// travel through the medium. The estimator below multiplies throughput by
// exp(-σ_a · segmentLength) over each segment spent inside the volume.
// White color (-ln 1 = 0) or the no-volume distance sentinel ⇒ σ_a = 0, so a
// thin pane (tiny interior path) and untagged glass absorb nothing — only a
// genuinely thick solid tints, which is the whole point.
float3 mediumSigmaA(PbrMaterial m) {
    if (m.attenuationDistance <= 0.0 || m.attenuationDistance >= 1e6) return float3(0.0);
    return -log(max(m.attenuationColor, float3(1e-4))) / m.attenuationDistance;
}

// Emissive area-light sample (NEE)
// Pick a triangle ∝ power via the alias table, then a uniform point on it.
// The returned pdf is the full solid-angle density: selection × area-pdf ×
// (dist² / cosθ_light). The triangle area cancels against the selection weight
// (p_select = area·lum(Le)/totalPower), leaving the closed form below — which
// is exactly what the BSDF-leg MIS recomputes from a hit, so the two strategies
// agree. Double-sided (abs cosθ) to match the unconditional emissive pickup.
struct EmissiveLightSample {
    float3 L;          // direction: shading point → sampled point on emitter
    float3 radiance;   // Le at the emitter (emissiveFactor; textures not sampled here)
    float  dist;
    float  pdf;        // solid-angle pdf, includes power-proportional selection
    bool   valid;
};

EmissiveLightSample sampleEmissive(inout uint rng, float3 P) {
    EmissiveLightSample es;
    es.L = float3(0.0); es.radiance = float3(0.0); es.dist = 0.0; es.pdf = 0.0; es.valid = false;

    uint n = frameUBO.emissiveTriCount;
    if (n == 0u || frameUBO.totalEmissivePower <= 0.0) return es;

    // Alias pick: bucket i, keep with prob, else jump to its alias.
    uint i = min(n - 1u, uint(rand(rng) * float(n)));
    AliasEntry a = emissiveAlias[i];
    uint idx = (rand(rng) < a.prob) ? i : a.alias;

    EmissiveTri t = emissiveTris[idx];
    float3 p0 = t.p0Area.xyz;
    float3 e1 = t.e1LeR.xyz;
    float3 e2 = t.e2LeG.xyz;
    float3 Le = float3(t.e1LeR.w, t.e2LeG.w, t.nLeB.w);   // emissiveFactor
    float3 nL = t.nLeB.xyz;

    // Uniform point on the triangle (sqrt warp keeps it area-uniform). b1/b2
    // are the barycentric weights on edges e1/e2 (vertices 1/2); vertex 0 gets
    // 1 - b1 - b2. Kept around so the emissive-texture UV below interpolates at
    // the exact sampled point.
    float u1 = rand(rng);
    float u2 = rand(rng);
    float su = sqrt(u1);
    float b1 = su * (1.0 - u2);
    float b2 = su * u2;
    float3 Q = p0 + e1 * b1 + e2 * b2;

    float3 toL  = Q - P;
    float  dist = length(toL);
    if (dist < 1e-4) return es;
    float3 L = toL / dist;

    float cosL = abs(dot(nL, L));
    if (cosL < 1e-4) return es;

    // Texture the emitted radiance to match the BSDF-leg pickup (hit.emissive =
    // emissiveFactor × emissiveTex). Refetch the triangle's UVs from the global
    // buffers and sample the emissive map at the chosen point. Only the RADIANCE
    // is textured — the selection pdf below stays factor-based so it matches the
    // alias-table weighting and the BSDF-leg MIS pLight, keeping the estimator
    // unbiased.
    float3 radiance = Le;
    if (t.meta.z >= 0) {
        uint io  = (uint)t.meta.x;
        uint pi  = (uint)t.meta.y;
        uint vi0 = fetchTriangleIndex(globalIndices, io, pi, 0u);
        uint vi1 = fetchTriangleIndex(globalIndices, io, pi, 1u);
        uint vi2 = fetchTriangleIndex(globalIndices, io, pi, 2u);
        float2 uv = fetchVertexUV(globalVertices, vi0) * (1.0 - b1 - b2)
                  + fetchVertexUV(globalVertices, vi1) * b1
                  + fetchVertexUV(globalVertices, vi2) * b2;
        radiance *= textures[NonUniformResourceIndex(t.meta.z)].SampleLevel(samplers[0], uv, 0.0).rgb;
    }

    es.L        = L;
    es.radiance = radiance;
    es.dist     = dist;
    es.pdf      = luminance(Le) * dist * dist / (frameUBO.totalEmissivePower * cosL);
    es.valid    = es.pdf > 0.0;
    return es;
}

//Entry point
//8x8 tile keeps subgroup occupancy high on both nvidia and amd
[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID, uint3 dim : SV_GroupThreadID){
    // pixel reject + RNG seed
    uint rngState = initRNG(gid.xy, dim.xy, frameUBO.frameIndex);

    // Sub-pixel jitter each sample — without it every sample fires through the
    // pixel center and aliasing is baked in no matter how many land. tentWarp
    // gives a triangular offset in [-1,1)² around the pixel center: a radius-1px
    // tent reconstruction filter, softer edges than a plain box (uniform) jitter
    // with negligible extra blur. Swap to a gaussian / Mitchell kernel here for
    // even softer falloff at the cost of more blur.
    float2 jitter = float2(tentWarp(rand(rngState)), tentWarp(rand(rngState)));
    float2 px     = float2(gid.xy) + 0.5f + jitter;
    float2 ndc    = (px / frameUBO.screenSize) * 2.0f - 1.0f;
    ndc.x        *= frameUBO.screenSize.x / frameUBO.screenSize.y;

    // primary ray: camera-space helpers in PTUtils, then lift to world with
    // invView so origin lands at the actual camera and forward points along
    // -view. transformRayToWorld preserves the fisheye TMax<0 sentinel.
    RayDesc ray = transformRayToWorld(generatePrimaryRay(ndc, rngState, (CameraMode)MODE, frameUBO),
                                      frameUBO.invView);

    float3 radiance   = float3(0.0);
    float3 throughput = float3(1.0);

    // Beer-Lambert: absorption coefficient of the medium the current ray
    // segment travels through. Vacuum (0) until a refraction event enters a
    // volume; reset on exit. Carried across bounces so TIR inside a solid keeps
    // absorbing. Assumes watertight, two-sided glass (every entry has a
    // matching exit) — single-quad panes must be authored as no-volume.
    float3 sigmaA = float3(0.0);

    // MIS state carried from the previous bounce's BSDF sample. When this
    // bounce lands on an emissive surface, its emission is weighted against the
    // light-sampling strategy using `prevBsdfPdf`. `prevWasDelta` starts true so
    // a directly-visible emitter (primary ray, no prior light-sampling chance)
    // is taken at full weight.
    float prevBsdfPdf  = 0.0;
    bool  prevWasDelta = true;

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
    const float FIREFLY_CLAMP = 256;

    uint cap = min(MAX_BOUNCES, frameUBO.bounceCap);
    for (uint b = 0u; b < cap; ++b) {
        HitInfo hit = traceRay(ray, rngState, b);

        // Miss: sample environment (full-res envCube, not the 128²
        //    GGX-prefiltered chain) and terminate.
        if (!hit.hit) {
            float3 env  = envCube.SampleLevel(ray.Direction, 0.0).rgb;
            float3 envC = throughput * env * frameUBO.scaleIBLAmbient;
            float  m    = max(max(envC.r, envC.g), envC.b);
            if (m > FIREFLY_CLAMP) envC *= FIREFLY_CLAMP / m;
            radiance += envC;
            break;
        }

        // Beer-Lambert absorption over the segment just traversed (previous
        // vertex → this hit). No-op in vacuum. Applied before any contribution
        // at this hit so emissive / NEE picked up here are already tinted by
        // the medium the path came through.
        if (any(sigmaA > 0.0)) throughput *= exp(-sigmaA * hit.hitT);

        // Pick up emission at the hit, MIS-weighted against the emissive-light
        // NEE strategy (which also samples this surface). The weight uses the
        // light-sampling pdf this emitter WOULD have had as seen from the
        // previous vertex — computed from the material's emissiveFactor (not the
        // textured hit.emissive) so it matches the alias-table weighting and
        // stays exact for untextured emitters. Full weight when the previous
        // bounce was specular/delta (NEE couldn't have sampled this direction)
        // or on the primary ray.
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

        float3 V = -ray.Direction;   // toward camera/incoming ray's origin

        // NEE — analytic lights + emissive geometry folded into a SINGLE shadow
        // ray. Each bounce stochastically picks a light category and fires one
        // occlusion ray; dividing by the category-selection probability keeps the
        // estimator unbiased. Halves NEE traversal vs sampling both categories
        // every bounce. The selection prob (pAnalytic) only enters as the 1/p
        // estimator factor — the power-heuristic MIS weights are unchanged (they
        // just partition the integrand), so the BSDF-leg misEm pickup above stays
        // exact. When only one category exists its prob is 1 → no variance change;
        // when both exist, 50/50 trades some direct-light variance for the ray.
        if (ENABLE_NEE) {
            bool haveAnalytic = frameUBO.lightCount > 0u;
            bool haveEmissive = frameUBO.emissiveTriCount > 0u;
            if (haveAnalytic || haveEmissive) {
                float pAnalytic = (haveAnalytic && haveEmissive) ? 0.5
                                : (haveAnalytic ? 1.0 : 0.0);

                if (rand(rngState) < pAnalytic) {
                    // Analytic-light RIS. No BSDF MIS — the BSDF sampler can't hit
                    // these lights, so there's no double-count to weight against.
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
                                // evaluateFullBrdf includes the clearcoat lobe + base
                                // attenuation; without clearcoat (factor=0) it collapses
                                // to the standard PBR BRDF.
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
                    // Emissive geometry (area lights), MIS-combined with the BSDF
                    // sampler — the counterpart to the misEm pickup above, so the
                    // two strategies don't double-count.
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

        // BSDF sample for indirect. Split diffuse / GGX-VNDF sampler
        //    Fresnel-weighted lobe pick keeps glossy surfaces actually finding
        //    their highlights instead of relying on the diffuse lobe to ever
        //    hit them by accident.
        BsdfSample bs = sampleBsdf(hit, V, rngState);
        if (!bs.valid) break;
        throughput *= bs.weight;

        // Carry this bounce's sampling pdf for the next hit's emissive MIS.
        // Delta lobes (smooth refraction) flag so the next emitter hit is taken
        // at full weight — NEE can't sample a singular direction.
        prevBsdfPdf  = bs.pdf;
        prevWasDelta = bs.isDelta;

        // Belt-and-braces: throughput should never go non-finite, but if a
        // numerical edge case in refract / pdf math produces NaN/Inf here we
        // terminate the path. Without this the accumulator gets infected and
        // the pixel renders black for the rest of the session.
        if (any(isnan(throughput)) || any(isinf(throughput))) break;

        // Russian roulette after the first MIN_BOUNCES bounces. Floor at
        //    0.05 stops near-black throughput from killing the path before
        //    it can pick up an emissive / NEE contribution downstream.
        if (b >= MIN_BOUNCES && RUSSIAN_ROULETTE) {
            float rr = clamp(luminance(throughput), 0.05, 0.95);
            if (rand(rngState) > rr) break;
            throughput /= rr;
        }

        // Update the participating medium on refraction. Entering (front face)
        // adopts this material's absorption; exiting (back face) returns to
        // vacuum. TIR keeps bs.transmit == false, so sigmaA is preserved and
        // the trapped ray keeps absorbing as it bounces inside the solid.
        if (bs.transmit) {
            sigmaA = hit.frontFace ? mediumSigmaA(materials[hit.matidx]) : float3(0.0);
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