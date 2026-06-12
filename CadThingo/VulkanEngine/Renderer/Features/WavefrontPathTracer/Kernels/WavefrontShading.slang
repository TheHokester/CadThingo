// WavefrontShading - shared scene/shading helpers for the wavefront kernels.
//
// These are DUPLICATED from PTCompute.slang (the megakernel), not extracted: the
// megakernel runs an 8x8 dispatch and uses implicit-LOD `.Sample`, whereas the
// wavefront kernels are 1D ([numthreads(64,1,1)]) with no derivative quad, so every
// texture fetch here is explicit-LOD `.SampleLevel(.., 0)`. Keeping a private copy
// lets the wavefront path evolve (and stay 1D-correct) without touching the shipping
// megakernel. `import`-ed by every kernel; re-exports WavefrontBindings (which itself
// re-exports PTUtils/PbrUtils/CommonTypes), so a kernel that imports this sees the
// whole set/binding layout + the math helpers.
module WavefrontShading;

__exported import WavefrontBindings;

// On bounces >= this, resolveHit skips the normal / metallic-roughness / clearcoat /
// transmission detail-texture fetches and falls back to material factors -- the bulk of
// per-hit memory traffic, ~invisible on indirect bounces (mirror of PTCompute).
public static const uint MATERIAL_LOD_BOUNCE = 2u;

// Per-contribution cap (max channel). Tiny BRDF/pdf ratios at grazing specular / near-
// degenerate refraction produce one-sample radiances in the thousands; clamping is the
// standard firefly fix (biased low at tiny sample counts, converges back).
public static const float WF_FIREFLY_CLAMP = 256.0;

public float3 wfFireflyClamp(float3 c) {
    float m = max(max(c.r, c.g), c.b);
    return (m > WF_FIREFLY_CLAMP) ? c * (WF_FIREFLY_CLAMP / m) : c;
}


// --- P3.1 Shading-class taxonomy (material-sorted shading) -------------------
// C = 4 routing classes, superset-ordered so DIFFUSE subset CONDUCTOR subset DIELECTRIC subset
// FULL (lobe sets are additive). classifyShadingClass returns the MINIMAL covering class for a
// material, from factors + bound-texture presence only (never resolved texels) -- cheap enough to
// run in Extend's routing tail, and conservative: a metallic-/transmission-TEXTURED material routes
// to the wider class even at factor 0, and anything ambiguous falls to FULL (== the verbatim P2
// BSDF), so a misjudgement only ever over-shades, never mis-shades. Used by Extend (P3.3) to bin
// the shade queue; P3b feeds the class into a SHADING_CLASS spec constant to dead-strip lobes.
public static const uint SC_DIFFUSE = 0u, SC_CONDUCTOR = 1u, SC_DIELECTRIC = 2u, SC_FULL = 3u;
public static const uint SC_COUNT   = WF_SHADE_CLASSES;   // canonical count lives in WavefrontBindings

// Lobe mask for a shading class -> drives compile-time lobe stripping (P3b). Superset-ordered:
// every class has diffuse + base-spec; DIELECTRIC adds the transmit lobe; FULL adds clearcoat.
// Spec-folded in Shade (from SHADING_CLASS) so the stripped lobes -- and their resolveHit texel
// fetches + sampleBsdf branches -- dead-strip for the opaque classes. LOBE_* live in PTUtils.
public uint classLobeMask(uint shadingClass) {
    uint m = LOBE_DIFFUSE | LOBE_SPEC;
    if (shadingClass >= SC_DIELECTRIC) m |= LOBE_TRANSMIT;    // DIELECTRIC, FULL
    if (shadingClass >= SC_FULL)       m |= LOBE_CLEARCOAT;   // FULL only
    return m;
}

public uint classifyShadingClass(PbrMaterial m) {
    // Classify on FACTORS ONLY. This engine leaves UNSET texture slots pointing at shared
    // GltfDefaults textures -- valid bindless indices (>= 0), NOT -1 -- so a `tex >= 0` presence
    // test is ~always true and would dump every material into FULL (the first branch). And textures
    // compose MULTIPLICATIVELY here (metallic = metallicFactor * tex, transmission/clearcoat likewise,
    // per resolveHit), so a zero factor kills its lobe regardless of the bound texture -- making
    // factors alone both correct and conservative. Order matters: widest lobe set first (supersets win).
    if (m.clearcoatFactor    > 0.0) return SC_FULL;        // coat => all lobes
    if (m.transmissionFactor > 0.0) return SC_DIELECTRIC;  // + transmit lobe
    if (m.metallicFactor     > 0.0) return SC_CONDUCTOR;   // metal possible (same lobes as DIFFUSE; coherence split)
    return SC_DIFFUSE;
}


// --- UV / alpha --------------------------------------------------------------
public float2 computeUV(uint indexOffset, uint primIdx, float2 bary) {
    uint i0 = fetchTriangleIndex(globalIndices, indexOffset, primIdx, 0u);
    uint i1 = fetchTriangleIndex(globalIndices, indexOffset, primIdx, 1u);
    uint i2 = fetchTriangleIndex(globalIndices, indexOffset, primIdx, 2u);
    float2 uv0 = fetchVertexUV(globalVertices, i0);
    float2 uv1 = fetchVertexUV(globalVertices, i1);
    float2 uv2 = fetchVertexUV(globalVertices, i2);
    float  w0  = 1.0 - bary.x - bary.y;
    return uv0 * w0 + uv1 * bary.x + uv2 * bary.y;
}

public float computeAlpha(PbrMaterial mat, uint indexOffset, uint primIdx, float2 barycentrics) {
    int baseColorIdx = mat.baseColorTex;
    if (baseColorIdx < 0 || mat.flags == 0) return 1.0f;
    float2 uv = computeUV(indexOffset, primIdx, barycentrics);
    return textures[NonUniformResourceIndex(baseColorIdx)].SampleLevel(samplers[0], uv, 0.0).a * mat.baseColorFactor.a;
}


// --- Hit resolve (off the stored compact hit; no re-trace) -------------------
// The post-commit material/geometry block of PTCompute.slang::traceRay, run from the
// stored recordIndex+primIdx+bary+t instead of a live RayQuery. `bounce` drives the
// MATERIAL_LOD detail-fetch skip on deep bounces.
public HitInfo resolveHit(uint2 prim, float triT, float2 bary, float3 rayOrigin, float3 rayDir, uint bounce, uint lobeMask = LOBE_ALL) {
    HitInfo h = (HitInfo)0;
    h.hit      = true;
    h.hitT     = triT;
    h.worldPos = rayOrigin + rayDir * triT;
    bool lowDetail = bounce >= MATERIAL_LOD_BOUNCE;

    uint   recordIndex = prim.x;
    uint   primIdx     = prim.y;
    float3 baryW       = float3(1.0 - bary.x - bary.y, bary.x, bary.y);

    ShadowEntityInfo info = entityInfo[recordIndex];
    h.matidx = info.materialIndex;

    uint i0 = fetchTriangleIndex(globalIndices, info.indexOffset, primIdx, 0u);
    uint i1 = fetchTriangleIndex(globalIndices, info.indexOffset, primIdx, 1u);
    uint i2 = fetchTriangleIndex(globalIndices, info.indexOffset, primIdx, 2u);

    float2 uv = fetchVertexUV(globalVertices, i0) * baryW.x
              + fetchVertexUV(globalVertices, i1) * baryW.y
              + fetchVertexUV(globalVertices, i2) * baryW.z;

    float3 nObj = fetchVertexNormal(globalVertices, i0) * baryW.x
                + fetchVertexNormal(globalVertices, i1) * baryW.y
                + fetchVertexNormal(globalVertices, i2) * baryW.z;

    float3 p0 = fetchVertexPos(globalVertices, i0);
    float3 p1 = fetchVertexPos(globalVertices, i1);
    float3 p2 = fetchVertexPos(globalVertices, i2);
    float3 ngObj = normalize(cross(p1 - p0, p2 - p0));

    float4 t0 = fetchVertexTangent(globalVertices, i0);
    float4 t1 = fetchVertexTangent(globalVertices, i1);
    float4 t2 = fetchVertexTangent(globalVertices, i2);
    float3 tObj = t0.xyz * baryW.x + t1.xyz * baryW.y + t2.xyz * baryW.z;
    float  bitangentSign = t0.w;

    float3x3 obj2worldR = entityNormalMatrix(info);
    h.normal          = normalize(mul(obj2worldR, nObj));
    h.geometricNormal = normalize(mul(obj2worldR, ngObj));
    float3 tangent    = normalize(mul(obj2worldR, tObj));

    h.frontFace = dot(h.geometricNormal, rayDir) < 0.0;
    if (!h.frontFace) {
        h.normal          = -h.normal;
        h.geometricNormal = -h.geometricNormal;
    }

    PbrMaterial  mat = materials[h.matidx];
    SamplerState s   = samplers[0];

    h.baseColor = mat.baseColorFactor;
    if (mat.baseColorTex >= 0)
        h.baseColor *= textures[NonUniformResourceIndex(mat.baseColorTex)].SampleLevel(s, uv, 0.0);

    h.metallic  = mat.metallicFactor;
    h.roughness = mat.roughnessFactor;
    if (mat.physicalDescriptorTex >= 0 && !lowDetail) {
        float2 mr   = textures[NonUniformResourceIndex(mat.physicalDescriptorTex)].SampleLevel(s, uv, 0.0).bg;
        h.metallic  *= mr.x;
        h.roughness *= mr.y;
    }

    h.emissive = mat.emissiveFactor;
    if (mat.emissiveTex >= 0)
        h.emissive *= textures[NonUniformResourceIndex(mat.emissiveTex)].SampleLevel(s, uv, 0.0).rgb;

    if (mat.normalTex >= 0 && !lowDetail) {
        float3 nm = textures[NonUniformResourceIndex(mat.normalTex)].SampleLevel(s, uv, 0.0).xyz * 2.0 - 1.0;
        float3 T  = normalize(tangent);
        float3 B  = cross(h.normal, T) * bitangentSign;
        h.normal  = normalize(nm.x * T + nm.y * B + nm.z * h.normal);
    }

    // P3b: skip the transmission / clearcoat reads entirely for classes whose PSO strips those
    // lobes. The factors live in the already-loaded material struct (cheap), but the texel fetches
    // are real memory traffic -- and since unset tex slots point at valid GltfDefaults textures
    // (tex >= 0 is ~always true here), they would otherwise ALWAYS fire on bounces 0-1. Gated on
    // the spec-folded lobeMask so the SampleLevel calls dead-strip for opaque classes.
    h.transmission = 0.0;
    if ((lobeMask & LOBE_TRANSMIT) != 0u) {
        h.transmission = mat.transmissionFactor;
        if (mat.transmissionTex >= 0 && !lowDetail)
            h.transmission *= textures[NonUniformResourceIndex(mat.transmissionTex)].SampleLevel(s, uv, 0.0).r;
    }
    h.ior = mat.ior;

    float f0_eta = (mat.ior - 1.0) / (mat.ior + 1.0);
    h.F0 = f0_eta * f0_eta;

    h.clearcoat          = 0.0;
    h.clearcoatRoughness = mat.clearcoatRoughnessFactor;   // floored below; value unused when clearcoat=0
    if ((lobeMask & LOBE_CLEARCOAT) != 0u) {
        h.clearcoat = mat.clearcoatFactor;
        if (mat.clearcoatTex >= 0 && !lowDetail)
            h.clearcoat *= textures[NonUniformResourceIndex(mat.clearcoatTex)].SampleLevel(s, uv, 0.0).r;
        if (mat.clearcoatRoughnessTex >= 0 && !lowDetail)
            h.clearcoatRoughness *= textures[NonUniformResourceIndex(mat.clearcoatRoughnessTex)].SampleLevel(s, uv, 0.0).g;
    }

    // Floor roughness so GGX D() stays evaluable (0 -> 0/0 NaN at the mirror dir).
    h.clearcoatRoughness = max(h.clearcoatRoughness, 0.045);
    h.roughness          = max(h.roughness,          0.045);
    h.clearcoatNormal    = h.normal;

    return h;
}


// --- Shadow / occlusion ray (NEE visibility) ---------------------------------
// ACCEPT_FIRST_HIT_AND_END_SEARCH; combined alpha + transmission gate so MASK / BLEND
// foliage and tinted glass cast correctly-graded soft shadows as samples accumulate.
public bool isOccluded(float3 origin, float3 dir, float tMax, inout uint rngState) {
    RayDesc r;
    r.Origin    = origin;
    r.Direction = dir;
    r.TMin      = 0.0;
    r.TMax      = tMax;

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
                transmittance *= textures[NonUniformResourceIndex(cmat.transmissionTex)].SampleLevel(samplers[0], uv, 0.0).r;
            }

            float passAlpha = 1.0;
            if ((cmat.flags & 1u) != 0u) passAlpha = (alpha >= cmat.alphaCutoff) ? 1.0 : 0.0;   // MASK
            else if ((cmat.flags & 4u) != 0u) passAlpha = alpha;                                 // BLEND

            float blockProb = passAlpha * (1.0 - transmittance);
            if (rand(rngState) < blockProb) q.CommitNonOpaqueTriangleHit();
        }
    }
    return q.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
}


// --- KHR_materials_volume absorption (Beer-Lambert) --------------------------
public float3 mediumSigmaA(PbrMaterial m) {
    if (m.attenuationDistance <= 0.0 || m.attenuationDistance >= 1e6) return float3(0.0);
    return -log(max(m.attenuationColor, float3(1e-4))) / m.attenuationDistance;
}


// --- Analytic-light RIS pick -------------------------------------------------
public static const uint RIS_LIGHT_SAMPLE_COUNT = 8u;

public struct LightPick {
    public LightSample sample;
    public float       weight;   // W; multiply into the direct-light estimator
    public bool        valid;
};

public LightPick sampleLightRIS(inout uint rngState, float3 worldPos, float3 N) {
    LightPick best;
    best.sample = (LightSample)0;
    best.weight = 0.0;
    best.valid  = false;

    if (frameUBO.lightCount == 0u) return best;

    uint M = (frameUBO.lightCount == 1u) ? 1u : RIS_LIGHT_SAMPLE_COUNT;

    float wSum        = 0.0;
    float bestPTarget = 0.0;
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

        if (rand(rngState) * wSum < w) {
            bestLs      = ls;
            bestPTarget = pTarget;
        }
    }

    if (bestPTarget <= 0.0 || wSum <= 0.0) return best;

    best.sample = bestLs;
    best.weight = wSum / (float(M) * bestPTarget);
    best.valid  = true;
    return best;
}


// --- Emissive area-light sample (NEE) ----------------------------------------
public struct EmissiveLightSample {
    public float3 L;          // shading point -> sampled point on emitter
    public float3 radiance;   // Le at the emitter (textured to match the BSDF-leg pickup)
    public float  dist;
    public float  pdf;        // solid-angle pdf incl. power-proportional selection
    public bool   valid;
};

public EmissiveLightSample sampleEmissive(inout uint rng, float3 P) {
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
    float3 Le = float3(t.e1LeR.w, t.e2LeG.w, t.nLeB.w);   // emissiveFactor
    float3 nL = t.nLeB.xyz;

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