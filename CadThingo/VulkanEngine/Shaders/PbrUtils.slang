// PbrUtils — the BRDF + analytic-lighting + IBL math layer.
//
// `module PbrUtils;` with `__exported import CommonTypes;` so a consumer that
// writes `import PbrUtils;` transitively sees PbrMaterial / PbrLight / PI / the
// vertex helpers etc. without a second import line.
//
// Everything here is pure / re-entrant. Functions that need GPU resources take
// them as parameters (see evaluateIBL) rather than reading globals, so the
// module never pins a binding layout.

module PbrUtils;

__exported import CommonTypes;

//normal distribution function (D) - GGX/trowbridge-reitz
//describes the statistical distribution of material microfacet normals
public float DistributionGGX(float NdotH, float roughness){
    float a = roughness * roughness;//remapping for perceptual linearity
    float a2 = a * a;
    float NdotH2 = NdotH * NdotH;

    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    return a2 / denom;
}

//Geometry function (G) - Smith's method
//models self shadowing and masking between microfacets
public float GeometrySmith(float NdotV,float NdotL,float roughness){
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;

    //geometry obstruction from view direction (masking)
    float ggx1 = NdotV / (NdotV * (1.0 - k) + k);
    //geometry obstruction from light direction (shadowing)
    float ggx2 = NdotL / (NdotL * (1.0 - k) + k);

    return ggx1 * ggx2; //both obstructions are combined
}
//Fresnels Reflectance (F)
public float3 FresnelSchlick(float cosTheta, float3 F0){
return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

// Roughness-aware Schlick Fresnel for IBL split-sum eval. The standard
// FresnelSchlick saturates F0 at grazing angles for smooth surfaces; this
// variant pulls the limit down on rough surfaces so the kS term doesn't
// overestimate the specular grab when integrating against an environment.
// Source: Karis 2013 / Lazarov 2013.
public float3 FresnelSchlickRoughness(float cosTheta, float3 F0, float roughness){
    return F0 + (max(float3(1.0 - roughness, 1.0 - roughness, 1.0 - roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0);
}

//  Analytic-light evaluation

// Resolves a PbrLight against a shading point into the canonical direct-light
// terms every consumer (deferred PBR, transparent, pathtracer NEE) needs:
//   * L         — unit vector FROM surface TO light (or -dir for directional)
//   * distance  — used for shadow-ray TMax; 1e6 for directional (any finite
//                 number large enough to clear the scene works)
//   * radiance  — colour × intensity × distance attenuation × spot mask;
//                 callers still multiply by the BRDF and NdotL themselves
// Returns false when the light contributes zero (intensity<=0, outside spot
// cone). Range-based attenuation may produce a near-zero but still-positive
// radiance; callers can skip with their own threshold if desired.
public struct LightSample {
    public float3 L;
    public float  distance;
    public float3 radiance;
    public bool   castShadows;
    public float  lightRadius;   // spotCones.w — only meaningful for soft shadows
};

public bool evaluateLight(PbrLight lt, float3 worldPos, out LightSample ls) {
    ls = (LightSample)0;
    uint   type      = (uint)lt.directionType.w;
    float3 lColor    = lt.colorIntensity.rgb;
    float  intensity = lt.colorIntensity.a;
    if (intensity <= 0.0) return false;

    float3 L;
    float  distance;
    float  attenuation;
    float  spotMask = 1.0;

    if (type == LIGHT_TYPE_DIRECTIONAL) {
        // Directional: field stores the *light-to-scene* direction, so flip.
        L           = normalize(-lt.directionType.xyz);
        distance    = 1e6;
        attenuation = 1.0;
    } else {
        float3 toLight = lt.positionRange.xyz - worldPos;
        distance       = length(toLight);
        L              = toLight / max(distance, 1e-4);
        attenuation    = 1.0 / max(distance * distance, 1e-4);

        // Smooth range cutoff so contribution falls to 0 at light.range.
        float range = lt.positionRange.w;
        if (range > 0.0) {
            float t      = saturate(1.0 - pow(distance / range, 4.0));
            attenuation *= t * t;
        }

        if (type == LIGHT_TYPE_SPOT) {
            float3 spotDir = normalize(lt.directionType.xyz);
            float  cosA    = dot(-L, spotDir);
            float  inner   = lt.spotCones.x;
            float  outer   = lt.spotCones.y;
            spotMask       = smoothstep(outer, inner, cosA);
            if (spotMask <= 0.0) return false;
        }
    }

    ls.L           = L;
    ls.distance    = distance;
    ls.radiance    = lColor * intensity * attenuation * spotMask;
    ls.castShadows = lt.spotCones.z > 0.5;
    ls.lightRadius = lt.spotCones.w;
    return true;
}

// Single analytic-light Cook-Torrance term (physically-correct Lambert: /PI).
// Returns the outgoing-radiance contribution ALREADY multiplied by NdotL —
// accumulate the return value straight into Lo. `albedo` is the diffuse colour
// (baseColor.rgb); pass an already-resolved `radiance` (colour·intensity·
// attenuation·spot) and `F0` (lerp(0.04, baseColor, metallic)).
//
// This mirrors PbrShader.slang's in-loop BRDF. NOTE: Transparent.slang's loop
// currently uses a `* PI` diffuse term (vs `/ PI` here) — see CLAUDE notes /
// the refactor write-up — so it deliberately does NOT call this yet.
public float3 cookTorranceDirect(float3 N, float3 V, float3 L, float3 radiance,
                                 float3 albedo, float metallic, float roughness, float3 F0) {
    float3 H = normalize(V + L);

    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.0);
    float NdotH = max(dot(N, H), 0.0);
    float HdotV = max(dot(H, V), 0.0);

    float  D = DistributionGGX(NdotH, roughness);
    float  G = GeometrySmith(NdotV, NdotL, roughness);
    float3 F = FresnelSchlick(HdotV, F0);

    float3 specular = (D * G * F) / (4.0 * NdotV * NdotL + 0.0001);

    float3 kS = F;
    float3 kD = (float3(1.0, 1.0, 1.0) - kS) * (1.0 - metallic);

    return (kD * albedo / PI + specular) * radiance * NdotL;
}

// Split-sum IBL ambient + reflection-probe blend
// Pulls the diffuse irradiance, the prefiltered specular (probe-blended, with a
// global-IBL fallback via the coverage lerp), and the BRDF LUT into the final
// ambient term. Resources are passed in so the deferred PBR pass and the
// transparent pass share one definition while owning separate descriptor sets.
//   pixelPos — integer pixel coords (SV_Position.xy cast to uint) for the
//              cluster lookup; matches the tile grid used by light culling.
public float3 evaluateIBL(
    float3 N, float3 V, float3 R,
    float3 baseColor, float metallic, float roughness, float ao, float3 F0,
    float3 worldPos, uint2 pixelPos,
    SamplerCube irradianceCube, SamplerCube prefilteredCube, Sampler2D brdfLut,
    SamplerCubeArray probeCubeArray,
    StructuredBuffer<ProbeRecord> probes,
    StructuredBuffer<ProbeClusterRange> probeClusterRange,
    StructuredBuffer<uint> probeIndexList,
    float prefilteredCubeMipLevels, float scaleIBLAmbient,
    uint probeClusterDimsX, uint probeClusterDimsY, float probeMipLevels)
{
    float  NdotV_ibl = max(dot(N, V), 0.0);
    float3 F_ibl     = FresnelSchlickRoughness(NdotV_ibl, F0, roughness);
    float3 kS_ibl    = F_ibl;
    float3 kD_ibl    = (float3(1.0) - kS_ibl) * (1.0 - metallic);

    float3 irradianceVal = irradianceCube.Sample(N).rgb;
    float3 diffuseIBL    = irradianceVal * baseColor;

    float  maxMip  = max(prefilteredCubeMipLevels - 1.0, 0.0);
    float2 envBRDF = brdfLut.SampleLevel(float2(NdotV_ibl, roughness), 0).xy;

    // Probe-blend specular: walk the cluster's probe list and accumulate
    // distance-weighted samples from probeCubeArray. Fragments outside any
    // probe's influence sphere fall back to the global IBL prefiltered cube.
    float3 specularInput = float3(0.0);
    {
        uint cx = min(pixelPos.x / TILE_SIZE_PX, probeClusterDimsX - 1u);
        uint cy = min(pixelPos.y / TILE_SIZE_PX, probeClusterDimsY - 1u);
        // Z slice = 0 today; when 3D froxels land cz is computed from view-Z
        // and the same cluster index math handles it without further edits.
        uint cz = 0u;
        uint clusterIdx = cx + cy * probeClusterDimsX
                       + cz * probeClusterDimsX * probeClusterDimsY;

        ProbeClusterRange r = probeClusterRange[clusterIdx];

        float maxMipProbe = max(probeMipLevels - 1.0, 0.0);
        float3 accum  = float3(0.0);
        float  totalW = 0.0;
        for (uint p = 0u; p < r.count; p++) {
            uint probeIdx = probeIndexList[r.offset + p];
            ProbeRecord rec = probes[probeIdx];

            float3 d   = worldPos - rec.positionRadius.xyz;
            float dist = length(d);
            float rad  = rec.positionRadius.w;
            if (rad <= 0.0 || dist >= rad) continue;

            // Plateau-then-fade weight: full strength out to ~half the radius,
            // then a smooth cubic fall-off to 0 at the boundary.
            float w = 1.0 - smoothstep(rad * 0.5, rad, dist);

            float3 s = probeCubeArray.SampleLevel(
                float4(R, (float)rec.cubeArraySlot),
                roughness * maxMipProbe).rgb;
            accum  += s * w;
            totalW += w;
        }

        // totalW as coverage: lerp global IBL → probe blend so the sphere edge
        // (totalW≈0) leans on the global cube and the interior is pure probe.
        float3 globalIBL  = prefilteredCube.SampleLevel(R, roughness * maxMip).rgb;
        float  coverage   = saturate(totalW);
        float3 probeBlend = (totalW > 1e-5) ? (accum / totalW) : float3(0.0);
        specularInput = lerp(globalIBL, probeBlend, coverage);
    }
    float3 specularIBL = specularInput * (F_ibl * envBRDF.x + envBRDF.y);

    return (kD_ibl * diffuseIBL + specularIBL) * ao * scaleIBLAmbient;
}
