#ifndef PBR_UTILS_INCLUDED
#define PBR_UTILS_INCLUDED

static const float PI = 3.14159265359;

struct PbrMaterial {
    float4 baseColorFactor;        // Base color tint/multiplier
    float3 emissiveFactor;         // HDR emissive (glTF factor × KHR_materials_emissive_strength)
    float alphaCutoff;             // Packs into emissiveFactor's std430 trailing 4B
    float metallicFactor;          // Metallic property multiplier
    float roughnessFactor;         // Surface roughness multiplier
    uint flags;                    // bit 0 = alphaMask, bit 1 = doubleSided, bit 2 = alphaBlend, ...
    int baseColorTex;              // Texture binding index for base color (-1 = none)
    int physicalDescriptorTex;     // Texture binding for metallic/roughness
    int normalTex;                 // Texture binding for normal maps
    int occlusionTex;              // Texture binding for ambient occlusion
    int emissiveTex;               // Texture binding for emissive maps

    // KHR_materials_transmission + KHR_materials_ior. Fields are present on
    // every material upload but consumers can ignore them — opaque materials
    // have transmissionFactor=0 / ior=1.5 (glTF defaults) and read the same.
    float transmissionFactor;
    float ior;

    // KHR_materials_clearcoat. Same opt-in model: clearcoatFactor=0 means
    // the coat lobe contributes nothing, so a BRDF that doesn't implement
    // clearcoat yet simply leaves these fields unread.
    float clearcoatFactor;
    float clearcoatRoughnessFactor;
    int transmissionTex;           // R channel multiplies transmissionFactor
    int clearcoatTex;              // R channel multiplies clearcoatFactor
    int clearcoatRoughnessTex;     // G channel multiplies clearcoatRoughnessFactor
    int clearcoatNormalTex;        // tangent-space normal for the coat
};
struct InstanceData {
    float4x4 model;
    uint materialIndex;
    uint _pad0, _pad1, _pad2;
};

// Per-light record packed by Renderer_Resources.UpdateLightingBuffers.
// Conventions:
//   positionRange.w  =  range (point/spot)     — directional uses -1 sentinel
//   directionType.w  =  LightType encoded as float (0=Dir, 1=Point, 2=Spot)
//   spotCones.z      =  CastShadows (0/1)
struct PbrLight {
    float4 positionRange;//   positionRange.w  =  range (point/spot)     — directional uses -1 sentinel
    float4 colorIntensity;
    float4 directionType;//   directionType.w  =  LightType encoded as float (0=Dir, 1=Point, 2=Spot)
    float4 spotCones;//   spotCones.z      =  CastShadows (0/1)
};

// Light type tags — must match CadThingo.VulkanEngine.LightType.
static const uint LIGHT_TYPE_DIRECTIONAL = 0u;
static const uint LIGHT_TYPE_POINT       = 1u;
static const uint LIGHT_TYPE_SPOT        = 2u;

//normal distribution function (D) - GGX/trowbridge-reitz
//describes the statistical distribution of material microfacet normals
float DistributionGGX(float NdotH, float roughness){
    float a = roughness * roughness;//remapping for perceptual linearity
    float a2 = a * a;
    float NdotH2 = NdotH * NdotH;
    
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    return a2 / denom;
}

//Geometry function (G) - Smith's method
//models self shadowing and masking between microfacets
float GeometrySmith(float NdotV,float NdotL,float roughness){
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    
    //geometry obstruction from view direction (masking)
    float ggx1 = NdotV / (NdotV * (1.0 - k) + k);
    //geometry obstruction from light direction (shadowing)
    float ggx2 = NdotL / (NdotL * (1.0 - k) + k);
    
    return ggx1 * ggx2; //both obstructions are combined
}
//Fresnels Reflectance (F)
float3 FresnelSchlick(float cosTheta, float3 F0){
return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

// Roughness-aware Schlick Fresnel for IBL split-sum eval. The standard
// FresnelSchlick saturates F0 at grazing angles for smooth surfaces; this
// variant pulls the limit down on rough surfaces so the kS term doesn't
// overestimate the specular grab when integrating against an environment.
// Source: Karis 2013 / Lazarov 2013.
float3 FresnelSchlickRoughness(float cosTheta, float3 F0, float roughness){
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
struct LightSample {
    float3 L;
    float  distance;
    float3 radiance;
    bool   castShadows;
    float  lightRadius;   // spotCones.w — only meaningful for soft shadows
};

bool evaluateLight(PbrLight lt, float3 worldPos, out LightSample ls) {
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

#endif // PBR_UTILS_INCLUDED
