static const float PI = 3.14159265359;

struct PbrMaterial {
    float4 baseColorFactor;             // Base color tint/multiplier
    float metallicFactor;               // Metallic property multiplier
    float roughnessFactor;              // Surface roughness multiplier
    float alphaCutoff;              // Alpha cutoff threshold
    uint flags;                    // bit 0 = alphaMask, bit 1 = doubleSided, ...
    int baseColorTex;            // Texture binding index for base color (-1 = none)
    int physicalDescriptorTex;   // Texture binding for metallic/roughness
    int normalTex;               // Texture binding for normal maps
    int occlusionTex;            // Texture binding for ambient occlusion
    int emissiveTex;             // Texture binding for emissive maps
};
struct InstanceData {
    float4x4 model;
    uint materialIndex;
    uint _pad0, _pad1, _pad2;
};

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

