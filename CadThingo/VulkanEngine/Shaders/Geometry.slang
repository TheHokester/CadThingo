struct GBufferUBO {
    float4x4 model;                     //model to world transformation matrix
    float4x4 view;                      // World-to-camera transformation matrix
    float4x4 proj;                      // Camera-to-clip space projection matrix
}
// Input from vertex buffer - Data sent per vertex from CPU
struct VSInput {
    float3 Position : POSITION;     // 3D position in model space
    float3 Normal : NORMAL;         // Surface normal for lighting calculations
    float2 UV : TEXCOORD0;          // Texture coordinates for material sampling
    float4 Tangent : TANGENT;       // Tangent vector for normal mapping (w component = handedness)
};

// Output from vertex shader / Input to fragment shader - Interpolated data
struct VSOutput {
    float4 Position : SV_POSITION; // Required clip space position for rasterization
    float3 WorldPos : POSITION;    // World space position for lighting calculations
    float3 Normal : NORMAL;        // World space normal (interpolated)
    float2 UV : TEXCOORD0;         // Texture coordinates (interpolated)
    float4 Tangent : TANGENT;      // World space tangent (interpolated)
};

struct GBufferOutput {
    float4 Position : SV_TARGET0;//xyz world pos, w unused
    float4 Normal : SV_TARGET1;//xyz = world normal, w unused
    float4 Albedo : SV_TARGET2;//xyz = basecolor, w = metallic
    float4 Material : SV_TARGET3;//x = roughness, y = ao, zw unused
}
#include "PbrUtils.slang"

[[vk::binding(0,0)]] ConstantBuffer<GBufferUBO> ubo; 
[[vk::binding(1, 0)]] Sampler2D baseColorSampler;
[[vk::binding(2, 0)]] Sampler2D metallicRoughnessSampler;
[[vk::binding(3, 0)]] Sampler2D normalSampler;
[[vk::binding(4, 0)]] Sampler2D occlusionSampler;
[[vk::binding(5, 0)]] Sampler2D emissiveSampler;

[[vk::push_constant]] PushConstants material; 
[[shader("vertex")]]
VSOutput VSMain(VSInput input){
    VSOutput output;
    
    // Transform vertex position through the rendering pipeline
    // Model -> World -> Camera -> Clip space transformation chain
    float4 worldPos = mul(ubo.model, float4(input.Position, 1.0));
    output.Position = mul(ubo.proj, mul(ubo.view, worldPos));

    // Pass world position for fragment lighting calculations
    // Fragment shader needs world space position to calculate light vectors
    output.WorldPos = worldPos.xyz;

    // Transform normal from model space to world space
    // Use only rotation/scale part of model matrix (upper-left 3x3)
    // Normalize to ensure unit length after transformation
    output.Normal = normalize(mul((float3x3)ubo.model, input.Normal));

    // Pass through texture coordinates unchanged
    // UV coordinates are typically in [0,1] range and don't need transformation
    output.UV = input.UV;

    // Pass tangent vector for normal mapping
    // Will be used in fragment shader to construct tangent-space basis
    output.Tangent = input.Tangent;

    return output;
}


[[shader("fragment")]]
GBufferOutput PSMain(VSOutput input){
    GBufferOutput output;
    
    float4 baseColor = baseColorSampler.Sample(input.UV) * material.baseColorFactor;
    float2 metallicRoughness = metallicRoughnessSampler.Sample(input.UV).bg;
    float metallic = metallicRoughness.x * material.metallicFactor;
    float roughness = metallicRoughness.y * material.roughnessFactor;
    float ao = occlusionSampler.Sample(input.UV).r;
    float3 emissive = emissiveSampler.Sample(input.UV).rgb;
    
    //TBN and normal mapping
    float3 N = normalize(input.Normal);
    if(material.normalTextureSet >= 0){
        float3 tangentNormal = normalSampler.Sample(input.UV).xyz * 2.0 - 1.0;
        float3 T = normalize(input.Tangent.xyz);
        float3 B = normalize(cross(N, T)) * input.Tangent.w;
        float3x3 TBN = float3x3(T,B,N);
        N = normalize(mul(tangentNormal, TBN)); 
    }
    
    //pack into GBuffer - NO lighting NO pbr math
    output.Position = float4(input.WorldPos, 1.0);
    output.Normal = float4(N, 0.0);
    
    output.Albedo = float4(baseColor.rgb, metallic);
    output.Material = float4(roughness, ao, 0.0, 0.0);
    
    return output;
}
