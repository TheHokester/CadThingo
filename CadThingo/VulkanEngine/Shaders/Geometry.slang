// Per-frame view+proj. Model matrix lives in the per-instance SSBO (PbrUtils.slang::InstanceData).
struct FrameUBO {
    float4x4 view;
    float4x4 proj;
}

struct VSInput {
    float3 Position : POSITION;
    float3 Normal   : NORMAL;
    float2 UV       : TEXCOORD0;
    float4 Tangent  : TANGENT;
};

struct VSOutput {
    float4 Position      : SV_POSITION;
    float3 WorldPos      : POSITION;
    float3 Normal        : NORMAL;
    float2 UV            : TEXCOORD0;
    float4 Tangent       : TANGENT;
    nointerpolation uint MaterialIndex : TEXCOORD1;
};

struct GBufferOutput {
    float4 Position : SV_TARGET0;
    float4 Normal   : SV_TARGET1;
    float4 Albedo   : SV_TARGET2;
    float4 Material : SV_TARGET3;
    float4 Emissive : SV_TARGET4;
}

#include "PbrUtils.slang"

// Set 0: per-frame UBO (view+proj). Bound once per frame.
[[vk::binding(0,0)]] ConstantBuffer<FrameUBO> frame;

// Set 1: bindless material/instance/texture/sampler arrays. Bound once per frame.
[[vk::binding(0,1)]] StructuredBuffer<PbrMaterial>  materials;
[[vk::binding(1,1)]] StructuredBuffer<InstanceData> instances;
[[vk::binding(2,1)]] Texture2D                      textures[];
[[vk::binding(3,1)]] SamplerState                   samplers[];

[[shader("vertex")]]
VSOutput VSMain(VSInput input, uint instanceIndex : SV_InstanceID)
{
    InstanceData inst = instances[instanceIndex];

    float4 worldPos = mul(inst.model, float4(input.Position, 1.0));

    VSOutput output;
    output.Position      = mul(frame.proj, mul(frame.view, worldPos));
    output.WorldPos      = worldPos.xyz;
    output.Normal        = normalize(mul((float3x3)inst.model, input.Normal));
    output.UV            = input.UV;
    output.Tangent       = input.Tangent;
    output.MaterialIndex = inst.materialIndex;
    return output;
}

[[shader("fragment")]]
GBufferOutput PSMain(VSOutput input)
{
    PbrMaterial mat = materials[input.MaterialIndex];

    // Sampler 0 is the shared linear-repeat default. All bindless textures use it for now.
    SamplerState s = samplers[0];

    float4 baseColor       = textures[mat.baseColorTex].Sample(s, input.UV) * mat.baseColorFactor;
    float2 metallicRoughness = textures[mat.physicalDescriptorTex].Sample(s, input.UV).bg;
    float metallic         = metallicRoughness.x * mat.metallicFactor;
    float roughness        = metallicRoughness.y * mat.roughnessFactor;
    float ao               = textures[mat.occlusionTex].Sample(s, input.UV).r;
    float3 emissive        = textures[mat.emissiveTex].Sample(s, input.UV).rgb;

    float3 N = normalize(input.Normal);
    {
        float3 tangentNormal = textures[mat.normalTex].Sample(s, input.UV).xyz * 2.0 - 1.0;
        float3 T = normalize(input.Tangent.xyz);
        float3 B = normalize(cross(N, T)) * input.Tangent.w;
        float3x3 TBN = float3x3(T, B, N);
        N = normalize(mul(tangentNormal, TBN));
    }

    GBufferOutput output;
    output.Position = float4(input.WorldPos, 1.0);
    output.Normal   = float4(N, 0.0);
    output.Albedo   = float4(baseColor.rgb, metallic);
    output.Material = float4(roughness, ao, 0.0, 0.0);
    output.Emissive = float4(emissive, 1.0);
    return output;
}