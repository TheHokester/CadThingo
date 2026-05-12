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

                // SV_InstanceID is the *relative* instance id within a draw call (HLSL
                // semantic). Slang lowers it to gl_InstanceIndex - gl_BaseInstance, which
                // is always 0 here (instanceCount=1). We need the *absolute* index to look
                // up instances[firstInstance], so re-add SV_StartInstanceLocation (=
                // gl_BaseInstance, the firstInstance the indirect cmd was issued with).
[[shader("vertex")]]
VSOutput VSMain(VSInput input,
                uint instanceIndex : SV_InstanceID,
                uint baseInstance  : SV_StartInstanceLocation)
{
    InstanceData inst = instances[instanceIndex + baseInstance];

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

// DEBUG: visualize MaterialIndex directly. If every prim shows ONE color, the
// VS→FS materialIndex pipeline is broken (likely firstInstance). If different
// prims show different colors, materialIndex is correctly delivered and the bug
// is downstream in materials[]/textures[]. Set to 0 to disable.
#define DEBUG_MATERIAL_INDEX 0

[[shader("fragment")]]
GBufferOutput PSMain(VSOutput input)
{
#if DEBUG_MATERIAL_INDEX
    GBufferOutput dbg;
    uint mi = input.MaterialIndex;
    float r = ((mi >> 0) & 1u) ? 1.0 : 0.0;
    float g = ((mi >> 1) & 1u) ? 1.0 : 0.0;
    float b = ((mi >> 2) & 1u) ? 1.0 : 0.0;
    float scale = 0.3 + 0.7 * (((mi >> 3) & 1u) ? 1.0 : 0.0);
    dbg.Position = float4(input.WorldPos, 1.0);
    dbg.Normal   = float4(normalize(input.Normal), 0.0);
    dbg.Albedo   = float4(r * scale, g * scale, b * scale, 0.0);
    dbg.Material = float4(0.5, 1.0, 0.0, 0.0);
    dbg.Emissive = float4(r * scale, g * scale, b * scale, 1.0); // emissive too so unlit areas still show
    return dbg;
#endif

    PbrMaterial mat = materials[input.MaterialIndex];

    // Sampler 0 is the shared linear-repeat default. All bindless textures use it for now.
    SamplerState s = samplers[0];

    // NonUniformResourceIndex is required for descriptor-array indices that may vary
    // across a wave (different primitives in the same wave can carry different material
    // indices). Without it, Vulkan leaves the result undefined and most drivers sample
    // the first lane's texture for the whole wave — i.e. neighbouring prims/models
    // appear to share textures.
    float4 baseColor       = textures[NonUniformResourceIndex(mat.baseColorTex)].Sample(s, input.UV) * mat.baseColorFactor;

    // MASK alpha test (bit 0 of flags). BLEND materials (bit 2) route to the dedicated
    // transparent pass and shouldn't reach the geometry pass after the partition lands —
    // but their glTF alphaCutoff defaults to 0 so this discard wouldn't fire for them
    // even if they did.
    if ((mat.flags & 1u) != 0u && baseColor.a < mat.alphaCutoff) discard;

    float2 metallicRoughness = textures[NonUniformResourceIndex(mat.physicalDescriptorTex)].Sample(s, input.UV).bg;
    float metallic         = metallicRoughness.x * mat.metallicFactor;
    float roughness        = metallicRoughness.y * mat.roughnessFactor;
    float ao               = textures[NonUniformResourceIndex(mat.occlusionTex)].Sample(s, input.UV).r;
    float3 emissive        = textures[NonUniformResourceIndex(mat.emissiveTex)].Sample(s, input.UV).rgb;

    float3 N = normalize(input.Normal);
    {
        float3 tangentNormal = textures[NonUniformResourceIndex(mat.normalTex)].Sample(s, input.UV).xyz * 2.0 - 1.0;
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