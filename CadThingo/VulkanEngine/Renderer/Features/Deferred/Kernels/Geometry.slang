import SceneBindings; // scene set 0 (materials/instances/textures/samplers); exports CommonTypes

// Per-pass constants in the reserved (0,0) dynamic slot (descriptor-system section 7).
// Model matrix lives in the per-instance SSBO (CommonTypes::InstanceData).
struct FrameUBO {
    float4x4 view;
    float4x4 proj;
}

struct VSInput {
    float3 Position : POSITION;
    float2 Normal   : NORMAL;   // octahedral (R16G16_UNORM)
    float2 UV       : TEXCOORD0;
};

struct VSOutput {
    float4 Position      : SV_POSITION;
    float3 WorldPos      : POSITION;
    float3 Normal        : NORMAL;
    float2 UV            : TEXCOORD0;
    nointerpolation uint MaterialIndex : TEXCOORD1;
};

struct GBufferOutput {
    float4 Position : SV_TARGET0;
    float4 Normal   : SV_TARGET1;
    float4 Albedo   : SV_TARGET2;
    float4 Material : SV_TARGET3;
    float4 Emissive : SV_TARGET4;
}

[[vk::binding(0,0)]] ConstantBuffer<FrameUBO> frame;

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
    InstanceData inst = sceneInstances[instanceIndex + baseInstance];

    float4 worldPos = mul(inst.model, float4(input.Position, 1.0));

    VSOutput output;
    output.Position      = mul(frame.proj, mul(frame.view, worldPos));
    output.WorldPos      = worldPos.xyz;
    output.Normal        = normalize(mul((float3x3)inst.model, octDecodeNormal(input.Normal)));
    output.UV            = input.UV;
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

    PbrMaterial mat = sceneMaterials[input.MaterialIndex];

    // Sampler 0 is the shared linear-repeat default. All bindless textures use it for now.
    SamplerState s = sceneSamplers[0];


    float4 baseColor       = sceneTextures[NonUniformResourceIndex(mat.baseColorTex)].Sample(s, input.UV) * mat.baseColorFactor;

    // MASK alpha test (bit 0 of flags). discards if geometry that should be opaque is transparent. 
    if ((mat.flags & 1u) != 0u && baseColor.a < mat.alphaCutoff) discard;

    float2 metallicRoughness = sceneTextures[NonUniformResourceIndex(mat.physicalDescriptorTex)].Sample(s, input.UV).rg;
    float metallic         = metallicRoughness.x * mat.metallicFactor;
    float roughness        = metallicRoughness.y * mat.roughnessFactor;
    float ao               = sceneTextures[NonUniformResourceIndex(mat.occlusionTex)].Sample(s, input.UV).r;
    float3 emissive        = sceneTextures[NonUniformResourceIndex(mat.emissiveTex)].Sample(s, input.UV).rgb * mat.emissiveFactor;

    float3 N = normalize(input.Normal);
    {
        float3 tangentNormal = decodeTangentNormal(sceneTextures[NonUniformResourceIndex(mat.normalTex)].Sample(s, input.UV).rg);
        N = applyNormalMap(N, input.WorldPos, input.UV, tangentNormal);
    }
    //writes gbuffer
    GBufferOutput output;
    output.Position = float4(input.WorldPos, 1.0);
    output.Normal   = float4(N, 0.0);
    output.Albedo   = float4(baseColor.rgb, metallic);
    output.Material = float4(roughness, ao, 0.0, 0.0);
    output.Emissive = float4(emissive, 1.0);
    return output;
}