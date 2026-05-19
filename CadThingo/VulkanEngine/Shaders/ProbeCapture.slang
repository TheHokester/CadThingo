// ProbeCapture.slang
//
// Multiview capture shader for runtime reflection probes. Renders all 6
// cubemap faces in one draw via SV_ViewID — the pipeline's VkRenderingInfo
// carries viewMask = 0x3F so the rasterizer fans each primitive across all
// six layers, and SV_ViewID picks the matching per-face view matrix.
//
// Lighting is deliberately simplified: a single hardcoded directional "sun"
// + per-material base color. Specular IBL is excluded to avoid a feedback
// loop (probes can't sample themselves), and direct lights / shadows are
// deferred to a later phase so the capture stays cheap. The prefilter pass
// downstream sees raw lit color, which is all the split-sum approximation
// needs as input.

#include "PbrUtils.slang"

struct CaptureUBO {
    float4x4 views[6];   // one per cube face, picked via SV_ViewID
    float4x4 proj;
};
struct PushConst {
    float4x4 model;
    uint     materialIndex;
    uint     _pad0;
    uint     _pad1;
    uint     _pad2;
}; 

struct VSInput {
    float3 Position : POSITION;
    float3 Normal   : NORMAL;
    float2 UV       : TEXCOORD0;
    float4 Tangent  : TANGENT;
};

struct VSOutput {
    float4 Position : SV_POSITION;
    float3 WorldPos : POSITION;
    float3 Normal   : NORMAL;
    float2 UV       : TEXCOORD0;
};
[[vk::push_constant]] PushConst pc;
[[vk::binding(0, 0)]] ConstantBuffer<CaptureUBO> frame;

// Set 1 mirrors the bindless layout owned by ResourceManager so we can reuse
// the same descriptor set this whole pass long. We only sample baseColor here;
// the metallic/normal/etc. fields stay unused.
[[vk::binding(0, 1)]] StructuredBuffer<PbrMaterial>  materials;
[[vk::binding(1, 1)]] StructuredBuffer<InstanceData> instances;
[[vk::binding(2, 1)]] Texture2D                      textures[];
[[vk::binding(3, 1)]] SamplerState                   samplers[];

[[shader("vertex")]]
VSOutput VSMain(VSInput input, uint viewIndex : SV_ViewID)
{
    float4 worldPos = mul(pc.model, float4(input.Position, 1.0));

    VSOutput o;
    o.Position = mul(frame.proj, mul(frame.views[viewIndex], worldPos));
    o.WorldPos = worldPos.xyz;
    // World-space normal — capture shader skips normal maps for cost, the
    // mesh normal is sufficient input for a prefiltered specular bake.
    o.Normal   = normalize(mul((float3x3)pc.model, input.Normal));
    o.UV       = input.UV;
    return o;
}

[[shader("fragment")]]
float4 PSMain(VSOutput input) : SV_TARGET
{
    PbrMaterial mat = materials[pc.materialIndex];
    SamplerState s  = samplers[0];

    float4 baseColor = textures[NonUniformResourceIndex(mat.baseColorTex)].Sample(s, input.UV)
                       * mat.baseColorFactor;

    // MASK alpha test — discards transparent texels so silhouettes survive
    // the capture. BLEND materials would need a separate pass; for v1 the
    // probe just sees them as opaque.
    if ((mat.flags & 1u) != 0u && baseColor.a < mat.alphaCutoff) discard;

    // Hardcoded sun direction (world space, normalized). Phase-4 probes ship
    // with placeholder lighting; the full direct-light + irradiance pass lands
    // when we wire the scene light SSBO into this pipeline.
    const float3 sunDir   = normalize(float3(0.45, 0.9, 0.25));
    const float3 sunColor = float3(1.0, 0.95, 0.85);
    const float  ambient  = 0.15;

    float3 N = normalize(input.Normal);
    float  NdotL = saturate(dot(N, sunDir));
    float3 lit = baseColor.rgb * (sunColor * NdotL + ambient);

    return float4(lit, 1.0);
}
