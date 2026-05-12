// Final post-processing pass: samples the linear HDR scene target written by
// the lighting pass (and later the transparent forward+ pass), applies the
// selected tone-map operator + inverse gamma, and outputs LDR sRGB-ready
// color for the swapchain blit. Push constants carry the two tunables — they
// change rarely so they don't belong on the per-frame UBO.

struct VSOutput {
    float4 Position : SV_POSITION;
    float2 UV       : TEXCOORD0;
};

struct TonemapParams {
    float exposure;
    float gamma;
};

[[vk::binding(0, 0)]] Sampler2D hdrInput;
[[vk::push_constant]] TonemapParams params;

// Specialization constant — picked up via VkSpecializationInfo at pipeline build.
// 0 = Reinhard (cheap, low contrast, preserves the pre-refactor look).
// 1 = Filmic   (Narkowicz ACES-fitted curve — soft shoulder, stronger contrast).
// Flip Renderer.tonemapOperator and rebuild the tonemap pipeline.
[[vk::constant_id(0)]] const uint TONEMAP_OPERATOR = 0;

static const uint TONEMAP_REINHARD = 0;
static const uint TONEMAP_FILMIC   = 1;

float3 ReinhardTonemap(float3 hdr) {
    return hdr / (hdr + float3(1.0, 1.0, 1.0));
}

// Narkowicz 2015 ACES-fitted approximation — the standard cheap "filmic" curve.
// Already clamped to [0,1] by the saturate.
float3 AcesFilmicTonemap(float3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}


[[shader("vertex")]]
VSOutput VSMain(uint vertexID : SV_VertexID)
{
    float2 positions[3] = {
        float2(-1.0, -1.0),
        float2( 3.0, -1.0),
        float2(-1.0,  3.0),
    };
    float2 uvs[3] = {
        float2(0.0, 0.0),
        float2(2.0, 0.0),
        float2(0.0, 2.0),
    };

    VSOutput output;
    output.Position = float4(positions[vertexID], 0.0, 1.0);
    output.UV       = uvs[vertexID];
    return output;
}

[[shader("fragment")]]
float4 PSMain(VSOutput input) : SV_TARGET
{
    float3 hdr = hdrInput.Sample(input.UV).rgb * params.exposure;

    // Spec-constant branch — folds to a single operator at pipeline build.
    float3 mapped;
    if (TONEMAP_OPERATOR == TONEMAP_FILMIC)
        mapped = AcesFilmicTonemap(hdr);
    else
        mapped = ReinhardTonemap(hdr);

    // Inverse gamma for sRGB display. Output is RGBA8_UNORM.
    float invG = 1.0 / params.gamma;
    mapped = pow(mapped, float3(invG, invG, invG));

    return float4(mapped, 1.0);
}