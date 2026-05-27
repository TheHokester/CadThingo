// Specular-IBL bake. For one cube mip per dispatch, run GGX importance
// sampling at the mip's roughness value and accumulate into prefilteredCube.
//
// Filtered-importance sampling: sample mip is biased by the PDF so low-PDF
// taps fall back to coarser env mips, removing the sparkle/firefly artifacts
// you get from straight env-mip-0 reads at high roughness.

import IblCommon;

[[vk::binding(0, 0)]] SamplerCube                                            envCube;
[[vk::binding(1, 0)]] [[vk::image_format("rgba16f")]] RWTexture2DArray<float4> outMip;

[[vk::push_constant]]
cbuffer PC
{
    float roughness;     // 0..1, distributed mip / (numMips-1) CPU-side
    uint  mipFaceSize;   // width = height of this output mip's face
    uint  envCubeSize;   // base env face size, for filtered-IS mip pick
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= mipFaceSize || id.y >= mipFaceSize) return;

    float2 uv01 = (float2(id.xy) + 0.5) / float(mipFaceSize);
    float3 N    = cubeFaceDir(id.z, uv01);
    // Split-sum approximation collapses R/V to N — the cone is symmetric
    // around the surface normal, not the view vector.
    float3 R = N;
    float3 V = R;

    const uint SAMPLE_COUNT = 1024u;
    float  totalWeight = 0.0;
    float3 prefiltered = float3(0, 0, 0);

    for (uint i = 0u; i < SAMPLE_COUNT; i++)
    {
        float2 Xi = Hammersley(i, SAMPLE_COUNT);
        float3 H  = ImportanceSampleGGX(Xi, N, roughness);
        float3 L  = normalize(2.0 * dot(V, H) * H - V);

        float NdotL = max(dot(N, L), 0.0);
        if (NdotL > 0.0)
        {
            // Filtered-importance sampling (Krivanek/Colbert 2008): tap a mip
            // sized to the sample's solid angle so high-roughness sparkle
            // doesn't escape through the alias filter.
            float NdotH = max(dot(N, H), 0.0);
            float HdotV = max(dot(H, V), 0.0);
            float a     = roughness * roughness;
            float a2    = a * a;
            float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
            float D     = a2 / (PI * denom * denom);
            float pdf   = (D * NdotH) / (4.0 * HdotV + 1e-4) + 1e-4;

            float saTexel  = 4.0 * PI / (6.0 * float(envCubeSize) * float(envCubeSize));
            float saSample = 1.0 / (float(SAMPLE_COUNT) * pdf);
            float mipLevel = roughness == 0.0 ? 0.0 : 0.5 * log2(saSample / saTexel);

            prefiltered += envCube.SampleLevel(L, mipLevel).rgb * NdotL;
            totalWeight += NdotL;
        }
    }
    prefiltered /= max(totalWeight, 1e-4);
    outMip[int3(id.xy, id.z)] = float4(prefiltered, 1.0);
}
