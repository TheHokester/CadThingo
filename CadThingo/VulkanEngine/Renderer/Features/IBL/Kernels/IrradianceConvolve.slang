// Diffuse-IBL bake. For each output cube texel, integrate the Lambert
// hemisphere of incoming radiance from envCube and store the result in
// irradianceCube. Output face size is small (32²) so the sample-delta walk is
// affordable even at 0.025rad spacing (~15k samples per output texel).

import IblCommon;

[[vk::binding(0, 0)]] SamplerCube                                            envCube;
[[vk::binding(1, 0)]] [[vk::image_format("rgba16f")]] RWTexture2DArray<float4> outIrradiance;

[[vk::push_constant]]
cbuffer PC
{
    uint faceSize;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= faceSize || id.y >= faceSize) return;

    float2 uv01 = (float2(id.xy) + 0.5) / float(faceSize);
    float3 N    = cubeFaceDir(id.z, uv01);

    // Build a tangent basis around N. Bias the "up" reference vector away from
    // N so the cross product never degenerates on poles.
    float3 up    = abs(N.y) < 0.999 ? float3(0, 1, 0) : float3(0, 0, 1);
    float3 right = normalize(cross(up, N));
    up           = cross(N, right);

    float3 irradiance = float3(0, 0, 0);
    float  nSamples   = 0.0;
    const float sampleDelta = 0.025;

    for (float phi = 0.0; phi < TWO_PI; phi += sampleDelta)
    {
        for (float theta = 0.0; theta < 0.5 * PI; theta += sampleDelta)
        {
            // Spherical (tangent-space) → world by rotating into the (right, up, N) basis.
            float3 tangentSample = float3(sin(theta) * cos(phi),
                                          sin(theta) * sin(phi),
                                          cos(theta));
            float3 sampleVec = tangentSample.x * right
                             + tangentSample.y * up
                             + tangentSample.z * N;

            irradiance += envCube.SampleLevel(sampleVec, 0).rgb * cos(theta) * sin(theta);
            nSamples   += 1.0;
        }
    }

    // Riemann sum × PI / N. The PI factor closes the Lambert BRDF normalization so
    // the runtime shader can multiply the sampled irradiance directly by baseColor.
    irradiance = PI * irradiance * (1.0 / nSamples);
    outIrradiance[int3(id.xy, id.z)] = float4(irradiance, 1.0);
}
