// Unwrap an equirectangular HDR environment into a cubemap. Run once per face
// (one dispatch covers all 6 layers via the z dimension of the thread grid).
// Output is the base mip of envCube; mip chain is generated CPU-side via blit.

#include "IblCommon.slang"

[[vk::binding(0, 0)]] Sampler2D                                              equirect;
[[vk::binding(1, 0)]] [[vk::image_format("rgba16f")]] RWTexture2DArray<float4> outCube;

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
    float3 dir  = cubeFaceDir(id.z, uv01);
    float2 eq   = equirectUV(dir);
    float4 c    = equirect.SampleLevel(eq, 0);
    outCube[int3(id.xy, id.z)] = c;
}
