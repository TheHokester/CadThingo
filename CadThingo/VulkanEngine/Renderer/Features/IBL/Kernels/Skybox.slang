// Fullscreen skybox draw. Runs between the deferred lighting pass and the
// transparent forward+ pass; samples envCube along a per-pixel view ray
// reconstructed from inverse-view-projection. Depth test is EQUAL @ 1.0 so it
// only writes where geometry didn't — no depth write, no blending.

struct SkyboxUBO {
    float4x4 invViewProj;   // inverse of (proj * view); reconstructs world dir from NDC
    float4   camPos;        // .xyz = camera world pos, .w unused
    float    intensity;     // matches scaleIBLAmbient; scales envCube sample
    float    _pad0;
    float    _pad1;
    float    _pad2;
};

import FeatureEnv; // feature set 4: envCube

[[vk::binding(0, 0)]] ConstantBuffer<SkyboxUBO> ubo;

struct VSOut {
    float4 Position : SV_POSITION;  // clip space — z=1.0 lands on far plane
    float2 NDC      : TEXCOORD0;    // raw [-1,1] NDC for ray reconstruction
};

[[shader("vertex")]]
VSOut VSMain(uint id : SV_VertexID)
{
    // Standard fullscreen-triangle covering NDC [-1,1] x [-1,1].
    float2 positions[3] = {
        float2(-1.0, -1.0),
        float2( 3.0, -1.0),
        float2(-1.0,  3.0),
    };

    VSOut o;
    // Write clip-space z=w=1 so the perspective divide leaves frag depth = 1.0
    // → matches the depth buffer's cleared "no geometry" value for sky pixels.
    o.Position = float4(positions[id], 1.0, 1.0);
    o.NDC      = positions[id];
    return o;
}

[[shader("fragment")]]
float4 PSMain(VSOut input) : SV_TARGET
{
    // Reconstruct a far-plane world point from the pixel's NDC, then subtract
    // the camera position to get the view ray direction. Normalize so envCube
    // sampling sees a unit vector (which is what SamplerCube expects anyway).
    float4 worldH = mul(ubo.invViewProj, float4(input.NDC, 1.0, 1.0));
    float3 world  = worldH.xyz / worldH.w;
    float3 dir    = normalize(world - ubo.camPos.xyz);

    // Sample mip 0 of the env cube - we want the sharp environment, not the
    // prefiltered roughness chain. The cube is already in HDR-linear space so
    // the downstream tonemap pass handles exposure + gamma uniformly.
    float3 color = envCube.SampleLevel(dir, 0).rgb * ubo.intensity;
    return float4(color, 1.0);
}
