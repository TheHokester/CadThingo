// IblCommon — shared utilities for the IBL bake compute shaders. Cube face math,
// low-discrepancy sequences, GGX importance sampling. All routines are pure /
// re-entrant and depend only on inputs — no descriptor reads happen here.
//
// `module IblCommon;` consumed via `import IblCommon;` by EquirectToCube,
// IrradianceConvolve, PrefilterEnv, and BrdfLutGen. PI / TWO_PI / INV_PI now
// come from CommonTypes (re-exported) instead of being redeclared here.

module IblCommon;

__exported import CommonTypes;

// Standard Vulkan / OpenGL cubemap face conventions. `face` is the array
// layer (0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z) and `uv01` is the in-face texel
// center in [0,1]. Returned direction is normalized world-space.
public float3 cubeFaceDir(uint face, float2 uv01)
{
    float2 uv = uv01 * 2.0 - 1.0;
    float a = uv.x;
    float b = uv.y;
    float3 d;
    switch (face)
    {
        case 0u: d = float3( 1.0,  -b,  -a); break; // +X
        case 1u: d = float3(-1.0,  -b,   a); break; // -X
        case 2u: d = float3(   a, 1.0,   b); break; // +Y
        case 3u: d = float3(   a,-1.0,  -b); break; // -Y
        case 4u: d = float3(   a,  -b, 1.0); break; // +Z
        default: d = float3(  -a,  -b,-1.0); break; // -Z
    }
    return normalize(d);
}

// Equirectangular UV from a unit direction. Wraps longitude through atan2 and
// maps latitude through asin → both into [0,1].
public float2 equirectUV(float3 d)
{
    float lon = atan2(d.z, d.x);
    float lat = asin(clamp(d.y, -1.0, 1.0));
    return float2(lon / TWO_PI + 0.5, lat / PI + 0.5);
}

// Van der Corput radical inverse — the second axis of the Hammersley sequence.
public float radicalInverseVdC(uint bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

public float2 Hammersley(uint i, uint N)
{
    return float2(float(i) / float(N), radicalInverseVdC(i));
}

// GGX-importance-sampled half vector. Tangent → world via an arbitrary basis
// around N. Used by both prefilter and BRDF LUT bake.
public float3 ImportanceSampleGGX(float2 Xi, float3 N, float roughness)
{
    float a   = roughness * roughness;
    float phi = TWO_PI * Xi.x;
    float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    float3 H = float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

    float3 up = abs(N.z) < 0.999 ? float3(0, 0, 1) : float3(1, 0, 0);
    float3 T  = normalize(cross(up, N));
    float3 B  = cross(N, T);

    return normalize(T * H.x + B * H.y + N * H.z);
}

// Smith geometry term, IBL form. The direct-lighting variant in PbrUtils uses
// k = (roughness+1)²/8; the IBL split-sum derivation calls for k = α²/2 (see
// Karis 2013 "Real Shading in UE4"). Don't reuse the direct-light G here.
public float G_SchlickGGX_IBL(float NdotV, float roughness)
{
    float k = (roughness * roughness) / 2.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

public float G_Smith_IBL(float NdotV, float NdotL, float roughness)
{
    return G_SchlickGGX_IBL(NdotV, roughness) * G_SchlickGGX_IBL(NdotL, roughness);
}
