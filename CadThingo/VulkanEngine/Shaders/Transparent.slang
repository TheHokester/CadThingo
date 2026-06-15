// Forward+ transparent pass — samples materials directly and uses the same
// tile light list + ray-traced shadows as the deferred PBR pass. Renders
// after the lighting pass into HDRColor with src-alpha / one-minus-src-alpha
// blending, depth-tested LESS_EQUAL against the geometry pass's depth buffer
// (no depth write so multiple transparent layers can stack).

import PbrUtils;   // BRDF/light/IBL math + re-exported CommonTypes

struct FrameUBO {
    float4x4 view;
    float4x4 proj;
    float4 camPos;                       // .w unused
    uint   lightCount;
    uint   tileCountX;
    uint   tileCountY;
    uint   _pad0;
    float2 screenSize;
    // IBL params — repurposed from former trailing pad slots (same bytes as
    // PbrShader.slang::LightingFrameUBO so the CPU update path stays symmetric).
    float  prefilteredCubeMipLevels;
    float  scaleIBLAmbient;
    // Reflection-probe cluster dims + mip count — mirrors the deferred path so
    // both shaders compute the same cluster index for the same fragment.
    uint   probeClusterDimsX;
    uint   probeClusterDimsY;
    uint   probeClusterDimsZ;
    float  probeMipLevels;
};

// ProbeRecord / ProbeClusterRange now come from CommonTypes (via import PbrUtils).

struct DrawPC {
    float4x4 model;
    uint     materialIndex;
    uint     _pad0;
    uint     _pad1;
    uint     _pad2;
};

// TILE_SIZE_PX / MAX_LIGHTS_PER_TILE come from CommonTypes (via import PbrUtils).

// Set 0 — own UBO + pointers to lighting buffers owned by other pipelines.
[[vk::binding(0, 0)]] ConstantBuffer<FrameUBO>              frame;
[[vk::binding(1, 0)]] StructuredBuffer<PbrLight>            lights;
[[vk::binding(2, 0)]] RaytracingAccelerationStructure       tlas;
[[vk::binding(3, 0)]] StructuredBuffer<uint>                tileLightCount;
[[vk::binding(4, 0)]] StructuredBuffer<uint>                tileLightIndices;
// IBL split-sum — same handles as PbrShader.slang. Pipelines own separate
// descriptor sets but the underlying VkImage / Sampler resources are shared.
[[vk::binding(5, 0)]] SamplerCube                           irradianceCube;
[[vk::binding(6, 0)]] SamplerCube                           prefilteredCube;
[[vk::binding(7, 0)]] Sampler2D                             brdfLut;

// Probe bindings mirror PbrShader.slang set 0 slots 8-11. Empty / unregistered
// slots leave clusterRange.count == 0 so the loop is a no-op and the global
// IBL fallback (via the lerp coverage path) takes over cleanly.
[[vk::binding(8,  0)]] SamplerCubeArray                     probeCubeArray;
[[vk::binding(9,  0)]] StructuredBuffer<ProbeRecord>        probes;
[[vk::binding(10, 0)]] StructuredBuffer<ProbeClusterRange>  probeClusterRange;
[[vk::binding(11, 0)]] StructuredBuffer<uint>               probeIndexList;

// Set 1 — bindless materials/instances/textures/samplers (shared layout with GeometryPipeline).
[[vk::binding(0, 1)]] StructuredBuffer<PbrMaterial>         materials;
[[vk::binding(1, 1)]] StructuredBuffer<InstanceData>        instances;     // layout parity; unused here
[[vk::binding(2, 1)]] Texture2D                             textures[];
[[vk::binding(3, 1)]] SamplerState                          samplers[];

[[vk::push_constant]] DrawPC draw;

// Mirror of PbrShader.slang's SOFT_SHADOWS spec constant; same toggle path.
[[vk::constant_id(0)]] const bool SOFT_SHADOWS = false;

//  Shadow helpers (copy of PbrShader.slang's; keeps each shader self-contained) 

static const float SHADOW_RAY_EPS  = 0.002;
static const int   PENUMBRA_SAMPLES = 8;
static const float GOLDEN_ANGLE     = 2.39996323;
static const float MAX_CONE_HA      = 0.25;

bool isOccluded(float3 origin, float3 dir, float tMax)
{
    RayDesc ray;
    ray.Origin    = origin;
    ray.Direction = dir;
    ray.TMin      = 0.0;
    ray.TMax      = tMax;

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    q.TraceRayInline(tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, ray);
    q.Proceed();
    return q.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
}

float blockerSearch(float3 origin, float3 dir, float tMax)
{
    RayDesc ray;
    ray.Origin    = origin;
    ray.Direction = dir;
    ray.TMin      = 0.0;
    ray.TMax      = tMax;

    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(tlas, RAY_FLAG_NONE, 0xFF, ray);
    while (q.Proceed()) {}
    return (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) ? q.CommittedRayT() : -1.0;
}

void buildLightBasis(float3 L, out float3 T, out float3 B)
{
    float3 up = abs(L.y) < 0.99 ? float3(0, 1, 0) : float3(1, 0, 0);
    T = normalize(cross(up, L));
    B = cross(L, T);
}

float pixelJitter(float2 px)
{
    return frac(sin(dot(px, float2(12.9898, 78.233))) * 43758.5453) * 6.2831853;
}

float softShadow(float3 origin, float3 L, float distanceToLight, float lightRadius, float jitterAngle)
{
    float tMax = max(distanceToLight - SHADOW_RAY_EPS, 0.0);

    float tBlocker = blockerSearch(origin, L, tMax);
    if (tBlocker < 0.0)        return 1.0;
    if (lightRadius <= 1e-4)   return 0.0;

    float coneHalfAngle = (distanceToLight > 1e5)
        ? lightRadius
        : (lightRadius * tBlocker) / (distanceToLight * max(distanceToLight - tBlocker, 1e-4));
    coneHalfAngle = min(coneHalfAngle, MAX_CONE_HA);

    float3 T, B;
    buildLightBasis(L, T, B);

    float unblocked = 0.0;
    for (int i = 0; i < PENUMBRA_SAMPLES; i++) {
        float r     = sqrt((i + 0.5) / float(PENUMBRA_SAMPLES));
        float theta = i * GOLDEN_ANGLE + jitterAngle;
        float2 disk = float2(cos(theta), sin(theta)) * r * coneHalfAngle;

        float3 dir = normalize(L + T * disk.x + B * disk.y);
        if (!isOccluded(origin, dir, tMax)) unblocked += 1.0;
    }
    return unblocked / float(PENUMBRA_SAMPLES);
}

// Vertex / fragment

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

[[shader("vertex")]]
VSOutput VSMain(VSInput input)
{
    float4 worldPos = mul(draw.model, float4(input.Position, 1.0));

    VSOutput output;
    output.Position      = mul(frame.proj, mul(frame.view, worldPos));
    output.WorldPos      = worldPos.xyz;
    output.Normal        = normalize(mul((float3x3)draw.model, input.Normal));
    output.UV            = input.UV;
    output.Tangent       = input.Tangent;
    output.MaterialIndex = draw.materialIndex;
    return output;
}

[[shader("fragment")]]
float4 PSMain(VSOutput input) : SV_TARGET
{
    PbrMaterial mat = materials[input.MaterialIndex];
    SamplerState s  = samplers[0];

    float4 baseColor         = textures[NonUniformResourceIndex(mat.baseColorTex)].Sample(s, input.UV) * mat.baseColorFactor;
    float2 metallicRoughness = textures[NonUniformResourceIndex(mat.physicalDescriptorTex)].Sample(s, input.UV).rg;
    float  metallic          = metallicRoughness.x * mat.metallicFactor;
    float  roughness         = metallicRoughness.y * mat.roughnessFactor;
    float  ao                = textures[NonUniformResourceIndex(mat.occlusionTex)].Sample(s, input.UV).r;
    float3 emissiveTex       = textures[NonUniformResourceIndex(mat.emissiveTex)].Sample(s, input.UV).rgb * mat.emissiveFactor;

    // Tangent-space normal map (same as geometry pass).
    float3 N = normalize(input.Normal);
    {
        float3 tangentNormal = decodeTangentNormal(textures[NonUniformResourceIndex(mat.normalTex)].Sample(s, input.UV).rg);
        float3 T = normalize(input.Tangent.xyz);
        float3 B = normalize(cross(N, T)) * input.Tangent.w;
        float3x3 TBN = float3x3(T, B, N);
        N = normalize(mul(tangentNormal, TBN));
    }

    float3 WorldPos = input.WorldPos;
    float3 V = normalize(frame.camPos.xyz - WorldPos);
    float3 R = reflect(-V, N);

    float3 F0 = float3(0.04, 0.04, 0.04);
    F0 = lerp(F0, baseColor.rgb, metallic);

    float3 Lo = float3(0.0, 0.0, 0.0);

    // Same tile-light walk as PbrShader.slang. SV_Position.xy is pixel-space.
    uint tileX   = (uint)input.Position.x / TILE_SIZE_PX;
    uint tileY   = (uint)input.Position.y / TILE_SIZE_PX;
    uint tileIdx = tileY * frame.tileCountX + tileX;
    uint perTile = (tileX < frame.tileCountX && tileY < frame.tileCountY)
                 ? tileLightCount[tileIdx] : 0u;

    for (uint t = 0u; t < perTile; t++) {
        uint     lightIdx = tileLightIndices[tileIdx * MAX_LIGHTS_PER_TILE + t];
        PbrLight lt       = lights[lightIdx];

        uint   type      = (uint)lt.directionType.w;
        float3 lColor    = lt.colorIntensity.rgb;
        float  intensity = lt.colorIntensity.a;
        if (intensity <= 0.0) continue;

        float3 L;
        float  distance;
        float  attenuation;
        float  spotMask = 1.0;

        if (type == LIGHT_TYPE_DIRECTIONAL) {
            L           = normalize(-lt.directionType.xyz);
            distance    = 1e6;
            attenuation = 1.0;
        } else {
            float3 toLight = lt.positionRange.xyz - WorldPos;
            distance       = length(toLight);
            L              = toLight / max(distance, 1e-4);
            attenuation    = 1.0 / max(distance * distance, 1e-4);

            float range = lt.positionRange.w;
            if (range > 0.0) {
                float ta = saturate(1.0 - pow(distance / range, 4.0));
                attenuation *= ta * ta;
            }

            if (type == LIGHT_TYPE_SPOT) {
                float3 spotDir = normalize(lt.directionType.xyz);
                float  cosA    = dot(-L, spotDir);
                float  inner   = lt.spotCones.x;
                float  outer   = lt.spotCones.y;
                spotMask       = smoothstep(outer, inner, cosA);
                if (spotMask <= 0.0) continue;
            }
        }

        float3 radiance = lColor * intensity * attenuation * spotMask;

        bool castShadows = lt.spotCones.z > 0.5;
        if (castShadows) {
            float3 shadowOrigin = WorldPos + N * SHADOW_RAY_EPS;
            float  tMax         = max(distance - SHADOW_RAY_EPS, 0.0);

            if (SOFT_SHADOWS) {
                float lightRadius = lt.spotCones.w;
                float jitter      = pixelJitter(input.Position.xy);
                float visibility  = softShadow(shadowOrigin, L, distance, lightRadius, jitter);
                if (visibility <= 0.0) continue;
                radiance *= visibility;
            } else {
                if (isOccluded(shadowOrigin, L, tMax)) continue;
            }
        }

        float3 H = normalize(V + L);
        float NdotL = max(dot(N, L), 0.0);
        float NdotV = max(dot(N, V), 0.0);
        float NdotH = max(dot(N, H), 0.0);
        float HdotV = max(dot(H, V), 0.0);

        float D = DistributionGGX(NdotH, roughness);
        float G = GeometrySmith(NdotV, NdotL, roughness);
        float3 F = FresnelSchlick(HdotV, F0);

        float3 specular = (D * G * F) / (4.0 * NdotV * NdotL + 0.0001);

        float3 kS = F;
        float3 kD = float3(1.0, 1.0, 1.0) - kS;
        kD *= 1.0 - metallic;

        Lo += (kD * baseColor.rgb / baseColor.a * PI + specular) * radiance * NdotL;
    }

    // Split-sum IBL — shared evaluator in PbrUtils, same call the deferred pass
    // makes. Zero contribution until an environment is baked; the prior 0.03
    // floor is intentionally gone because transparents need physically-correct
    // reflection of the env once it lands.
    float3 ambient = evaluateIBL(
        N, V, R, baseColor.rgb, metallic, roughness, ao, F0,
        WorldPos, uint2(input.Position.xy),
        irradianceCube, prefilteredCube, brdfLut, probeCubeArray,
        probes, probeClusterRange, probeIndexList,
        frame.prefilteredCubeMipLevels, frame.scaleIBLAmbient,
        frame.probeClusterDimsX, frame.probeClusterDimsY, frame.probeMipLevels);
    float3 color   = ambient + Lo + emissiveTex;

    // Fresnel-modulated alpha: at grazing angles the surface becomes more reflective,
    // so push alpha toward 1 there. Keeps specular highlights from being damped to
    // matte by the SRC_ALPHA blend on low-alpha clearcoats/glass.
    float fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.0);
    float outAlpha = saturate(baseColor.a + (1.0 - baseColor.a) * fresnel);

    // Output linear HDR. Tonemap pass downstream applies exposure + Reinhard + inverse gamma.
    return float4(color, outAlpha);
}