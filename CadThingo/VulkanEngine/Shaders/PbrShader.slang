// Output from vertex shader / Input to fragment shader - Interpolated data
struct LightingVSOutput {
    float4 Position : SV_POSITION; // Required clip space position for rasterization
    float2 UV : TEXCOORD0;         // Texture coordinates (interpolated)
};

// Per-frame shading constants. The light array lives in a separate
// StructuredBuffer<PbrLight> at binding 1; tileCounts + screenSize key the
// per-tile light list (Phase 4).
struct LightingFrameUBO {
    float4 camPos;                      // Camera position for view-dependent effects
    float exposure;                     // HDR exposure control
    float gamma;                        // Gamma correction value (typically 2.2)
    float prefilteredCubeMipLevels;     // IBL prefiltered environment map mip levels
    float scaleIBLAmbient;              // IBL ambient contribution scale
    uint  lightCount;                   // Number of valid entries in lights[]
    uint  tileCountX;                   // Phase 4 — tiles across the screen
    uint  tileCountY;                   // Phase 4 — tiles down the screen
    uint  _pad0;
    float2 screenSize;                  // pixels (matches CPU swapChainExtent)
    uint  _pad1;
    uint  _pad2;
};

#include "PbrUtils.slang"

// Tile cull constants — must match Renderer_Resources.cs.
static const uint TILE_SIZE_PX        = 16u;
static const uint MAX_LIGHTS_PER_TILE = 64u;

// Resource bindings - Connect CPU resources to GPU shader registers
[[vk::binding(0, 0)]] ConstantBuffer<LightingFrameUBO>      ubo;
[[vk::binding(1, 0)]] StructuredBuffer<PbrLight>            lights;
[[vk::binding(2, 0)]] RaytracingAccelerationStructure       tlas;   // scene TLAS for ray-traced shadows
[[vk::binding(3, 0)]] StructuredBuffer<uint>                tileLightCount;
[[vk::binding(4, 0)]] StructuredBuffer<uint>                tileLightIndices;

// GBuffer inputs — Set 1 in your descriptor layout
[[vk::binding(0, 1)]] Sampler2D gBufferPositionSampler;
[[vk::binding(1, 1)]] Sampler2D gBufferNormalSampler;
[[vk::binding(2, 1)]] Sampler2D gBufferAlbedoSampler;
[[vk::binding(3, 1)]] Sampler2D gBufferMaterialSampler;
[[vk::binding(4, 1)]] Sampler2D gBufferEmissiveSampler;

// Self-shadow bias — pushes the shadow ray origin off the surface along N so
// floating-point error in the gbuffer position doesn't make the surface
// shadow itself. Tune up if you see acne, down if you see "peter-panning."
static const float SHADOW_RAY_EPS = 0.002;

// Returns true iff a ray from `origin` toward `dir` hits any geometry within
// `tMax`. Uses ACCEPT_FIRST_HIT_AND_END_SEARCH so the traversal exits on the
// first intersection — perfect for binary visibility.
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


[[shader("vertex")]]
LightingVSOutput VSMain(uint vertexID : SV_VertexID)
{
    // hardcoded fullscreen triangle — covers NDC in one draw call, to give the frag shader something to write on
       float2 positions[3] = {
           float2(-1.0, -1.0),
           float2( 3.0, -1.0),
           float2(-1.0,  3.0)
       };
       float2 uvs[3] = {
           float2(0.0, 0.0),
           float2(2.0, 0.0),
           float2(0.0, 2.0)
       };

       LightingVSOutput output;
       output.Position = float4(positions[vertexID], 0.0, 1.0);
       output.UV       = uvs[vertexID];
       return output;
}

// Fragment shader entry point - Executes once per pixel
[[shader("fragment")]]
float4 PSMain(LightingVSOutput input) : SV_TARGET
{
    //unpack GBUFFER
    float3 WorldPos = gBufferPositionSampler.Sample(input.UV).xyz;
    float3 N = normalize(gBufferNormalSampler.Sample(input.UV)).xyz;

    float4 albedoMet = gBufferAlbedoSampler.Sample(input.UV);
    float3 baseColor = albedoMet.rgb;
    float metallic = albedoMet.w;
    float2 mat = gBufferMaterialSampler.Sample(input.UV).xy;
    float roughness = mat.x;
    float ao = mat.y;
    // === LIGHTING SETUP ===
    // Calculate view direction (fragment to camera)
    float3 V = normalize(ubo.camPos.xyz - WorldPos);

    // Calculate reflection vector for environment mapping
    float3 R = reflect(-V, N);

    // === PBR MATERIAL SETUP ===
    // Calculate F0 (reflectance at normal incidence)
    // Non-metals: low reflectance (~0.04), Metals: colored reflectance from base color
    float3 F0 = float3(0.04, 0.04, 0.04);  // Dielectric default
    F0 = lerp(F0, baseColor.rgb, metallic); // Lerp to metallic behavior

    // Initialize outgoing radiance accumulator
    float3 Lo = float3(0.0, 0.0, 0.0);

    // === DIRECT LIGHTING LOOP ===
    // Map the pixel to its 16×16 tile and walk only the lights the cull pass
    // assigned to this tile. Pixels off the tile grid (shouldn't normally happen)
    // fall back to zero lights — Lo stays 0 for them.
    uint tileX    = (uint)input.Position.x / TILE_SIZE_PX;
    uint tileY    = (uint)input.Position.y / TILE_SIZE_PX;
    uint tileIdx  = tileY * ubo.tileCountX + tileX;
    uint perTile  = (tileX < ubo.tileCountX && tileY < ubo.tileCountY)
                  ? tileLightCount[tileIdx] : 0u;

    for (uint t = 0u; t < perTile; t++) {
        uint     lightIdx = tileLightIndices[tileIdx * MAX_LIGHTS_PER_TILE + t];
        PbrLight lt       = lights[lightIdx];

        uint  type      = (uint)lt.directionType.w;
        float3 lColor   = lt.colorIntensity.rgb;
        float  intensity = lt.colorIntensity.a;
        if (intensity <= 0.0) continue;

        // ── Per-type direction + attenuation ──
        float3 L;
        float  distance;
        float  attenuation;
        float  spotMask = 1.0;

        if (type == LIGHT_TYPE_DIRECTIONAL) {
            // Directional: light direction is the field; distance is "infinite"
            // for shadow ray purposes — use a large finite tMax instead of FLT_MAX
            // so traversal still terminates cleanly when nothing is hit.
            L            = normalize(-lt.directionType.xyz);
            distance     = 1e6;
            attenuation  = 1.0;
        } else {
            // Point / spot: standard inverse-square falloff.
            float3 toLight = lt.positionRange.xyz - WorldPos;
            distance       = length(toLight);
            L              = toLight / max(distance, 1e-4);
            attenuation    = 1.0 / max(distance * distance, 1e-4);

            // Smooth radius cutoff so contribution drops to 0 at light.range.
            float range = lt.positionRange.w;
            if (range > 0.0) {
                float t   = saturate(1.0 - pow(distance / range, 4.0));
                attenuation *= t * t;
            }

            if (type == LIGHT_TYPE_SPOT) {
                // Spot cone: smoothstep between innerCos and outerCos along -direction.
                float3 spotDir = normalize(lt.directionType.xyz);
                float  cosA    = dot(-L, spotDir);
                float  inner   = lt.spotCones.x;
                float  outer   = lt.spotCones.y;
                spotMask       = smoothstep(outer, inner, cosA);
                if (spotMask <= 0.0) continue;
            }
        }

        float3 radiance = lColor * intensity * attenuation * spotMask;

        // Shadow ray: WorldPos -> light, with self-shadow bias along N.
        // Stop the ray a hair short of the light so we don't hit the light itself.
        // Skip the ray query for lights with shadow casting disabled.
        bool castShadows = lt.spotCones.z > 0.5;
        if (castShadows) {
            float3 shadowOrigin = WorldPos + N * SHADOW_RAY_EPS;
            float  tMax         = max(distance - SHADOW_RAY_EPS, 0.0);
            if (isOccluded(shadowOrigin, L, tMax)) continue;
        }

        // Calculate half vector (between view and light directions)
        float3 H = normalize(V + L);

        // === BRDF EVALUATION ===
        float NdotL = max(dot(N, L), 0.0);  // Lambertian falloff
        float NdotV = max(dot(N, V), 0.0);  // View angle
        float NdotH = max(dot(N, H), 0.0);  // Half vector for specular
        float HdotV = max(dot(H, V), 0.0);  // For Fresnel calculation

        // Evaluate Cook-Torrance BRDF components
        float D = DistributionGGX(NdotH, roughness);    // Normal distribution
        float G = GeometrySmith(NdotV, NdotL, roughness); // Geometry function
        float3 F = FresnelSchlick(HdotV, F0);           // Fresnel reflectance

        // Specular BRDF
        float3 numerator = D * G * F;
        float denominator = 4.0 * NdotV * NdotL + 0.0001; // Prevent division by zero
        float3 specular = numerator / denominator;

        // === ENERGY CONSERVATION ===
        float3 kS = F;                          // Specular contribution
        float3 kD = float3(1.0, 1.0, 1.0) - kS; // Diffuse contribution
        kD *= 1.0 - metallic;                   // Metals have no diffuse reflection

        // === RADIANCE ACCUMULATION ===
        Lo += (kD * baseColor.rgb / PI + specular) * radiance * NdotL;
    }

    // === AMBIENT AND EMISSIVE ===
    // Add simple ambient lighting (should be replaced with IBL in production)
    float3 ambient = float3(0.03, 0.03, 0.03) * baseColor.rgb * ao;

    // Emissive radiance written by the geometry pass — added pre-tone-map so
    // exposure/gamma apply uniformly with the rest of the lighting.
    float3 emissive = gBufferEmissiveSampler.Sample(input.UV).rgb;

    // Combine all lighting contributions
    float3 color = ambient + Lo + emissive;

    // === HDR TONE MAPPING AND GAMMA CORRECTION ===
    // Apply Reinhard tone mapping to compress HDR values to [0,1] range
    color = color / (color + float3(1.0, 1.0, 1.0));

    // Apply gamma correction for sRGB display (inverse gamma)
    color = pow(color, float3(1.0 / ubo.gamma, 1.0 / ubo.gamma, 1.0 / ubo.gamma));

    return float4(color, 1.0);
}