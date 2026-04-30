// Output from vertex shader / Input to fragment shader - Interpolated data
struct LightingVSOutput {
    float4 Position : SV_POSITION; // Required clip space position for rasterization
    float2 UV : TEXCOORD0;         // Texture coordinates (interpolated)
};

// Uniform buffer - Global data shared across all vertices/fragments
struct LightingUBO {
    float4 lightPositions[4];           // Light positions in world space
    float4 lightColors[4];              // Light intensities and colors
    float4 camPos;                      // Camera position for view-dependent effects
    float exposure;                     // HDR exposure control
    float gamma;                        // Gamma correction value (typically 2.2)
    float prefilteredCubeMipLevels;     // IBL prefiltered environment map mip levels
    float scaleIBLAmbient;              // IBL ambient contribution scale
};

#include "PbrUtils.slang"

// Resource bindings - Connect CPU resources to GPU shader registers
[[vk::binding(0, 0)]] ConstantBuffer<LightingUBO> ubo;
[[vk::binding(1, 0)]] RaytracingAccelerationStructure tlas;   // scene TLAS for ray-traced shadows
[[vk::push_constant]] PushConstants material;

// GBuffer inputs — Set 1 in your descriptor layout
[[vk::binding(0, 1)]] Sampler2D gBufferPositionSampler;
[[vk::binding(1, 1)]] Sampler2D gBufferNormalSampler;
[[vk::binding(2, 1)]] Sampler2D gBufferAlbedoSampler;
[[vk::binding(3, 1)]] Sampler2D gBufferMaterialSampler;

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
    // Calculate contribution from each light source. Each light's BRDF result is
    // gated on a shadow-ray visibility test — occluded lights contribute nothing.
    for (int i = 0; i < 4; i++) {
        float3 lightPos = ubo.lightPositions[i].xyz;
        float3 lightColor = ubo.lightColors[i].rgb;

        // Skip zero-intensity slots (we hand-wired only light 0 right now).
        if (dot(lightColor, lightColor) <= 0.0) continue;

        // Calculate light direction and attenuation
        float3 L = normalize(lightPos - WorldPos);      // Light direction
        float distance = length(lightPos - WorldPos);   // Distance for falloff
        float attenuation = 1.0 / (distance * distance);     // Inverse square falloff
        float3 radiance = lightColor * attenuation;           // Attenuated light color

        // Shadow ray: WorldPos -> light, with self-shadow bias along N.
        // Stop the ray a hair short of the light so we don't hit the light itself
        // (zero-area in this scene, but cheap insurance).
        float3 shadowOrigin = WorldPos + N * SHADOW_RAY_EPS;
        if (isOccluded(shadowOrigin, L, distance - SHADOW_RAY_EPS)) continue;

        // Calculate half vector (between view and light directions)
        float3 H = normalize(V + L);

        // === BRDF EVALUATION ===
        // Calculate all necessary dot products for BRDF terms
        float NdotL = max(dot(N, L), 0.0);  // Lambertian falloff
        float NdotV = max(dot(N, V), 0.0);  // View angle
        float NdotH = max(dot(N, H), 0.0);  // Half vector for specular
        float HdotV = max(dot(H, V), 0.0);  // For Fresnel calculation

        // Evaluate Cook-Torrance BRDF components
        float D = DistributionGGX(NdotH, roughness);    // Normal distribution
        float G = GeometrySmith(NdotV, NdotL, roughness); // Geometry function
        float3 F = FresnelSchlick(HdotV, F0);           // Fresnel reflectance

        // Calculate specular BRDF
        float3 numerator = D * G * F;
        float denominator = 4.0 * NdotV * NdotL + 0.0001; // Prevent division by zero
        float3 specular = numerator / denominator;

        // === ENERGY CONSERVATION ===
        // Fresnel term represents specular reflection ratio
        float3 kS = F;                          // Specular contribution
        float3 kD = float3(1.0, 1.0, 1.0) - kS; // Diffuse contribution (energy conservation)
        kD *= 1.0 - metallic;                   // Metals have no diffuse reflection

        // === RADIANCE ACCUMULATION ===
        // Combine diffuse (Lambertian) and specular (Cook-Torrance) terms
        // Multiply by incident radiance and cosine foreshortening
        Lo += (kD * baseColor.rgb / PI + specular) * radiance * NdotL;
    }

    // === AMBIENT AND EMISSIVE ===
    // Add simple ambient lighting (should be replaced with IBL in production)
    float3 ambient = float3(0.03, 0.03, 0.03) * baseColor.rgb * ao;

    // Combine all lighting contributions
    float3 color = ambient + Lo;

    // === HDR TONE MAPPING AND GAMMA CORRECTION ===
    // Apply Reinhard tone mapping to compress HDR values to [0,1] range
    color = color / (color + float3(1.0, 1.0, 1.0));

    // Apply gamma correction for sRGB display (inverse gamma)
    color = pow(color, float3(1.0 / ubo.gamma, 1.0 / ubo.gamma, 1.0 / ubo.gamma));

    return float4(color, 1.0);
}

