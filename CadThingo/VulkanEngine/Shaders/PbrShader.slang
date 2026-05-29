import PbrUtils;   // BRDF/light/IBL math + re-exported CommonTypes (types, consts, vertex helpers)

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
    float _padExposure;                 // formerly exposure — tone-map moved to dedicated post pass
    float _padGamma;                    // formerly gamma — tone-map moved to dedicated post pass
    float prefilteredCubeMipLevels;     // global IBL prefiltered env map mip levels
    float scaleIBLAmbient;              // IBL ambient contribution scale
    uint  lightCount;                   // Number of valid entries in lights[]
    uint  tileCountX;                   // Phase 4 — tiles across the screen
    uint  tileCountY;                   // Phase 4 — tiles down the screen
    uint  _pad0;
    float2 screenSize;                  // pixels (matches CPU swapChainExtent)
    uint  _pad1;
    uint  _pad2;
    // Reflection-probe cluster dims. Tile-only today (Z = 1); when 3D froxels
    // land Z > 1 and the cluster index computation below picks up the slice.
    uint  probeClusterDimsX;
    uint  probeClusterDimsY;
    uint  probeClusterDimsZ;
    float probeMipLevels;               // mip count of probe cubemap array (256² → 9)
};

// ProbeRecord, ProbeClusterRange, TILE_SIZE_PX, MAX_LIGHTS_PER_TILE now come
// from CommonTypes (via `import PbrUtils`).

// Resource bindings - Connect CPU resources to GPU shader registers
[[vk::binding(0, 0)]] ConstantBuffer<LightingFrameUBO>      ubo;
[[vk::binding(1, 0)]] StructuredBuffer<PbrLight>            lights;
[[vk::binding(2, 0)]] RaytracingAccelerationStructure       tlas;   // scene TLAS for ray-traced shadows
[[vk::binding(3, 0)]] StructuredBuffer<uint>                tileLightCount;
[[vk::binding(4, 0)]] StructuredBuffer<uint>                tileLightIndices;
// IBL split-sum: irradianceCube + prefilteredCube give diffuse and per-roughness
// specular environment contribution; brdfLut is the view-independent F0 fit.
// Cubes are baked from .hdr; black-cleared until LoadEnvironmentHdr runs, which
// gives a zero IBL contribution until then.
[[vk::binding(5, 0)]] SamplerCube                           irradianceCube;
[[vk::binding(6, 0)]] SamplerCube                           prefilteredCube;
[[vk::binding(7, 0)]] Sampler2D                             brdfLut;

// Reflection-probe bindings. Empty / zero-probe scenes still bind well-defined
// resources — empty clusterRange.count makes the loop a no-op and the global
// IBL fallback takes over.
[[vk::binding(8,  0)]] SamplerCubeArray                     probeCubeArray;
[[vk::binding(9,  0)]] StructuredBuffer<ProbeRecord>        probes;
[[vk::binding(10, 0)]] StructuredBuffer<ProbeClusterRange>  probeClusterRange;
[[vk::binding(11, 0)]] StructuredBuffer<uint>               probeIndexList;

// Specialization constants — picked up via VkSpecializationInfo at pipeline build.
// SOFT_SHADOWS off (default) → single-ray binary occlusion (the original path).
// SOFT_SHADOWS on            → blocker-search + Vogel-disk penumbra taps.
// Flip CadThingo's Renderer.softShadowsEnabled and rebuild the PBR pipeline.
[[vk::constant_id(0)]] const bool SOFT_SHADOWS = false;

// GBuffer inputs — Set 1 in your descriptor layout
[[vk::binding(0, 1)]] Sampler2D gBufferPositionSampler;
[[vk::binding(1, 1)]] Sampler2D gBufferNormalSampler;
[[vk::binding(2, 1)]] Sampler2D gBufferAlbedoSampler;
[[vk::binding(3, 1)]] Sampler2D gBufferMaterialSampler;
[[vk::binding(4, 1)]] Sampler2D gBufferEmissiveSampler;

// Set 2: shadow-alpha plumbing
// ShadowEntityInfo[i] = info for the entity whose TLAS InstanceCustomIndex == i.
// Filled CPU-side in Renderer_Ray_Query.cs::RebuildTlas. Read in the shadow-ray
// Proceed loop when the candidate is non-opaque (CPU sets ForceNoOpaqueBit on
// MASK + BLEND TLAS instances; opaque geometry never enters the Proceed loop).
// ShadowEntityInfo layout lives in CommonTypes.
[[vk::binding(0, 2)]] StructuredBuffer<ShadowEntityInfo> entityShadowInfo;
[[vk::binding(1, 2)]] ByteAddressBuffer                  globalIndices;
[[vk::binding(2, 2)]] ByteAddressBuffer                  globalVertices;

// Set 3: bindless materials/instances/textures/samplers
// Layout shared with the geometry pass (owned by ResourceManager). Only the
// alpha-test path reads materials + textures + samplers; instances[] is
// layout parity only.
[[vk::binding(0, 3)]] StructuredBuffer<PbrMaterial>      materials;
[[vk::binding(1, 3)]] StructuredBuffer<InstanceData>     instances;     // unused here
[[vk::binding(2, 3)]] Texture2D                          textures[];
[[vk::binding(3, 3)]] SamplerState                       samplers[];

// Vertex stride / fetch helpers now live in CommonTypes; they take the
// ByteAddressBuffer as a parameter (globalVertices / globalIndices below).

// Decides whether a candidate non-opaque hit should occlude the shadow ray.
//   BLEND  → false (light passes through unattenuated).
//   MASK   → alpha-tests sampled baseColor.a against material.alphaCutoff.
//   other  → defensive: returns true so a stray non-opaque-tagged opaque
//            instance still casts a shadow.
bool shadowAlphaPasses(uint recordIndex, uint primIdx, float2 bary)
{
    // recordIndex is the flat per-geometry slot: the candidate instance's custom
    // index (cluster geometry-block base) + the candidate geometry index.
    ShadowEntityInfo info = entityShadowInfo[recordIndex];
    if ((info.flags & 4u) != 0u) return false;   // BLEND → no shadow
    if ((info.flags & 1u) == 0u) return true;    // not MASK → commit (defensive)

    PbrMaterial mat = materials[info.materialIndex];
    
    if(mat.alphaCutoff == 0.0f) return true;

    uint i0 = fetchTriangleIndex(globalIndices, info.indexOffset, primIdx, 0u);
    uint i1 = fetchTriangleIndex(globalIndices, info.indexOffset, primIdx, 1u);
    uint i2 = fetchTriangleIndex(globalIndices, info.indexOffset, primIdx, 2u);
    float2 uv0 = fetchVertexUV(globalVertices, i0);
    float2 uv1 = fetchVertexUV(globalVertices, i1);
    float2 uv2 = fetchVertexUV(globalVertices, i2);
    float  w0  = 1.0 - bary.x - bary.y;
    float2 uv  = uv0 * w0 + uv1 * bary.x + uv2 * bary.y;

    float alpha = textures[NonUniformResourceIndex(mat.baseColorTex)]
                    .SampleLevel(samplers[0], uv, 0).a
                * mat.baseColorFactor.a;

    return alpha >= mat.alphaCutoff;
}

// Self-shadow bias — pushes the shadow ray origin off the surface along N so
// floating-point error in the gbuffer position doesn't make the surface
// shadow itself. Tune up if you see acne, down if you see "peter-panning."
static const float SHADOW_RAY_EPS = 0.002;

// Distance-to-occluder soft shadow tunables.
//   PENUMBRA_SAMPLES — penumbra-tap rays per shaded pixel (cost scales linearly).
//   GOLDEN_ANGLE     — Vogel-disk increment; sqrt(i/N) radius + golden-angle theta
//                      gives a low-discrepancy disk pattern with no precomputed table.
//   MAX_CONE_HA      — clamps the penumbra cone half-angle so a blocker hugging the
//                      light doesn't produce a hemispherical spray of samples.
static const int   PENUMBRA_SAMPLES = 8;
static const float GOLDEN_ANGLE     = 2.39996323;
static const float MAX_CONE_HA      = 0.25;

// Returns true iff a ray from `origin` toward `dir` hits any geometry within
// `tMax`. Opaque BLAS geometry auto-commits on first hit; non-opaque (MASK +
// BLEND, tagged ForceNoOpaque CPU-side) routes through shadowAlphaPasses in
// the Proceed loop. ACCEPT_FIRST_HIT_AND_END_SEARCH still ends traversal on
// the first *committed* hit — works for both opaque and alpha-passing cases.
bool isOccluded(float3 origin, float3 dir, float tMax)
{
    RayDesc ray;
    ray.Origin    = origin;
    ray.Direction = dir;
    ray.TMin      = 0.0;
    ray.TMax      = tMax;

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    q.TraceRayInline(tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, ray);
    while (q.Proceed())
    {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            if (shadowAlphaPasses(
                    q.CandidateInstanceID() + q.CandidateGeometryIndex(),
                    q.CandidatePrimitiveIndex(),
                    q.CandidateTriangleBarycentrics()))
            {
                q.CommitNonOpaqueTriangleHit();
            }
        }
    }
    return q.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
}

// Closest-hit blocker search along the center shadow ray. Returns the hit
// distance (CommittedRayT), or -1 on miss. Non-opaque candidates go through
// shadowAlphaPasses; BLEND geometry never commits so it can't seed the
// penumbra cone (matches "glass casts no shadow" semantics elsewhere).
float blockerSearch(float3 origin, float3 dir, float tMax)
{
    RayDesc ray;
    ray.Origin    = origin;
    ray.Direction = dir;
    ray.TMin      = 0.0;
    ray.TMax      = tMax;

    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(tlas, RAY_FLAG_NONE, 0xFF, ray);
    while (q.Proceed())
    {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            if (shadowAlphaPasses(
                    q.CandidateInstanceID() + q.CandidateGeometryIndex(),
                    q.CandidatePrimitiveIndex(),
                    q.CandidateTriangleBarycentrics()))
            {
                q.CommitNonOpaqueTriangleHit();
            }
        }
    }
    return (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) ? q.CommittedRayT() : -1.0;
}

// Orthonormal basis around the light direction L. Used to place Vogel-disk
// samples in the plane perpendicular to L.
void buildLightBasis(float3 L, out float3 T, out float3 B)
{
    float3 up = abs(L.y) < 0.99 ? float3(0, 1, 0) : float3(1, 0, 0);
    T = normalize(cross(up, L));
    B = cross(L, T);
}

// Per-pixel hash → rotation angle for the Vogel disk. Different pattern per
// pixel breaks up banding and approximates blue-noise distribution cheaply.
float pixelJitter(float2 px)
{
    return frac(sin(dot(px, float2(12.9898, 78.233))) * 43758.5453) * 6.2831853;
}

// Distance-to-occluder soft shadow. Returns [0..1] visibility.
//   distanceToLight: full ray length from surface to light center
//                    (set to 1e6 for directional — used as a sentinel below)
//   lightRadius:     world-space radius for point/spot; tan(angularRadius) for
//                    directional (sun ≈ 0.005). 0 falls back to hard shadow.
//
// Two-phase: closest-hit blocker search seeds a PCSS-style penumbra estimate,
// then N Vogel-disk taps inside that cone produce a soft edge. When the blocker
// hugs the surface the cone collapses to ~0 and you get a crisp contact shadow;
// when the blocker hugs the light the cone widens, softening the silhouette.
float softShadow(float3 origin, float3 L, float distanceToLight, float lightRadius, float jitterAngle)
{
    float tMax = max(distanceToLight - SHADOW_RAY_EPS, 0.0);

    float tBlocker = blockerSearch(origin, L, tMax);
    if (tBlocker < 0.0)        return 1.0;   // unobstructed
    if (lightRadius <= 1e-4)   return 0.0;   // point light → hard shadow

    // Penumbra-width formula (PCSS, expressed as a cone half-angle around L):
    //   finite light:  α = lightRadius · tBlocker / (dLight · (dLight − tBlocker))
    //   directional:   α ≈ lightRadius   (lightRadius is already tan(angularRadius))
    // The 1e5 cutoff matches the 1e6 sentinel CPU-side writes for directional lights.
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
    // gBufferPosition.w == 1.0 wherever Geometry.slang ran; == 0.0 on the
    // cleared sky pixels. Bail early so we don't sample irradiance with an
    // undefined-normalized normal (and so the skybox pass can paint here).
    float4 WorldPosW = gBufferPositionSampler.Sample(input.UV);
    if (WorldPosW.w < 0.5) return float4(0, 0, 0, 1);
    float3 WorldPos = WorldPosW.xyz;
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

        // Per-type direction + attenuation 
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
        // Spec-constant gate picks soft (multi-tap penumbra) vs hard (1-ray binary).
        // The branch on SOFT_SHADOWS is resolved at pipeline-build time, so the
        // unused arm is compiled out entirely — no per-pixel cost for the toggle.
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

        // Cook-Torrance direct term (shared evaluator in PbrUtils). baseColor.rgb
        // is the diffuse albedo; `radiance` already folds in colour · intensity ·
        // attenuation · spot mask. Return value is pre-multiplied by NdotL.
        Lo += cookTorranceDirect(N, V, L, radiance, baseColor.rgb, metallic, roughness, F0);
    }

    // === IBL — SPLIT-SUM AMBIENT (shared evaluator in PbrUtils) ===
    // Diffuse irradiance + probe-blended prefiltered specular + BRDF LUT, with a
    // global-IBL fallback baked into the coverage lerp. The deferred pass and the
    // transparent pass call the same evaluateIBL with their own descriptor sets.
    // When no environment is loaded the cubes are black and this returns 0 — the
    // scene falls back to direct lighting only.
    float3 ambient = evaluateIBL(
        N, V, R, baseColor.rgb, metallic, roughness, ao, F0,
        WorldPos, uint2(input.Position.xy),
        irradianceCube, prefilteredCube, brdfLut, probeCubeArray,
        probes, probeClusterRange, probeIndexList,
        ubo.prefilteredCubeMipLevels, ubo.scaleIBLAmbient,
        ubo.probeClusterDimsX, ubo.probeClusterDimsY, ubo.probeMipLevels);

    // Emissive radiance written by the geometry pass — added pre-tone-map so
    // exposure/gamma apply uniformly with the rest of the lighting.
    float3 emissive = gBufferEmissiveSampler.Sample(input.UV).rgb;

    // Combine all lighting contributions
    float3 color = ambient + Lo + emissive;

    // Output linear scene-referred HDR — the dedicated tone-map pass downstream
    // applies exposure, Reinhard, and inverse gamma before writing the LDR
    // FinalColor target.
    return float4(color, 1.0);
}