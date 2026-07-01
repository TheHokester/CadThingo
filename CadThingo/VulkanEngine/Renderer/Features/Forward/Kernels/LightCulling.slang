// Tiled light culling — Phase 4. One thread group per 16×16 screen tile;
// 256 threads cooperatively classify the live light set, atomically appending
// touching lights into a per-tile slot list. The lighting fragment shader
// then reads the per-tile list keyed by gl_FragCoord / 16.

// Must match Renderer_Resources.cs constants.
static const uint TILE_SIZE           = 16u;
static const uint MAX_LIGHTS_PER_TILE = 64u;

// Mirrors PbrUtils.slang::PbrLight (kept self-contained so this shader doesn't
// inherit the rest of the PBR utility include).
struct PbrLight {
    float4 positionRange;     // xyz pos, w range (-1 sentinel = directional)
    float4 colorIntensity;    // rgb color, a intensity
    float4 directionType;     // xyz dir, w type (0=Dir, 1=Point, 2=Spot) as float
    float4 spotCones;         // x innerCos, y outerCos, z castShadows, w pad
};

static const uint LIGHT_TYPE_DIRECTIONAL = 0u;
static const uint LIGHT_TYPE_POINT       = 1u;
static const uint LIGHT_TYPE_SPOT        = 2u;

struct CullParams {
    float4x4 invViewProj;     // un-projects NDC -> world for tile corner rays
    float4   camPos;          // xyz only
    float2   screenSize;      // pixels
    uint2    tileCounts;      // how many tiles fit on screen this frame
    uint     lightCount;      // how many entries in lights[] are valid
    uint     _pad0;
    uint     _pad1;
    uint     _pad2;
};

[[vk::push_constant]] CullParams params;

[[vk::binding(0, 0)]] StructuredBuffer<PbrLight>     lights;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>       tileLightCount;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>       tileLightIndices;

// Per-tile workspace: shared between the 256 threads in this group.
groupshared uint  sharedCount;
groupshared float4 sharedPlanes[4];   // L, R, B, T (normals point INTO the frustum)

// Project an NDC point at the far plane back to a world-space direction from
// the camera. Used to build the 4 corner rays of the tile sub-frustum.
float3 ndcToWorldDir(float2 ndc, float4x4 invVP, float3 camPos)
{
    float4 farClip   = float4(ndc, 1.0, 1.0);
    float4 farWorld4 = mul(invVP, farClip);
    float3 farWorld  = farWorld4.xyz / farWorld4.w;
    return normalize(farWorld - camPos);
}

// Plane through `origin` whose normal is the cross product of two rays.
// `signRef` is a third ray known to be inside the sub-frustum; we flip the
// normal if needed so it points INTO the frustum (dot(n, signRef) >= 0).
float4 planeFromCorner(float3 origin, float3 rayA, float3 rayB, float3 signRef)
{
    float3 n = normalize(cross(rayA, rayB));
    if (dot(n, signRef) < 0.0) n = -n;
    float d = -dot(n, origin);
    return float4(n, d);
}

bool sphereInsidePlane(float4 plane, float3 c, float r)
{
    return dot(plane.xyz, c) + plane.w + r >= 0.0;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void Main(uint3 gid : SV_GroupID, uint flatIdx : SV_GroupIndex)
{
    uint tx = gid.x;
    uint ty = gid.y;
    if (tx >= params.tileCounts.x || ty >= params.tileCounts.y) return;
    uint tileIdx = ty * params.tileCounts.x + tx;

    //  Thread 0 builds the per-tile state 
    if (flatIdx == 0)
    {
        sharedCount = 0u;

        // Pixel-space tile bounds.
        float2 tileMin = float2(tx * TILE_SIZE, ty * TILE_SIZE);
        float2 tileMax = tileMin + float2(TILE_SIZE, TILE_SIZE);

        // Convert pixel-space bounds to NDC. Vulkan NDC y goes downward in our
        // screen convention (we Y-flip the projection matrix in the renderer),
        // but the cull frustum is built from invViewProj of the same Y-flipped
        // matrix, so the math is consistent — the visible volume is the same.
        float2 sz   = params.screenSize;
        float2 ndcMin = float2(tileMin.x / sz.x, tileMin.y / sz.y) * 2.0 - 1.0;
        float2 ndcMax = float2(tileMax.x / sz.x, tileMax.y / sz.y) * 2.0 - 1.0;

        // Four corner rays at the far plane. Order: TL, TR, BL, BR.
        float3 rayTL = ndcToWorldDir(float2(ndcMin.x, ndcMin.y), params.invViewProj, params.camPos.xyz);
        float3 rayTR = ndcToWorldDir(float2(ndcMax.x, ndcMin.y), params.invViewProj, params.camPos.xyz);
        float3 rayBL = ndcToWorldDir(float2(ndcMin.x, ndcMax.y), params.invViewProj, params.camPos.xyz);
        float3 rayBR = ndcToWorldDir(float2(ndcMax.x, ndcMax.y), params.invViewProj, params.camPos.xyz);

        // Reference direction inside the sub-frustum — average of corners.
        float3 mid = normalize(rayTL + rayTR + rayBL + rayBR);

        // Side planes — each plane spans two adjacent corner rays + the camera origin.
        sharedPlanes[0] = planeFromCorner(params.camPos.xyz, rayTL, rayBL, mid); // Left
        sharedPlanes[1] = planeFromCorner(params.camPos.xyz, rayBR, rayTR, mid); // Right
        sharedPlanes[2] = planeFromCorner(params.camPos.xyz, rayBL, rayBR, mid); // Bottom
        sharedPlanes[3] = planeFromCorner(params.camPos.xyz, rayTR, rayTL, mid); // Top
    }

    // All threads wait for thread 0 to publish sharedPlanes + sharedCount.
    GroupMemoryBarrierWithGroupSync();

    //  Cooperative light classification
    
    // 256 threads stride through the light list: thread `flatIdx` tests
    // lights at indices flatIdx, 256+flatIdx, 512+flatIdx, ...
    for (uint li = flatIdx; li < params.lightCount; li += 256u)
    {
        PbrLight L = lights[li];
        uint type   = (uint)L.directionType.w;
        bool inside;

        if (type == LIGHT_TYPE_DIRECTIONAL)
        {
            // Directional lights have no spatial bound — always touch every tile.
            inside = true;
        }
        else
        {
            // Point and spot: sphere of radius=range against the 4 side planes.
            // (Spot is over-approximated as a sphere here; tighter cone-vs-frustum
            // is a follow-up optimization.)
            float3 c = L.positionRange.xyz;
            float  r = L.positionRange.w;
            inside = sphereInsidePlane(sharedPlanes[0], c, r) &&
                     sphereInsidePlane(sharedPlanes[1], c, r) &&
                     sphereInsidePlane(sharedPlanes[2], c, r) &&
                     sphereInsidePlane(sharedPlanes[3], c, r);
        }

        if (inside)
        {
            uint slot;
            InterlockedAdd(sharedCount, 1u, slot);
            if (slot < MAX_LIGHTS_PER_TILE)
                tileLightIndices[tileIdx * MAX_LIGHTS_PER_TILE + slot] = li;
        }
    }

    // Wait for all threads to finish appending before thread 0 publishes the count.
    GroupMemoryBarrierWithGroupSync();

    if (flatIdx == 0)
    {
        tileLightCount[tileIdx] = min(sharedCount, MAX_LIGHTS_PER_TILE);
    }
}