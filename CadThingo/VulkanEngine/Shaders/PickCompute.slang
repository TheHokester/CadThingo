// Object picking — a single ray-query against the scene TLAS.
//
// Builds a camera ray through the clicked pixel using the same NDC→world
// convention as LightCulling.slang (so the ray lines up with the rasterized
// image and the PT image alike), traverses the TLAS, commits the closest hit —
// opaque or not, so glass / foliage are pickable too — and writes the hit
// instance's custom index to pickResult[0].
//
// Instances are per spatial cluster now, so InstanceCustomIndex is the cluster's
// geometry-block base, not the entity. The hit's identity comes from
// entityInfo[InstanceCustomIndex + GeometryIndex].entityIndex (RebuildTlas fills
// that flat table) — a stable RenderableHandle slot; the CPU maps the result back
// via GpuScene.ResolveSlot. Miss PICK_NONE.

import CommonTypes;   // ShadowEntityInfo

struct PickParams {
    float4x4 invViewProj;   // un-projects NDC -> world (Y-flipped proj, matches raster)
    float4   camPos;        // xyz = ray origin (eye); w unused
    float2   screenSize;    // render-target pixels
    uint2    pixel;         // clicked pixel, render-target space (top-left origin)
};

[[vk::push_constant]] PickParams params;

[[vk::binding(0, 0)]] RaytracingAccelerationStructure tlas;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>        pickResult;
[[vk::binding(2, 0)]] StructuredBuffer<ShadowEntityInfo> entityInfo;

static const uint PICK_NONE = 0xFFFFFFFFu;

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
    float2 ndc = (float2(params.pixel) + 0.5) / params.screenSize * 2.0 - 1.0;

    // Un-project the far-plane NDC point and aim from the eye through it. Any
    // valid clip-space depth works for the direction; the far plane keeps w
    // well-conditioned.
    float4 farClip  = float4(ndc, 1.0, 1.0);
    float4 farWorld = mul(params.invViewProj, farClip);
    float3 dir      = normalize(farWorld.xyz / farWorld.w - params.camPos.xyz);

    RayDesc ray;
    ray.Origin    = params.camPos.xyz;
    ray.Direction = dir;
    ray.TMin      = 0.0;
    ray.TMax      = 1e30;

    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(tlas, RAY_FLAG_NONE, 0xFF, ray);
    // Commit every candidate so non-opaque instances (MASK / BLEND /
    // transmissive, flagged ForceNoOpaque in the TLAS) stay selectable. No
    // alpha test — picking the frontmost surface is what the user expects.
    while (q.Proceed()) {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
            q.CommitNonOpaqueTriangleHit();
    }

    pickResult[0] = (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
                  ? entityInfo[q.CommittedInstanceID() + q.CommittedGeometryIndex()].entityIndex
                  : PICK_NONE;
}