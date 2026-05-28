// Selection-mask pass — full-screen sibling of PickCompute. One primary ray
// per pixel against the scene TLAS; writes 1.0 where the frontmost hit belongs
// to the selected entity, 0.0 elsewhere. Because it rides the same TLAS the
// other ray paths use, the mask is occlusion-correct and identical in the
// deferred, forward+ and pathtraced render modes — which is what lets the
// outline overlay work everywhere.
//
// The committed InstanceCustomIndex is the scene entity index (RebuildTlas sets
// InstanceCustomIndex = i); the CPU passes the selected entity's index in
// params.selectedIndex.

struct MaskParams {
    float4x4 invViewProj;   // un-projects NDC -> world (matches the raster image)
    float4   camPos;        // xyz = ray origin (eye)
    float2   screenSize;    // render-target pixels
    uint     selectedIndex; // scene entity index to mask
    uint     _pad;
};

[[vk::push_constant]] MaskParams params;

[[vk::binding(0, 0)]] RaytracingAccelerationStructure tlas;
[[vk::binding(1, 0)]] RWTexture2D<float>             mask;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint2 px = gid.xy;
    if (px.x >= (uint)params.screenSize.x || px.y >= (uint)params.screenSize.y) return;

    float2 ndc = (float2(px) + 0.5) / params.screenSize * 2.0 - 1.0;
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
    while (q.Proceed()) {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
            q.CommitNonOpaqueTriangleHit();
    }

    float hit = (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT &&
                 q.CommittedInstanceID() == params.selectedIndex) ? 1.0 : 0.0;
    mask[px] = hit;
}