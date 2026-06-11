import WavefrontBindings;
import PTUtils;
import CommonTypes;

[shader("compute")]
[numthreads(64,1,1)]
void main(uint3 tid : SV_DispatchThreadID){

    if(tid.x >= counters[RAY_COUNT]) return;
    
    uint p = rdRay(tid.x);
    RayDesc ray;
    
    ray.Origin = psRayOrigin[p].xyz;
    ray.Direction = psRayDir[p].xyz;
    ray.TMin = 0.0;
    ray.TMax = INFINITY;
    
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(tlas, RAY_FLAG_NONE, 0xFF, ray);
    q.Proceed();
    if(q.CommittedStatus() != COMMITTED_TRIANGLE_HIT){
        psRadiance[p].xyz += psThroughput[p].xyz * envCube.SampleLevel(ray.Direction, 0.0).xyz;
        return;
    }
    
    hitRecPrim[p] = uint2(q.CommittedInstanceID() + q.CommittedGeometryIndex(), q.CommittedPrimitiveIndex());
    hitT[p] = q.CommittedRayT();
    hitBary[p] = q.CommittedTriangleBarycentrics();
    uint slot;
    InterlockedAdd(counters[SHADE_COUNT_0], 1u, slot);
    shadeQueue[slot] = p;
}
