import WavefrontBindings;
import PTUtils;
import CommonTypes;

[shader("compute")]
[numthreads(64,1,1)]
void main(uint3 tid : SV_DispatchThreadID){
    
    if(tid.x >= counters[SHADE_COUNT_0]) return;

    uint p = shadeQueue[tid.x];
    HitInfo h = resolveHit(hitRecPrim[p], hitT[p], hitBary[p], psRayOrigin[p].xyz, psRayDir[p].xyz, pc.bounce);

}



HitInfo resolveHit(uint2 prim, float triIdx, float2 bary, float3 rayOrigin, float3 rayDir, uint bounce){
    HitInfo h;
    
    return h;
}