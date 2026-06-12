import WavefrontBindings;
import PTUtils;




[[vk::constant_id(0)]] const uint mode = CameraMode.PINHOLE;




[shader("compute")] 
[numthreads(8,8,1)] 
void main(uint3 gid : SV_DispatchThreadID) {
    uint2 dim = uint2(frameUBO.screenSize); if (any(gid.xy >= dim)) return;
    uint px = gid.y*dim.x + gid.x;
    //initialize rng
    uint rng = initRNG(gid.xy, dim, frameUBO.frameIndex);
    //integrate over pixel and calculate normalized device coordinates
    float2 j = float2(tentWarp(rand(rng)), tentWarp(rand(rng)));
    float2 ndc = ((float2(gid.xy)+0.5+j)/frameUBO.screenSize)*2.0-1.0;
    ndc.x *= frameUBO.screenSize.x/frameUBO.screenSize.y;
    //generate primary ray
    RayDesc ray = transformRayToWorld(generatePrimaryRay(ndc, rng, (CameraMode)mode, frameUBO), frameUBO.invView);
    //store ray data. MIS state seeds in the w-lanes: psRayDir.w = prevBsdfPdf (0 on the
    //primary), psThroughput.w = prevWasDelta flag = 1 so a directly-visible emitter is taken
    //at full weight (NEE had no prior chance to sample it).
    psRayOrigin[px]=float4(ray.Origin,0);
    psRayDir[px]=float4(ray.Direction,0);
    psThroughput[px]=float4(1,1,1,1.0);
    psRadiance[px]=float4(0,0,0,0);
    psSigmaA[px]=float4(0,0,0,0);   // start in vacuum (no medium absorption); P2.6 Beer-Lambert
    psRng[px]=rng; 
    rayQueue0[px]=px;
    
    //set ray counters + the first stage's indirect args if this is the first ray. This folds
    //the old PrepExtend(0) dispatch in: seed every counter the bounce chain consumes (the
    //arrival ticket COMPLETED_WG must start at 0) and write extend-args[0] = wfGroupCount(N) so
    //Extend(0) launches without a separate prep. N is dense + known here, so no last-group dance.
    if (px==0) {
        uint n = dim.x*dim.y;
        counters[RAY_COUNT]      = n;
        counters[NEXT_RAY_COUNT] = 0;
        counters[SHADOW_COUNT]   = 0;
        counters[COMPLETED_WG]   = 0;
        for(uint c = 0u; c < WF_SHADE_CLASSES; c++) counters[SHADE_COUNT_0 + c] = 0u;  // all C bins
        wfWriteArgs(wfExtendArgsOffset(0u), wfGroupCount(n));
    }
}