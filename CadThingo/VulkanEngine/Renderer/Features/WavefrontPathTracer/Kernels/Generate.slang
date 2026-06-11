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
    RayDesc ray = transformRayToWorld(generatePrimaryRay(ndc, rng), frameUBO.invView);
    //store ray data
    psRayOrigin[px]=float4(ray.Origin,0);
    psRayDir[px]=float4(ray.Direction,0);
    psThroughput[px]=float4(1,1,1,0); 
    psRadiance[px]=float4(0,0,0,0); 
    psRng[px]=rng; 
    rayQueue0[px]=px;
    
    //set ray counters if this is the first ray
    if (px==0) { 
        counters[RAY_COUNT]=dim.x*dim.y; 
        counters[NEXT_RAY_COUNT]=0; 
    }
}