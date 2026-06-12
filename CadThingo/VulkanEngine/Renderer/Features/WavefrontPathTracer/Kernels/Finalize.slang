import WavefrontBindings;
import PTUtils;

[shader("compute")]
[numthreads(8,8,1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint2 dim = uint2(frameUBO.screenSize);
    if(any(gid.xy >= dim)) return;
    
    uint p = gid.y * dim.x + gid.x;
    float3 L = psRadiance[p].xyz;
    if(any(isnan(L)) | any(isinf(L))) L = float3(0.0);
    if(frameUBO.resetAccum != 0) accumulator[gid.xy] = float4(L, 1.0);
    else accumulator[gid.xy] += float4(L, 1.0);
    
    float4 a = accumulator[gid.xy];
    
    outColor[gid.xy] = float4(a.rgb /max(a.a, 1.0), 1.0);
}