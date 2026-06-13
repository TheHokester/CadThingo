import WavefrontBindings;
import PTUtils;

[shader("compute")]
[numthreads(8,8,1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint2 dim = uint2(frameUBO.screenSize);
    if(any(gid.xy >= dim)) return;
    
    uint p = gid.y * dim.x + gid.x;
    float3 L = wfUnpackHalf4(psRadiance[p]).xyz;
    // Connect's NEE contributions live in their own accumulator (Connect is its sole
    // writer all frame, which is what frees it to overlap the bounce chain). Fold it in.
    L += wfUnpackHalf4(shadowRadiance[p]).xyz;
    // Escaped paths: Extend stashes the (already firefly-clamped) env contribution in
    // psThroughput with w = -1, instead of adding to psRadiance -- keeps Extend off
    // psRadiance so nothing it overlaps conflicts. Fold it in here.
    float4 t = wfUnpackHalf4(psThroughput[p]);
    if(t.w < -0.5) L += t.xyz;
    if(any(isnan(L)) | any(isinf(L))) L = float3(0.0);
    if(frameUBO.resetAccum != 0) accumulator[gid.xy] = float4(L, 1.0);
    else accumulator[gid.xy] += float4(L, 1.0);
    
    float4 a = accumulator[gid.xy];
    
    outColor[gid.xy] = float4(a.rgb /max(a.a, 1.0), 1.0);
}