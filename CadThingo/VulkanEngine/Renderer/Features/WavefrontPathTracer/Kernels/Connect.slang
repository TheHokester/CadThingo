import WavefrontShading;   // re-exports WavefrontBindings + PTUtils (+ isOccluded)

// P2.5 Connect: for each shadow record Shade emitted, fire the occlusion ray and add the
// precomputed contribution if the light is visible. shadowPath is unique per bounce (<=1
// record per path) so the radiance add needs no atomics. shadowDir.w flags whether the
// light casts shadows (0 = unconditionally visible, skip the trace).
[shader("compute")]
[numthreads(64,1,1)]
void main(uint3 tid : SV_DispatchThreadID){
    if(tid.x >= counters[SHADOW_COUNT]) return;

    uint   p       = shadowPath[tid.x];
    float4 o       = shadowOrigin[tid.x];   // xyz = origin, w = tMax
    float4 d       = shadowDir[tid.x];      // xyz = dir,    w = castShadows
    float3 contrib = shadowLe[tid.x].xyz;   // precomputed beta*f*Le*NdotL*W

    // isOccluded's MASK/BLEND alpha gate consumes the path RNG; thread it through psRng.
    uint rng = psRng[p];
    bool occluded = (d.w > 0.5) && isOccluded(o.xyz, d.xyz, o.w, rng);
    psRng[p] = rng;

    if(!occluded) psRadiance[p].xyz += contrib;
}