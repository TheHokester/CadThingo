import WavefrontShading;   // re-exports WavefrontBindings + PTUtils (+ isOccluded)

// P2.5 Connect: for each shadow record Shade emitted, fire the occlusion ray and add the
// precomputed contribution if the light is visible. shadowPath is unique per bounce (<=1
// record per path) so the radiance add needs no atomics. shadowDir.w flags whether the
// light casts shadows (0 = unconditionally visible, skip the trace).
//
// DECOUPLED from the bounce chain: Connect(b) overlaps the NEXT BOUNCE's workers (its only
// downstream consumers are Shade(b+2), which reuses its parity record set, and Finalize),
// so it must not touch anything those overlapped dispatches touch --
//  - thread gate comes from connectArgs[parity][b].count, NOT counters[SHADOW_COUNT]
//    (Extend(b+1)'s tail zeroes that mid-flight);
//  - the alpha-gate RNG is hash-derived per (record, bounce, frame), NOT the psRng stream
//    (Extend(b+1) advances that for surviving paths). Decorrelation is all the gate needs.
//  - contributions accumulate in shadowRadiance, NOT psRadiance: Connect is that buffer's
//    only writer all frame (Generate zeroes, Finalize folds), so it conflicts with no
//    graphics pass at all. shadowPath is unique per bounce; consecutive Connects are
//    ordered on their own queue, so the += needs no atomics.
[shader("compute")]
[numthreads(64,1,1)]
void main(uint3 tid : SV_DispatchThreadID){
    if(tid.x >= wfConnectCount(pc.bounce)) return;

    uint   par     = pc.bounce & 1u;
    uint   p       = rdShadowPath(par, tid.x);
    float4 o       = rdShadowOrigin(par, tid.x);   // xyz = origin, w = tMax
    float4 d       = rdShadowDir(par, tid.x);      // xyz = dir,    w = castShadows
    float3 contrib = rdShadowLe(par, tid.x);       // precomputed beta*f*Le*NdotL*W

    // Private stream: nested pcgHash like initRNG, but salted so it can never collide with
    // a pixel stream (initRNG folds pixel.y in the middle; this folds the bounce + a salt).
    uint rng = pcgHash(p + pcgHash(pc.bounce + pcgHash(frameUBO.frameIndex ^ 0x9E3779B9u)));
    bool occluded = (d.w > 0.5) && isOccluded(o.xyz, normalize(d.xyz), o.w, rng);

    if(!occluded){
        float4 sr = wfUnpackHalf4(shadowRadiance[p]);
        sr.xyz += contrib;
        shadowRadiance[p] = wfPackHalf4(sr);
    }
}