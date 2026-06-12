import WavefrontShading;   // re-exports WavefrontBindings + PTUtils + CommonTypes

// P2.2 Extend: trace the live rays with the alpha-test Proceed loop (MASK + stochastic
// BLEND), env on miss, atomic-append committed hits to the shade queue. The dispatch fans out
// over RAY_COUNT rays; the per-path work is gated by `active` (NOT an early return) so every
// thread reaches the fused tail below in uniform control flow.
//
// Fused tail (was the PrepShade dispatch): the last group to arrive writes shade-args[bounce]
// from the final hit count and zeroes the counters Shade will accumulate into (SHADOW_COUNT,
// NEXT_RAY_COUNT). wfGroupCount clamps to >=1, so even a zero-hit bounce launches one Shade
// group whose own tail keeps the chain's args fresh.
[shader("compute")]
[numthreads(64,1,1)]
void main(uint3 tid : SV_DispatchThreadID, uint3 git : SV_GroupThreadID){

    uint rayCount = counters[RAY_COUNT];
    bool active   = tid.x < rayCount;

    if(active){
        uint p   = rdRay(tid.x);
        uint rng = psRng[p];   // threaded so BLEND's stochastic-alpha coin flip doesn't reuse samples

        RayDesc ray;
        ray.Origin    = psRayOrigin[p].xyz;
        ray.Direction = psRayDir[p].xyz;
        ray.TMin      = 0.0;
        ray.TMax      = INFINITY;

        RayQuery<RAY_FLAG_NONE> q;
        q.TraceRayInline(tlas, RAY_FLAG_NONE, 0xFF, ray);

        // Alpha-test loop. ForceNoOpaque-flagged instances (MASK / BLEND / transmissive) ride
        // through Proceed; pure-opaque hits never enter the body. MASK = hard cutoff; BLEND =
        // Bernoulli on alpha (stochastic transparency -> soft foliage edges as samples accumulate).
        while(q.Proceed()){
            if(q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE){
                ShadowEntityInfo cei = entityInfo[q.CandidateInstanceID() + q.CandidateGeometryIndex()];
                PbrMaterial      cmat = materials[cei.materialIndex];
                float alpha = computeAlpha(cmat, cei.indexOffset,
                                           q.CandidatePrimitiveIndex(),
                                           q.CandidateTriangleBarycentrics());
                bool accept = true;
                if((cmat.flags & 1u) != 0u) accept = alpha >= cmat.alphaCutoff;   // MASK
                if((cmat.flags & 4u) != 0u) accept = rand(rng) < alpha;           // BLEND
                if(accept) q.CommitNonOpaqueTriangleHit();
            }
        }

        if(q.CommittedStatus() != COMMITTED_TRIANGLE_HIT){
            // Miss: environment, scaled by IBL intensity to match the megakernel, then
            // firefly-clamped before it lands in the accumulator. Terminates the path.
            float3 env = envCube.SampleLevel(ray.Direction, 0.0).xyz * frameUBO.scaleIBLAmbient;
            psRadiance[p].xyz += wfFireflyClamp(psThroughput[p].xyz * env);
        } else {
            float tHit = q.CommittedRayT();

            // P2.6 Beer-Lambert: absorb over the segment just traversed (previous vertex -> this hit)
            // when the path is inside a participating medium. No-op in vacuum (sigmaA == 0). Applied
            // here, before Shade picks up emissive/NEE at this hit, so those are already tinted by the
            // medium. Miss paths skip this (they escaped the boundary) -- matches the megakernel.
            float3 sigmaA = psSigmaA[p].xyz;
            if(any(sigmaA > 0.0)) psThroughput[p].xyz *= exp(-sigmaA * tHit);

            uint recordIndex = q.CommittedInstanceID() + q.CommittedGeometryIndex();
            hitRecPrim[p] = uint2(recordIndex, q.CommittedPrimitiveIndex());
            hitT[p]       = tHit;
            hitBary[p]    = q.CommittedTriangleBarycentrics();
            psRng[p]      = rng;   // hand the advanced stream to Shade for this path

            // P3.3 route by shading class into a per-class bin (factors + bound-texture presence
            // only -- no resolved texels, so it's cheap + conservative). Class c owns the slice
            // shadeQueue[c*pathCount, (c+1)*pathCount); the base is fixed so the append is a single
            // per-class atomic, no scan. pathCount is the alloc stride = the dense pixel count.
            uint matidx    = entityInfo[recordIndex].materialIndex;
            uint c         = classifyShadingClass(materials[matidx]);
            uint pathCount = uint(frameUBO.screenSize.x) * uint(frameUBO.screenSize.y);
            uint slot;
            InterlockedAdd(counters[SHADE_COUNT_0 + c], 1u, slot);
            shadeQueue[c * pathCount + slot] = p;
        }
    }

    // ---- fused PrepShade(bounce): one indirect command per class + reset shadow/next ----
    if(wfFuseTailElect(git.x, rayCount)){
        for(uint c = 0u; c < WF_SHADE_CLASSES; c++)
            wfWriteArgs(wfShadeArgsOffset(pc.bounce, c), wfGroupCount(wfCounterLoad(SHADE_COUNT_0 + c)));
        counters[SHADOW_COUNT]   = 0u;   // Shade accumulates shadow records from zero
        counters[NEXT_RAY_COUNT] = 0u;   // Shade accumulates survivors from zero
    }
}