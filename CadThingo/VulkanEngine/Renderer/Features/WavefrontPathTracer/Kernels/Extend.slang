import WavefrontShading;   // re-exports WavefrontBindings + PTUtils + CommonTypes

// P2.2 Extend: trace the live rays with the alpha-test Proceed loop (MASK + stochastic
// BLEND), env on miss, atomic-append committed hits to the shade queue.
[shader("compute")]
[numthreads(64,1,1)]
void main(uint3 tid : SV_DispatchThreadID){

    if(tid.x >= counters[RAY_COUNT]) return;

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
        return;
    }

    float tHit = q.CommittedRayT();

    // P2.6 Beer-Lambert: absorb over the segment just traversed (previous vertex -> this hit) when
    // the path is inside a participating medium. No-op in vacuum (sigmaA == 0). Applied here, before
    // Shade picks up emissive/NEE at this hit, so those are already tinted by the medium. Miss paths
    // skip this (they escaped the boundary) -- matches the megakernel.
    float3 sigmaA = psSigmaA[p].xyz;
    if(any(sigmaA > 0.0)) psThroughput[p].xyz *= exp(-sigmaA * tHit);

    hitRecPrim[p] = uint2(q.CommittedInstanceID() + q.CommittedGeometryIndex(), q.CommittedPrimitiveIndex());
    hitT[p]       = tHit;
    hitBary[p]    = q.CommittedTriangleBarycentrics();
    psRng[p]      = rng;   // hand the advanced stream to Shade for this path

    uint slot;
    InterlockedAdd(counters[SHADE_COUNT_0], 1u, slot);
    shadeQueue[slot] = p;
}