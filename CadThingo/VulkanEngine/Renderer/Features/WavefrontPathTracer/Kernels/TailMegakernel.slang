import WavefrontShading;   // resolveHit (cone LOD) + BSDF/NEE/RNG helpers + the set/binding layout

// Megakernel tail: continue the bounce-cap survivors through the REMAINING bounces in one dispatch
// instead of more per-bounce Extend/Shade/Connect round-trips. Past a cutoff the wavefront's deep
// bounces are sparse (mostly transmissive/specular paths) -- the per-bounce dispatch + arg-gen +
// barrier overhead dwarfs the actual work. This loops bounces [pc.bounce, pc.maxBounces) per path,
// reading the path's live state out of the persistent SoA (so it picks up exactly where Shade left
// off, MIS state included) and folding everything into psRadiance for Finalize. NEE is done inline
// (isOccluded directly) rather than deferred to Connect -- the tail set is small, so the deferred
// machinery would cost more than it saves.
//
// Dispatched indirectly over RAY_COUNT (the bounce-cap survivor count Shade's fused tail promoted),
// via extend-args[pc.bounce]; pc.srcParity selects the survivor queue (rdRay). Image-identical to
// running the same bounces through the wavefront chain -- this is purely an overhead reduction.
[shader("compute")]
[numthreads(64,1,1)]
void main(uint3 tid : SV_DispatchThreadID){
    if(tid.x >= counters[RAY_COUNT]) return;

    uint p = rdRay(tid.x);

    // Resume the path's state at the cutoff bounce. psRayOrigin.w carries the packed ray cone;
    // psRayDir.w the previous BSDF pdf; psThroughput.w the prevWasDelta flag (all set by Shade).
    float4 ro4 = psRayOrigin[p];
    RayDesc ray;
    ray.Origin    = ro4.xyz;
    ray.Direction = psRayDir[p].xyz;
    ray.TMin      = 0.0;
    ray.TMax      = INFINITY;

    float2 cone        = wfUnpackCone(ro4.w);
    float  coneWidth   = cone.x;
    float  coneSpread  = cone.y;
    float4 thr4        = wfUnpackHalf4(psThroughput[p]);
    float3 throughput  = thr4.xyz;
    bool   prevWasDelta = thr4.w > 0.5;
    float  prevBsdfPdf = psRayDir[p].w;
    float3 L           = wfUnpackHalf4(psRadiance[p]).xyz;
    float3 sigmaA      = wfUnpackHalf4(psSigmaA[p]).xyz;
    uint   rng         = psRng[p];

    for(uint b = pc.bounce; b < pc.maxBounces; b++){
        // ---- trace (inline alpha-test loop, mirrors Extend) ----
        RayQuery<RAY_FLAG_NONE> q;
        q.TraceRayInline(tlas, RAY_FLAG_NONE, 0xFF, ray);
        while(q.Proceed()){
            if(q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE){
                ShadowEntityInfo cei = entityInfo[q.CandidateInstanceID() + q.CandidateGeometryIndex()];
                // Transmissive candidates commit straight through off the entity-record flags -- no
                // materials[] read (closest-hit treats glass as opaque; see Extend). Only MASK/BLEND
                // fetch the fat material to alpha-test.
                if((cei.flags & 5u) == 0u){          // 5 = MASK(1) | BLEND(4)
                    q.CommitNonOpaqueTriangleHit();
                } else {
                    PbrMaterial cmat = materials[cei.materialIndex];
                    float alpha = computeAlpha(cmat, cei.indexOffset,
                                               q.CandidatePrimitiveIndex(),
                                               q.CandidateTriangleBarycentrics(),
                                               ALPHA_LOD_VISIBILITY);
                    bool accept = true;
                    if((cmat.flags & 1u) != 0u) accept = alpha >= cmat.alphaCutoff;   // MASK
                    if((cmat.flags & 4u) != 0u) accept = rand(rng) < alpha;           // BLEND
                    if(accept) q.CommitNonOpaqueTriangleHit();
                }
            }
        }

        if(q.CommittedStatus() != COMMITTED_TRIANGLE_HIT){
            float3 env = envCube.SampleLevel(ray.Direction, 0.0).xyz * frameUBO.scaleIBLAmbient;
            L += wfFireflyClamp(throughput * env);
            break;
        }

        float tHit = q.CommittedRayT();
        if(any(sigmaA > 0.0)) throughput *= exp(-sigmaA * tHit);   // Beer-Lambert over the segment

        uint   jr        = rng;                 // decorrelated mip jitter (transport stream untouched)
        float  lodJitter = rand(jr);
        uint2  prim      = uint2(q.CommittedInstanceID() + q.CommittedGeometryIndex(), q.CommittedPrimitiveIndex());
        HitInfo h = resolveHit(prim, tHit, q.CommittedTriangleBarycentrics(), ray.Origin, ray.Direction,
                               b, LOBE_ALL, coneWidth, coneSpread, lodJitter, b >= 1u);
        float3 V = -ray.Direction;

        // ---- emissive pickup, MIS-weighted against emissive NEE (mirrors Shade 2.3) ----
        float misEm = 1.0;
        if(!prevWasDelta && frameUBO.emissiveTriCount > 0u && frameUBO.totalEmissivePower > 0.0){
            float lumLe = luminance(materials[h.matidx].emissiveFactor);
            float cosL  = abs(dot(h.geometricNormal, ray.Direction));
            if(lumLe > 0.0 && cosL > 1e-4){
                float pLight = lumLe * h.hitT * h.hitT / (frameUBO.totalEmissivePower * cosL);
                misEm = powerHeuristic(prevBsdfPdf, pLight);
            }
        }
        L += wfFireflyClamp(throughput * h.emissive * misEm);

        // ---- inline NEE: one light category, one shadow ray (mirrors PathTraceRT) ----
        bool haveAnalytic = frameUBO.lightCount > 0u;
        bool haveEmissive = frameUBO.emissiveTriCount > 0u && frameUBO.totalEmissivePower > 0.0;
        if(haveAnalytic || haveEmissive){
            float pAnalytic = (haveAnalytic && haveEmissive) ? 0.5 : (haveAnalytic ? 1.0 : 0.0);
            if(rand(rng) < pAnalytic){
                LightPick pick = sampleLightRIS(rng, h.worldPos, h.normal);
                if(pick.valid){
                    float NdotL = max(dot(h.normal, pick.sample.L), 0.0);
                    if(NdotL > 0.0){
                        float3 sOrigin = h.worldPos + h.geometricNormal * 1e-3;
                        float  tMax    = max(pick.sample.distance - 2e-3, 0.0);
                        bool   shadowed = pick.sample.castShadows && isOccluded(sOrigin, pick.sample.L, tMax, rng);
                        if(!shadowed){
                            float3 brdf = evaluateFullBrdf(h, V, pick.sample.L);
                            L += wfFireflyClamp(throughput * brdf * pick.sample.radiance * NdotL * pick.weight / pAnalytic);
                        }
                    }
                }
            } else {
                EmissiveLightSample es = sampleEmissive(rng, h.worldPos);
                if(es.valid){
                    float NdotL = max(dot(h.normal, es.L), 0.0);
                    if(NdotL > 0.0){
                        float3 sOrigin = h.worldPos + h.geometricNormal * 1e-3;
                        float  tMax    = max(es.dist - 2e-3, 0.0);
                        if(!isOccluded(sOrigin, es.L, tMax, rng)){
                            float  pB   = bsdfPdfForDir(h, V, es.L);
                            float  w    = powerHeuristic(es.pdf, pB);
                            float3 brdf = evaluateFullBrdf(h, V, es.L);
                            L += wfFireflyClamp(throughput * brdf * es.radiance * NdotL * w / (es.pdf * (1.0 - pAnalytic)));
                        }
                    }
                }
            }
        }

        // ---- scatter + Russian roulette (mirrors Shade 2.7) ----
        BsdfSample bs = sampleBsdf(h, V, rng);
        if(!bs.valid) break;
        throughput  *= bs.weight;
        prevBsdfPdf  = bs.pdf;
        prevWasDelta = bs.isDelta;
        if(any(isnan(throughput)) || any(isinf(throughput))) break;

        if(b >= MIN_BOUNCES){
            float rr = clamp(luminance(throughput), 0.05, 0.95);
            if(rand(rng) > rr) break;
            throughput *= 1.0 / rr;
        }

        // Advance the ray cone (footprint -> next origin width; non-delta widens the spread).
        coneWidth += coneSpread * tHit;
        if(!bs.isDelta) coneSpread += WF_CONE_SPREAD_K * max(h.roughness, 0.3);

        if(bs.transmit)
            sigmaA = h.frontFace ? mediumSigmaA(materials[h.matidx]) : float3(0.0);

        float3 oN = bs.transmit ? -h.geometricNormal : h.geometricNormal;
        ray.Origin    = h.worldPos + oN * 1e-3;
        ray.Direction = bs.wi;
        ray.TMin      = 0.0;
        ray.TMax      = INFINITY;
    }

    if(any(isnan(L)) || any(isinf(L))) L = float3(0.0);
    psRadiance[p] = wfPackHalf4(float4(L, 0.0));
    psRng[p]      = rng;
}