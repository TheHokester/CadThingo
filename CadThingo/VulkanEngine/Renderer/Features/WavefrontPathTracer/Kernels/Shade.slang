import WavefrontShading;   // re-exports WavefrontBindings + PTUtils + PbrUtils + CommonTypes

// P2 Shade: re-derive the hit, MIS-weighted emissive pickup (2.3), next-event estimation
// into a precomputed shadow record (2.4), then BSDF-sample + NaN scrub + Russian roulette
// for the indirect ray (2.7). MIS state round-trips the queue in the w-lanes: psRayDir.w =
// prevBsdfPdf, psThroughput.w = prevWasDelta flag (set by the PREVIOUS bounce's Shade).
// The dispatch fans out over SHADE_COUNT_0 hits; per-path work is gated by `active` (NOT an
// early return) and path death falls through via `alive` instead of returning, so every thread
// reaches the fused tail in uniform control flow.
//
// Fused tail (was PrepConnect + PrepExtend(bounce+1)): the last group to arrive writes
// connect-args[bounce] from the final shadow-record count, and -- unless this is the last
// bounce -- promotes the survivor count into RAY_COUNT and writes extend-args[bounce+1], so
// Extend(bounce+1) launches without a separate prep. wfGroupCount clamps to >=1 so an empty
// next bounce still launches one Extend group whose tail refreshes the chain.
[shader("compute")]
[numthreads(64,1,1)]
void main(uint3 tid : SV_DispatchThreadID, uint3 git : SV_GroupThreadID){

    // P3: this pass shades ONE material-class bin, selected by the push constant (the dispatch is
    // launched from that class's indirect args, so the bound count + queue slice must match).
    uint cls        = pc.argsClass;
    uint shadeCount = counters[SHADE_COUNT_0 + cls];
    bool active     = tid.x < shadeCount;

    if(active){
        uint   pathCount = uint(frameUBO.screenSize.x) * uint(frameUBO.screenSize.y);
        uint   p     = shadeQueue[cls * pathCount + tid.x];   // class bin: [cls*pathCount, ...)
        uint   rng   = psRng[p];
        float3 inDir = psRayDir[p].xyz;            // incoming ray direction at this hit
        HitInfo h    = resolveHit(hitRecPrim[p], hitT[p], hitBary[p], psRayOrigin[p].xyz, inDir, pc.bounce);
        float3 V     = -inDir;
        float3 thr   = psThroughput[p].xyz;        // throughput INTO this bounce (NEE + emissive use it)

        // ---- 2.3 emissive pickup, MIS-weighted against the emissive-NEE strategy ----
        // Full weight when the previous bounce was delta (NEE couldn't sample it) or on the
        // primary ray; otherwise power-heuristic vs the light-sampling pdf this emitter would
        // have had from the previous vertex (factor-based, so it matches the alias weighting).
        float prevBsdfPdf  = psRayDir[p].w;
        bool  prevWasDelta = psThroughput[p].w > 0.5;
        float misEm = 1.0;
        if(!prevWasDelta && frameUBO.emissiveTriCount > 0u && frameUBO.totalEmissivePower > 0.0){
            float lumLe = luminance(materials[h.matidx].emissiveFactor);
            float cosL  = abs(dot(h.geometricNormal, inDir));
            if(lumLe > 0.0 && cosL > 1e-4){
                float pLight = lumLe * h.hitT * h.hitT / (frameUBO.totalEmissivePower * cosL);
                misEm = powerHeuristic(prevBsdfPdf, pLight);
            }
        }
        psRadiance[p].xyz += wfFireflyClamp(thr * h.emissive * misEm);

        // ---- 2.4 NEE: stochastically pick ONE light category and emit ONE shadow record.
        // Shade precomputes the full contribution (Connect has no BSDF/light data); the dense
        // shadowPath is unique per bounce so Connect's radiance add needs no atomics. The
        // occlusion trace itself is deferred to Connect (2.5).
        bool haveAnalytic = frameUBO.lightCount > 0u;
        bool haveEmissive = frameUBO.emissiveTriCount > 0u && frameUBO.totalEmissivePower > 0.0;
        if(haveAnalytic || haveEmissive){
            float  pAnalytic = (haveAnalytic && haveEmissive) ? 0.5 : (haveAnalytic ? 1.0 : 0.0);
            float3 sOrigin   = h.worldPos + h.geometricNormal * 1e-3;   // offset along GEOMETRIC normal
            float3 sDir      = float3(0.0);
            float  sTMax     = 0.0;
            float3 contrib   = float3(0.0);
            float  castSh    = 1.0;
            bool   valid     = false;

            if(rand(rng) < pAnalytic){
                // Analytic-light RIS. No BSDF MIS (the BSDF sampler can't hit punctual lights).
                LightPick pick = sampleLightRIS(rng, h.worldPos, h.normal);
                if(pick.valid){
                    float NdotL = max(dot(h.normal, pick.sample.L), 0.0);
                    if(NdotL > 0.0){
                        sDir    = pick.sample.L;
                        sTMax   = max(pick.sample.distance - 2e-3, 0.0);
                        float3 brdf = evaluateFullBrdf(h, V, pick.sample.L);
                        contrib = thr * brdf * pick.sample.radiance * NdotL * pick.weight / pAnalytic;
                        castSh  = pick.sample.castShadows ? 1.0 : 0.0;
                        valid   = true;
                    }
                }
            } else {
                // Emissive geometry, MIS-combined with the BSDF sampler (counterpart to misEm).
                EmissiveLightSample es = sampleEmissive(rng, h.worldPos);
                if(es.valid){
                    float NdotL = max(dot(h.normal, es.L), 0.0);
                    if(NdotL > 0.0){
                        sDir    = es.L;
                        sTMax   = max(es.dist - 2e-3, 0.0);
                        float  pB   = bsdfPdfForDir(h, V, es.L);
                        float  w    = powerHeuristic(es.pdf, pB);
                        float3 brdf = evaluateFullBrdf(h, V, es.L);
                        contrib = thr * brdf * es.radiance * NdotL * w / (es.pdf * (1.0 - pAnalytic));
                        castSh  = 1.0;
                        valid   = true;
                    }
                }
            }

            if(valid){
                uint s;
                InterlockedAdd(counters[SHADOW_COUNT], 1u, s);
                shadowPath[s]   = p;
                shadowOrigin[s] = float4(sOrigin, sTMax);
                shadowDir[s]    = float4(sDir, castSh);                 // w: 1 = test occlusion, 0 = unshadowed
                shadowLe[s]     = float4(wfFireflyClamp(contrib), 0.0);
            }
        }

        // ---- indirect ray: BSDF sample + NaN scrub + Russian roulette (2.7) ----
        // Path death falls through (alive=false) instead of returning so the thread still
        // reaches the fused tail. rng is written back once at the end -- it captures every
        // advance (NEE pick, sampleBsdf, the RR draw), matching the old per-return writes.
        BsdfSample bs = sampleBsdf(h, V, rng);
        bool alive = bs.valid;
        if(alive) thr *= bs.weight;
        if(alive && (any(isnan(thr)) || any(isinf(thr)))) alive = false;
        if(alive && pc.bounce >= MIN_BOUNCES){
            float rr = clamp(luminance(thr), 0.05, 0.95);
            if(rand(rng) > rr) alive = false;          // terminated
            else               thr *= 1.0 / rr;
        }

        if(alive){
            // P2.6 update the participating medium on a refraction event: entering (front face)
            // adopts this material's absorption, exiting (back face) returns to vacuum. Non-transmit
            // (reflect / diffuse / TIR) leaves psSigmaA untouched, so a ray trapped in a solid keeps
            // absorbing.
            if(bs.transmit)
                psSigmaA[p] = float4(h.frontFace ? mediumSigmaA(materials[h.matidx]) : float3(0.0), 0.0);

            // Next-bounce state + MIS state for the next Shade (psRayDir.w / psThroughput.w).
            float3 oN = bs.transmit ? -h.geometricNormal : h.geometricNormal;
            psRayOrigin[p]  = float4(h.worldPos + oN * 1e-3f, 0);
            psRayDir[p]     = float4(bs.wi, bs.pdf);
            psThroughput[p] = float4(thr, bs.isDelta ? 1.0 : 0.0);

            uint slot;
            InterlockedAdd(counters[NEXT_RAY_COUNT], 1u, slot);
            wrRay(slot, p);
        }

        psRng[p] = rng;   // advanced stream persists whether or not the path survived
    }

    // ---- fused PrepConnect(bounce) + PrepExtend(bounce+1) ----
    
    if(cls == WF_SHADE_CLASSES - 1u && wfFuseTailElect(git.x, shadeCount)){
        uint nShadow = wfCounterLoad(SHADOW_COUNT);
        // Connect is a leaf: plain ceil, fine for it to launch 0 groups when there are no records.
        wfWriteArgs(wfConnectArgsOffset(pc.bounce), (nShadow + 63u) / 64u);

        if(pc.bounce + 1u < pc.maxBounces){
            uint nNext = wfCounterLoad(NEXT_RAY_COUNT);
            counters[RAY_COUNT] = nNext;       // promote survivors -> next bounce's ray count
            for(uint c = 0u; c < WF_SHADE_CLASSES; c++) counters[SHADE_COUNT_0 + c] = 0u;  // all C bins
            wfWriteArgs(wfExtendArgsOffset(pc.bounce + 1u), wfGroupCount(nNext));
        }
    }
}