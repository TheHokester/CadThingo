// PTUtils — pathtracer math: raygen for the four camera modes, RNG, BSDF
// sampling + layered (base + clearcoat) BRDF evaluation.
//
// `module PTUtils;` with `__exported import PbrUtils;` so PTCompute gets the
// shared BRDF terms (DistributionGGX / GeometrySmith / FresnelSchlick),
// evaluateLight / LightSample, and the CommonTypes layer transitively.
//
//  All helpers take height-NDC with aspect baked into x:
//
//      float2 px  = float2(gid.xy) + 0.5;
//      float2 ndc = (px / ubo.screenSize) * 2.0 - 1.0;       // [-1, 1]
//      ndc.x *= ubo.screenSize.x / ubo.screenSize.y;          // aspect → x
//
//  Output rays are in **camera space** (origin near 0, forward = -Z, up = +Y).
//  Caller transforms to world space via the camera basis (rows of invView).
//  NDC convention: ndc.y > 0 = bottom of screen; helpers flip internally so
//  the returned direction has world +Y = up.

module PTUtils;

__exported import PbrUtils;

// PI / TWO_PI / INV_PI / HALF_PI / INV_TWO_PI come from CommonTypes. The
// constants below are pathtracer-specific and stay local.
public static const float INFINITY        = asfloat(0x7F800000);

// Sentinel returned by raygen helpers for pixels that fall outside the
// projection's valid region (fisheye corner clip, etc.). Caller skips the
// pixel when TMax < 0.
public static const float RAY_INVALID_TMAX = -1.0;
public static const uint  MIN_BOUNCES      = 1;

// Per-hit shading info returned by traceRay().  Lives in registers across the
// bounce loop — size affects occupancy more than direct perf.  Fields slotted
// so float3 + float pairs pack into 16B vectors.
public struct HitInfo {
    public RayDesc ray;              // 32B - input ray (handy for outgoing dir = -ray.Direction)
    public float3 worldPos;           // 12B - world-space hit position
    public float metallic;            //  4B - packs with worldPos
    public float4 baseColor;          // 16B - factor * texture; .a kept for alpha-blend math
    public float3 normal;             // 12B - SHADING normal (after normal-map perturbation)
    public float roughness;           //  4B - packs with normal
    public float3 geometricNormal;    // 12B - geometry-only; use for ray offset + front/back test
    public float hitT;                //  4B - committed ray distance; worldPos = ray.Origin + dir*hitT
    public float3 tangent;            // 12B - world-space T axis of the TBN frame
    public float bitangentSign;       //  4B - glTF tangent.w; B = cross(N, T) * bitangentSign
    public float3 emissive;           // 12B - factor * texture
    public float ior;                 //  4B - index of refraction (~1.5 default)
    public float3 clearcoatNormal;    // 12B - normal for the coat lobe (usually = shading N)
    public float clearcoat;           //  4B - KHR_materials_clearcoat factor × tex.r
    public float F0;                  //  4B - dielectric F0; derived from IOR
    public float clearcoatRoughness;  //  4B - clearcoat lobe roughness (factor × tex.g)
    public float2 uv;                 //  8B - interpolated UV at the hit
    public float transmission;        //  4B - glass / transparency factor [0..1]
    public int alphaMode;             //  4B - 0=opaque, 1=mask, 4=blend (mat.flags bits)
    public uint instID;               //  4B - committed instance — debug
    public uint matidx;               //  4B - material index
    public bool hit;                  //  4B - false ⇒ ray missed the scene
    public bool frontFace;            //  4B - dot(geometricNormal, ray.Direction) < 0
    public bool isGlass;              //  4B
    public bool isLiquid;             //  4B
};


// Camera-space → world-space
// Every raygen helper emits its ray in camera space (origin near 0, forward
// = -Z).  Apply this once on the result to plant the ray at the actual
// camera position pointing along the actual view direction.
//
// `invView` is the inverse of the camera's view matrix — System.Numerics
// row-major + Slang column-major-default storage cancel so the standard
// `mul(M, v)` idiom matches the engine's other shaders (Skybox, Geometry).
//
// Preserves the TMax < 0 sentinel from generateFisheyeRay so out-of-circle
// pixels still short-circuit downstream.
public RayDesc transformRayToWorld(RayDesc cameraSpace, float4x4 invView) {
    if (cameraSpace.TMax < 0.0) return cameraSpace;
    RayDesc world;
    world.Origin    = mul(invView, float4(cameraSpace.Origin, 1.0)).xyz;
    world.Direction = normalize(mul((float3x3)invView, cameraSpace.Direction));
    world.TMin      = cameraSpace.TMin;
    world.TMax      = cameraSpace.TMax;
    return world;
}


// Pinhole
// Standard perspective. tanHalfFovY = tan(verticalFOV / 2). Aspect is
// already in ndc.x, so no aspect parameter here.
public RayDesc generatePinholeRay(float2 ndc, float tanHalfFovY) {
    float3 dir = normalize(float3(
         ndc.x * tanHalfFovY,
        -ndc.y * tanHalfFovY,   // flip: NDC.y > 0 is bottom → ray points -Y
        -1.0));
    return RayDesc(float3(0.0), 0.0, dir, INFINITY);
}


// Thin lens / DoF
// Builds the pinhole ray, picks a focal point along it at `focusDistance`,
// then re-aims from a sampled point on the aperture disk through that focal
// point. `lensUV` is a uniform [0,1]² random pair (one per ray sample).
//
// `apertureRadius` is the physical aperture half-width.  If you want
// photographer-style controls, compute it in the caller:
//     apertureRadius = focalLength / (2 * fStop);
// — that keeps this function decoupled from f-number semantics.
public RayDesc generateThinLensRay(float2 ndc, float tanHalfFovY, float2 lensUV,
                            float focusDistance, float apertureRadius) {
    RayDesc pin = generatePinholeRay(ndc, tanHalfFovY);

    // Where the pinhole ray crosses the focal plane (Z = -focusDistance).
    // -direction.z accounts for the -Z forward convention.
    float  t          = focusDistance / -pin.Direction.z;
    float3 focusPoint = pin.Direction * t;

    // Uniform disk sample: sqrt on the radius is the area-preserving
    // transform from a uniform [0,1] square to a uniform disk.
    float  theta = lensUV.x * TWO_PI;
    float  r     = sqrt(lensUV.y) * apertureRadius;
    float3 origin = float3(cos(theta) * r, sin(theta) * r, 0.0);

    return RayDesc(origin, 0.0, normalize(focusPoint - origin), INFINITY);
}


// Generalized Panini
// `cameraFovAngle` is the full FOV (radians) you want at the screen edge in
// the Panini-projected image. `paniniDistance` d ∈ [0, 1]: 0 collapses to
// pinhole, 1 is the classic Panini look. `verticalCompression` ∈ [0, 1]
// trades vertical stretch (0) for face-preserving compression (1).
//
// NOTE: the M / x / z form below is a non-canonical inverse parameterization
// pulled from the original draft.  Boundary check (d=0 collapses to pinhole)
// passes, but I haven't validated it against Sharpless's canonical inverse.
// Worth eyeballing against a reference Panini render once it's running — if
// it bows wrong at wide FOV, swap in the quadratic-solve Sharpless inverse.
public RayDesc generatePaniniRay(float2 ndc, float cameraFovAngle, float paniniDistance,
                          float verticalCompression) {
    float halfFov       = cameraFovAngle * 0.5;
    float halfPaniniFov = atan2(sin(halfFov), cos(halfFov) + paniniDistance);

    // Single FOV scaling — the original applied it twice (raw FOV then
    // halfPaniniFov), which dimensioned hv in radians². ndc maps [-1, 1]
    // to [-halfPaniniFov, +halfPaniniFov] in the projected angle space.
    float2 hv = ndc * halfPaniniFov;

    float d = paniniDistance;
    float M = sqrt(1.0 - pow(sin(hv.x) * d, 2.0) + d * cos(hv.x));

    float x =  sin(hv.x) * M;
    float z =  cos(hv.x) * M - d;
    float y = -lerp(tan(hv.y) * (d + z),
                    tan(hv.y) * z,
                    verticalCompression);   // negate: NDC.y > 0 = bottom

    return RayDesc(float3(0.0), 0.0, normalize(float3(x, y, -z)), INFINITY);
}


// Equidistant fisheye
// True equidistant projection: r ∝ θ from the optical axis (not lat/long).
// `cameraFovAngle` is the full diameter FOV in radians (PI = 180° fisheye).
// Pixels outside the fisheye circle return TMax = RAY_INVALID_TMAX; the
// raygen entry point must check and write a black sample for those.
//
// To switch projections, replace the `theta = r * halfFov` line:
//   equisolid     : theta = 2.0 * asin(r * sin(halfFov * 0.5));
//   stereographic : theta = 2.0 * atan(r * tan(halfFov * 0.5));
public RayDesc generateFisheyeRay(float2 ndc, float cameraFovAngle) {
    float halfFov = min(cameraFovAngle, TWO_PI) * 0.5;
    float r       = length(ndc);

    // Outside the fisheye image circle (corners of non-square viewports,
    // or any pixel that would map past the optical-axis hemisphere).
    if (r * halfFov > HALF_PI) {
        return RayDesc(float3(0.0), 0.0, float3(0.0), RAY_INVALID_TMAX);
    }

    float theta = r * halfFov;            // angle from optical axis (-Z)
    float phi   = atan2(ndc.y, ndc.x);    // azimuth around -Z
    float sinT  = sin(theta);

    float3 dir = float3(
         sinT * cos(phi),
        -sinT * sin(phi),                 // flip: NDC.y > 0 = bottom
        -cos(theta));
    return RayDesc(float3(0.0), 0.0, dir, INFINITY);
}
//RNG generation

public uint jenkinsHash(uint x) {
    x += x << 10;
    x ^= x >> 6;
    x += x << 3;
    x ^= x >> 11;
    x += x << 15;
    return x;
}

// Mix both pixel coordinates and the frame into the seed via independent
// prime multipliers, then jenkins-hash. The previous formula collapsed when
// the local thread index was 0, giving identical seeds along every 8th
// column → visible vertical streaks in the noise. `gtid` is no longer needed
// for decorrelation; left in the signature for callers that still pass it.
public uint initRNG(uint2 pixel, uint2 gtid, uint frame) {
    uint seed = pixel.x * 1973u
              ^ pixel.y * 9277u
              ^ frame   * 26699u;
    return jenkinsHash(seed);
}

public float uintToFloat(uint x) {
    return asfloat(0x3f800000 | (x >> 9)) - 1.0;
}

public uint xorshift(inout uint state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

public float rand(inout uint rngState) {
    return uintToFloat(xorshift(rngState));
}

public float luminance(float3 c) {
    return 0.2126 * c.r +  0.7152 * c.g +  0.0722 * c.b;
}

// Orthonormal basis from a single normal
// Pixar / Frisvad revised: branchless, handles N.z < 0 without the original
// 2003 numerical breakdown near -Z. Use to lift a tangent-space sample into
// world space when you only have N (no vertex tangent / bitangent).
public void buildOrthonormalBasis(float3 n, out float3 t, out float3 b) {
    float s  = (n.z >= 0.0) ? 1.0 : -1.0;
    float a  = -1.0 / (s + n.z);
    float bx = n.x * n.y * a;
    t = float3(1.0 + s * n.x * n.x * a, s * bx, -s * n.x);
    b = float3(bx, s + n.y * n.y * a, -n.y);
}

// Cosine-weighted hemisphere sample around `N`. Returns the world-space
// direction; writes the PDF (NdotL / PI). Lambert importance — cheap and
// matches the diffuse lobe; specular gets full BRDF eval but is under-sampled
// at high roughness (acceptable starting point; swap for split BRDF / GGX
// VNDF sampling when noise on glossy surfaces becomes the dominant artifact).
public float3 sampleCosineHemisphere(float3 N, inout uint rngState, out float pdf) {
    float u1 = rand(rngState);
    float u2 = rand(rngState);

    float r   = sqrt(u1);
    float phi = TWO_PI * u2;
    // Tangent-space direction with z = up.
    float3 dTan = float3(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u1)));

    float3 T, B;
    buildOrthonormalBasis(N, T, B);
    float3 worldDir = dTan.x * T + dTan.y * B + dTan.z * N;

    pdf = dTan.z * INV_PI;
    return normalize(worldDir);
}

// Clearcoat F0 is fixed by KHR_materials_clearcoat — the coat layer is always
// IOR 1.5, so Schlick reduces to the standard dielectric 0.04.
public static const float CLEARCOAT_F0 = 0.04;

// glTF metallic-roughness BRDF eval
// Returns f_r(V, L) WITHOUT the cosine term — caller multiplies by NdotL.
// Mirrors the inline math in PbrShader.slang's lighting loop; consolidating
// it here lets both NEE (light-sampled) and the BSDF-sampled indirect leg of
// the pathtracer reuse the same evaluation. This is the BASE BRDF only —
// callers that want the clearcoat layer on top use evaluateFullBrdf below.
public float3 evaluatePbrBrdf(HitInfo h, float3 V, float3 L) {
    float3 N     = h.normal;
    float  NdotL = dot(N, L);
    float  NdotV = dot(N, V);
    if (NdotL <= 0.0 || NdotV <= 0.0) return float3(0.0);

    float3 H     = normalize(V + L);
    float  NdotH = saturate(dot(N, H));
    float  HdotV = saturate(dot(H, V));

    float3 baseRGB = h.baseColor.rgb;
    float3 F0      = lerp(float3(h.F0), baseRGB, h.metallic);

    float  D = DistributionGGX(NdotH, h.roughness);
    float  G = GeometrySmith(NdotV, NdotL, h.roughness);
    float3 F = FresnelSchlick(HdotV, F0);

    float3 specular = (D * G * F) / max(4.0 * NdotV * NdotL, 1e-4);
    // (1-transmission) scales out the diffuse share of energy that has been
    // promised to refraction — without it, glass would double-count the
    // transmitted light path as both refracted and diffusely scattered.
    float3 kD       = (float3(1.0) - F) * (1.0 - h.metallic) * (1.0 - h.transmission);
    float3 diffuse  = kD * baseRGB * INV_PI;
    return diffuse + specular;
}

// Clearcoat lobe
// Pure GGX-Smith reflection with the clearcoat's fixed IOR-1.5 Fresnel.
// Returns the raw lobe BRDF — caller multiplies by NdotL.  No diffuse / no
// transmission: the spec is reflection-only.  Uses `clearcoatNormal` so a
// separate normal-mapped coat (orange peel) just plugs in when populated.
public float evaluateClearcoatBrdf(HitInfo h, float3 V, float3 L) {
    float3 N    = h.clearcoatNormal;
    float NdotL = dot(N, L);
    float NdotV = dot(N, V);
    if (NdotL <= 0.0 || NdotV <= 0.0) return 0.0;

    float3 H    = normalize(V + L);
    float NdotH = saturate(dot(N, H));
    float HdotV = saturate(dot(H, V));

    float D = DistributionGGX(NdotH, h.clearcoatRoughness);
    float G = GeometrySmith(NdotV, NdotL, h.clearcoatRoughness);
    float F = CLEARCOAT_F0 + (1.0 - CLEARCOAT_F0) * pow(1.0 - HdotV, 5.0);

    return (D * G * F) / max(4.0 * NdotV * NdotL, 1e-4);
}

// Full layered BRDF: base + clearcoat
// Single-Fresnel approximation: light reflected by the coat is removed from
// what reaches the base via baseAtten = (1 - clearcoat * F_cc(NdotV)). The
// "correct" two-fresnel version applies the same factor on the way out too
// (squared), which is darker. Single-fresnel matches what UE / Filament /
// most realtime PBRs do; swap to squared if you want stricter physical
// correctness at the cost of energy.
public float3 evaluateFullBrdf(HitInfo h, float3 V, float3 L) {
    float3 base = evaluatePbrBrdf(h, V, L);
    if (h.clearcoat <= 0.0) return base;

    float NdotV = max(dot(h.clearcoatNormal, V), 0.0);
    float Fcc_v = CLEARCOAT_F0 + (1.0 - CLEARCOAT_F0) * pow(1.0 - NdotV, 5.0);
    float baseAtten = 1.0 - h.clearcoat * Fcc_v;

    float ccBrdf = evaluateClearcoatBrdf(h, V, L);
    return base * baseAtten + float3(ccBrdf * h.clearcoat);
}

// Smith G1 for GGX (sampling-side, closed-form)
// Heitz's VNDF derivation uses the true Smith G1; the PbrUtils
// `GeometrySmith` is the Schlick-GGX approximation that's cheaper but won't
// match the sampling pdf exactly. Using the proper form here keeps the
// BSDF-sample weight unbiased.
public float G1_GGX_Smith(float NdotX, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float n2 = NdotX * NdotX;
    return 2.0 * NdotX / max(NdotX + sqrt(a2 + (1.0 - a2) * n2), 1e-6);
}

// GGX VNDF half-vector sample (Heitz 2018)
// Samples only those microfacet normals visible from V — eliminates the
// "wasted samples behind the horizon" that plain GGX H sampling produces.
// `Vt` is the view direction in tangent space (N = +Z). Returns a sampled
// half-vector Ht, also in tangent space.
public float3 sampleGGXVNDF(float3 Vt, float roughness, float u1, float u2) {
    float alpha = max(roughness * roughness, 1e-4);

    // Step 1: stretch to roughness=1 hemisphere.
    float3 Vh = normalize(float3(alpha * Vt.x, alpha * Vt.y, Vt.z));

    // Step 2: orthonormal basis on the hemisphere — pick the +X axis when
    // Vh is nearly aligned with +Z to avoid a divide-by-zero.
    float  lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    float3 T1    = lensq > 0.0
                   ? float3(-Vh.y, Vh.x, 0.0) * rsqrt(lensq)
                   : float3(1.0, 0.0, 0.0);
    float3 T2    = cross(Vh, T1);

    // Step 3: parameterise the projected hemisphere as a unit disk that
    // gets squashed along T2 to match the (visible) hemisphere area.
    float r   = sqrt(u1);
    float phi = TWO_PI * u2;
    float t1  = r * cos(phi);
    float t2  = r * sin(phi);
    float s   = 0.5 * (1.0 + Vh.z);
    t2        = (1.0 - s) * sqrt(max(0.0, 1.0 - t1 * t1)) + s * t2;

    // Step 4: reproject to the hemisphere.
    float3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;

    // Step 5: unstretch.
    return normalize(float3(alpha * Nh.x, alpha * Nh.y, max(0.0, Nh.z)));
}

// Split BSDF sample (diffuse + GGX specular + delta transmission)
// Three lobes picked stochastically per call:
//   * Specular reflect  : GGX VNDF importance sample
//   * Specular transmit : delta refraction (Snell + Fresnel), smooth glass
//                          only — rough-glass GGX transmission TBD
//   * Diffuse           : cosine-weighted hemisphere
//
// Lobe selection probabilities derive from a Fresnel/transmission/metal
// split. Energy bookkeeping:
//   Fl  = luminance(F(NdotV))                       — reflected share
//   Tl  = (1 - Fl) * transmission                   — transmitted share
//   Dl  = (1 - Fl) * (1 - transmission) * (1 - met) — diffuse share
// then normalised. Metals → Fl ≈ 1, no transmit / diffuse. Full glass →
// Tl ≈ 1 - Fl, no diffuse.
//
// `weight` already includes the cosine and 1/pdf — multiply straight into
// throughput. `transmit` flags the refraction path; caller must offset the
// next ray along the *opposite* face.
public struct BsdfSample {
    public float3 wi;
    public float  pdf;
    public float3 weight;
    public bool   valid;
    public bool   transmit;   // wi crosses the surface (refraction)
};

// Combined non-delta direction PDF given the lobe pick probabilities. Used
// for the spec / diffuse / clearcoat branches; the delta transmit lobe has
// zero contribution to non-delta directions and is excluded.
public float bsdfCombinedPdf(HitInfo h, float3 V, float3 wi,
                     float Pcc, float Pspec, float Pdiff) {
    float NdotV = max(dot(h.normal, V), 0.0);
    float NdotL = max(dot(h.normal, wi), 0.0);
    if (NdotL <= 0.0 || NdotV <= 0.0) return 0.0;

    float3 H     = normalize(V + wi);
    float  NdotH = saturate(dot(h.normal, H));

    float pdfDiff = NdotL * INV_PI;

    float D_b   = DistributionGGX(NdotH, h.roughness);
    float G1V_b = G1_GGX_Smith(NdotV, h.roughness);
    float pdfSpec = (G1V_b * D_b) / max(4.0 * NdotV, 1e-6);

    float pdfCC = 0.0;
    if (Pcc > 0.0) {
        // Clearcoat normal currently mirrors shading normal; revisit when a
        // separate clearcoat tangent frame is plumbed for the orange-peel
        // normal-map case.
        float D_cc   = DistributionGGX(NdotH, h.clearcoatRoughness);
        float G1V_cc = G1_GGX_Smith(NdotV, h.clearcoatRoughness);
        pdfCC = (G1V_cc * D_cc) / max(4.0 * NdotV, 1e-6);
    }

    return Pcc * pdfCC + Pspec * pdfSpec + Pdiff * pdfDiff;
}

public BsdfSample sampleBsdf(HitInfo h, float3 V, inout uint rngState) {
    BsdfSample s;
    s.wi       = float3(0.0);
    s.pdf      = 0.0;
    s.weight   = float3(0.0);
    s.valid    = false;
    s.transmit = false;

    float3 N     = h.normal;
    float  NdotV = dot(N, V);
    if (NdotV <= 0.0) return s;

    // Tangent-space basis from the shading normal.
    float3 T, B;
    buildOrthonormalBasis(N, T, B);
    float3 Vt = float3(dot(V, T), dot(V, B), NdotV);

    // Lobe weighting
    // Clearcoat sits on top: its Fresnel determines what fraction of energy
    // it grabs before light reaches the base. The remaining (1 - cc*Fcc_v)
    // splits between the base lobes (spec / transmit / diffuse) just like
    // before. Metals push spec high; transmissive materials push transmit
    // high; clearcoated materials reserve some energy for the coat lobe.
    float3 F0  = lerp(float3(h.F0), h.baseColor.rgb, h.metallic);
    float3 Fv  = FresnelSchlick(NdotV, F0);
    float  Fl  = clamp(luminance(Fv), 0.05, 0.95);

    float Fcc_v     = CLEARCOAT_F0 + (1.0 - CLEARCOAT_F0) * pow(1.0 - NdotV, 5.0);
    float ccRaw     = h.clearcoat * Fcc_v;
    float baseAtten = 1.0 - ccRaw;

    float Pcc_raw   = ccRaw;
    float Pspec_raw = Fl * baseAtten;
    float Pt_raw    = (1.0 - Fl) * h.transmission * baseAtten;
    float Pd_raw    = (1.0 - Fl) * (1.0 - h.transmission) * (1.0 - h.metallic) * baseAtten;
    float Sum = Pcc_raw + Pspec_raw + Pt_raw + Pd_raw;
    if (Sum <= 1e-6) return s;

    float Pcc     = Pcc_raw   / Sum;
    float Pspec   = Pspec_raw / Sum;
    float Ptrans  = Pt_raw    / Sum;
    float Pdiff   = Pd_raw    / Sum;

    float u  = rand(rngState);

    if (u < Pcc) {
        // Clearcoat GGX-VNDF reflection
        float u1 = rand(rngState);
        float u2 = rand(rngState);
        float3 Ht = sampleGGXVNDF(Vt, h.clearcoatRoughness, u1, u2);
        float3 Lt = reflect(-Vt, Ht);
        if (Lt.z <= 0.0) return s;

        s.wi  = normalize(Lt.x * T + Lt.y * B + Lt.z * N);
        s.pdf = bsdfCombinedPdf(h, V, s.wi, Pcc, Pspec, Pdiff);
        if (s.pdf <= 0.0) return s;

        float NdotL = Lt.z;
        s.weight = evaluateFullBrdf(h, V, s.wi) * NdotL / s.pdf;
        s.valid  = true;
        return s;
    }

    if (u < Pcc + Pspec) {
        // Base specular GGX-VNDF reflection
        float u1 = rand(rngState);
        float u2 = rand(rngState);
        float3 Ht = sampleGGXVNDF(Vt, h.roughness, u1, u2);
        float3 Lt = reflect(-Vt, Ht);
        if (Lt.z <= 0.0) return s;

        s.wi  = normalize(Lt.x * T + Lt.y * B + Lt.z * N);
        s.pdf = bsdfCombinedPdf(h, V, s.wi, Pcc, Pspec, Pdiff);
        if (s.pdf <= 0.0) return s;

        float NdotL = Lt.z;
        s.weight = evaluateFullBrdf(h, V, s.wi) * NdotL / s.pdf;
        s.valid  = true;
        return s;
    }

    if (u < Pcc + Pspec + Ptrans) {
        // Delta refraction (smooth glass)
        float etaI = h.frontFace ? 1.0   : h.ior;
        float etaT = h.frontFace ? h.ior : 1.0;
        float eta  = etaI / etaT;

        float cos2t = 1.0 - eta * eta * (1.0 - NdotV * NdotV);
        if (cos2t < 0.0) return s;   // TIR — bail; spec lobe handles it via
                                     // its own importance sample with Pspec.

        float3 wt = refract(-V, N, eta);
        if (dot(wt, wt) < 1e-6) return s;
        s.wi       = normalize(wt);
        s.transmit = true;

        // Delta weight. baseAtten scales the transmitted energy because light
        // first has to pass through the clearcoat layer. eta² is the standard
        // radiometric refraction term for the change in solid-angle measure.
        float  transmittance = 1.0 - Fl;
        float3 tint          = h.baseColor.rgb;
        s.pdf    = Ptrans;
        s.weight = tint * transmittance * (eta * eta) * baseAtten / max(Ptrans, 1e-6);
        s.valid  = true;
        return s;
    }

    // Cosine-weighted diffuse
    float u1 = rand(rngState);
    float u2 = rand(rngState);
    float r   = sqrt(u1);
    float phi = TWO_PI * u2;
    float3 Lt = float3(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u1)));
    s.wi = normalize(Lt.x * T + Lt.y * B + Lt.z * N);

    float NdotL = Lt.z;
    if (NdotL <= 0.0) return s;

    s.pdf = bsdfCombinedPdf(h, V, s.wi, Pcc, Pspec, Pdiff);
    if (s.pdf <= 0.0) return s;

    s.weight = evaluateFullBrdf(h, V, s.wi) * NdotL / s.pdf;
    s.valid  = true;
    return s;
}
