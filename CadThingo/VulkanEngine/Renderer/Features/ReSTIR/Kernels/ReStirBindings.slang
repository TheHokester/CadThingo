// Shared ReSTIR DI record types + the graph-owned working set (set 2). Imported by all three
// ReSTIR kernels (the ReStirDI trace + BuildTemporal + SpatialShade compute passes) so the buffer
// bindings and struct strides are declared ONCE and can never drift apart between them -- a silent
// drift (e.g. a working buffer at set 3 in one kernel, set 2 in another) renders pure black.
module ReStirBindings;

// Per-pixel DI reservoir over the UNIFIED light set (analytic punctual/dir lights + emissive
// triangles). Scalar/tight Slang packing (float3 stays 12B) -> std430 stride 64B; MUST match
// ReStirDIPipeline.ReservoirBytes.
public struct Reservoir {
    public uint   sampleType;   // SAMPLE_NONE / SAMPLE_ANALYTIC / SAMPLE_EMISSIVE
    public uint   index;        // analytic: light index (emissive: unused)
    public float3 samplePos;    // emissive: sampled world point Q (analytic: unused)
    public float3 sampleLe;     // emissive: radiance Le at Q (analytic: unused)
    public float  sampleEsPdf;  // emissive: NEE solid-angle pdf at build (NEE<->BSDF MIS)
    public uint   M;            // candidates streamed
    public float  wSum;         // running sum of RIS weights
    public float  W;            // unbiased contribution weight
    public float  pHat;         // target fn of the selected sample
};

// Fat G-buffer = the compact hit record the closesthit packs. Stored per pixel so the deferred
// SpatialShade pass reconstructs the BRDF without re-tracing. 40B data, rounded to 48B stride
// (float3 align); MUST match ReStirDIPipeline.PackedHitBytes.
public struct PackedHit {
    public float3 worldPos;    // full precision (shadow-ray origins / NEE)
    public uint   nOct;        // shading normal, oct 2x16
    public uint   ngOct;       // geometric normal, oct 2x16
    public uint   baseIor;     // baseColor.rgb unorm8 | ior unorm8 over [1,3]
    public uint   emisRG;      // emissive.rg as f16x2
    public uint   scal0;       // metallic | roughness | transmission | clearcoat (unorm8x4)
    public uint   scal1;       // clearcoatRoughness unorm8 | emissive.b f16 in bits 16..31
    public uint   matFlags;    // matidx:24 | hit:bit24 | frontFace:bit25
};

public static const uint SAMPLE_NONE     = 0u;
public static const uint SAMPLE_ANALYTIC = 1u;
public static const uint SAMPLE_EMISSIVE = 2u;

// Set 2 (graph-shared, graph-owned): reservoir + fat-G-buffer ping-pong pair + deferred
// sceneRadiance hand-off. Ping-pong is by frame parity (push constant), both halves bound at once.
public [[vk::binding(0, 2)]] RWStructuredBuffer<Reservoir> reservoirA;
public [[vk::binding(1, 2)]] RWStructuredBuffer<Reservoir> reservoirB;
public [[vk::binding(2, 2)]] RWStructuredBuffer<PackedHit> gbufferA;
public [[vk::binding(3, 2)]] RWStructuredBuffer<PackedHit> gbufferB;
public [[vk::binding(4, 2)]] RWStructuredBuffer<float4>    sceneRadiance;