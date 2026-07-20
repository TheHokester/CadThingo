// FeaturePTIO descriptor set (set 5): the progressive-accumulation IO pair every path tracer
// writes - accumulator holds the running sum, outColor the resolved image tonemap samples.
// All four tracers (PTCompute, PathTraceRT, ReSTIR, Wavefront) target the SAME two renderer-owned
// images, so this is one registry-owned set rather than an identical pass set rebuilt per pipeline.
// The render graph still imports the images separately for barrier tracking; that is about
// layout transitions, not binding.
module FeaturePTIO;

public [[vk::binding(0, 5)]] RWTexture2D<float4> accumulator;
public [[vk::binding(1, 5)]] RWTexture2D<float4> outColor;