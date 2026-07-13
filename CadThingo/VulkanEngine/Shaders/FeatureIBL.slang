// FeatureIBL descriptor set (set 2): the image-based-lighting split-sum + reflection-probe
// resources consumed by the raster lighting shaders (PBR deferred + Transparent forward).
// Shaders import this module instead of declaring their own; DescriptorRegistry reflects it
// once to build the VkDescriptorSetLayout and owns the per-frame set instances. The underlying
// resources stay owned by IblSystem / ReflectionProbeSystem, which register them by name.
//
// Fixed global set index 2 (ShaderSets.FirstFeature). A shader using this feature but not the
// pass set (1) leaves that slot as an empty placeholder in its pipeline layout.
module FeatureIBL;

__exported import CommonTypes; // ProbeRecord / ProbeClusterRange

// Global IBL split-sum: irradiance (diffuse) + prefiltered (per-roughness specular) cubes and
// the view-independent BRDF LUT. Black-cleared until an environment is loaded.
public [[vk::binding(0, 2)]] SamplerCube      irradianceCube;
public [[vk::binding(1, 2)]] SamplerCube      prefilteredCube;
public [[vk::binding(2, 2)]] Sampler2D        brdfLut;

// Reflection probes: prefiltered cube array + per-probe records and the cluster lookup tables.
public [[vk::binding(3, 2)]] SamplerCubeArray                     probeCubeArray;
public [[vk::binding(4, 2)]] StructuredBuffer<ProbeRecord>        probes;
public [[vk::binding(5, 2)]] StructuredBuffer<ProbeClusterRange>  probeClusterRange;
public [[vk::binding(6, 2)]] StructuredBuffer<uint>              probeIndexList;