// FeatureIBL descriptor set (set 3): the image-based-lighting split-sum + reflection-probe
// resources consumed by the raster lighting shaders (PBR deferred + Transparent forward).
// Shaders import this module instead of declaring their own; DescriptorRegistry reflects it
// once to build the VkDescriptorSetLayout and owns the per-frame set instances. The underlying
// resources stay owned by IblSystem / ReflectionProbeSystem, which register them by name.

module FeatureIBL;

__exported import CommonTypes; // ProbeRecord / ProbeClusterRange

// Global IBL split-sum: irradiance (diffuse) + prefiltered (per-roughness specular) cubes and
// the view-independent BRDF LUT. Black-cleared until an environment is loaded.
public [[vk::binding(0, 3)]] SamplerCube      irradianceCube;
public [[vk::binding(1, 3)]] SamplerCube      prefilteredCube;
public [[vk::binding(2, 3)]] Sampler2D        brdfLut;

// Reflection probes: prefiltered cube array + per-probe records and the cluster lookup tables.
public [[vk::binding(3, 3)]] SamplerCubeArray                     probeCubeArray;
public [[vk::binding(4, 3)]] StructuredBuffer<ProbeRecord>        probes;
public [[vk::binding(5, 3)]] StructuredBuffer<ProbeClusterRange>  probeClusterRange;
public [[vk::binding(6, 3)]] StructuredBuffer<uint>              probeIndexList;