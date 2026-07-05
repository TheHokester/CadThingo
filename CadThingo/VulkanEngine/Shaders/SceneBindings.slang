// Canonical scene descriptor set (set 0) - the single source of truth for the unified
// scene layout (docs/descriptor-system.md section 5). Shaders import this module instead
// of declaring their own scene resources; DescriptorRegistry reflects it once to build
// the canonical VkDescriptorSetLayout and owns the per-frame set instances.
//
// Binding 0 is deliberately NOT declared here: it is the reserved per-pass constant slot.
// An entry shader declares its own
//     [[vk::binding(0, 0)]] ConstantBuffer<MyPassParams> gPass;
// and the canonical layout carries binding 0 as UNIFORM_BUFFER_DYNAMIC pointing into the
// per-frame constant arena (SPIR-V does not encode dynamic-ness, so the shader's plain
// ConstantBuffer stays layout-compatible).
//
// The unbounded texture array must stay the HIGHEST binding (variable-count rule).
module SceneBindings;

__exported import CommonTypes;
import PTUtils;

public [[vk::binding( 1, 0)]] StructuredBuffer<PbrMaterial>      sceneMaterials;
public [[vk::binding( 2, 0)]] StructuredBuffer<InstanceData>     sceneInstances;
public [[vk::binding( 3, 0)]] StructuredBuffer<PbrLight>         sceneLights;
public [[vk::binding( 4, 0)]] StructuredBuffer<ShadowEntityInfo> sceneEntityInfo;
public [[vk::binding( 5, 0)]] RaytracingAccelerationStructure    sceneTlas;
public [[vk::binding( 6, 0)]] ByteAddressBuffer                  sceneVertices;
public [[vk::binding( 7, 0)]] ByteAddressBuffer                  sceneIndices;
public [[vk::binding( 8, 0)]] StructuredBuffer<EmissiveTri>      sceneEmissiveTris;
public [[vk::binding( 9, 0)]] StructuredBuffer<AliasEntry>       sceneEmissiveAlias;
public [[vk::binding(10, 0)]] SamplerState                       sceneSamplers[16];
public [[vk::binding(11, 0)]] Texture2D                          sceneTextures[];