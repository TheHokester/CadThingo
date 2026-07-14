// FeatureEnv descriptor set (set 4): the raw environment cube, consumed by the shaders that
// sample the background / miss direction directly (skybox + the path tracers). Kept separate
// from FeatureIBL because those consumers are disjoint from the raster lighting shaders.
// DescriptorRegistry reflects this module for the layout + per-frame sets; IblSystem owns the
// cube resource and registers it by name.
module FeatureEnv;

public [[vk::binding(0, 4)]] SamplerCube envCube;