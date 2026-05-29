// CommonTypes — the data layer shared by every shader in the engine.
//
// Anything whose memory layout must agree with a C# struct lives here so there
// is exactly one definition to keep in sync. This is a Slang *module*: consumers
// write `import CommonTypes;` (or get it transitively via `import PbrUtils;`).
//
// Visibility note: Slang module symbols default to `internal` (invisible across
// an `import`). Everything meant to be used by other shaders is marked `public`
// — and for structs whose FIELDS are read directly, every field is `public` too
// (a `public struct` alone does not export its members).

module CommonTypes;

// Math constants — previously redeclared per-file (PbrUtils, IblCommon, PTUtils).
public static const float PI         = 3.14159265359;
public static const float TWO_PI     = 6.28318530718;
public static const float INV_PI     = 0.31830988618;
public static const float INV_TWO_PI = 0.15915494309;
public static const float HALF_PI    = 1.57079632679;

// Material / instance / light layouts

public struct PbrMaterial {
    public float4 baseColorFactor;        // Base color tint/multiplier
    public float3 emissiveFactor;         // HDR emissive (glTF factor × KHR_materials_emissive_strength)
    public float alphaCutoff;             // Packs into emissiveFactor's std430 trailing 4B
    public float metallicFactor;          // Metallic property multiplier
    public float roughnessFactor;         // Surface roughness multiplier
    public uint flags;                    // bit 0 = alphaMask, bit 1 = doubleSided, bit 2 = alphaBlend, ...
    public int baseColorTex;              // Texture binding index for base color (-1 = none)
    public int physicalDescriptorTex;     // Texture binding for metallic/roughness
    public int normalTex;                 // Texture binding for normal maps
    public int occlusionTex;              // Texture binding for ambient occlusion
    public int emissiveTex;               // Texture binding for emissive maps

    // KHR_materials_transmission + KHR_materials_ior. Fields are present on
    // every material upload but consumers can ignore them — opaque materials
    // have transmissionFactor=0 / ior=1.5 (glTF defaults) and read the same.
    public float transmissionFactor;
    public float ior;

    // KHR_materials_clearcoat. Same opt-in model: clearcoatFactor=0 means
    // the coat lobe contributes nothing, so a BRDF that doesn't implement
    // clearcoat yet simply leaves these fields unread.
    public float clearcoatFactor;
    public float clearcoatRoughnessFactor;
    public int transmissionTex;           // R channel multiplies transmissionFactor
    public int clearcoatTex;              // R channel multiplies clearcoatFactor
    public int clearcoatRoughnessTex;     // G channel multiplies clearcoatRoughnessFactor
    public int clearcoatNormalTex;        // tangent-space normal for the coat

    // KHR_materials_volume (absorption only — thickness / scattering ignored).
    // attenuationColor = the color light becomes after traveling
    // attenuationDistance through the medium. White color OR a sentinel
    // distance (≤0 / ≥1e6) ⇒ no absorption, so thin panes and untagged glass
    // are unchanged. float3 lands at a 16B-aligned offset; no padding needed.
    public float3 attenuationColor;
    public float  attenuationDistance;
};

public struct InstanceData {
    public float4x4 model;
    public uint materialIndex;
    public uint _pad0, _pad1, _pad2;
};

// Per-light record packed by Renderer_Resources.UpdateLightingBuffers.
// Conventions:
//   positionRange.w  =  range (point/spot)     — directional uses -1 sentinel
//   directionType.w  =  LightType encoded as float (0=Dir, 1=Point, 2=Spot)
//   spotCones.z      =  CastShadows (0/1)
public struct PbrLight {
    public float4 positionRange;//   positionRange.w  =  range (point/spot)     — directional uses -1 sentinel
    public float4 colorIntensity;
    public float4 directionType;//   directionType.w  =  LightType encoded as float (0=Dir, 1=Point, 2=Spot)
    public float4 spotCones;//   spotCones.z      =  CastShadows (0/1)
};

// Light type tags — must match CadThingo.VulkanEngine.LightType.
public static const uint LIGHT_TYPE_DIRECTIONAL = 0u;
public static const uint LIGHT_TYPE_POINT       = 1u;
public static const uint LIGHT_TYPE_SPOT        = 2u;

// Reflection-probe records

// Layout mirrors ReflectionProbeSystem.ProbeGpuRecord (32B std430).
public struct ProbeRecord {
    public float4 positionRadius;
    public uint   cubeArraySlot;
    public uint   _pad0;
    public uint   _pad1;
    public uint   _pad2;
};
// Layout mirrors ProbeClusterGrid.ClusterRange (8B std430).
public struct ProbeClusterRange {
    public uint offset;
    public uint count;
};

// Shadow / TLAS geometry lookup
// One record per (cluster instance, geometry) pair. A TLAS instance's
// InstanceCustomIndex is the base offset of its cluster's geometry block, so
// the flat record index is `InstanceCustomIndex + GeometryIndex`. Filled
// CPU-side in Renderer_Ray_Query.cs::RebuildTlas.
public struct ShadowEntityInfo {
    public uint indexOffset;     // first uint of this primitive's triangle list inside globalIndices
    public uint materialIndex;   // index into materials[]
    public uint flags;           // bit 0 = MASK, bit 2 = BLEND
    public uint entityIndex;     // scene entity index — pick/selection resolve through this
    // Per-geometry world transform (column-vector 3x4, same packing as the BLAS
    // build transform). The cluster BLAS bakes positions to world space with an
    // IDENTITY instance transform, so CommittedObjectToWorld3x4 can't rotate the
    // fetched object-space normals — apply this 3x3 instead. Rows i = (m_i0,m_i1,
    // m_i2, t_i); only .xyz (the linear part) is used for directions.
    public float4 xform0;
    public float4 xform1;
    public float4 xform2;
}

// 3x3 linear part of the per-geometry world transform, for rotating fetched
// object-space normals/tangents into world space (uniform-scale assumption,
// matching the old ObjectToWorld path).
public float3x3 entityNormalMatrix(ShadowEntityInfo info) {
    return float3x3(info.xform0.xyz, info.xform1.xyz, info.xform2.xyz);
};

// Tile-cull constants — must match Renderer_Resources.cs.
public static const uint TILE_SIZE_PX        = 16u;
public static const uint MAX_LIGHTS_PER_TILE = 64u;

// Vertex stream layout + fetch helpers
// Must match the C# Vertex struct in MeshComponent.cs:
//   Position(12) + Normal(12) + TexCoord(8) + Tangent(16) = 48 bytes
// Helpers take the ByteAddressBuffer as a parameter so each shader passes its
// own binding — the module stays binding-agnostic.
public static const uint VERTEX_STRIDE         = 48u;
public static const uint VERTEX_POS_OFFSET     = 0u;
public static const uint VERTEX_NORMAL_OFFSET  = 12u;   // 12 (Position)
public static const uint VERTEX_UV_OFFSET      = 24u;   // 12 (Position) + 12 (Normal)
public static const uint VERTEX_TANGENT_OFFSET = 32u;   // + 8 (TexCoord)

public float3 fetchVertexPos(ByteAddressBuffer vbuf, uint vi) {
    return asfloat(vbuf.Load3(vi * VERTEX_STRIDE + VERTEX_POS_OFFSET));
}
public float3 fetchVertexNormal(ByteAddressBuffer vbuf, uint vi) {
    return asfloat(vbuf.Load3(vi * VERTEX_STRIDE + VERTEX_NORMAL_OFFSET));
}
public float2 fetchVertexUV(ByteAddressBuffer vbuf, uint vi) {
    return asfloat(vbuf.Load2(vi * VERTEX_STRIDE + VERTEX_UV_OFFSET));
}
public float4 fetchVertexTangent(ByteAddressBuffer vbuf, uint vi) {
    return asfloat(vbuf.Load4(vi * VERTEX_STRIDE + VERTEX_TANGENT_OFFSET));
}

public uint fetchTriangleIndex(ByteAddressBuffer ibuf, uint indexOffset, uint primIdx, uint corner) {
    return ibuf.Load((indexOffset + primIdx * 3u + corner) * 4u);
}
