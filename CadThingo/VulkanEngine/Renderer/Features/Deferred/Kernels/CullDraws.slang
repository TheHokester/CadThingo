// GPU frustum-cull pass. One thread per scene renderable. Threads that pass
// the frustum test atomically append a VkDrawIndexedIndirectCommand and an
// InstanceData row to the post-cull output buffers and bump a count.
//
// Outputs feed:
//   - vkCmdDrawIndexedIndirectCount        (indirectCmd + indirectCount)
//   - geometry pipeline VS via SV_InstanceID (instanceData, indexed by firstInstance)

struct RenderableInput {
    float4x4 model;
    float4   sphereLocal;   // xyz center, w radius — local space
    uint     indexCount;
    uint     firstIndex;
    uint     materialIndex;
    uint     _pad;
};

// Mirrors VkDrawIndexedIndirectCommand exactly.
struct DrawCmd {
    uint indexCount;
    uint instanceCount;
    uint firstIndex;
    int  vertexOffset;
    uint firstInstance;
};

// Mirrors PbrUtils.slang::InstanceData. Re-declared here so this shader is
// self-contained; layout must match the geometry pipeline's instance SSBO.
struct InstanceData {
    float4x4 model;
    uint     materialIndex;
    uint     _pad0, _pad1, _pad2;
};

struct CullParams {
    // Plane equation Ax+By+Cz+D = 0; (xyz) is the unit normal pointing INTO the
    // frustum, w is the plane distance. A point p is inside-or-on iff
    // dot(plane.xyz, p) + plane.w >= 0. A sphere of radius r is fully outside iff
    // dot(n, c) + d + r < 0.
    float4 planeL;
    float4 planeR;
    float4 planeB;
    float4 planeT;
    float4 planeN;
    float4 planeF;
    uint   renderableCount;
    uint   _pad0;
    uint   _pad1;
    uint   _pad2;
};

[[vk::push_constant]] CullParams params;

// Pass set (set 0) — every binding is a graph resource, so these names are the contract the
// CullPass Read/Write binds match against by name. Renaming one here renames it in the graph.
[[vk::binding(0, 0)]] StructuredBuffer<RenderableInput>      renderables;
[[vk::binding(1, 0)]] RWStructuredBuffer<DrawCmd>            indirectCmd;
[[vk::binding(2, 0)]] RWStructuredBuffer<InstanceData>       instanceData;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>               indirectCount;

bool sphereInsidePlane(float4 plane, float3 c, float r)
{
    return dot(plane.xyz, c) + plane.w + r >= 0.0;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void Main(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= params.renderableCount) return;

    RenderableInput r = renderables[i];

    // World-space sphere center. Slang/HLSL row-vector convention: mul(model, vec)
    // applies the matrix as (row × col), matching our CPU upload of System.Numerics
    // Matrix4x4 (row-major).
    float3 c = mul(r.model, float4(r.sphereLocal.xyz, 1.0)).xyz;

    // Bloat radius by the worst-case axis scale extracted from the upper-3x3.
    // Lengths of the basis-vector rows give per-axis scale magnitudes.
    float sx = length(float3(r.model._m00, r.model._m01, r.model._m02));
    float sy = length(float3(r.model._m10, r.model._m11, r.model._m12));
    float sz = length(float3(r.model._m20, r.model._m21, r.model._m22));
    float radius = r.sphereLocal.w * max(sx, max(sy, sz));

    // 6-plane frustum test. Early-out as soon as we find a separating plane.
    if (!sphereInsidePlane(params.planeL, c, radius)) return;
    if (!sphereInsidePlane(params.planeR, c, radius)) return;
    if (!sphereInsidePlane(params.planeB, c, radius)) return;
    if (!sphereInsidePlane(params.planeT, c, radius)) return;
    if (!sphereInsidePlane(params.planeN, c, radius)) return;
    if (!sphereInsidePlane(params.planeF, c, radius)) return;

    // Visible - claim a slot atomically. InterlockedAdd returns the original value.
    uint slot;
    InterlockedAdd(indirectCount[0], 1u, slot);

    DrawCmd cmd;
    cmd.indexCount    = r.indexCount;
    cmd.instanceCount = 1;
    cmd.firstIndex    = r.firstIndex;
    cmd.vertexOffset  = 0;
    // firstInstance is what SV_InstanceID resolves to in the VS for indirect draws
    // with instanceCount==1, so we use the slot index itself as the instance lookup.
    cmd.firstInstance = slot;
    indirectCmd[slot] = cmd;

    InstanceData inst;
    inst.model         = r.model;
    inst.materialIndex = r.materialIndex;
    inst._pad0 = 0;
    inst._pad1 = 0;
    inst._pad2 = 0;
    instanceData[slot] = inst;
}