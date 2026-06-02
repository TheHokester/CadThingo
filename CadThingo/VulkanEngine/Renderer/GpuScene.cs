using System.Numerics;
using System.Runtime.InteropServices;
using CadThingo.VulkanEngine.GLTF;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer;

[StructLayout(LayoutKind.Sequential)]
public struct EmissiveTriGpu
{
    public Vector4 P0Area;   // xyz = p0 (world),       w = triangle area (world)
    public Vector4 E1LeR;    // xyz = p1 - p0,          w = Le.r
    public Vector4 E2LeG;    // xyz = p2 - p0,          w = Le.g
    public Vector4 NLeB;     // xyz = geometric normal, w = Le.b
    public int IndexOffset;  // element offset into globalIndices (= Mesh.offset)
    public int PrimIndex;    // triangle index within the mesh
    public int EmissiveTex;  // bindless emissive-texture index (-1 = none)
    public int _pad;
}

[StructLayout(LayoutKind.Sequential)]
public struct ShadowEntityInfo
{
    public uint IndexOffset; // uint elements into GlobalIndexBuffer for this mesh's first triangle
    public uint MaterialIndex; // index into the materials SSBO
    public uint Flags; // copy of PbrMaterial.Flags — bit 0 = MASK, bit 2 = BLEND

    public uint EntityIndex; // scene entity index — pick/selection resolve through this

    // Per-geometry world transform (column-vector 3x4 rows). The cluster BLAS is
    // world-space with an identity instance transform, so the shader applies this
    // to rotate fetched object-space normals into world space.
    public Vector4 Xform0;
    public Vector4 Xform1;
    public Vector4 Xform2;
}

[StructLayout(LayoutKind.Sequential)]
public struct PbrMaterial
{
    public Vector4 BaseColorFactor;
    public Vector3 EmissiveFactor;
    public float AlphaCutoff;
    public float MetallicFactor;
    public float RoughnessFactor;
    public uint Flags; //bit 0 alphaMask, bit 1 doubleSided, bit 2 alphaBlend, ...
    public int BaseColorTex;
    public int PhysicalDescriptorTex;
    public int NormalTex;
    public int OcclusionTex;
    public int EmissiveTex;

    // KHR_materials_transmission + KHR_materials_ior
    public float TransmissionFactor; // 0 = opaque, 1 = full transmission
    public float Ior; // index of refraction; glTF default 1.5

    // KHR_materials_clearcoat
    public float ClearcoatFactor; // 0 = no coat, 1 = full coat
    public float ClearcoatRoughnessFactor;

    public int TransmissionTex; // R channel multiplies TransmissionFactor
    public int ClearcoatTex; // R channel multiplies ClearcoatFactor
    public int ClearcoatRoughnessTex; // G channel multiplies ClearcoatRoughnessFactor
    public int ClearcoatNormalTex; // tangent-space normal for the coat layer

    // KHR_materials_volume (Beer-Lambert absorption). White color or the
    // no-absorption sentinel distance means glass behaves as before.
    public Vector3 AttenuationColor; // glTF default white (1,1,1) = no tint
    public float AttenuationDistance; // distance to reach AttenuationColor
}

[StructLayout(LayoutKind.Sequential)]
public struct RenderableInputGpu
{
    public Matrix4x4 model; // 64B
    public Vector4 sphereLocal; // 16B  (xyz center, w radius — local space)
    public uint indexCount; //  4B  ┐
    public uint firstIndex; //  4B  │  pulled straight from Mesh
    public uint materialIndex; //  4B  │
    public uint _pad; //  4B  ┘  std430 16B alignment
}

[StructLayout(LayoutKind.Sequential)]
public struct PbrLightGpu
{
    public Vector4 positionRange; // xyz = world pos, w = range (point/spot; ignored for directional)
    public Vector4 colorIntensity; // rgb = linear color, a = intensity
    public Vector4 directionType; // xyz = direction (dir/spot, world-space), w = LightType as float
    public Vector4 spotCones; // x = innerCos, y = outerCos, z = castShadows (0/1), w = lightRadius
}

public readonly struct RenderableHandle
{
    public readonly uint Index;
    public readonly uint Generation;

    public RenderableHandle(uint index, uint generation)
    {
        Index = index;
        Generation = generation;
    }
}

readonly ref struct RenderView
{
    private readonly uint FrameIndex;
    private readonly Extent2D RenderExtent;

    private readonly Matrix4x4 View, Proj, ViewProj, InvView, InvProj, InvViewProj;
    private readonly Vector3 CamPos;

    private readonly uint RenderableCount, LightCount, MaterialCount;
    private readonly DescriptorSet SceneSet;
}

/// <summary>
/// the canonical GPU-resident mirror of the scene's render-relevant data and
/// (as the refactor progresses) the single place that reads the ECS for rendering.
///
/// Migration is incremental (renderer-refactor.md, L2 steps 1–8). This first slice
/// owns the per-frame light SSBO and hosts the light + material *extractors* (the
/// former <c>Renderer.UpdateLights</c> / <c>UpdateMaterials</c> bodies, moved here
/// verbatim). <see cref="Renderer"/> keeps same-named forwarders so existing call
/// sites — the consuming pipelines and the DrawX paths — compile unchanged.
///
/// Still to fold in over the later steps: renderable packing (out of
/// <c>DrawCullPipeline.Record</c>), the acceleration-structure buffers, stable
/// <see cref="RenderableHandle"/> identity, the cached transform pass, and the
/// single bindless scene descriptor set. Those buffer fields are intentionally
/// NOT declared yet — they arrive with the step that populates them.
///
/// Depends only on <see cref="GraphicsDevice"/>; the <see cref="Scene"/> is passed
/// per-extract, never retained.
/// </summary>
public sealed unsafe class GpuScene : IDisposable
{
    private readonly GraphicsDevice _gfx;

    // Per-frame-in-flight light SSBO. Holds packed PbrLightGpu records; bound by
    // every pipeline that needs scene lighting (deferred, light-cull, transparent,
    // PT, RT). Persistently mapped, stable for the renderer's lifetime. Owned here
    // (not by the deferred lighting pipeline) so non-deferred paths can read it
    // without that pipeline existing.
    private readonly UboBuffer[] _lightBuffers = new UboBuffer[Renderer.MAX_CONCURRENT_FRAMES];
    private readonly List<LightComponent> _lightScratch = new();

    /// <summary>Number of valid lights packed by the most recent <see cref="UpdateLights"/>.</summary>
    public uint LightCount { get; private set; }

    public GpuScene(GraphicsDevice gfx)
    {
        _gfx = gfx;
    }

    /// <summary>Light SSBO for the given frame slot. Stable for the renderer's lifetime.</summary>
    public Buffer GetLightStorageBuffer(uint frame) => _lightBuffers[frame].buffer;

    /// <summary>Allocates the per-frame light SSBOs. Call once at init, before any
    /// pipeline binds them.</summary>
    public void CreateLightBuffers()
    {
        for (int i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            _gfx.CreateMappedStorageBuffer(
                (ulong)(Renderer.MAX_LIGHTS * (uint)sizeof(PbrLightGpu)),
                ref _lightBuffers[i]);
        }
    }

    /// <summary>
    /// Copy every scene material into the per-frame material SSBO, plus a
    /// fallback entry at slot [MaterialCount] for legacy / procedural geometry
    /// without an assigned material index. Every rendering path (deferred,
    /// forward+, pathtracer) calls this once per frame so live edits in the
    /// inspector show up on the next frame regardless of which renderer is
    /// active. Returns the material count actually written (excluding the
    /// fallback).
    /// </summary>
    public uint UpdateMaterials(uint frameIndex, Scene scene)
    {
        int matCount = scene.MaterialCount;
        if (matCount + 1 > (int)Renderer.MAX_MATERIALS)
            throw new InvalidOperationException(
                $"Scene material count ({matCount}) exceeds MAX_MATERIALS ({Renderer.MAX_MATERIALS}).");

        PbrMaterial* matPtr = (PbrMaterial*)Engine.ResourceManager.GetMaterialMapped(frameIndex);
        for (int mi = 0; mi < matCount; mi++) matPtr[mi] = scene.Materials[mi];

        matPtr[matCount] = new PbrMaterial
        {
            BaseColorFactor          = new Vector4(1, 1, 1, 1),
            EmissiveFactor           = Vector3.Zero,
            AlphaCutoff              = 0f,
            MetallicFactor           = 0.3f,
            RoughnessFactor          = 0.7f,
            Flags                    = 0,
            BaseColorTex             = GltfDefaults.BaseColorIndex,
            PhysicalDescriptorTex    = GltfDefaults.MetallicRoughnessIndex,
            NormalTex                = GltfDefaults.NormalIndex,
            OcclusionTex             = GltfDefaults.OcclusionIndex,
            EmissiveTex              = GltfDefaults.EmissiveIndex,
            TransmissionFactor       = 0f,
            Ior                      = 1.5f,
            ClearcoatFactor          = 0f,
            ClearcoatRoughnessFactor = 0f,
            TransmissionTex          = GltfDefaults.OcclusionIndex,
            ClearcoatTex             = GltfDefaults.OcclusionIndex,
            ClearcoatRoughnessTex    = GltfDefaults.OcclusionIndex,
            ClearcoatNormalTex       = GltfDefaults.NormalIndex,
            AttenuationColor         = Vector3.One,
            AttenuationDistance      = PbrMaterialVolume.NoAbsorptionDistance,
        };

        return (uint)matCount;
    }

    /// <summary>Walks scene lights into the per-frame light SSBO. Returns the
    /// packed light count, also cached in <see cref="LightCount"/>. Every
    /// rendering path (deferred, forward+, pathtracer) calls this once per frame
    /// before recording its draws.</summary>
    public uint UpdateLights(uint frameIndex, Scene scene)
    {
        scene.EnumerateLights(_lightScratch);

        PbrLightGpu* lightPtr = (PbrLightGpu*)_lightBuffers[frameIndex].mapped;
        uint count = 0;
        foreach (var light in _lightScratch)
        {
            if (count >= Renderer.MAX_LIGHTS) break;

            // World-space position from the owner transform if present.
            Vector3 worldPos = Vector3.Zero;
            if (light.Owner != null)
            {
                var t = light.Owner->GetComponent<TransformComponent>();
                if (t != null)
                {
                    var w = *t.GetWorldMatrix();
                    worldPos = new Vector3(w.M41, w.M42, w.M43);
                }
            }

            // Normalize direction — guard against zero-vector default.
            Vector3 dir = light.Direction.LengthSquared() > 1e-8f
                ? Vector3.Normalize(light.Direction)
                : new Vector3(0, -1, 0);

            // Range: -1 sentinel marks directional lights so the shader can branch
            // on attenuation without inspecting Type for the most common test.
            float range = light.Type == LightType.Directional ? -1f : light.Range;

            lightPtr[count] = new PbrLightGpu
            {
                positionRange  = new Vector4(worldPos, range),
                colorIntensity = new Vector4(light.Color, light.Intensity),
                directionType  = new Vector4(dir, (float)(uint)light.Type),
                spotCones      = new Vector4(light.InnerConeCos, light.OuterConeCos,
                    light.CastShadows ? 1f : 0f, light.Radius),
            };
            count++;
        }

        LightCount = count;
        return count;
    }

    public void Dispose()
    {
        for (int i = 0; i < _lightBuffers.Length; i++)
        {
            if (_lightBuffers[i].buffer.Handle != 0)
                _gfx.DestroyBuffer(_lightBuffers[i].buffer, _lightBuffers[i].alloc);
        }
    }
}