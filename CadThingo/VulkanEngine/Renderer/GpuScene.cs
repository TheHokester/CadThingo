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

    public uint EntityIndex; // RenderableHandle slot index — pick/selection resolve through this (L2 step 6)

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
    public uint indexCount;     //  4B  |
    public uint firstIndex;     //  4B  | pulled straight from Mesh
    public uint materialIndex;  //  4B  |
    public uint _pad;           //  4B  |  std430 16B alignment
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
    private readonly UboBuffer[] _lightBuffers = new UboBuffer[RenderConfig.MAX_CONCURRENT_FRAMES];
    private readonly List<LightComponent> _lightScratch = new();

    /// <summary>Number of valid lights packed by the most recent <see cref="UpdateLights"/>.</summary>
    public uint LightCount { get; private set; }

    // Per-frame cull-input SSBO — one RenderableInputGpu row per OPAQUE/MASK
    // renderable. Formerly owned + packed by DrawCullPipeline; now extracted here
    // (L2 step 4) so the cull compute pass is pure frustum cull over this buffer.
    private readonly UboBuffer[] _renderableBuffers = new UboBuffer[RenderConfig.MAX_CONCURRENT_FRAMES];
    // BLEND-mode renderables, captured during extraction WITHOUT a sort key — the
    // back-to-front sort is view-dependent (camera), so it stays downstream in the
    // cull pass which owns the camera. This list is the extraction half only.
    private readonly List<TransparentDraw> _transparentCandidates = new();

    /// <summary>OPAQUE/MASK renderable rows packed by the most recent
    /// <see cref="ExtractRenderables"/> — drives the cull dispatch + maxDrawCount.</summary>
    public uint ExtractedRenderableCount { get; private set; }

    //dirty driven scene extraction
    private const uint AllSlotsDirty = (1u << (int)RenderConfig.MAX_CONCURRENT_FRAMES) - 1u;
    private uint _materialDirty   = AllSlotsDirty;
    private uint _lightDirty      = AllSlotsDirty;
    private uint _renderableDirty = AllSlotsDirty;
    private uint _lastMaterialCount;

    /// <summary>Marks the GPU mirror stale for every frame slot. The next time each
    /// extract runs for a given slot it re-packs that slot, then leaves it clean until
    /// the next edit. Idempotent and cheap — call it generously; over-triggering only
    /// costs a re-pack, under-triggering would render stale data.</summary>
    public void MarkSceneDirty() => _materialDirty = _lightDirty = _renderableDirty = AllSlotsDirty;

    // Test-and-clear one frame-slot's dirty bit. Returns whether that slot still
    // needed a re-pack (and clears it so a clean slot is skipped next time).
    private static bool TakeSlot(ref uint mask, uint frame)
    {
        uint bit = 1u << (int)frame;
        bool was = (mask & bit) != 0u;
        mask &= ~bit;
        return was;
    }

    public GpuScene(GraphicsDevice gfx)
    {
        _gfx = gfx;
    }

    // Renderable identity 
    // Generational slot allocator. Generalizes Scene's material free-list idea to
    // a stable (index, generation) handle that survives entity-list reorder and
    // removal. This handle — not Scene.IndexOf — becomes the canonical identity for
    // the TLAS InstanceCustomIndex, pick results, ShadowEntityInfo.EntityIndex, and
    // the selection index (those consumers are repointed in L2 step 6). For now this
    // only *maintains* the map; nothing reads the handles yet.
    private readonly List<uint> _slotGeneration = new();   // current generation per slot
    private readonly List<nint> _slotEntity      = new();   // entity ptr per slot; 0 = free
    private readonly Stack<int> _freeSlots        = new();   // recycled slot indices
    private readonly Dictionary<nint, RenderableHandle> _handleOf = new();
    // SyncRenderables scratch — reused so a per-edit reconcile allocates nothing.
    private readonly HashSet<nint> _present = new();
    private readonly List<nint>    _toFree  = new();

    /// <summary>Number of entities currently holding a live renderable handle.</summary>
    public int RenderableCount => _handleOf.Count;

    /// <summary>Returns the entity's stable handle, allocating a slot on first sight.
    /// Idempotent — the same entity maps to the same slot until it is freed.</summary>
    public RenderableHandle Register(Entity* e)
    {
        nint key = (nint)e;
        if (_handleOf.TryGetValue(key, out var existing)) return existing;

        int slot;
        if (_freeSlots.Count > 0)
        {
            slot = _freeSlots.Pop();
            _slotEntity[slot] = key;
        }
        else
        {
            slot = _slotGeneration.Count;
            _slotGeneration.Add(0);
            _slotEntity.Add(key);
        }
        var handle = new RenderableHandle((uint)slot, _slotGeneration[slot]);
        _handleOf[key] = handle;
        return handle;
    }

    /// <summary>Releases the entity's slot, bumping its generation so any cached
    /// handle to it is invalidated. The slot is recycled by the next Register.</summary>
    public void Free(Entity* e)
    {
        nint key = (nint)e;
        if (!_handleOf.TryGetValue(key, out var h)) return;
        int slot = (int)h.Index;
        _slotGeneration[slot]++;      // invalidate outstanding handles to this slot
        _slotEntity[slot] = 0;
        _freeSlots.Push(slot);
        _handleOf.Remove(key);
    }

    /// <summary>True if the handle still refers to the live entity it was issued for.</summary>
    public bool IsValid(RenderableHandle h) =>
        h.Index < _slotGeneration.Count
        && _slotGeneration[(int)h.Index] == h.Generation
        && _slotEntity[(int)h.Index] != 0;

    /// <summary>Resolves a handle back to its entity, or null if the handle is stale.</summary>
    public Entity* Resolve(RenderableHandle h) =>
        IsValid(h) ? (Entity*)_slotEntity[(int)h.Index] : null;

    /// <summary>Resolves a bare slot index (as carried in ShadowEntityInfo.EntityIndex /
    /// returned by the pick pass) back to its entity, or null if the slot is free /
    /// out of range. Generation-less by design: the GPU only stores the slot index,
    /// and the pick read-back is synchronous, so no slot can be recycled in between.</summary>
    public Entity* ResolveSlot(uint index) =>
        index < (uint)_slotEntity.Count && _slotEntity[(int)index] != 0
            ? (Entity*)_slotEntity[(int)index] : null;

    /// <summary>Looks up an already-registered entity's handle.</summary>
    public bool TryGetHandle(Entity* e, out RenderableHandle h) =>
        _handleOf.TryGetValue((nint)e, out h);

    /// <summary>
    /// Reconciles the handle map against the scene's current renderable set:
    /// registers entities that have a live mesh, frees handles whose entity left the
    /// scene (or lost its mesh). The registration predicate matches RebuildTlas's
    /// gather, so the handle population tracks the TLAS instance population exactly.
    /// Identity is independent of active/visibility state, so a visibility toggle
    /// doesn't churn generations. Called on the TLAS-rebuild cadence (dirty-driven),
    /// not per frame.
    /// </summary>
    public void SyncRenderables(Scene scene)
    {
        // 1. Register current renderables (entities with a live mesh).
        _present.Clear();
        for (int i = 0; i < scene.EntityCount; i++)
        {
            Entity* e = scene.GetEntity(i);
            if (e == null) continue;
            var mesh = e->GetComponent<MeshComponent>();
            if (mesh == null || mesh.mesh == null) continue;
            _present.Add((nint)e);
            Register(e);
        }

        // 2. Free handles whose entity is no longer a present renderable.
        _toFree.Clear();
        foreach (var key in _handleOf.Keys)
            if (!_present.Contains(key)) _toFree.Add(key);
        foreach (var key in _toFree) Free((Entity*)key);
    }

    // ── Cached world transforms (L2 step 5) ──────────────────────────────────
    // TransformComponent.GetWorldMatrix() walks the parent chain on every call (no
    // cross-frame cache), and lights / cull / TLAS each call it independently. This
    // memoizes the composed world matrix per entity for the duration of one extract
    // cycle, so the chain is walked at most once per entity per frame instead of
    // once per consumer. Reset at the top of each extract cycle (the per-frame draw
    // path + RebuildTlas) via BeginTransforms; computed lazily on first request.
    private readonly Dictionary<nint, Matrix4x4> _worldCache = new();

    /// <summary>Clears the per-cycle world-transform cache. MUST be called once at the
    /// start of each extract cycle (before any <see cref="WorldOf"/> read) — otherwise
    /// consumers would see last cycle's stale matrices.</summary>
    public void BeginTransforms() => _worldCache.Clear();

    /// <summary>Composed world matrix for <paramref name="e"/>, memoized for the current
    /// extract cycle (<see cref="BeginTransforms"/>). Identity if the entity has no
    /// TransformComponent. Behaviourally identical to <c>*GetWorldMatrix()</c> — just
    /// computed once per entity per cycle instead of once per consumer.</summary>
    public Matrix4x4 WorldOf(Entity* e)
    {
        nint key = (nint)e;
        if (_worldCache.TryGetValue(key, out var cached)) return cached;
        var t = e->GetComponent<TransformComponent>();
        Matrix4x4 world = t != null ? *t.GetWorldMatrix() : Matrix4x4.Identity;
        _worldCache[key] = world;
        return world;
    }

    /// <summary>Light SSBO for the given frame slot. Stable for the renderer's lifetime.</summary>
    public Buffer GetLightStorageBuffer(uint frame) => _lightBuffers[frame].buffer;

    /// <summary>Allocates the per-frame light SSBOs. Call once at init, before any
    /// pipeline binds them.</summary>
    public void CreateLightBuffers()
    {
        for (int i = 0; i < RenderConfig.MAX_CONCURRENT_FRAMES; i++)
        {
            _gfx.CreateMappedStorageBuffer(
                RenderConfig.MAX_LIGHTS * (uint)sizeof(PbrLightGpu),
                ref _lightBuffers[i], preferDeviceLocal: true);
        }
    }

    /// <summary>Cull-input SSBO for the given frame slot — bound by the cull pass
    /// at binding 0. Stable for the renderer's lifetime.</summary>
    public Buffer GetRenderablesBuffer(uint frame) => _renderableBuffers[frame].buffer;

    /// <summary>BLEND-mode renderables extracted this frame, unsorted. The cull pass
    /// applies the view-dependent back-to-front sort and hands the result to the
    /// transparent pass.</summary>
    public IReadOnlyList<TransparentDraw> TransparentCandidates => _transparentCandidates;

    /// <summary>Allocates the per-frame cull-input SSBOs. Call once at init, before
    /// the cull pipeline binds them.</summary>
    public void CreateRenderableBuffers()
    {
        for (int i = 0; i < RenderConfig.MAX_CONCURRENT_FRAMES; i++)
        {
            _gfx.CreateMappedStorageBuffer(
                RenderConfig.MAX_INSTANCES * (uint)sizeof(RenderableInputGpu),
                ref _renderableBuffers[i]);
        }
    }

    /// <summary>
    /// Walks the scene once, packing OPAQUE/MASK renderables into the cull-input
    /// SSBO and siphoning BLEND renderables into <see cref="TransparentCandidates"/>
    /// (unsorted — the back-to-front sort is view-dependent, so it stays in the cull
    /// pass which owns the camera). Returns the opaque count, also cached in
    /// <see cref="ExtractedRenderableCount"/>. This is the extraction half of the
    /// former DrawCullPipeline.Record walk; GPU frustum cull reads this buffer.
    /// </summary>
    public uint ExtractRenderables(uint frameIndex, Scene scene)
    {
        // Dirty-driven: skip the walk if this slot is already current.
        // The cull pass still re-sorts transparents by camera every frame - only the
        // (view-independent) packing is gated here, and the buffer keeps last extract's
        // rows for this slot, which stay valid until the next MarkSceneDirty.
        if (!TakeSlot(ref _renderableDirty, frameIndex)) return ExtractedRenderableCount;

        RenderableInputGpu* inputPtr = (RenderableInputGpu*)_renderableBuffers[frameIndex].mapped;
        uint count = 0;
        int matCount = scene.MaterialCount;
        int fallbackMatIdx = matCount; // matches the fallback slot UpdateMaterials writes

        _transparentCandidates.Clear();

        for (int i = 0; i < scene.EntityCount; i++)
        {
            Entity* e = scene.GetEntity(i);
            if (e == null) continue;
            var meshComp = e->GetComponent<MeshComponent>();
            if (meshComp == null || meshComp.mesh == null) continue;
            var transform = e->GetComponent<TransformComponent>();
            if (transform == null) continue;

            int  matIdx = meshComp.materialIndex >= 0 ? meshComp.materialIndex : fallbackMatIdx;
            Mesh m      = *meshComp.mesh;
            var  world  = WorldOf(e); // cached (L2 step 5); transform != null checked above

            // Fallback material is implicitly opaque. Real materials carry their
            // mode in the Flags bitfield.
            AlphaMode mode = AlphaMode.Opaque;
            if (matIdx >= 0 && matIdx < matCount)
                mode = scene.Materials[matIdx].GetAlphaMode();

            if (mode == AlphaMode.Blend)
            {
                // ViewDepth left 0 — the cull pass fills + sorts it (camera-dependent).
                _transparentCandidates.Add(new TransparentDraw
                {
                    Model         = world,
                    MaterialIndex = (uint)matIdx,
                    IndexCount    = (uint)m.count,
                    FirstIndex    = (uint)m.offset,
                    ViewDepth     = 0f,
                });
                continue;
            }

            if (count >= RenderConfig.MAX_INSTANCES)
                throw new InvalidOperationException($"Renderable count exceeds MAX_INSTANCES ({RenderConfig.MAX_INSTANCES}).");

            inputPtr[count] = new RenderableInputGpu
            {
                model         = world,
                sphereLocal   = m.sphereLocal,
                indexCount    = (uint)m.count,
                firstIndex    = (uint)m.offset,
                materialIndex = (uint)matIdx,
            };
            count++;
        }

        ExtractedRenderableCount = count;
        return count;
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
        // Dirty-driven (L2 step 7): this slot's material SSBO is already current.
        if (!TakeSlot(ref _materialDirty, frameIndex)) return _lastMaterialCount;

        int matCount = scene.MaterialCount;
        if (matCount + 1 > (int)RenderConfig.MAX_MATERIALS)
            throw new InvalidOperationException(
                $"Scene material count ({matCount}) exceeds MAX_MATERIALS ({RenderConfig.MAX_MATERIALS}).");

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

        _lastMaterialCount = (uint)matCount;
        return _lastMaterialCount;
    }

    /// <summary>Walks scene lights into the per-frame light SSBO. Returns the
    /// packed light count, also cached in <see cref="LightCount"/>. Every
    /// rendering path (deferred, forward+, pathtracer) calls this once per frame
    /// before recording its draws.</summary>
    public uint UpdateLights(uint frameIndex, Scene scene)
    {
        // Dirty-driven (L2 step 7): this slot's light SSBO is already current.
        if (!TakeSlot(ref _lightDirty, frameIndex)) return LightCount;

        scene.EnumerateLights(_lightScratch);

        PbrLightGpu* lightPtr = (PbrLightGpu*)_lightBuffers[frameIndex].mapped;
        uint count = 0;
        foreach (var light in _lightScratch)
        {
            if (count >= RenderConfig.MAX_LIGHTS) break;

            // World-space position from the owner transform if present. Cached
            // (L2 step 5); WorldOf returns identity for an owner without a
            // TransformComponent — same result as the old t == null path.
            Vector3 worldPos = Vector3.Zero;
            if (light.Owner != null)
            {
                var w = WorldOf(light.Owner);
                worldPos = new Vector3(w.M41, w.M42, w.M43);
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
        for (int i = 0; i < _renderableBuffers.Length; i++)
        {
            if (_renderableBuffers[i].buffer.Handle != 0)
                _gfx.DestroyBuffer(_renderableBuffers[i].buffer, _renderableBuffers[i].alloc);
        }
    }
}