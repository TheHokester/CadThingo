using System.Numerics;
using System.Runtime.InteropServices;
using CadThingo.VulkanEngine.Renderer;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine;

/// <summary>
/// Mesh allocated in unmanaged memory so MeshComponent can hold a raw Mesh*.
/// <para><c>sphereLocal</c> is the bounding sphere in mesh-local space:
/// xyz = center, w = radius. Computed in <see cref="ResourceManager.UploadMesh"/>
/// from vertex positions so the GPU cull pass can frustum-test each renderable
/// without re-walking vertex data.</para>
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct Mesh
{
    public int offset;
    public int count;
    public Vector4 sphereLocal;
    // VB-side range — only consumed by ResourceManager.FreeMesh so the global
    // VB free-list can reclaim the vertex slot when the owning MeshResource
    // unloads. Renderer/shader code addresses vertices through the index buffer,
    // so these fields don't need to be GPU-visible.
    public int vertexOffset;
    public int vertexCount;
}

public class Scene
{
    private Camera Cam;

    // Entity* stored as nint so the managed List doesn't need pinning.
    private readonly List<nint> _entityList = new();

    // Persistent hierarchy mirror of TransformComponent.Parent edges. Maintained
    // incrementally on AddEntity / AddChild / Reparent so the editor outliner
    // doesn't have to rebuild it every frame. Roots are entities whose
    // TransformComponent.Parent is null (or who have no TransformComponent).
    private readonly List<nint>                   _roots      = new();
    private readonly Dictionary<nint, List<nint>> _childrenOf = new();

    // Read-only view of the entity list for editor/UI iteration. Each nint is an
    // Entity* — callers cast back to Entity* inside an `unsafe` block.
    public IReadOnlyList<nint> Entities => _entityList;

    /// <summary>Entities with no parent (top-level scene nodes).</summary>
    public IReadOnlyList<nint> Roots => _roots;

    /// <summary>
    /// Returns the children of <paramref name="parent"/> as registered through
    /// AddChild / Reparent. Empty list if no children — never null.
    /// </summary>
    public unsafe IReadOnlyList<nint> GetChildren(Entity* parent)
        => _childrenOf.TryGetValue((nint)parent, out var list) ? list : Array.Empty<nint>();

    // Bindless materials live in a CPU mirror keyed by the int index stored on MeshComponent.
    // The geometry pipeline uploads this list into a per-frame StructuredBuffer<PbrMaterial>
    // before recording the geometry pass. Index -1 means "use the default material at slot 0".
    private readonly List<PbrMaterial> _materials = new();

    // Indices freed via ReleaseMaterial. AddMaterial pops from here before
    // extending the list, so a destroy/reload cycle reuses slots instead of
    // burning through the MAX_MATERIALS budget. Freed slots are zeroed in the
    // backing list so the per-frame upload of [0, MaterialCount) doesn't keep
    // shipping stale bindless texture indices.
    private readonly Stack<int> _materialFree = new();

    public int EntityCount   => _entityList.Count;
    public int MaterialCount => _materials.Count;
    public IReadOnlyList<PbrMaterial> Materials => _materials;

    /// <summary>
    /// In-place view of the material list. PbrMaterial is a struct, so editors
    /// need a `ref` into the backing array to mutate factors/flags without a
    /// read–copy–write round-trip. The renderer's per-frame upload copies from
    /// this list straight into the material SSBO, so writes take effect on the
    /// next frame.
    /// </summary>
    public Span<PbrMaterial> MaterialsMutable => CollectionsMarshal.AsSpan(_materials);

    // vk/device/physicalDevice are retained on the signature for call-site stability; the
    // deferred chain now lives on the renderer's FrameGraph, not a Scene-owned RenderGraph.
    public Scene()
    {
        Cam = new Camera();
    }

    /// <summary>
    /// Registers an entity as a root (no parent). If the entity already has a
    /// TransformComponent.Parent set, the existing parent edge is honoured —
    /// callers that want to nest should use AddChild instead.
    /// </summary>
    public unsafe void AddEntity(Entity* entity)
    {
        if (entity == null) return;
        _entityList.Add((nint)entity);

        var t = entity->GetComponent<TransformComponent>();
        if (t != null && t.Parent != null)
            RegisterAsChildOf(t.Parent, entity);
        else
            _roots.Add((nint)entity);
    }

    /// <summary>
    /// Bulk variant of <see cref="AddEntity"/> for file imports (e.g. an entire
    /// glTF scene loaded at once). Caller is responsible for having set up
    /// TransformComponent.Parent pointers between members of the batch before
    /// the call — the hierarchy mirror is built from those pointers.
    /// </summary>
    public unsafe void AddEntities(ReadOnlySpan<nint> entities)
    {
        foreach (var ptr in entities)
            AddEntity((Entity*)ptr);
    }

    /// <summary>
    /// Parents <paramref name="child"/> under <paramref name="parent"/> and registers it.
    /// Sets the child's TransformComponent.Parent so GetWorldMatrix() walks up the chain.
    /// </summary>
    public unsafe void AddChild(Entity* parent, Entity* child)
    {
        if (child == null) return;
        var t = child->GetComponent<TransformComponent>();
        if (t != null) t.Parent = parent;
        _entityList.Add((nint)child);
        if (parent != null) RegisterAsChildOf(parent, child);
        else                _roots.Add((nint)child);
    }

    /// <summary>
    /// Moves <paramref name="child"/> under a new parent (or null for root).
    /// Updates both the TransformComponent.Parent pointer and the cached
    /// hierarchy mirror. Cheap: O(siblings) in the previous parent group.
    /// </summary>
    public unsafe void Reparent(Entity* child, Entity* newParent)
    {
        if (child == null) return;
        var t = child->GetComponent<TransformComponent>();
        Entity* oldParent = t != null ? t.Parent : null;
        if (oldParent == newParent) return;

        if (oldParent != null && _childrenOf.TryGetValue((nint)oldParent, out var oldList))
            oldList.Remove((nint)child);
        else
            _roots.Remove((nint)child);

        if (t != null) t.Parent = newParent;

        if (newParent != null) RegisterAsChildOf(newParent, child);
        else                   _roots.Add((nint)child);
    }

    private unsafe void RegisterAsChildOf(Entity* parent, Entity* child)
    {
        var key = (nint)parent;
        if (!_childrenOf.TryGetValue(key, out var list))
        {
            list = new List<nint>();
            _childrenOf[key] = list;
        }
        list.Add((nint)child);
    }

    public unsafe Entity* GetEntity(int index) => (Entity*)_entityList[index];

    /// <summary>
    /// Position of <paramref name="entity"/> in the flat entity list, or -1 if
    /// absent. NOTE: this is a plain list-position lookup and is NO LONGER the
    /// GPU-side identity — pick / selection / TLAS now resolve through
    /// <see cref="GpuScene"/>'s stable RenderableHandle (L2 step 6), which survives
    /// list reorder/removal. Use a handle, not this index, for anything GPU-facing.
    /// </summary>
    public unsafe int IndexOf(Entity* entity) =>
        entity == null ? -1 : _entityList.IndexOf((nint)entity);

    /// <summary>
    /// Removes a single entity from the scene's flat list and hierarchy mirror.
    /// TransformComponent.Parent is left intact so a later AddEntity rebuilds
    /// the same parent-child edge. The entity's native memory is NOT freed —
    /// callers that want a destroy should Entity.Destroy() after removal.
    /// </summary>
    public unsafe void RemoveEntity(Entity* entity)
    {
        if (entity == null) return;
        var ptr = (nint)entity;
        _entityList.Remove(ptr);
        _roots.Remove(ptr);

        var t = entity->GetComponent<TransformComponent>();
        if (t != null && t.Parent != null
            && _childrenOf.TryGetValue((nint)t.Parent, out var siblings))
        {
            siblings.Remove(ptr);
            if (siblings.Count == 0)
                _childrenOf.Remove((nint)t.Parent);
        }

        // Drop our own children mirror entry — descendants are expected to be
        // removed by the caller (RemoveSubtree handles this) but the bookkeeping
        // entry would otherwise leak if a child was destroyed first.
        _childrenOf.Remove(ptr);
    }

    /// <summary>
    /// Collects an entity and every descendant in pre-order, then RemoveEntity's
    /// each. Returns the ordered list so callers can re-add them later via
    /// <see cref="AddEntity"/> in the same order (the AddEntity path rebuilds
    /// parent edges from each entity's TransformComponent.Parent, which we
    /// deliberately leave intact).
    /// </summary>
    public unsafe List<nint> RemoveSubtree(Entity* root)
    {
        var collected = new List<nint>();
        if (root == null) return collected;

        Collect(root, collected);
        foreach (var ptr in collected)
            RemoveEntity((Entity*)ptr);
        return collected;

        void Collect(Entity* e, List<nint> output)
        {
            output.Add((nint)e);
            if (!_childrenOf.TryGetValue((nint)e, out var children)) return;
            // Snapshot — Collect's grandchildren-recursion mutates nothing,
            // but RemoveEntity below will, so we copy the list defensively.
            var snapshot = children.ToArray();
            foreach (var c in snapshot)
                Collect((Entity*)c, output);
        }
    }

    /// <summary>
    /// Appends every enabled <see cref="LightComponent"/> attached to an active entity
    /// to <paramref name="output"/>. Used by <c>UpdateLightingBuffers</c> to pack the
    /// per-frame StructuredBuffer&lt;PbrLight&gt;.
    ///
    /// Returns a list (not an iterator) because the body dereferences raw Entity*
    /// pointers — C# disallows the <c>unsafe</c> modifier on yield-return iterators.
    /// </summary>
    public unsafe void EnumerateLights(List<LightComponent> output)
    {
        output.Clear();
        foreach (var ptr in _entityList)
        {
            var e = (Entity*)ptr;
            if (e == null || !e->IsActive) continue;
            var light = e->GetComponent<LightComponent>();
            if (light == null || !light.Enabled) continue;
            output.Add(light);
        }
    }

    /// <summary>
    /// Appends every <see cref="ReflectionProbeComponent"/> attached to an active
    /// entity (alongside the entity pointer cast to nint — C# rejects Entity* as
    /// a generic type argument). Callers re-cast back: <c>(Entity*)tuple.entity</c>.
    /// </summary>
    public unsafe void EnumerateProbes(List<(nint entity, ReflectionProbeComponent probe)> output)
    {
        output.Clear();
        foreach (var ptr in _entityList)
        {
            var e = (Entity*)ptr;
            if (e == null || !e->IsActive) continue;
            var probe = e->GetComponent<ReflectionProbeComponent>();
            if (probe == null) continue;
            output.Add(((nint)e, probe));
        }
    }

    /// <summary>
    /// Registers a bindless material on the scene. Returns the int index stored on
    /// MeshComponent.materialIndex so the geometry pass can resolve the material in
    /// the per-frame material SSBO.
    /// </summary>
    public int AddMaterial(PbrMaterial mat)
    {
        if (_materialFree.Count > 0)
        {
            int idx = _materialFree.Pop();
            _materials[idx] = mat;
            return idx;
        }
        _materials.Add(mat);
        return _materials.Count - 1;
    }

    /// <summary>
    /// Returns the material slot at <paramref name="idx"/> to the free pool so a
    /// later <see cref="AddMaterial"/> can reuse it. Backing list entry is zeroed
    /// so the per-frame upload doesn't keep shipping stale bindless texture
    /// indices that might have been re-bound to a different texture in the
    /// meantime. No-op for out-of-range / already-free indices.
    /// </summary>
    public void ReleaseMaterial(int idx)
    {
        if (idx < 0 || idx >= _materials.Count) return;
        _materials[idx] = default;
        _materialFree.Push(idx);
    }

    /// <summary>
    /// Returns the PbrMaterial for <paramref name="idx"/>, or default if out of range.
    /// </summary>
    public PbrMaterial GetMaterial(int idx)
        => idx >= 0 && idx < _materials.Count ? _materials[idx] : default;

    public bool RayCast(Ray ray, ref RayCastHit hit, float rayLength)
    {
        return false;
    }
}

public struct Ray
{
    public Vector3 origin;
    public Vector3 direction;
}

public struct RayCastHit
{
    public Vector3 point;
}