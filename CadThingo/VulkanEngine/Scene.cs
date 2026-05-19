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
}

public class Scene
{
    private static Camera Cam;

    public RenderGraph renderGraph;

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

    public Scene(Vk vk, Device device, PhysicalDevice physicalDevice)
    {
        renderGraph = new RenderGraph(vk, device, physicalDevice);
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
        _materials.Add(mat);
        return _materials.Count - 1;
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