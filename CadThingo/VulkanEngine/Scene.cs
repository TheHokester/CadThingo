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

    // Bindless materials live in a CPU mirror keyed by the int index stored on MeshComponent.
    // The geometry pipeline uploads this list into a per-frame StructuredBuffer<PbrMaterial>
    // before recording the geometry pass. Index -1 means "use the default material at slot 0".
    private readonly List<PbrMaterial> _materials = new();

    public int EntityCount   => _entityList.Count;
    public int MaterialCount => _materials.Count;
    public IReadOnlyList<PbrMaterial> Materials => _materials;

    public Scene(Vk vk, Device device, PhysicalDevice physicalDevice)
    {
        renderGraph = new RenderGraph(vk, device, physicalDevice);
        Cam = new Camera();
    }

    public unsafe void AddEntity(Entity* entity) => _entityList.Add((nint)entity);

    /// <summary>
    /// Parents <paramref name="child"/> under <paramref name="parent"/> and registers it.
    /// Sets the child's TransformComponent.Parent so GetWorldMatrix() walks up the chain.
    /// </summary>
    public unsafe void AddChild(Entity* parent, Entity* child)
    {
        if (child == null) return;
        var t = child->GetComponent<TransformComponent>();
        if (t != null) t.Parent = parent;
        AddEntity(child);
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