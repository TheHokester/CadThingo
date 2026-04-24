using System.Numerics;
using CadThingo.VulkanEngine.Renderer;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine;

/// <summary>
/// Mesh allocated in unmanaged memory so MeshComponent can hold a raw Mesh*.
/// </summary>
public unsafe struct Mesh
{
    public int offset;
    public int count;
}
  
/// <summary>
/// Material allocated in unmanaged memory.
/// SetUniform takes a Matrix4x4* to avoid a 64-byte copy on every draw call.
/// </summary>
public unsafe struct Material
{
    public Vector4 baseColorFactor;
    public float metallicFactor;
    public float roughnessFactor;
    public int baseColorTextureSet;
    public int physicalDescriptorTextureSet;
    public int normalTextureSet;
    public int occlusionTextureSet;
    public int emissiveTextureSet;
    public float alphaMask;
    public float alphaMaskCutoff;


    public void Bind()
    {
    }

    public void SetUniform(string name, Matrix4x4* value)
    {
    }

    public void SetUniform(string name, float value)
    {
    }
}

public class Scene
{

    private Entity[] _entities;
    private List<ResourceHandle<MaterialResource>> _materials;
    private List<ResourceHandle<TextureResource>> _textures;

    private static Camera Cam;

    public RenderGraph renderGraph;

    // Entity* stored as nint so the managed List doesn't need pinning.
    private readonly List<nint> _entityList = new();

    public int EntityCount => _entityList.Count;

    public Scene(Vk vk, Device device, PhysicalDevice physicalDevice)
    {
        renderGraph = new RenderGraph(vk, device, physicalDevice);
        Cam = new Camera();
    }

    public unsafe void AddEntity(Entity* entity) => _entityList.Add((nint)entity);

    public unsafe Entity* GetEntity(int index) => (Entity*)_entityList[index];

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