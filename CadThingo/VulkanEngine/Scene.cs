using System.Numerics;
using CadThingo.VulkanEngine.Renderer;

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

    private RenderGraph renderGraph;

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