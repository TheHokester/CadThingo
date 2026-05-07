namespace CadThingo.VulkanEngine.GLTF;

/// <summary>
/// MeshResource subclass that uploads pre-extracted vertex/index arrays from a glTF primitive.
/// The actual SharpGLTF accessor reads happen in GltfLoader; this class just feeds the
/// arrays into the global VB/IB via the existing MeshResource.Load pipeline.
/// </summary>
public unsafe class GltfMeshResource : MeshResource
{
    private Vertex[]? _vertices;
    private uint[]?   _indices;

    public GltfMeshResource(string id, ResourceManager manager, Vertex[] vertices, uint[] indices)
        : base(id, manager)
    {
        _vertices = vertices;
        _indices  = indices;
    }

    protected override bool LoadMeshData(out Vertex[] vertices, out uint[] indices)
    {
        if (_vertices == null || _indices == null)
        {
            vertices = Array.Empty<Vertex>();
            indices  = Array.Empty<uint>();
            return false;
        }
        vertices = _vertices;
        indices  = _indices;
        // Drop the managed refs so the GC can reclaim them after upload completes.
        _vertices = null;
        _indices  = null;
        return true;
    }
}