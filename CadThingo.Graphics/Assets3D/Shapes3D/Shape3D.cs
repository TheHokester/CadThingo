using CadThingo.Graphics.Assets3D;
using CadThingo.Graphics.Assets3D.Geometry;
using CadThingo.Graphics.Rendering;

namespace CadThingo.Graphics.Assets3D.Shapes3D;

public abstract class Shape3D : Object3DBase, IRenderable3D
{
    public Material3D Material { get; set; } = new();
    
    public abstract IEnumerable<RenderTriangle> GetTriangles();
    public abstract IEnumerable<Edge3D> GetEdges();
}