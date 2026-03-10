using CadThingo.Graphics.Assets3D.Geometry;
using CadThingo.Graphics.Rendering;

namespace CadThingo.Graphics.Assets3D;

public interface IRenderable3D
{
    IEnumerable<RenderTriangle> GetTriangles(); 
    IEnumerable<Edge3D> GetEdges();
}