using Cadthingo.Assets3D.Geometry;
using CadThingo.Rendering;

namespace Cadthingo.Assets3D;

public interface IRenderable3D
{
    IEnumerable<RenderTriangle> GetTriangles(); 
    IEnumerable<Edge3D> GetEdges();
}