using System.Drawing;
using System.Numerics;

namespace CadThingo.Graphics.Assets3D.Geometry;

public struct RenderTriangle
{
    public Vertex A, B, C;
    public Color color;
    
    public RenderTriangle(Vertex a, Vertex b, Vertex c, Color color)
    {
        A = a;
        B = b;
        C = c;
        this.color = color;
    }
}
public struct Vertex
{
    public Vector3 Position;  
    public Vertex(Vector3 point) => Position = point;
    // camera space or world space
}