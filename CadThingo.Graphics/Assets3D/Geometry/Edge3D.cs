namespace CadThingo.Graphics.Assets3D.Geometry;

public struct Edge3D
{
    public Vertex A, B;

    public Edge3D(Vertex a, Vertex b)
    {
        A = a;
        B = b;
    }
}