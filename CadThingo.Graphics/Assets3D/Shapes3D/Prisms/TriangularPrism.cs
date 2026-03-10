using System.Numerics;
using CadThingo.Graphics.Assets3D.Shapes3D.Prisms;

namespace CadThingo.Graphics.Assets3D.Shapes3D.Prisms;

public sealed class TriangularPrism : Prism3D
{
    public Vector2 A { get;  }
    public Vector2 B { get;  }
    public Vector2 C { get;  }
    
    public override string ObjectType => "TriangularPrism";
    
    public TriangularPrism(Vector2 a, Vector2 b, Vector2 c, float length, Vector3 position)
        : base([a, b, c], length, position)
    {
        A = a ;
        B = b;
        C = c;
    }
    public enum TriangleType{
        Equilateral,
        Scalene,
        LRightAngled,
        RRightAngled,
    }

    // public TriangularPrism(float length, TriangleType type, float triBase, float triHeight) : base(length)
    // {
    //     //TODO: Implement later
    // }
     
}