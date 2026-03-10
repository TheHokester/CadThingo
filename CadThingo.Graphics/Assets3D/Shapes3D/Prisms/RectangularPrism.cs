using System.Numerics;
using CadThingo.Graphics.Assets3D.Shapes3D.Prisms;

namespace CadThingo.Graphics.Assets3D.shapes3D.Prisms;

public sealed class RectangularPrism : Prism3D

{
    public float Width { get;}
    public float Height { get; }
    
    public override string ObjectType => "RectangularPrism";

    public RectangularPrism(float length, float width, float height, Vector3 position) :
        base(CreateProfile(width, height), length, position) 
    {
        if (width <= 0)
            throw new ArgumentOutOfRangeException(nameof(width), "Width must be greater than 0");
        if (height <= 0)
            throw new ArgumentOutOfRangeException(nameof(height), "Height must be greater than 0");
        
        Width = width;
        Height = height;
    }

    private static IReadOnlyList<Vector2> CreateProfile(float width, float height)
    {
        if(width <= 0f) 
            throw new ArgumentOutOfRangeException(nameof(width), "Width must be greater than 0");
        if (height <= 0)
            throw new ArgumentOutOfRangeException(nameof(height), "Height must be greater than 0");
        var halfWidth = width * 0.5f;
        var halfHeight = height * 0.5f;

        return
        [
            new Vector2(-halfWidth, -halfHeight),
            new Vector2(halfWidth, -halfHeight),
            new Vector2(halfWidth, halfHeight),
            new Vector2(-halfWidth, halfHeight)
        ];
    }
        
    
}