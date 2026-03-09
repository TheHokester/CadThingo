using System.Numerics;
using Cadthingo.Assets3D.Shapes3D.Prisms;

namespace Cadthingo.Assets3D.Shapes3D.Prisms;

public sealed class RegularPolygonPrism : Prism3D
{
    public int SideCount { get; }
    public float Radius { get; }
    public override string ObjectType => "RegularPolyGonPrism";
    
    public RegularPolygonPrism(float length,float radius, int sideCount, Vector3 position) 
        : base(CreateProfile(sideCount, radius), length, position )
    {
        if (sideCount < 3)
            throw new ArgumentOutOfRangeException(nameof(sideCount), "Side count must be greater than 2");
        if (radius <= 0)
            throw new ArgumentOutOfRangeException(nameof(radius), "Radius must be greater than 0");
        
        SideCount = sideCount;
        Radius = radius;
    }

    private static IReadOnlyList<Vector2> CreateProfile(int sideCount, float radius)
    {
        if (sideCount < 3)
            throw new ArgumentOutOfRangeException(nameof(sideCount), "Side count must be greater than 2");
        if (radius <= 0)
            throw new ArgumentOutOfRangeException(nameof(radius), "Radius must be greater than 0");

        var points = new Vector2[sideCount];
        float step = MathF.Tau / sideCount;
        float startAngle = -MathF.PI * 0.5f;

        for (var i = 0; i < sideCount; i++)
        {
            float angle = startAngle + i * step;
            points[i] = new Vector2(
                radius * MathF.Cos(angle),
                radius * MathF.Sin(angle));
        }
        return points;
    }

    
}