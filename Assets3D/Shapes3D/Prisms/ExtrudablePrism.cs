using System.Numerics;
using Cadthingo.Assets3D.Shapes3D.Prisms;

namespace CadThingo.Assets3D.Shapes3D.Prisms;

public sealed class ExtrudablePrism : Prism3D
{
    public override string ObjectType => "ExtrudablePrism";

    public ExtrudablePrism(IReadOnlyList<Vector2> profile,float length, Vector3 position) 
        : base(profile, length, position)
    {
        //TODO: Implement later 
    }

    
}