using CadThingo.Assets3D;

namespace Cadthingo.Assets3D;

public class Container3D : Object3DBase
{
    public override string ObjectType => "Container";

    public override Bounds3D GetLocalBounds()
    {
        if(Children.Count == 0)
            return Bounds3D.Empty;

        Bounds3D? combined = null;
        foreach (var child in Children)
        {
            var childBounds = child.GetWorldBounds();

            combined = combined is null
                ? childBounds
                : Bounds3D.Union(combined.Value, childBounds);
        }
        
        return combined ?? Bounds3D.Empty;
        
    }
}
