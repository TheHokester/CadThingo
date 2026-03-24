using System.Numerics;
namespace CadThingo.Graphics.Assets3D;

public interface IObject3D : ITransform3D, IIdentity
{
    
    
    bool IsVisible { get; set; }
    bool IsSelectable { get; set; }
    bool IsLocked { get; set; }
    IObject3D? Parent { get; }
    IReadOnlyList<IObject3D> Children { get; }
    void AddChild(IObject3D child);
    bool RemoveChild(IObject3D child);
   
   Bounds3D GetLocalBounds();
   Bounds3D GetWorldBounds();
}