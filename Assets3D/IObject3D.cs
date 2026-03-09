using System.Numerics;
namespace CadThingo.Assets3D;

public interface IObject3D
{
    Guid Id { get; }
    string? Name { get; }
    string? Description { get; }
    
    bool IsVisible { get; set; }
    bool IsSelectable { get; set; }
    bool IsLocked { get; set; }
    
    Vector3 Position { get; set; }
    Quaternion Rotation { get; set; }
    Vector3 Scale { get; set; }
    Vector3 Pivot { get; set; }
    
    IObject3D? Parent { get; }
    IReadOnlyList<IObject3D> Children { get; }

   Matrix4x4 LocalMatrix { get; }
   Matrix4x4 WorldMatrix { get; }

   void AddChild(IObject3D child);
   bool RemoveChild(IObject3D child);
   
   Bounds3D GetLocalBounds();
   Bounds3D GetWorldBounds();
}