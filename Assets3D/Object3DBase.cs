using CadThingo;
using System.Numerics;
using CadThingo.Assets3D;

namespace Cadthingo.Assets3D;

public abstract class Object3DBase : IObject3D
{
    private readonly List<IObject3D> _children = new();
    
    public Guid Id { get; } = Guid.NewGuid();
    public abstract string ObjectType { get; }
    
    public string? Name { get; set; }
    public string? Description { get; set; }
    
    public bool IsVisible { get; set; } = true;
    public bool IsSelectable { get; set; } = true;
    public bool IsLocked { get; set; } = false;
    
    public Vector3 Position { get; set; }
    public Quaternion Rotation { get; set; } = Quaternion.Identity;
    public Vector3 Scale { get; set; } = Vector3.One;
    public Vector3 Pivot { get; set; } = Vector3.Zero; 
    
    public IObject3D? Parent { get; private set; }
    public IReadOnlyList<IObject3D> Children => _children;

    public Matrix4x4 LocalMatrix
    {
        get
        {
            var moveToPivot = Matrix4x4.CreateTranslation(-Pivot);
            var scale = Matrix4x4.CreateScale(Scale);
            var rotate = Matrix4x4.CreateFromQuaternion(Rotation);
            var moveBack = Matrix4x4.CreateTranslation(Pivot);
            var translate = Matrix4x4.CreateTranslation(Position);
            return moveToPivot * scale * rotate * moveBack * translate;
        }
    }

    public Matrix4x4 WorldMatrix
    {
        get
        {
          if(Parent is null)
              return LocalMatrix;
          return LocalMatrix * Parent.WorldMatrix;
        }
    }

    public virtual void AddChild(IObject3D child)
    {
        ArgumentNullException.ThrowIfNull(child);
        
        if(ReferenceEquals(child, this))
            throw new ArgumentException("Cannot add self as child");
        
        if (_children.Contains(child))
        {
            throw new ArgumentException("Child already added");
           
        }

        if (child is Object3DBase obj && obj.Parent is not null)
        {
            obj.Parent.RemoveChild(obj);
        }
        if(child is Object3DBase attachable) 
            attachable.Parent = this;
        else
        {
            throw new ArgumentException("Child must be of type Object3DBase");
        }
        _children.Add(child);
        
    }

    public virtual bool RemoveChild(IObject3D child)
    {
        if (!_children.Remove(child))
            return false;
        if(child is Object3DBase childBase)
            childBase.Parent = null;
        return true;
    }
    public virtual Bounds3D GetLocalBounds() => Bounds3D.Empty;

    public virtual Bounds3D GetWorldBounds()
    {
        return GetLocalBounds().Transform(WorldMatrix);
    }
}


