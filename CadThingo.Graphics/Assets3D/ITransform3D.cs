

using System.Numerics;

namespace CadThingo.Graphics.Assets3D;

public interface ITransform3D
{
    Vector3 Position { get; set; }
    Quaternion Rotation { get; set; }
    Vector3 Scale { get; set; }
    Vector3 Pivot { get; set; } 
    Matrix4x4 LocalMatrix { get; }
    Matrix4x4 WorldMatrix { get; }
}