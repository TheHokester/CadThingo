using System.Numerics;
using Cadthingo.Assets3D;

namespace CadThingo.Assets3D;

public sealed class Camera : Object3DBase
{
    public override string ObjectType => "Camera";
    public Vector3 Position { get; set; }
    public Vector3 Target = Vector3.Zero;
    
    public float FOV = MathF.PI / 3f;
    public float near = 0.1f;
    public float far = 100f;
    
    public Matrix4x4 ViewMatrix => Matrix4x4.CreateLookAt(Position, Target, Vector3.UnitY);
    public Matrix4x4 ProjectionMatrix(float aspect) => 
        Matrix4x4.CreatePerspectiveFieldOfView(FOV, aspect, near, far);
}