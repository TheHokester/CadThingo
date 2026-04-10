using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace CadThingo.VulkanEngine;

/// <summary>
/// Stores position, rotation and scale; caches the model matrix.
/// </summary>
public sealed unsafe class TransformComponent : Component
{
    private Vector3 _position = Vector3.Zero;
    private Quaternion _rotation = Quaternion.Identity;
    private Vector3 _scale = Vector3.One;
    
    private readonly Matrix4x4* _Matrix;
    
    private bool _dirty = true;



    public TransformComponent()
    {
        _Matrix = (Matrix4x4*)NativeMemory.Alloc((nuint)sizeof(Matrix4x4));
        *_Matrix = Matrix4x4.Identity;
    }

    protected override void OnDestroy()
    {
        if (_Matrix != null)
        {
            NativeMemory.Free(_Matrix);
        }
    }
    
    // ---- Setters -------------------------------------------------------

    public void SetPosition(Vector3 pos)
    {
        _position = pos;
        _dirty = true;
    }
    public void SetRotation(Quaternion rot)
    {
        _rotation = rot;
        _dirty = true;
    }
    public void SetScale(Vector3* scale)
    {
        _scale = *scale;
        _dirty = true;
    }
    
    
    public Vector3    GetPosition() => _position;
    public Quaternion GetRotation() => _rotation;
    public Vector3    GetScale()    => _scale;

    ///<summary>
    /// Returns a pointer to the cached transformation matrix, recalculates if dirty
    /// The pointer is valid for component lifetime
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Matrix4x4* GetTransformMatrix()
    {
        if (_dirty)
        {
            var t = Matrix4x4.CreateTranslation(_position);
            var r = Matrix4x4.CreateFromQuaternion(_rotation);
            var s = Matrix4x4.CreateScale(_scale);
            *_Matrix = s * r * t;
            _dirty = false;

        }
        return _Matrix;
    }
    
}