using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace CadThingo.VulkanEngine;

/// <summary>
/// Stores position, rotation and scale; caches the local + world transformation matrices.
/// World matrix is composed by walking up the Parent chain.
/// </summary>
public sealed unsafe class TransformComponent : Component
{
    private Vector3 _position = Vector3.Zero;
    private Quaternion _rotation = Quaternion.Identity;
    private Vector3 _scale = Vector3.One;

    // Parent entity in the scenegraph. Null = this transform is a root.
    public Entity* Parent;

    // Both matrices live in unmanaged memory so callers can hold stable pointers
    // across GC moves (the renderer reads world via pointer in the geometry pass).
    private readonly Matrix4x4* _localMatrix;
    private readonly Matrix4x4* _worldMatrix;

    private bool _localDirty = true;

    public TransformComponent()
    {
        _localMatrix = (Matrix4x4*)NativeMemory.Alloc((nuint)sizeof(Matrix4x4));
        _worldMatrix = (Matrix4x4*)NativeMemory.Alloc((nuint)sizeof(Matrix4x4));
        *_localMatrix = Matrix4x4.Identity;
        *_worldMatrix = Matrix4x4.Identity;
    }

    protected override void OnDestroy()
    {
        if (_localMatrix != null) NativeMemory.Free(_localMatrix);
        if (_worldMatrix != null) NativeMemory.Free(_worldMatrix);
    }

    // ---- Setters -------------------------------------------------------

    public void SetPosition(Vector3 pos)
    {
        _position = pos;
        _localDirty = true;
    }
    public void SetRotation(Quaternion rot)
    {
        _rotation = rot;
        _localDirty = true;
    }
    public void SetScale(Vector3 scale)
    {
        _scale = scale;
        _localDirty = true;
    }


    public Vector3    GetPosition() => _position;
    public Quaternion GetRotation() => _rotation;
    public Vector3    GetScale()    => _scale;

    ///<summary>
    /// Returns a pointer to the cached LOCAL transformation matrix, recalculates if dirty.
    /// Pointer is valid for component lifetime.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Matrix4x4* GetModelMatrix()
    {
        if (_localDirty)
        {
            var t = Matrix4x4.CreateTranslation(_position);
            var r = Matrix4x4.CreateFromQuaternion(_rotation);
            var s = Matrix4x4.CreateScale(_scale);
            *_localMatrix = s * r * t;
            _localDirty = false;
        }
        return _localMatrix;
    }

    ///<summary>
    /// Returns a pointer to the WORLD transformation matrix.
    /// Walks the Parent chain on every call (no cross-frame cache) — depth is bounded
    /// by glTF scene depth so this is cheap. Pointer is valid for component lifetime.
    /// </summary>
    public Matrix4x4* GetWorldMatrix()
    {
        var local = *GetModelMatrix();
        if (Parent == null)
        {
            *_worldMatrix = local;
            return _worldMatrix;
        }
        var parentTransform = Parent->GetComponent<TransformComponent>();
        if (parentTransform == null)
        {
            *_worldMatrix = local;
            return _worldMatrix;
        }
        // C# Matrix4x4 is row-major; child_world = child_local * parent_world composes
        // correctly when uploaded and read as column-major in the shader.
        *_worldMatrix = local * *parentTransform.GetWorldMatrix();
        return _worldMatrix;
    }
}