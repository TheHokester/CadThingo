using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using CadThingo.VulkanEngine.Renderer;

namespace CadThingo.VulkanEngine;

//TODO: Frustum struct STUB

public sealed unsafe class CameraComponent : Component
{
    private float _fovRadians;
    private float _aspectRatio;
    private float _nearPlane;
    private float _farPlane;
    private Frustum _frustum;
    
    //projection matrix in unmanaged memory
    private Matrix4x4* _proj;
    private bool _projDirty = true;

    public CameraComponent()
    {
        _proj = (Matrix4x4*)NativeMemory.Alloc((nuint)sizeof(Matrix4x4));
        *_proj = Matrix4x4.Identity;
    }
    protected override void OnDestroy()
    {
        if (_proj != null)
        {
            NativeMemory.Free(_proj);
        }
    }

    public void SetPerspective(float fovDegrees, float aspect, float near, float far)
    {
        _fovRadians = fovDegrees * (float)Math.PI / 180.0f;
        _aspectRatio = aspect;
       _nearPlane = near;
       _farPlane = far;
       _projDirty= true;
    }

    ///<summary>Computes the view matrix from the owners TransformComponent</summary>
    public Matrix4x4 GetViewMatrix()
    {
        TransformComponent* t = Owner-> GetComponent<TransformComponent>();
        if (t == null) return Matrix4x4.Identity;

        var pos = t->GetPosition();
        var rot = t->GetRotation();
        var forward = Vector3.Transform(new Vector3(0f,0f,-1f), rot);
        var up = Vector3.Transform(new Vector3(0f, 1f, 0f), rot);
        
        return Matrix4x4.CreateLookAt(pos, pos + forward, up);
    }

    ///<summary> returns a pointer to the cached proj matrix. </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Matrix4x4* GetProjectionMatrix()
    {
        if (_projDirty)
        {
            *_proj = Matrix4x4.CreatePerspectiveFieldOfView(_fovRadians, _aspectRatio, _nearPlane, _farPlane);
            _projDirty = false;
        }
        return _proj;
    }

    public Frustum GetFrustum()
    {
        return default;
    }
    
}