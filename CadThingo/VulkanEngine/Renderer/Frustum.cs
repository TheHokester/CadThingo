using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace CadThingo.VulkanEngine.Renderer;

/// <summary>A world-space plane stored as (unit normal, signed distance from origin).</summary>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct Plane
{
    public Vector4 Data;

    public Vector3 Normal   => new(Data.X, Data.Y, Data.Z);
    public float   Distance => Data.W;

    /// <summary>Builds a plane from the coefficients of Ax + By + Cz + D = 0. The normal is
    /// normalised, so distance comparisons come out in world units.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Plane FromCoefficients(float a, float b, float c, float d)
    {
        float invLen = 1f / MathF.Sqrt(a * a + b * b + c * c);
        return new Plane
        {
            Data = new Vector4(a * invLen, b * invLen, c * invLen, d * invLen)
        };
    }

    /// <summary>Signed distance from a point to this plane. Positive means the same side as the
    /// normal.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float SignedDistance(Vector3 point) =>
        Data.X * point.X + Data.Y * point.Y + Data.Z * point.Z + Data.W;
}

/// <summary>Six view-frustum planes in a fixed inline array, so the GC never touches an instance
/// and culling can walk it through a pointer.</summary>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct Frustum
{
    // Plane order matches the standard convention: left, right, bottom, top, near, far.
    private fixed float _planeData[6 * 4];

    public const int Left   = 0;
    public const int Right  = 1;
    public const int Bottom = 2;
    public const int Top    = 3;
    public const int Near   = 4;
    public const int Far    = 5;

    /// <summary>Reads a plane by index through a pointer into the fixed array.</summary>
    public Plane GetPlane(int index)
    {
        fixed (float* p = _planeData)
        {
            float* slot = p + index * 4;
            return new Plane
            {
                Data = new Vector4(slot[0], slot[1], slot[2], slot[3])
            };
        }
    }

    /// <summary>Writes a plane by index.</summary>
    public void SetPlane(int index, Plane plane)
    {
        fixed (float* p = _planeData)
        {
            float* slot = p + index * 4;
            slot[0] = plane.Data.X;
            slot[1] = plane.Data.Y;
            slot[2] = plane.Data.Z;
            slot[3] = plane.Data.W;
        }
    }

    public Plane PlaneLeft   { get => GetPlane(Left);   set => SetPlane(Left,   value); }
    public Plane PlaneRight  { get => GetPlane(Right);  set => SetPlane(Right,  value); }
    public Plane PlaneBottom { get => GetPlane(Bottom); set => SetPlane(Bottom, value); }
    public Plane PlaneTop    { get => GetPlane(Top);    set => SetPlane(Top,    value); }
    public Plane PlaneNear   { get => GetPlane(Near);   set => SetPlane(Near,   value); }
    public Plane PlaneFar    { get => GetPlane(Far);    set => SetPlane(Far,    value); }

    /// <summary>Extracts the six planes from a combined view-projection matrix by the Gribb and
    /// Hartmann row method.</summary>
    /// <param name="vulkanNDC">True for Vulkan's z range of [0, 1], false for OpenGL's [-1, 1].
    /// The two conventions differ only in the near plane.</param>
    public static Frustum FromViewProjection(Matrix4x4 vp, bool vulkanNDC = true)
    {
        // System.Numerics.Matrix4x4 is row-major, so the planes come out of row sums:
        //   left = r4 + r1, right = r4 - r1, bottom = r4 + r2, top = r4 - r2, far = r4 - r3.
        Frustum f = default;

        f.SetPlane(Left,   Plane.FromCoefficients(
            vp.M14 + vp.M11, vp.M24 + vp.M21, vp.M34 + vp.M31, vp.M44 + vp.M41));

        f.SetPlane(Right,  Plane.FromCoefficients(
            vp.M14 - vp.M11, vp.M24 - vp.M21, vp.M34 - vp.M31, vp.M44 - vp.M41));

        f.SetPlane(Bottom, Plane.FromCoefficients(
            vp.M14 + vp.M12, vp.M24 + vp.M22, vp.M34 + vp.M32, vp.M44 + vp.M42));

        f.SetPlane(Top,    Plane.FromCoefficients(
            vp.M14 - vp.M12, vp.M24 - vp.M22, vp.M34 - vp.M32, vp.M44 - vp.M42));

        if (vulkanNDC)
        {
            // Vulkan z in [0, 1]: the near plane is row 3 alone.
            f.SetPlane(Near, Plane.FromCoefficients(
                vp.M13, vp.M23, vp.M33, vp.M43));
        }
        else
        {
            // OpenGL z in [-1, 1]: the near plane is r4 + r3.
            f.SetPlane(Near, Plane.FromCoefficients(
                vp.M14 + vp.M13, vp.M24 + vp.M23, vp.M34 + vp.M33, vp.M44 + vp.M43));
        }

        f.SetPlane(Far, Plane.FromCoefficients(
            vp.M14 - vp.M13, vp.M24 - vp.M23, vp.M34 - vp.M33, vp.M44 - vp.M43));

        return f;
    }

    /// <summary>Tests an axis-aligned box against the frustum by the positive-vertex method: the
    /// box is outside when the corner furthest along a plane's normal sits behind that plane.
    /// Rejects on the first failing plane, so a typical scene tests one or two.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool Intersects(BoundingBox box)
    {
        // Walks the fixed array through a pointer rather than copying each Plane out of GetPlane.
        fixed (float* p = _planeData)
        {
            for (int i = 0; i < 6; i++)
            {
                float* slot = p + i * 4;

                float nx = slot[0];
                float ny = slot[1];
                float nz = slot[2];
                float d  = slot[3];

                float px = nx >= 0f ? box.Max.X : box.Min.X;
                float py = ny >= 0f ? box.Max.Y : box.Min.Y;
                float pz = nz >= 0f ? box.Max.Z : box.Min.Z;

                if (nx * px + ny * py + nz * pz + d < 0f)
                    return false;
            }
        }
        return true;
    }

    /// <summary>Pointer overload, for callers already holding a <c>BoundingBox*</c> that would
    /// otherwise copy the struct onto the managed stack.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool Intersects(BoundingBox* box) => Intersects(*box);
}