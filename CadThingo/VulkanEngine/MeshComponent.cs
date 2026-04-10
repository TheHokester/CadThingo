using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Silk.NET.Assimp;
using Silk.NET.Maths;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine;

// Alias so call-sites can write Matrix4x4 instead of the Silk generic form.
using Mat4  = Silk.NET.Maths.Matrix4X4<float>;
using Mat3  = Silk.NET.Maths.Matrix3X3<float>;
using Mat34 = Silk.NET.Maths.Matrix3X4<float>;  // replaces glm::mat3x4
using Vec4  = Silk.NET.Maths.Vector4D<float>;
using Vec3  = Silk.NET.Maths.Vector3D<float>;

public static class NormalMatrixOps
{
    // ── Extract upper-left 3×3 from a 4×4 ────────────────────
    // glm::mat3(mat4)
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Mat3 ExtractMat3(Mat4 m) => new(
        m.M11, m.M12, m.M13,
        m.M21, m.M22, m.M23,
        m.M31, m.M32, m.M33);
 
    // ── Transpose ─────────────────────────────────────────────
    // glm::transpose(mat3)
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Mat3 Transpose(Mat3 m) => new(
        m.M11, m.M21, m.M31,
        m.M12, m.M22, m.M32,
        m.M13, m.M23, m.M33);
 
    // ── Inverse via adjugate ──────────────────────────────────
    // Silk.NET.Maths has no Matrix3X3.Ops file, so we implement
    // this ourselves using the cofactor / adjugate method.
    public static Mat3 Invert(Mat3 m)
    {
        // Cofactors along each row
        float c11 =  (m.M22 * m.M33 - m.M23 * m.M32);
        float c12 = -(m.M21 * m.M33 - m.M23 * m.M31);
        float c13 =  (m.M21 * m.M32 - m.M22 * m.M31);
        float c21 = -(m.M12 * m.M33 - m.M13 * m.M32);
        float c22 =  (m.M11 * m.M33 - m.M13 * m.M31);
        float c23 = -(m.M11 * m.M32 - m.M12 * m.M31);
        float c31 =  (m.M12 * m.M23 - m.M13 * m.M22);
        float c32 = -(m.M11 * m.M23 - m.M13 * m.M21);
        float c33 =  (m.M11 * m.M22 - m.M12 * m.M21);
 
        float det = m.M11 * c11 + m.M12 * c12 + m.M13 * c13;
        if (MathF.Abs(det) < 1e-6f)
            return Mat3.Identity;
 
        float r = 1f / det;
        // Inverse = (1/det) * adjugate = (1/det) * Transpose(cofactors)
        // We transpose by swapping row/col indices on the cofactor matrix.
        return new Mat3(
            c11 * r, c21 * r, c31 * r,
            c12 * r, c22 * r, c32 * r,
            c13 * r, c23 * r, c33 * r);
    }
 
    // ── Combined: glm::transpose(glm::inverse(glm::mat3(m4))) ─
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Mat3 NormalMatrix(Mat4 modelMatrix)
        => Transpose(Invert(ExtractMat3(modelMatrix)));
}
public static class Mat34Helpers
{
    /// <summary>
    /// Packs a 3×3 normal matrix into a Matrix3X4 for GPU upload.
    /// Each ROW of the result holds one COLUMN of <paramref name="n"/>, w=0.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Mat34 Pack(Mat3 n) => new(
        // Row0 = column 0 of n  (n.M11, n.M21, n.M31, 0)
        n.M11, n.M21, n.M31, 0f,
        // Row1 = column 1 of n  (n.M12, n.M22, n.M32, 0)
        n.M12, n.M22, n.M32, 0f,
        // Row2 = column 2 of n  (n.M13, n.M23, n.M33, 0)
        n.M13, n.M23, n.M33, 0f);
 
    /// <summary>
    /// Unpacks a GPU Matrix3X4 back into a Matrix3X3.
    /// Reverses Pack() — each COLUMN of the result is one ROW of <paramref name="packed"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Mat3 Unpack(Mat34 packed) => new(
        packed.M11, packed.M12, packed.M13,   // row 0 ← row0 of packed
        packed.M21, packed.M22, packed.M23,   // row 1 ← row1 of packed
        packed.M31, packed.M32, packed.M33);  // row 2 ← row2 of packed
}

/// <summary>
/// Per-instance GPU data: model matrix + normal matrix + material index.<br/>
///
///   Offset   0 : ModelMatrix   — 64 bytes  (Mat4  = Matrix4X4&lt;float&gt;)<br/>
///   Offset  64 : NormalMatrix  — 48 bytes  (Mat34 = Matrix3X4&lt;float&gt;)<br/>
///   Offset 112 : MaterialIndex —  4 bytes  (uint)<br/>
///   Total      : 116 bytes<br/>
///
/// Pack=4 prevents the compiler from inserting padding between fields.
/// </summary>
public unsafe struct InstanceData
{
    //model matrix
    public Matrix4X4<float> ModelMatrix = default;
    //normal matrix as mat 3x4 3columns of vec4
    public Matrix3X4<float> NormalMatrix = default;

    private uint MaterialIndex;

    public InstanceData()
    {
        MaterialIndex = 0;
        ModelMatrix = Matrix4X4<float>.Identity;
        NormalMatrix.Row1 = new Vector4D<float>(1.0f, 0.0f, 0.0f, 0.0f);
        NormalMatrix.Row2 = new Vector4D<float>(0.0f, 1.0f, 0.0f, 0.0f);
        NormalMatrix.Row3 = new Vector4D<float>(0.0f, 0.0f, 1.0f, 0.0f);
        
    }
    /// <summary>
    /// Creates an instance in accordance with the given transform and material index, computes the normal matrix.
    /// </summary>
    /// <param name="transform"></param>
    /// <param name="materialIndex"></param>
    public InstanceData(Mat4 transform, uint materialIndex)
    {
       ModelMatrix = transform;
       NormalMatrix = Mat34Helpers.Pack(NormalMatrixOps.NormalMatrix(ModelMatrix));
       MaterialIndex = materialIndex;
    }
    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Mat4 GetModelMatrix() => ModelMatrix;
    
    /// <summary>
    /// Sets the model matrix and recomputes the normal matrix.
    /// </summary>
    public void SetModelMatrix(Mat4 matrix)
    {
        ModelMatrix = matrix;
        NormalMatrix = Mat34Helpers.Pack(NormalMatrixOps.NormalMatrix(ModelMatrix));
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Mat3 GetNormalMatrix() => Mat34Helpers.Unpack(NormalMatrix);


    public static VertexInputBindingDescription GetBindingDescription()
    {
        VertexInputBindingDescription bindingDescription = new()
        {
            Binding = 1,
            Stride = (uint)Unsafe.SizeOf<InstanceData>(),
            InputRate = VertexInputRate.Instance
        };
        return bindingDescription;
    }

    public static VertexInputAttributeDescription[] GetAttributeDescriptions()
    {
        uint modelBase = (uint)Marshal.OffsetOf<InstanceData>(nameof(ModelMatrix));
        uint normalBase = (uint)Marshal.OffsetOf<InstanceData>(nameof(NormalMatrix));
        uint vec4size = (uint)Unsafe.SizeOf<Vector4D<float>>();
        return new VertexInputAttributeDescription[]
        {
            //bindings 4-7 for model matrix columns
            new()
            {   
                Location = 4,
                Binding = 1,
                Format = Format.R32G32B32A32Sfloat,
                Offset = modelBase + 0 * vec4size
            },
            new()
            {   
                Location = 5,
                Binding = 1,
                Format = Format.R32G32B32A32Sfloat,
                Offset = modelBase + 1 * vec4size
            },
            new()
            {   
                Location = 6,
                Binding = 1,
                Format = Format.R32G32B32A32Sfloat,
                Offset = modelBase + 2 * vec4size
            },
            new()
            {   
                Location = 7,
                Binding = 1,
                Format = Format.R32G32B32A32Sfloat,
                Offset = modelBase + 3 * vec4size
            }
            //Normal matrix columns (8-10)
            ,new()
            {   
                Location = 8,
                Binding = 1,
                Format = Format.R32G32B32A32Sfloat,
                Offset = normalBase + 0 * vec4size
            },
            new()
            {   
                Location = 9,
                Binding = 1,
                Format = Format.R32G32B32A32Sfloat,
                Offset = normalBase + 1 * vec4size
            },
            new()
            {   
                Location = 10,
                Binding = 1,
                Format = Format.R32G32B32A32Sfloat,
                Offset = normalBase + 2 * vec4size
            }
        };
    }

    public static VertexInputAttributeDescription[] GetModelMatrixAttributeDescriptions()
    {
        uint modelBase = (uint)Marshal.OffsetOf<InstanceData>(nameof(ModelMatrix));
        uint vec4size = (uint)Unsafe.SizeOf<Vector4D<float>>();
        return new VertexInputAttributeDescription[]
        {
            //bindings 4-7 for model matrix columns
            new()
            {
                Location = 4,
                Binding = 1,
                Format = Format.R32G32B32A32Sfloat,
                Offset = modelBase + 0 * vec4size
            },
            new()
            {
                Location = 5,
                Binding = 1,
                Format = Format.R32G32B32A32Sfloat,
                Offset = modelBase + 1 * vec4size
            },
            new()
            {
                Location = 6,
                Binding = 1,
                Format = Format.R32G32B32A32Sfloat,
                Offset = modelBase + 2 * vec4size
            },
            new()
            {
                Location = 7,
                Binding = 1,
                Format = Format.R32G32B32A32Sfloat,
                Offset = modelBase + 3 * vec4size
            }
        };
    }

    public static VertexInputAttributeDescription[] GetNormalMatrixAttributeDescriptions()
    {
        uint normalBase = (uint)Marshal.OffsetOf<InstanceData>(nameof(NormalMatrix));
        uint vec4size = (uint)Unsafe.SizeOf<Vector4D<float>>();

        return new VertexInputAttributeDescription[]
        {
            new()
            {
                Location = 8,
                Binding = 1,
                Format = Format.R32G32B32A32Sfloat,
                Offset = normalBase + 0 * vec4size
            },
            new()
            {
                Location = 9,
                Binding = 1,
                Format = Format.R32G32B32A32Sfloat,
                Offset = normalBase + 1 * vec4size
            },
            new()
            {
                Location = 10,
                Binding = 1,
                Format = Format.R32G32B32A32Sfloat,
                Offset = normalBase + 2 * vec4size
            }
        };
    }
    
    
    
}
public unsafe struct Vertex
{
    public Vector3 Position;
    public Vector3 Normal;
    public Vector2 TexCoord;
    public Vector4 Tangent;

    public static VertexInputBindingDescription GetBindingDescription()
    {
        VertexInputBindingDescription bindingDescription = new()
        {
            Binding = 0,
            Stride = (uint)sizeof(Vertex),
            InputRate = VertexInputRate.Vertex,
        };
        return bindingDescription;
    }

    public static VertexInputAttributeDescription[] GetAttributeDescriptions()
    {
        return new VertexInputAttributeDescription[]
        {
            new()
            {
                Binding = 0,
                Location = 0,
                Format = Format.R32G32B32A32Sfloat,
                Offset = (uint)Marshal.OffsetOf<Vertex>(nameof(Position)),
            },
            new()
            {
                Binding = 0,
                Location = 1,
                Format = Format.R32G32B32Sfloat,
                Offset = (uint)Marshal.OffsetOf<Vertex>(nameof(Normal))
            },
            new()
            {
                Binding = 0,
                Location = 2,
                Format = Format.R32G32Sfloat,
                Offset = (uint)Marshal.OffsetOf<Vertex>(nameof(TexCoord))
            },
            new()
            {
                Binding = 0,
                Location = 3,
                Format = Format.R32G32B32A32Sfloat,
                Offset = (uint)Marshal.OffsetOf<Vertex>(nameof(Tangent))
            }
        };

    }
}


public sealed unsafe class MeshComponent : Component
{
    public Mesh* mesh;
    public Material* material;
    public BoundingBox* boundingBox;
    
    public MeshComponent() { }

    public MeshComponent(Mesh* mesh, Material* material)
    {
        this.mesh = mesh;
        this.material = material;
    }
    
    public override void Render()
    {
        if (mesh == null || material == null) return;
        
        TransformComponent* transform = Owner->GetComponent<TransformComponent>();
        if (transform == null) return;

        material->Bind();
        material->SetUniform("modelMatrix", transform->GetTransformMatrix());
        mesh->Render();
    }
    public BoundingBox GetBoundingBox()
    {
        return *boundingBox;
    }
}

//-----------------------------------------
// TODO: Stubs exist, implement them.
// ----------------------------------------
///<summary>
/// Boundingbox</summary>
///
public unsafe struct BoundingBox
{
    public Vector3 Min, Max;
    public BoundingBox(Vector3 min, Vector3 max)
    {
        Min = min;
        Max = max;
    }
    public Vector3 Size => Max - Min;
    
    public Vector3 Center => (Min + Max) * 0.5f;
    
    public bool Contains(Vector3 point) => point.X >= Min.X && point.X <= Max.X && point.Y >= Min.Y && point.Y <= Max.Y && point.Z >= Min.Z && point.Z <= Max.Z;

    public bool Union(BoundingBox box)
    {
        return false;
    }
    public void Transform(Matrix4x4* matrix)
    {
        var corners = new[]
        {
            new Vector3(Min.X, Min.Y, Min.Z),
            new Vector3(Max.X, Min.Y, Min.Z),
            new Vector3(Min.X, Max.Y, Min.Z),
            new Vector3(Max.X, Max.Y, Min.Z),
            new Vector3(Min.X, Min.Y, Max.Z),
            new Vector3(Max.X, Min.Y, Max.Z),
            new Vector3(Min.X, Max.Y, Max.Z),
            new Vector3(Max.X, Max.Y, Max.Z)

        };
        var transformed = corners
            .Select(c => Vector3.Transform(c, *matrix))
            .ToArray();
        var min = transformed.Aggregate(Vector3.Min);
        var max = transformed.Aggregate(Vector3.Max);
        this = new BoundingBox(min, max);
    }
    
}
/// <summary>
/// Mesh allocated in unmanaged memory so MeshComponent can hold a raw Mesh*.
/// </summary>
public unsafe struct Mesh
{
    public void Render()
    {
        
    }
}
 
/// <summary>
/// Material allocated in unmanaged memory.
/// SetUniform takes a Matrix4x4* to avoid a 64-byte copy on every draw call.
/// </summary>
public unsafe struct Material
{
    public void Bind()
    {
        
    }

    public void SetUniform(string name, Matrix4x4* value)
    {
        
    }

    public void SetUniform(string name, float value)
    {
        
    }
}