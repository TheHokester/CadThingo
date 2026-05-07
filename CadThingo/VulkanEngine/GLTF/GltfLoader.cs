using System.Numerics;
using SharpGLTF.Memory;
using SharpGLTF.Schema2;

namespace CadThingo.VulkanEngine.GLTF;

/// <summary>
/// Loads a .glb (or .gltf) file via SharpGLTF.Core, builds engine-side mesh/material
/// resources, and instantiates a hierarchy of Entities under <see cref="Scene"/>.
/// Returns the root entity so the caller can position the whole asset.
///
/// Supported channels: BaseColor, MetallicRoughness, Normal, Occlusion, Emissive.
/// Required vertex attributes: POSITION, NORMAL, TEXCOORD_0, TANGENT.
/// Out of scope (Phase 1): animations, skinning, morph targets, embedded cameras, embedded lights.
/// </summary>
public static unsafe class GltfLoader
{
    public static Entity* Load(
        string path,
        string idPrefix,
        ResourceManager rm,
        Renderer.Renderer renderer,
        Scene scene)
    {
        // 1. Fallback textures must exist before any material build runs.
        GltfDefaults.EnsureRegistered(rm, renderer);

        // 2. Parse the .glb/.gltf file. SharpGLTF reads embedded image bytes for .glb
        //    and resolves relative URIs for .gltf — both work through this single API.
        var model = ModelRoot.Load(path);

        // 3. Materials pass — build PbrMaterial structs (with bindless texture indices) and
        //    register on the scene. Map SharpGLTF Material → engine Scene material index.
        var materialIndexMap = new Dictionary<SharpGLTF.Schema2.Material, int>();
        for (int matIdx = 0; matIdx < model.LogicalMaterials.Count; matIdx++)
        {
            var gltfMat = model.LogicalMaterials[matIdx];
            var pbr = GltfMaterialResource.BuildAndRegister(idPrefix, matIdx, gltfMat, rm, renderer);
            int sceneMatIdx = scene.AddMaterial(pbr);
            materialIndexMap[gltfMat] = sceneMatIdx;
        }

        // 4. Meshes pass — extract vertex/index arrays per primitive and upload via
        //    GltfMeshResource → ResourceManager.UploadMesh. Map (mesh, primIdx) → Mesh*
        //    address (stored as nint for Dictionary compatibility) so the scene-tree
        //    pass can attach MeshComponents pointing at stable NativeMemory.
        var primitiveMeshMap = new Dictionary<(int meshIdx, int primIdx), nint>();
        var primitiveMaterialMap = new Dictionary<(int meshIdx, int primIdx), int>();

        for (int meshIdx = 0; meshIdx < model.LogicalMeshes.Count; meshIdx++)
        {
            var gltfMesh = model.LogicalMeshes[meshIdx];
            for (int primIdx = 0; primIdx < gltfMesh.Primitives.Count; primIdx++)
            {
                var prim = gltfMesh.Primitives[primIdx];
                var (verts, indices) = ExtractPrimitive(prim, $"{path} mesh[{meshIdx}].primitive[{primIdx}]");

                string meshId = $"{idPrefix}:mesh:{meshIdx}:prim:{primIdx}";
                rm.Load<MeshResource>(meshId, _ => new GltfMeshResource(meshId, rm, verts, indices));
                Mesh* meshPtr = rm.GetMesh(meshId);
                primitiveMeshMap[(meshIdx, primIdx)] = (nint)meshPtr;

                int sceneMatIdx = -1;
                if (prim.Material != null && materialIndexMap.TryGetValue(prim.Material, out var found))
                    sceneMatIdx = found;
                primitiveMaterialMap[(meshIdx, primIdx)] = sceneMatIdx;
            }
        }

        // 5. Scene tree pass — walk the default scene; instantiate one Entity per node,
        //    plus one child entity per primitive on nodes that carry a mesh. Use
        //    Scene.AddChild so the hierarchical TransformComponent.Parent chain is set.
        var defaultScene = model.DefaultScene ?? model.LogicalScenes[0];

        // Synthetic root so the caller has a single handle to the whole asset.
        Entity* root = Entity.Create($"{idPrefix}_root");
        root->AddComponent(new TransformComponent());
        scene.AddEntity(root);

        foreach (var node in defaultScene.VisualChildren)
        {
            VisitNode(node, root, model, scene, idPrefix, primitiveMeshMap, primitiveMaterialMap);
        }

        return root;
    }

    private static void VisitNode(
        Node node,
        Entity* parent,
        ModelRoot model,
        Scene scene,
        string idPrefix,
        Dictionary<(int meshIdx, int primIdx), nint> primitiveMeshMap,
        Dictionary<(int meshIdx, int primIdx), int> primitiveMaterialMap)
    {
        Entity* entity =
            Entity.Create(string.IsNullOrEmpty(node.Name) ? $"{idPrefix}_node_{node.LogicalIndex}" : node.Name);

        var transform = new TransformComponent();
        var local = node.LocalTransform;
        var affine = local.GetDecomposed();
        if (affine.IsSRT)
        {
            transform.SetPosition(affine.Translation);
            transform.SetRotation(affine.Rotation);
            transform.SetScale(affine.Scale);
        }

        entity->AddComponent(transform);
        scene.AddChild(parent, entity);

        // Attach a child MeshComponent entity per primitive so per-draw material
        // binding stays one-to-one with one MeshComponent → one Mesh* + one materialIndex.
        if (node.Mesh != null)
        {
            int meshIdx = node.Mesh.LogicalIndex;
            for (int primIdx = 0; primIdx < node.Mesh.Primitives.Count; primIdx++)
            {
                // Cast back to Mesh* — points into the stable NativeMemory owned by
                // the GltfMeshResource registered during the meshes pass.
                Mesh* meshPtr = (Mesh*)primitiveMeshMap[(meshIdx, primIdx)];
                int matIdx = primitiveMaterialMap[(meshIdx, primIdx)];

                Entity* primEntity = Entity.Create($"{idPrefix}_node_{node.LogicalIndex}_prim_{primIdx}");
                primEntity->AddComponent(
                    new TransformComponent()); // identity local — node entity holds the world transform
                primEntity->AddComponent(new MeshComponent(meshPtr, matIdx));
                scene.AddChild(entity, primEntity);
            }
        }

        foreach (var child in node.VisualChildren)
        {
            VisitNode(child, entity, model, scene, idPrefix, primitiveMeshMap, primitiveMaterialMap);
        }
    }

    private static (Vertex[] verts, uint[] indices) ExtractPrimitive(MeshPrimitive prim, string contextLabel)
    {
        var positionsAcc = prim.GetVertexAccessor("POSITION") ?? throw new InvalidDataException(
            $"glTF primitive {contextLabel} is missing required POSITION accessor.");
        var normalsAcc = prim.GetVertexAccessor("NORMAL") ?? throw new InvalidDataException(
            $"glTF primitive {contextLabel} is missing required NORMAL accessor.");
        var uvsAcc = prim.GetVertexAccessor("TEXCOORD_0");
        var uvsFound = uvsAcc != null;

        var tangentsAcc = prim.GetVertexAccessor("TANGENT");
        var tangentsFound = tangentsAcc != null;


        var positions = positionsAcc.AsVector3Array().ToArray();
        var normals = normalsAcc.AsVector3Array().ToArray();
        var uvs = uvsFound ? uvsAcc.AsVector2Array().ToArray() : new Vector2[positions.Length];
        {
            //kept ones level lower as a native c# array to ensure that 
            var tangents = tangentsFound ? tangentsAcc.AsVector4Array().ToArray() : GenerateTangentsForPrimitive(prim);

            var n = positions.Length;
            if (normals.Length != n || uvs.Length != n || tangents.Length != n)
                throw new InvalidDataException(
                    $"glTF primitive {contextLabel} has mismatched vertex-attribute counts " +
                    $"(positions={n}, normals={normals.Length}, uvs={uvs.Length}, tangents={tangents.Length}).");

            var verts = new Vertex[n];
            for (int i = 0; i < n; i++)
            {
                verts[i] = new Vertex
                {
                    Position = positions[i],
                    Normal = normals[i],
                    TexCoord = uvs[i], // glTF UV origin is top-left, same as Vulkan — no flip needed
                    Tangent = tangents[i],
                };
            }

            var indexList = prim.GetIndices();
            var indices = new uint[indexList.Count];
            for (int i = 0; i < indexList.Count; i++) indices[i] = indexList[i];

            return (verts, indices);
        }
    }

    public static Vector4[] GenerateTangentsForPrimitive(MeshPrimitive primitive)
    {
        // 1. Extract raw data from glTF Accessors
        var positions = primitive.GetVertexAccessor("POSITION").AsVector3Array();
        var normals = primitive.GetVertexAccessor("NORMAL").AsVector3Array();
        var uvsAccessor = primitive.GetVertexAccessor("TEXCOORD_0");
        var uvs = (uvsAccessor != null) ? uvsAccessor.AsVector2Array().ToArray() : new Vector2[positions.Count];
        
        var indices = primitive.GetIndices();

        int vertexCount = positions.Count;
        var tan1 = new Vector3[vertexCount];
        var tan2 = new Vector3[vertexCount];
        var resultTangents = new Vector4[vertexCount];


        // 2. Calculate Tangent and Bitangent for each triangle
        for (int i = 0; i < indices.Count; i += 3)
        {
            uint i1 = indices[i];
            uint i2 = indices[i + 1];
            uint i3 = indices[i + 2];

            Vector3 v1 = positions[(int)i1];
            Vector3 v2 = positions[(int)i2];
            Vector3 v3 = positions[(int)i3];

            Vector2 w1 = uvs[(int)i1];
            Vector2 w2 = uvs[(int)i2];
            Vector2 w3 = uvs[(int)i3];

            float x1 = v2.X - v1.X;
            float x2 = v3.X - v1.X;
            float y1 = v2.Y - v1.Y;
            float y2 = v3.Y - v1.Y;
            float z1 = v2.Z - v1.Z;
            float z2 = v3.Z - v1.Z;

            float s1 = w2.X - w1.X;
            float s2 = w3.X - w1.X;
            float t1 = w2.Y - w1.Y;
            float t2 = w3.Y - w1.Y;

            float r = 1.0f / (s1 * t2 - s2 * t1);
            Vector3 sdir = new Vector3((t2 * x1 - t1 * x2) * r, (t2 * y1 - t1 * y2) * r,
                (t2 * z1 - t1 * z2) * r);
            Vector3 tdir = new Vector3((s1 * x2 - s2 * x1) * r, (s1 * y2 - s2 * y1) * r,
                (s1 * z2 - s2 * z1) * r);

            tan1[i1] += sdir;
            tan1[i2] += sdir;
            tan1[i3] += sdir;

            tan2[i1] += tdir;
            tan2[i2] += tdir;
            tan2[i3] += tdir;
        }

        // 3. Finalize Tangent (Gram-Schmidt Orthogonalization)
        for (int a = 0; a < vertexCount; a++)
        {
            Vector3 n = normals[a];
            Vector3 t = tan1[a];

            // Gram-Schmidt orthogonalize
            Vector3 tangentXYZ = Vector3.Normalize(t - n * Vector3.Dot(n, t));

            // Calculate handedness (W component)
            float handedness = (Vector3.Dot(Vector3.Cross(n, t), tan2[a]) < 0.0f) ? -1.0f : 1.0f;

            resultTangents[a] = new Vector4(tangentXYZ, handedness);
        }

        return resultTangents;
    }
}