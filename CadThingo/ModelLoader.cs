using System.Numerics;
using System.Runtime.CompilerServices;
using CadThingo.GraphicsPipeline;
using Silk.NET.Assimp;

namespace CadThingo;

public unsafe class ModelLoader
{
    public static void LoadObjModel(VulkanContext ctx, string path)
    {
        using var assimp = Assimp.GetApi();
        var scene = assimp.ImportFile(path , (uint)PostProcessPreset.TargetRealTimeMaximumQuality);
        
        var vertexMap = new Dictionary<VkVertex, uint>();
        var vertices = new List<VkVertex>();
        var indices = new List<uint>();

        VisitSceneNode(scene->MRootNode);
        assimp.ReleaseImport(scene);

        ctx.Vertices = vertices.ToArray();
        ctx.Indices = indices.ToArray();

        void VisitSceneNode(Node* node)
        {
            for (var m = 0; m < node->MNumMeshes; m++)
            {
                var mesh = scene->MMeshes[node->MMeshes[m]];

                for (var f = 0; f < mesh->MNumFaces; f++)
                {
                    var face = mesh->MFaces[f];

                    for (var i = 0; i < face.MNumIndices; i++)
                    {
                        var index = face.MIndices[i];
                        
                        var position = mesh->MVertices[index];
                        var texture = mesh->MTextureCoords[0][index];

                        VkVertex vertex = new()
                        {
                            pos = new Vector3(position.X, position.Y, position.Z),
                            color = new Vector3(1, 1, 1),
                            uv = new Vector2(texture.X, 1.0f - texture.Y)
                        };

                        if (vertexMap.TryGetValue(vertex, out var meshIndex))
                        {
                            indices.Add(meshIndex);
                        }
                        else
                        {
                            indices.Add((uint)vertices.Count);
                            vertexMap[vertex] = (uint)vertices.Count;
                            vertices.Add(vertex);
                        }
                    }
                }
            }

            for (var c = 0; c < node->MNumChildren; c++)
            {
                VisitSceneNode(node->MChildren[c]);
            }
        }

    }
} 