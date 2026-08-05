using System.Numerics;
using System.Runtime.InteropServices;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer;

/// <summary>Per-instance row in the geometry pipeline's instance SSBO. Mirrors
/// PbrUtils.slang::InstanceData under std430 at an 80B stride; the padding keeps a
/// <c>Span&lt;InstanceDataGPU&gt;</c> blitting cleanly into the shader struct.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct InstanceDataGPU
{
    public Matrix4x4 model;
    public uint materialIndex;
    public uint _pad0, _pad1, _pad2;
}

/// <summary>CPU-side transparent draw record. The forward+ transparent pass walks these sorted
/// back-to-front by view-space Z and issues one push constant plus vkCmdDrawIndexed each. Not a GPU
/// struct: the per-draw model and material index ride in push constants because the count is
/// small.</summary>
public struct TransparentDraw
{
    public Matrix4x4 Model;
    public uint      MaterialIndex;
    public uint      IndexCount;
    public uint      FirstIndex;     // into Engine.ResourceManager.GlobalIndexBuffer
    public float     ViewDepth;      // view-space Z of the entity origin, and the sort key
}

/// <summary>Mirrors VkDrawIndexedIndirectCommand. The cull compute shader writes one per surviving
/// renderable and vkCmdDrawIndexedIndirectCount consumes them.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct DrawIndexedIndirectCommandGpu
{
    public uint indexCount;
    public uint instanceCount;
    public uint firstIndex;
    public int  vertexOffset;
    public uint firstInstance;
}

/// <summary>Per-frame mapped UBO/SSBO bundle. Storage belongs to the renderer's
/// <see cref="GpuMemoryAllocator"/>, and Dispose is absent on purpose: callers free through
/// <c>Renderer.DestroyBuffer</c> so the handle and the suballocation go together.</summary>
unsafe struct UboBuffer
{
    public Buffer    buffer;
    public SubAlloc  alloc;
    public void*     mapped;
}

/// <summary>glTF alpha mode, packed into <c>PbrMaterial.Flags</c>.</summary>
public enum AlphaMode : uint
{
    Opaque = 0,
    Mask   = 1,
    Blend  = 2,
}

public static class PbrMaterialVolume
{
    /// <summary>Stands in for "no participating volume", matching the shader guard that treats
    /// mediumSigmaA above 1e6 as no absorption. glTF's default is +infinity, which does not upload
    /// cleanly; this reads identically.</summary>
    public const float NoAbsorptionDistance = 1e30f;
}

/// <summary>Alpha-mode decoding for <see cref="PbrMaterial"/>.</summary>
public static class PbrMaterialAlphaExtensions
{
    /// <summary>Blend wins over mask if both bits are set, which glTF never produces since the two
    /// are mutually exclusive there.</summary>
    public static AlphaMode GetAlphaMode(this in PbrMaterial mat)
    {
        if ((mat.Flags & 0x4u) != 0u) return AlphaMode.Blend;
        if ((mat.Flags & 0x1u) != 0u) return AlphaMode.Mask;
        return AlphaMode.Opaque;
    }
}