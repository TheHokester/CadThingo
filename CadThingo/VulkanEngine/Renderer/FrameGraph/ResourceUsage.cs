using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.FrameGraph;

public enum ResourceUsage
{
    ColorAttachment, DepthAttachment, DepthRead,
    SampledFragment, SampledCompute,
    StorageReadCompute, StorageWriteCompute, StorageRWCompute,
    IndirectArg, IndexBuffer, VertexBuffer, UniformRead,
    TransferSrc, TransferDst,
    StorageRT, AccelStructBuild, AccelStructRead,
    Present,
}

internal readonly record struct UsageInfo(
    PipelineStageFlags2 Stage,
    AccessFlags2 Access,
    ImageLayout Layout,
    bool IsWrite);

internal static class UsageTable
{
    public static UsageInfo Of(ResourceUsage u) => u switch
    {
        ResourceUsage.ColorAttachment => new(
            PipelineStageFlags2.ColorAttachmentOutputBit,
            AccessFlags2.ColorAttachmentReadBit | AccessFlags2.ColorAttachmentWriteBit,
            ImageLayout.ColorAttachmentOptimal, true),

        ResourceUsage.DepthAttachment => new(
            PipelineStageFlags2.EarlyFragmentTestsBit | PipelineStageFlags2.LateFragmentTestsBit,
            AccessFlags2.DepthStencilAttachmentReadBit | AccessFlags2.DepthStencilAttachmentWriteBit,
            ImageLayout.DepthStencilAttachmentOptimal, true),

        ResourceUsage.DepthRead => new(
            PipelineStageFlags2.EarlyFragmentTestsBit | PipelineStageFlags2.FragmentShaderBit,
            AccessFlags2.DepthStencilAttachmentReadBit,
            ImageLayout.DepthStencilReadOnlyOptimal, false),

        ResourceUsage.SampledFragment => new(
            PipelineStageFlags2.FragmentShaderBit,
            AccessFlags2.ShaderSampledReadBit,
            ImageLayout.ShaderReadOnlyOptimal, false),

        ResourceUsage.SampledCompute => new(
            PipelineStageFlags2.ComputeShaderBit,
            AccessFlags2.ShaderSampledReadBit,
            ImageLayout.ShaderReadOnlyOptimal, false),

        ResourceUsage.StorageReadCompute => new(
            PipelineStageFlags2.ComputeShaderBit,
            AccessFlags2.ShaderStorageReadBit,
            ImageLayout.General, false),

        ResourceUsage.StorageWriteCompute => new(
            PipelineStageFlags2.ComputeShaderBit,
            AccessFlags2.ShaderStorageWriteBit,
            ImageLayout.General, true),

        ResourceUsage.StorageRWCompute => new(
            PipelineStageFlags2.ComputeShaderBit,
            AccessFlags2.ShaderStorageReadBit | AccessFlags2.ShaderStorageWriteBit,
            ImageLayout.General, true),

        ResourceUsage.IndirectArg => new(
            PipelineStageFlags2.DrawIndirectBit,
            AccessFlags2.IndirectCommandReadBit,
            ImageLayout.Undefined /* buffer only */, false),

        ResourceUsage.IndexBuffer => new(),

        ResourceUsage.VertexBuffer => new(),

        ResourceUsage.UniformRead => new(),

        ResourceUsage.TransferSrc => new(
            PipelineStageFlags2.AllTransferBit, 
            AccessFlags2.TransferReadBit,
            ImageLayout.TransferSrcOptimal, false),

        ResourceUsage.TransferDst => new(
            PipelineStageFlags2.AllTransferBit, 
            AccessFlags2.TransferWriteBit,
            ImageLayout.TransferDstOptimal, true),
        
        ResourceUsage.StorageRT => new(
            PipelineStageFlags2.RayTracingShaderBitKhr,
            AccessFlags2.ShaderStorageReadBit | AccessFlags2.ShaderStorageWriteBit,
            ImageLayout.General, true),

        ResourceUsage.AccelStructBuild => new(
            PipelineStageFlags2.AccelerationStructureBuildBitKhr,
            AccessFlags2.AccelerationStructureWriteBitKhr,
            ImageLayout.Undefined, false),

        ResourceUsage.AccelStructRead => new(
            PipelineStageFlags2.RayTracingShaderBitKhr | PipelineStageFlags2.ComputeShaderBit | PipelineStageFlags2.FragmentShaderBit,
            AccessFlags2.AccelerationStructureReadBitKhr, 
            ImageLayout.Undefined, false),

        ResourceUsage.Present => new(
            PipelineStageFlags2.BottomOfPipeBit,
            AccessFlags2.None,
            ImageLayout.PresentSrcKhr, false),

        _ => throw new ArgumentOutOfRangeException(nameof(u), u, null)
    };
}