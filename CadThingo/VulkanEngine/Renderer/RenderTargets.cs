using Silk.NET.Vulkan;
// ReSharper disable InconsistentNaming

namespace CadThingo.VulkanEngine.Renderer;

/// <summary>
/// What the host owns of the render targets: FinalColor and the shared g-buffer sampler. Each
/// technique-private target sits on the feature that uses it. The g-buffers / depth / HDR are graph
/// transients, the accumulator / out-color pair belongs to
/// <see cref="Features.PathTracer.PathTracingSystem"/>, and the coverage mask to
/// <see cref="Features.Selection.SelectionSystem"/>.
///
/// FinalColor stays here because it is consumed outside every feature: the swapchain blit and the
/// ImGui viewport panel both sample it. Depends only on <see cref="GraphicsDevice"/>.
///
/// The orchestrator (Renderer) keeps RebuildRenderTargets / ResizeRenderTargets, which pump the
/// feature resize, and delegates the realloc here via <see cref="ReallocateSizeDependent"/>. The
/// g-buffer sampler survives a resize.
/// </summary>
public sealed unsafe class RenderTargets : IDisposable
{
    private readonly GraphicsDevice _gfx;
    private readonly Vk vk;
    private Extent2D _extent;

    public RenderTargets(GraphicsDevice gfx)
    {
        _gfx = gfx;
        vk = gfx.Vk;
    }

    public Extent2D RenderExtent => _extent;

    /// <summary>The current extent + FinalColor, as handed to the features on every resize.</summary>
    public HostTargets Snapshot => new(_extent, FinalColor);

    // Extent-independent, so a resize leaves it alone. Every deferred g-buffer read binds it.
    public Sampler       GBufferSampler  { get; private set; }

    public ImageResource FinalColor    { get; private set; } = null!;

    public void SetExtent(Extent2D extent) => _extent = extent;

    /// <summary>First-time allocation at the current extent: g-buffer sampler + FinalColor.</summary>
    public void AllocateAll()
    {
        CreateGBufferSampler();
        CreateFinalColorResources();
    }

    /// <summary>Resize path: disposes + recreates FinalColor at the new extent. Keeps the g-buffer
    /// sampler (extent-independent). The caller pumps the feature resize after this, which is where
    /// every technique-private target is reallocated and the fresh FinalColor re-imported.</summary>
    public void ReallocateSizeDependent(Extent2D extent)
    {
        DisposeSizeDependent();
        _extent = extent;
        CreateFinalColorResources();
    }

    private void DisposeSizeDependent()
    {
        FinalColor?.Dispose();
    }

    public void Dispose()
    {
        DisposeSizeDependent();
        if (GBufferSampler.Handle != 0) vk.DestroySampler(_gfx.Device, GBufferSampler, null);
    }

    /// <summary>
    /// Allocates FinalColor: the LDR target the FrameGraph's tonemap pass writes, the per-frame
    /// swapchain blit sources, and the ImGui viewport panel samples. The graph imports rather than
    /// owns it, because consumers outside the graph need a stable handle. TransferSrc/Dst cover the
    /// blit, including the PT path blitting out-color in. Left Undefined here, since the graph
    /// discards and regenerates it each frame and hands it back in ShaderReadOnly.
    /// </summary>
    private void CreateFinalColorResources()
    {
        FinalColor = new ImageResource(vk, _gfx.Device, "FinalColor", Format.R8G8B8A8Unorm,
            new Extent2D(_extent.Width, _extent.Height),
            ImageUsageFlags.ColorAttachmentBit | ImageUsageFlags.TransferSrcBit
            | ImageUsageFlags.TransferDstBit | ImageUsageFlags.SampledBit,
            ImageLayout.Undefined, ImageLayout.ShaderReadOnlyOptimal);
        // High priority: FinalColor is regenerated and sampled every frame (the viewport source).
        FinalColor.Allocate(_gfx.PhysicalDevice, GpuMemoryAllocator.PriorityHigh);
    }

    private void CreateGBufferSampler()
    {
        SamplerCreateInfo samplerInfo = new()
        {
            SType = StructureType.SamplerCreateInfo,
            MagFilter = Filter.Nearest,
            MinFilter = Filter.Nearest,
            AddressModeU = SamplerAddressMode.ClampToEdge,
            AddressModeV = SamplerAddressMode.ClampToEdge,
            AddressModeW = SamplerAddressMode.ClampToEdge,
            AnisotropyEnable = true,
            MaxAnisotropy = 16,
            BorderColor = BorderColor.FloatOpaqueBlack,
            UnnormalizedCoordinates = false,
            CompareEnable = false,
            CompareOp = CompareOp.Always,
            MipmapMode = SamplerMipmapMode.Nearest,
            MinLod = 0.0f,
            MaxLod = 1.0f,
            MipLodBias = 0.0f,
        };
        if (vk.CreateSampler(_gfx.Device, &samplerInfo, null, out var sampler) != Result.Success)
        {
            throw new Exception("Failed to create gBuffer sampler");
        }
        GBufferSampler = sampler;
    }

}