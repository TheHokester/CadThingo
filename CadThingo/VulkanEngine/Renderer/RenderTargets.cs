using Silk.NET.Vulkan;
// ReSharper disable InconsistentNaming

namespace CadThingo.VulkanEngine.Renderer;

/// <summary>
/// Owns the render-target attachments sized to the render extent: the depth buffer,
/// the five deferred g-buffers + their shared sampler, and the path-tracer storage
/// images (accumulator / out-color) + the selection coverage mask.
///
/// L1.5 of the renderer refactor — the interim home for these targets. Depth + the
/// g-buffers are <see cref="ImageResource"/> *objects only*; the render graph allocates
/// their VkImages inside Compile(). The PT / selection images live outside the graph,
/// so this allocates and lays them out directly. Depends only on <see cref="GraphicsDevice"/>.
///
/// The orchestrator (Renderer) keeps RebuildRenderTargets / ResizeRenderTargets — those
/// re-issue cross-pipeline descriptor writes — but delegates the realloc here via
/// <see cref="ReallocateSizeDependent"/>. The g-buffer sampler survives a resize.
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

    //render targets now only keeys ownership of shared gbuffer sampler 
    public Sampler       GBufferSampler  { get; private set; }

    public ImageResource PtAccumulator { get; private set; } = null!;
    public ImageResource PtOutColor    { get; private set; } = null!;
    public ImageResource SelectionMask { get; private set; } = null!;
    public ImageResource FinalColor    { get; private set; } = null!;

    public void SetExtent(Extent2D extent) => _extent = extent;

    /// <summary>First-time allocation at the current extent — g-buffer sampler + PT +
    /// selection + FinalColor. (The g-buffers / depth / HDR live on the FrameGraph now.)</summary>
    public void AllocateAll()
    {
        CreateGBufferSampler();
        CreatePathTracingResources();
        CreateSelectionResources();
        CreateFinalColorResources();
        TransitionSizeDependentImages();   // single batched submit for all initial layouts
    }

    /// <summary>Resize path: disposes + recreates the size-dependent targets at the new
    /// extent. Keeps the g-buffer sampler (extent-independent). The caller rebuilds the
    /// deferred FrameGraph after this (it owns + reallocates the g-buffer/depth/HDR
    /// transients and re-imports the fresh FinalColor handle).</summary>
    public void ReallocateSizeDependent(Extent2D extent)
    {
        DisposeSizeDependent();
        _extent = extent;
        CreatePathTracingResources();
        CreateSelectionResources();
        CreateFinalColorResources();
        TransitionSizeDependentImages();   // one drain instead of one per image group
    }

    private void DisposeSizeDependent()
    {
        PtAccumulator?.Dispose();
        PtOutColor?.Dispose();
        SelectionMask?.Dispose();
        FinalColor?.Dispose();
    }

    public void Dispose()
    {
        DisposeSizeDependent();
        if (GBufferSampler.Handle != 0) vk.DestroySampler(_gfx.Device, GBufferSampler, null);
    }

    /// <summary>
    /// Allocates FinalColor - the LDR target the FrameGraph's tonemap pass writes,
    /// the per-frame swapchain blit sources, and the ImGui viewport panel samples. The graph
    /// IMPORTS it (it doesn't own it) because consumers outside the graph need a stable
    /// handle; TransferSrc/Dst cover the blit (and the PT path blitting ptOutColor in).
    /// Left Undefined here — the graph discards + fully regenerates it each frame and hands
    /// it back in ShaderReadOnly.
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

    private void CreatePathTracingResources()
    {
        var width = _extent.Width;
        var height = _extent.Height;

        // ptOutColor needs TransferSrcBit so DrawPathtraced can blit it into
        // FinalColor (the viewport's source) at the end of the dispatch.
        PtAccumulator = new ImageResource(vk, _gfx.Device, "accumulator", Format.R32G32B32A32Sfloat,
            new Extent2D(width, height),
            ImageUsageFlags.StorageBit | ImageUsageFlags.SampledBit,
            ImageLayout.Undefined, ImageLayout.General);

        PtOutColor = new ImageResource(vk, _gfx.Device, "outColor", Format.R32G32B32A32Sfloat,
            new Extent2D(width, height),
            ImageUsageFlags.StorageBit | ImageUsageFlags.SampledBit | ImageUsageFlags.TransferSrcBit,
            ImageLayout.Undefined, ImageLayout.General);

        // Render graph normally calls Allocate inside Compile — these images live
        // outside the graph, so allocate explicitly. The Undefined → General transition
        // is batched with the selection mask's in TransitionSizeDependentImages.
        // High residency priority: the accumulator + out-color are the path tracer's hot
        // working set, touched every frame - keep them resident ahead of cold resources
        // when the process is over its WDDM budget.
        PtAccumulator.Allocate(_gfx.PhysicalDevice, GpuMemoryAllocator.PriorityHigh);
        PtOutColor.Allocate(_gfx.PhysicalDevice, GpuMemoryAllocator.PriorityHigh);
    }

    /// <summary>
    /// Allocates the R32F selection-coverage mask used by the outline overlay.
    /// </summary>
    private void CreateSelectionResources()
    {
        var width = _extent.Width;
        var height = _extent.Height;

        SelectionMask = new ImageResource(vk, _gfx.Device, "selectionMask", Format.R32Sfloat,
            new Extent2D(width, height),
            ImageUsageFlags.StorageBit | ImageUsageFlags.SampledBit,
            ImageLayout.Undefined, ImageLayout.General);
        SelectionMask.Allocate(_gfx.PhysicalDevice);
        // Settles in ShaderReadOnly (see TransitionSizeDependentImages) so the per-frame
        // block's opening ShaderReadOnly→General transition has a valid source layout.
    }

    /// <summary>
    /// One batched single-time submit that lays out every freshly-allocated size-dependent
    /// image into the layout its first use expects: the PT accumulator/out-color into General
    /// (so the first dispatch can imageStore with no first-use branch), and the selection mask
    /// into ShaderReadOnly. Replaces the previous per-group submits — one GPU drain per resize
    /// instead of two — so dragging the viewport hitches less. FinalColor needs no transition
    /// (the FrameGraph imports it Undefined and regenerates it every frame).
    /// </summary>
    private void TransitionSizeDependentImages()
    {
        var cmd = _gfx.BeginSingleTimeCommands();
        _gfx.TransitionImageLayout(cmd, PtAccumulator.Image, PtAccumulator._format,
            ImageLayout.Undefined, ImageLayout.General);
        _gfx.TransitionImageLayout(cmd, PtOutColor.Image, PtOutColor._format,
            ImageLayout.Undefined, ImageLayout.General);
        _gfx.TransitionImageLayout(cmd, SelectionMask.Image, SelectionMask._format,
            ImageLayout.Undefined, ImageLayout.General);
        _gfx.TransitionImageLayout(cmd, SelectionMask.Image, SelectionMask._format,
            ImageLayout.General, ImageLayout.ShaderReadOnlyOptimal);
        _gfx.EndSingleTimeCommands(cmd);
    }
}