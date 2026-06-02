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

    public ImageResource Depth          { get; private set; } = null!;
    public ImageResource GBufferPosition { get; private set; } = null!;
    public ImageResource GBufferNormal   { get; private set; } = null!;
    public ImageResource GBufferAlbedo   { get; private set; } = null!;
    public ImageResource GBufferMaterial { get; private set; } = null!;
    public ImageResource GBufferEmissive { get; private set; } = null!;
    public Sampler       GBufferSampler  { get; private set; }

    public ImageResource PtAccumulator { get; private set; } = null!;
    public ImageResource PtOutColor    { get; private set; } = null!;
    public ImageResource SelectionMask { get; private set; } = null!;

    public void SetExtent(Extent2D extent) => _extent = extent;

    /// <summary>First-time allocation at the current extent — depth + g-buffers + sampler
    /// + PT + selection.</summary>
    public void AllocateAll()
    {
        CreateDepthResources();
        CreateGBufferResources();
        CreateGBufferSampler();
        CreatePathTracingResources();
        CreateSelectionResources();
    }

    /// <summary>Resize path: disposes + recreates the size-dependent targets at the new
    /// extent. Keeps the g-buffer sampler (extent-independent). The caller disposes the
    /// render graph first (it holds references to depth + g-buffers) and rebuilds it after.</summary>
    public void ReallocateSizeDependent(Extent2D extent)
    {
        DisposeSizeDependent();
        _extent = extent;
        CreateDepthResources();
        CreateGBufferResources();
        CreatePathTracingResources();
        CreateSelectionResources();
    }

    private void DisposeSizeDependent()
    {
        Depth?.Dispose();
        GBufferPosition?.Dispose();
        GBufferNormal?.Dispose();
        GBufferAlbedo?.Dispose();
        GBufferMaterial?.Dispose();
        GBufferEmissive?.Dispose();
        PtAccumulator?.Dispose();
        PtOutColor?.Dispose();
        SelectionMask?.Dispose();
    }

    public void Dispose()
    {
        DisposeSizeDependent();
        if (GBufferSampler.Handle != 0) vk.DestroySampler(_gfx.Device, GBufferSampler, null);
    }

    private void CreateDepthResources()
    {
        // Depth buffer matches the render target extent — sampled / depth-tested
        // by the deferred passes which all run at renderExtent. ImageResource object
        // only; the render graph allocates the VkImage inside Compile().
        var width = _extent.Width;
        var height = _extent.Height;

        Depth = new ImageResource(vk, _gfx.Device, "Depth", Format.D32Sfloat, new Extent2D(width, height),
            ImageUsageFlags.DepthStencilAttachmentBit | ImageUsageFlags.InputAttachmentBit,
            ImageLayout.Undefined, ImageLayout.DepthStencilAttachmentOptimal);
    }

    private void CreateGBufferResources()
    {
        // G-buffers + lighting all run at the render extent (== panel size in
        // editor mode), not necessarily the swapchain size.
        var width = _extent.Width;
        var height = _extent.Height;
        // SampledBit required: the lighting pass samples these via CombinedImageSampler descriptors.
        const ImageUsageFlags gBufferUsage =
            ImageUsageFlags.ColorAttachmentBit | ImageUsageFlags.InputAttachmentBit | ImageUsageFlags.SampledBit;
        GBufferPosition = new ImageResource(vk, _gfx.Device, "GBuffer_Position", Format.R32G32B32A32Sfloat,
            new Extent2D(width, height), gBufferUsage,
            ImageLayout.Undefined, ImageLayout.ShaderReadOnlyOptimal);
        GBufferNormal = new ImageResource(vk, _gfx.Device, "GBuffer_Normal", Format.R32G32B32A32Sfloat,
            new Extent2D(width, height), gBufferUsage,
            ImageLayout.Undefined, ImageLayout.ShaderReadOnlyOptimal);
        GBufferAlbedo = new ImageResource(vk, _gfx.Device, "GBuffer_Albedo", Format.R8G8B8A8Unorm,
            new Extent2D(width, height), gBufferUsage,
            ImageLayout.Undefined, ImageLayout.ShaderReadOnlyOptimal);
        GBufferMaterial = new ImageResource(vk, _gfx.Device, "GBuffer_Material", Format.R8G8B8A8Unorm,
            new Extent2D(width, height), gBufferUsage,
            ImageLayout.Undefined, ImageLayout.ShaderReadOnlyOptimal);
        // Emissive G-buffer: HDR range. Geometry pass writes emissiveSampler.rgb;
        // lighting pass adds it to the final color before tone-mapping.
        GBufferEmissive = new ImageResource(vk, _gfx.Device, "GBuffer_Emissive", Format.R16G16B16A16Sfloat,
            new Extent2D(width, height), gBufferUsage,
            ImageLayout.Undefined, ImageLayout.ShaderReadOnlyOptimal);
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
        // outside the graph, so allocate explicitly.
        PtAccumulator.Allocate(_gfx.PhysicalDevice);
        PtOutColor.Allocate(_gfx.PhysicalDevice);

        // One-shot Undefined → General transition so the very first dispatch can
        // do imageStore without a per-frame "first use" branch. Subsequent
        // dispatches keep both images in General between frames.
        var cmd = _gfx.BeginSingleTimeCommands();
        _gfx.TransitionImageLayout(cmd, PtAccumulator.Image, PtAccumulator._format,
            ImageLayout.Undefined, ImageLayout.General);
        _gfx.TransitionImageLayout(cmd, PtOutColor.Image, PtOutColor._format,
            ImageLayout.Undefined, ImageLayout.General);
        _gfx.EndSingleTimeCommands(cmd);
    }

    /// <summary>
    /// Allocates the R32F selection-coverage mask used by the outline overlay.
    /// Storage (compute write) + sampled (outline fragment read). Left in
    /// ShaderReadOnly between frames — RecordSelectionOutline flips it to General
    /// and back each active frame, and that flip doubles as the cross-frame
    /// write-after-read guard on the single shared image.
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

        // Settle in ShaderReadOnly so the per-frame block's opening
        // ShaderReadOnly→General transition has a valid source layout.
        var cmd = _gfx.BeginSingleTimeCommands();
        _gfx.TransitionImageLayout(cmd, SelectionMask.Image, SelectionMask._format,
            ImageLayout.Undefined, ImageLayout.General);
        _gfx.TransitionImageLayout(cmd, SelectionMask.Image, SelectionMask._format,
            ImageLayout.General, ImageLayout.ShaderReadOnlyOptimal);
        _gfx.EndSingleTimeCommands(cmd);
    }
}