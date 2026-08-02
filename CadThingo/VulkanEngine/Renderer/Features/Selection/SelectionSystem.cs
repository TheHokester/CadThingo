using System.Numerics;
using System.Runtime.CompilerServices;
using CadThingo.VulkanEngine.Renderer.FeatureLifecycle;
using CadThingo.VulkanEngine.Renderer.FeatureLifecycle.RenderFeatureInterfaces;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Features.Selection;

/// <summary>
/// Editor selection. Bundles the three ray-query pipelines behind picking + the outline overlay:
///   PickPipeline          one-ray-per-click compute pick (returns an entity)
///   SelectionMaskPipeline ray-query coverage mask of the selected entity
///   OutlinePipeline       composites an outer ring around that mask into FinalColor
/// The coverage mask is this system's own image - nothing outside these three pipelines reads it.
/// The pick + mask pipelines read the TLAS + ShadowEntityInfo table from the registry-owned scene
/// set (sceneTlas / sceneEntityInfo), so no per-pipeline TLAS/entity-info fan-out is needed.
///
/// Three phases: <see cref="PreDraw"/> resolves a pending click out-of-band, <see cref="PostDraw"/>
/// composites the outline once the active core has produced FinalColor, and <see cref="Resize"/>
/// reallocates the mask and re-points its descriptors. Order 100 puts it after every core, which is
/// what makes "after the frame's image exists" true for PostDraw without the host sequencing it.
/// </summary>
public unsafe sealed class SelectionSystem
    : IPreDrawFeature, IPostDrawFeature, IResizeFeature,
      ISelfRegisteringFeature<SelectionSystem>, INeedsGpu, INeedsHost
{
    public static FeatureDesc Desc =>
        new(Order: 100, Gate: _ => true, Make: () => new SelectionSystem());

    [ModuleInitializer]
    internal static void _Reg() => FeatureCatalog.Register<SelectionSystem>();

    public string Name => "Selection (pick + outline)";

    private GpuContext _gpu;
    private Renderer   _host = null!;
    GpuContext INeedsGpu.Gpu   { set => _gpu  = value; }
    Renderer   INeedsHost.Host { set => _host = value; }

    // Last snapshot the host handed over, re-read on every Resize.
    private HostTargets _targets;

    private GraphicsDevice Gfx => _gpu.Gfx;

    internal PickPipeline          pickPipeline          = null!;
    internal SelectionMaskPipeline selectionMaskPipeline = null!;
    internal OutlinePipeline       outlinePipeline       = null!;

    // Selection changes arrive as events, from here (the viewport pick) and from the panels
    // (outliner click, probe spawn). This system applying them is what keeps
    // EditorState.SelectedEntity single-writer - publishers state the intent and nothing else.
    private IDisposable _selectionSub = null!;

    public void Initialize()
    {
        _selectionSub = Engine.EventBus.Subscribe<SceneEntitySelectedEvent>(OnEntitySelected);

        // Object picking owns only a tiny result SSBO; the TLAS is bound later.
        pickPipeline = new PickPipeline(_gpu, _host);
        pickPipeline.Initialize();

        // Mask pipeline writes the coverage image; outline reads it back. Both mask descriptors
        // point at an extent-sized view, so they are written in Resize (primed once at boot).
        selectionMaskPipeline = new SelectionMaskPipeline(_gpu, _host);
        selectionMaskPipeline.Initialize();

        outlinePipeline = new OutlinePipeline(_gpu, _host);
        outlinePipeline.Initialize();
    }

    // R32F coverage mask, sized to the render extent. The mask compute pass writes it, the outline
    // pass samples it, and nothing else reads it, so this system owns it.
    private ImageResource _mask = null!;
    private ImageResource Mask => _mask;

    /// <summary>Reallocates the coverage mask at the new extent and points both mask descriptors at
    /// the fresh view: storage side on the compute pipeline, sampled side on the outline
    /// pass.</summary>
    public void Resize(in HostTargets targets)
    {
        _targets = targets;

        _mask?.Dispose();
        _mask = new ImageResource(Gfx.Vk, Gfx.Device, "selectionMask", Format.R32Sfloat,
            targets.Extent,
            ImageUsageFlags.StorageBit | ImageUsageFlags.SampledBit,
            ImageLayout.Undefined, ImageLayout.General);
        _mask.Allocate(Gfx.PhysicalDevice);

        // Settles in ShaderReadOnly so PostDraw's opening ShaderReadOnly->General transition has a
        // valid source layout on the first frame.
        var cmd = Gfx.BeginSingleTimeCommands();
        Gfx.TransitionImageLayout(cmd, _mask.Image, _mask._format,
            ImageLayout.Undefined, ImageLayout.General);
        Gfx.TransitionImageLayout(cmd, _mask.Image, _mask._format,
            ImageLayout.General, ImageLayout.ShaderReadOnlyOptimal);
        Gfx.EndSingleTimeCommands(cmd);

        selectionMaskPipeline.WriteMaskImageDescriptor(_mask.ImageView);
        outlinePipeline.WriteMaskDescriptor(_mask.ImageView);
    }

    /// <summary>
    /// Consumes a pending viewport pick (posted by ViewportPanel as a render-target pixel) and
    /// resolves it to an entity by casting one ray through the TLAS in a compute dispatch. The pick
    /// pass returns the hit's ShadowEntityInfo.EntityIndex - a stable RenderableHandle slot (see
    /// RebuildTlas) - which <see cref="GpuScene.ResolveSlot"/> maps back to the entity. Runs as a
    /// self-contained single-time submit (QueueWaitIdle), so the host-visible result is ready
    /// immediately, and so this belongs in PreDraw rather than in the frame's command buffer.
    /// No-op unless ray queries are supported and a TLAS exists.
    /// </summary>
    public void PreDraw(in RenderView view)
    {
        var req = ImGui.EditorState.RequestedPick;
        if (!req.HasValue) return;
        ImGui.EditorState.RequestedPick = null;

        if (!_host.RayInfraReady) return;

        var extent = view.RenderExtent;
        var camera = view.Camera;

        uint px = req.Value.x;
        uint py = req.Value.y;
        if (px >= extent.Width || py >= extent.Height) return;

        // Same Y-flipped projection the geometry / lighting / light-cull passes
        // use, so the pick ray lines up with the rasterized image (and the PT
        // image, which flips the same way).
        Matrix4x4 viewMat = camera.GetViewMatrix();
        Matrix4x4 proj = camera.GetProjectionMatrix(
            (float)extent.Width / extent.Height, 0.1f, 100.0f);
        proj.M22 *= -1f;
        if (!Matrix4x4.Invert(viewMat * proj, out Matrix4x4 invVP)) return;

        var cmd = Gfx.BeginSingleTimeCommands();
        pickPipeline.Record(cmd, invVP, camera.GetPosition(),
            view, px, py);
        Gfx.EndSingleTimeCommands(cmd);   // QueueWaitIdle - result buffer is now valid

        uint idx = pickPipeline.ReadResult();
        Engine.EventBus.PublishEvent(new SceneEntitySelectedEvent(
            idx == PickPipeline.PickNone ? null : _host.gpuScene.ResolveSlot(idx)));
    }

    /// <summary>
    /// Applies a selection change. Editor-category events deliver immediately, so the store is
    /// updated before the publishing panel's next line runs and the outliner highlight lands on
    /// the same frame as the click. The accumulator restart is because the outline composites
    /// into FinalColor - a different selection is a different image.
    /// </summary>
    private void OnEntitySelected(SceneEntitySelectedEvent e)
    {
        if (ImGui.EditorState.SelectedEntity == e.GetEntity) return;

        ImGui.EditorState.SelectedEntity = e.GetEntity;
        _host.MarkAccumulatorDirty();
    }

    /// <summary>
    /// Composites the selection outline into FinalColor. The PostDraw phase runs after the active
    /// render mode has produced FinalColor (left in ShaderReadOnly by both the deferred and PT
    /// paths) and before the swapchain blit, so one phase covers every mode without the host
    /// naming this system. Ray-queries the TLAS into the coverage mask, then draws an outer ring
    /// around the selected entity's silhouette. No-op unless a mesh-bearing entity is selected and
    /// ray queries are available; leaves FinalColor in ShaderReadOnly exactly as it found it.
    /// </summary>
    public void PostDraw(CommandBuffer cmd, in RenderView view)
    {
        if (ImGui.EditorState.SelectedEntity == null) return;
        if (!_host.RayInfraReady) return;

        // Resolve the selection to its RenderableHandle slot - the same token the
        // mask shader compares against ShadowEntityInfo.EntityIndex. No handle
        // => not a renderable (e.g. a light picked in the outliner) => no
        // outline, which is correct.
        if (!_host.gpuScene.TryGetHandle(ImGui.EditorState.SelectedEntity, out var selHandle)) return;
        uint idx = selHandle.Index;

        var extent = view.RenderExtent;
        var camera = view.Camera;

        Matrix4x4 viewMat = camera.GetViewMatrix();
        Matrix4x4 proj = camera.GetProjectionMatrix(
            (float)extent.Width / extent.Height, 0.1f, 100.0f);
        proj.M22 *= -1f;
        if (!Matrix4x4.Invert(viewMat * proj, out Matrix4x4 invVP)) return;

        var mask = Mask;

        // Mask sits in ShaderReadOnly between frames. Flip to General before the
        // compute write; this fragment-read->compute-write barrier also serializes
        // the previous frame's outline read against this frame's overwrite.
        Gfx.TransitionImageLayout(cmd, mask.Image, mask._format,
            ImageLayout.ShaderReadOnlyOptimal, ImageLayout.General);

        selectionMaskPipeline.Record(cmd, invVP, view, idx);

        // compute write -> outline fragment read
        Gfx.TransitionImageLayout(cmd, mask.Image, mask._format,
            ImageLayout.General, ImageLayout.ShaderReadOnlyOptimal);

        var finalColor = _targets.FinalColor;
        Gfx.TransitionImageLayout(cmd, finalColor.Image, finalColor._format,
            ImageLayout.ShaderReadOnlyOptimal, ImageLayout.ColorAttachmentOptimal);

        outlinePipeline.Record(cmd, extent, finalColor.ImageView);

        // Back to ShaderReadOnly - exactly where the swapchain blit and the ImGui
        // viewport sampler expect FinalColor to be.
        Gfx.TransitionImageLayout(cmd, finalColor.Image, finalColor._format,
            ImageLayout.ColorAttachmentOptimal, ImageLayout.ShaderReadOnlyOptimal);
    }

    public void Dispose()
    {
        _selectionSub?.Dispose();
        outlinePipeline?.Dispose();
        selectionMaskPipeline?.Dispose();
        pickPipeline?.Dispose();
        _mask?.Dispose();
    }
}
