using System.Numerics;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Features.Selection;

/// <summary>
/// Host-owned editor selection. Bundles the three ray-query pipelines behind
/// picking + the outline overlay:
///   PickPipeline          one-ray-per-click compute pick (returns an entity)
///   SelectionMaskPipeline ray-query coverage mask of the selected entity
///   OutlinePipeline       composites an outer ring around that mask into FinalColor
/// The coverage mask image itself lives on <see cref="RenderTargets"/> (resized
/// with the swapchain); this system only binds it. All three pipelines borrow
/// the renderer-owned TLAS + ShadowEntityInfo table, wired via
/// <see cref="WriteTlasDescriptor"/> / <see cref="WriteEntityInfoDescriptor"/>
/// once InitRayQuery has built them.
/// </summary>
public unsafe sealed class SelectionSystem : IDisposable
{
    private readonly Renderer _renderer;

    internal PickPipeline          pickPipeline;
    internal SelectionMaskPipeline selectionMaskPipeline;
    internal OutlinePipeline       outlinePipeline;

    public SelectionSystem(Renderer renderer)
    {
        _renderer = renderer;

        // Object picking owns only a tiny result SSBO; the TLAS is bound later.
        pickPipeline = new PickPipeline(renderer);
        pickPipeline.Initialize();

        // Mask pipeline writes the coverage image; outline reads it back.
        selectionMaskPipeline = new SelectionMaskPipeline(renderer);
        selectionMaskPipeline.Initialize();
        selectionMaskPipeline.WriteMaskImageDescriptor(Mask.ImageView);

        outlinePipeline = new OutlinePipeline(renderer);
        outlinePipeline.Initialize();
        outlinePipeline.WriteMaskDescriptor(Mask.ImageView);
    }

    // The coverage mask is a RenderTargets-owned, size-dependent image.
    private ImageResource Mask => _renderer.renderTargets.SelectionMask;

    /// Binds the TLAS into both ray-query pipelines. Called after InitRayQuery
    /// builds (or rebuilds) the acceleration structure.
    public void WriteTlasDescriptor(AccelerationStructureKHR tlas)
    {
        pickPipeline.WriteTlasDescriptor(tlas);
        selectionMaskPipeline.WriteTlasDescriptor(tlas);
    }

    /// Binds the flat ShadowEntityInfo table both pipelines resolve hits through
    /// (InstanceCustomIndex + GeometryIndex -> entity slot).
    public void WriteEntityInfoDescriptor()
    {
        pickPipeline.WriteEntityInfoDescriptor();
        selectionMaskPipeline.WriteEntityInfoDescriptor();
    }

    /// Re-points the mask descriptors after a swapchain resize rebuilt the view
    /// (storage side on the compute pipeline, sampled side on the outline pass).
    public void RebindMask()
    {
        selectionMaskPipeline.WriteMaskImageDescriptor(Mask.ImageView);
        outlinePipeline.WriteMaskDescriptor(Mask.ImageView);
    }

    /// <summary>
    /// Consumes a pending viewport pick (posted by ViewportPanel as a render-
    /// target pixel) and resolves it to an entity by casting one ray through the
    /// TLAS in a compute dispatch. The pick pass returns the hit's
    /// ShadowEntityInfo.EntityIndex - a stable RenderableHandle slot (see
    /// RebuildTlas) - which <see cref="GpuScene.ResolveSlot"/> maps back to the
    /// entity. Runs as a self-contained single-time submit (QueueWaitIdle), so the
    /// host-visible result is ready immediately. No-op unless ray queries are
    /// supported and a TLAS exists.
    /// </summary>
    public void ProcessPickRequest()
    {
        var req = ImGui.EditorState.RequestedPick;
        if (!req.HasValue) return;
        ImGui.EditorState.RequestedPick = null;

        if (!_renderer.RayInfraReady) return;

        var extent = _renderer.renderExtent;
        var camera = _renderer.Camera;

        uint px = req.Value.x;
        uint py = req.Value.y;
        if (px >= extent.Width || py >= extent.Height) return;

        // Same Y-flipped projection the geometry / lighting / light-cull passes
        // use, so the pick ray lines up with the rasterized image (and the PT
        // image, which flips the same way).
        Matrix4x4 view = camera.GetViewMatrix();
        Matrix4x4 proj = camera.GetProjectionMatrix(
            (float)extent.Width / extent.Height, 0.1f, 100.0f);
        proj.M22 *= -1f;
        if (!Matrix4x4.Invert(view * proj, out Matrix4x4 invVP)) return;

        var cmd = _renderer.BeginSingleTimeCommands();
        pickPipeline.Record(cmd, invVP, camera.GetPosition(),
            new Vector2(extent.Width, extent.Height), px, py);
        _renderer.EndSingleTimeCommands(cmd);   // QueueWaitIdle - result buffer is now valid

        uint idx = pickPipeline.ReadResult();
        ImGui.EditorState.SelectedEntity =
            idx == PickPipeline.PickNone ? null : _renderer.gpuScene.ResolveSlot(idx);
    }

    /// <summary>
    /// Composites the selection outline into FinalColor. Runs after the active
    /// render mode has produced FinalColor (left in ShaderReadOnly by both the
    /// deferred and PT paths) and before the swapchain blit, so one insertion
    /// point covers every mode. Ray-queries the TLAS into the coverage mask,
    /// then draws an outer ring around the selected entity's silhouette. No-op
    /// unless a mesh-bearing entity is selected and ray queries are available;
    /// leaves FinalColor in ShaderReadOnly exactly as it found it.
    /// </summary>
    public void RecordOutline(CommandBuffer cmd)
    {
        if (ImGui.EditorState.SelectedEntity == null) return;
        if (!_renderer.RayInfraReady) return;

        // Resolve the selection to its RenderableHandle slot - the same token the
        // mask shader compares against ShadowEntityInfo.EntityIndex. No handle
        // => not a renderable (e.g. a light picked in the outliner) => no
        // outline, which is correct.
        if (!_renderer.gpuScene.TryGetHandle(ImGui.EditorState.SelectedEntity, out var selHandle)) return;
        uint idx = selHandle.Index;

        var extent = _renderer.renderExtent;
        var camera = _renderer.Camera;

        Matrix4x4 view = camera.GetViewMatrix();
        Matrix4x4 proj = camera.GetProjectionMatrix(
            (float)extent.Width / extent.Height, 0.1f, 100.0f);
        proj.M22 *= -1f;
        if (!Matrix4x4.Invert(view * proj, out Matrix4x4 invVP)) return;

        var mask = Mask;

        // Mask sits in ShaderReadOnly between frames. Flip to General before the
        // compute write; this fragment-read->compute-write barrier also serializes
        // the previous frame's outline read against this frame's overwrite.
        _renderer.TransitionImageLayout(cmd, mask.Image, mask._format,
            ImageLayout.ShaderReadOnlyOptimal, ImageLayout.General);

        selectionMaskPipeline.Record(cmd, invVP, camera.GetPosition(), extent, idx);

        // compute write -> outline fragment read
        _renderer.TransitionImageLayout(cmd, mask.Image, mask._format,
            ImageLayout.General, ImageLayout.ShaderReadOnlyOptimal);

        var finalColor = _renderer.renderTargets.FinalColor;
        _renderer.TransitionImageLayout(cmd, finalColor.Image, finalColor._format,
            ImageLayout.ShaderReadOnlyOptimal, ImageLayout.ColorAttachmentOptimal);

        outlinePipeline.Record(cmd, extent, finalColor.ImageView);

        // Back to ShaderReadOnly - exactly where the swapchain blit and the ImGui
        // viewport sampler expect FinalColor to be.
        _renderer.TransitionImageLayout(cmd, finalColor.Image, finalColor._format,
            ImageLayout.ColorAttachmentOptimal, ImageLayout.ShaderReadOnlyOptimal);
    }

    public void Dispose()
    {
        outlinePipeline?.Dispose();
        selectionMaskPipeline?.Dispose();
        pickPipeline?.Dispose();
    }
}