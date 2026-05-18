namespace CadThingo.VulkanEngine.ImGui;

/// <summary>
/// How the viewport ImGui.Image fits inside its dock-node when the image aspect
/// doesn't match the panel aspect. Identical-looking when AutoResolution is on
/// and the image is allocated at panel size — diverges only when the user locks
/// the render resolution.
/// </summary>
public enum ViewportFitMode
{
    Fill,       // Stretch to panel — ignores aspect.
    Letterbox,  // Scale uniformly to fit inside; pad with background on the short axis.
    Cover,      // Scale uniformly to cover the panel; crop overflow on the long axis.
}

/// <summary>
/// Cross-panel editor state. Static so any panel can read/write without plumbing
/// references through every Begin() call. SelectedEntity is the single source of
/// truth the Inspector binds to.
///
/// ViewportFocused / ViewportHovered are written by ViewportPanel each frame and
/// read by camera input gating (later — once the viewport is wired) so WASD/RMB
/// only drive the camera when the viewport panel has focus.
/// </summary>
public static unsafe class EditorState
{
    public static Entity* SelectedEntity;

    public static bool ViewportFocused;
    public static bool ViewportHovered;

    public static bool ShowViewport      = true;
    public static bool ShowSceneOutliner = true;
    public static bool ShowInspector     = true;
    public static bool ShowStats         = true;
    public static bool ShowCamera        = true;

    // ── Viewport panel state ─────────────────────────────────────
    public static ViewportFitMode ViewportFitMode    = ViewportFitMode.Fill;
    public static bool            ViewportAutoResolution = true;
    public static bool            ViewportFullscreen     = false;

    /// <summary>
    /// Render-target resize request posted by ViewportPanel. Consumed at the top
    /// of Renderer.DrawFrame, which then calls ResizeRenderTargets and clears it.
    /// Nullable so the renderer can early-out when no resize is pending.
    /// </summary>
    public static (uint w, uint h)? RequestedRenderExtent;

    // ── Scene Outliner state ─────────────────────────────────────
    /// <summary>
    /// When on, the Scene Outliner skips entities that carry neither a
    /// MeshComponent nor a LightComponent — i.e. pure transform-only
    /// intermediates from nested glTF imports. Their children still render at
    /// the parent's indent level, so meaningful descendants don't disappear.
    /// </summary>
    public static bool HideEmptyEntities = false;

    // ── Renderer settings ────────────────────────────────────────
    /// <summary>
    /// Toggles the skybox draw between the lighting and transparent passes.
    /// When off, sky pixels show the lighting pass's black clear instead of
    /// the env cube. Phase 5 surfaces this as a checkbox in the Renderer
    /// Settings panel.
    /// </summary>
    public static bool SkyboxEnabled = true;
}
