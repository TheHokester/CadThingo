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

    public static bool ShowViewport         = true;
    public static bool ShowSceneOutliner    = true;
    public static bool ShowInspector        = true;
    public static bool ShowStats            = true;
    public static bool ShowCamera           = true;
    public static bool ShowRendererSettings = true;
    public static bool ShowFileBrowser      = true;

    //  Viewport panel state 
    public static ViewportFitMode ViewportFitMode    = ViewportFitMode.Fill;
    public static bool            ViewportAutoResolution = true;
    public static bool            ViewportFullscreen     = false;

    /// <summary>
    /// Render-target resize request posted by ViewportPanel. Consumed at the top
    /// of Renderer.DrawFrame, which then calls ResizeRenderTargets and clears it.
    /// Nullable so the renderer can early-out when no resize is pending.
    /// </summary>
    public static (uint w, uint h)? RequestedRenderExtent;

    /// <summary>
    /// Object-pick request posted by ViewportPanel on a left-click, in render-
    /// target pixel coordinates (top-left origin). Consumed by
    /// Renderer.ProcessPickRequest at the top of DrawFrame, which ray-queries
    /// the TLAS and updates <see cref="SelectedEntity"/>. Null when no click is
    /// pending.
    /// </summary>
    public static (uint x, uint y)? RequestedPick;

    // Scene Outliner state 
    /// <summary>
    /// When on, the Scene Outliner skips entities that carry neither a
    /// MeshComponent nor a LightComponent — i.e. pure transform-only
    /// intermediates from nested glTF imports. Their children still render at
    /// the parent's indent level, so meaningful descendants don't disappear.
    /// </summary>
    public static bool HideEmptyEntities = false;

    //  Renderer settings 
    /// <summary>
    /// Toggles the skybox draw between the lighting and transparent passes.
    /// When off, sky pixels show the lighting clear (BackgroundColor) instead
    /// of the env cube.
    /// </summary>
    public static bool SkyboxEnabled = true;

    /// <summary>
    /// Multiplier on the env-cube sample written by the skybox pass. Useful
    /// when the HDR is too bright for a comfortable backdrop but you still
    /// want IBL contributing at full strength to surface lighting.
    /// </summary>
    public static float SkyboxIntensity = 1.0f;

    /// <summary>
    /// scaleIBLAmbient — multiplier on the IBL ambient term in both the PBR
    /// deferred and transparent forward+ shaders. The studio HDRIs that ship
    /// with this project are quite bright; default tuned to land in the same
    /// rough range as the pre-IBL flat ambient floor.
    /// </summary>
    public static float IblIntensity = 0.3f;

    /// <summary>
    /// Used as the lighting-pass clear color (the sky behind the scene when
    /// the skybox is disabled). RGB only; alpha is always 1.
    /// </summary>
    public static System.Numerics.Vector3 BackgroundColor =
        new System.Numerics.Vector3(0.0f, 0.0f, 0.0f);
}
