using System.Numerics;
using ImGuiNET;

namespace CadThingo.VulkanEngine.ImGui.Panels;

/// <summary>
/// Displays the renderer's FinalColor target as a docked (or fullscreen) ImGui
/// image. Drives the renderer's <c>renderExtent</c> via
/// <see cref="EditorState.RequestedRenderExtent"/> when AutoResolution is on,
/// debounced to avoid reallocating on every drag-resize frame.
///
/// In fullscreen mode the panel covers the whole main viewport; the rest of the
/// editor chrome (menu bar, other panels) is skipped by <c>ImGuiUI.Draw</c>.
/// </summary>
public static class ViewportPanel
{
    // Number of frames the panel size has to remain stable before we post a
    // resize request. Resizing rebuilds the entire deferred chain, so we don't
    // want to churn during a drag.
    const int ResizeStableFrames = 4;

    static (uint w, uint h) _lastSeenSize;
    static int              _framesSinceChange;

    public static void Draw()
    {
        if (!EditorState.ShowViewport && !EditorState.ViewportFullscreen) return;

        if (EditorState.ViewportFullscreen) DrawFullscreen();
        else                                DrawDocked();
    }

    static void DrawDocked()
    {
        // Edge-to-edge image: drop window padding so the scene reaches the dock
        // borders. The toolbar lives in a child window above the image.
        ImGuiNET.ImGui.PushStyleVar(ImGuiStyleVar.WindowPadding, Vector2.Zero);
        if (!ImGuiNET.ImGui.Begin("Viewport", ref EditorState.ShowViewport))
        {
            ImGuiNET.ImGui.End();
            ImGuiNET.ImGui.PopStyleVar();
            return;
        }

        EditorState.ViewportFocused = ImGuiNET.ImGui.IsWindowFocused();

        ImGuiNET.ImGui.PopStyleVar();  // pop before drawing the toolbar so it gets normal frame padding

        DrawToolbar();

        var imageArea = ImGuiNET.ImGui.GetContentRegionAvail();
        DrawImage(imageArea);

        ImGuiNET.ImGui.End();
    }

    static void DrawFullscreen()
    {
        var vp = ImGuiNET.ImGui.GetMainViewport();
        
        ImGuiNET.ImGui.SetNextWindowPos(vp.WorkPos);
        ImGuiNET.ImGui.SetNextWindowSize(vp.WorkSize);

        const ImGuiWindowFlags flags =
            ImGuiWindowFlags.NoDecoration |
            ImGuiWindowFlags.NoMove |
            ImGuiWindowFlags.NoResize |
            ImGuiWindowFlags.NoSavedSettings |
            ImGuiWindowFlags.NoBringToFrontOnFocus |
            ImGuiWindowFlags.NoNavFocus;

        ImGuiNET.ImGui.PushStyleVar(ImGuiStyleVar.WindowPadding, Vector2.Zero);
        bool open = true;
        if (!ImGuiNET.ImGui.Begin("ViewportFullscreen", ref open, flags))
        {
            ImGuiNET.ImGui.End();
            ImGuiNET.ImGui.PopStyleVar();
            return;
        }

        EditorState.ViewportFocused = ImGuiNET.ImGui.IsWindowFocused();

        ImGuiNET.ImGui.PopStyleVar();

        DrawToolbar();
        DrawImage(ImGuiNET.ImGui.GetContentRegionAvail());

        ImGuiNET.ImGui.End();
    }

    static void DrawToolbar()
    {
        // Compact toolbar: fit mode + auto-resolution + fullscreen.
        var pt = Engine.renderer.ptComputePipeline;
        
        ImGuiNET.ImGui.AlignTextToFramePadding();
        ImGuiNET.ImGui.Text("Fit");
        ImGuiNET.ImGui.SameLine();
        ImGuiNET.ImGui.SetNextItemWidth(110f);
        int mode = (int)EditorState.ViewportFitMode;
        if (ImGuiNET.ImGui.Combo("##fitmode", ref mode, "Fill\0Letterbox\0Cover\0\0"))
            EditorState.ViewportFitMode = (ViewportFitMode)mode;

        ImGuiNET.ImGui.SameLine();
        bool auto = EditorState.ViewportAutoResolution;
        if (ImGuiNET.ImGui.Checkbox("Auto-res", ref auto))
        {
            EditorState.ViewportAutoResolution = auto;
            // Force a re-request on next frame so the renderer catches up.
            _framesSinceChange = 0;
        }

        // Show the current render extent so the user knows what's actually
        // happening on the GPU side.
        var ext = Engine.renderer?.RenderExtent;
        if (ext.HasValue)
        {
            ImGuiNET.ImGui.SameLine();
            ImGuiNET.ImGui.TextDisabled($"{ext.Value.Width}x{ext.Value.Height}");
        }

        if (EditorState.ViewportFullscreen && Engine.renderer.renderMode == Renderer.Renderer.RenderMode.RayCompute)
        {
            ImGuiNET.ImGui.SameLine();
            ImGuiNET.ImGui.Text($"Samples accumulated: {pt.CurrentSampleCount}");
        }
        ImGuiNET.ImGui.SameLine(ImGuiNET.ImGui.GetWindowWidth() - 110f);
        string fsLabel = EditorState.ViewportFullscreen ? "Exit fullscreen" : "Fullscreen";
        if (ImGuiNET.ImGui.Button(fsLabel))
            EditorState.ViewportFullscreen = !EditorState.ViewportFullscreen;
    }

    static void DrawImage(Vector2 area)
    {
        EditorState.ViewportHovered = false;  // overwritten below only if the image actually draws

        var texId = Engine.renderer?.imGuiUtils?.ViewportTextureId ?? 0;
        if (area.X <= 1 || area.Y <= 1)
        {
            return;
        }

        uint panelW = (uint)MathF.Max(1, area.X);
        uint panelH = (uint)MathF.Max(1, area.Y);

        // Auto-resize: request the renderer reallocate FinalColor + gbuffers to
        // match panel size after a few stable frames.
        if (EditorState.ViewportAutoResolution)
            MaybeRequestResize(panelW, panelH);

        if (texId == 0)
        {
            DrawPlaceholder(area, "Viewport target not bound");
            return;
        }

        var renderExt = Engine.renderer!.RenderExtent;
        ComputeFit(area, new Vector2(renderExt.Width, renderExt.Height),
            EditorState.ViewportFitMode, out var imgPos, out var imgSize);

        // Offset cursor by imgPos to centre the image. SetCursorPos uses
        // window-relative coords; combine with the current cursor.
        var cursor = ImGuiNET.ImGui.GetCursorPos();
        ImGuiNET.ImGui.SetCursorPos(cursor + imgPos);
        ImGuiNET.ImGui.Image(texId, imgSize);
        // Image-specific hover so RMB-drag on the toolbar / menubar doesn't
        // start camera rotation — only clicks landing on the scene image do.
        EditorState.ViewportHovered = ImGuiNET.ImGui.IsItemHovered();

        // Left-click picks the entity under the cursor. Map the click from the
        // (possibly fit-scaled) on-screen image rect into render-target pixels;
        // the renderer ray-queries the TLAS for that pixel next frame. RMB is
        // camera look, so only LMB triggers a pick.
        if (EditorState.ViewportHovered &&
            ImGuiNET.ImGui.IsMouseClicked(ImGuiMouseButton.Left))
        {
            var rectMin  = ImGuiNET.ImGui.GetItemRectMin();
            var rectSize = ImGuiNET.ImGui.GetItemRectSize();
            if (rectSize.X > 0 && rectSize.Y > 0)
            {
                var mouse = ImGuiNET.ImGui.GetMousePos();
                float u = Math.Clamp((mouse.X - rectMin.X) / rectSize.X, 0f, 0.999999f);
                float v = Math.Clamp((mouse.Y - rectMin.Y) / rectSize.Y, 0f, 0.999999f);
                EditorState.RequestedPick =
                    ((uint)(u * renderExt.Width), (uint)(v * renderExt.Height));
            }
        }
    }

    static void MaybeRequestResize(uint w, uint h)
    {
        if ((w, h) == _lastSeenSize)
        {
            _framesSinceChange++;
        }
        else
        {
            _lastSeenSize = (w, h);
            _framesSinceChange = 0;
        }

        var rendererExt = Engine.renderer?.RenderExtent;
        if (!rendererExt.HasValue) return;

        bool differs = rendererExt.Value.Width != w || rendererExt.Value.Height != h;
        if (differs && _framesSinceChange >= ResizeStableFrames)
        {
            EditorState.RequestedRenderExtent = (w, h);
        }
    }

    /// <summary>
    /// Computes the on-screen image position + size inside <paramref name="panel"/>
    /// given source <paramref name="img"/> dimensions and a fit mode. Position is
    /// relative to the panel's upper-left (i.e. add to current cursor).
    /// </summary>
    static void ComputeFit(Vector2 panel, Vector2 img, ViewportFitMode mode,
        out Vector2 pos, out Vector2 size)
    {
        if (mode == ViewportFitMode.Fill || img.X <= 0 || img.Y <= 0)
        {
            pos  = Vector2.Zero;
            size = panel;
            return;
        }

        float panelAspect = panel.X / panel.Y;
        float imgAspect   = img.X   / img.Y;

        if (mode == ViewportFitMode.Letterbox)
        {
            // Scale uniformly to FIT inside the panel — the shorter axis is letterboxed.
            if (imgAspect > panelAspect)
            {
                size = new Vector2(panel.X, panel.X / imgAspect);
                pos  = new Vector2(0f, (panel.Y - size.Y) * 0.5f);
            }
            else
            {
                size = new Vector2(panel.Y * imgAspect, panel.Y);
                pos  = new Vector2((panel.X - size.X) * 0.5f, 0f);
            }
        }
        else // Cover
        {
            // Scale uniformly to COVER the panel — overflow is cropped via panel scissor.
            if (imgAspect > panelAspect)
            {
                size = new Vector2(panel.Y * imgAspect, panel.Y);
                pos  = new Vector2((panel.X - size.X) * 0.5f, 0f);
            }
            else
            {
                size = new Vector2(panel.X, panel.X / imgAspect);
                pos  = new Vector2(0f, (panel.Y - size.Y) * 0.5f);
            }
        }
    }

    static void DrawPlaceholder(Vector2 area, string label)
    {
        var textSize = ImGuiNET.ImGui.CalcTextSize(label);
        var cursor = ImGuiNET.ImGui.GetCursorPos();
        ImGuiNET.ImGui.SetCursorPos(new Vector2(
            cursor.X + (area.X - textSize.X) * 0.5f,
            cursor.Y + (area.Y - textSize.Y) * 0.5f));
        ImGuiNET.ImGui.TextDisabled(label);
    }
}
