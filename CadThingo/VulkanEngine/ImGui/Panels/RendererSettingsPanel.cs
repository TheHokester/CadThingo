using System;
using System.IO;
using System.Linq;
using System.Numerics;
using CadThingo.VulkanEngine.Renderer;
using CadThingo.VulkanEngine.Renderer.Pipelines;
using ImGuiNET;

namespace CadThingo.VulkanEngine.ImGui.Panels;

/// <summary>
/// Renderer-level controls — environment / IBL / tonemap / shadows /
/// background. Most fields are live (slider drag updates the next frame's
/// UBO); spec-constant toggles (soft shadows, tonemap operator) require an
/// explicit pipeline rebuild via the Apply button next to each.
///
/// HDR file picker is a dropdown of *.hdr files discovered under
/// Assets/Textures/. Refreshed manually (the Reload button) so drag-drop
/// hotloading isn't on by default.
/// </summary>
public static class RendererSettingsPanel
{
    static string[] _hdrFiles      = Array.Empty<string>();
    static string[] _hdrLabels     = Array.Empty<string>();
    static int      _hdrSelected   = -1;
    static bool     _hdrListLoaded = false;

    // Pending-spec-constant tracking. Lit when the user flips a toggle that
    // requires a pipeline rebuild; reset by the Apply button.
    static bool _pbrRebuildPending     = false;
    static bool _tonemapRebuildPending = false;

    public static void Draw()
    {
        if (!EditorState.ShowRendererSettings) return;

        if (!ImGuiNET.ImGui.Begin("Renderer Settings", ref EditorState.ShowRendererSettings))
        {
            ImGuiNET.ImGui.End();
            return;
        }

        var renderer = Engine.renderer;
        if (renderer == null)
        {
            ImGuiNET.ImGui.TextDisabled("Renderer not initialized.");
            ImGuiNET.ImGui.End();
            return;
        }

        DrawRenderMode(renderer);
        DrawPathtracer(renderer);
        DrawEnvironment(renderer);
        DrawReflectionProbes(renderer);
        DrawTonemap(renderer);
        DrawShadows(renderer);
        DrawBackground();

        ImGuiNET.ImGui.End();
    }

    // ── Render mode ────────────────────────────────────────────────────────

    static readonly string[] _renderModeLabels =
    {
        "Deferred",
        "Forward+ (ray-queried)",
        "Pathtracer (compute + ray query)",
        "Pathtracer (RT pipeline) [TODO]",
    };

    static void DrawRenderMode(Renderer.Renderer renderer)
    {
        ImGuiNET.ImGui.SeparatorText("Render Mode");

        int modeIdx = (int)renderer.renderMode;
        if (ImGuiNET.ImGui.Combo("Mode", ref modeIdx, _renderModeLabels, _renderModeLabels.Length))
        {
            renderer.renderMode = (Renderer.Renderer.RenderMode)modeIdx;
            // Mode switch invalidates whatever was in FinalColor — drop any in-
            // progress PT accumulation so we don't get a frame of stale pixels
            // when toggling back on.
            renderer.MarkAccumulatorDirty();
        }
    }

    // ── Pathtracer ─────────────────────────────────────────────────────────
    // Visible only when render mode is RayCompute. Sliders push their value
    // straight to the pipeline and mark the accumulator dirty so the next
    // frame restarts integration.

    static readonly string[] _cameraModeLabels =
    {
        "Pinhole",
        "Thin lens (DoF)",
        "Generalized Panini",
        "Fisheye (equidistant)",
        "Orthographic [TODO]",
    };

    static void DrawPathtracer(Renderer.Renderer renderer)
    {
        if (renderer.renderMode != Renderer.Renderer.RenderMode.RayCompute) return;

        var pt = renderer.ptComputePipeline;
        if (pt == null)
        {
            ImGuiNET.ImGui.SeparatorText("Pathtracer");
            ImGuiNET.ImGui.TextDisabled("Pathtracer pipeline not initialized.");
            return;
        }

        ImGuiNET.ImGui.SeparatorText("Pathtracer");

        // Sample counter (progressive accumulation depth) — read-only feedback.
        ImGuiNET.ImGui.Text($"Samples accumulated: {pt.CurrentSampleCount}");
        ImGuiNET.ImGui.SameLine();
        if (ImGuiNET.ImGui.SmallButton("Restart")) renderer.MarkAccumulatorDirty();

        // Bounce cap. Hard ceiling MAX_BOUNCES is baked into the pipeline at
        // build (spec const); this slider clamps below that at runtime.
        uint bounceCap = pt.BounceCap;
        int  bounceTmp = (int)bounceCap;
        if (ImGuiNET.ImGui.SliderInt("Bounce cap", ref bounceTmp, 1, (int)pt.MaxBouncesHardCap))
        {
            pt.BounceCap = (uint)bounceTmp;
            renderer.MarkAccumulatorDirty();
        }

        // FOV is used by every projection mode — vertical FOV in degrees, the
        // pipeline derives radians + tanHalfFov for the UBO.
        float fov = pt.FovDeg;
        if (ImGuiNET.ImGui.SliderFloat("FOV (deg)", ref fov, 10f, 170f, "%.1f"))
        {
            pt.FovDeg = fov;
            renderer.MarkAccumulatorDirty();
        }

        // Camera mode — switches which PSO Record binds. No rebuild, no hitch.
        int modeIdx = (int)pt.Mode;
        if (ImGuiNET.ImGui.Combo("Camera mode", ref modeIdx, _cameraModeLabels, _cameraModeLabels.Length))
        {
            pt.Mode = (PTComputePipeline.CameraMode)modeIdx;
            renderer.MarkAccumulatorDirty();
        }

        // Mode-specific sliders. Branch on the just-set Mode so the UI updates
        // the same frame the user changes mode.
        switch (pt.Mode)
        {
            case PTComputePipeline.CameraMode.ThinLens:
                DrawThinLensSliders(renderer, pt);
                break;
            case PTComputePipeline.CameraMode.Panini:
                DrawPaniniSliders(renderer, pt);
                break;
            case PTComputePipeline.CameraMode.Fisheye:
                ImGuiNET.ImGui.TextDisabled("Equidistant fisheye uses just FOV — no extra controls.");
                break;
            case PTComputePipeline.CameraMode.Pinhole:
            case PTComputePipeline.CameraMode.Orthographic:
                // no extra sliders
                break;
        }
    }

    static void DrawThinLensSliders(Renderer.Renderer renderer, PTComputePipeline pt)
    {
        ImGuiNET.ImGui.Indent();
        float aperture = pt.Aperture;
        if (ImGuiNET.ImGui.SliderFloat("Aperture radius", ref aperture, 0f, 0.5f, "%.3f"))
        {
            pt.Aperture = aperture;
            renderer.MarkAccumulatorDirty();
        }
        float focus = pt.FocusDistance;
        if (ImGuiNET.ImGui.SliderFloat("Focus distance", ref focus, 0.1f, 50f, "%.2f"))
        {
            pt.FocusDistance = focus;
            renderer.MarkAccumulatorDirty();
        }
        ImGuiNET.ImGui.TextDisabled("aperture = focalLength / (2 · fStop) — set to 0 to disable DoF.");
        ImGuiNET.ImGui.Unindent();
    }

    static void DrawPaniniSliders(Renderer.Renderer renderer, PTComputePipeline pt)
    {
        ImGuiNET.ImGui.Indent();
        float d = pt.PaniniDistance;
        if (ImGuiNET.ImGui.SliderFloat("Panini distance (d)", ref d, 0f, 1f, "%.2f"))
        {
            pt.PaniniDistance = d;
            renderer.MarkAccumulatorDirty();
        }
        float vc = pt.VerticalCompression;
        if (ImGuiNET.ImGui.SliderFloat("Vertical compression", ref vc, 0f, 1f, "%.2f"))
        {
            pt.VerticalCompression = vc;
            renderer.MarkAccumulatorDirty();
        }
        ImGuiNET.ImGui.TextDisabled("d=0 collapses to pinhole; d=1 is the classic Panini look.");
        ImGuiNET.ImGui.Unindent();
    }

    // ── Reflection probes ─────────────────────────────────────────────────

    static void DrawReflectionProbes(Renderer.Renderer renderer)
    {
        ImGuiNET.ImGui.SeparatorText("Reflection Probes");

        var probeSys = renderer.reflectionProbeSystem;
        if (probeSys == null)
        {
            ImGuiNET.ImGui.TextDisabled("Probe system not ready.");
            return;
        }

        // Slot usage summary. Capture pipeline status is surfaced here so the
        // user knows whether ProbeCapture.spv was found at startup — without
        // it, registered probes register but never capture.
        int used = probeSys.Probes.Count;
        int total = (int)ReflectionProbeSystem.MaxProbes;
        ImGuiNET.ImGui.Text($"Slots: {used} / {total}");
        if (probeSys.capturePipeline == null)
        {
            ImGuiNET.ImGui.SameLine();
            ImGuiNET.ImGui.TextColored(new Vector4(1f, 0.5f, 0.3f, 1f),
                "  Capture shader missing (run shaderCompile.bat)");
        }

        if (ImGuiNET.ImGui.Button("Spawn probe at camera"))
            SpawnProbeAtCamera(renderer);
        ImGuiNET.ImGui.SameLine();
        if (ImGuiNET.ImGui.Button("Force recapture all"))
        {
            foreach (var p in probeSys.Probes) p.Dirty = true;
        }
    }

    static unsafe void SpawnProbeAtCamera(Renderer.Renderer renderer)
    {
        // Position the new probe at the camera's eye. The scheduler picks it up
        // on the next frame (first-time probes go to the front of the queue),
        // so the user sees the probe contribute within a frame or two.
        var probeSys = renderer.reflectionProbeSystem;
        if (probeSys.Probes.Count >= ReflectionProbeSystem.MaxProbes)
        {
            Console.WriteLine("[Probe] Cannot spawn — slot pool full.");
            return;
        }

        var camPos = renderer.Camera.GetPosition();

        Entity* e = Entity.Create($"Probe_{probeSys.Probes.Count}");
        e->AddComponent(new TransformComponent());
        e->GetComponent<TransformComponent>()?.SetPosition(camPos);
        e->AddComponent(new ReflectionProbeComponent
        {
            InfluenceRadius = 8f,
            UpdatePolicy    = ProbeUpdatePolicy.OnDirty,
        });
        e->Initialize();
        renderer.Scene.AddEntity(e);

        var probe = e->GetComponent<ReflectionProbeComponent>();
        if (probe != null) probeSys.Register(probe);

        // Auto-select the newly spawned probe so the user immediately sees its
        // component in the Inspector and can tweak the radius.
        EditorState.SelectedEntity = e;
    }

    // ── Environment / IBL ─────────────────────────────────────────────────

    static void DrawEnvironment(Renderer.Renderer renderer)
    {
        ImGuiNET.ImGui.SeparatorText("Environment");

        if (!_hdrListLoaded) RefreshHdrList(renderer);

        if (_hdrFiles.Length == 0)
        {
            ImGuiNET.ImGui.TextDisabled("No .hdr files in Assets/Textures/");
        }
        else
        {
            ImGuiNET.ImGui.SetNextItemWidth(220);
            ImGuiNET.ImGui.Combo("##hdrCombo", ref _hdrSelected, _hdrLabels, _hdrLabels.Length);
            ImGuiNET.ImGui.SameLine();
            if (ImGuiNET.ImGui.Button("Load") && _hdrSelected >= 0 && _hdrSelected < _hdrFiles.Length)
            {
                try { renderer.LoadEnvironmentHdr(_hdrFiles[_hdrSelected]); }
                catch (Exception e) { Console.WriteLine($"LoadEnvironmentHdr failed: {e.Message}"); }
            }
            ImGuiNET.ImGui.SameLine();
            if (ImGuiNET.ImGui.Button("Refresh")) RefreshHdrList(renderer);
        }

        var current = renderer.CurrentEnvironmentPath;
        ImGuiNET.ImGui.TextDisabled($"Loaded: {(string.IsNullOrEmpty(current) ? "(none)" : Path.GetFileName(current))}");

        ImGuiNET.ImGui.Spacing();
        ImGuiNET.ImGui.SliderFloat("IBL intensity",    ref EditorState.IblIntensity,    0f, 3f, "%.2f");
        ImGuiNET.ImGui.Checkbox  ("Skybox enabled",    ref EditorState.SkyboxEnabled);
        ImGuiNET.ImGui.SliderFloat("Skybox intensity", ref EditorState.SkyboxIntensity, 0f, 3f, "%.2f");
    }

    static void RefreshHdrList(Renderer.Renderer renderer)
    {
        // Assets path is relative to the project layout — same convention the
        // renderer uses for the rest of its asset lookups.
        string dir = Path.Combine(AppContext.BaseDirectory, "Assets", "Textures");
        if (!Directory.Exists(dir))
        {
            // Fall back to the source-tree location when running uninstalled.
            dir = @"C:\Users\jamie\RiderProjects\CadThingo\CadThingo\Assets\Textures";
        }
        if (Directory.Exists(dir))
        {
            _hdrFiles  = Directory.GetFiles(dir, "*.hdr", SearchOption.TopDirectoryOnly);
            _hdrLabels = _hdrFiles.Select(Path.GetFileName).ToArray()!;
        }
        else
        {
            _hdrFiles  = Array.Empty<string>();
            _hdrLabels = Array.Empty<string>();
        }
        _hdrListLoaded = true;

        // Pre-select whatever's currently loaded, if it's still on the list.
        _hdrSelected = -1;
        if (!string.IsNullOrEmpty(renderer.CurrentEnvironmentPath))
        {
            for (int i = 0; i < _hdrFiles.Length; i++)
            {
                if (string.Equals(_hdrFiles[i], renderer.CurrentEnvironmentPath, StringComparison.OrdinalIgnoreCase))
                {
                    _hdrSelected = i;
                    break;
                }
            }
        }
        if (_hdrSelected < 0 && _hdrFiles.Length > 0) _hdrSelected = 0;
    }

    // ── Tonemap ────────────────────────────────────────────────────────────

    static void DrawTonemap(Renderer.Renderer renderer)
    {
        ImGuiNET.ImGui.SeparatorText("Tonemap");

        var tm = renderer.tonemapPipeline;
        if (tm == null)
        {
            ImGuiNET.ImGui.TextDisabled("Tonemap pipeline not ready.");
            return;
        }

        float exposure = tm.Exposure;
        if (ImGuiNET.ImGui.SliderFloat("Exposure", ref exposure, 0.1f, 10f, "%.2f"))
            tm.Exposure = exposure;
        float gamma = tm.Gamma;
        if (ImGuiNET.ImGui.SliderFloat("Gamma",    ref gamma,    1.0f, 3f, "%.2f"))
            tm.Gamma = gamma;

        // Operator is a spec constant — changing it needs a pipeline rebuild
        // (it doesn't actually rebuild here; we just flag _tonemapRebuildPending
        // until the user clicks Apply).
        int opIdx = (int)renderer.tonemapOperator;
        if (ImGuiNET.ImGui.Combo("Operator", ref opIdx, new[] { "Reinhard", "Filmic" }, 2))
        {
            renderer.tonemapOperator = (TonemapOperator)opIdx;
            _tonemapRebuildPending = true;
        }
        if (_tonemapRebuildPending)
        {
            ImGuiNET.ImGui.SameLine();
            if (ImGuiNET.ImGui.Button("Apply##tonemap"))
            {
                // Queue the rebuild for the top of the next frame — running it
                // here would tear down a pipeline this frame's command buffer
                // has already bound, which is a use-after-free at submit time.
                renderer.pendingTonemapRebuild = true;
                _tonemapRebuildPending         = false;
            }
        }
    }

    // ── Shadows ────────────────────────────────────────────────────────────

    static void DrawShadows(Renderer.Renderer renderer)
    {
        ImGuiNET.ImGui.SeparatorText("Shadows");

        if (!renderer.RayShadowsSupported)
        {
            ImGuiNET.ImGui.TextDisabled("Ray-traced shadows unavailable on this device.");
            return;
        }

        bool soft = renderer.softShadowsEnabled;
        if (ImGuiNET.ImGui.Checkbox("Soft (PCSS-style) shadows", ref soft))
        {
            renderer.softShadowsEnabled = soft;
            _pbrRebuildPending = true;
        }
        ImGuiNET.ImGui.TextDisabled("Trades a few hundred microseconds per pixel for distance-to-occluder penumbra.");

        if (_pbrRebuildPending)
        {
            if (ImGuiNET.ImGui.Button("Apply##pbr"))
            {
                // Defer to next frame — see tonemap apply for reasoning.
                renderer.pendingPbrRebuild = true;
                _pbrRebuildPending         = false;
            }
        }
    }

    // ── Background ────────────────────────────────────────────────────────

    static void DrawBackground()
    {
        ImGuiNET.ImGui.SeparatorText("Background");

        var bg = EditorState.BackgroundColor;
        if (ImGuiNET.ImGui.ColorEdit3("Clear color", ref bg)) EditorState.BackgroundColor = bg;
        ImGuiNET.ImGui.TextDisabled("Visible only when the skybox is off.");
    }
}
