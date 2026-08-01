using System;
using System.IO;
using System.Linq;
using System.Numerics;
using CadThingo.VulkanEngine.Renderer;
using CadThingo.VulkanEngine.Renderer.Features.WavefrontPathTracer;
using CadThingo.VulkanEngine.Renderer.Features.Tonemapping;
using CadThingo.VulkanEngine.Renderer.Features.PathTracer;
using CadThingo.VulkanEngine.Renderer.Features.IBL;
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

    // Staged spec-constant edits. These are panel state only - "the user moved this
    // control and hasn't hit Apply", which is what decides whether the Apply button
    // shows and what the widget displays meanwhile. The renderer is not told anything
    // until Apply publishes the intent, and it owns the rebuild flag from there. The
    // panel used to keep these *and* write the renderer's pending-rebuild fields, which
    // put one intent in two flags on two classes.
    static bool             _pbrRebuildPending     = false;
    static bool             _tonemapRebuildPending = false;
    static bool             _stagedSoftShadows;
    static TonemapOperator  _stagedTonemapOperator;

    // Every control here that changes what the camera sees without changing the scene
    // restarts progressive integration. Published rather than called so the panel does
    // not need the renderer for it; the renderer subscribes and owns the dirty flag.
    static void InvalidateAccumulator() =>
        Engine.EventBus.PublishEvent(new PathTracingAccumulatorInvalidatedEvent());

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

    //  Render mode
    // Registry-driven: the combo lists the renderer's registered cores by index (construction
    // order) using each core's Name, and hands the chosen index back via RequestCoreIndex. There
    // is no fixed label array to maintain -- a new core (or one gated out by missing device
    // support, which then never registers) appears/disappears here automatically.

    static void DrawRenderMode(Renderer.Renderer renderer)
    {
        ImGuiNET.ImGui.SeparatorText("Render Mode");

        var cores = renderer.RenderCores;
        if (cores.Count == 0) return;
        int current = renderer.ActiveCoreIndex;
        if (current < 0) current = 0;

        if (ImGuiNET.ImGui.BeginCombo("Mode", cores[current].Name))
        {
            for (int i = 0; i < cores.Count; i++)
            {
                bool selected = i == current;
                if (ImGuiNET.ImGui.Selectable(cores[i].Name, selected))
                {
                    renderer.RequestCoreIndex(i);
                    // Mode switch invalidates whatever was in FinalColor — drop any in-progress PT
                    // accumulation so we don't get a frame of stale pixels when toggling back on.
                    InvalidateAccumulator();
                }
                if (selected) ImGuiNET.ImGui.SetItemDefaultFocus();
            }
            ImGuiNET.ImGui.EndCombo();
        }
    }

    //  Pathtracer
    // Visible for both path-tracer modes (megakernel RayCompute + RayWavefront).
    // The camera-mode combo + lens sliders are shared via IPathTracerCamera since
    // both tracers bake the same four CameraMode PSOs and read the same DoF fields;
    // the per-tracer bits (sample counter, bounce-cap semantics) branch above.
    // Sliders push straight to the pipeline and mark the accumulator dirty so the
    // next frame restarts integration.

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
        var mode        = renderer.renderMode;
        bool isCompute   = mode == Renderer.Renderer.RenderMode.RayCompute;
        bool isWavefront = mode == Renderer.Renderer.RenderMode.RayWavefront;
        if (!isCompute && !isWavefront) return;

        ImGuiNET.ImGui.SeparatorText(isWavefront ? "Pathtracer (wavefront)" : "Pathtracer");

        if (isCompute)
        {
            var pt = renderer.ptComputePipeline;
            if (pt == null)
            {
                ImGuiNET.ImGui.TextDisabled("Pathtracer pipeline not initialized.");
                return;
            }

            DrawSampleCounter(renderer, pt.CurrentSampleCount);

            // Bounce cap. Hard ceiling MAX_BOUNCES is baked into the pipeline at
            // build (spec const); this slider clamps below that at runtime.
            int bounceTmp = (int)pt.BounceCap;
            if (ImGuiNET.ImGui.SliderInt("Bounce cap", ref bounceTmp, 1, (int)pt.MaxBouncesHardCap))
            {
                pt.BounceCap = (uint)bounceTmp;
                InvalidateAccumulator();
            }

            DrawFov(renderer);
            DrawCameraControls(renderer, pt);
        }
        else // RayWavefront
        {
            var wf = renderer.wavefrontPipeline;
            if (wf == null)
            {
                ImGuiNET.ImGui.TextDisabled("Wavefront pipeline not initialized.");
                return;
            }

            DrawSampleCounter(renderer, wf.CurrentSampleCount);

            // Bounce count is structural for the wavefront tracer: the graph
            // unrolls MaxBounces bodies at build, so there's no runtime clamp to
            // slide. Changing it is a code edit + graph rebuild (see
            // WavefrontPTPipeline.MaxBounces), hence read-only here.
            ImGuiNET.ImGui.Text($"Bounces: {WavefrontPTPipeline.MaxBounces} (structural)");

            DrawFov(renderer);
            DrawCameraControls(renderer, wf);
        }
    }

    // Sample counter (progressive accumulation depth) -- read-only feedback +
    // an explicit restart, shared by both tracers.
    static void DrawSampleCounter(Renderer.Renderer renderer, uint samples)
    {
        ImGuiNET.ImGui.Text($"Samples accumulated: {samples}");
        ImGuiNET.ImGui.SameLine();
        if (ImGuiNET.ImGui.SmallButton("Restart")) InvalidateAccumulator();
    }

    // FOV is the camera's vertical FOV (degrees), shared with the raster modes so
    // switching renderers doesn't change the framing. Both PT pipelines derive
    // radians + tanHalfFov from it for every projection mode. Range matches the
    // camera's own clamp.
    static void DrawFov(Renderer.Renderer renderer)
    {
        float fov = renderer.Camera.Fov;
        if (ImGuiNET.ImGui.SliderFloat("FOV (deg)", ref fov, 1f, 120f, "%.1f"))
        {
            renderer.Camera.Fov = fov;
            InvalidateAccumulator();
        }
    }

    // Camera mode + mode-specific lens sliders, driven through IPathTracerCamera
    // so the megakernel and wavefront pipelines share one code path. The mode
    // combo switches which Generate PSO Record binds -- no rebuild, no hitch.
    static void DrawCameraControls(Renderer.Renderer renderer, IPathTracerCamera cam)
    {
        int modeIdx = (int)cam.Mode;
        if (ImGuiNET.ImGui.Combo("Camera mode", ref modeIdx, _cameraModeLabels, _cameraModeLabels.Length))
        {
            cam.Mode = (PTComputePipeline.CameraMode)modeIdx;
            InvalidateAccumulator();
        }

        // Branch on the just-set Mode so the UI updates the same frame.
        switch (cam.Mode)
        {
            case PTComputePipeline.CameraMode.ThinLens:
                DrawThinLensSliders(renderer, cam);
                break;
            case PTComputePipeline.CameraMode.Panini:
                DrawPaniniSliders(renderer, cam);
                break;
            case PTComputePipeline.CameraMode.Fisheye:
                ImGuiNET.ImGui.TextDisabled("Equidistant fisheye uses just FOV - no extra controls.");
                break;
            case PTComputePipeline.CameraMode.Pinhole:
            case PTComputePipeline.CameraMode.Orthographic:
                // no extra sliders
                break;
        }
    }

    static void DrawThinLensSliders(Renderer.Renderer renderer, IPathTracerCamera cam)
    {
        ImGuiNET.ImGui.Indent();
        float aperture = cam.Aperture;
        if (ImGuiNET.ImGui.SliderFloat("Aperture radius", ref aperture, 0f, 0.5f, "%.3f"))
        {
            cam.Aperture = aperture;
            InvalidateAccumulator();
        }
        float focus = cam.FocusDistance;
        if (ImGuiNET.ImGui.SliderFloat("Focus distance", ref focus, 0.1f, 50f, "%.2f"))
        {
            cam.FocusDistance = focus;
            InvalidateAccumulator();
        }
        ImGuiNET.ImGui.TextDisabled("aperture = focalLength / (2 * fStop) - set to 0 to disable DoF.");
        ImGuiNET.ImGui.Unindent();
    }

    static void DrawPaniniSliders(Renderer.Renderer renderer, IPathTracerCamera cam)
    {
        ImGuiNET.ImGui.Indent();
        float d = cam.PaniniDistance;
        if (ImGuiNET.ImGui.SliderFloat("Panini distance (d)", ref d, 0f, 1f, "%.2f"))
        {
            cam.PaniniDistance = d;
            InvalidateAccumulator();
        }
        float vc = cam.VerticalCompression;
        if (ImGuiNET.ImGui.SliderFloat("Vertical compression", ref vc, 0f, 1f, "%.2f"))
        {
            cam.VerticalCompression = vc;
            InvalidateAccumulator();
        }
        ImGuiNET.ImGui.TextDisabled("d=0 collapses to pinhole; d=1 is the classic Panini look.");
        ImGuiNET.ImGui.Unindent();
    }

    // Reflection probes

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
        // user knows whether ProbeCapture compiled at startup — without it,
        // probes register but never capture.
        int used = probeSys.Probes.Count;
        int total = (int)probeSys.MaxProbes;
        ImGuiNET.ImGui.Text($"Slots: {used} / {total}");
        if (probeSys.capturePipeline == null)
        {
            ImGuiNET.ImGui.SameLine();
            ImGuiNET.ImGui.TextColored(new Vector4(1f, 0.5f, 0.3f, 1f),
                "  Capture shader failed to compile (see console)");
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
        if (probeSys.Probes.Count >= probeSys.MaxProbes)
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
        //editor events immediately fire so inspector will immediately show the probe
        Engine.EventBus.PublishEvent(new SceneEntitySelectedEvent(e));
    }

    //  Environment / IBL 

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
                try { renderer.Ibl.LoadEnvironmentHdr(_hdrFiles[_hdrSelected]); }
                catch (Exception e) { Console.WriteLine($"LoadEnvironmentHdr failed: {e.Message}"); }
            }
            ImGuiNET.ImGui.SameLine();
            if (ImGuiNET.ImGui.Button("Refresh")) RefreshHdrList(renderer);
        }

        var current = renderer.Ibl.CurrentEnvironmentPath;
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
        if (!string.IsNullOrEmpty(renderer.Ibl.CurrentEnvironmentPath))
        {
            for (int i = 0; i < _hdrFiles.Length; i++)
            {
                if (string.Equals(_hdrFiles[i], renderer.Ibl.CurrentEnvironmentPath, StringComparison.OrdinalIgnoreCase))
                {
                    _hdrSelected = i;
                    break;
                }
            }
        }
        if (_hdrSelected < 0 && _hdrFiles.Length > 0) _hdrSelected = 0;
    }

    // Tonemap

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

        // Operator is a spec constant - changing it needs a pipeline rebuild, so the
        // combo only stages a value locally and the renderer hears nothing until Apply.
        int opIdx = (int)(_tonemapRebuildPending ? _stagedTonemapOperator : renderer.tonemapOperator);
        if (ImGuiNET.ImGui.Combo("Operator", ref opIdx, new[] { "Reinhard", "Filmic" }, 2))
        {
            _stagedTonemapOperator = (TonemapOperator)opIdx;
            _tonemapRebuildPending = true;
        }
        if (_tonemapRebuildPending)
        {
            ImGuiNET.ImGui.SameLine();
            if (ImGuiNET.ImGui.Button("Apply##tonemap"))
            {
                // Fire-and-forget: the renderer stores the operator and rebuilds at the
                // top of the next frame. Rebuilding inline would tear down a pipeline
                // this frame's command buffer has already bound - a use-after-free at
                // submit time - which is why this is a queued event rather than a call.
                Engine.EventBus.PublishEvent(new TonemapFilterChangedEvent(_stagedTonemapOperator));
                _tonemapRebuildPending = false;
            }
        }
    }

    // Shadows

    static void DrawShadows(Renderer.Renderer renderer)
    {
        ImGuiNET.ImGui.SeparatorText("Shadows");

        if (!renderer.RayShadowsSupported)
        {
            ImGuiNET.ImGui.TextDisabled("Ray-traced shadows unavailable on this device.");
            return;
        }

        bool soft = _pbrRebuildPending ? _stagedSoftShadows : renderer.softShadowsEnabled;
        if (ImGuiNET.ImGui.Checkbox("Soft (PCSS-style) shadows", ref soft))
        {
            _stagedSoftShadows = soft;
            _pbrRebuildPending = true;
        }
        ImGuiNET.ImGui.TextDisabled("Trades a few hundred microseconds per pixel for distance-to-occluder penumbra.");

        if (_pbrRebuildPending)
        {
            if (ImGuiNET.ImGui.Button("Apply##pbr"))
            {
                // Deferred to next frame by the bus - see tonemap apply for reasoning.
                Engine.EventBus.PublishEvent(new PbrSoftShadowingChangedEvent(_stagedSoftShadows));
                _pbrRebuildPending = false;
            }
        }
    }

    // Background

    static void DrawBackground()
    {
        ImGuiNET.ImGui.SeparatorText("Background");

        var bg = EditorState.BackgroundColor;
        if (ImGuiNET.ImGui.ColorEdit3("Clear color", ref bg)) EditorState.BackgroundColor = bg;
        ImGuiNET.ImGui.TextDisabled("Visible only when the skybox is off.");
    }
}
