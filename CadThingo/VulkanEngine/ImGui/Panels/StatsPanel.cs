using System.Numerics;
using ImGuiNET;

namespace CadThingo.VulkanEngine.ImGui.Panels;

/// <summary>
/// Frame timing + scene counters. Frametime history is a fixed ring buffer
/// rendered as a sparkline for quick visual perf reads.
/// </summary>
public static class StatsPanel
{
    private const int HistorySize = 200;
    private static readonly float[] _frameTimesMs = new float[HistorySize];
    private static int _frameTimeHead;

    public static void Draw()
    {
        if (!EditorState.ShowStats) return;

        // Push the current frametime even when the window is hidden, so toggling
        // the panel back on shows a continuous history rather than a gap.
        float ms = Engine.DeltaTime * 1000f;
        _frameTimesMs[_frameTimeHead] = ms;
        _frameTimeHead = (_frameTimeHead + 1) % HistorySize;

        if (!ImGuiNET.ImGui.Begin("Stats", ref EditorState.ShowStats))
        {
            ImGuiNET.ImGui.End();
            return;
        }

        float fps = Engine.DeltaTime > 0f ? 1f / Engine.DeltaTime : 0f;
        ImGuiNET.ImGui.Text($"Frame  {ms,6:F2} ms   ({fps,5:F0} fps)");
        ImGuiNET.ImGui.Text($"Time   {Engine.TotalTime,8:F2} s");

        ImGuiNET.ImGui.PlotLines("##frametime", ref _frameTimesMs[0], HistorySize, _frameTimeHead,
            $"{ms:F2} ms", 0f, 33.3f, new Vector2(0, 60));

        ImGuiNET.ImGui.Separator();
        var scene = Engine.renderer?.Scene;
        if (scene != null)
        {
            ImGuiNET.ImGui.Text($"Entities   {scene.EntityCount}");
            ImGuiNET.ImGui.Text($"Materials  {scene.MaterialCount}");
        }
        else
        {
            ImGuiNET.ImGui.TextDisabled("Scene not initialized.");
        }

        DrawRenderGraph();

        ImGuiNET.ImGui.End();
    }

    // Deferred FrameGraph instrumentation: per-pass GPU/CPU timing, barrier/cull counts,
    // optional pipeline statistics, and a DOT export. All values come from the renderer's
    // GraphStats snapshot (GPU timings are resolved a frame late, so they lag slightly).
    private static void DrawRenderGraph()
    {
        var stats = Engine.renderer?.DeferredGraphStats;
        if (stats is not { } gs) return;

        ImGuiNET.ImGui.Separator();
        if (!ImGuiNET.ImGui.CollapsingHeader("Render Graph (deferred)")) return;

        ImGuiNET.ImGui.Text($"{gs.LivePassCount} passes ({gs.CulledPassCount} culled)   "
                          + $"{gs.BarrierCount} barriers   compile {gs.CompileMs:F2} ms");
        ImGuiNET.ImGui.Text($"GPU frame {gs.FrameGpuMs,6:F3} ms");
        if (!gs.GpuTimingAvailable) ImGuiNET.ImGui.TextDisabled("(GPU timestamps unavailable on this device)");

        bool ps = gs.PipelineStatsOn;
        if (ImGuiNET.ImGui.Checkbox("Pipeline stats", ref ps) && Engine.renderer != null)
            Engine.renderer.DeferredGraphPipelineStats = ps;
        ImGuiNET.ImGui.SameLine();
        if (ImGuiNET.ImGui.Button("Copy DOT") && Engine.renderer != null)
            ImGuiNET.ImGui.SetClipboardText(Engine.renderer.DeferredGraphDot());

        int cols = gs.PipelineStatsOn ? 6 : 3;
        var flags = ImGuiNET.ImGuiTableFlags.Borders | ImGuiNET.ImGuiTableFlags.RowBg
                  | ImGuiNET.ImGuiTableFlags.SizingStretchProp;
        if (ImGuiNET.ImGui.BeginTable("##fgpasses", cols, flags))
        {
            ImGuiNET.ImGui.TableSetupColumn("Pass");
            ImGuiNET.ImGui.TableSetupColumn("GPU ms");
            ImGuiNET.ImGui.TableSetupColumn("CPU ms");
            if (gs.PipelineStatsOn)
            {
                ImGuiNET.ImGui.TableSetupColumn("VS");
                ImGuiNET.ImGui.TableSetupColumn("prim");
                ImGuiNET.ImGui.TableSetupColumn("FS");
            }
            ImGuiNET.ImGui.TableHeadersRow();

            foreach (var t in gs.Passes)
            {
                ImGuiNET.ImGui.TableNextRow();
                ImGuiNET.ImGui.TableNextColumn(); ImGuiNET.ImGui.Text(t.Name);
                ImGuiNET.ImGui.TableNextColumn(); ImGuiNET.ImGui.Text($"{t.GpuMs:F3}");
                ImGuiNET.ImGui.TableNextColumn(); ImGuiNET.ImGui.Text($"{t.CpuRecordMs:F3}");
                if (gs.PipelineStatsOn)
                {
                    ImGuiNET.ImGui.TableNextColumn(); ImGuiNET.ImGui.Text($"{t.Stats.VertexInvocations}");
                    ImGuiNET.ImGui.TableNextColumn(); ImGuiNET.ImGui.Text($"{t.Stats.Primitives}");
                    ImGuiNET.ImGui.TableNextColumn(); ImGuiNET.ImGui.Text($"{t.Stats.FragmentInvocations}");
                }
            }
            ImGuiNET.ImGui.EndTable();
        }
    }
}