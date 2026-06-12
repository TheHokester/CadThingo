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

        DrawGpuMemory();
        DrawRenderGraph();
        DrawWavefrontCompaction();

        ImGuiNET.ImGui.End();
    }

    // TEMP/debug: per-bounce indirect launch sizes for the wavefront path tracer, read back from
    // the dispatchArgs buffer. Shrinking Extend/Shade/Connect columns down the bounces == compaction
    // working. Counts are workgroups (x64 ~= max rays). Only shown when wavefront is the active core.
    // Remove with the pipeline's _argsReadback feature.
    private static void DrawWavefrontCompaction()
    {
        var counts = Engine.renderer?.WavefrontDispatchCounts;
        if (counts == null || counts.Length < 3) return;

        ImGuiNET.ImGui.Separator();
        if (!ImGuiNET.ImGui.CollapsingHeader("Wavefront compaction (workgroups, x64 ~= rays)")) return;

        int bounces = counts.Length / 3;
        var flags = ImGuiNET.ImGuiTableFlags.Borders | ImGuiNET.ImGuiTableFlags.RowBg
                  | ImGuiNET.ImGuiTableFlags.SizingStretchProp;
        if (ImGuiNET.ImGui.BeginTable("##wfcompact", 4, flags))
        {
            ImGuiNET.ImGui.TableSetupColumn("Bounce");
            ImGuiNET.ImGui.TableSetupColumn("Extend");
            ImGuiNET.ImGui.TableSetupColumn("Shade");
            ImGuiNET.ImGui.TableSetupColumn("Connect");
            ImGuiNET.ImGui.TableHeadersRow();
            for (int b = 0; b < bounces; b++)
            {
                ImGuiNET.ImGui.TableNextRow();
                ImGuiNET.ImGui.TableNextColumn(); ImGuiNET.ImGui.Text($"{b}");
                ImGuiNET.ImGui.TableNextColumn(); ImGuiNET.ImGui.Text($"{counts[b * 3 + 0]}");
                ImGuiNET.ImGui.TableNextColumn(); ImGuiNET.ImGui.Text($"{counts[b * 3 + 1]}");
                ImGuiNET.ImGui.TableNextColumn(); ImGuiNET.ImGui.Text($"{counts[b * 3 + 2]}");
            }
            ImGuiNET.ImGui.EndTable();
        }
    }

    // GPU-memory occupancy + resize-churn counter. The headline number is the
    // Reserved−Used gap: if it climbs while you resize the viewport and does NOT
    // fall back when you return to the original size, the allocator is holding
    // empty blocks it never releases (the spp/s-degradation root cause under test).
    private static void DrawGpuMemory()
    {
        var r = Engine.renderer;
        if (r == null) return;

        ImGuiNET.ImGui.Separator();
        var m = r.GpuMemoryStats;
        const double MB = 1024.0 * 1024.0;
        double reserved = m.ReservedBytes / MB;
        double used     = m.UsedBytes / MB;
        double gap      = reserved - used;

        var ext = r.RenderExtent;
        ImGuiNET.ImGui.Text($"renderExtent {ext.Width}x{ext.Height}   rebuilds {r.RenderTargetRebuilds}");
        ImGuiNET.ImGui.Text($"VRAM reserved {reserved,8:F1} MB   used {used,8:F1} MB");
        ImGuiNET.ImGui.Text($"  retained (empty) {gap,8:F1} MB   blocks {m.PooledBlocks}+{m.DedicatedBlocks} ded");

        // Authoritative WDDM budget (VK_EXT_memory_budget). Headroom is what stands between
        // us and the OS demand-paging our working set — colour it red as it runs out. This
        // is the gauge that turns the silent residency cliff into something you can see.
        var b = r.GpuMemoryBudget;
        if (b.Available)
        {
            double budgetMb = b.Budget / MB;
            double usageMb  = b.Usage  / MB;
            double headMb   = b.Headroom / MB;
            ImGuiNET.ImGui.Text($"WDDM budget {usageMb,8:F1} / {budgetMb,8:F1} MB  ({b.Fraction * 100f,4:F0}%)");
            var col = b.Fraction >= 0.95f ? new Vector4(1f, 0.3f, 0.3f, 1f)
                    : b.Fraction >= 0.80f ? new Vector4(1f, 0.8f, 0.2f, 1f)
                    : new Vector4(0.5f, 0.9f, 0.5f, 1f);
            ImGuiNET.ImGui.TextColored(col, $"  headroom {headMb,8:F1} MB");
        }
        else
        {
            ImGuiNET.ImGui.TextDisabled("WDDM budget unavailable (VK_EXT_memory_budget off)");
        }

        // Per-rebuild history. X axis = rebuild #, so a rising line means each
        // resize adds memory that never comes back (leak); a flat line after the
        // first step means one-time high-water (no leak).
        int len  = r.MemHistoryLength;
        int head = r.MemHistoryHead;
        var usedH = r.UsedMbHistory;
        var resH  = r.ReservedMbHistory;
        if (usedH.Length == len && r.RenderTargetRebuilds > 0)
        {
            ImGuiNET.ImGui.PlotLines("##usedhist", ref usedH[0], len, head,
                "used MB / rebuild", float.MaxValue, float.MaxValue, new Vector2(0, 50));
            ImGuiNET.ImGui.PlotLines("##reshist", ref resH[0], len, head,
                "reserved MB / rebuild", float.MaxValue, float.MaxValue, new Vector2(0, 50));
        }
    }

    // Deferred FrameGraph instrumentation: per-pass GPU/CPU timing, barrier/cull counts,
    // optional pipeline statistics, and a DOT export. All values come from the renderer's
    // GraphStats snapshot (GPU timings are resolved a frame late, so they lag slightly).
    private static void DrawRenderGraph()
    {
        var stats = Engine.renderer?.ActiveGraphStats;
        if (stats is not { } gs) return;

        ImGuiNET.ImGui.Separator();
        if (!ImGuiNET.ImGui.CollapsingHeader($"Render Graph ({Engine.renderer!.ActiveCoreName})")) return;

        ImGuiNET.ImGui.Text($"{gs.LivePassCount} passes ({gs.CulledPassCount} culled)   "
                          + $"{gs.BarrierCount} barriers   compile {gs.CompileMs:F2} ms");
        ImGuiNET.ImGui.Text($"GPU frame {gs.FrameGpuMs,6:F3} ms");
        if (!gs.GpuTimingAvailable) ImGuiNET.ImGui.TextDisabled("(GPU timestamps unavailable on this device)");

        bool ps = gs.PipelineStatsOn;
        if (ImGuiNET.ImGui.Checkbox("Pipeline stats", ref ps) && Engine.renderer != null)
            Engine.renderer.ActiveGraphPipelineStats = ps;
        ImGuiNET.ImGui.SameLine();
        if (ImGuiNET.ImGui.Button("Copy DOT") && Engine.renderer != null)
            ImGuiNET.ImGui.SetClipboardText(Engine.renderer.ActiveGraphDot());

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