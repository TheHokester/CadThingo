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

        ImGuiNET.ImGui.End();
    }
}