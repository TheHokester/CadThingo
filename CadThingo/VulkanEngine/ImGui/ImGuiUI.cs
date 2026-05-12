using System.Numerics;
using ImGuiNET;

namespace CadThingo.VulkanEngine.ImGui;

public static class ImGuiUI
{
    public static void Draw()
    {
        DrawCameraOverlay();
    }

    static void DrawCameraOverlay()
    {
        var camera = Engine.renderer?.Camera;
        if (camera is null) return;

        const float pad = 10f;
        var viewport = ImGuiNET.ImGui.GetMainViewport();
        var pos = new Vector2(viewport.WorkPos.X + pad, viewport.WorkPos.Y + pad);
        ImGuiNET.ImGui.SetNextWindowPos(pos, ImGuiCond.Always, new Vector2(0f, 0f));
        ImGuiNET.ImGui.SetNextWindowBgAlpha(0.35f);

        const ImGuiWindowFlags flags =
            ImGuiWindowFlags.NoDecoration |
            ImGuiWindowFlags.AlwaysAutoResize |
            ImGuiWindowFlags.NoSavedSettings |
            ImGuiWindowFlags.NoFocusOnAppearing |
            ImGuiWindowFlags.NoNav |
            ImGuiWindowFlags.NoMove;

        if (ImGuiNET.ImGui.Begin("Camera", flags))
        {
            var p = camera.GetPosition();
            var f = camera.GetFront();
            ImGuiNET.ImGui.Text("Camera");
            ImGuiNET.ImGui.Separator();
            ImGuiNET.ImGui.Text($"Pos   X:{p.X,8:F2} Y:{p.Y,8:F2} Z:{p.Z,8:F2}");
            ImGuiNET.ImGui.Text($"Front X:{f.X,8:F2} Y:{f.Y,8:F2} Z:{f.Z,8:F2}");
        }
        ImGuiNET.ImGui.End();
    }
}