using ImGuiNET;
using CadThingo.VulkanEngine.ImGui.Panels;

namespace CadThingo.VulkanEngine.ImGui;

public static class ImGuiUI
{
    public static void Draw()
    {
        // F11 toggles viewport fullscreen. Skip while a text-input has focus so
        // the key doesn't get consumed mid-typing in a panel input.
        if (!ImGuiNET.ImGui.GetIO().WantTextInput && ImGuiNET.ImGui.IsKeyPressed(ImGuiKey.F11))
            EditorState.ViewportFullscreen = !EditorState.ViewportFullscreen;

        // Always create the dockspace — even in fullscreen mode — so docked
        // windows keep their dock-node attachments. In fullscreen we just skip
        // drawing the chrome and let ViewportPanel paint a fullscreen overlay
        // on top.
        ImGuiNET.ImGui.DockSpaceOverViewport(ImGuiNET.ImGui.GetMainViewport().ID,
            ImGuiNET.ImGui.GetMainViewport(),
            ImGuiDockNodeFlags.PassthruCentralNode);

        if (EditorState.ViewportFullscreen)
        {
            // Only the fullscreen viewport draws. The other panels (and the
            // docked Viewport window) are simply not Begin()'d this frame —
            // ImGui keeps their saved dock positions so the layout is intact
            // when fullscreen toggles off.
            ViewportPanel.Draw();
            return;
        }

        DrawMainMenuBar();

        ViewportPanel.Draw();
        SceneOutlinerPanel.Draw();
        InspectorPanel.Draw();
        StatsPanel.Draw();
        CameraPanel.Draw();
    }

    static void DrawMainMenuBar()
    {
        if (!ImGuiNET.ImGui.BeginMainMenuBar()) return;

        if (ImGuiNET.ImGui.BeginMenu("View"))
        {
            ImGuiNET.ImGui.MenuItem("Viewport",       null, ref EditorState.ShowViewport);
            ImGuiNET.ImGui.MenuItem("Scene Outliner", null, ref EditorState.ShowSceneOutliner);
            ImGuiNET.ImGui.MenuItem("Inspector",      null, ref EditorState.ShowInspector);
            ImGuiNET.ImGui.MenuItem("Stats",          null, ref EditorState.ShowStats);
            ImGuiNET.ImGui.MenuItem("Camera",         null, ref EditorState.ShowCamera);
            ImGuiNET.ImGui.Separator();
            ImGuiNET.ImGui.MenuItem("Viewport fullscreen", "F11", ref EditorState.ViewportFullscreen);
            ImGuiNET.ImGui.EndMenu();
        }

        ImGuiNET.ImGui.EndMainMenuBar();
    }
}
