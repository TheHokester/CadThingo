using System;
using System.Numerics;
using ImGuiNET;

namespace CadThingo.VulkanEngine.ImGui.Panels;

/// <summary>
/// Editable camera state + CAD view presets + frame-selected.
///
/// DragFloat callers assign back unconditionally when ImGui reports a change so
/// the camera's property setters get a chance to re-derive front/right/up.
/// </summary>
public static unsafe class CameraPanel
{
    public static void Draw()
    {
        if (!EditorState.ShowCamera) return;

        if (!ImGuiNET.ImGui.Begin("Camera", ref EditorState.ShowCamera))
        {
            ImGuiNET.ImGui.End();
            return;
        }

        var camera = Engine.renderer?.Camera;
        if (camera == null)
        {
            ImGuiNET.ImGui.TextDisabled("Camera not initialized.");
            ImGuiNET.ImGui.End();
            return;
        }

        DrawPose(camera);
        DrawLens(camera);
        DrawViewPresets(camera);
        DrawFocus(camera);
        DrawControlFeel(camera);

        ImGuiNET.ImGui.End();
    }

    static void DrawPose(Camera camera)
    {
        ImGuiNET.ImGui.SeparatorText("Pose");

        var pos = camera.Position;
        if (ImGuiNET.ImGui.DragFloat3("Position", ref pos, 0.05f))
            camera.Position = pos;

        float yaw = camera.Yaw;
        if (ImGuiNET.ImGui.DragFloat("Yaw",   ref yaw,   0.5f, -360f, 360f, "%.2f°"))
            camera.Yaw = yaw;

        float pitch = camera.Pitch;
        if (ImGuiNET.ImGui.DragFloat("Pitch", ref pitch, 0.5f, -89f,   89f, "%.2f°"))
            camera.Pitch = pitch;

        var f = camera.GetFront();
        ImGuiNET.ImGui.TextDisabled($"Front  {f.X,7:F2} {f.Y,7:F2} {f.Z,7:F2}");
    }

    static void DrawLens(Camera camera)
    {
        ImGuiNET.ImGui.SeparatorText("Lens");

        float fov = camera.Fov;
        if (ImGuiNET.ImGui.DragFloat("FOV", ref fov, 0.5f, 1f, 120f, "%.1f°"))
            camera.Fov = fov;
    }

    static void DrawViewPresets(Camera camera)
    {
        ImGuiNET.ImGui.SeparatorText("View");

        // Two rows of six buttons; ImGui.SameLine packs them horizontally.
        if (ImGuiNET.ImGui.Button("Front"))  camera.SetViewFront();
        ImGuiNET.ImGui.SameLine();
        if (ImGuiNET.ImGui.Button("Back"))   camera.SetViewBack();
        ImGuiNET.ImGui.SameLine();
        if (ImGuiNET.ImGui.Button("Right"))  camera.SetViewRight();
        ImGuiNET.ImGui.SameLine();
        if (ImGuiNET.ImGui.Button("Left"))   camera.SetViewLeft();

        if (ImGuiNET.ImGui.Button("Top"))    camera.SetViewTop();
        ImGuiNET.ImGui.SameLine();
        if (ImGuiNET.ImGui.Button("Bottom")) camera.SetViewBottom();
        ImGuiNET.ImGui.SameLine();
        if (ImGuiNET.ImGui.Button("Iso"))    camera.SetViewIso();
    }

    static void DrawFocus(Camera camera)
    {
        ImGuiNET.ImGui.SeparatorText("Focus");

        var entity = EditorState.SelectedEntity;
        bool canFocus = entity != null;

        if (!canFocus) ImGuiNET.ImGui.BeginDisabled();
        if (ImGuiNET.ImGui.Button("Focus Selected") && canFocus)
            FocusOnEntity(camera, entity);
        if (!canFocus) ImGuiNET.ImGui.EndDisabled();

        if (!canFocus)
            ImGuiNET.ImGui.TextDisabled("Select an entity in the Scene Outliner.");
    }

    static void FocusOnEntity(Camera camera, Entity* entity)
    {
        var transform = entity->GetComponent<TransformComponent>();
        if (transform == null) return;

        var world = *transform.GetWorldMatrix();
        // Default frame: entity's world translation with unit radius — used when
        // there's no mesh to read a bounding sphere from.
        var target = new Vector3(world.M41, world.M42, world.M43);
        float radius = 1f;

        var meshComp = entity->GetComponent<MeshComponent>();
        if (meshComp != null && meshComp.mesh != null)
        {
            var sphere = meshComp.mesh->sphereLocal;
            var localCenter = new Vector3(sphere.X, sphere.Y, sphere.Z);
            target = Vector3.Transform(localCenter, world);

            // Approximate sphere-radius scaling for non-uniform scale: pick the
            // largest world-space axis length so the sphere still covers the AABB.
            float sx = new Vector3(world.M11, world.M12, world.M13).Length();
            float sy = new Vector3(world.M21, world.M22, world.M23).Length();
            float sz = new Vector3(world.M31, world.M32, world.M33).Length();
            radius = sphere.W * MathF.Max(sx, MathF.Max(sy, sz));
        }

        camera.FocusOn(target, radius);
    }

    static void DrawControlFeel(Camera camera)
    {
        ImGuiNET.ImGui.SeparatorText("Control feel");

        float speed = camera.MovementSpeed;
        if (ImGuiNET.ImGui.DragFloat("Move speed", ref speed, 0.05f, 0f, 100f, "%.2f u/s"))
            camera.MovementSpeed = speed;

        float sens = camera.MouseSensitivity;
        if (ImGuiNET.ImGui.DragFloat("Mouse sens", ref sens, 0.005f, 0f, 5f, "%.3f"))
            camera.MouseSensitivity = sens;
    }
}