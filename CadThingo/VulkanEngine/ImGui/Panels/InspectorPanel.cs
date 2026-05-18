using System.Numerics;
using ImGuiNET;

namespace CadThingo.VulkanEngine.ImGui.Panels;

/// <summary>
/// Per-entity component editor. Collapsing sections are gated on whether the
/// selected entity actually carries that component — no empty headers for
/// missing data. Each section reads from the live component and writes back
/// through its setter (or directly for plain fields), so changes flow into the
/// renderer on the next frame without explicit synchronisation.
/// </summary>
public static unsafe class InspectorPanel
{
    // Cached Euler representation of the selected entity's rotation. ImGui drags
    // edit Euler degrees, which we convert to a quaternion on write. We keep the
    // cache so consecutive frames of dragging stay numerically stable (extracting
    // Euler from a quaternion is lossy near gimbal lock — feeding the cached
    // value back avoids drift).
    static Entity*    _eulerEntity;
    static Quaternion _eulerLastQuat;
    static Vector3    _eulerCacheDeg;

    public static void Draw()
    {
        if (!EditorState.ShowInspector) return;

        if (!ImGuiNET.ImGui.Begin("Inspector", ref EditorState.ShowInspector))
        {
            ImGuiNET.ImGui.End();
            return;
        }

        var entity = EditorState.SelectedEntity;
        if (entity == null)
        {
            ImGuiNET.ImGui.TextDisabled("Select an entity from the Scene Outliner.");
            ImGuiNET.ImGui.End();
            return;
        }

        DrawHeader(entity);

        var transform = entity->GetComponent<TransformComponent>();
        if (transform != null) DrawTransform(entity, transform);

        var mesh = entity->GetComponent<MeshComponent>();
        if (mesh != null) DrawMesh(mesh);

        var light = entity->GetComponent<LightComponent>();
        if (light != null) DrawLight(light);

        if (transform == null && mesh == null && light == null)
            ImGuiNET.ImGui.TextDisabled("Entity has no editable components.");

        ImGuiNET.ImGui.End();
    }

    static void DrawHeader(Entity* entity)
    {
        ImGuiNET.ImGui.Text(entity->Name);
        ImGuiNET.ImGui.SameLine();
        ImGuiNET.ImGui.TextDisabled($"0x{(nint)entity:X}");

        bool active = entity->IsActive;
        if (ImGuiNET.ImGui.Checkbox("Active", ref active))
            entity->IsActive = active;

        ImGuiNET.ImGui.Separator();
    }

    static void DrawTransform(Entity* entity, TransformComponent t)
    {
        if (!ImGuiNET.ImGui.CollapsingHeader("Transform", ImGuiTreeNodeFlags.DefaultOpen))
            return;

        var pos = t.GetPosition();
        if (ImGuiNET.ImGui.DragFloat3("Position", ref pos, 0.05f))
            t.SetPosition(pos);

        var quat = t.GetRotation();
        var euler = GetEulerDeg(entity, quat);
        if (ImGuiNET.ImGui.DragFloat3("Rotation", ref euler, 0.5f, 0f, 0f, "%.2f°"))
            SetEulerDeg(entity, euler, t);

        var scale = t.GetScale();
        if (ImGuiNET.ImGui.DragFloat3("Scale", ref scale, 0.01f, 0.0001f, float.MaxValue))
            t.SetScale(scale);

        if (t.Parent != null)
            ImGuiNET.ImGui.TextDisabled($"Parent: {t.Parent->Name}");
    }

    static void DrawMesh(MeshComponent m)
    {
        if (!ImGuiNET.ImGui.CollapsingHeader("Mesh", ImGuiTreeNodeFlags.DefaultOpen))
            return;

        if (m.mesh != null)
        {
            ImGuiNET.ImGui.Text($"Index range: [{m.mesh->offset} .. +{m.mesh->count})");
            var s = m.mesh->sphereLocal;
            ImGuiNET.ImGui.Text($"Bound sphere: ({s.X:F2}, {s.Y:F2}, {s.Z:F2}) r={s.W:F2}");
        }
        else
        {
            ImGuiNET.ImGui.TextDisabled("No mesh assigned.");
        }

        var scene = Engine.renderer?.Scene;
        int matCount = scene?.MaterialCount ?? 0;

        // Material index — clamp upper bound to the live material count so users
        // don't dial in indices that point past the SSBO.
        int matIdx = m.materialIndex;
        int matMax = Math.Max(0, matCount - 1);
        int matMin = -1;
        if (ImGuiNET.ImGui.DragInt("Material", ref matIdx, 0.1f, matMin, matMax))
            m.materialIndex = matIdx;
        ImGuiNET.ImGui.SameLine();
        ImGuiNET.ImGui.TextDisabled($"of {matCount}");

        if (scene != null && m.materialIndex >= 0 && m.materialIndex < matCount)
            DrawMaterialEditor(scene, m.materialIndex);
    }

    /// <summary>
    /// Inline editor for the PbrMaterial at <paramref name="idx"/> on the scene.
    /// Writes go through MaterialsMutable so the per-frame upload picks them up
    /// next frame — no explicit dirty flag needed. Texture indices are
    /// read-only here; a richer view with previews and slot rebinding is the
    /// next step.
    /// </summary>
    static void DrawMaterialEditor(Scene scene, int idx)
    {
        ImGuiNET.ImGui.SeparatorText($"Material {idx}");
        ref var mat = ref scene.MaterialsMutable[idx];

        ImGuiNET.ImGui.ColorEdit4("Base color", ref mat.BaseColorFactor,
            ImGuiColorEditFlags.AlphaBar | ImGuiColorEditFlags.AlphaPreview | ImGuiColorEditFlags.Float);

        // Emissive is HDR — strength can exceed 1.0 (KHR_materials_emissive_strength).
        ImGuiNET.ImGui.ColorEdit3("Emissive", ref mat.EmissiveFactor,
            ImGuiColorEditFlags.Float | ImGuiColorEditFlags.HDR);

        ImGuiNET.ImGui.SliderFloat("Metallic",  ref mat.MetallicFactor,  0f, 1f);
        ImGuiNET.ImGui.SliderFloat("Roughness", ref mat.RoughnessFactor, 0f, 1f);

        // Alpha flags — Mask and Blend are conceptually mutually exclusive
        // (glTF treats them that way). We don't enforce here; the runtime
        // resolver in PbrMaterialAlphaExtensions gives Blend precedence.
        bool alphaMask    = (mat.Flags & 0x1u) != 0u;
        bool doubleSided  = (mat.Flags & 0x2u) != 0u;
        bool alphaBlend   = (mat.Flags & 0x4u) != 0u;

        if (ImGuiNET.ImGui.Checkbox("Alpha mask", ref alphaMask))
            mat.Flags = SetBit(mat.Flags, 0x1u, alphaMask);
        ImGuiNET.ImGui.SameLine();
        if (ImGuiNET.ImGui.Checkbox("Double sided", ref doubleSided))
            mat.Flags = SetBit(mat.Flags, 0x2u, doubleSided);
        ImGuiNET.ImGui.SameLine();
        if (ImGuiNET.ImGui.Checkbox("Alpha blend", ref alphaBlend))
            mat.Flags = SetBit(mat.Flags, 0x4u, alphaBlend);

        if (alphaMask)
            ImGuiNET.ImGui.SliderFloat("Alpha cutoff", ref mat.AlphaCutoff, 0f, 1f);

        // Texture binding indices — bindless slot numbers into the texture
        // descriptor array. -1 = none. Read-only until the texture browser
        // lands.
        ImGuiNET.ImGui.TextDisabled(
            $"Tex base={FmtTex(mat.BaseColorTex)}  mr={FmtTex(mat.PhysicalDescriptorTex)}  " +
            $"nrm={FmtTex(mat.NormalTex)}  ao={FmtTex(mat.OcclusionTex)}  em={FmtTex(mat.EmissiveTex)}");
    }

    static uint SetBit(uint flags, uint bit, bool on) => on ? (flags | bit) : (flags & ~bit);
    static string FmtTex(int idx) => idx < 0 ? "—" : idx.ToString();

    static void DrawLight(LightComponent l)
    {
        if (!ImGuiNET.ImGui.CollapsingHeader("Light", ImGuiTreeNodeFlags.DefaultOpen))
            return;

        ImGuiNET.ImGui.Checkbox("Enabled", ref l.Enabled);
        ImGuiNET.ImGui.SameLine();
        ImGuiNET.ImGui.Checkbox("Cast shadows", ref l.CastShadows);

        int type = (int)l.Type;
        if (ImGuiNET.ImGui.Combo("Type", ref type, "Directional\0Point\0Spot\0\0"))
            l.Type = (LightType)type;

        var color = l.Color;
        if (ImGuiNET.ImGui.ColorEdit3("Color", ref color, ImGuiColorEditFlags.Float))
            l.Color = color;

        ImGuiNET.ImGui.DragFloat("Intensity", ref l.Intensity, 0.05f, 0f, float.MaxValue);

        // Range is meaningless for directional lights — the shader encodes
        // directional as range=-1, so don't expose the slider in that mode.
        if (l.Type != LightType.Directional)
            ImGuiNET.ImGui.DragFloat("Range", ref l.Range, 0.1f, 0f, float.MaxValue);

        // Direction is consumed by directional + spot only. Renormalise on edit
        // so accidental zero vectors don't drift the shader's lighting math.
        if (l.Type == LightType.Directional || l.Type == LightType.Spot)
        {
            var dir = l.Direction;
            if (ImGuiNET.ImGui.DragFloat3("Direction", ref dir, 0.01f))
                l.Direction = dir.LengthSquared() > 1e-6f ? Vector3.Normalize(dir) : Vector3.UnitY * -1f;
        }

        // Spot cones — stored as cos for shader-side fast comparison. Edit in
        // degrees so the UI matches glTF authoring conventions.
        if (l.Type == LightType.Spot)
        {
            float innerDeg = MathF.Acos(Math.Clamp(l.InnerConeCos, -1f, 1f)) * (180f / MathF.PI);
            float outerDeg = MathF.Acos(Math.Clamp(l.OuterConeCos, -1f, 1f)) * (180f / MathF.PI);
            if (ImGuiNET.ImGui.DragFloat("Inner cone", ref innerDeg, 0.25f, 0f, 89f, "%.2f°"))
                l.InnerConeCos = MathF.Cos(innerDeg * (MathF.PI / 180f));
            if (ImGuiNET.ImGui.DragFloat("Outer cone", ref outerDeg, 0.25f, 0f, 89f, "%.2f°"))
                l.OuterConeCos = MathF.Cos(outerDeg * (MathF.PI / 180f));
        }

        // Radius drives ray-queried soft shadows. World-space radius for
        // point/spot; tan(angularRadius) for directional (sun ≈ 0.005).
        ImGuiNET.ImGui.DragFloat("Radius", ref l.Radius, 0.005f, 0f, float.MaxValue);
    }

    // ── Euler ↔ Quaternion ──────────────────────────────────────
    // ZYX-intrinsic (= XYZ-extrinsic) Tait-Bryan. Pick any convention and stay
    // consistent — the cache below means we only round-trip when the quaternion
    // changes externally, so accumulated drift during dragging stays at zero.

    static Vector3 GetEulerDeg(Entity* e, Quaternion q)
    {
        if (e != _eulerEntity || !RoughlyEqual(_eulerLastQuat, q))
        {
            _eulerEntity   = e;
            _eulerCacheDeg = ExtractEulerDeg(q);
            _eulerLastQuat = q;
        }
        return _eulerCacheDeg;
    }

    static void SetEulerDeg(Entity* e, Vector3 deg, TransformComponent t)
    {
        var q = EulerDegToQuat(deg);
        _eulerEntity   = e;
        _eulerCacheDeg = deg;
        _eulerLastQuat = q;
        t.SetRotation(q);
    }

    static Vector3 ExtractEulerDeg(Quaternion q)
    {
        // Standard ZYX-intrinsic extraction.
        float x = MathF.Atan2(2f * (q.W * q.X + q.Y * q.Z),
                              1f - 2f * (q.X * q.X + q.Y * q.Y));
        float sy = 2f * (q.W * q.Y - q.Z * q.X);
        float y = MathF.Abs(sy) >= 1f
            ? MathF.CopySign(MathF.PI * 0.5f, sy)
            : MathF.Asin(sy);
        float z = MathF.Atan2(2f * (q.W * q.Z + q.X * q.Y),
                              1f - 2f * (q.Y * q.Y + q.Z * q.Z));
        return new Vector3(x, y, z) * (180f / MathF.PI);
    }

    static Quaternion EulerDegToQuat(Vector3 deg)
    {
        var r = deg * (MathF.PI / 180f);
        var qx = Quaternion.CreateFromAxisAngle(Vector3.UnitX, r.X);
        var qy = Quaternion.CreateFromAxisAngle(Vector3.UnitY, r.Y);
        var qz = Quaternion.CreateFromAxisAngle(Vector3.UnitZ, r.Z);
        // ZYX intrinsic = Z * Y * X applied to vectors.
        return qz * qy * qx;
    }

    static bool RoughlyEqual(Quaternion a, Quaternion b)
    {
        // q and -q represent the same rotation, so compare via abs(dot).
        float dot = a.X * b.X + a.Y * b.Y + a.Z * b.Z + a.W * b.W;
        return MathF.Abs(dot) > 0.99999f;
    }
}
