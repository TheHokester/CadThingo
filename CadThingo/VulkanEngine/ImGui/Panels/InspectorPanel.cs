using System.Numerics;
using CadThingo.VulkanEngine.Renderer;
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

        var probe = entity->GetComponent<ReflectionProbeComponent>();
        if (probe != null) DrawProbe(probe);

        if (transform == null && mesh == null && light == null && probe == null)
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
        {
            t.SetPosition(pos);
            Engine.renderer.MarkAccumulatorDirty();
            Engine.renderer.MarkTlasDirty();
        }

        var quat = t.GetRotation();
        var euler = GetEulerDeg(entity, quat);
        if (ImGuiNET.ImGui.DragFloat3("Rotation", ref euler, 0.5f, 0f, 0f, "%.2f°"))
        {
            SetEulerDeg(entity, euler, t);
            Engine.renderer.MarkAccumulatorDirty();
            Engine.renderer.MarkTlasDirty();
        }

        var scale = t.GetScale();
        if (ImGuiNET.ImGui.DragFloat3("Scale", ref scale, 0.01f, 0.0001f, float.MaxValue))
        {
            Engine.renderer.MarkAccumulatorDirty();
            Engine.renderer.MarkTlasDirty();
            t.SetScale(scale);
        }

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
        {
            Engine.renderer.MarkAccumulatorDirty();
            m.materialIndex = matIdx;
        }
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

        if (ImGuiNET.ImGui.ColorEdit4("Base color", ref mat.BaseColorFactor,
                ImGuiColorEditFlags.AlphaBar | ImGuiColorEditFlags.AlphaPreview | ImGuiColorEditFlags.Float))
        {
            Engine.renderer.MarkAccumulatorDirty();
        }

        // Emissive is HDR — strength can exceed 1.0 (KHR_materials_emissive_strength).
        if (ImGuiNET.ImGui.ColorEdit3("Emissive", ref mat.EmissiveFactor,
                ImGuiColorEditFlags.Float | ImGuiColorEditFlags.HDR))
        {
            Engine.renderer.MarkAccumulatorDirty();
        }

        if (ImGuiNET.ImGui.SliderFloat("Metallic", ref mat.MetallicFactor, 0f, 1f))
        {
            Engine.renderer.MarkAccumulatorDirty();
        }

        if (ImGuiNET.ImGui.SliderFloat("Roughness", ref mat.RoughnessFactor, 0f, 1f))
        {
            Engine.renderer.MarkAccumulatorDirty();
        }

        // Alpha flags — Mask and Blend are conceptually mutually exclusive
        // (glTF treats them that way). We don't enforce here; the runtime
        // resolver in PbrMaterialAlphaExtensions gives Blend precedence.
        bool alphaMask    = (mat.Flags & 0x1u) != 0u;
        bool doubleSided  = (mat.Flags & 0x2u) != 0u;
        bool alphaBlend   = (mat.Flags & 0x4u) != 0u;

        if (ImGuiNET.ImGui.Checkbox("Alpha mask", ref alphaMask))
        {
            mat.Flags = SetBit(mat.Flags, 0x1u, alphaMask);
            Engine.renderer.MarkAccumulatorDirty();
        }
        ImGuiNET.ImGui.SameLine();
        if (ImGuiNET.ImGui.Checkbox("Double sided", ref doubleSided))
        {
            mat.Flags = SetBit(mat.Flags, 0x2u, doubleSided);
            Engine.renderer.MarkAccumulatorDirty();
        }
        ImGuiNET.ImGui.SameLine();
        if (ImGuiNET.ImGui.Checkbox("Alpha blend", ref alphaBlend))
        {
            mat.Flags = SetBit(mat.Flags, 0x4u, alphaBlend);
            Engine.renderer.MarkAccumulatorDirty();
        }

        if (alphaMask)
            if (ImGuiNET.ImGui.SliderFloat("Alpha cutoff", ref mat.AlphaCutoff, 0f, 1f))
            {
                Engine.renderer.MarkAccumulatorDirty();
            }

        // KHR_materials_transmission + KHR_materials_ior
        // Transmission only matters for non-opaque dielectrics; collapsed by
        // default so opaque-material editing stays uncluttered, but always
        // visible because flipping a material to transmissive is a one-slider
        // operation now.
        if (ImGuiNET.ImGui.TreeNode("Transmission / IOR"))
        {
            if (ImGuiNET.ImGui.SliderFloat("Transmission", ref mat.TransmissionFactor, 0f, 1f))
            {
                Engine.renderer.MarkAccumulatorDirty();
            }
            // 1.0 = vacuum / no refraction, 2.4 covers everything up to diamond.
            // Glass sits at 1.5, water 1.33, plastic 1.46–1.55.
            if (ImGuiNET.ImGui.SliderFloat("IOR", ref mat.Ior, 1.0f, 2.5f))
            {
                Engine.renderer.MarkAccumulatorDirty();
            }

            // KHR_materials_volume (Beer-Lambert absorption). Tint = color light
            // becomes after Distance units inside the solid; white = clear.
            // Needs watertight, two-sided glass — a single-quad pane has no
            // interior, so push Distance to max (no absorption) for those.
            if (ImGuiNET.ImGui.ColorEdit3("Attenuation tint", ref mat.AttenuationColor))
            {
                Engine.renderer.MarkAccumulatorDirty();
            }
            // Log slider: short distance = strong absorption (small gem),
            // huge distance ≈ no absorption (clear / thin glass).
            if (ImGuiNET.ImGui.SliderFloat("Attenuation distance", ref mat.AttenuationDistance,
                                           0.01f, 100f, "%.3f", ImGuiNET.ImGuiSliderFlags.Logarithmic))
            {
                Engine.renderer.MarkAccumulatorDirty();
            }
            ImGuiNET.ImGui.TreePop();
        }

        // KHR_materials_clearcoat
        // Same collapsed-by-default treatment. Pathtracer currently doesn't
        // sample clearcoat; uploaded for future BSDF work + the deferred
        // renderer can read it whenever that lobe lands too.
        if (ImGuiNET.ImGui.TreeNode("Clearcoat"))
        {
            if (ImGuiNET.ImGui.SliderFloat("Clearcoat", ref mat.ClearcoatFactor, 0f, 1f))
            {
                Engine.renderer.MarkAccumulatorDirty();
            };
            if (ImGuiNET.ImGui.SliderFloat("Clearcoat roughness", ref mat.ClearcoatRoughnessFactor, 0f, 1f))
            {
                Engine.renderer.MarkAccumulatorDirty();
            }
            ImGuiNET.ImGui.TreePop();
        }

        // Texture binding indices — bindless slot numbers into the texture
        // descriptor array. -1 = none. Read-only until the texture browser
        // lands.
        ImGuiNET.ImGui.TextDisabled(
            $"Tex base={FmtTex(mat.BaseColorTex)}  mr={FmtTex(mat.PhysicalDescriptorTex)}  " +
            $"nrm={FmtTex(mat.NormalTex)}  ao={FmtTex(mat.OcclusionTex)}  em={FmtTex(mat.EmissiveTex)}");
        ImGuiNET.ImGui.TextDisabled(
            $"Ext trans={FmtTex(mat.TransmissionTex)}  cc={FmtTex(mat.ClearcoatTex)}  " +
            $"ccR={FmtTex(mat.ClearcoatRoughnessTex)}  ccN={FmtTex(mat.ClearcoatNormalTex)}");
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
        {
            l.Type = (LightType)type;
            Engine.renderer.MarkAccumulatorDirty();
        }

        var color = l.Color;
        if (ImGuiNET.ImGui.ColorEdit3("Color", ref color, ImGuiColorEditFlags.Float))
        {
            Engine.renderer.MarkAccumulatorDirty();
            l.Color = color;
        }

        if (ImGuiNET.ImGui.DragFloat("Intensity", ref l.Intensity, 0.05f, 0f, float.MaxValue))
        {
            Engine.renderer.MarkAccumulatorDirty();
        }

        // Range is meaningless for directional lights — the shader encodes
        // directional as range=-1, so don't expose the slider in that mode.
        if (l.Type != LightType.Directional)
            if (ImGuiNET.ImGui.DragFloat("Range", ref l.Range, 0.1f, 0f, float.MaxValue))
            {
                Engine.renderer.MarkAccumulatorDirty();
            }

        // Direction is consumed by directional + spot only. Renormalise on edit
        // so accidental zero vectors don't drift the shader's lighting math.
        if (l.Type == LightType.Directional || l.Type == LightType.Spot)
        {
            var dir = l.Direction;
            if (ImGuiNET.ImGui.DragFloat3("Direction", ref dir, 0.01f))
            {
                l.Direction = dir.LengthSquared() > 1e-6f ? Vector3.Normalize(dir) : Vector3.UnitY * -1f;
                Engine.renderer.MarkAccumulatorDirty();
            }
        }

        // Spot cones — stored as cos for shader-side fast comparison. Edit in
        // degrees so the UI matches glTF authoring conventions.
        if (l.Type == LightType.Spot)
        {
            float innerDeg = MathF.Acos(Math.Clamp(l.InnerConeCos, -1f, 1f)) * (180f / MathF.PI);
            float outerDeg = MathF.Acos(Math.Clamp(l.OuterConeCos, -1f, 1f)) * (180f / MathF.PI);
            if (ImGuiNET.ImGui.DragFloat("Inner cone", ref innerDeg, 0.25f, 0f, 89f, "%.2f°"))
            {
                Engine.renderer.MarkAccumulatorDirty();
                l.InnerConeCos = MathF.Cos(innerDeg * (MathF.PI / 180f));
            }
            if (ImGuiNET.ImGui.DragFloat("Outer cone", ref outerDeg, 0.25f, 0f, 89f, "%.2f°"))
            {
                Engine.renderer.MarkAccumulatorDirty();
                l.OuterConeCos = MathF.Cos(outerDeg * (MathF.PI / 180f));
            }
        }

        // Radius drives ray-queried soft shadows. World-space radius for
        // point/spot; tan(angularRadius) for directional (sun ≈ 0.005).
        if (ImGuiNET.ImGui.DragFloat("Radius", ref l.Radius, 0.005f, 0f, float.MaxValue))
        {
            Engine.renderer.MarkAccumulatorDirty();
        };
    }

    static void DrawProbe(ReflectionProbeComponent probe)
    {
        if (!ImGuiNET.ImGui.CollapsingHeader("Reflection Probe", ImGuiTreeNodeFlags.DefaultOpen))
            return;

        // Slot assignment & status. Slot < 0 = system didn't register the probe
        // (out-of-slots) — the entity exists but never contributes to lighting.
        if (probe.CubeArraySlot >= 0)
            ImGuiNET.ImGui.Text($"Slot {probe.CubeArraySlot} / {ReflectionProbeSystem.MaxProbes - 1}");
        else
            ImGuiNET.ImGui.TextColored(new Vector4(1f, 0.5f, 0.3f, 1f),
                "Unregistered — slot pool full at probe creation.");

        ImGuiNET.ImGui.SameLine();
        if (probe.Dirty)
            ImGuiNET.ImGui.TextColored(new Vector4(1f, 0.85f, 0.2f, 1f), "  Dirty");
        else
            ImGuiNET.ImGui.TextDisabled($"  Last capture: frame {probe.LastCaptureFrame}");

        // Influence sphere — the probe contributes specular within this radius.
        // The shader uses smoothstep(0.5·r, r, dist) for the fall-off so doubling
        // this slider widens both the plateau and the transition band linearly.
        ImGuiNET.ImGui.DragFloat("Influence radius", ref probe.InfluenceRadius, 0.05f, 0.1f, 200f, "%.2f");

        // Face resolution is captured at *registration* into the shared cube-
        // array slot, so it's effectively read-only after registration. Show as
        // a disabled field with a TODO note.
        ImGuiNET.ImGui.BeginDisabled();
        int faceSize = (int)probe.FaceSize;
        ImGuiNET.ImGui.DragInt("Face size", ref faceSize);
        ImGuiNET.ImGui.EndDisabled();
        ImGuiNET.ImGui.TextDisabled("Face size is fixed per cube-array slot — Phase-9 work.");

        // Update policy combo. Once / OnDirty / EveryNFrames mirror the
        // ProbeUpdatePolicy enum exactly.
        int policy = (int)probe.UpdatePolicy;
        if (ImGuiNET.ImGui.Combo("Update policy", ref policy, "Once\0OnDirty\0EveryNFrames\0\0"))
            probe.UpdatePolicy = (ProbeUpdatePolicy)policy;

        if (probe.UpdatePolicy == ProbeUpdatePolicy.EveryNFrames)
        {
            int interval = (int)probe.UpdateIntervalFrames;
            if (ImGuiNET.ImGui.DragInt("Interval (frames)", ref interval, 1f, 1, 3600))
                probe.UpdateIntervalFrames = (uint)Math.Max(1, interval);
        }

        if (ImGuiNET.ImGui.Button("Force recapture"))
            probe.Dirty = true;
    }

    // Euler ↔ Quaternion
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
