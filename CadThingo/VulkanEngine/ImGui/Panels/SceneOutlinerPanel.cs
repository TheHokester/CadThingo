using ImGuiNET;

namespace CadThingo.VulkanEngine.ImGui.Panels;

/// <summary>
/// Hierarchical entity browser. Reads the persistent parent/child mirror that
/// Scene maintains incrementally on AddEntity / AddChild / Reparent — no
/// per-frame rebuild.
///
/// "Hide empty" suppresses nodes that carry neither MeshComponent nor
/// LightComponent (typically transform-only intermediates from nested glTF
/// imports). Children of suppressed nodes still draw — they appear at the
/// parent's depth, flattening hierarchy plumbing the user can't act on.
/// </summary>
public static unsafe class SceneOutlinerPanel
{
    public static void Draw()
    {
        if (!EditorState.ShowSceneOutliner) return;

        if (!ImGuiNET.ImGui.Begin("Scene Outliner", ref EditorState.ShowSceneOutliner))
        {
            ImGuiNET.ImGui.End();
            return;
        }

        var scene = Engine.renderer?.Scene;
        if (scene == null)
        {
            ImGuiNET.ImGui.TextDisabled("Scene not initialized.");
            ImGuiNET.ImGui.End();
            return;
        }

        ImGuiNET.ImGui.Text($"{scene.EntityCount} entit{(scene.EntityCount == 1 ? "y" : "ies")}");
        ImGuiNET.ImGui.SameLine();
        ImGuiNET.ImGui.Checkbox("Hide empty", ref EditorState.HideEmptyEntities);
        ImGuiNET.ImGui.Separator();

        // Scrolling child so deep glTF imports don't blow the dock node out.
        if (ImGuiNET.ImGui.BeginChild("##outliner_scroll", default, ImGuiChildFlags.None))
        {
            var roots = scene.Roots;
            for (int i = 0; i < roots.Count; i++)
                DrawNode(scene, (Entity*)roots[i], i);
        }
        ImGuiNET.ImGui.EndChild();

        ImGuiNET.ImGui.End();
    }

    static void DrawNode(Scene scene, Entity* entity, int siblingIndex)
    {
        if (entity == null) return;

        bool isEmpty = IsEmpty(entity);
        bool hide    = EditorState.HideEmptyEntities && isEmpty;

        // Empty-and-hidden: skip the node header, recurse so descendants stay
        // visible at this depth.
        if (hide)
        {
            DrawChildren(scene, entity);
            return;
        }

        var  children    = scene.GetChildren(entity);
        bool hasChildren = children.Count > 0;
        bool selected    = EditorState.SelectedEntity == entity;

        var flags = ImGuiTreeNodeFlags.OpenOnArrow
                  | ImGuiTreeNodeFlags.OpenOnDoubleClick
                  | ImGuiTreeNodeFlags.SpanAvailWidth;
        if (selected)     flags |= ImGuiTreeNodeFlags.Selected;
        if (!hasChildren) flags |= ImGuiTreeNodeFlags.Leaf | ImGuiTreeNodeFlags.NoTreePushOnOpen;

        // Pointer is the only stable per-entity ID. Sibling index disambiguates
        // in the (unlikely) case of two entities at the same address across
        // sessions — defensive, costs nothing.
        var label = $"{entity->Name}##entity_{(nint)entity}_{siblingIndex}";

        // Dim transform-only nodes so the eye can pick out meaningful subtrees.
        bool pushedColor = false;
        if (isEmpty)
        {
            ImGuiNET.ImGui.PushStyleColor(ImGuiCol.Text, 0xFF888888u);
            pushedColor = true;
        }

        bool open = ImGuiNET.ImGui.TreeNodeEx(label, flags);
        // IsItemToggledOpen filters out the arrow-click that just expands the
        // node - we don't want expanding to also change selection.
        if (ImGuiNET.ImGui.IsItemClicked() && !ImGuiNET.ImGui.IsItemToggledOpen())
            Engine.EventBus.PublishEvent(new SceneEntitySelectedEvent(entity));

        if (pushedColor) ImGuiNET.ImGui.PopStyleColor();

        if (open && hasChildren)
        {
            DrawChildren(scene, entity);
            ImGuiNET.ImGui.TreePop();
        }
    }

    static void DrawChildren(Scene scene, Entity* entity)
    {
        var children = scene.GetChildren(entity);
        for (int i = 0; i < children.Count; i++)
            DrawNode(scene, (Entity*)children[i], i);
    }

    /// <summary>
    /// "Empty" = no MeshComponent and no LightComponent. A TransformComponent
    /// alone doesn't count — most entities have one, including intermediates.
    /// </summary>
    static bool IsEmpty(Entity* e)
        => e->GetComponent<MeshComponent>() == null
        && e->GetComponent<LightComponent>() == null;
}
