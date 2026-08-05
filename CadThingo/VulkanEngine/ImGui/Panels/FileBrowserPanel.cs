using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using CadThingo.VulkanEngine.GLTF;
using ImGuiNET;

namespace CadThingo.VulkanEngine.ImGui.Panels;

/// <summary>
/// Per-file state machine:<br/>
///   Loaded   → entities present in the scene, drawing.<br/>
///   Detached → entities still alive (Entity* pointers held by the panel) but
///              removed from Scene._entityList. GPU resources untouched.<br/>
///   Unloaded → entities destroyed (Entity.Destroy). GPU mesh/texture memory is
///              not reclaimed (engine has no per-file free path), but the panel
///              remembers the path so a single-click reload re-imports the file.
/// </summary>
public enum FileLoadState
{
    Loaded,
    Detached,
    Unloaded,
}

internal sealed unsafe class LoadedSceneFile
{
    public string         Path        = "";
    public string         DisplayName = "";
    public string         IdPrefix    = "";
    public FileLoadState  State       = FileLoadState.Loaded;
    public Entity*        Root;
    // Pre-order list of every entity GltfLoader produced for this file.
    // Stored as nint so the managed List doesn't need pinning.
    public List<nint>     Entities    = new();
    public bool           Visible     = true;
    // Resource manifest filled by GltfLoader.Load — used by Destroy to free
    // mesh ranges, texture VkImages, bindless slots, and BLAS entries.
    public LoadManifest?  Manifest;
}

/// <summary>
/// File browser and scene loader, and the only path that calls <c>GltfLoader.Load</c>. Listens for
/// File / Open in the main menu, and for the Browse button on the panel.
/// </summary>
public static unsafe class FileBrowserPanel
{
    static readonly List<LoadedSceneFile> _files = new();
    static int _loadCounter; // disambiguates idPrefixes when the user loads the same path twice

    /// <summary>Returns the default directory the file picker opens to.</summary>
    static string InitialDirectory()
    {
        // App-relative Assets folder first (Debug/net*/Assets/Models when running
        // out of bin/), falling back to the source tree the rest of the codebase
        // already hardcodes for the HDR picker.
        string app = Path.Combine(AppContext.BaseDirectory, "Assets", "Models");
        if (Directory.Exists(app)) return app;

        string src = @"C:\Users\jamie\RiderProjects\CadThingo\CadThingo\Assets\Models";
        if (Directory.Exists(src)) return src;

        return AppContext.BaseDirectory;
    }

    /// <summary>Opens the native file picker and loads the chosen file as a new entry.</summary>
    public static void BrowseAndLoad()
    {
        var path = FilePicker.Open(
            "Open scene file",
            InitialDirectory(),
            "glTF / GLB", "*.glb;*.gltf",
            "All files",  "*.*");
        if (string.IsNullOrEmpty(path)) return;
        LoadFromPath(path);
    }

    static void LoadFromPath(string path)
    {
        var renderer = Engine.renderer;
        var scene    = renderer?.Scene;
        if (renderer == null || scene == null)
        {
            Console.Error.WriteLine("[FileBrowser] Renderer/scene not initialized.");
            return;
        }

        // Unique idPrefix per load so re-importing the same file after destroy
        // doesn't collide with stale ResourceManager entries (those keep mesh*
        // pointers that the destroyed entities used to reference).
        string display  = Path.GetFileName(path);
        string idPrefix = $"{Path.GetFileNameWithoutExtension(path)}#{_loadCounter++}";

        GltfLoader.LoadResult result;
        try
        {
            result = GltfLoader.Load(path, idPrefix, Engine.ResourceManager, renderer.gfx, scene);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[FileBrowser] GltfLoader.Load failed for '{path}': {ex}");
            return;
        }

        var entry = new LoadedSceneFile
        {
            Path        = path,
            DisplayName = display,
            IdPrefix    = idPrefix,
            Root        = result.Root,
            State       = FileLoadState.Loaded,
            Visible     = true,
            Manifest    = result.Manifest,
            Entities    = new List<nint>(result.Manifest.Entities),
        };

        _files.Add(entry);
        renderer.OnSceneEntitiesChanged();
    }

    //  ImGui 
    public static void Draw()
    {
        if (!EditorState.ShowFileBrowser) return;

        if (!ImGuiNET.ImGui.Begin("Files", ref EditorState.ShowFileBrowser))
        {
            ImGuiNET.ImGui.End();
            return;
        }

        if (ImGuiNET.ImGui.Button("Browse..."))
            BrowseAndLoad();
        ImGuiNET.ImGui.SameLine();
        ImGuiNET.ImGui.TextDisabled($"{_files.Count} file{(_files.Count == 1 ? "" : "s")}");
        
        // Reclaimed VB/IB bytes on the global mesh buffers. Useful sanity check
        // that destroying a file actually returns its mesh ranges to the
        // free-list. Values stay at zero until the first destroy.
        var (vbFree, ibFree) = Engine.ResourceManager.GetMeshFreeStats();
        if (vbFree > 0 || ibFree > 0)
        {
            ImGuiNET.ImGui.SameLine();
            ImGuiNET.ImGui.TextDisabled($"  free: VB {vbFree}  IB {ibFree}");
        }

        ImGuiNET.ImGui.Separator();

        if (_files.Count == 0)
        {
            ImGuiNET.ImGui.TextDisabled("No files loaded. Use Browse to add a .glb / .gltf.");
            ImGuiNET.ImGui.End();
            return;
        }

        // Deferred actions — we can't mutate _files while iterating it.
        LoadedSceneFile? toRemove = null;

        for (int i = 0; i < _files.Count; i++)
        {
            var f = _files[i];
            ImGuiNET.ImGui.PushID(i);

            DrawRow(f, ref toRemove);

            ImGuiNET.ImGui.PopID();
        }

        if (toRemove != null)
        {
            // Destroy + drop the entry entirely.
            DestroyEntities(toRemove);
            _files.Remove(toRemove);
            Engine.renderer?.OnSceneEntitiesChanged();
        }

        ImGuiNET.ImGui.End();
    }

    static void DrawRow(LoadedSceneFile f, ref LoadedSceneFile? toRemove)
    {
        // Header line: visible-checkbox + name + state badge.
        bool visible = f.Visible;
        if (ImGuiNET.ImGui.Checkbox("##vis", ref visible))
        {
            f.Visible = visible;
            if (f.State == FileLoadState.Loaded)
                SetSubtreeActive(f, visible);
        }
        if (ImGuiNET.ImGui.IsItemHovered())
            ImGuiNET.ImGui.SetTooltip("Toggle visibility (IsActive on every entity in this file)");

        ImGuiNET.ImGui.SameLine();
        ImGuiNET.ImGui.TextUnformatted(f.DisplayName);
        if (ImGuiNET.ImGui.IsItemHovered())
            ImGuiNET.ImGui.SetTooltip(f.Path);

        ImGuiNET.ImGui.SameLine();
        DrawStateBadge(f.State);

        // Action row: Detach/Attach, Destroy, Reload.
        ImGuiNET.ImGui.Indent();
        switch (f.State)
        {
            case FileLoadState.Loaded:
                if (ImGuiNET.ImGui.SmallButton("Detach")) Detach(f);
                ImGuiNET.ImGui.SameLine();
                if (ImGuiNET.ImGui.SmallButton("Destroy")) toRemove = f;
                break;

            case FileLoadState.Detached:
                if (ImGuiNET.ImGui.SmallButton("Re-attach")) Reattach(f);
                ImGuiNET.ImGui.SameLine();
                if (ImGuiNET.ImGui.SmallButton("Destroy")) toRemove = f;
                break;

            case FileLoadState.Unloaded:
                if (ImGuiNET.ImGui.SmallButton("Reload")) Reload(f);
                ImGuiNET.ImGui.SameLine();
                if (ImGuiNET.ImGui.SmallButton("Remove from list")) toRemove = f;
                break;
        }
        ImGuiNET.ImGui.Text($"{f.Entities.Count} entities");
        ImGuiNET.ImGui.Unindent();

        ImGuiNET.ImGui.Separator();
    }

    static void DrawStateBadge(FileLoadState state)
    {
        var (label, color) = state switch
        {
            FileLoadState.Loaded   => ("LOADED",   new Vector4(0.3f, 0.9f, 0.4f, 1f)),
            FileLoadState.Detached => ("DETACHED", new Vector4(0.9f, 0.7f, 0.2f, 1f)),
            FileLoadState.Unloaded => ("UNLOADED", new Vector4(0.7f, 0.3f, 0.3f, 1f)),
            _                      => ("?",        new Vector4(1f, 1f, 1f, 1f)),
        };
        ImGuiNET.ImGui.TextColored(color, label);
    }

    //  State transitions

    static void SetSubtreeActive(LoadedSceneFile f, bool active)
    {
        foreach (var ptr in f.Entities)
        {
            var e = (Entity*)ptr;
            if (e == null) continue;
            e->IsActive = active;
        }
        // RebuildTlas now skips !IsActive entities, so a visibility toggle
        // genuinely removes them from the ray-traced shadow set after the next
        // TLAS flush. PT accumulator restarts because the lit scene changed.
        // Both of those are what SceneDirtyEvent means, so the panel states the
        // fact and leaves the renderer to decide what it invalidates.
        Engine.EventBus.PublishEvent(new SceneDirtyEvent());
    }

    static void Detach(LoadedSceneFile f)
    {
        var scene = Engine.renderer?.Scene;
        if (scene == null || f.Root == null) return;

        // RemoveSubtree leaves TransformComponent.Parent pointers intact so
        // re-attach can rebuild the hierarchy by re-running AddEntity in order.
        scene.RemoveSubtree(f.Root);
        f.State = FileLoadState.Detached;
        Engine.renderer?.OnSceneEntitiesChanged();
    }

    static void Reattach(LoadedSceneFile f)
    {
        var scene = Engine.renderer?.Scene;
        if (scene == null) return;

        foreach (var ptr in f.Entities)
        {
            var e = (Entity*)ptr;
            if (e == null) continue;
            scene.AddEntity(e);
            // Restore visibility flag — Detach didn't touch it, but if a user
            // detaches a hidden file then re-attaches, they probably want the
            // panel's Visible toggle to drive IsActive again.
            e->IsActive = f.Visible;
        }
        f.State = FileLoadState.Loaded;
        Engine.renderer?.OnSceneEntitiesChanged();
    }

    static void DestroyEntities(LoadedSceneFile f)
    {
        var renderer = Engine.renderer;
        var scene    = renderer?.Scene;

        // If still in scene, detach first so we don't leave dangling pointers
        // in _entityList / _roots / _childrenOf.
        if (f.State == FileLoadState.Loaded && scene != null && f.Root != null)
            scene.RemoveSubtree(f.Root);

        // Destroy in reverse order so children go before parents — TransformComponent's
        // Parent pointer becomes dangling for a moment otherwise (harmless today,
        // defensive against future Component.Dispose code reading it).
        for (int i = f.Entities.Count - 1; i >= 0; i--)
        {
            var e = (Entity*)f.Entities[i];
            if (e == null) continue;
            Entity.Destroy(e);
        }
        f.Entities.Clear();
        f.Root = null;

        // Drain the device before destroying GPU-side resources — BLAS storage
        // and texture VkImages can still be referenced by command buffers that
        // captured them last frame. ResourceManager.UnregisterBindless rewrites
        // the descriptor slot to the white default, which also wants idle.
        var manifest = f.Manifest;
        if (renderer != null && manifest != null)
        {
            FreeManifestResources(renderer, manifest);
        }
        f.Manifest = null;
        f.State    = FileLoadState.Unloaded;
    }

    /// <summary>
    /// Walks every per-file resource on the manifest and frees its GPU memory:
    /// BLAS entries, bindless slots, texture VkImages, mesh VB/IB ranges, and
    /// the scene material slots so the next load reuses indices instead of
    /// pushing past MAX_MATERIALS. Bindless slots are rewritten to the white
    /// default first so any stale material entry still pointing at them reads
    /// safe data.
    /// </summary>
    static void FreeManifestResources(Renderer.Renderer renderer, LoadManifest m)
    {
        // Single DeviceWaitIdle covers every Vk*Destroy below.
        renderer.vk!.DeviceWaitIdle(renderer.device);

        // BLAS first — they reference the global VB/IB ranges we're about to
        // mark free. After this call the cache no longer holds those handles,
        // so the next OnSceneEntitiesChanged won't rediscover them.
        renderer.DestroyBlasFor(m.MeshPtrs);

        // Bindless slots → white default, then return to free stack. We use
        // GltfDefaults.BaseColor (registered globally, never freed) so we can
        // be sure the descriptor target survives.
        var rm = Engine.ResourceManager;
        foreach (var idx in m.BindlessIndices)
            rm.UnregisterBindless(idx, GltfDefaults.BaseColor);

        // Resource buckets — Texture/Mesh resources. Releasing a MeshResource
        // also returns its (vbOffset, ibOffset, ...) range to the free-list
        // via MeshResource.Unload → manager.FreeMesh.
        foreach (var (type, ids) in m.ResourceIdsByType)
        {
            foreach (var id in ids)
                ReleaseByRuntimeType(rm, type, id);
        }

        // Material slots → scene free stack. Comes after the bindless unregister
        // so the zeroed-out PbrMaterial entries (BaseColorTex=0 etc.) land on a
        // descriptor slot that's still guaranteed to point at GltfDefaults.BaseColor.
        var scene = renderer.Scene;
        if (scene != null)
        {
            foreach (var idx in m.MaterialIndices)
                scene.ReleaseMaterial(idx);
        }
    }

    // Reflection trampoline so we can call ResourceManager.ReleaseResource<T>
    // with the runtime Type captured on the manifest. Cached MethodInfo to
    // keep the destroy path's overhead negligible compared to the underlying
    // DeviceWaitIdle.
    static readonly System.Reflection.MethodInfo _releaseGeneric =
        typeof(ResourceManager).GetMethod(nameof(ResourceManager.ReleaseResource))!;
    static readonly Dictionary<Type, System.Reflection.MethodInfo> _releaseCache = new();

    static void ReleaseByRuntimeType(ResourceManager rm, Type t, string id)
    {
        if (!_releaseCache.TryGetValue(t, out var mi))
        {
            mi = _releaseGeneric.MakeGenericMethod(t);
            _releaseCache[t] = mi;
        }
        mi.Invoke(rm, new object[] { id });
    }

    static void Reload(LoadedSceneFile f)
    {
        // Reload after a destroy goes through the full file parse + upload
        // path again. We use a fresh idPrefix so the new resources land in
        // newly-allocated free-list slots / bindless indices — the previous
        // load's slots have been freed and could be reused by another file.
        var renderer = Engine.renderer;
        var scene    = renderer?.Scene;
        if (renderer == null || scene == null) return;

        string idPrefix = $"{Path.GetFileNameWithoutExtension(f.Path)}#{_loadCounter++}";

        GltfLoader.LoadResult result;
        try
        {
            result = GltfLoader.Load(f.Path, idPrefix, Engine.ResourceManager, renderer.gfx, scene);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[FileBrowser] Reload failed for '{f.Path}': {ex.Message}");
            return;
        }

        f.IdPrefix  = idPrefix;
        f.Root      = result.Root;
        f.Manifest  = result.Manifest;
        f.Entities  = new List<nint>(result.Manifest.Entities);
        f.State     = FileLoadState.Loaded;

        // Restore the user's last-seen visibility preference.
        if (!f.Visible) SetSubtreeActive(f, false);

        renderer.OnSceneEntitiesChanged();
    }
}