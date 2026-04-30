using System.Runtime.InteropServices;
using System.Collections.Generic;
using System.Runtime.CompilerServices;


namespace CadThingo.VulkanEngine;

/// <summary>
/// Entity is allocated on the unmanaged heap via NativeMemory so that
/// components can hold a raw Entity* without needing a GC pin.
///
/// Usage:
///   Entity* e = Entity.Create("Player");
///   e->AddComponent(new TransformComponent());
///   Entity.Destroy(e);
/// </summary>
public unsafe struct Entity : IDisposable
{
    // ── Fields ───────────────────────────────────────────────

    // Fixed-capacity component slot array — avoids List<T> and its
    // managed array resizing entirely.
    // Each slot holds the raw IntPtr of a GCHandle (stored as ulong).
    private const int MaxComponents = 64;
      
    
    private fixed ulong _componentSlots[MaxComponents];
    private int _componentCount;

    // Type-to-slot map lives on the managed heap; we reach it via GCHandle.
    private GCHandle _mapHandle;
    private GCHandle _nameHandle;

    public bool IsActive;

    // ── Factory helpers ──────────────────────────────────────

    /// <summary>Allocates an Entity in unmanaged memory.</summary>
    public static Entity* Create(string name)
    {
        Entity* e = (Entity*)NativeMemory.AllocZeroed((nuint)sizeof(Entity));
        e->IsActive = true;
        e->_componentCount = 0;
        e->_mapHandle = GCHandle.Alloc(new Dictionary<Type, int>());
        e->_nameHandle = GCHandle.Alloc(name);
        return e;
    }

    /// <summary>Disposes and frees an Entity previously created with Create().</summary>
    public static void Destroy(Entity* e)
    {
        if (e == null) return;
        e->Dispose();
        NativeMemory.Free(e);
    }

    // ── Accessors ────────────────────────────────────────────

    public string Name => (string)_nameHandle.Target!;

    private Dictionary<Type, int> ComponentMap =>
        (Dictionary<Type, int>)_mapHandle.Target!;

    // ── Lifecycle ────────────────────────────────────────────

    public void Initialize()
    {
        for (int i = 0; i < _componentCount; i++)
            GetComponentAtSlot(i)?.Initialize();
    }

    public void Update(float deltaTime)
    {
        if (!IsActive) return;
        for (int i = 0; i < _componentCount; i++)
            GetComponentAtSlot(i)?.Update(deltaTime);
    }

    public void Render()
    {
        if (!IsActive) return;
        for (int i = 0; i < _componentCount; i++)
            GetComponentAtSlot(i)?.Render();
    }
    
        // ── Component management ─────────────────────────────────
 
    /// <summary>
    /// Attaches a component. A Pinned GCHandle is stored in the fixed slot
    /// array so the GC cannot move or collect it while Entity holds the handle.
    /// Returns a raw pointer to the component for zero-overhead access.
    /// </summary>
    public T* AddComponent<T>(T component) where T : Component
    {
        var map = ComponentMap;
        if (map.TryGetValue(typeof(T), out int existingSlot))
            return GetTypedPointer<T>(existingSlot);
 
        if (_componentCount >= MaxComponents)
            throw new InvalidOperationException("Entity component capacity exceeded.");
 
        int slot = _componentCount++;
 
        // GCHandleType.Pinned fixes the object at a stable address in memory.
        // The GC will not move it, so our raw pointer remains valid.
        var handle = GCHandle.Alloc(component, GCHandleType.Pinned);
 
        fixed (ulong* slots = _componentSlots)
            slots[slot] = (ulong)(nint)GCHandle.ToIntPtr(handle);
 
        map[typeof(T)] = slot;
 
        fixed (Entity* self = &this)
            component.SetOwner(self);
 
        return GetTypedPointer<T>(slot);
    }
 
    /// <summary>
    /// O(1) typed component lookup — returns a raw pointer.
    /// No null-boxing, no managed reference root creation on the caller's stack.
    /// Caller must not store the pointer beyond the Entity's lifetime.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public T? GetComponent<T>() where T : Component
    {
        if (!ComponentMap.TryGetValue(typeof(T), out int slot))
            return null;
        return (T)GetComponentAtSlot(slot)!;
    }
    /// <summary>
    /// Returns a list of all components of type T.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <returns></returns>
    public List<T?> GetComponents<T>() where T : Component
    {
        var list = new List<T?>();
        foreach (var slot in ComponentMap.Values)
        {
            var component = GetComponentAtSlot(slot);
            if (component is T typed)
                list.Add(typed);
        }
        return list;
    }
    
    /// <summary>Detaches and disposes the component of type T.</summary>
    public bool RemoveComponent<T>() where T : Component
    {
        var map = ComponentMap;
        if (!map.TryGetValue(typeof(T), out int slot))
            return false;
 
        GetComponentAtSlot(slot)?.Dispose();
        FreeSlot(slot);
        map.Remove(typeof(T));
        return true;
    }
 
    // ── Private helpers ──────────────────────────────────────
 
    private T* GetTypedPointer<T>(int slot) where T : Component
    {
        fixed (ulong* slots = _componentSlots)
        {
            var handle = GCHandle.FromIntPtr((nint)slots[slot]);
            // AddrOfPinnedObject gives the stable unmanaged address.
            return (T*)(void*)handle.AddrOfPinnedObject();
        }
    }
 
    private Component? GetComponentAtSlot(int slot)
    {
        fixed (ulong* slots = _componentSlots)
        {
            if (slots[slot] == 0) return null;
            return GCHandle.FromIntPtr((nint)slots[slot]).Target as Component;
        }
    }
 
    private void FreeSlot(int slot)
    {
        fixed (ulong* slots = _componentSlots)
        {
            if (slots[slot] == 0) return;
            var handle = GCHandle.FromIntPtr((nint)slots[slot]);
            if (handle.IsAllocated) handle.Free();
            slots[slot] = 0;
        }
    }
 
    // ── IDisposable ──────────────────────────────────────────
 
    public void Dispose()
    {
        for (int i = 0; i < _componentCount; i++)
        {
            GetComponentAtSlot(i)?.Dispose();
            FreeSlot(i);
        }
        _componentCount = 0;
 
        if (_mapHandle.IsAllocated)  _mapHandle.Free();
        if (_nameHandle.IsAllocated) _nameHandle.Free();
    }
} 