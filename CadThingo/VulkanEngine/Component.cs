namespace CadThingo.VulkanEngine;

public abstract unsafe class Component : IDisposable
{
    public enum State { Uninitialized, Initializing, Active, Destroying, Destroyed }
 
    private State _state = State.Uninitialized;
    private bool  _disposed;
 
    // Stored as a raw pointer rather than a managed reference.
    // The Entity owns its own memory; this pointer is valid for the
    // component's entire lifetime, so no GC pin is required.
    private Entity* _owner;
 
    public bool    IsActive => _state == State.Active;
    public Entity* Owner    => _owner;
 
    // Called only by Entity — not public API.
    internal void SetOwner(Entity* owner) => _owner = owner;
 
    // ── Lifecycle ────────────────────────────────────────────
 
    public void Initialize()
    {
        if (_state != State.Uninitialized) return;
        _state = State.Initializing;
        OnInitialize();
        _state = State.Active;
    }
 
    public void Destroy()
    {
        if (_state != State.Active) return;
        _state = State.Destroying;
        OnDestroy();
        _state = State.Destroyed;
    }
 
    protected virtual void OnInitialize() { }
    protected virtual void OnDestroy()    { }
 
    public virtual void Update(float deltaTime) { }
    public virtual void Render()               { }
 
    // ── IDisposable ──────────────────────────────────────────
 
    public void Dispose()
    {
        if (_disposed) return;
        if (_state != State.Destroyed)
        {
            OnDestroy();
            _state = State.Destroyed;
        }
        _disposed = true;
        GC.SuppressFinalize(this);
    }
}


public static class ComponentTypeID
{
    private static int _nextID = 0;
    
    public static int For<T>() where T : Component => TypeIdHolder<T>.ID;
    
    private static class TypeIdHolder<T> where T : Component
    {
        public static readonly int ID = Interlocked.Increment(ref _nextID) -1;
    }
}
