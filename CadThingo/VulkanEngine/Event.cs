using Silk.NET.Input;

namespace CadThingo.VulkanEngine;

public abstract class Event 
{
    // You don't need this — it's already built-in
    public virtual Type GetEventType() => GetType();

    public abstract Event Clone();

    protected virtual EventCategory GetCategoryFlags() => 0;
    
    public bool IsInCategory(EventCategory category) => (GetCategoryFlags() & category) == category;
}

class WindowResizeEvent : Event
{
    private int Width;
    private int Height;
    
    public WindowResizeEvent(int w, int h)
    {
        Width = w;
        Height = h;
    }

    public int GetWidth() => Width;
    public int GetHeight() => Height;

    public override Event Clone()
    {
        return new WindowResizeEvent(Width, Height);
    }
    
    protected override EventCategory GetCategoryFlags() => EventCategory.Window;
}

class KeyPressEvent : Event
{
    private int KeyCode;
    private bool repeat;

    public KeyPressEvent(int keycode)
    {
        KeyCode = keycode;
    }

    public int GetKeyCode() => KeyCode;

    public override Event Clone()
    {
        return new KeyPressEvent(KeyCode);
    }
    protected override EventCategory GetCategoryFlags() => EventCategory.Input | EventCategory.Keyboard;
}

class KeyReleaseEvent : Event
{
    private int KeyCode;
    
    public KeyReleaseEvent(int keycode)
    {
        KeyCode = keycode;
    }

    public override Event Clone()
    {
        return new KeyReleaseEvent(KeyCode);
    }
    protected override EventCategory GetCategoryFlags() => EventCategory.Input | EventCategory.Keyboard;
}

class MouseMoveEvent : Event
{
    private float dX;
    private float dY;
    
    public MouseMoveEvent(float dx, float dy)
    {
        dX = dx;
        dY = dy;
    }
    
    public float GetX() => dX;
    public float GetY() => dY;
    
    public override Event Clone()
    {
        return new MouseMoveEvent(dX, dY);
    }
    protected override EventCategory GetCategoryFlags() => EventCategory.Input | EventCategory.Mouse;
}

class MouseKeyDownEvent : Event
{
    MouseButton KeyCode; 
    
    public MouseKeyDownEvent(MouseButton keycode)
    {
        KeyCode = keycode;
    }
    
    public override Event Clone()
    {
        return new MouseKeyDownEvent(KeyCode);
    }
    protected override EventCategory GetCategoryFlags() => EventCategory.Input | EventCategory.MouseButton;
}

class MouseKeyReleaseEvent : Event
{
    private MouseButton KeyCode;
    
    public MouseKeyReleaseEvent(MouseButton keycode)
    {
        KeyCode = keycode;
    }
    public override Event Clone()
    {
        return new MouseKeyReleaseEvent(KeyCode);
    }
    protected override EventCategory GetCategoryFlags() => EventCategory.Input | EventCategory.MouseButton;
}

class MouseScrollEvent : Event
{
    private float _y;

    public MouseScrollEvent(float y) => _y = y;

    public override Event Clone()
    {
        return new MouseScrollEvent(_y);
    }
    
    protected override EventCategory GetCategoryFlags() => EventCategory.Input | EventCategory.Mouse;
}



// public sealed unsafe class CollisionEvent : Event
// {
//     public Entity* Entity1;
//     public Entity* Entity2;
//
//     public CollisionEvent(Entity* e1, Entity* e2)
//     {
//         Entity1 = e1;
//         Entity2 = e2;
//     }
// }

public interface IEventListener
{
    void OnEvent(Event evt);
}

//event dispatcher
public class EventDispatcher
{
    private Event evt;

    public EventDispatcher(Event e)
    {
        evt = e;
    }    
    //Dispatch event to handler if the types match
    public bool Dispatch<T, F>(F handler) where T : Event where F : IEventListener
    {
        if (evt.GetEventType() == typeof(T))
        {
            handler.OnEvent(evt);
            return true;
        }
        return false;
    }
}

public class EventBus
{
    
    private Dictionary<IEventListener, EventCategory> listeners;
    private Queue<Event> eventQueue;
    private object eventQueueLock;
    private bool immediateMode = true;

    public EventBus()
    {
        listeners = new Dictionary<IEventListener, EventCategory>();
        eventQueue = new Queue<Event>();
        eventQueueLock = new object();
    }
    
    public void SetImmediateMode(bool immediate) => immediateMode = immediate;
    
    public void AddListener(IEventListener listener, EventCategory category) => listeners.Add(listener, category);
    public void RemoveListener(IEventListener listener) => listeners.Remove(listener);
        
     
    public void PublishEvent(Event evt)
    {
        if (immediateMode)
        {
            foreach (var listener in listeners)
            {
                if (evt.IsInCategory(listener.Value))
                    listener.Key.OnEvent(evt);
            }
        }
        else
        {
            lock (eventQueueLock)
            {
                eventQueue.Enqueue(evt.Clone());
                Monitor.Pulse(eventQueueLock);
            }
        }
    }
    
    public void ProcessEvents()
    {
        if (immediateMode) return;
        
        
        lock (eventQueueLock)
        {
            while (eventQueue.Count > 0)
            {
                var evt = eventQueue.Dequeue();
                foreach (var listener in listeners)
                {
                    if (evt.IsInCategory(listener.Value))
                        listener.Key.OnEvent(evt);
                }
            }
            Monitor.Pulse(eventQueueLock);
        }
    }
}



    ///<summary>
    /// DispatchEvent uses stackalloc (<=64 listeners) or ArrayPool (>64)
    /// for its snapshot - zero heap allocations.
    ///</summary>
public sealed class EventSystem
{
    private readonly List<IEventListener> _listeners = new();

    public void AddListener(IEventListener l)
    {
        if( !_listeners.Contains(l) )
            _listeners.Add(l);
    }
    public void RemoveListener(IEventListener l)
    {
        _listeners.Remove(l);
    }
    public void DispatchEvent(Event evt)
    {
        int count = _listeners.Count;
        if (count == 0) return;
        
        
        //arraypool: reuses pooled array avoids fresh heap alloc
        var rented = System.Buffers.ArrayPool<IEventListener>.Shared.Rent(count);
        _listeners.CopyTo(rented, 0);
        for(int i = 0; i < count; i++) rented[i]?.OnEvent(evt);
        System.Buffers.ArrayPool<IEventListener>.Shared.Return(rented, clearArray:true);
        
    }
}
[Flags]
public enum EventCategory
{
    None = 0,
    Application = 1 << 0,
    Input = 1 << 1,
    Keyboard = 1 << 2,
    Mouse = 1 << 3,
    MouseButton = 1 << 4,
    Window = 1 << 5,
}
