using CadThingo.VulkanEngine.Renderer.Features.Tonemapping;
using Silk.NET.Input;

namespace CadThingo.VulkanEngine;

/// <summary>
/// Base for every event on the <see cref="EventBus"/>.
///
/// Payload is immutable: an event is built at the publish site, handed to the bus and never
/// touched again. That is what makes it safe to publish from a background thread (the resource
/// worker) and deliver on the main thread without copying it first. <see cref="Handled"/> is the
/// one mutable member, and it is dispatch state rather than payload - only ever written by a
/// handler, only ever on the thread that is draining the bus.
/// </summary>
public abstract class Event
{
    /// <summary>Category bits this event carries. Broadcast listeners subscribe by mask.</summary>
    public abstract EventCategory Category { get; }

    /// <summary>
    /// Marks the event consumed. Subscribers registered with skipHandled:true are skipped once
    /// this is set, so an earlier-ordered handler (ImGui, when it wants the keyboard) can stop a
    /// later one (the camera) from acting on the same input.
    /// </summary>
    public bool Handled { get; set; }

    /// <summary>True if this event carries any of the bits in <paramref name="mask"/>.</summary>
    public bool IsInCategory(EventCategory mask) => (Category & mask) != 0;
}

sealed class WindowResizeEvent : Event
{
    private readonly int Width;
    private readonly int Height;

    public WindowResizeEvent(int w, int h)
    {
        Width = w;
        Height = h;
    }

    public int GetWidth() => Width;
    public int GetHeight() => Height;

    public override EventCategory Category => EventCategory.Window;
}

sealed class KeyPressEvent : Event
{
    private readonly Key KeyCode;
    private readonly IKeyboard kb;

    public KeyPressEvent(IKeyboard kb, Key keycode)
    {
        KeyCode = keycode;
        this.kb = kb;
    }

    public Key GetKeyCode => KeyCode;
    public IKeyboard GetKeyboard => kb;

    public override EventCategory Category => EventCategory.Input | EventCategory.Keyboard;
}

sealed class KeyReleaseEvent : Event
{
    private readonly Key KeyCode;
    private readonly IKeyboard kb;

    public KeyReleaseEvent(IKeyboard kb, Key keycode)
    {
        KeyCode = keycode;
        this.kb = kb;
    }

    public Key GetKeyCode => KeyCode;
    public IKeyboard GetKeyboard => kb;

    public override EventCategory Category => EventCategory.Input | EventCategory.Keyboard;
}

sealed class MouseMoveEvent : Event
{
    private readonly float absX;
    private readonly float absY;
    private readonly float dX;
    private readonly float dY;
    private readonly IMouse mouse;

    public MouseMoveEvent(IMouse mouse, float absX, float absY, float dx, float dy)
    {
        this.mouse = mouse;
        this.absX = absX;
        this.absY = absY;
        dX = dx;
        dY = dy;
    }

    public float GetAbsX() => absX;
    public float GetAbsY() => absY;
    public float GetX() => dX;
    public float GetY() => dY;
    public IMouse GetMouse => mouse;

    public override EventCategory Category => EventCategory.Input | EventCategory.Mouse;
}

sealed class MouseKeyDownEvent : Event
{
    private readonly MouseButton KeyCode;
    private readonly IMouse mouse;

    public MouseKeyDownEvent(IMouse mouse, MouseButton keycode)
    {
        this.mouse = mouse;
        KeyCode = keycode;
    }

    public MouseButton GetButton => KeyCode;
    public IMouse GetMouse => mouse;

    public override EventCategory Category => EventCategory.Input | EventCategory.MouseButton;
}

sealed class MouseKeyReleaseEvent : Event
{
    private readonly MouseButton KeyCode;
    private readonly IMouse mouse;

    public MouseKeyReleaseEvent(IMouse mouse, MouseButton keycode)
    {
        this.mouse = mouse;
        KeyCode = keycode;
    }

    public MouseButton GetButton => KeyCode;
    public IMouse GetMouse => mouse;

    public override EventCategory Category => EventCategory.Input | EventCategory.MouseButton;
}

sealed class MouseScrollEvent : Event
{
    private readonly float _x;
    private readonly float _y;
    private readonly IMouse mouse;

    public MouseScrollEvent(IMouse mouse, float x, float y)
    {
        this.mouse = mouse;
        _x = x;
        _y = y;
    }

    public float GetX() => _x;
    public float GetY() => _y;
    public IMouse GetMouse => mouse;

    public override EventCategory Category => EventCategory.Input | EventCategory.Mouse;
}

// ---- Renderer / editor intents -------------------------------------------------------------
// These carry the Renderer category, which the bus delivers queued: they are published from UI
// callbacks and from the resource worker, and their handlers rebuild pipelines or touch GPU
// state, so they must land at the frame's drain point rather than part-way through a frame.
// Drain runs in Engine's Update handler, ahead of DrawFrame, so an intent raised while drawing
// frame N is applied at the top of frame N+1 - the same timing the pending-flag fields had.

/// <summary>
/// The scene's contents or transforms changed: the TLAS is stale and any progressive
/// integration has to restart. Publish this instead of reaching for MarkTlasDirty +
/// MarkAccumulatorDirty from code that has no business holding a Renderer.
/// </summary>
sealed class SceneDirtyEvent : Event
{
    public override EventCategory Category => EventCategory.Renderer;
}

/// <summary>
/// Something changed what the camera sees without changing the scene, so accumulation restarts
/// but the TLAS stands. Lighter than <see cref="SceneDirtyEvent"/> - use that one for edits.
/// </summary>
sealed class PathTracingAccumulatorInvalidatedEvent : Event
{
    public override EventCategory Category => EventCategory.Renderer;
}

/// <summary>
/// Requests the tonemap curve. The operator is a specialization constant, so the handler stores
/// it and flags a pipeline rebuild rather than applying it live.
/// </summary>
sealed class TonemapFilterChangedEvent : Event
{
    private readonly TonemapOperator _operator;

    public TonemapFilterChangedEvent(TonemapOperator op) => _operator = op;

    public TonemapOperator GetOperator => _operator;

    public override EventCategory Category => EventCategory.Renderer;
}

/// <summary>
/// Requests PCSS-style soft shadows on or off. Also a specialization constant, and it feeds two
/// pipelines (deferred PBR + transparent forward), so it is a rebuild rather than a live flag.
/// </summary>
sealed class PbrSoftShadowingChangedEvent : Event
{
    private readonly bool _enabled;

    public PbrSoftShadowingChangedEvent(bool enabled) => _enabled = enabled;

    public bool GetEnabled => _enabled;

    public override EventCategory Category => EventCategory.Renderer;
}

/// <summary>
/// Selection changed. Carries the raw entity pointer, which is why this one is Editor category
/// and therefore delivered immediately: a queued selection could be drained after the entity it
/// names has been destroyed, leaving handlers with a dangling pointer.
/// </summary>
sealed unsafe class SceneEntitySelectedEvent : Event
{
    private readonly Entity* entity;

    public SceneEntitySelectedEvent(Entity* entity) => this.entity = entity;

    /// <summary>The newly selected entity, or null when the selection was cleared.</summary>
    public Entity* GetEntity => entity;

    public override EventCategory Category => EventCategory.Editor;
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