namespace CadThingo.VulkanEngine;

/// <summary>How a category's events reach their handlers.</summary>
public enum EventDelivery
{
    /// <summary>Delivered inside PublishEvent, on the calling thread. Bus thread only.</summary>
    Immediate,

    /// <summary>Queued, then delivered from ProcessEvents at a fixed point in the frame.</summary>
    Queued,
}

/// <summary>
/// Pub/sub bus for engine events. Two ways to subscribe:
///
///  - <see cref="AddListener"/>: broadcast by category mask. For consumers that want a whole
///    stream in order and switch on it themselves (ImGui, the camera).
///  - <see cref="Subscribe{T}"/>: one event type, one delegate. For intents with a single owner
///    (a settings panel asking for a tonemap rebuild), where a category switch is pure noise.
///
/// Both return a token; disposing it unsubscribes. Features hold their tokens and dispose them
/// with themselves, so a torn-down feature cannot be handed an event and dereference GPU handles
/// it has already destroyed.
///
/// Threading: publishing is safe from any thread. Delivery is not - handlers touch Vulkan, ImGui
/// and scene state, all of which belong to the thread that called <see cref="BindToCurrentThread"/>
/// (the main thread). Anything published from off that thread is therefore queued regardless of
/// its category's delivery mode, and lands in the next <see cref="ProcessEvents"/>. That is the
/// path the resource worker uses to report a finished load.
/// </summary>
public sealed class EventBus
{
    /// <summary>
    /// Cap on drain passes in one <see cref="ProcessEvents"/>. A handler is allowed to publish,
    /// and what it publishes is picked up by the next pass rather than mutating the buffer being
    /// iterated. The cap turns a publish cycle into a warning instead of a hung frame.
    /// </summary>
    private const int MaxDrainPasses = 8;

    private sealed class Subscription : IDisposable
    {
        public readonly EventBus Bus;
        public readonly Type? Type;          // null for a category broadcast subscription
        public readonly EventCategory Mask;  // only meaningful when Type is null
        public readonly object? Owner;       // listener instance, for RemoveListener
        public readonly Action<Event> Invoke;
        public readonly int Order;
        public readonly bool SkipHandled;

        // Read by the dispatch loop off a snapshot array, written by Dispose from any thread.
        // Volatile so a mid-dispatch unsubscribe takes effect for handlers not yet reached in
        // this same event, rather than only from the next publish.
        public volatile bool Alive = true;

        public Subscription(EventBus bus, Type? type, EventCategory mask, object? owner,
                            Action<Event> invoke, int order, bool skipHandled)
        {
            Bus = bus;
            Type = type;
            Mask = mask;
            Owner = owner;
            Invoke = invoke;
            Order = order;
            SkipHandled = skipHandled;
        }

        public void Dispose()
        {
            if (!Alive) return;
            Alive = false;
            Bus.Unregister(this);
        }
    }

    // Subscription state is copy-on-write: writers rebuild an array under _writeLock and publish
    // it by reference assignment, readers just grab the reference. The dispatch loop therefore
    // iterates an array that cannot change under it, which is what makes subscribing or
    // unsubscribing from inside a handler safe (the old code iterated the live dictionary and
    // threw InvalidOperationException the moment a handler touched the listener set).
    private readonly object _writeLock = new();
    private volatile Dictionary<Type, Subscription[]> _typed = new();
    private volatile Subscription[] _broadcast = [];

    // Deferred queue, double-buffered so the drain never holds the lock while running handlers.
    private readonly object _queueLock = new();
    private Queue<Event> _queue = new();
    private Queue<Event> _drain = new();

    // Which categories are queued, as a bitmask rather than a per-category map: PublishEvent is
    // on the mouse-move path, so resolving delivery has to be one masked compare with no lock and
    // no allocation. Input and window events stay immediate (latency, and the previous
    // behaviour); renderer intents are queued so a UI callback cannot rebuild a pipeline
    // part-way through a frame.
    private volatile uint _queuedMask = (uint)EventCategory.Renderer;

    private int _busThreadId = Environment.CurrentManagedThreadId;

    /// <summary>True when the caller is on the thread that owns delivery.</summary>
    public bool OnBusThread => Environment.CurrentManagedThreadId == _busThreadId;

    /// <summary>
    /// Names the calling thread as the one events are delivered on. The constructor assumes
    /// whichever thread built the bus; Engine.Start calls this explicitly so the binding does not
    /// depend on where a static initialiser happened to run.
    /// </summary>
    public void BindToCurrentThread() => _busThreadId = Environment.CurrentManagedThreadId;

    /// <summary>Overrides how one category is delivered. Affects events published afterwards.</summary>
    public void SetDelivery(EventCategory category, EventDelivery mode)
    {
        lock (_writeLock)
        {
            _queuedMask = mode == EventDelivery.Queued
                ? _queuedMask | (uint)category
                : _queuedMask & ~(uint)category;
        }
    }

    // ---- subscribing -------------------------------------------------------------------------

    /// <summary>
    /// Broadcast subscription: <paramref name="listener"/> receives every event sharing any bit
    /// with <paramref name="category"/>. Lower <paramref name="order"/> runs first; a handler
    /// registered with <paramref name="skipHandled"/> is passed over once something earlier has
    /// set <see cref="Event.Handled"/>.
    /// </summary>
    public IDisposable AddListener(IEventListener listener, EventCategory category,
                                   int order = 0, bool skipHandled = false)
    {
        ArgumentNullException.ThrowIfNull(listener);
        var sub = new Subscription(this, null, category, listener, listener.OnEvent, order, skipHandled);
        Register(sub);
        return sub;
    }

    /// <summary>
    /// Drops every subscription made by <paramref name="listener"/> through
    /// <see cref="AddListener"/>. Prefer disposing the token; this exists for listeners that do
    /// not keep one.
    /// </summary>
    public void RemoveListener(IEventListener listener)
    {
        lock (_writeLock)
        {
            var live = _broadcast;
            var keep = new List<Subscription>(live.Length);
            foreach (var sub in live)
            {
                if (ReferenceEquals(sub.Owner, listener)) sub.Alive = false;
                else keep.Add(sub);
            }

            if (keep.Count != live.Length) _broadcast = keep.ToArray();
        }
    }

    /// <summary>
    /// Typed subscription: <paramref name="handler"/> receives only <typeparamref name="T"/>, so
    /// there is no category switch to write. Dispose the returned token to unsubscribe.
    /// </summary>
    public IDisposable Subscribe<T>(Action<T> handler, int order = 0, bool skipHandled = false)
        where T : Event
    {
        ArgumentNullException.ThrowIfNull(handler);
        var sub = new Subscription(this, typeof(T), EventCategory.None, handler,
                                   e => handler((T)e), order, skipHandled);
        Register(sub);
        return sub;
    }

    /// <summary>
    /// Completes on the next <typeparamref name="T"/>, for async flows that need to wait on a
    /// point in the frame (a swapchain rebuild, a finished bake) without wiring a callback.
    ///
    /// The continuation resumes on the thread pool, never inline on the drain: an await that
    /// resumed inside dispatch would run engine work part-way through event delivery. Publish an
    /// event to get back onto the bus thread if the continuation needs to touch GPU state.
    /// </summary>
    public Task<T> NextAsync<T>(CancellationToken ct = default) where T : Event
    {
        var tcs = new TaskCompletionSource<T>(TaskCreationOptions.RunContinuationsAsynchronously);

        // Built before Register so the one-shot dispose below cannot observe a null: the
        // subscription is only reachable by the dispatcher after Register publishes the array.
        Subscription? sub = null;
        sub = new Subscription(this, typeof(T), EventCategory.None, null,
                               e =>
                               {
                                   if (tcs.TrySetResult((T)e)) sub!.Dispose();
                               },
                               order: 0, skipHandled: false);
        Register(sub);

        if (ct.CanBeCanceled)
        {
            var reg = ct.Register(() =>
            {
                if (tcs.TrySetCanceled(ct)) sub!.Dispose();
            });
            tcs.Task.ContinueWith(static (_, state) => ((CancellationTokenRegistration)state!).Dispose(),
                                  reg, TaskContinuationOptions.ExecuteSynchronously);
        }

        return tcs.Task;
    }

    private void Register(Subscription sub)
    {
        lock (_writeLock)
        {
            if (sub.Type is null)
            {
                _broadcast = Insert(_broadcast, sub);
                return;
            }

            // Copy the dictionary too, not just the bucket: readers hold the map reference for
            // the duration of a dispatch and must not see a rehash.
            var next = new Dictionary<Type, Subscription[]>(_typed);
            next[sub.Type] = Insert(next.GetValueOrDefault(sub.Type, []), sub);
            _typed = next;
        }
    }

    private void Unregister(Subscription sub)
    {
        lock (_writeLock)
        {
            if (sub.Type is null)
            {
                _broadcast = Remove(_broadcast, sub);
                return;
            }

            if (!_typed.TryGetValue(sub.Type, out var bucket)) return;

            var next = new Dictionary<Type, Subscription[]>(_typed);
            var pruned = Remove(bucket, sub);
            if (pruned.Length == 0) next.Remove(sub.Type);
            else next[sub.Type] = pruned;
            _typed = next;
        }
    }

    // Insertion keeps the array sorted by Order, and stable within an Order so subscribers
    // registered at the same priority still run in registration order.
    private static Subscription[] Insert(Subscription[] source, Subscription sub)
    {
        int at = source.Length;
        for (int i = 0; i < source.Length; i++)
        {
            if (source[i].Order > sub.Order) { at = i; break; }
        }

        var next = new Subscription[source.Length + 1];
        Array.Copy(source, 0, next, 0, at);
        next[at] = sub;
        Array.Copy(source, at, next, at + 1, source.Length - at);
        return next;
    }

    private static Subscription[] Remove(Subscription[] source, Subscription sub)
    {
        int at = Array.IndexOf(source, sub);
        if (at < 0) return source;

        var next = new Subscription[source.Length - 1];
        Array.Copy(source, 0, next, 0, at);
        Array.Copy(source, at + 1, next, at, source.Length - at - 1);
        return next;
    }

    // ---- publishing --------------------------------------------------------------------------

    /// <summary>
    /// Publishes an event. Safe from any thread. Delivered inline only when the category is
    /// Immediate and the caller is on the bus thread; otherwise queued for
    /// <see cref="ProcessEvents"/>.
    /// </summary>
    public void PublishEvent(Event evt)
    {
        ArgumentNullException.ThrowIfNull(evt);

        if (OnBusThread && ResolveDelivery(evt.Category) == EventDelivery.Immediate)
        {
            Dispatch(evt);
            return;
        }

        lock (_queueLock) _queue.Enqueue(evt);
    }

    /// <summary>
    /// Delivers everything queued since the last call. Must run on the bus thread, at one fixed
    /// point in the frame.
    /// </summary>
    public void ProcessEvents()
    {
        for (int pass = 0; pass < MaxDrainPasses; pass++)
        {
            lock (_queueLock)
            {
                if (_queue.Count == 0) return;

                // Normally _drain is empty and the buffers just swap. It is only non-empty when a
                // handler threw out of a previous drain and left events behind; append rather
                // than swap so those keep their place at the front.
                if (_drain.Count == 0) (_queue, _drain) = (_drain, _queue);
                else while (_queue.Count > 0) _drain.Enqueue(_queue.Dequeue());
            }

            while (_drain.Count > 0)
                Dispatch(_drain.Dequeue());
        }

        lock (_queueLock)
        {
            if (_queue.Count > 0)
                Console.Error.WriteLine(
                    $"[EventBus] Drain hit {MaxDrainPasses} passes with {_queue.Count} event(s) still " +
                    "queued - a handler is publishing in a cycle. Deferred to next frame.");
        }
    }

    /// <summary>Queue depth. Diagnostics only.</summary>
    public int PendingCount
    {
        get { lock (_queueLock) return _queue.Count; }
    }

    private void Dispatch(Event evt)
    {
        // Broadcast first: those are the input consumers, and they are the ones that decide
        // Handled. Typed subscribers (single-owner intents) see the outcome of that decision.
        Invoke(_broadcast, evt, broadcast: true);

        // Walk the base chain so a subscription to a future grouping type (a MouseEvent base,
        // say) still fires. The old EventDispatcher compared types exactly and silently missed
        // every subclass.
        var typed = _typed;
        for (Type? t = evt.GetType(); t is not null && t != typeof(object); t = t.BaseType)
        {
            if (typed.TryGetValue(t, out var bucket))
                Invoke(bucket, evt, broadcast: false);
        }
    }

    private static void Invoke(Subscription[] subs, Event evt, bool broadcast)
    {
        foreach (var sub in subs)
        {
            if (!sub.Alive) continue;
            if (evt.Handled && sub.SkipHandled) continue;
            if (broadcast && !evt.IsInCategory(sub.Mask)) continue;
            sub.Invoke(evt);
        }
    }

    /// <summary>
    /// An event carrying several category bits is queued if any of them is queued. Delivery mode
    /// is a safety property, so the stricter bit wins.
    /// </summary>
    private EventDelivery ResolveDelivery(EventCategory category) =>
        ((uint)category & _queuedMask) != 0 ? EventDelivery.Queued : EventDelivery.Immediate;
}