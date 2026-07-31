using System.Diagnostics;
using CadThingo.VulkanEngine.Renderer.Slang;
using Silk.NET.Vulkan;
using Silk.NET.Vulkan.Extensions.EXT;

namespace CadThingo.VulkanEngine.Renderer.FrameGraph;

public unsafe class FrameGraph : IDisposable
{
    private readonly GraphicsDevice gfx;

    private int _nextResourceId;
    private bool _disposed;
    // Restores imported images to their declared final layout after the last pass, so the
    // baked first-use barriers stay valid every frame. Empty when nothing needs restoring.
    private ImageMemoryBarrier2[] _closingImageBarriers = [];

    // Debug/profiling: GPU+CPU timings, pipeline stats, debug-utils labels/names. Created
    // in Compile once the schedule is known; null only before the first Compile.
    private GraphDebug? _debug;
    private double _compileMs;
    private int _culledCount;
    private int _barrierCount;
    private bool[] _live = [];   // per-pass-id liveness, kept for ToDot
    // Labelled DAG edges captured during Compile, purely for ToDot (allows duplicates).
    private readonly List<(int from, int to, string label, bool dashed)> _dotEdges = [];

    // ---- Multi-queue submission plan (async compute) --------------------------------------
    // When a pass declares QueueClass.AsyncCompute AND the device has a dedicated compute
    // family (QueuePlan.HasRealAsyncCompute), Compile partitions the schedule into per-queue
    // SUBMIT CHUNKS: contiguous runs of same-queue passes, split exactly where a cross-queue
    // edge needs a timeline-semaphore signal (after the producer) or wait (before the
    // consumer). Execute then records each chunk into a graph-owned command buffer and
    // submits them itself (the host's command buffer receives nothing; host work submitted
    // later on the graphics queue lands after the chunks by submission order). Graphs with
    // no async passes keep the original record-into-host-cmd path untouched.
    private QueuePlan _plan;
    private QueueClass[] _effQueue = [];           // per pass id: declared queue collapsed to what the device has
    private readonly List<SubmitChunk> _chunks = [];   // creation order == schedule order per queue
    private bool _chunked;                         // true iff any live pass landed on the async queue
    private int _gfxSignalCount, _cmpSignalCount;  // relative timeline values consumed per frame
    private int _lastGfxChunk = -1;                // closing image barriers are appended here
    private Silk.NET.Vulkan.Semaphore _gfxTimeline, _cmpTimeline;
    private ulong _gfxCursor, _cmpCursor;          // monotonic absolute timeline bases across frames
    private CommandPool _gfxChunkPool, _cmpChunkPool;
    private CommandBuffer[][] _chunkCmds = [];     // [frameInFlight][chunkIndex], parallel to _chunks
    // Absolute timeline bases stored by ExecuteChunked for SubmitGfxChunks (called externally).
    private ulong _lastSubmitGfxBase, _lastSubmitCmpBase;
    private uint  _lastSubmitFr;
    // DAG retained from Compile step 1 for the chunk planner (cross-queue edge discovery).
    private HashSet<int>[] _adj = [], _preds = [];

    /// <summary>True when the device has a dedicated compute family, i.e. a pass declared
    /// <see cref="QueueClass.AsyncCompute"/> will really run on a second queue. Modules use
    /// this to pick between an async layout and a single-queue fallback.</summary>
    public bool AsyncComputeAvailable => _plan.HasRealAsyncCompute;

    /// <summary>True when the compiled graph has async-compute passes. In this mode
    /// <see cref="Execute"/> self-submits only the async chunks; the caller must follow up with
    /// <see cref="SubmitGfxChunks"/> to submit the graphics chunks together with its own host
    /// command buffer, keeping all graphics work in one unified submission for profiler
    /// visibility.</summary>
    public bool HasPendingGfxChunks => _chunked;

    /// <summary>One per-queue submission: the contiguous run of scheduled passes it records,
    /// the relative timeline value it signals on its own queue's semaphore (0 = none), and
    /// the (queue, relative value) pairs its submit waits on.</summary>
    private sealed class SubmitChunk
    {
        public QueueClass Queue;
        public readonly List<int> Order = [];                    // positions in _executionOrder
        public ulong Signal;                                     // relative value on own timeline; 0 = none
        public readonly List<(QueueClass q, ulong rel)> Waits = [];
    }

    public FrameGraph(GraphicsDevice device)
    {
        gfx = device;
        _plan = QueuePlan.Resolve(device);
    }

    private Dictionary<int, GraphResource> _resources = [];
    private List<GraphPass> _passes = [];

    internal int CurrentPassIndex = 0;
    internal bool Compiled = false;
    private List<int> _executionOrder = [];

    // Resource ids the caller declared as graph outputs (swapchain / FinalColor / a
    // read-back buffer …). Seeds dead-pass culling (step 2): a pass survives iff it
    // (transitively) feeds one of these, writes an Imported resource, or is flagged
    // HasSideEffects. Without at least one root, every pass is culled.
    private readonly HashSet<int> _outputs = [];

    internal GraphResource? GetResource(int resourceId)
    {
        _resources.TryGetValue(resourceId, out var r);
        
        return r;
    }

    /// <summary>Marks a resource as a graph output so culling never drops the chain that
    /// produces it. Call once per externally-observed target before <see cref="Compile"/>.</summary>
    public void MarkOutput(GraphImage h)  => _outputs.Add(h.resourceId);
    public void MarkOutput(GraphBuffer h) => _outputs.Add(h.resourceId);

    /// <summary>The root authoring scope (empty prefix). Top-level graph building and module
    /// <see cref="IGraphModule{TInputs,TOutputs}.Build"/> calls go through a
    /// <see cref="GraphScope"/>; nest deeper with <see cref="GraphScope.Child"/>.</summary>
    public GraphScope RootScope() => new(this, "");

    // ---- Resource declaration -------------------------------------------------
    // Transients are graph-owned: virtual until Compile()'s step 5 allocates them.
    // Imports adopt an externally-owned handle and are never allocated or freed here.

    public GraphImage CreateImage(in ImageDesc desc, string name)
    {
        int id = _nextResourceId++;
        _resources[id] = new GraphResource
        {
            Id = id, Name = name, IsImage = true, Residency = ResidencyKind.Transient,
            ImageDesc = desc, InitialLayout = ImageLayout.Undefined,
        };
        return new GraphImage(id, 0);
    }

    public GraphBuffer CreateBuffer(in BufferDesc desc, string name)
    {
        int id = _nextResourceId++;
        _resources[id] = new GraphResource
        {
            Id = id, Name = name, IsImage = false, Residency = ResidencyKind.Transient,
            BufferDesc = desc,
        };
        return new GraphBuffer(id, 0);
    }

    /// <summary>Imports an externally-owned image. <paramref name="currentLayout"/> is the
    /// layout it arrives in each frame; <paramref name="finalLayout"/> is what the graph
    /// guarantees on exit (default: hand it back unchanged).</summary>
    public GraphImage ImportImage(Image image, ImageView view, in ImageDesc desc,
        ImageLayout currentLayout, string name, ImageLayout? finalLayout = null)
    {
        int id = _nextResourceId++;
        _resources[id] = new GraphResource
        {
            Id = id, Name = name, IsImage = true, Residency = ResidencyKind.Imported,
            ImageDesc = desc, InitialLayout = currentLayout, FinalLayout = finalLayout ?? currentLayout,
            PhysImage = image, PhysView = view,
        };
        return new GraphImage(id, 0);
    }

    public GraphBuffer ImportBuffer(Buffer buffer, in BufferDesc desc, string name)
    {
        int id = _nextResourceId++;
        _resources[id] = new GraphResource
        {
            Id = id, Name = name, IsImage = false, Residency = ResidencyKind.Imported,
            BufferDesc = desc, PhysBuffer = buffer,
        };
        return new GraphBuffer(id, 0);
    }

    /// <summary>Imports a double-buffered buffer (one handle per frame-in-flight). The
    /// graph derives barriers as usual; at Execute it resolves the right per-frame handle.
    /// <paramref name="perFrame"/> must be indexable by <c>RenderView.FrameIndex</c>.</summary>
    public GraphBuffer ImportBufferPerFrame(Buffer[] perFrame, in BufferDesc desc, string name)
    {
        int id = _nextResourceId++;
        _resources[id] = new GraphResource
        {
            Id = id, Name = name, IsImage = false, Residency = ResidencyKind.Imported,
            BufferDesc = desc, PhysBufferFrames = perFrame,
        };
        return new GraphBuffer(id, 0);
    }


    public void AddPass(string name, PassType type, QueueClass queue,
        PassSetup setup, PassExecute execute,
        bool preferAsync = false, bool hasSideEffects = false, string scope = "")
    {
        var pass = new GraphPass
        {
            Name = name, Scope = scope, Type = type, Queue = queue, PreferAsync = preferAsync, HasSideEffects = hasSideEffects, Execute = execute
        };
        _passes.Add(pass);
        CurrentPassIndex = _passes.Count - 1;
        setup(new GraphBuilder(this, pass)); //Read/Write mutate version ledger + pass lists
        ValidatePassBindings(pass);
    }

    // Fail-loud at authoring time: a named Read/Write must have a matching UsePassSet binding.
    // Order-independent (UsePassSet may follow the named accesses in the setup lambda).
    private static void ValidatePassBindings(GraphPass pass)
    {
        bool anyNamed = false;
        foreach (var a in pass.Reads)  anyNamed |= a.Bind != null;
        foreach (var a in pass.Writes) anyNamed |= a.Bind != null;
        if (!anyNamed) return;

        if (pass.PassSet is null)
            throw new InvalidOperationException(
                $"FrameGraph: pass '{pass.Name}' names a pass-set binding but never called UsePassSet.");

        var spec = pass.PassSet.Value;
        void Check(string? bind)
        {
            if (bind is null) return;
            foreach (var b in spec.Bindings) if (b.Name == bind) return;
            throw new InvalidOperationException(
                $"FrameGraph: pass '{pass.Name}' binds '{bind}' but its pass set has no such parameter " +
                $"(known: {string.Join(", ", spec.Bindings.Select(b => b.Name))}).");
        }
        foreach (var a in pass.Reads)  Check(a.Bind);
        foreach (var a in pass.Writes) Check(a.Bind);
    }

    /// <summary>
    /// 1. Build DAG from versions: <br/>
    /// - RAW: foreach read of (resource, version) -> edge, Producers[v] -> this pass <br/>
    /// - WAW: edge Producers[v-1] -> Producers[v] <br/>
    /// - WAR: foreach reader of (resource, version), edge reader -> Producers[v] <br/>
    /// (All from the version ledger - nolonger j > 1), no declaration-order dependencies. <br/><br/>
    /// 2. Cull dead passes: roots = passes that write an Imported resource or HasSideEffects,
    /// or write a resource reachable from a MarkOutput'd handle. Reverse- reachability; drop the rest + the transient resources. <br/><br/>
    /// 3. Schedule: Kahn topological sort. Initially signle queue so so output is one ordered list. Keep it stable, (enqueue in id order)
    /// so DOT output is deterministic. <br/><br/>
    /// 4. Lifetimes: first/last scheduled index each transient is touched.
    /// (Stored now; only consumed by later aliasing in phase 4, computed now so data is ready) <br/><br/>
    /// 5. Allocate: transients via the allocator, imported resources adopt existing handles. <br/><br/>
    /// 6. Generate Sync: walk the scheduled order maintaining a per-resource "last access"
    /// (stage/access/layout). For each pass, batch the transitions of all its reads + writes into one
    /// DependencyInfo. Store the baked barrier batch on the pass. <br/><br/>
    /// 7. Bake: freeze ordered passes + per-pass barrier batch + resolved physical map.
    /// </summary>
    /// <returns>Compilation success</returns>
    public bool Compile()
    {
        if (Compiled)
            throw new InvalidOperationException("FrameGraph already compiled.");

        var compileSw = Stopwatch.StartNew();
        var n = _passes.Count;

        // ---- 1. Build the DAG from the version ledger ----------------------------
        // Edges point producer -> consumer ("from must run before to"). Dedupe through a
        // HashSet: a duplicate edge would double-count inDegree below and strand Kahn's
        // sort, falsely reporting a cycle. preds is the inverse adjacency, used by the
        // reverse-reachability cull in step 2.
        var adj   = new HashSet<int>[n];
        var preds = new HashSet<int>[n];
        for (int i = 0; i < n; i++) { adj[i] = []; preds[i] = []; }

        // readers[(res, ver)] = every pass that READS exactly that version. Drives WAR:
        // a write that bumps to v must wait for all readers of v-1.
        var readers = new Dictionary<(int res, int ver), List<int>>();
        for (int p = 0; p < n; p++)
            foreach (var rd in _passes[p].Reads)
            {
                var key = (rd.ResourceId, rd.Version);
                if (!readers.TryGetValue(key, out var list)) readers[key] = list = [];
                list.Add(p);
            }

        _dotEdges.Clear();
        for (int p = 0; p < n; p++)
        {
            var pass = _passes[p];

            // RAW: a read of (res, v) depends on whoever produced v.
            foreach (var rd in pass.Reads)
            {
                int producer = ProducerOf(rd.ResourceId, rd.Version);
                AddEdge(producer, p);
                if (producer >= 0)
                    _dotEdges.Add((producer, p,
                        $"{_resources[rd.ResourceId].Name}@{rd.Version} {rd.Usage}", false));
            }

            // WAW + WAR: a write produces (res, v) from the prior version (res, v-1).
            foreach (var wr in pass.Writes)
            {
                // Version 0 is the -1 producer sentinel (fresh/imported); the first real
                // write lands at v >= 1, so a previous version only exists from there.
                if (wr.Version < 1) continue;

                // WAW: serialize after the previous version's producer.
                int prevProducer = ProducerOf(wr.ResourceId, wr.Version - 1);
                AddEdge(prevProducer, p);
                if (prevProducer >= 0)
                    _dotEdges.Add((prevProducer, p, $"{_resources[wr.ResourceId].Name} WAW", true));

                // WAR: every reader of the previous version must finish before we overwrite.
                if (readers.TryGetValue((wr.ResourceId, wr.Version - 1), out var rs))
                    foreach (var reader in rs)
                    {
                        AddEdge(reader, p);
                        _dotEdges.Add((reader, p, $"{_resources[wr.ResourceId].Name} WAR", true));
                    }
            }
        }

        // ---- 2. Cull dead passes -------------------------------------------------
        // Seed with roots, then mark every predecessor of a live pass live too
        // (reverse reachability). Anything still unmarked produces nothing observed.
        var live = new bool[n];
        var work = new Stack<int>();
        for (int p = 0; p < n; p++)
            if (IsRoot(_passes[p])) { live[p] = true; work.Push(p); }

        while (work.Count > 0)
        {
            int p = work.Pop();
            foreach (var pred in preds[p])
                if (!live[pred]) { live[pred] = true; work.Push(pred); }
        }

        // ---- 3. Schedule (Kahn, stable) over the LIVE subgraph -------------------
        // inDegree counts only live->live edges so culled passes don't gate the sort.
        var inDeg = new int[n];
        for (int from = 0; from < n; from++)
            if (live[from])
                foreach (var to in adj[from])
                    if (live[to]) inDeg[to]++;

        // SortedSet => always take the lowest ready id: deterministic order, so DOT
        // dumps and RenderDoc captures are reproducible frame to frame.
        var ready = new SortedSet<int>();
        for (int p = 0; p < n; p++)
            if (live[p] && inDeg[p] == 0) ready.Add(p);

        _executionOrder.Clear();
        while (ready.Count > 0)
        {
            int p = ready.Min;
            ready.Remove(p);
            _executionOrder.Add(p);
            foreach (var to in adj[p])
                if (live[to] && --inDeg[to] == 0) ready.Add(to);
        }

        int liveCount = 0;
        for (int p = 0; p < n; p++) if (live[p]) liveCount++;
        if (_executionOrder.Count != liveCount)
            throw new InvalidOperationException(
                "FrameGraph: cycle detected among live passes — the version ledger produced a back-edge.");

        _live = live;                  // kept for ToDot
        _culledCount = n - liveCount;
        _adj = adj; _preds = preds;    // retained for the chunk planner (cross-queue edges)

        // ---- 3.5 Queue assignment + submit chunking (no-op without real async passes) -----
        PlanChunks();

        // ---- 4. Transient lifetimes (first/last touch in schedule order) ---------
        // Consumed by aliasing in Phase 4; computed now so the schedule stays the single
        // source of truth for ordering and the data is ready when aliasing lands.
        foreach (var res in _resources.Values) { res.FirstUse = int.MaxValue; res.LastUse = -1; }
        for (int order = 0; order < _executionOrder.Count; order++)
        {
            var pass = _passes[_executionOrder[order]];
            foreach (var a in pass.Reads)  TouchLifetime(a.ResourceId, order);
            foreach (var a in pass.Writes) TouchLifetime(a.ResourceId, order);
        }

        // ---- 5. Allocate transients / adopt imports ------------------------------
        AllocateResources();

        // ---- 6/7. Derive + bake the per-pass Sync2 barrier batches ---------------
        // _executionOrder + the baked per-pass barriers + the resolved physical handles
        // are now the frozen plan; Execute() replays them with no per-frame analysis.
        BakeSync();

        // ---- Graph-baked pass descriptor sets (opt-in via UsePassSet) ------------
        // Allocate + fill each opting pass's per-frame set from the now-resolved handles.
        BakePassSets();

        // ---- Graph-owned shared set (opt-in via UseGraphSharedSet) ---------------
        // One set for the whole graph, shared by every pass (the PT working set).
        BakeGraphSharedSet();

        // ---- Debug instrumentation: query pools, labels, object names ------------
        SetupDebug();

        Compiled = true;
        _compileMs = compileSw.Elapsed.TotalMilliseconds;
        return true;

        void AddEdge(int from, int to)
        {
            // from < 0  == "version 0 sentinel / imported — no producing pass" -> no edge.
            if (from < 0 || to < 0 || from == to) return;
            if (adj[from].Add(to)) preds[to].Add(from);
        }

        // ---- locals --------------------------------------------------------------
        int ProducerOf(int resId, int version)
        {
            var producers = _resources[resId].Producers;
            return version >= 0 && version < producers.Count ? producers[version] : -1;
        }

        bool IsRoot(GraphPass pass)
        {
            if (pass.HasSideEffects) return true;
            foreach (var wr in pass.Writes)
            {
                var res = _resources[wr.ResourceId];
                if (res.Residency == ResidencyKind.Imported || _outputs.Contains(wr.ResourceId))
                    return true;
            }
            return false;
        }

        void TouchLifetime(int resId, int order)
        {
            var r = _resources[resId];
            if (order < r.FirstUse) r.FirstUse = order;
            if (order > r.LastUse)  r.LastUse  = order;
        }
    }

    /// ---- Step 3.5: queue assignment + submit chunking ----------------------------
    /// Collapse declared queues onto what the device has, then split the schedule into
    /// per-queue submit chunks at the cross-queue edges. Rules:
    ///  - a graphics pass with an async producer (any RAW/WAW/WAR pred) starts a NEW chunk
    ///    whose submit waits that producer chunk's timeline value (a wait gates the whole
    ///    submit, so passes that must NOT wait stay in earlier chunks);
    ///  - a graphics pass with an async CONSUMER closes its chunk with a signal right after
    ///    it, so the consumer has something to wait on;
    ///  - every async pass is its own chunk (v1) and always signals -- graphics consumers
    ///    and next-frame transitive ordering hang off that value.
    /// Cross-queue MEMORY sync rides the semaphores (signal makes all writes available, wait
    /// makes them visible), so BakeSync skips barriers on cross-queue transitions.
    private void PlanChunks()
    {
        _chunks.Clear();
        _gfxSignalCount = _cmpSignalCount = 0;
        _lastGfxChunk = -1;
        _chunked = false;

        _effQueue = new QueueClass[_passes.Count];
        for (int id = 0; id < _passes.Count; id++)
        {
            bool isAsync = _passes[id].Queue == QueueClass.AsyncCompute && _plan.HasRealAsyncCompute;
            _effQueue[id] = isAsync ? QueueClass.AsyncCompute : QueueClass.Graphics;
            _chunked |= isAsync && _live[id];
        }
        if (!_chunked) return;

        // v1 restrictions: async passes must be compute/transfer work and touch BUFFERS only.
        // Image layouts are tracked on the graphics timeline; letting an async pass transition
        // one would need cross-queue layout handoff (queue family ownership transfer) -- out of
        // scope until a feature needs it.
        foreach (int id in _executionOrder)
        {
            if (_effQueue[id] != QueueClass.AsyncCompute) continue;
            var p = _passes[id];
            // Only Compute (and Transfer) work can ride the async-compute queue. Graphics and
            // RayTrace (CmdTraceRays) both require the graphics queue -- the dedicated compute
            // family advertises neither, so declaring one AsyncCompute is an authoring error.
            if (p.Type is PassType.Graphics or PassType.RayTrace)
                throw new InvalidOperationException(
                    $"FrameGraph: pass '{p.Name}' is PassType.{p.Type} but declared QueueClass.AsyncCompute; " +
                    "only Compute/Transfer passes can run on the async-compute queue.");
            foreach (var a in p.Reads)
                if (a.IsImage) throw new InvalidOperationException(
                    $"FrameGraph: async pass '{p.Name}' accesses image '{_resources[a.ResourceId].Name}' -- async passes may only touch buffers (v1).");
            foreach (var a in p.Writes)
                if (a.IsImage) throw new InvalidOperationException(
                    $"FrameGraph: async pass '{p.Name}' accesses image '{_resources[a.ResourceId].Name}' -- async passes may only touch buffers (v1).");
        }

        var passChunk = new int[_passes.Count];
        Array.Fill(passChunk, -1);
        int curIdx = -1;   // open graphics chunk index, -1 = none

        for (int pos = 0; pos < _executionOrder.Count; pos++)
        {
            int id = _executionOrder[pos];
            if (_effQueue[id] == QueueClass.Graphics)
            {
                // Async producers force a fresh chunk fronted by their waits. Every async
                // chunk signals (see below), so the producer's value always exists.
                List<(QueueClass, ulong)>? waits = null;
                foreach (var pred in _preds[id])
                    if (_effQueue[pred] == QueueClass.AsyncCompute)
                        (waits ??= []).Add((QueueClass.AsyncCompute, _chunks[passChunk[pred]].Signal));
                if (waits != null && curIdx >= 0)
                {
                    // The open chunk is about to be closed. Give it an explicit gfx timeline
                    // signal so profilers (Nsight) can anchor it. Add the matching gfx wait to
                    // the new chunk's waits -- redundant with same-queue ordering, but makes
                    // every chunk reachable via the semaphore dependency graph.
                    _chunks[curIdx].Signal = (ulong)++_gfxSignalCount;
                    waits.Add((QueueClass.Graphics, (ulong)_gfxSignalCount));
                    curIdx = -1;
                }
                if (curIdx < 0)
                {
                    _chunks.Add(new SubmitChunk { Queue = QueueClass.Graphics });
                    curIdx = _chunks.Count - 1;
                    _lastGfxChunk = curIdx;
                }
                if (waits != null) MergeWaits(_chunks[curIdx], waits);
                _chunks[curIdx].Order.Add(pos);
                passChunk[id] = curIdx;

                // An async consumer needs a value to wait on: close this chunk with a signal.
                foreach (var succ in _adj[id])
                    if (_effQueue[succ] == QueueClass.AsyncCompute)
                    {
                        _chunks[curIdx].Signal = (ulong)++_gfxSignalCount;
                        curIdx = -1;
                        break;
                    }
            }
            else
            {
                var chunk = new SubmitChunk { Queue = QueueClass.AsyncCompute, Signal = (ulong)++_cmpSignalCount };
                List<(QueueClass, ulong)>? waits = null;
                foreach (var pred in _preds[id])
                {
                    var predChunk = _chunks[passChunk[pred]];
                    // A graphics pred's chunk was closed with a signal when this pass was seen
                    // among its successors; an async pred always signals.
                    (waits ??= []).Add((predChunk.Queue, predChunk.Signal));
                }
                if (waits != null) MergeWaits(chunk, waits);
                chunk.Order.Add(pos);
                _chunks.Add(chunk);
                passChunk[id] = _chunks.Count - 1;
            }
        }

        CreateSubmitResources();

        static void MergeWaits(SubmitChunk chunk, List<(QueueClass q, ulong rel)> waits)
        {
            // One wait per queue, at the max value (a timeline wait at V covers every <= V).
            foreach (var (q, rel) in waits)
            {
                if (rel == 0) throw new InvalidOperationException(
                    "FrameGraph: cross-queue producer chunk has no signal -- chunk planner bug.");
                int existing = chunk.Waits.FindIndex(w => w.q == q);
                if (existing < 0) chunk.Waits.Add((q, rel));
                else if (chunk.Waits[existing].rel < rel) chunk.Waits[existing] = (q, rel);
            }
        }
    }

    /// <summary>Per-queue command pools + per-frame chunk command buffers + the two timeline
    /// semaphores. Created once per Compile (graphs rebuild on resize); torn down in Dispose.</summary>
    private void CreateSubmitResources()
    {
        var vk  = gfx.Vk;
        var dev = gfx.Device;

        CommandPool MakePool(uint family)
        {
            var info = new CommandPoolCreateInfo
            {
                SType = StructureType.CommandPoolCreateInfo,
                Flags = CommandPoolCreateFlags.ResetCommandBufferBit,
                QueueFamilyIndex = family,
            };
            if (vk.CreateCommandPool(dev, &info, null, out var pool) != Result.Success)
                throw new Exception("FrameGraph: failed to create chunk command pool");
            return pool;
        }
        _gfxChunkPool = MakePool(_plan.Graphics.Family);
        _cmpChunkPool = MakePool(_plan.AsyncCompute!.Value.Family);

        _chunkCmds = new CommandBuffer[RenderConfig.MAX_CONCURRENT_FRAMES][];
        for (int f = 0; f < RenderConfig.MAX_CONCURRENT_FRAMES; f++)
        {
            _chunkCmds[f] = new CommandBuffer[_chunks.Count];
            for (int c = 0; c < _chunks.Count; c++)
            {
                var alloc = new CommandBufferAllocateInfo
                {
                    SType = StructureType.CommandBufferAllocateInfo,
                    CommandPool = _chunks[c].Queue == QueueClass.Graphics ? _gfxChunkPool : _cmpChunkPool,
                    Level = CommandBufferLevel.Primary,
                    CommandBufferCount = 1,
                };
                fixed (CommandBuffer* p = &_chunkCmds[f][c])
                    if (vk.AllocateCommandBuffers(dev, &alloc, p) != Result.Success)
                        throw new Exception("FrameGraph: failed to allocate chunk command buffer");
            }
        }

        Silk.NET.Vulkan.Semaphore MakeTimeline()
        {
            var type = new SemaphoreTypeCreateInfo
            {
                SType = StructureType.SemaphoreTypeCreateInfo,
                SemaphoreType = SemaphoreType.Timeline,
                InitialValue = 0,
            };
            var info = new SemaphoreCreateInfo { SType = StructureType.SemaphoreCreateInfo, PNext = &type };
            if (vk.CreateSemaphore(dev, &info, null, out var sem) != Result.Success)
                throw new Exception("FrameGraph: failed to create timeline semaphore");
            return sem;
        }
        _gfxTimeline = MakeTimeline();
        _cmpTimeline = MakeTimeline();
        _gfxCursor = _cmpCursor = 0;
    }

    /// ---- Physical-handle resolution (used by PassResources during Execute) --------
    /// Valid only after Compile()'s step 5 has populated the phys handles; until then
    /// these throw, which is the correct signal that Execute ran before allocation.
    internal ImageView ResolveView(GraphImage h) =>
        _resources[h.resourceId].PhysView ?? throw new InvalidOperationException(
            $"FrameGraph: image '{_resources[h.resourceId].Name}' has no view (not allocated/imported).");

    internal Image ResolveImage(GraphImage h) =>
        _resources[h.resourceId].PhysImage ?? throw new InvalidOperationException(
            $"FrameGraph: image '{_resources[h.resourceId].Name}' not allocated.");

    internal Buffer ResolveBuffer(GraphBuffer h)
    {
        var r = _resources[h.resourceId];
        if (r.PhysBuffer is { } single) return single;
        throw new InvalidOperationException(r.PhysBufferFrames != null
            ? $"FrameGraph: buffer '{r.Name}' is per-frame — read it from its owning pipeline in the pass body, not via PassResources."
            : $"FrameGraph: buffer '{r.Name}' not allocated.");
    }

    // Resolves a buffer resource's handle for a specific frame-in-flight (used to patch
    // baked buffer barriers each Execute). Single-handle resources ignore the frame.
    private Buffer ResolveBufferFrame(int resId, uint frame)
    {
        var r = _resources[resId];
        return r.PhysBufferFrames is { } frames ? frames[frame] : r.PhysBuffer ?? default;
    }

    // ---- Step 5: physical allocation -----------------------------------------
    // Transients touched by a live pass get backing memory + (images) a view; imports
    // already carry their handle; culled resources (LastUse < 0) are skipped.
    private void AllocateResources()
    {
        foreach (var res in _resources.Values)
        {
            if (res.Residency == ResidencyKind.Imported || res.LastUse < 0) continue;
            if (res.IsImage) AllocateImage(res);
            else             AllocateBuffer(res);
        }
    }

    private void AllocateImage(GraphResource res)
    {
        var vk  = gfx.Vk;
        var dev = gfx.Device;
        var d   = res.ImageDesc!.Value;
        uint mips    = d.Mips   == 0 ? 1 : d.Mips;
        uint layers  = d.Layers == 0 ? 1 : d.Layers;
        var  samples = d.Samples == 0 ? SampleCountFlags.Count1Bit : d.Samples;

        var info = new ImageCreateInfo
        {
            SType = StructureType.ImageCreateInfo,
            ImageType = ImageType.Type2D,
            Format = d.Format,
            Extent = new Extent3D(d.Extent.Width, d.Extent.Height, 1),
            MipLevels = mips,
            ArrayLayers = layers,
            Samples = samples,
            Tiling = ImageTiling.Optimal,
            Usage = d.Usage,
            SharingMode = SharingMode.Exclusive,
            InitialLayout = ImageLayout.Undefined,
        };
        if (vk.CreateImage(dev, ref info, null, out var image) != Result.Success)
            throw new Exception($"FrameGraph: failed to create image '{res.Name}'.");
        
        // High residency priority: graph transients are the g-buffers / depth / HDR
        // targets, written and read every frame — keep them resident ahead of cold
        // resources under WDDM budget pressure.
        var alloc = gfx.Allocator.AllocateForImage(image, MemoryPropertyFlags.DeviceLocalBit,
            ImageTiling.Optimal, GpuMemoryAllocator.PriorityHigh);

        bool depth = IsDepthFormat(d.Format);
        var viewInfo = new ImageViewCreateInfo
        {
            SType = StructureType.ImageViewCreateInfo,
            Image = image,
            ViewType = layers > 1 ? ImageViewType.Type2DArray : ImageViewType.Type2D,
            Format = d.Format,
            SubresourceRange = new ImageSubresourceRange(
                depth ? ImageAspectFlags.DepthBit : ImageAspectFlags.ColorBit, 0, mips, 0, layers),
        };
        if (vk.CreateImageView(dev, ref viewInfo, null, out var view) != Result.Success)
            throw new Exception($"FrameGraph: failed to create image view '{res.Name}'.");

        res.PhysImage = image;
        res.Alloc     = alloc;
        res.PhysView  = view;
    }

    private void AllocateBuffer(GraphResource res)
    {
        var d = res.BufferDesc!.Value;
        gfx.CreateBuffer(d.size, d.Usage, MemoryPropertyFlags.DeviceLocalBit,
            out var buffer, out var alloc);
        res.PhysBuffer = buffer;
        res.Alloc      = alloc;
    }

    // ---- Step 6/7: derive + bake the per-pass barrier batches -----------------
    private void BakeSync()
    {
        // Per-resource running cursor = the stage/access/layout the last access left it in.
        // Same shape as a UsageTable entry (it IS one - the seed and every post-barrier
        // state are just usage snapshots), so we reuse UsageInfo rather than a twin struct.
        // Transients start Undefined (fresh memory, contents are garbage); imports start in
        // their declared incoming layout, conservatively visible to all prior work.
        var state = new Dictionary<int, UsageInfo>();
        foreach (var res in _resources.Values)
        {
            if (res.LastUse < 0) continue;
            state[res.Id] = res.Residency == ResidencyKind.Imported
                ? new UsageInfo(PipelineStageFlags2.AllCommandsBit,
                    AccessFlags2.MemoryReadBit | AccessFlags2.MemoryWriteBit, res.InitialLayout, false)
                : new UsageInfo(PipelineStageFlags2.TopOfPipeBit, AccessFlags2.None,
                    ImageLayout.Undefined, false);
        }

        // Queue of each resource's last WRITER (graphics until proven otherwise). A cross-
        // queue access derives NO barrier: the consumer chunk's timeline-semaphore wait is
        // the execution AND memory dependency (signal = all writes available, wait = all
        // available memory visible). Empty when the graph isn't chunked.
        var writeQueue = new Dictionary<int, QueueClass>();

        foreach (var passIdx in _executionOrder)
        {
            var pass = _passes[passIdx];
            var q = _effQueue.Length > 0 ? _effQueue[passIdx] : QueueClass.Graphics;
            var imgs = new List<ImageMemoryBarrier2>();
            var bufs = new List<BufferMemoryBarrier2>();
            var bufRes = new List<int>();
            foreach (var a in pass.Reads)  ProcessAccess(in a, q, state, writeQueue, imgs, bufs, bufRes);
            foreach (var a in pass.Writes) ProcessAccess(in a, q, state, writeQueue, imgs, bufs, bufRes);
            pass.ImageBarriers     = imgs.ToArray();
            pass.BufferBarriers    = bufs.ToArray();
            pass.BufferBarrierRes  = bufRes.ToArray();
        }

        // Closing: hand each imported image back in its declared final layout, so next
        // frame's baked first-use barrier (which assumes InitialLayout) is valid and any
        // external consumer (swapchain blit, ImGui sampler) finds it where it expects.
        var closing = new List<ImageMemoryBarrier2>();
        foreach (var res in _resources.Values)
        {
            if (res.Residency != ResidencyKind.Imported || !res.IsImage) continue;
            // FinalLayout == Undefined means "don't care / no restore" (the import default
            // for fully-regenerated targets); you can never transition TO Undefined anyway.
            if (res.FinalLayout == ImageLayout.Undefined) continue;
            if (!state.TryGetValue(res.Id, out var cur) || cur.Layout == res.FinalLayout) continue;
            closing.Add(MakeImageBarrier(res,
                cur.Stage, cur.Access, cur.Layout,
                PipelineStageFlags2.AllCommandsBit, AccessFlags2.MemoryReadBit, res.FinalLayout));
        }
        _closingImageBarriers = closing.ToArray();
    }

    private void ProcessAccess(in ResourceAccess a, QueueClass q, Dictionary<int, UsageInfo> state,
        Dictionary<int, QueueClass> writeQueue, List<ImageMemoryBarrier2> imgs,
        List<BufferMemoryBarrier2> bufs, List<int> bufRes)
    {
        var next = UsageTable.Of(a.Usage);
        var cur  = state[a.ResourceId];

        // Cross-queue access (resource last written on the OTHER queue): the chunk-level
        // timeline wait derived from this same DAG edge is the dependency; record no barrier.
        // A cross-queue READ also leaves the stored state untouched, so later consumers on
        // the writer's own queue still derive their barrier against the real write (the
        // reader's execution is tracked by the semaphore, not the state cursor). A cross-
        // queue WRITE adopts the resource: state and owning queue move to this queue.
        if (writeQueue.TryGetValue(a.ResourceId, out var wq) ? wq != q : q != QueueClass.Graphics)
        {
            if (next.IsWrite)
            {
                state[a.ResourceId]      = next;
                writeQueue[a.ResourceId] = q;
            }
            return;
        }

        bool readAfterRead = !next.IsWrite && !cur.IsWrite;
        // Concurrent-after-concurrent: both accesses declared StorageConcurrentCompute, whose
        // contract is "mutually safe by construction" -- skip the barrier exactly like
        // read-after-read so the later pass can overlap the earlier on the queue. The widened
        // state still carries IsWrite/IsConcurrent, so the eventual transition OUT of the
        // concurrent run (an ordinary read or write) emits a barrier covering every accessor.
        bool concurrentRepeat = next.IsConcurrent && cur.IsConcurrent;
        bool layoutChange  = a.IsImage && cur.Layout != next.Layout;

        // Pure read-after-read in a compatible layout needs no barrier — but widen the
        // visibility set so the eventual writer waits on EVERY prior reader, not just the
        // last (two readers on different stages would otherwise drop the first).
        if ((readAfterRead || concurrentRepeat) && !layoutChange)
        {
            state[a.ResourceId] = cur with { Stage = cur.Stage | next.Stage, Access = cur.Access | next.Access };
            return;
        }

        var res = _resources[a.ResourceId];
        if (a.IsImage)
        {
            imgs.Add(MakeImageBarrier(res, cur.Stage, cur.Access, cur.Layout,
                                      next.Stage, next.Access, next.Layout));
        }
        else
        {
            bufs.Add(MakeBufferBarrier(res, cur.Stage, cur.Access, next.Stage, next.Access));
            bufRes.Add(res.Id);   // so Execute can patch per-frame buffer handles
        }

        state[a.ResourceId] = next;   // next IS a UsageInfo — no copy needed
    }

    private ImageMemoryBarrier2 MakeImageBarrier(GraphResource res,
        PipelineStageFlags2 srcStage, AccessFlags2 srcAccess, ImageLayout oldLayout,
        PipelineStageFlags2 dstStage, AccessFlags2 dstAccess, ImageLayout newLayout)
    {
        var d = res.ImageDesc!.Value;
        bool depth = IsDepthFormat(d.Format);
        return new ImageMemoryBarrier2
        {
            SType = StructureType.ImageMemoryBarrier2,
            SrcStageMask = srcStage, SrcAccessMask = srcAccess,
            DstStageMask = dstStage, DstAccessMask = dstAccess,
            OldLayout = oldLayout,   NewLayout = newLayout,
            SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
            Image = res.PhysImage!.Value,
            SubresourceRange = new ImageSubresourceRange(
                depth ? ImageAspectFlags.DepthBit : ImageAspectFlags.ColorBit,
                0, d.Mips == 0 ? 1 : d.Mips, 0, d.Layers == 0 ? 1 : d.Layers),
        };
    }

    private static BufferMemoryBarrier2 MakeBufferBarrier(GraphResource res,
        PipelineStageFlags2 srcStage, AccessFlags2 srcAccess,
        PipelineStageFlags2 dstStage, AccessFlags2 dstAccess) =>
        new()
        {
            SType = StructureType.BufferMemoryBarrier2,
            SrcStageMask = srcStage, SrcAccessMask = srcAccess,
            DstStageMask = dstStage, DstAccessMask = dstAccess,
            SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
            // Per-frame imports have a null PhysBuffer here; Execute patches the .Buffer
            // field with the right frame's handle (via BufferBarrierRes) before submitting.
            Buffer = res.PhysBuffer ?? default, Offset = 0, Size = Vk.WholeSize,
        };

    private static bool IsDepthFormat(Format f) =>
        f is Format.D32Sfloat or Format.D24UnormS8Uint or Format.D16Unorm
          or Format.D32SfloatS8Uint or Format.D16UnormS8Uint;

    // ---- Graph-baked pass descriptor sets  ---------
    // Owns the pass sets because their lifetime IS the graph's: allocated here at Compile
    // (which runs under device-idle on init/resize) and freed in Dispose. Each opting pass
    // gets one set per frame-in-flight from its pipeline-owned layout; every binding a named
    // Read/Write referenced is written NOW with that frame's resolved handle. No per-frame
    // rewrite is needed -- the sets are never mutated in flight (a resize rebuilds the whole
    // graph), so Execute only hands the right frame's set to the pass body. The scene set (a
    // frame-lifetime, in-flight-mutated set) stays with DescriptorRegistry; these do not.
    private DescriptorPool _passSetPool;

    private void BakePassSets()
    {
        uint frames = RenderConfig.MAX_CONCURRENT_FRAMES;

        var opting = new List<GraphPass>();
        foreach (var idx in _executionOrder)
            if (_passes[idx].PassSet is not null) opting.Add(_passes[idx]);
        if (opting.Count == 0) return;

        // Pool sized to the exact per-type descriptor demand across every opting pass * frames.
        var poolSizes = new Dictionary<DescriptorType, uint>();
        uint maxSets = 0;
        foreach (var pass in opting)
        {
            var spec = pass.PassSet!.Value;
            maxSets += frames;
            foreach (var a in NamedAccesses(pass))
            {
                var b = FindBinding(spec, a.Bind!);
                poolSizes[b.Type] = poolSizes.GetValueOrDefault(b.Type) + frames;
            }
        }

        var sizes = poolSizes.Select(kv => new DescriptorPoolSize { Type = kv.Key, DescriptorCount = kv.Value }).ToArray();
        fixed (DescriptorPoolSize* pSizes = sizes)
        {
            var poolInfo = new DescriptorPoolCreateInfo
            {
                SType = StructureType.DescriptorPoolCreateInfo,
                PoolSizeCount = (uint)sizes.Length,
                PPoolSizes = pSizes,
                MaxSets = maxSets,
            };
            if (gfx.Vk.CreateDescriptorPool(gfx.Device, &poolInfo, null, out _passSetPool) != Result.Success)
                throw new Exception("FrameGraph: failed to create pass-set descriptor pool");
        }

        // One layout-array scratch reused per pass (every set in a pass shares its layout).
        var layouts = stackalloc DescriptorSetLayout[(int)frames];
        foreach (var pass in opting)
        {
            var spec = pass.PassSet!.Value;

            for (int f = 0; f < frames; f++) layouts[f] = spec.Layout;
            var alloc = new DescriptorSetAllocateInfo
            {
                SType = StructureType.DescriptorSetAllocateInfo,
                DescriptorPool = _passSetPool,
                DescriptorSetCount = frames,
                PSetLayouts = layouts,
            };
            var sets = new DescriptorSet[frames];
            fixed (DescriptorSet* pSets = sets)
                if (gfx.Vk.AllocateDescriptorSets(gfx.Device, &alloc, pSets) != Result.Success)
                    throw new Exception($"FrameGraph: failed to allocate pass set for '{pass.Name}'");
            pass.PassSets = sets;

            foreach (var a in NamedAccesses(pass))
                for (uint f = 0; f < frames; f++)
                    WritePassBinding(sets[f], spec, in a, f);
        }
    }

    // ---- Graph-owned shared set  ---------
    // ONE descriptor set for the whole graph, shared by every pass (vs pass sets, one per pass).
    // Fits a technique whose passes all touch the same working buffers (the PT SoA / ReSTIR
    // reservoirs): the pipeline still owns the buffers + the layout, the graph owns just the set.
    // Single instance (not per-frame): the handles never change in flight -- a resize rebuilds the
    // whole graph -- and the buffer contents are ordered by the passes' declared barriers. The core
    // reads GraphSharedSet after Compile and hands it to the pipeline for its record-time binds.
    private GraphSharedSetSpec? _graphSharedSpec;
    private DescriptorSet _graphSharedSet;
    private DescriptorPool _graphSharedPool;

    /// <summary>Opts the graph into owning one shared descriptor set (see <see cref="GraphSharedSetSpec"/>),
    /// allocated + written at Compile from the pipeline-supplied buffer handles. Call once during Build.</summary>
    public void UseGraphSharedSet(in GraphSharedSetSpec spec) => _graphSharedSpec = spec;

    /// <summary>The graph-owned shared set, valid after <see cref="Compile"/> (default handle if the
    /// graph never opted in). Bound by the pipeline at the spec's set index.</summary>
    public DescriptorSet GraphSharedSet => _graphSharedSet;

    private void BakeGraphSharedSet()
    {
        if (_graphSharedSpec is not { } spec || spec.Bindings.Count == 0) return;

        var poolSizes = new Dictionary<DescriptorType, uint>();
        foreach (var b in spec.Bindings) poolSizes[b.Type] = poolSizes.GetValueOrDefault(b.Type) + 1;
        var sizes = poolSizes.Select(kv => new DescriptorPoolSize { Type = kv.Key, DescriptorCount = kv.Value }).ToArray();
        fixed (DescriptorPoolSize* pSizes = sizes)
        {
            var poolInfo = new DescriptorPoolCreateInfo
            {
                SType = StructureType.DescriptorPoolCreateInfo,
                PoolSizeCount = (uint)sizes.Length, PPoolSizes = pSizes, MaxSets = 1,
            };
            if (gfx.Vk.CreateDescriptorPool(gfx.Device, &poolInfo, null, out _graphSharedPool) != Result.Success)
                throw new Exception("FrameGraph: failed to create graph-shared descriptor pool");
        }

        var layout = spec.Layout;
        var alloc = new DescriptorSetAllocateInfo
        {
            SType = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool = _graphSharedPool, DescriptorSetCount = 1, PSetLayouts = &layout,
        };
        fixed (DescriptorSet* p = &_graphSharedSet)
            if (gfx.Vk.AllocateDescriptorSets(gfx.Device, &alloc, p) != Result.Success)
                throw new Exception("FrameGraph: failed to allocate graph-shared set");

        foreach (var b in spec.Bindings)
        {
            var info = new DescriptorBufferInfo { Buffer = b.Buffer, Offset = 0, Range = Vk.WholeSize };
            var write = new WriteDescriptorSet
            {
                SType = StructureType.WriteDescriptorSet, DstSet = _graphSharedSet,
                DstBinding = b.Binding, DescriptorType = b.Type, DescriptorCount = 1, PBufferInfo = &info,
            };
            gfx.Vk.UpdateDescriptorSets(gfx.Device, 1, &write, 0, null);
        }
    }

    // The named (pass-set) accesses of a pass, reads then writes.
    private static IEnumerable<ResourceAccess> NamedAccesses(GraphPass pass)
    {
        foreach (var a in pass.Reads)  if (a.Bind != null) yield return a;
        foreach (var a in pass.Writes) if (a.Bind != null) yield return a;
    }

    private static BindingDesc FindBinding(in PassSetSpec spec, string name)
    {
        foreach (var b in spec.Bindings) if (b.Name == name) return b;
        throw new InvalidOperationException($"FrameGraph: pass set has no binding named '{name}'.");
    }

    // Writes one binding of a baked pass set for frame f. Image layout is derived from the
    // access usage (the same UsageTable entry that baked the barrier), so the descriptor
    // layout and the barrier's target layout can never drift. A CombinedImageSampler binding
    // gets its sampler from the pipeline-owned layout's IMMUTABLE sampler (the write's sampler
    // field is then ignored by Vulkan) -- so every image case writes just view + layout and the
    // graph never has to own or plumb a sampler.
    private void WritePassBinding(DescriptorSet set, in PassSetSpec spec, in ResourceAccess a, uint frame)
    {
        var b   = FindBinding(spec, a.Bind!);
        var res = _resources[a.ResourceId];
        var write = new WriteDescriptorSet
        {
            SType = StructureType.WriteDescriptorSet,
            DstSet = set,
            DstBinding = b.Binding,
            DescriptorType = b.Type,
            DescriptorCount = 1,
        };

        if (a.IsImage)
        {
            var info = new DescriptorImageInfo
            {
                ImageView = res.PhysView ?? throw new InvalidOperationException(
                    $"FrameGraph: pass-set image '{res.Name}' has no view."),
                ImageLayout = UsageTable.Of(a.Usage).Layout,
            };
            write.PImageInfo = &info;
            gfx.Vk.UpdateDescriptorSets(gfx.Device, 1, &write, 0, null);
        }
        else
        {
            var buf = res.PhysBufferFrames is { } fr ? fr[frame]
                    : res.PhysBuffer ?? throw new InvalidOperationException(
                        $"FrameGraph: pass-set buffer '{res.Name}' not allocated.");
            var info = new DescriptorBufferInfo { Buffer = buf, Offset = 0, Range = Vk.WholeSize };
            write.PBufferInfo = &info;
            gfx.Vk.UpdateDescriptorSets(gfx.Device, 1, &write, 0, null);
        }
    }

    /// <summary>
    /// Replays the baked plan: per scheduled pass, emit its barrier batch then record its
    /// body; finally hand imported images back in their declared final layout. No
    /// per-frame analysis — <see cref="Compile"/> did it all.
    ///
    /// When the compiled plan contains async-compute chunks, the host's <paramref name="cmd"/>
    /// receives NOTHING: the graph records each chunk into its own command buffers and submits
    /// them itself (graphics chunks first, then async). Host work recorded into
    /// <paramref name="cmd"/> and submitted later on the graphics queue executes after the
    /// chunks by submission order, exactly as it did when the passes lived inside it.
    /// </summary>
    public void Execute(CommandBuffer cmd, in RenderView frame)
    {
        if (!Compiled) throw new InvalidOperationException("FrameGraph: call Compile() before Execute().");

        if (_chunked) { ExecuteChunked(in frame); return; }

        _debug?.BeginFrame(cmd, frame.FrameIndex);
        for (int i = 0; i < _executionOrder.Count; i++)
        {
            var pass = _passes[_executionOrder[i]];
            // Patch per-frame buffer handles into this pass's baked buffer barriers.
            for (int k = 0; k < pass.BufferBarriers.Length; k++)
                pass.BufferBarriers[k].Buffer = ResolveBufferFrame(pass.BufferBarrierRes[k], frame.FrameIndex);

            _debug?.BeginPass(cmd, frame.FrameIndex, i);   // debug label + begin timestamp/stats

            EmitBarriers(cmd, pass.ImageBarriers, pass.BufferBarriers);
            var resources = new PassResources(this,
                pass.PassSets.Length > 0 ? pass.PassSets[frame.FrameIndex] : default);
            pass.Execute(cmd, resources, in frame);
            _debug?.EndPass(cmd, frame.FrameIndex, i);      // end timestamp/stats + pop label
        }
        EmitBarriers(cmd, _closingImageBarriers, []);
    }

    /// <summary>Record every chunk into this frame slot's command buffers. Async compute chunks
    /// are self-submitted here. Graphics chunks are NOT submitted -- the caller must follow up
    /// with <see cref="SubmitGfxChunks"/> to merge them with its own host command buffer into
    /// one vkQueueSubmit2, giving profilers (Nsight, RenderDoc) a single unified gfx submission
    /// to show all graphics passes alongside the blit and UI work.</summary>
    private void ExecuteChunked(in RenderView frame)
    {
        uint fr = frame.FrameIndex;
        var vk = gfx.Vk;

        bool firstGfx = true;
        for (int c = 0; c < _chunks.Count; c++)
        {
            var chunk = _chunks[c];
            var cb = _chunkCmds[fr][c];
            vk.ResetCommandBuffer(cb, 0);
            var begin = new CommandBufferBeginInfo
            {
                SType = StructureType.CommandBufferBeginInfo,
                Flags = CommandBufferUsageFlags.OneTimeSubmitBit,
            };
            if (vk.BeginCommandBuffer(cb, &begin) != Result.Success)
                throw new Exception("FrameGraph: failed to begin chunk command buffer");

            // Query-pool resets live in the first graphics chunk; async chunks may still
            // write timestamps -- their submits wait on graphics values signalled after the
            // reset, so the reset is ordered before every async write.
            if (chunk.Queue == QueueClass.Graphics && firstGfx)
            {
                _debug?.BeginFrame(cb, fr);
                firstGfx = false;
            }

            foreach (var pos in chunk.Order)
            {
                var pass = _passes[_executionOrder[pos]];
                for (int k = 0; k < pass.BufferBarriers.Length; k++)
                    pass.BufferBarriers[k].Buffer = ResolveBufferFrame(pass.BufferBarrierRes[k], fr);

                _debug?.BeginPass(cb, fr, pos);
                EmitBarriers(cb, pass.ImageBarriers, pass.BufferBarriers);
                var resources = new PassResources(this,
                    pass.PassSets.Length > 0 ? pass.PassSets[fr] : default);
                pass.Execute(cb, resources, in frame);
                _debug?.EndPass(cb, fr, pos);
            }

            if (c == _lastGfxChunk) EmitBarriers(cb, _closingImageBarriers, []);
            if (vk.EndCommandBuffer(cb) != Result.Success)
                throw new Exception("FrameGraph: failed to end chunk command buffer");
        }

        // Map this frame's relative timeline values onto the monotonic cursors and store them
        // so SubmitGfxChunks can resolve the baked relative signal/wait values to absolute ones.
        ulong gfxBase = _gfxCursor; _gfxCursor += (ulong)_gfxSignalCount;
        ulong cmpBase = _cmpCursor; _cmpCursor += (ulong)_cmpSignalCount;
        _lastSubmitGfxBase = gfxBase;
        _lastSubmitCmpBase = cmpBase;
        _lastSubmitFr      = fr;

        // Async compute chunks submit immediately on the compute queue (they must go to a
        // different queue family; nothing about their submission needs the frame fence or
        // presentation semaphores). Timeline waits may precede their signals, so the async
        // submit can happen before the gfx submit without correctness issues.
        SubmitChunksFor(QueueClass.AsyncCompute, _plan.AsyncCompute!.Value.Handle, fr, gfxBase, cmpBase);
    }

    /// <summary>Submit all pending graphics chunks plus the caller's <paramref name="hostCmd"/>
    /// in one vkQueueSubmit2 call. Each graphics chunk retains its baked timeline waits/signals
    /// (async-compute synchronisation); the host command buffer goes last with the supplied
    /// binary frame-pacing semaphores and <paramref name="fence"/>. Call this after
    /// <see cref="Execute"/> when <see cref="HasPendingGfxChunks"/> is true, once the host
    /// command buffer is fully recorded and ended.</summary>
    public unsafe void SubmitGfxChunks(
        Queue gfxQueue,
        SemaphoreSubmitInfo imgAvailWait,
        SemaphoreSubmitInfo renderDoneSignal,
        CommandBuffer hostCmd,
        Fence fence)
    {
        uint  fr      = _lastSubmitFr;
        ulong gfxBase = _lastSubmitGfxBase;
        ulong cmpBase = _lastSubmitCmpBase;

        int n = 0, totalWaits = 0, totalSignals = 0;
        foreach (var chunk in _chunks)
        {
            if (chunk.Queue != QueueClass.Graphics) continue;
            n++;
            totalWaits   += chunk.Waits.Count;
            if (chunk.Signal != 0) totalSignals++;
        }

        // +1 slot for the host cmd (one binary wait, one binary signal).
        int total = n + 1;
        var submits     = stackalloc SubmitInfo2[total];
        var cmdInfos    = stackalloc CommandBufferSubmitInfo[total];
        var waitInfos   = stackalloc SemaphoreSubmitInfo[Math.Max(1, totalWaits + 1)];
        var signalInfos = stackalloc SemaphoreSubmitInfo[Math.Max(1, totalSignals + 1)];

        int si = 0, wi = 0, gi = 0;
        for (int c = 0; c < _chunks.Count; c++)
        {
            var chunk = _chunks[c];
            if (chunk.Queue != QueueClass.Graphics) continue;

            cmdInfos[si] = new CommandBufferSubmitInfo
            {
                SType = StructureType.CommandBufferSubmitInfo,
                CommandBuffer = _chunkCmds[fr][c],
            };

            int waitStart = wi;
            foreach (var (wq, rel) in chunk.Waits)
                waitInfos[wi++] = new SemaphoreSubmitInfo
                {
                    SType     = StructureType.SemaphoreSubmitInfo,
                    Semaphore = wq == QueueClass.Graphics ? _gfxTimeline : _cmpTimeline,
                    Value     = (wq == QueueClass.Graphics ? gfxBase : cmpBase) + rel,
                    StageMask = PipelineStageFlags2.AllCommandsBit,
                };

            int sigStart = gi;
            if (chunk.Signal != 0)
                signalInfos[gi++] = new SemaphoreSubmitInfo
                {
                    SType     = StructureType.SemaphoreSubmitInfo,
                    Semaphore = _gfxTimeline,
                    Value     = gfxBase + chunk.Signal,
                    StageMask = PipelineStageFlags2.AllCommandsBit,
                };

            submits[si++] = new SubmitInfo2
            {
                SType = StructureType.SubmitInfo2,
                CommandBufferInfoCount    = 1,
                PCommandBufferInfos       = &cmdInfos[si - 1],
                WaitSemaphoreInfoCount    = (uint)(wi - waitStart),
                PWaitSemaphoreInfos       = wi > waitStart ? &waitInfos[waitStart] : null,
                SignalSemaphoreInfoCount  = (uint)(gi - sigStart),
                PSignalSemaphoreInfos     = gi > sigStart ? &signalInfos[sigStart] : null,
            };
        }

        // Host cmd: binary frame-pacing wait + signal, plus the frame fence.
        waitInfos[wi]   = imgAvailWait;
        signalInfos[gi] = renderDoneSignal;
        cmdInfos[si] = new CommandBufferSubmitInfo
        {
            SType = StructureType.CommandBufferSubmitInfo,
            CommandBuffer = hostCmd,
        };
        submits[si] = new SubmitInfo2
        {
            SType = StructureType.SubmitInfo2,
            CommandBufferInfoCount   = 1,
            PCommandBufferInfos      = &cmdInfos[si],
            WaitSemaphoreInfoCount   = 1,
            PWaitSemaphoreInfos      = &waitInfos[wi],
            SignalSemaphoreInfoCount = 1,
            PSignalSemaphoreInfos    = &signalInfos[gi],
        };

        if (gfx.Vk.QueueSubmit2(gfxQueue, (uint)total, submits, fence) != Result.Success)
            throw new Exception("FrameGraph: SubmitGfxChunks QueueSubmit2 failed");
    }

    private void SubmitChunksFor(QueueClass q, Queue queue, uint fr, ulong gfxBase, ulong cmpBase)
    {
        int n = 0, totalWaits = 0, totalSignals = 0;
        foreach (var chunk in _chunks)
        {
            if (chunk.Queue != q) continue;
            n++;
            totalWaits += chunk.Waits.Count;
            if (chunk.Signal != 0) totalSignals++;
        }
        if (n == 0) return;

        var submits = stackalloc SubmitInfo2[n];
        var cmdInfos = stackalloc CommandBufferSubmitInfo[n];
        var waitInfos = stackalloc SemaphoreSubmitInfo[Math.Max(1, totalWaits)];
        var signalInfos = stackalloc SemaphoreSubmitInfo[Math.Max(1, totalSignals)];

        int si = 0, wi = 0, gi = 0;
        for (int c = 0; c < _chunks.Count; c++)
        {
            var chunk = _chunks[c];
            if (chunk.Queue != q) continue;

            cmdInfos[si] = new CommandBufferSubmitInfo
            {
                SType = StructureType.CommandBufferSubmitInfo,
                CommandBuffer = _chunkCmds[fr][c],
            };

            int waitStart = wi;
            foreach (var (wq, rel) in chunk.Waits)
                waitInfos[wi++] = new SemaphoreSubmitInfo
                {
                    SType = StructureType.SemaphoreSubmitInfo,
                    Semaphore = wq == QueueClass.Graphics ? _gfxTimeline : _cmpTimeline,
                    Value = (wq == QueueClass.Graphics ? gfxBase : cmpBase) + rel,
                    StageMask = PipelineStageFlags2.AllCommandsBit,
                };

            int sigStart = gi;
            if (chunk.Signal != 0)
                signalInfos[gi++] = new SemaphoreSubmitInfo
                {
                    SType = StructureType.SemaphoreSubmitInfo,
                    Semaphore = q == QueueClass.Graphics ? _gfxTimeline : _cmpTimeline,
                    Value = (q == QueueClass.Graphics ? gfxBase : cmpBase) + chunk.Signal,
                    StageMask = PipelineStageFlags2.AllCommandsBit,
                };

            submits[si] = new SubmitInfo2
            {
                SType = StructureType.SubmitInfo2,
                CommandBufferInfoCount = 1,
                PCommandBufferInfos = &cmdInfos[si],
                WaitSemaphoreInfoCount = (uint)(wi - waitStart),
                PWaitSemaphoreInfos = wi > waitStart ? &waitInfos[waitStart] : null,
                SignalSemaphoreInfoCount = (uint)(gi - sigStart),
                PSignalSemaphoreInfos = gi > sigStart ? &signalInfos[sigStart] : null,
            };
            si++;
        }

        if (gfx.Vk.QueueSubmit2(queue, (uint)n, submits, default) != Result.Success)
            throw new Exception($"FrameGraph: QueueSubmit2 failed for {q} chunks");
    }

    
    private void EmitBarriers(CommandBuffer cmd, ImageMemoryBarrier2[] imgs, BufferMemoryBarrier2[] bufs)
    {
        uint imgCount = (uint)imgs.Length;
        uint bufCount = (uint)bufs.Length;
        if (imgCount == 0 && bufCount == 0) return;

        fixed (ImageMemoryBarrier2*  pImg = imgs)
        fixed (BufferMemoryBarrier2* pBuf = bufs)
        {
            var dep = new DependencyInfo
            {
                SType = StructureType.DependencyInfo,
                ImageMemoryBarrierCount  = imgCount, PImageMemoryBarriers  = pImg,
                BufferMemoryBarrierCount = bufCount, PBufferMemoryBarriers = pBuf,
            };
            gfx.Vk.CmdPipelineBarrier2(cmd, &dep);
        }
    }

    // ---- Debug: instrumentation setup, stats, DOT export ---------------------

    /// <summary>Per-pass GPU/CPU timings + pipeline stats + counts from the last resolved
    /// frame, or null before the first Compile. See <see cref="GraphStats"/>.</summary>
    public GraphStats? Stats => _debug?.Snapshot(_compileMs, _culledCount, _barrierCount);

    /// <summary>Runtime toggle for pipeline-statistics collection (no-op when unsupported).</summary>
    public bool CollectPipelineStats
    {
        get => _debug?.CollectPipelineStats ?? false;
        set { if (_debug != null) _debug.CollectPipelineStats = value; }
    }

    private void SetupDebug()
    {
        int live = _executionOrder.Count;
        var names    = new string[live];
        var queues   = new QueueClass[live];
        var types    = new PassType[live];
        var eligible = new bool[live];
        for (int i = 0; i < live; i++)
        {
            int id = _executionOrder[i];
            var pass = _passes[id];
            var effQ = _effQueue.Length > 0 ? _effQueue[id] : QueueClass.Graphics;
            names[i]    = pass.Name;
            queues[i]   = effQ;   // effective (post-collapse) queue, not the declared one
            types[i]    = pass.Type;
            // Pipeline stats: draw/dispatch only, and NEVER on the async queue -- the stats
            // pool carries graphics counters (IA/VS/...) that a compute-only queue cannot
            // begin a query for. Timestamps are still written there (reset is ordered before
            // the async chunks via their graphics-timeline waits).
            eligible[i] = pass.Type != PassType.Transfer && effQ != QueueClass.AsyncCompute;
        }

        _barrierCount = _closingImageBarriers.Length;
        foreach (var idx in _executionOrder)
            _barrierCount += _passes[idx].ImageBarriers.Length + _passes[idx].BufferBarriers.Length;

        _debug = new GraphDebug(gfx);
        _debug.Initialize(names, queues, types, eligible, RenderConfig.MAX_CONCURRENT_FRAMES);

        // Name physical resources so captures read "GBuffer_Position" instead of a raw handle.
        foreach (var res in _resources.Values)
        {
            if (res.IsImage)
            {
                if (res.PhysImage is { } img)  _debug.NameObject(ObjectType.Image,     img.Handle,  res.Name);
                if (res.PhysView  is { } view) _debug.NameObject(ObjectType.ImageView, view.Handle, res.Name + "/view");
            }
            else if (res.PhysBuffer is { } buf)
            {
                _debug.NameObject(ObjectType.Buffer, buf.Handle, res.Name);
            }
            else if (res.PhysBufferFrames is { } frames)
            {
                for (int f = 0; f < frames.Length; f++)
                    _debug.NameObject(ObjectType.Buffer, frames[f].Handle, $"{res.Name}#{f}");
            }
        }
    }

    /// <summary>
    /// Graphviz dump of the compiled DAG. Passes are nodes (filled by queue, greyed if culled,
    /// prefixed with their schedule index), grouped into nested <c>subgraph cluster_*</c> boxes
    /// by module scope so the module hierarchy is visible (a module nested in another nests its
    /// box). Solid edges are RAW data deps labelled <c>resource@version usage</c>; dashed edges
    /// are WAW/WAR ordering deps; an edge whose endpoints live in different modules is drawn in
    /// blue (and thicker) so module crossings stand out. Two boundary nodes sit outside every
    /// cluster: <c>INPUTS</c> (dotted edges to the first pass that touches each imported
    /// resource) and <c>OUTPUTS</c> (dotted edges from the last writer of each MarkOutput'd
    /// resource), labelled with the resource name -- so what enters and leaves the graph is
    /// explicit. Paste into any Graphviz viewer.
    /// </summary>
    public string ToDot()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("digraph FrameGraph {");
        sb.AppendLine("  rankdir=LR;");
        sb.AppendLine("  compound=true;");
        sb.AppendLine("  node [shape=box, style=filled, fontname=\"Consolas\"];");

        // Build the module-scope cluster tree from each pass's Scope ("" = top level,
        // "Deferred", "Deferred/Sub", ...). Intermediate levels are materialized even when they
        // hold no passes directly, so a module that only contains submodules still nests.
        var root = new DotCluster("");
        for (int p = 0; p < _passes.Count; p++)
            ClusterFor(root, _passes[p].Scope).Passes.Add(p);

        EmitCluster(root, 0);

        // Boundary: imported resources enter at their first live access (Inputs source node);
        // marked outputs leave from their last live writer (Outputs sink node). Both sit outside
        // every module cluster -- they are the graph's external surface, not part of a module.
        var inEdges = new List<(int pass, string res)>();
        foreach (var res in _resources.Values.OrderBy(r => r.Id))
        {
            if (res.Residency != ResidencyKind.Imported) continue;
            int pi = FirstLiveAccess(res.Id);
            if (pi >= 0) inEdges.Add((pi, res.Name));
        }
        var outEdges = new List<(int pass, string res)>();
        foreach (int resId in _outputs.OrderBy(x => x))
        {
            if (!_resources.TryGetValue(resId, out var ores)) continue;
            int pi = LastLiveWriter(resId);
            if (pi < 0) pi = FirstLiveAccess(resId);   // imported + marked but never written: pass-through
            if (pi >= 0) outEdges.Add((pi, ores.Name));
        }
        if (inEdges.Count > 0)
        {
            sb.AppendLine("  INPUTS [shape=cds, style=filled, fillcolor=\"#fff0c0\", fontname=\"Consolas\", label=\"inputs\\n(imported)\"];");
            foreach (var (pi, res) in inEdges)
                sb.AppendLine($"  INPUTS -> p{pi} [label=\"{Escape(res)}\", fontsize=9, style=dotted, color=\"#b08000\"];");
        }
        if (outEdges.Count > 0)
        {
            sb.AppendLine("  OUTPUTS [shape=cds, style=filled, fillcolor=\"#ffd0d0\", fontname=\"Consolas\", label=\"outputs\\n(marked)\"];");
            foreach (var (pi, res) in outEdges)
                sb.AppendLine($"  p{pi} -> OUTPUTS [label=\"{Escape(res)}\", fontsize=9, style=dotted, color=\"#b03030\"];");
        }

        // Edges last (Graphviz wants nodes defined in their cluster first). Cross-module edges
        // -- from/to passes in different scopes -- are accented so module boundaries are obvious.
        foreach (var (from, to, lbl, dashed) in _dotEdges)
        {
            bool crossModule = !string.Equals(_passes[from].Scope, _passes[to].Scope, StringComparison.Ordinal);
            string color  = crossModule ? "\"#3060c0\"" : dashed ? "\"#999999\"" : "\"#333333\"";
            string dash   = dashed ? ", style=dashed" : "";
            string weight = crossModule ? ", penwidth=1.6" : "";
            sb.AppendLine($"  p{from} -> p{to} [label=\"{Escape(lbl)}\", fontsize=9, color={color}{dash}{weight}];");
        }

        sb.AppendLine("}");
        return sb.ToString();

        // ---- locals --------------------------------------------------------------
        // Navigate/create the cluster node for a "/"-delimited scope path.
        DotCluster ClusterFor(DotCluster r, string scope)
        {
            if (string.IsNullOrEmpty(scope)) return r;
            var node = r;
            var path = "";
            foreach (var seg in scope.Split('/'))
            {
                path = path.Length == 0 ? seg : $"{path}/{seg}";
                if (!node.Children.TryGetValue(seg, out var child))
                    node.Children[seg] = child = new DotCluster(path);
                node = child;
            }
            return node;
        }

        // Emit this cluster's direct passes, then recurse into child module clusters.
        void EmitCluster(DotCluster node, int depth)
        {
            string ind = new(' ', (depth + 1) * 2);
            foreach (int p in node.Passes) sb.AppendLine(ind + NodeLine(p));
            foreach (var (seg, child) in node.Children)
            {
                sb.AppendLine($"{ind}subgraph cluster_{Sanitize(child.Path)} {{");
                sb.AppendLine($"{ind}  label=\"{Escape(seg)}\"; labeljust=l; fontname=\"Consolas\"; fontsize=11;");
                sb.AppendLine($"{ind}  style=\"filled,rounded\"; fillcolor=\"#f4f4fb\"; color=\"#b9b9c8\";");
                EmitCluster(child, depth + 1);
                sb.AppendLine($"{ind}}}");
            }
        }

        string NodeLine(int p)
        {
            var pass = _passes[p];
            bool isLive = p < _live.Length && _live[p];
            int order = _executionOrder.IndexOf(p);
            string leaf = Leaf(pass.Name);
            string fill = isLive ? QueueFill(pass.Queue) : "\"#dddddd\"";
            string label = isLive
                ? $"{order}: {Escape(leaf)}\\n{pass.Type}/{pass.Queue}"
                : $"{Escape(leaf)}\\n(culled)";
            string extra = isLive ? "" : ", fontcolor=\"#888888\"";
            return $"p{p} [label=\"{label}\", fillcolor={fill}{extra}];";
        }

        // First live pass (schedule order) that reads or writes a resource, or -1.
        int FirstLiveAccess(int resId)
        {
            foreach (int pi in _executionOrder)
            {
                foreach (var a in _passes[pi].Reads)  if (a.ResourceId == resId) return pi;
                foreach (var a in _passes[pi].Writes) if (a.ResourceId == resId) return pi;
            }
            return -1;
        }

        // Last live pass (schedule order) that writes a resource, or -1.
        int LastLiveWriter(int resId)
        {
            for (int k = _executionOrder.Count - 1; k >= 0; k--)
                foreach (var a in _passes[_executionOrder[k]].Writes)
                    if (a.ResourceId == resId) return _executionOrder[k];
            return -1;
        }

        // Leaf = the segment after the last "/" (the unscoped pass name).
        static string Leaf(string name)
        {
            int i = name.LastIndexOf('/');
            return i < 0 ? name : name[(i + 1)..];
        }

        // Graphviz cluster ids must be alphanumeric; map "/" and punctuation to "_".
        static string Sanitize(string s)
        {
            var chars = s.ToCharArray();
            for (int i = 0; i < chars.Length; i++)
                if (!char.IsLetterOrDigit(chars[i])) chars[i] = '_';
            return new string(chars);
        }

        static string QueueFill(QueueClass q) => q switch
        {
            QueueClass.Graphics     => "\"#cfe8cf\"",
            QueueClass.AsyncCompute => "\"#cfe0f5\"",
            QueueClass.Transfer     => "\"#f5e3cf\"",
            _ => "\"#eeeeee\"",
        };
        static string Escape(string s) => s.Replace("\"", "\\\"");
    }

    // Module-scope tree, built transiently by ToDot to emit nested Graphviz clusters. Children
    // are sorted so the DOT output is deterministic frame to frame.
    private sealed class DotCluster(string path)
    {
        public readonly string Path = path;
        public readonly SortedDictionary<string, DotCluster> Children = new(StringComparer.Ordinal);
        public readonly List<int> Passes = [];
    }

    public void Dispose()
    {
        if (_disposed) return;
        _debug?.Dispose();
        var vk  = gfx.Vk;
        var dev = gfx.Device;

        // Async-compute submission plan (no-ops when the graph never chunked). Pools free
        // their command buffers; the caller guarantees idle (graphs are rebuilt under
        // DeviceWaitIdle on resize/mode-switch, same contract as the transient images below).
        if (_gfxChunkPool.Handle != 0) vk.DestroyCommandPool(dev, _gfxChunkPool, null);
        if (_cmpChunkPool.Handle != 0) vk.DestroyCommandPool(dev, _cmpChunkPool, null);
        if (_gfxTimeline.Handle  != 0) vk.DestroySemaphore(dev, _gfxTimeline, null);
        if (_cmpTimeline.Handle  != 0) vk.DestroySemaphore(dev, _cmpTimeline, null);

        // Graph-baked pass sets + the graph-owned shared set: freed with their pools.
        if (_passSetPool.Handle != 0) vk.DestroyDescriptorPool(dev, _passSetPool, null);
        if (_graphSharedPool.Handle != 0) vk.DestroyDescriptorPool(dev, _graphSharedPool, null);

        foreach (var res in _resources.Values)
        {
            if (res.Residency != ResidencyKind.Transient) continue;   // never free imports
            if (res.IsImage)
            {
                if (res.PhysView  is { Handle: not 0 } v)  vk.DestroyImageView(dev, v, null);
                if (res.PhysImage is { Handle: not 0 } im) vk.DestroyImage(dev, im, null);
                if (res.Alloc is { } a) gfx.Allocator.Free(a);
            }
            else if (res.PhysBuffer is { } b)
            {
                gfx.DestroyBuffer(b, res.Alloc ?? default);
            }
        }
        _resources.Clear();
        _disposed = true;
        GC.SuppressFinalize(this);
    }
}