using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.FrameGraph;

public unsafe class FrameGraph : IDisposable
{
    private readonly GraphicsDevice gfx;

    private int _nextResourceId;
    private bool _disposed;
    // Restores imported images to their declared final layout after the last pass, so the
    // baked first-use barriers stay valid every frame. Empty when nothing needs restoring.
    private ImageMemoryBarrier2[] _closingImageBarriers = [];

    public FrameGraph(GraphicsDevice device) => gfx = device;

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


    public void AddPass(string name, PassType type, QueueClass queue,
        PassSetup setup, PassExecute execute,
        bool preferAsync = false, bool hasSideEffects = false)
    {
        var pass = new GraphPass
        {
            Name = name, Type = type, Queue = queue, PreferAsync = preferAsync, HasSideEffects = hasSideEffects, Execute = execute
        };
        _passes.Add(pass);
        CurrentPassIndex = _passes.Count - 1;
        setup(new GraphBuilder(this, pass)); //Read/Write mutate version ledger + pass lists
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

        int n = _passes.Count;

        // ---- 1. Build the DAG from the version ledger ----------------------------
        // Edges point producer -> consumer ("from must run before to"). Dedupe through a
        // HashSet: a duplicate edge would double-count inDegree below and strand Kahn's
        // sort, falsely reporting a cycle. preds is the inverse adjacency, used by the
        // reverse-reachability cull in step 2.
        var adj   = new HashSet<int>[n];
        var preds = new HashSet<int>[n];
        for (int i = 0; i < n; i++) { adj[i] = []; preds[i] = []; }

        void AddEdge(int from, int to)
        {
            // from < 0  == "version 0 sentinel / imported — no producing pass" -> no edge.
            if (from < 0 || to < 0 || from == to) return;
            if (adj[from].Add(to)) preds[to].Add(from);
        }

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

        for (int p = 0; p < n; p++)
        {
            var pass = _passes[p];

            // RAW: a read of (res, v) depends on whoever produced v.
            foreach (var rd in pass.Reads)
                AddEdge(ProducerOf(rd.ResourceId, rd.Version), p);

            // WAW + WAR: a write produces (res, v) from the prior version (res, v-1).
            foreach (var wr in pass.Writes)
            {
                // Version 0 is the -1 producer sentinel (fresh/imported); the first real
                // write lands at v >= 1, so a previous version only exists from there.
                if (wr.Version < 1) continue;

                // WAW: serialize after the previous version's producer.
                AddEdge(ProducerOf(wr.ResourceId, wr.Version - 1), p);

                // WAR: every reader of the previous version must finish before we overwrite.
                if (readers.TryGetValue((wr.ResourceId, wr.Version - 1), out var rs))
                    foreach (var reader in rs) AddEdge(reader, p);
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

        Compiled = true;
        return true;

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

    // ---- Physical-handle resolution (used by PassResources during Execute) --------
    // Valid only after Compile()'s step 5 has populated the phys handles; until then
    // these throw, which is the correct signal that Execute ran before allocation.
    internal ImageView ResolveView(GraphImage h) =>
        _resources[h.resourceId].PhysView ?? throw new InvalidOperationException(
            $"FrameGraph: image '{_resources[h.resourceId].Name}' has no view (not allocated/imported).");

    internal Image ResolveImage(GraphImage h) =>
        _resources[h.resourceId].PhysImage ?? throw new InvalidOperationException(
            $"FrameGraph: image '{_resources[h.resourceId].Name}' not allocated.");

    internal Buffer ResolveBuffer(GraphBuffer h) =>
        _resources[h.resourceId].PhysBuffer ?? throw new InvalidOperationException(
            $"FrameGraph: buffer '{_resources[h.resourceId].Name}' not allocated.");

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
        
        var alloc = gfx.Allocator.AllocateForImage(image, MemoryPropertyFlags.DeviceLocalBit);

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

        foreach (var passIdx in _executionOrder)
        {
            var pass = _passes[passIdx];
            var imgs = new List<ImageMemoryBarrier2>();
            var bufs = new List<BufferMemoryBarrier2>();
            foreach (var a in pass.Reads)  ProcessAccess(in a, state, imgs, bufs);
            foreach (var a in pass.Writes) ProcessAccess(in a, state, imgs, bufs);
            pass.ImageBarriers  = imgs.ToArray();
            pass.BufferBarriers = bufs.ToArray();
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

    private void ProcessAccess(in ResourceAccess a, Dictionary<int, UsageInfo> state,
        List<ImageMemoryBarrier2> imgs, List<BufferMemoryBarrier2> bufs)
    {
        var next = UsageTable.Of(a.Usage);
        var cur  = state[a.ResourceId];

        bool readAfterRead = !next.IsWrite && !cur.IsWrite;
        bool layoutChange  = a.IsImage && cur.Layout != next.Layout;

        // Pure read-after-read in a compatible layout needs no barrier — but widen the
        // visibility set so the eventual writer waits on EVERY prior reader, not just the
        // last (two readers on different stages would otherwise drop the first).
        if (readAfterRead && !layoutChange)
        {
            state[a.ResourceId] = cur with { Stage = cur.Stage | next.Stage, Access = cur.Access | next.Access };
            return;
        }

        var res = _resources[a.ResourceId];
        if (a.IsImage)
            imgs.Add(MakeImageBarrier(res, cur.Stage, cur.Access, cur.Layout,
                                      next.Stage, next.Access, next.Layout));
        else
            bufs.Add(MakeBufferBarrier(res, cur.Stage, cur.Access, next.Stage, next.Access));

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
            Buffer = res.PhysBuffer!.Value, Offset = 0, Size = Vk.WholeSize,
        };

    private static bool IsDepthFormat(Format f) =>
        f is Format.D32Sfloat or Format.D24UnormS8Uint or Format.D16Unorm
          or Format.D32SfloatS8Uint or Format.D16UnormS8Uint;

    /// <summary>
    /// Replays the baked plan: per scheduled pass, emit its barrier batch then record its
    /// body; finally hand imported images back in their declared final layout. No
    /// per-frame analysis — <see cref="Compile"/> did it all.
    /// </summary>
    public void Execute(CommandBuffer cmd, in Renderer.FrameContext frame)
    {
        if (!Compiled) throw new InvalidOperationException("FrameGraph: call Compile() before Execute().");

        var resources = new PassResources(this);
        foreach (var passIdx in _executionOrder)
        {
            var pass = _passes[passIdx];
            EmitBarriers(cmd, pass.ImageBarriers, pass.BufferBarriers);
            pass.Execute(cmd, resources, in frame);
        }
        EmitBarriers(cmd, _closingImageBarriers, []);
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

    public void Dispose()
    {
        if (_disposed) return;
        var vk  = gfx.Vk;
        var dev = gfx.Device;
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