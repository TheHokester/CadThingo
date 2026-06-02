# Render Graph — Implementation Guide

The companion to [`render-graph.md`](render-graph.md). That doc is the **design** (the
"why" and the target shape); this one is the **how**, grounded in the types that exist
in this repo today. It is a build order: a sequence of steps that each compile and ship,
mapped onto concrete files, signatures, and call sites.

Read `render-graph.md` first for the model (versioned virtual resources, usage-derived
sync, queue classes, modules). Everything here assumes that vocabulary.

> **Build between every step.** From bash: `/c/Users/jamie/.dotnet/dotnet.exe build
> CadThingo.sln -c Debug` (bare `dotnet` can report "No SDKs found" here). The tree is
> expected to be green after each numbered step below — if it isn't, stop and fix before
> moving on. This is the same discipline as `renderer-refactor.md`.

---

## 0. Where the code actually is today

| Concern | File | State |
|---|---|---|
| Current graph | `Renderer/RenderGraph.cs` | image-only, single-queue, `j>i` ordering hack, v1 `CmdPipelineBarrier` with `AllCommandsBit`/`MemoryReadBit` |
| Pass record | `Renderer/Pass.cs` | `struct Pass` — string in/out lists + `Action<CommandBuffer, FrameContext>` |
| Image wrapper | `Renderer/Renderer_Resources.cs` → `ImageResource` | owns `VkImage`/`SubAlloc`/`ImageView`; `Allocate(physicalDevice)` |
| Allocator | `Renderer/GpuMemoryAllocator.cs` | `AllocateForBuffer/Image`, `Free`, `GetMapped`, dedicated + suballoc blocks |
| Device surface | `Renderer/GraphicsDevice.cs` | `Vk/Device/PhysicalDevice/Allocator`, 4 queues, `QueueFamilyIndices`, caps. **Sync2 + timeline semaphores already enabled** (`vulkan13Features.Synchronization2`, `vulkan12Features.TimelineSemaphore`). |
| Queue discovery | `GraphicsDevice.FindQueueFamilies` | has the exact "compute family == graphics family @ index 0" trap §3 of the design warns about |
| Scene mirror (L2) | `Renderer/GpuScene.cs` | owns light + renderable SSBOs, `RenderableHandle` allocator, `RenderView` (stub), `Extract*` |
| Frame ctx | `Renderer_Rendering.cs` → `FrameContext` | `FrameIndex/Camera/Scene/RenderExtent` |
| Graph wiring | `Renderer_Rendering.cs` → `SetupDeferredRenderer` | declares g-buffers/HDR/Final, 5 passes (Geometry→Lighting→Skybox→Transparent→Tonemap) |
| Hand-recorded (NOT in graph) | `Renderer_Rendering.cs` → `DrawPathtraced`, `DrawRayTraced`, `ProcessPickRequest`, `RecordSelectionOutline`; `DrawDeferred` for cull/light-cull | manual `CmdPipelineBarrier` everywhere |
| New scaffold (started) | `Renderer/FrameGraph/QueuePlan.cs`, `IGraphBuilder.cs`, `IGraphModule.cs` | **stubs** — `QueuePlan` fields are private and unpopulated, `IGraphBuilder` is empty, `IGraphModule.Build` has `in TOutputs` (should be `out`) |

### Decisions locked for this guide

1. **New type, side-by-side.** Build the new graph as `FrameGraph` in the existing
   `CadThingo.VulkanEngine.Renderer.FrameGraph` namespace. The old `RenderGraph` keeps
   running the deferred chain until `FrameGraph` reaches parity; the final cutover
   deletes `RenderGraph.cs` and `Pass.cs`. No big-bang rewrite.
2. **Pipelines keep doing their own `CmdBeginRendering` in Phase 1.** The graph resolves
   physical handles and emits all sync; pass bodies stay near-identical to today's
   closures. Hoisting attachment setup into the graph is a *named follow-up step*
   (1.9), not a precondition — keeps each step bisectable.
3. **Present / blit / ImGui stay outside the graph.** The graph's output is `FinalColor`
   (imported). The host's step-7 blit + overlay is unchanged. We are replacing the
   *scene* graph, not the swapchain plumbing.
4. **Phase target is 1 + 2.** Async compute (3) and aliasing (4) are sketched but the
   guide goes deep only through the transfer queue.

---

## 1. Phase 1 — versioned virtual resources, buffers+images, Sync2, single queue

The high-value core. End state of Phase 1: the deferred chain *plus* cull and light-cull
run through `FrameGraph`; every barrier is derived from a usage table via
`CmdPipelineBarrier2`; the `j>i` hack is gone; DOT export + per-pass GPU timing + a
compile-time validation pass exist.

### File plan (all under `Renderer/FrameGraph/`)

```
FrameGraph.cs            // the graph: registry, builder impl, compile, execute, dispose
GraphResources.cs        // GraphImage, GraphBuffer, ImageDesc, BufferDesc, ResidencyKind, virtual records
ResourceUsage.cs         // enum ResourceUsage + the usage→(stage2,access2,layout) table  ← the sync engine
GraphPass.cs             // PassType, PassDesc, recorded access lists, PassResources
GraphBuilder.cs          // IGraphBuilder concrete impl (replaces the empty interface body)
GraphDebug.cs            // DOT export, timestamp pool, GraphStats, validation
QueuePlan.cs             // (exists) — finish in Phase 2; Phase 1 uses Graphics only
```

### 1.1 Resource model (`GraphResources.cs`)

Handles are versioned value types; descs are creation parameters; the graph holds the
mutable virtual records.

```csharp
public readonly record struct GraphImage(int ResourceId, int Version);
public readonly record struct GraphBuffer(int ResourceId, int Version);

public struct ImageDesc
{
    public Format Format;
    public Extent2D Extent;
    public uint Mips, Layers;
    public ImageUsageFlags Usage;
    public SampleCountFlags Samples;   // default Count1Bit
}

public struct BufferDesc
{
    public ulong Size;
    public BufferUsageFlags Usage;
}

public enum ResidencyKind { Transient, Imported }

// One per virtual resource id. Image and buffer share the array via Kind.
internal sealed class GraphResource
{
    public int Id;
    public string Name;
    public bool IsImage;
    public ResidencyKind Residency;

    public ImageDesc  ImageDesc;     // valid when IsImage
    public BufferDesc BufferDesc;    // valid when !IsImage

    // Versioning: producers[v] = pass index that produced version v (or -1 = initial).
    public List<int> Producers = new() { -1 };
    public int CurrentVersion => Producers.Count - 1;

    // Filled at compile. Imported resources adopt an externally-owned handle.
    public Image       PhysImage;
    public ImageView   PhysView;
    public VkBuffer    PhysBuffer;
    public SubAlloc    Alloc;        // transients only
    public ImageLayout InitialLayout;  // imported: caller-declared current layout
}
```

`Producers` is the SSA ledger. A write appends a new producer; `CurrentVersion` is what a
read captures. This is the structure that replaces the `j>i` restriction — the topo sort
reads edges off `Producers`, never off declaration order.

### 1.2 The usage table (`ResourceUsage.cs`) — this *is* the sync engine

```csharp
public enum ResourceUsage
{
    ColorAttachment, DepthAttachment, DepthRead,
    SampledFragment, SampledCompute,
    StorageReadCompute, StorageWriteCompute, StorageRWCompute,
    IndirectArg, IndexBuffer, VertexBuffer, UniformRead,
    TransferSrc, TransferDst,
    StorageRT, AccelStructBuild, AccelStructRead,
    Present,
}

internal readonly record struct UsageInfo(
    PipelineStageFlags2 Stage, AccessFlags2 Access, ImageLayout Layout, bool IsWrite);

internal static class UsageTable
{
    // The whole point: one place maps usage → barrier params. Add a row, get correct sync.
    public static UsageInfo Of(ResourceUsage u) => u switch
    {
        ResourceUsage.ColorAttachment => new(
            PipelineStageFlags2.ColorAttachmentOutputBit,
            AccessFlags2.ColorAttachmentWriteBit | AccessFlags2.ColorAttachmentReadBit,
            ImageLayout.ColorAttachmentOptimal, true),

        ResourceUsage.DepthAttachment => new(
            PipelineStageFlags2.EarlyFragmentTestsBit | PipelineStageFlags2.LateFragmentTestsBit,
            AccessFlags2.DepthStencilAttachmentWriteBit | AccessFlags2.DepthStencilAttachmentReadBit,
            ImageLayout.DepthStencilAttachmentOptimal, true),

        ResourceUsage.DepthRead => new(
            PipelineStageFlags2.EarlyFragmentTestsBit | PipelineStageFlags2.FragmentShaderBit,
            AccessFlags2.DepthStencilAttachmentReadBit,
            ImageLayout.DepthStencilReadOnlyOptimal, false),

        ResourceUsage.SampledFragment => new(
            PipelineStageFlags2.FragmentShaderBit, AccessFlags2.ShaderSampledReadBit,
            ImageLayout.ShaderReadOnlyOptimal, false),

        ResourceUsage.SampledCompute => new(
            PipelineStageFlags2.ComputeShaderBit, AccessFlags2.ShaderSampledReadBit,
            ImageLayout.ShaderReadOnlyOptimal, false),

        ResourceUsage.StorageReadCompute => new(
            PipelineStageFlags2.ComputeShaderBit, AccessFlags2.ShaderStorageReadBit,
            ImageLayout.General, false),
        ResourceUsage.StorageWriteCompute => new(
            PipelineStageFlags2.ComputeShaderBit, AccessFlags2.ShaderStorageWriteBit,
            ImageLayout.General, true),
        ResourceUsage.StorageRWCompute => new(
            PipelineStageFlags2.ComputeShaderBit,
            AccessFlags2.ShaderStorageReadBit | AccessFlags2.ShaderStorageWriteBit,
            ImageLayout.General, true),

        ResourceUsage.IndirectArg => new(
            PipelineStageFlags2.DrawIndirectBit, AccessFlags2.IndirectCommandReadBit,
            ImageLayout.Undefined /* buffer */, false),

        ResourceUsage.TransferSrc => new(
            PipelineStageFlags2.AllTransferBit, AccessFlags2.TransferReadBit,
            ImageLayout.TransferSrcOptimal, false),
        ResourceUsage.TransferDst => new(
            PipelineStageFlags2.AllTransferBit, AccessFlags2.TransferWriteBit,
            ImageLayout.TransferDstOptimal, true),

        ResourceUsage.StorageRT => new(
            PipelineStageFlags2.RayTracingShaderBitKhr,
            AccessFlags2.ShaderStorageReadBit | AccessFlags2.ShaderStorageWriteBit,
            ImageLayout.General, true),
        ResourceUsage.AccelStructBuild => new(
            PipelineStageFlags2.AccelerationStructureBuildBitKhr,
            AccessFlags2.AccelerationStructureWriteBitKhr, ImageLayout.Undefined, true),
        ResourceUsage.AccelStructRead => new(
            PipelineStageFlags2.RayTracingShaderBitKhr | PipelineStageFlags2.ComputeShaderBit,
            AccessFlags2.AccelerationStructureReadBitKhr, ImageLayout.Undefined, false),

        ResourceUsage.Present => new(
            PipelineStageFlags2.BottomOfPipeBit, AccessFlags2.None,
            ImageLayout.PresentSrcKhr, false),

        _ => throw new ArgumentOutOfRangeException(nameof(u)),
    };
}
```

Notes specific to this codebase:
- Silk.NET 2.23 exposes `PipelineStageFlags2` / `AccessFlags2` / `ImageMemoryBarrier2` /
  `BufferMemoryBarrier2` / `DependencyInfo` / `CmdPipelineBarrier2` because core 1.3
  Synchronization2 is on. Use those, not the v1 `*Flags` the old graph uses.
- `DepthAttachment` is a *write* usage (depth test+write); `DepthRead` is the read-only
  variant for a pass that samples or depth-tests without writing (the Skybox/Transparent
  passes depth-test EQUAL/LE with no write — they should declare `DepthRead`, which is
  the fix for today's "both write Depth so the writer→writer edge serializes them" hack).

### 1.3 Pass model (`GraphPass.cs`)

```csharp
public enum PassType { Graphics, Compute, RayTrace, Transfer }

public delegate void PassSetup(IGraphBuilder b);
public delegate void PassExecute(CommandBuffer cmd, PassResources res, in FrameContext frame);
// NOTE: keep FrameContext for Phase 1 (camera/frame/scene). Swap to RenderView when
// GpuScene's RenderView lands (L2 step done) — same shape, narrower surface.

internal struct ResourceAccess
{
    public int ResourceId;
    public int Version;          // version observed (read) or produced (write)
    public ResourceUsage Usage;
    public bool IsWrite;
    public bool IsImage;
}

internal sealed class GraphPass
{
    public string Name;
    public PassType Type;
    public QueueClass Queue;     // Phase 1: always Graphics
    public bool PreferAsync;     // Phase 3 hint; ignored in 1–2
    public bool HasSideEffects;  // keep through dead-code cull even with no consumer
    public PassExecute Execute;

    public List<ResourceAccess> Reads  = new();
    public List<ResourceAccess> Writes = new();

    // Filled at compile (graphics passes): color/depth attachments for auto-rendering (1.9).
    public List<int> ColorTargets = new();
    public int DepthTarget = -1;
}
```

`PassResources` is what the execute body gets — it resolves declared handles to physical
Vulkan objects, so a body never touches graph internals or string names:

```csharp
public readonly struct PassResources
{
    private readonly FrameGraph _g;
    internal PassResources(FrameGraph g) => _g = g;
    public ImageView View(GraphImage h)   => _g.ResolveView(h);
    public Image     Image(GraphImage h)  => _g.ResolveImage(h);
    public VkBuffer  Buffer(GraphBuffer h)=> _g.ResolveBuffer(h);
}
```

### 1.4 The builder (`GraphBuilder.cs`)

Replace the empty `IGraphBuilder`. Read captures current version; write appends a version
and returns the new handle.

```csharp
public interface IGraphBuilder
{
    GraphImage  CreateImage(in ImageDesc desc, string name);
    GraphBuffer CreateBuffer(in BufferDesc desc, string name);

    GraphImage  Read (GraphImage h,  ResourceUsage usage);
    GraphBuffer Read (GraphBuffer h, ResourceUsage usage);
    GraphImage  Write(GraphImage h,  ResourceUsage usage);   // → new version
    GraphBuffer Write(GraphBuffer h, ResourceUsage usage);   // → new version

    string Scope { get; }   // module namespacing (Phase 3); "" at top level
}
```

```csharp
internal sealed class GraphBuilder : IGraphBuilder
{
    private readonly FrameGraph _g;
    private readonly GraphPass _pass;       // the pass currently being set up
    public string Scope { get; }

    public GraphImage Read(GraphImage h, ResourceUsage usage)
    {
        _pass.Reads.Add(new ResourceAccess {
            ResourceId = h.ResourceId, Version = h.Version,
            Usage = usage, IsWrite = false, IsImage = true });
        return h;
    }

    public GraphImage Write(GraphImage h, ResourceUsage usage)
    {
        var r = _g.Resource(h.ResourceId);
        r.Producers.Add(_g.CurrentPassIndex);     // new version, this pass produces it
        var nv = new GraphImage(h.ResourceId, r.CurrentVersion);
        _pass.Writes.Add(new ResourceAccess {
            ResourceId = h.ResourceId, Version = nv.Version,
            Usage = usage, IsWrite = true, IsImage = true });
        if (usage is ResourceUsage.ColorAttachment) _pass.ColorTargets.Add(h.ResourceId);
        if (usage is ResourceUsage.DepthAttachment) _pass.DepthTarget = h.ResourceId;
        return nv;
    }
    // buffer overloads identical sans layout/attachment bookkeeping
}
```

The public `FrameGraph.AddPass` runs `Setup` immediately (records accesses), storing the
`Execute` for replay:

```csharp
public void AddPass(string name, PassType type, QueueClass queue,
                    PassSetup setup, PassExecute execute,
                    bool preferAsync = false, bool hasSideEffects = false)
{
    var pass = new GraphPass { Name = name, Type = type, Queue = queue,
        PreferAsync = preferAsync, HasSideEffects = hasSideEffects, Execute = execute };
    _passes.Add(pass);
    _currentPassIndex = _passes.Count - 1;
    setup(new GraphBuilder(this, pass));   // Read/Write mutate version ledger + pass lists
}
```

### 1.5 Compile (`FrameGraph.Compile`)

Runs once per topology change, **not** per frame (same as today). Steps mirror
`render-graph.md` §6.

```
1. Build DAG from versions:
   - RAW: for each read of (res,v) → edge Producers[v] → this pass.
   - WAW: edge Producers[v-1] → Producers[v].
   - WAR: for each reader of (res, v-1), edge reader → Producers[v].
   (All from the version ledger — no j>i, no declaration-order writer→writer hack.)
2. Cull dead passes: roots = passes that write an Imported resource, or HasSideEffects,
   or write a resource reachable from a MarkOutput'd handle. Reverse-reachability; drop
   the rest + their transient resources.
3. Schedule: Kahn topo sort (reuse the existing algorithm in RenderGraph.Compile). Phase 1
   is single-queue, so the output is one ordered list. Keep it stable (enqueue in id order)
   so DOT output is deterministic.
4. Lifetimes: first/last scheduled index each transient is touched. (Stored now; only
   consumed by aliasing in Phase 4. Compute it here so the data's ready.)
5. Allocate: transients via the allocator (1.6); imported resources adopt their handle.
6. Generate sync: walk the scheduled order maintaining a per-resource "last access"
   (stage/access/layout). For each pass, batch the transitions of all its reads+writes
   into one DependencyInfo (1.7). Store the baked barrier batch on the pass.
7. Bake: freeze ordered passes + per-pass barrier batch + resolved physical map.
```

Reuse Kahn's algorithm verbatim from `RenderGraph.Compile` (lines 161–180) — only the
*edge construction* changes (versions, not `j>i`).

### 1.6 Allocation + import

Transients: build the `VkImage`/`VkBuffer`, bind via `GpuMemoryAllocator`. For images you
can lift `ImageResource.Allocate` almost verbatim (it already does
`CreateImage` → `Allocator.AllocateForImage` → `CreateImageView`). For buffers use
`AllocateForBuffer`. Free in `Dispose`.

Import (externals never allocated/aliased):

```csharp
public GraphImage ImportImage(Image img, ImageView view, in ImageDesc desc,
                              ImageLayout currentLayout, string name) { ... }
public GraphBuffer ImportBuffer(VkBuffer buf, in BufferDesc desc, string name) { ... }
```

Imports for this engine:
- **`FinalColor`** (host-owned `RenderTargets`/`ImageResource`) — graph output, left in
  `ShaderReadOnlyOptimal` for the host blit + ImGui sampler. Import it; don't let the
  graph own it (the host's step-7 blit dances its layout).
- **`GpuScene` SSBOs** — `GetLightStorageBuffer(frame)`, `GetRenderablesBuffer(frame)`,
  material SSBO, and (later) shadow-info/emissive/TLAS. Import per frame (they're
  per-frame-in-flight).
- **`DrawCullPipeline` outputs** — indirect-cmd + indirect-count + post-cull instance
  buffers. Import or let the cull pass `CreateBuffer` them; importing is less churn for
  the first migration since the cull pipeline already owns them.

### 1.7 Barrier emission (the replacement for `RenderGraph.Execute`'s three barrier loops)

Per pass, one `CmdPipelineBarrier2` covering every input/output transition:

```csharp
// During compile, per pass, after computing transitions:
foreach (access in pass.Reads.Concat(pass.Writes))
{
    var prev = lastAccess[access.ResourceId];        // stage/access/layout from prior pass
    var next = UsageTable.Of(access.Usage);
    if (access.IsImage)
        imageBarriers.Add(new ImageMemoryBarrier2 {
            SType = StructureType.ImageMemoryBarrier2,
            SrcStageMask = prev.Stage, SrcAccessMask = prev.Access,
            DstStageMask = next.Stage, DstAccessMask = next.Access,
            OldLayout = prev.Layout,   NewLayout = next.Layout,
            SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
            DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
            Image = Resource(access.ResourceId).PhysImage,
            SubresourceRange = AspectRangeFor(access.ResourceId) });
    else
        bufferBarriers.Add(new BufferMemoryBarrier2 { /* analogous, no layout */ });
    lastAccess[access.ResourceId] = next;
}
// bake: DependencyInfo { PImageMemoryBarriers, PBufferMemoryBarriers, counts }
```

Execute per frame just replays: for each scheduled pass, `CmdPipelineBarrier2(cmd, &dep)`
then `pass.Execute(cmd, new PassResources(this), in frame)`. No per-frame analysis. This
is strictly fewer, tighter barriers than today's three-`AllCommandsBit`-loops-per-pass.

> **Aspect mask:** reuse the depth-format test already duplicated across the codebase
> (`Format.D32Sfloat or D24UnormS8Uint or D16Unorm` → `ImageAspectFlags.DepthBit`). Put it
> in one helper on the graph; today it's copy-pasted in `RenderGraph`, `ImageResource`, and
> the draw paths.

### 1.8 Migrate the deferred chain (first real cutover)

Port `SetupDeferredRenderer` to the new API. Side-by-side translation of each pass — the
*bodies* barely change (they already fetch `ImageView`s and call `pipeline.Record`):

```csharp
// images
var pos  = g.CreateImage(GBufferPositionDesc, "GBuffer_Position");
// ... normal/albedo/material/emissive ...
var depth = g.CreateImage(DepthDesc, "Depth");
var hdr  = g.CreateImage(HdrDesc,   "HDRColor");
var final = g.ImportImage(finalColorImg, finalColorView, FinalDesc,
                          ImageLayout.ShaderReadOnlyOptimal, "FinalColor");

// cull + light-cull as REAL compute passes (delete their manual barriers in DrawDeferred)
var indirectCmd = g.ImportBuffer(drawCullPipeline.GetIndirectCmdBuffer(frame), ...);
g.AddPass("Cull", PassType.Compute, QueueClass.Graphics,
    b => { b.Read (renderablesSsbo, ResourceUsage.StorageReadCompute);
           indirectCmd = b.Write(indirectCmd, ResourceUsage.StorageWriteCompute); },
    (cmd, res, in f) => drawCullPipeline.Record(cmd, f.FrameIndex, f.Camera));

g.AddPass("LightCull", PassType.Compute, QueueClass.Graphics,
    b => { b.Read(lightsSsbo, ResourceUsage.StorageReadCompute);
           tileBuf = b.Write(tileBuf, ResourceUsage.StorageWriteCompute); },
    (cmd, res, in f) => lightCullPipeline.Record(cmd, f.FrameIndex, f.Camera, lc, tx, ty));

// geometry: reads the indirect buffer, writes the g-buffers + depth
g.AddPass("Geometry", PassType.Graphics, QueueClass.Graphics,
    b => { b.Read(indirectCmd, ResourceUsage.IndirectArg);
           pos = b.Write(pos, ResourceUsage.ColorAttachment); /* …+normal/albedo/mat/emissive… */
           depth = b.Write(depth, ResourceUsage.DepthAttachment); },
    (cmd, res, in f) => geometryPipeline.Record(cmd, f, indirectCmdHandle, indirectCount,
        drawCount, new GeometryPipeline.Attachments(res.View(pos), …, res.View(depth))));

// lighting: samples g-buffers + tile buffer, writes HDR
g.AddPass("Lighting", PassType.Graphics, QueueClass.Graphics,
    b => { b.Read(pos, ResourceUsage.SampledFragment); /* …normal/albedo/mat/emissive… */
           b.Read(depth, ResourceUsage.DepthRead);
           b.Read(tileBuf, ResourceUsage.StorageReadCompute);
           hdr = b.Write(hdr, ResourceUsage.ColorAttachment); },
    (cmd, res, in f) => PbrDeferredPipeline.Record(cmd, f, res.View(hdr)));

// skybox: depth-test only (DepthRead — the fix for the writer→writer hack), writes HDR
g.AddPass("Skybox", PassType.Graphics, QueueClass.Graphics,
    b => { hdr = b.Write(hdr, ResourceUsage.ColorAttachment);
           b.Read(depth, ResourceUsage.DepthRead); },
    (cmd, res, in f) => skyboxPipeline.Record(cmd, f, new(res.View(hdr), res.View(depth))));

// transparent: same shape as skybox
// tonemap: reads HDR, writes FinalColor
g.AddPass("Tonemap", PassType.Graphics, QueueClass.Graphics,
    b => { b.Read(hdr, ResourceUsage.SampledFragment);
           final = b.Write(final, ResourceUsage.ColorAttachment); },
    (cmd, res, in f) => tonemapPipeline.Record(cmd, f, res.View(final)));

g.MarkOutput(final);
g.Compile();
```

What this deletes from `DrawDeferred` (`Renderer_Rendering.cs:380–432`): the manual
`drawCullPipeline.Record` + `lightCullPipeline.Record` *ordering reliance* and any barrier
between them and geometry — the graph now serializes cull→geometry (indirect buffer RAW)
and light-cull→lighting (tile buffer RAW) by derivation. The per-frame `Update*` calls
stay (they're CPU SSBO packing, not GPU passes).

Versioning earns its keep here: `HDRColor` is written by Lighting (`@v1`), Skybox (`@v2`),
Transparent (`@v3`), read by Tonemap (`@v3`). A clean linear chain — declarable in any
order, no cycle, no `j>i`. `Depth` is written once (Geometry) and *read* by Lighting/
Skybox/Transparent as `DepthRead`; no more phantom writer→writer serialization.

### 1.9 (Follow-up) Hoist `CmdBeginRendering` into the graph

Once 1.8 is green, move dynamic-rendering begin/end out of the graphics pipelines and into
the graph: the graph already knows each graphics pass's `ColorTargets`/`DepthTarget`
(recorded in `Write`), so it can build `RenderingInfo` + `RenderingAttachmentInfo` and
wrap the pass body. Pass bodies then drop their `CmdBeginRendering`/`EndRendering` and just
bind + draw. Do this *after* parity so a regression here is isolated. Load-op policy
(CLEAR vs LOAD) becomes a per-write flag on the builder (`Write(h, usage, LoadOp.Clear)`),
which is how Skybox/Transparent keep LOAD semantics.

### 1.10 Debug & validation (`GraphDebug.cs`) — ship with Phase 1

- **Validation (compile-time, debug builds):** cycle detection (Kahn leaves nodes →
  cycle), read-before-write (a read of `@v` whose `Producers[v] == -1` and resource isn't
  Imported), write-with-no-consumer (warn → candidate for cull), and resources declared
  but never accessed. Throw on the first three; these catch wiring mistakes the old graph
  failed silently on.
- **DOT export:** `string ToDot()` — nodes = passes, edges labeled `name@version + usage`.
  Invaluable the first time the new topo order surprises you.
- **GPU timing:** one `QueryPool` of timestamps, two writes per pass (`CmdWriteTimestamp2`
  at begin/end), resolved one frame late (read frame N-1 while recording N) to avoid
  stalls. Surface `PassTiming[]` in the existing `StatsPanel` (there's already an ImGui
  panel — `ImGui/Panels/StatsPanel.cs`).

---

## 2. Phase 2 — transfer queue + cross-queue timeline semaphores

Now `PassType`/`QueueClass` stop being decoration. Transfer first (safest multi-queue
win): per-frame staging uploads and mip generation move off the graphics queue.

### 2.1 Finish `QueuePlan.cs`

The stub has private fields and no resolver. Make the fields `public`/`internal` and add a
builder that resolves from `GraphicsDevice`. **Fix the `FindQueueFamilies` trap** here
rather than in the device: the graph needs the *real* family flags, not the current
"first family with ComputeBit" (which is the graphics family). Add to `GraphicsDevice` a
raw accessor for `QueueFamilyProperties[]` (it already fetches them in `FindQueueFamilies`)
and resolve:

```csharp
public static QueuePlan Resolve(GraphicsDevice dev, QueueFamilyProperties[] families)
{
    var plan = new QueuePlan {
        Graphics = new QueueRef(dev.QueueFamilyIndices.graphicsFamily!.Value, 0, dev.GraphicsQueue) };

    // AsyncCompute = COMPUTE without GRAPHICS
    for (uint i = 0; i < families.Length; i++)
        if (Has(families[i], QueueFlags.ComputeBit) && !Has(families[i], QueueFlags.GraphicsBit))
            { plan.AsyncCompute = new QueueRef(i, 0, dev.ComputeQueue); break; }

    // Transfer = TRANSFER without GRAPHICS or COMPUTE (the DMA engine)
    for (uint i = 0; i < families.Length; i++)
        if (Has(families[i], QueueFlags.TransferBit)
            && !Has(families[i], QueueFlags.GraphicsBit) && !Has(families[i], QueueFlags.ComputeBit))
            { plan.Transfer = new QueueRef(i, 0, dev.TransferQueue); break; }

    // Same-handle guard: if a class resolved to graphics' (family,index), null it out.
    if (plan.AsyncCompute is { } ac && ac.Family == plan.Graphics.Family && ac.Index == plan.Graphics.Index)
        plan.AsyncCompute = null;
    if (plan.Transfer is { } tr && tr.Family == plan.Graphics.Family && tr.Index == plan.Graphics.Index)
        plan.Transfer = null;
    return plan;
}
```

> The device's existing `computeQueue`/`transferQueue` may alias the graphics queue
> (see `FindQueueFamilies` falling back to `graphicsFamily`). The same-handle guard is
> what makes "no real async/transfer family" degrade to graphics-queue + barriers instead
> of pretending a second queue exists. **Get this guard right — it's the single trap §3
> of the design doc is built around.**

### 2.2 Per-queue command buffers + scheduling

Compile now emits a command stream **per queue class present**. `FrameRing` (owns the
per-frame command buffers, `CadThingo/VulkanEngine/FrameRing.cs`) grows a second/third
command buffer per frame-in-flight for the transfer/async queues. Passes whose
`QueueClass` resolved to `null` collapse onto Graphics.

### 2.3 Cross-queue sync via timeline semaphores

Timeline semaphores are already enabled. One timeline per queue + a monotonic counter.
For a cross-queue edge, producer signals value `N`, consumer waits `≥ N` (encode in the
per-frame submit's `TimelineSemaphoreSubmitInfo`). The graph records, at compile, which
`(queue, value)` each pass signals and which values its consumers wait on; execute just
submits with those.

### 2.4 Queue-family ownership transfer

For `SharingMode.Exclusive` resources crossing families: emit a **release** barrier on the
source queue (`srcQueueFamilyIndex → dstQueueFamilyIndex`) and a matching **acquire** on
the destination. The graph emits both halves for any cross-family edge. Per-resource
escape hatch: a `Concurrent` flag on `ImageDesc`/`BufferDesc` (no transfers, small perf
cost) — default Exclusive for attachments, Concurrent for small SSBOs read on multiple
queues. The transfer-upload buffers are the natural first Concurrent candidates.

---

## 3. Phase 3 — async compute + subgraph modules (sketch)

- **Async:** `PreferAsync` passes get placed on `QueuePlan.AsyncCompute` *only* inside an
  overlap window (between the signal of their last graphics-queue dependency and the wait
  of their first consumer); never if it forces an immediate graphics-queue wait. Falls
  back to graphics + barriers when `AsyncCompute is null`. First candidates: light-cull,
  any SSAO, denoiser à-trous.
- **Modules:** finish `IGraphModule` — fix the signature to `out TOutputs`:
  ```csharp
  public interface IGraphModule<TInputs, TOutputs>
  { void Build(IGraphBuilder b, in TInputs inputs, out TOutputs outputs); }
  ```
  Internal resources auto-namespaced under `b.Scope`. Validate ports (format/usage/extent)
  at wire time. This is how PT/RT, the host post-stack (tonemap+outline), and even the
  deferred lighting chain become drop-in modules. Folding `DrawPathtraced`/`DrawRayTraced`/
  `RecordSelectionOutline`/`ProcessPickRequest` in here is what finally deletes their
  hand-rolled `CmdPipelineBarrier` blocks and the `_lastRenderMode` tonemap-rebind hack
  (because tonemap becomes one host module reading a stable `SceneColorHDR`/`FinalColor` —
  this is the L3 contract from `renderer-refactor.md`).

This phase composes with L3: each `IRenderCore.Render` builds its technique as a module
appended to the host graph.

---

## 4. Phase 4 — transient aliasing (sketch, highest bug surface — do last)

With lifetimes from 1.5(4) and queue assignment from Phase 2/3, alias transients with
disjoint lifetimes + compatible memory from a graph-owned transient pool. Emit an aliasing
barrier (+ undefined→target transition, contents are garbage) at the first use of the
second resource. **Hazard:** a transient aliased across the graphics/async boundary must
not alias anything live in the overlap window — so aliasing is computed *after* queue
assignment. Surface `AliasedSavedBytes` + an alias-map view in the debug panel.

---

## Cross-cutting gotchas specific to this engine

1. **Transients vs. frames-in-flight.** Today's `RenderGraph` allocates one set of
   g-buffers at `Compile` and reuses them every frame. With `MAX_CONCURRENT_FRAMES > 1`
   that's a latent cross-frame WAR hazard (frame N+1's geometry writes can race frame N's
   lighting reads; only the per-frame fence gates *CPU* command-buffer reuse, not GPU
   overlap). **Decide explicitly:** either allocate transients per-frame-in-flight
   (simplest, more VRAM) or have the graph emit a cross-frame acquire barrier on
   first-use. Imported per-frame SSBOs already dodge this because `GpuScene` double-buffers
   them. Pick per-frame-in-flight for transients in Phase 1 to match the SSBO cadence.
2. **`FinalColor` layout dance stays the host's job.** The host's step-7
   (`Renderer_Rendering.cs:268–346`) transitions FinalColor through `TransferSrcOptimal`
   for the blit and back to `ShaderReadOnlyOptimal`. Keep FinalColor *imported* and leave
   it in `ShaderReadOnlyOptimal` at graph end so the host's existing dance and the ImGui
   viewport sampler both stay valid. Don't pull the blit into the graph in Phase 1.
3. **Depth is read-not-written by Skybox/Transparent.** Declaring those reads as
   `DepthRead` (not a second `DepthAttachment` write) is the correct-by-construction
   replacement for the current writer→writer-edge hack. Verify the pipelines are created
   with depth-write disabled (they are: EQUAL/LE, no write).
4. **Pick + selection are out-of-band today.** `ProcessPickRequest` uses a single-time
   submit with `QueueWaitIdle`; `RecordSelectionOutline` runs after the mode dispatch.
   Leave both outside the graph until Phase 3 modules — they're editor concerns and the
   pick read-back is synchronous by design.
5. **Reuse Kahn's sort + the depth-format helper.** Don't rewrite the topo sort
   (`RenderGraph.Compile:161–180` is fine) or re-copy the depth-format switch a fourth
   time.
6. **`Globals.vk` is process-wide; never dispose it in the graph.** The graph owns only
   its transients + query pools + (Phase 2) its timelines/aux command buffers.

## Per-phase verification

| After | Check |
|---|---|
| 1.8 | Deferred output pixel-identical to pre-migration; validation layers clean; `ToDot()` shows Lighting→Skybox→Transparent→Tonemap as an HDRColor version chain |
| 1.9 | Still pixel-identical with begin/end-rendering hoisted; RenderDoc shows one render region per graphics pass |
| 1.10 | StatsPanel shows per-pass GPU ms; intentional cycle/read-before-write throws at compile |
| 2.x | On a discrete GPU with a DMA queue: RenderDoc/Nsight shows uploads on the transfer queue; on an iGPU (single family) everything collapses to graphics with no validation errors (same-handle guard) |
| 3.x | Removing a module call drops its passes (dead-cull); PT/RT no longer carry manual barriers; `_lastRenderMode` rebind deleted |
| 4.x | `AliasedSavedBytes > 0`; no corruption with aliasing force-toggled in the panel |

## Recommended order of attack

1.1 → 1.2 → 1.3/1.4 (types compile, no behavior) → 1.5/1.6/1.7 (graph compiles+executes an
empty/trivial graph) → **1.8 (deferred parity — the milestone)** → 1.10 (debug) → 1.9
(hoist rendering) → 2.1 → 2.2 → 2.3/2.4. Stop after Phase 2 if time-boxed (the design doc's
recommended stopping point): hand-rolled barriers gone, compute/RT/transfer unified, the
transfer-upload win banked, debug tooling shipped — without the async scheduler or aliasing
where the subtle bugs live.