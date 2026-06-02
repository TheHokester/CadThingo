# Render Graph — Parallel Track

A standalone redesign of `RenderGraph` into a real, multi-queue frame graph. This is
a **parallel track** to the L1→L3 ownership refactor (`renderer-refactor.md`): it does
not depend on those stages and they do not depend on it, but they compose cleanly —
the graph consumes the `GraphicsDevice` (L1) for queues/allocator/sync, imports
`GpuScene` resources (L2), and each render core (L3) builds its own (sub)graph.

> **Scope of ambition:** versioned virtual resources (images **and** buffers),
> usage-derived synchronization (barriers **and** cross-queue semaphores),
> Graphics / Compute / RayTrace / Transfer passes on their respective queues with
> async overlap, composable **subgraph modules** (append-and-wire features, e.g. a
> drop-in SVGF denoiser), transient memory aliasing, and first-class debug metrics.

## Contents

1. [Where we are today](#1-where-we-are-today)
2. [Design goals & principles](#2-design-goals--principles)
3. [Queue & family model](#3-queue--family-model)
4. [Resource model (virtual, versioned)](#4-resource-model-virtual-versioned)
5. [Pass model](#5-pass-model)
6. [Compilation pipeline](#6-compilation-pipeline)
7. [Synchronization](#7-synchronization)
8. [Async scheduling](#8-async-scheduling)
9. [Subgraph modules (features)](#9-subgraph-modules-features)
10. [Debug & metrics](#10-debug--metrics)
11. [Public API sketch](#11-public-api-sketch)
12. [Integration with L1/L2/L3](#12-integration-with-l1l2l3)
13. [Phasing](#13-phasing)

---

## 1. Where we are today

`RenderGraph` (in `Renderer_Rendering.cs`):

- **Images only.** `AddResource` takes an `ImageResource`; buffers can't participate.
- **Graphics passes only.** `Execute` hardcodes "inputs → `ShaderReadOnlyOptimal`,
  outputs → `ColorAttachment`/`DepthStencilAttachment`". Compute/RT/transfer can't use it,
  so cull, light-cull, PT, RT, probe capture, selection, and pick are hand-recorded
  outside it with manual barriers.
- **Coarse sync.** Every barrier is `AllCommandsBit` + `MemoryReadBit` — correct but a
  blunt instrument; no per-resource stage/access derivation.
- **Ordering is really declaration order.** `Compile` restricts reader matching to
  `j > i` and adds writer→writer edges in declaration order, specifically to dodge the
  false cycles you get when one resource (HDRColor, Depth) is written by several passes.
  That's a versioning problem wearing a topological-sort costume.
- **Single queue, no aliasing, no metrics.**

Everything below is the upgrade path.

## 2. Design goals & principles

- **Declarative, data-driven.** A pass declares *what* it touches and *how* (usage);
  the graph derives *all* synchronization. Hand-written barriers disappear.
- **Correct-by-construction sync.** Barriers and semaphores come from the dependency
  analysis, not from a human tracking layouts in comments.
- **Multi-queue with graceful degradation.** Use async-compute / transfer queues when
  the device exposes independent families; transparently collapse to the graphics queue
  + barriers when it doesn't.
- **Composability.** Features are subgraph modules with typed ports — append and wire,
  don't copy-paste passes.
- **Observability is a feature, not an afterthought.** Per-pass GPU timing, DOT export,
  barrier/semaphore logs, and a validation pass ship from Phase 1.
- **Non-goals (deliberate):** a fully *automatic* async scheduler (we use hints),
  sub-pass merging, and render-pass/tile-memory optimization (you use dynamic rendering;
  not relevant). Aliasing is opt-in and late (Phase 4) because it's where bugs live.

## 3. Queue & family model

(See the chat answer: capability ≠ a distinct queue. This section formalizes it.)

The graph works in **queue classes**, not raw families:

```csharp
enum QueueClass { Graphics, AsyncCompute, Transfer }
// RayTrace passes run on the Graphics-family queue (ray tracing is a graphics-family
// capability); they are a PassType, not a QueueClass.
```

At construction the graph queries the device once and builds a **queue plan**:

```csharp
struct QueuePlan
{
    QueueRef Graphics;                 // universal family, always present
    QueueRef? AsyncCompute;            // distinct COMPUTE family WITHOUT graphics bit, else null
    QueueRef? Transfer;                // distinct TRANSFER-only family, else null
    bool HasRealAsyncCompute => AsyncCompute is not null;
    bool HasDedicatedTransfer => Transfer is not null;
}
struct QueueRef { uint Family; uint Index; Queue Handle; }
```

Resolution rules:

1. **Graphics** = the universal family (graphics + compute + transfer).
2. **AsyncCompute** = a family advertising `COMPUTE` but **not** `GRAPHICS`. If none,
   `null` → async passes fall back to Graphics.
3. **Transfer** = a family advertising `TRANSFER` but neither `GRAPHICS` nor `COMPUTE`
   (the DMA engine). If none, `null` → transfer passes fall back to Graphics.
4. **Same-handle guard:** if two classes would resolve to the *same* `(family, index)`,
   they are merged into one logical queue. This is exactly the trap in the current
   `FindQueueFamilies` (compute family == graphics family, both index 0 ⇒ identical
   `VkQueue`). The graph must treat that as a single queue, not pretend it's async.

Consequence for sync: **same queue ⇒ pipeline barrier; different queue ⇒ timeline
semaphore (+ queue-family ownership transfer for `EXCLUSIVE` resources).** The graph
picks automatically based on the queue plan.

## 4. Resource model (virtual, versioned)

Resources are **virtual** during graph building — no GPU memory until compile. Two
kinds, unified under a handle:

```csharp
readonly struct GraphImage  { int ResourceId; int Version; }
readonly struct GraphBuffer { int ResourceId; int Version; }

struct ImageDesc  { Format Format; Extent2D Extent; uint Mips, Layers;
                    ImageUsageFlags Usage; SampleCountFlags Samples; }
struct BufferDesc { ulong Size; BufferUsageFlags Usage; }

enum ResidencyKind { Transient,   // graph-owned, lifetime-scoped, aliasable
                     Imported }   // external (swapchain, GpuScene SSBOs, IBL) — never aliased
```

### Versioning (the fix for the `j > i` hack)

Every **write produces a new version** of the resource (SSA form). Reads capture the
*current* version. This makes RAW/WAR/WAW dependencies unambiguous and **eliminates the
false cycles** that forced the declaration-order restriction:

```
LightingPass   writes HDRColor      → HDRColor@v1   (producer: Lighting)
SkyboxPass     reads  HDRColor@v1,
               writes HDRColor      → HDRColor@v2   (producer: Skybox, depends on Lighting)
TransparentPass reads HDRColor@v2,
               writes HDRColor      → HDRColor@v3   (producer: Transparent, depends on Skybox)
TonemapPass    reads  HDRColor@v3                   (depends on Transparent)
```

A linear chain of versions, no cycle, declarable in any order. Edges fall out of
"reader of @vN depends on producer of @vN."

### Usage = the single source of truth for sync

A read/write is tagged with a `ResourceUsage`, which the graph maps to
`(VkPipelineStage2, VkAccessFlags2, VkImageLayout)`:

| `ResourceUsage` | Stage2 | Access2 | Layout |
|---|---|---|---|
| `ColorAttachment` | `COLOR_ATTACHMENT_OUTPUT` | `COLOR_ATTACHMENT_WRITE` | `COLOR_ATTACHMENT_OPTIMAL` |
| `DepthAttachment` | `EARLY/LATE_FRAGMENT_TESTS` | `DEPTH_STENCIL_ATTACHMENT_*` | `DEPTH_STENCIL_ATTACHMENT_OPTIMAL` |
| `SampledFragment` | `FRAGMENT_SHADER` | `SHADER_SAMPLED_READ` | `SHADER_READ_ONLY_OPTIMAL` |
| `StorageReadCompute` | `COMPUTE_SHADER` | `SHADER_STORAGE_READ` | `GENERAL` |
| `StorageWriteCompute` | `COMPUTE_SHADER` | `SHADER_STORAGE_WRITE` | `GENERAL` |
| `IndirectArg` | `DRAW_INDIRECT` | `INDIRECT_COMMAND_READ` | n/a (buffer) |
| `TransferSrc/Dst` | `ALL_TRANSFER` | `TRANSFER_READ/WRITE` | `TRANSFER_SRC/DST_OPTIMAL` |
| `StorageRT` | `RAY_TRACING_SHADER` | `SHADER_STORAGE_*` | `GENERAL` |
| `AccelStructBuild` | `ACCELERATION_STRUCTURE_BUILD` | `ACCEL_STRUCT_WRITE` | n/a |
| `AccelStructRead` | `RAY_TRACING_SHADER`/`COMPUTE` | `ACCEL_STRUCT_READ` | n/a |

This table is the whole sync engine. Uses `Synchronization2` (already enabled via
`vulkan13Features.Synchronization2`) so barriers are per-resource `VkImageMemoryBarrier2`
inside a `VkDependencyInfo`, not the global `AllCommands` sledgehammer.

## 5. Pass model

```csharp
enum PassType { Graphics, Compute, RayTrace, Transfer }

// Two-phase: Setup declares dependencies (no GPU work); Execute records commands.
delegate void PassSetup(IGraphBuilder b);
delegate void PassExecute(CommandBuffer cmd, in PassResources res, in RenderView view);

struct PassDesc
{
    string     Name;
    PassType   Type;
    QueueClass Queue;        // requested; may be downgraded by the queue plan
    bool       PreferAsync;  // hint: scheduler may move to AsyncCompute if profitable
    PassSetup  Setup;
    PassExecute Execute;
}
```

- **Graphics** passes auto-`CmdBeginRendering`/`EndRendering` from their declared color/
  depth writes (the graph already knows them — hoist attachment setup out of pipelines).
- **RayTrace** passes bind the SBT and `CmdTraceRays`; scheduled on the graphics queue.
- **Compute/Transfer** passes can request `AsyncCompute`/`Transfer`.
- `PassResources` hands the execute callback the *physical* handles (resolved
  `ImageView`/`Buffer`) for everything it declared, so the body never touches the
  graph's internals.

## 6. Compilation pipeline

`Compile()` runs once per graph topology change (not per frame):

1. **Build the DAG** from versioned reads/writes (RAW/WAR/WAW edges). Cross-queue edges
   are flagged for semaphore generation.
2. **Cull dead passes.** Mark graph outputs (swapchain image, `SceneColorHDR`, anything
   `Imported` + written, or passes flagged `HasSideEffects`). Reverse-reachability from
   outputs; drop unreachable passes and their transient resources. (Lets features be
   registered unconditionally and compiled out when unused.)
3. **Schedule.** Topological sort that respects queue classes; emit a per-queue ordered
   command stream. Async passes placed to maximize overlap windows (§8).
4. **Compute resource lifetimes.** First/last use (in schedule order) per transient.
5. **Allocate / alias.** Transients allocated from a graph-owned transient pool via
   `GpuMemoryAllocator`; disjoint-lifetime + compatible resources may alias (Phase 4).
   Imported resources are never allocated/aliased.
6. **Generate sync.** Walk each queue stream; emit barriers (intra-queue) and
   semaphore wait/signal + ownership transfers (cross-queue) from the usage table (§7).
7. **Bake.** Cache the compiled plan: ordered passes per queue, barrier batches,
   semaphore graph, attachment infos, physical resource map.

`Execute(view)` per frame just replays the baked plan: for each queue stream, record
barriers + pass bodies into that queue's command buffer, then submit with the computed
semaphore dependencies. No per-frame analysis.

## 7. Synchronization

### Intra-queue (barriers)

For consecutive accesses to the same resource on the same queue, emit one
`VkImageMemoryBarrier2`/`VkBufferMemoryBarrier2`:
- `srcStage/srcAccess` = previous access's mapped stage/access,
- `dstStage/dstAccess` = next access's,
- `oldLayout/newLayout` from the usage table (images).

Barriers are **batched** per `VkDependencyInfo` at pass boundaries (one
`CmdPipelineBarrier2` covering all of a pass's input transitions), replacing the current
per-resource calls.

### Cross-queue (timeline semaphores)

When a dependency crosses queues, use **timeline semaphores** (already enabled via
`vulkan12Features.TimelineSemaphore`): producer signals value `N`, consumer waits `≥ N`.
One timeline per queue + a monotonic counter is enough; the graph records the
`(queue, value)` each pass signals and which values its consumers wait on.

### Queue-family ownership transfer

Resources with `SharingMode.Exclusive` that move between families need a **release
barrier on the source queue** (`srcQueueFamilyIndex → dstQueueFamilyIndex`) and a
matching **acquire barrier on the destination queue**. The graph emits both halves
automatically for any cross-family edge. Alternative: declare such resources
`Concurrent` (no transfers, simpler, small perf cost) — make it a per-resource flag,
default `Exclusive` for attachments, `Concurrent` for small SSBOs read everywhere.

### Aliasing barriers (Phase 4)

When two transients share memory, the first use of the second resource needs an
aliasing barrier (and an undefined→target layout transition, since the contents are
garbage). The graph inserts these from the alias map.

## 8. Async scheduling

Fully automatic async is fragile; the graph uses **hints + a deterministic scheduler**:

- A pass sets `PreferAsync` (e.g. light-cull, SSAO, a denoiser's à-trous passes).
- During scheduling, if `QueuePlan.HasRealAsyncCompute`, the scheduler tries to place
  the pass on the async queue inside an **overlap window** — between the signal of its
  last dependency on the graphics queue and the wait of its first consumer.
- It will **not** move a pass async if doing so would force the graphics queue to
  immediately wait on it (no overlap → pure overhead).
- **Fallback:** no async family (or same-handle merge from §3) ⇒ the pass stays on the
  graphics queue and cross-queue semaphores degrade to ordinary barriers. Same graph,
  same correctness, no code change in the pass.

Transfer queue: per-frame uploads (`GpuScene` staging copies, mip generation) are the
safest async win and should be the *first* multi-queue work enabled (Phase 2), well
before compute overlap.

Hazard the scheduler must guard: a transient aliased across the graphics/async boundary
cannot be aliased with anything live in the overlap window. Aliasing is therefore
computed **after** queue assignment.

## 9. Subgraph modules (features)

The "append-and-wire" requirement: a feature is a **module** that contributes a set of
passes + its own internal resources, exposing typed **ports**. Modules compose and
nest. The parent wires a module's input ports to existing handles and consumes its
output ports.

```csharp
interface IGraphModule
{
    // Registers this module's passes into `b`, reading `inputs`, returning outputs.
    // Internal resources are auto-namespaced under `b.Scope` (e.g. "svgf/variance").
    void Build(IGraphBuilder b, in TInputs inputs, out TOutputs outputs);
}
```

Ports are validated at wire time (format/usage/extent compatibility), so a mismatch is
a compile error, not a runtime corruption.

**Worked example — a drop-in SVGF denoiser** (illustrative; not to be implemented now):

```
Module: SvgfDenoiser
  Inputs : Color(noisy, RGBA16F), Albedo, Normal, Depth, MotionVectors, HistoryLength
  Output : Denoised (RGBA16F)
  Internal passes (all Compute, several PreferAsync):
    1. Reproject      (reads History + Motion → Moments, HistoryLength)   [Compute]
    2. VarianceEst    (reads Moments → Variance)                          [Compute]
    3. ATrous ×5      (ping-pong Color/Variance, edge-stopping)           [Compute, async]
    4. Modulate       (Color × Albedo → Denoised)                         [Compute]
  Internal resources: svgf/moments, svgf/variance, svgf/pingA, svgf/pingB, svgf/history
  History buffers are Imported (persist across frames; ping-pong owned by the module).
```

Wiring it into a path-traced graph is then:

```csharp
var ptColor = ptCore.Build(b);                 // raygen → noisy color handle
svgf.Build(b, new SvgfInputs(ptColor, gAlbedo, gNormal, depth, motion, histLen),
              out var svgfOut);
tonemap.Build(b, new TonemapInputs(svgfOut.Denoised), out var ldr);  // SceneColorHDR → FinalColor
```

Remove the denoiser by not calling `svgf.Build`; dead-pass culling (§6.2) drops its
resources automatically. This is how host post-stack (tonemap + outline) and even the
whole deferred lighting chain become modules too.

## 10. Debug & metrics

Shipped from Phase 1, surfaced in an ImGui "Render Graph" panel:

**GPU timing.** A per-queue timestamp query pool; the graph writes a timestamp at each
pass boundary and resolves results **one frame late** (read frame N-1's queries while
recording N) to avoid stalls. Reports per-pass GPU ms, per-queue totals, and whole-frame.

**Pipeline statistics (optional).** Per-pass `VkQueryPool` of pipeline-statistics
(VS invocations, primitives, FS invocations, compute invocations) behind a toggle.

**Debug labels.** `CmdBeginDebugUtilsLabelEXT` per pass and per module (nested), using
the `ExtDebugUtils` already loaded — passes and modules show up as named, colored,
hierarchical regions in RenderDoc / Nsight.

**DOT export.** Dump the compiled DAG to Graphviz: nodes = passes (colored by queue),
edges = resource dependencies (labeled `resource@version` + usage), dashed edges =
cross-queue semaphore syncs, boxes = module scopes. One method, invaluable for "why did
these reorder."

**Barrier / semaphore log.** Optional dump of every emitted barrier (`src/dst stage`,
`access`, layout transition, resource) and semaphore (`signal queue@value` →
`wait queue`), each annotated with the producing/consuming pass — so sync is auditable.

**Validation pass** (run at compile, in debug): cycle detection, read-before-write
(a read whose version has no producer), write-with-no-consumer (dead → warn), port
format/usage mismatches, aliasing overlap checks, and resources declared but never used.

**Live overlay data model:**

```csharp
struct PassTiming  { string Name; QueueClass Queue; double GpuMs; double CpuRecordMs; }
struct GraphStats  { double FrameGpuMs;
                     PassTiming[] Passes;          // hierarchical by module scope
                     int BarrierCount, SemaphoreCount, CulledPassCount;
                     long TransientBytes, AliasedSavedBytes;
                     double CompileMs; }            // last (re)compile cost
```

Panel features: per-pass bar chart, per-queue occupancy timeline (shows async overlap),
transient memory map + aliasing savings, and toggles to force-disable async / aliasing /
individual modules for A/B debugging.

## 11. Public API sketch

```csharp
var graph = new RenderGraph(device);             // device = GraphicsDevice (L1)

// Import externals (never aliased; graph tracks their state):
var swap   = graph.ImportImage(swapchainImage, ResourceState.Undefined, "swapchain");
var sceneSet = graph.ImportBuffer(gpuScene.Renderables, ...);   // L2

// Declare a compute pass that may run async:
graph.AddPass("LightCull", PassType.Compute, QueueClass.AsyncCompute, preferAsync: true,
    setup: b => {
        b.Read (sceneSet,           ResourceUsage.StorageReadCompute);
        tileBuf = b.Write(tileBuf,  ResourceUsage.StorageWriteCompute);
    },
    execute: (cmd, res, view) => lightCull.Record(cmd, res, view));

// Declare a graphics pass; attachments inferred from writes:
var hdr = graph.CreateImage(hdrDesc, "HDRColor");
graph.AddPass("Lighting", PassType.Graphics, QueueClass.Graphics, preferAsync: false,
    setup: b => {
        b.Read(gbufPosition, ResourceUsage.SampledFragment);
        b.Read(tileBuf,      ResourceUsage.StorageReadCompute);
        hdr = b.Write(hdr,   ResourceUsage.ColorAttachment);   // → new version
    },
    execute: (cmd, res, view) => pbr.Record(cmd, res, view));

graph.MarkOutput(swap);
graph.Compile();        // DAG, cull, schedule, alloc/alias, sync — cached
// per frame:
graph.Execute(in renderView);   // replays baked per-queue streams + submits
```

## 12. Integration with L1/L2/L3

- **L1 (`GraphicsDevice`):** the graph's only device dependency — queues (via the
  queue plan), `GpuMemoryAllocator` (transient pool), `Synchronization2`/timeline
  semaphores, debug-utils labels.
- **L2 (`GpuScene`):** scene SSBOs, TLAS, IBL, and the swapchain image are **imported**
  resources. The cached transform/extract work feeds `RenderView`, passed to `Execute`.
- **L3 (cores):** each `IRenderCore.Render` builds its technique as a graph (or a module
  appended to a host graph). The host post-stack (tonemap → outline → present) is itself
  a module. `SceneColorHDR` is the core's graph output, imported by the host module —
  which is exactly the L3 contract, now enforced by port wiring instead of convention.

This also retires the manual barrier code in `DrawDeferred`/`DrawPathtraced`/
`DrawRayTraced`/`RecordSelectionOutline`: those become passes whose sync is derived.

## 13. Phasing

| Phase | Delivers | Notes |
|---|---|---|
| **0** | (today) image-only, single queue, coarse barriers | baseline |
| **1** | versioned resources (kill `j>i`), **buffers + images**, usage-derived `Sync2` barriers, single queue, **debug: timings + DOT + validation** | the high-value core; migrate deferred fully incl. cull/light-cull as compute passes |
| **2** | `PassType` + **dedicated transfer queue** (uploads/mips), timeline-semaphore cross-queue sync, ownership transfers | transfer first — safest multi-queue win |
| **3** | **async compute** (hints + scheduler), **subgraph modules** | PT/RT denoiser-style features become drop-ins |
| **4** | transient **aliasing**, debug overlay polish (occupancy timeline, alias map) | aliasing last — highest bug surface |

> Recommended stopping point if time-boxed: **Phase 1 + Phase 2**. That removes the
> hand-rolled barriers, unifies compute/RT/transfer into one scheduler, gives you the
> transfer-queue upload win, and ships the debug tooling — without taking on the async
> scheduler or aliasing, which are where the subtle bugs live.