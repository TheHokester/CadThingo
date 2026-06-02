# Renderer Refactor Plan — L1 → L2 → L3

A staged refactor of the Vulkan renderer toward three goals: **simpler**, **more
extensible**, and **clear separation of concerns**. It attacks the three root pains
in a deliberate bottom-up order so each stage compiles, ships, and is testable
before the next begins:

| Stage | Name | Pain it kills | Risk |
|---|---|---|---|
| **L1** | `GraphicsDevice` decomposition | the `Renderer` god object | low (mechanical) |
| **L2** | `GpuScene` + Extract phase | the scene↔GPU tangle | medium |
| **L3** | Render-core model | the 4-mode `DrawX` switch + manual wiring | medium |

> **Ordering rationale:** L1 produces the device-services surface that both L2 and
> L3 depend on. L2 produces the `RenderView` that L3's cores consume. Doing them
> bottom-up means each stage makes the next one *smaller*, and you can bisect any
> regression to a single stage.

---

## Current state (what we're starting from)

- **`Renderer` is a god object** — one partial class (`Renderer_Core`,
  `_Rendering`, `_Resources`, `_Ibl`, `_Ray_Query`, `_Compute`, `_Utils`) owning the
  instance/device/queues/allocator/descriptor-pool/command-pool, the swapchain, the
  command-buffer ring + sync, every render target, **15 pipeline objects**, IBL, the
  acceleration structures, *and* the per-frame scene→GPU packing.
- **The render graph is half-used.** Only the deferred *graphics* chain
  (`Geometry → Lighting → Skybox → Transparent → Tonemap`) goes through
  `RenderGraph`. Cull/light-cull compute, the PT compute path, the RT path, probe
  capture, selection mask, and pick are all hand-recorded with manual barriers.
- **Scene→GPU is duplicated across ≥4 walkers.** `DrawCullPipeline.Record`,
  `UpdateLights`, `UpdateMaterials`, and `RebuildTlas` each independently walk
  `Scene._entityList`, each call `GetComponent` (64-slot scans), each recompute world
  matrices. Material classification lives in two places. Entity identity is the
  *flat list index*, reused as TLAS `InstanceCustomIndex`, pick result, and selection
  index — an implicit contract that breaks on reorder/removal.
- **Descriptor wiring is manual and order-dependent.** `Initialize()` is a long
  sequence of `WriteXDescriptor` calls; `RebuildPbrPipelines` / `RebuildRenderTargets`
  must re-issue ~10 of them by hand.

---

# L1 — Decompose the god object by ownership

**Goal:** peel *infrastructure* off `Renderer` into four lifetime-scoped owners,
leaving `Renderer` as a thin orchestrator. Do **not** touch technique code
(pipelines, graph, modes, IBL, ray query, scene packing) in this stage.

## Target components

```
GraphicsDevice   (app lifetime)      ← the RHI context everything depends on
   ▲
Swapchain        (resize lifetime)
   ▲
RenderTargets    (resize lifetime)
   ▲
FrameRing        (app lifetime, cycles per frame)
   ▲
Renderer         (orchestrator: holds the above + pipelines + mode dispatch)
```

Dependencies point **downward only**.

### `GraphicsDevice` — essentially all of `Renderer_Core.cs` + device-level helpers

| Owns (fields) | Owns (methods) |
|---|---|
| `vk`, `instance`, `debugUtils`, `debugMessenger`, `surface`, `khrSurface`, `physicalDevice`, `device`, `deviceExtensions`, the 4 queues, `queueFamilyIndices`, `memAllocator`, `descriptorPool`, `commandPool` | `CreateInstance`, `SetupDebugMessenger`, `CreateSurface`, `PickPhysicalDevice`, `AddSupportedOptionalExtensions`, `CreateLogicalDevice`, `CheckValidationLayerSupport`, `CheckDeviceExtensionSupport`, `IsDeviceSuitable`, `FindQueueFamilies`, `CreateDescriptorPool`, `CreateCommandPool`, `GetVkInstance` |
| **Capability flags:** `descriptorIndexEnabled`, `accelerationStructureEnabled`, `rayQueryEnabled`, `rayTracePipelineEnabled`, `serEnabled`, `multiviewEnabled` + derived `RayShadowsSupported`, `RayTracePipelineSupported`, `SerSupported` | `FindMemoryType`, `FindDepthFormat`, `FindSupportedFormat`, `CreateBuffer`, `CopyBuffer`, `UploadBufferData`, `DestroyBuffer`, `CreateMappedUniformBuffer`, `CreateMappedStorageBuffer`, `TransitionImageLayout`, `GenerateMipMaps`, `BeginSingleTimeCommands`, `EndSingleTimeCommands`, `CreateShaderModule`, `WriteBindlessTexture` |

### `Swapchain` — depends on `GraphicsDevice` + surface

- **Fields:** `swapChainKhr`, `swapChain`, `swapChainImages`, `swapChainImageFormat`,
  `swapChainExtent`, `swapChainImageViews`, `swapChainImageLayouts`.
- **Methods:** `CreateSwapChain`, `CreateImageViews`, `CleanupSwapChain`,
  `RecreateSwapChain`, `ChooseSwapExtent`, `QuerySwapChainSupport`,
  `ChooseSwapSurfaceFormat`, `ChooseSwapPresentMode`, `SetupDynamicRendering`.

### `RenderTargets` — depends on `GraphicsDevice`

- **Fields:** `renderExtent`, `depthImageResource`, the 5 g-buffers
  (`gBufferPosition/Normal/Albedo/Material/Emissive`), `gBufferSampler`,
  `ptAccumulator`, `ptOutColor`, `selectionMask`.
- **Methods:** `CreateDepthResources`, `CreateGBufferResources`,
  `CreateGBufferSampler`, `CreatePathTracingResources`, `CreateSelectionResources`.
- **Note:** `RebuildRenderTargets` / `ResizeRenderTargets` stay on the orchestrator
  (they re-issue cross-pipeline descriptor writes) but delegate the realloc here.
  L2 later moves the transient targets into the graph; this owner is the interim home.

### `FrameRing` — depends on `GraphicsDevice` + `Swapchain`

- **Fields:** `commandBuffers`, `imageAvailableSemaphores`, `renderFinishedSemaphores`,
  `inFlightFences`, `currentFrame`, `frameCounter`, `uploadsTimeline`,
  `lastTimelineValue`, `MAX_CONCURRENT_FRAMES`.
- **Methods:** `CreateCommandBuffers`, `CreateSyncObjects`, and the
  **acquire → begin → submit → present skeleton** of `DrawFrame` (steps 1–4, 7e–10).
  The frame *body* (mode switch, blit, outline) stays on the orchestrator and is
  handed `cmd` + `imageIndex`.

### `Renderer` (what remains after L1)

References to the four owners + the 15 pipeline fields + `renderMode`/`_lastRenderMode`
+ `DrawDeferred/DrawRayQueried/DrawPathtraced/DrawRayTraced` + `ProcessPickRequest` +
`RecordSelectionOutline` + `SetupDeferredRenderer` + the IBL & ray-query partials
(untouched) + the `Update*` scene packing (untouched — that's L2). `Initialize()`
becomes a **composition root**.

## Data model (L1 introduces no GPU structs — only a handle bundle)

```csharp
// Passed to every pipeline/core instead of the whole Renderer.
// This is the narrow "device services" surface.
sealed class GraphicsDevice
{
    Vk             Vk            { get; }
    Device         Device        { get; }
    PhysicalDevice PhysicalDevice{ get; }
    GpuMemoryAllocator Allocator { get; }
    DescriptorPool DescriptorPool{ get; }
    Queue          GraphicsQueue { get; }

    bool RayShadowsSupported       { get; }
    bool RayTracePipelineSupported { get; }
    bool SerSupported              { get; }
    bool DescriptorIndexingEnabled { get; }

    // helpers: CreateBuffer / CreateShaderModule / BeginSingleTimeCommands / ...
}
```

## What changes

1. **`PipelineBase.Renderer` → `PipelineBase.Device` (`GraphicsDevice`).** Every
   pipeline currently holds `protected readonly Renderer Renderer` and calls
   `Renderer.vk/.device/.physicalDevice/.CreateShaderModule/.CreateMappedStorageBuffer/
   .BeginSingleTimeCommands/.GetVkInstance()` + capability flags. **Repoint all of it
   to `GraphicsDevice`.** This is the edit that turns the hidden god-object coupling
   into a real, narrow dependency. The residual non-device needs are already passed at
   record time (`GeometryPipeline.Record(Attachments)`, `PbrDeferredPipeline.Record(hdrView)`);
   the few direct grabs (`DrawCullPipeline` reading `renderExtent`) become parameters.

2. **Teardown order is centralized, not scattered.** Today `Cleanup()` encodes
   hard-won ordering (AS handles before `ResourceManager.Dispose`; allocator dead-last
   because `Vk*Destroy` releases handles but not memory). Keep an explicit ordered
   teardown in the orchestrator: `pipelines → ray-query → IBL → RenderTargets →
   Swapchain → FrameRing → GraphicsDevice (strictly last)`. Each owner's `Dispose`
   frees only what it owns; the orchestrator owns the *order*.

3. **`Initialize()` becomes a composition root** — `new GraphicsDevice()` →
   `new Swapchain(device)` → `new RenderTargets(device)` → `new FrameRing(device, swapchain)`
   → construct + wire pipelines.

## Migration steps (compiles between each)

1. Extract `GraphicsDevice` (biggest win, unblocks the pipeline repoint).
2. Repoint `PipelineBase` to `GraphicsDevice`; convert stray `renderExtent` reads to params.
3. Extract `Swapchain`.
4. Extract `FrameRing`.
5. Extract `RenderTargets`.
6. Centralize teardown order in the orchestrator.

> **Scope guard:** do *not* fold IBL, ray-query, or `Update*` into L1 even though
> they share the partial. They're technique/data concerns (L2/L3). L1 is only the
> four infra owners + the pipeline repoint.

---

# L2 — `GpuScene`: one Extract phase, one canonical GPU mirror

**Goal:** make a single **Extract** step the *only* thing that reads the ECS for
rendering, producing a `GpuScene` (canonical GPU-resident mirror) and a per-frame
`RenderView` snapshot. Nothing downstream touches `Entity*` again.

## The pain being removed

The same `Scene._entityList` is walked independently, per frame, by:
`DrawCullPipeline.Record` (packs `RenderableInputGpu`, classifies opaque/blend, sorts
transparents), `UpdateLights` (packs `PbrLightGpu`, recomputes owner world pos),
`UpdateMaterials` (copies `Scene.Materials`), and `RebuildTlas` (packs instances +
`ShadowEntityInfo`). World matrices are computed 2–3× per entity per frame; identity
is the fragile flat list index.

## Data models

### Stable identity (replaces the flat-index contract)

```csharp
// Generational handle — generalizes Scene's existing material free-list idea.
// Stable for the entity's lifetime; survives list reorder/removal.
readonly struct RenderableHandle { uint Index; uint Generation; }
```

`GpuScene` owns a slot allocator handing these out at register time. TLAS
`InstanceCustomIndex`, pick results, `ShadowEntityInfo.EntityIndex`, and the selection
index all resolve through the handle — **not** `Scene.IndexOf`.

### `GpuScene` — owns the canonical buffers (these GPU structs already exist)

```csharp
// Existing, reused verbatim — GpuScene becomes their single owner:

struct RenderableInputGpu          // cull input (96B)
{ Matrix4x4 model; Vector4 sphereLocal; uint indexCount, firstIndex, materialIndex, _pad; }

struct PbrLightGpu                 // packed light (64B)
{ Vector4 positionRange, colorIntensity, directionType, spotCones; }

struct PbrMaterial                 // bindless material (BaseColor/Metallic/.../KHR ext factors + tex indices)
{ Vector4 BaseColorFactor; Vector3 EmissiveFactor; float AlphaCutoff, MetallicFactor, RoughnessFactor;
  uint Flags; int BaseColorTex, PhysicalDescriptorTex, NormalTex, OcclusionTex, EmissiveTex;
  float TransmissionFactor, Ior, ClearcoatFactor, ClearcoatRoughnessFactor; /* + clearcoat tex... */ }

struct ShadowEntityInfo            // per-cluster instance → entity + xform (ray query)
{ uint IndexOffset, MaterialIndex, Flags, EntityIndex; Vector4 Xform0, Xform1, Xform2; }

struct EmissiveTriGpu              // area-light triangle (80B)
{ Vector4 P0Area, E1LeR, E2LeG, NLeB; int IndexOffset, PrimIndex, EmissiveTex, _pad; }
```

```csharp
sealed class GpuScene  // owns the buffers for the renderer's lifetime
{
    // Retained, slot-addressed, double-buffered where written per frame:
    Buffer Transforms;       // cached world matrices (computed ONCE per dirty entity)
    Buffer Renderables;      // RenderableInputGpu[]   (was per-frame repacked in cull)
    Buffer Lights;           // PbrLightGpu[]
    Buffer Materials;        // PbrMaterial[] + fallback slot
    Buffer ShadowInfo;       // ShadowEntityInfo[]
    Buffer EmissiveTris;     // EmissiveTriGpu[] (+ alias table)
    AccelerationStructureKHR Tlas;

    // The single "scene" descriptor set every core binds (see below).
    DescriptorSet SceneSet(uint frame);

    RenderableHandle Register(Entity* e);   // allocates a stable slot
    void MarkDirty(RenderableHandle h);      // edit → re-pack only this slot next Extract
    void Free(RenderableHandle h);
}
```

### `RenderView` — the immutable per-frame snapshot (this *is* the fat FrameContext)

```csharp
readonly ref struct RenderView
{
    uint     FrameIndex;
    Extent2D RenderExtent;

    // Precomputed once per frame (was recomputed in cull/pick/lighting separately):
    Matrix4x4 View, Proj, ViewProj, InvViewProj;
    Vector3   CameraPos;

    // Counts + the bound buffers — cores read these, never the ECS:
    uint RenderableCount, LightCount, MaterialCount;
    DescriptorSet SceneSet;          // GpuScene's single set
    AccelerationStructureKHR Tlas;
}
```

## What changes

1. **Extract is the only ECS reader.** A single `GpuScene.Extract(scene, camera)`
   replaces the scene-walking bodies of `UpdateLights`, `UpdateMaterials`, the packing
   half of `DrawCullPipeline.Record`, and the instance/`ShadowEntityInfo` packing in
   `RebuildTlas`. Those subsystems now *consume* `GpuScene` buffers.

2. **Concerns split out of `cull.Record`.** Today it does extraction **+** frustum
   cull **+** transparent classification **+** back-to-front sort. After L2:
   - Extraction → `GpuScene.Extract`
   - Frustum cull → stays in the cull compute pass, reading `GpuScene.Renderables`
   - Transparent sort → a **view-dependent** step downstream of `GpuScene` (it is not
     extraction; it depends on the camera), producing the `TransparentDraw[]` list.

3. **Transforms computed once.** A transform pass writes cached world matrices into
   `GpuScene.Transforms`; lights/cull/TLAS read the cache instead of each calling
   `GetWorldMatrix()` (which walks the parent chain).

4. **Retained + dirty cadence.** A CAD scene is static most frames. Extraction is
   driven by dirty flags (formalizing the existing `tlasDirty` instinct + component
   lifecycle), not the frame clock. Full re-pack becomes the cold path (load / bulk
   import). Start *pull-with-dirty-flags*; escalate to *push via EventBus* only if a
   profile demands it.

5. **One "scene" descriptor set.** `GpuScene` exposes transforms/renderables/lights/
   materials/TLAS/shadow-info as a single bindless set. Cores bind that one set; resize
   no longer re-threads these (only transient targets get rebound). This is the lever
   that shrinks the manual descriptor wiring.

6. **`Scene` becomes pure authoring data** — entities, components, hierarchy, material
   registry. The `renderGraph` reference moves off `Scene` (it's a rendering concern).

## Migration steps

1. Add `GpuScene` owning the buffers currently created in `Renderer_Resources`
   (`lightStorageBuffers`, material SSBO, cull input buffers) + the AS buffers.
2. Add the `RenderableHandle` slot allocator; register on `Scene.AddEntity`.
3. Move `UpdateLights`/`UpdateMaterials` bodies into `GpuScene.Extract`.
4. Move renderable packing out of `cull.Record` into `Extract`; leave cull as cull.
5. Add the cached transform pass; repoint lights/TLAS to read it.
6. Repoint pick/selection/TLAS identity from `Scene.IndexOf` → `RenderableHandle`.
7. Introduce dirty tracking; switch Extract from per-frame to dirty-driven.
8. Build the single scene descriptor set; collapse the per-pipeline rebinds.

---

# L3 — Render-core model

**Goal:** replace the 4-mode `switch` + parallel `DrawX` methods with pluggable
**render cores**. `Renderer` becomes a **host** that owns shared resources and the
frame skeleton; each core owns one technique and produces a single agreed output.

## The pain being removed

`DrawFrame` switches `RenderMode` into `DrawDeferred/DrawRayQueried/DrawPathtraced/
DrawRayTraced`. `DrawPathtraced` and `DrawRayTraced` are ~80 near-identical lines.
Tonemap is inconsistent (graph pass in deferred, manual `tonemapPipeline.Record` in
PT/RT), forcing the `_lastRenderMode` HDR-input rebind hack under `DeviceWaitIdle`.
Adding a mode means a field + switch arm + `DrawX` + wiring in 3 places.

## The contract (the whole game)

> **A core consumes a `RenderFrame` and produces a single HDR scene-color resource
> at `RenderExtent`, left in a known layout (`ShaderReadOnly`). The host owns
> everything after: tonemap, selection outline, present, ImGui overlay.**

This single contract:
- collapses the PT/RT duplication into one shared base,
- makes **tonemap a host stage for all modes** → deletes the `_lastRenderMode`
  rebind hack entirely,
- keeps editor concerns (pick, outline, viewport) mode-agnostic and host-side.

## Data models

```csharp
// Per-frame context handed to a core. Composes L1 (device) + L2 (RenderView).
readonly ref struct RenderFrame
{
    GraphicsDevice Device;       // L1
    CommandBuffer  Cmd;
    RenderView     View;         // L2 — scene data, camera matrices, scene set
    ImageResource  SceneColorHDR;// the core's required output target (host-owned)
}

interface IRenderCore : IDisposable
{
    string Name { get; }
    void Initialize(GraphicsDevice device, RenderResources targets);
    void Resize(Extent2D extent);          // rebind transient targets
    void Render(in RenderFrame frame);     // MUST fill frame.SceneColorHDR
}
```

### Cores (one per current `DrawX`)

| Core | Replaces | Owns (technique-local) | Reads (shared) |
|---|---|---|---|
| `DeferredCore` | `DrawDeferred` + `SetupDeferredRenderer` | g-buffers, geometry/lighting/skybox/transparent pipelines, its sub-graph | `RenderView` (scene set, lights, TLAS) |
| `ForwardPlusCore` | `DrawRayQueried` (stub today) | forward+ pipelines | `RenderView` |
| `PathTraceComputeCore` | `DrawPathtraced` | `ptComputePipeline`, accumulator | `RenderView`, IBL, emissive |
| `PathTraceRTCore` | `DrawRayTraced` | `rtPipeline`, accumulator | `RenderView`, IBL, emissive |

`PathTraceComputeCore` and `PathTraceRTCore` share a base for the camera-dirty
accumulator reset + pre-dispatch barriers (the ~80 duplicated lines).

### Host (`Renderer` after L3)

- **Owns shared/global:** `GraphicsDevice`, `Swapchain`, `FrameRing`,
  `RenderResources`, `GpuScene`, IBL, the TLAS lifetime, the active core, **and the
  host post-stack: tonemap + outline + present + ImGui**.
- **Per frame:** `FrameRing` acquires → `GpuScene.Extract` (dirty-driven) builds the
  `RenderView` → `activeCore.Render(frame)` fills `SceneColorHDR` → host runs tonemap →
  `RecordSelectionOutline` → blit → ImGui → present.
- **Mode switch:** swap `activeCore`. With tonemap host-side reading a stable
  `SceneColorHDR`, no descriptor rebind / `DeviceWaitIdle` dance is needed.

## Shared vs. core-owned (resolves the ownership question)

| Resource | Owner | Why |
|---|---|---|
| Camera UBO, lights, materials, transforms, TLAS, scene set | **host / `GpuScene`** | every core reads them; extract once |
| IBL cubemaps, BRDF LUT, probe records | **host** | stable, shared across cores |
| `SceneColorHDR`, `FinalColor`, depth | **host** (`RenderResources`) | the contract surface + post-stack |
| g-buffers | `DeferredCore` | technique-local |
| PT accumulator / `ptOutColor` | PT cores | technique-local |
| tonemap / outline / pick pipelines | **host** | mode-agnostic post + editor |

## What changes

1. `DrawFrame` loses the `switch`; it runs the fixed host sequence and calls
   `activeCore.Render`.
2. Each `DrawX` body moves into its core; PT/RT dedupe into a shared base.
3. Tonemap moves out of the deferred graph and the PT/RT manual calls into **one host
   stage**; `_lastRenderMode` + the HDR-input rebind are deleted.
4. `RenderResources` exposes a stable `SceneColorHDR` that cores write and the host
   reads.
5. `RebuildPbrPipelines` / mode-switch rebinds shrink to core-local concerns.

## Migration steps

1. Define `IRenderCore` + `RenderFrame`; add a host post-stack (tonemap+outline+present)
   that reads a stable `SceneColorHDR`.
2. Wrap the existing `DrawDeferred` in `DeferredCore` (move its pipelines + sub-graph in).
3. Wrap PT and RT into cores; extract their shared base.
4. Replace the `DrawFrame` switch with `activeCore.Render`; delete the `_lastRenderMode`
   tonemap rebind.
5. Convert `ForwardPlusCore` from stub when ready — now a self-contained addition with
   zero edits to the host.

---

## End state

```
Engine
 └─ Renderer (host)
     ├─ GraphicsDevice            ← L1
     ├─ Swapchain                 ← L1
     ├─ FrameRing                 ← L1
     ├─ RenderResources           ← L1 (SceneColorHDR, FinalColor, depth)
     ├─ GpuScene                  ← L2 (extract, buffers, stable handles, scene set)
     ├─ IRenderCore activeCore    ← L3 (Deferred / Forward+ / PT-Compute / PT-RT)
     └─ host post-stack: tonemap → outline → blit → ImGui → present
Scene  = pure authoring data (entities, components, hierarchy, materials)
```

- **Simpler:** `Renderer` shrinks from "owns everything" to "owns shared resources +
  drives the frame." Each `DrawX` and its pipelines live in a self-contained core.
- **Extensible:** a new technique = one new `IRenderCore`, zero host edits. A new
  pass within a technique = one pass in that core's graph.
- **Separation of concerns:** ECS read happens in exactly one place (`Extract`);
  device ops behind `GraphicsDevice`; technique behind `IRenderCore`; the scene→GPU
  seam is `GpuScene` + `RenderView`, full stop.

## Open decisions to lock before starting

- **L2 identity:** generational `RenderableHandle` (recommended — kills the latent
  flat-index reorder bug).
- **L2 cadence:** retained + dirty (recommended for CAD) vs per-frame re-pack.
- **L2 transforms:** one cached transform pass (recommended) vs per-consumer recompute.
- **L3 core lifetime:** eager (all cores built up front — instant switch, more VRAM)
  vs lazy (build on switch — hitch, less VRAM). Eager is fine for an editor; keep the
  API from precluding lazy.
- **Graph scope (parallel track):** "explicit order + auto-sync over buffers+images+
  compute+RT with resource versioning" vs a full frame-graph. Recommended: the former.