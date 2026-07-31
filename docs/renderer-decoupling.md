# Renderer Decoupling & Coherence

The plan for stripping the `Renderer` god object to a thin composition root, making the base
renderer class closed against new features, giving each subsystem a single clear owner, and
routing every cross-subsystem need through a narrow channel. Adding a feature should be one new
file: no edit to `Renderer`, `DrawFrame`, or `Cleanup`.

This is a target and a migration plan, not the current state. Status lives in the migration order
at the end.

## Guiding principles

1. **One owner per concept.** Every piece of state has exactly one authoritative home. No mirrored
   copies, no parallel dirty flags, no "the panel keeps one and the renderer keeps another".
2. **Data flows one direction.** GpuScene owns scene GPU truth; consumers read it. The renderer
   owns the frame skeleton; features are driven by it. UI emits intent outward and owns no renderer
   state.
3. **The draw loop orchestrates per-frame work exactly once.** Technique-independent work (scene
   extraction, transforms) runs in the skeleton, and cores consume the result -- they never
   re-trigger it.
4. **Depend on the narrowest thing that works.** Never take `Renderer`. Take a capability, a
   provider contract, a constant table, or a per-frame view -- see the channels below.
5. **No ambient globals as the coupling substrate.** Statics are for genuinely process-global,
   single-instance infrastructure only, never a convenient reach into subsystem state.

## Architectural decisions (locked)

- **Features receive infrastructure via marker-interface setter injection** (`INeedsGpu`), and
  **collaborators via `INeedsFeature<T>`**; both are filled by the `FeatureHost` in one wiring
  pass. **Pipelines keep a single `GpuContext` ctor arg** (they are constructed explicitly by
  cores/modules, matching the existing `DeferredModule` style). Corollary settled in step 7: because
  injection covers everything, feature constructors and `Initialize` take **no** parameters -- there
  is no `FeatureContext`, and the transitional host reach is itself an injected interface
  (`INeedsHost`).
- **Features self-register via a `[ModuleInitializer]` descriptor** (factory + gate + order); the
  `FeatureHost` sorts, gates, and builds. Adding a feature file *is* the registration, and the
  descriptor is a `static abstract` interface member so the compiler rejects a feature without one.
- **Typed feature fields on `Renderer` become bridge accessors first**
  (`internal IblSystem Ibl => _features.Get<IblSystem>()!;`), so existing callers keep working
  while `Renderer` stops constructing/disposing them. Callers migrate off later, per-caller.

Open (non-blocking, current recommendation in the text): AS-verb grouping shape on `GfxDevice`
(recommend a grouped `gfx.As.*` facet); whether `EditorState.SelectedEntity` stays a store while
selection *changes* flow as events (recommend yes).

## Coupling audit

Ranked by *distinct* `Renderer` members touched. High volume + few distinct = narrow and fine;
many distinct = god-object use.

| Consumer                | reaches / distinct | Verdict                                            |
|-------------------------|--------------------|----------------------------------------------------|
| `RendererSettingsPanel` | 53 / 20            | Worst: UI mutating renderer internals              |
| `PbrDeferredPipeline`   | 40 / 15            | Pipeline pulling infra + features + consts + state |
| `InspectorPanel`        | 37 / 3             | Good: narrow, intent-only                          |
| `ResourceManager`       | 33 / 12            | Takes Renderer, uses only the RHI                  |
| `DeferredCore`          | 31 / 21            | Orchestrator fishing sibling pipelines from host   |
| `GpuScene`              | 13 / 5             | Constants + one behavior reach                     |

Counts are from the original audit and are not re-measured each step; `PbrDeferredPipeline`,
`DeferredCore` and `DrawCullPipeline` are all lighter than shown since step 5 (extraction and the
light/material packs no longer route through `Renderer`).

Renderer-coupling anti-patterns:

- **A1. UI mutating renderer internals** (`RendererSettingsPanel`). Flips raw fields
  (`pendingTonemapRebuild`, `tonemapOperator`, `softShadowsEnabled`, `renderMode`) and pokes
  individual pipelines. Fix: intent events / a command facade. `InspectorPanel` is the model:
  only `MarkAccumulatorDirty` / `MarkTlasDirty`. *(Partly done: the four raw-field flips and the
  accumulator reaches are now published intents. The pipeline pokes -- `tm.Exposure`, `cam.Mode`,
  `pt.BounceCap`, `probeSys` -- remain, and are step 9.)*
- **A2. Systems that only need the RHI** (`ResourceManager`). Every reach is device territory
  (`CreateBuffer`, `UploadBufferData`, `GpuMemoryAllocator`, `device`, `descriptorPool`,
  `WriteBindlessTexture`). Fix: inject `GpuContext`; drop `Renderer`.
- **A3. Constants sourced off Renderer** (`GpuScene`, every pipeline). `MAX_CONCURRENT_FRAMES`,
  `MAX_INSTANCES/MATERIALS/LIGHTS`, `TILE_SIZE`, `MAX_TILE_COUNT`. Fix: `static RenderConfig`.
- **A4. Renderer as an ad-hoc pipeline registry** (`DeferredCore`). Fishes `host.transparentPipeline`
  etc., yet its own `DeferredModule` already takes those by explicit injection. Fix: the core
  owns/constructs its pipelines and hands them to the module; never re-reads them from `host`.
- **A5. Behavior reach-through** (`UpdateLights`, `UpdateMaterials`). A subsystem's behavior parked
  on Renderer. Fix: a feature with a contract interface, resolved like IBL. *(Done for the two
  named: both forwarders are deleted, the bodies are GpuScene's, and the draw loop is the only
  caller. What remains under this heading is the AS path -- `SyncRenderables` and `RebuildTlas`
  hanging off `Renderer` -- which step 8 resolves as `SceneAS`.)*

Non-Renderer anti-patterns (broader coherence):

- **G1. Double-bookkeeping of the rebuild flags.** `RendererSettingsPanel` keeps its own
  `static _pbrRebuildPending` / `_tonemapRebuildPending` (`:34-35`) and copies them into the
  renderer's separate `pendingPbrRebuild` / `pendingTonemapRebuild` (`Renderer_Core.cs:221-222`).
  One intent, two flags, two classes. Fix: publish an event; a single owner holds one flag. *(Done.
  The panel's flags survive but now mean "Apply is pending", not "a rebuild is due".)*
- **G2. `Engine.renderer` public static mutable global** (68 reaches / 8 files). Ambient access
  and a `public` settable static (any code can reassign the renderer). Mostly falls out as UI moves
  to events + narrow interfaces; regardless, make it `{ get; private set; }` immediately.
- **G3. Instance state modeled as `static`.** `Scene.cs:31 private static Camera Cam;` -- all
  `Scene` instances share one camera (a latent bug for a second scene / headless bake).
  `Renderer_Ray_Query.cs:740 private static uint PreviousCount;` -- per-renderer TLAS state held in
  a static. Fix: instance fields.
- **G4. Render code reading editor globals.** `EditorState` is a `public static` mutable blob;
  fine for pure-UI flags, but render/pipeline code reads `SkyboxEnabled` / `IblIntensity` /
  `SkyboxIntensity` / `SelectedEntity` directly, coupling the render path to editor globals. Fix:
  those flow into the renderer as data (`RenderConfig` / frame settings) or events; leave the
  pure-UI flags alone.
- **G5. Generic `throw new Exception` x130.** No caller can distinguish failure types. Mostly fatal
  init paths, so a consistency nit, not an architecture flaw. Optional: a `VkCheck(result, "...")` /
  `VulkanException` helper. Orthogonal to everything else.

Checked and fine (do not chase): `switch(renderMode)` technique branching is already gone (cores
replaced it); descriptor-pool boilerplate is already centralized in `PipelineBase`;
`Component._nextID` is standard id generation; the `null!` two-phase-init fields resolve under the
`FeatureHost.Initialize()` pattern below.

## Target: dependency channels, none of them `Renderer`

Everything a system might grab off `Renderer` sorts into exactly one channel.

| Need                                             | Channel             | Mechanism                                        |
|--------------------------------------------------|---------------------|--------------------------------------------------|
| Device verbs (buffers, images, single-time cmd)  | `GpuContext`        | pipelines: ctor arg; features: `INeedsGpu`       |
| Bindable GPU resources (IBL cubes, g-buffer)     | `DescriptorRegistry`| named provider/consumer broker (exists)          |
| Feature collaborators (IBL scalars, probes)      | `FeatureHost`       | `INeedsFeature<T>` (or `Features.Get<T>()`)      |
| Constants (MAX_*, TILE_SIZE)                      | `RenderConfig`      | static; no injection                             |
| Per-frame state (extent, camera, renderables)    | `RenderFrame` / `RenderView` | produced by the draw loop; read at record time |
| UI -> engine intent                              | `EventBus`          | fire-and-forget events (no state on the panel)   |
| Off-thread notification (worker -> engine)       | `EventBus`          | publish from any thread; drained on the bus thread |

`GpuContext` is deliberately capped and never grows:

```csharp
public readonly record struct GpuContext(
    GraphicsDevice Gfx, DescriptorRegistry Registry, ShaderLibrary Shaders);
```

The discipline: nothing frame-scoped or feature-scoped goes in it. Any temptation to add a fifth
thing is frame state (to `RenderView`), a constant (to `RenderConfig`), or a feature (to
`INeedsFeature<T>`).

### Access mechanisms

Pipelines take `GpuContext` in the ctor (explicit, matches modules). Features declare their needs
as interfaces and the `FeatureHost` fills them in one wiring pass -- no ctor params, opt-in by
which interfaces a feature implements:

```csharp
interface INeedsGpu        { GpuContext Gpu { set; } }                      // infra
interface INeedsFeature<in T> where T : class { T Dependency { set; } }     // a collaborator contract
interface INeedsHost       { Renderer Host { set; } }                       // transitional; deleted at the end

// FeatureHost wiring pass (phase 2 of BuildAll), once, after all features are constructed:
foreach (var f in _all) {
    if (f is INeedsGpu g)  g.Gpu  = _gpu;
    if (f is INeedsHost h) h.Host = _host;
    WireFeatureDeps(f);   // for each INeedsFeature<T> implemented, resolve T from _all and set it
}
```

Infrastructure is a direct type test; collaborators need reflection, because the contract type is a
generic argument and is only knowable from the feature's implemented-interface list. That is once
per feature at boot, not per frame. An unresolvable dependency throws at wiring with both type names
and the likely cause -- usually a consumer that forgot the gate its provider has.

A feature needing two collaborators implements `INeedsFeature<T>` twice with explicit-interface
setters:

```csharp
sealed class ReflectionProbeSystem : IRenderFeature, IBakeFeature,
                                     INeedsGpu, INeedsFeature<IIblProvider>
{
    IIblProvider _ibl = null!;
    IIblProvider INeedsFeature<IIblProvider>.Dependency { set => _ibl = value; }
}
```

The injected value is a *reference* available after wiring. If a feature needs its collaborator
already *baked/initialized* (not merely referenced), it uses it in `Initialize()` / `Bake()`, where
descriptor `Order` guarantees the collaborator ran first. That is what dissolves the old
`ReflectionProbeSystem(renderer, Ibl)` ctor coupling.

### Feature-to-feature bindable resources

Consumers of a feature's *bindable* resources go through `DescriptorRegistry` and never name the
producer: IBL calls `RegisterImage("iblIrradiance", ...)` on bake (re-register on rebake is a free
hot-swap via the per-frame `BeginFrame` rewrite); a consumer declares the binding name. Only
non-bindable data (e.g. `prefilteredCubeMipLevels`) uses `INeedsFeature<IIblProvider>`, reading the
value at record time so rebakes never leave a stale copy.

## Feature lifecycle and the closed base renderer

*(Done, and exercised: eleven features, every phase pump with at least one implementer, and
`INeedsFeature<T>` resolving two contracts.)* Cores already proved
two thirds of the pattern (self-register via `RegisterCore`, list-disposed). It is now generalized,
and `DrawFrame`/`Cleanup`/`Initialize` no longer change per feature. Lives in
`Renderer/FeatureLifecycle/`.

```csharp
interface IRenderFeature : IDisposable { string Name { get; } void Initialize(); }
interface IResizeFeature   : IRenderFeature { void Resize(Extent2D extent); }
interface IBakeFeature     : IRenderFeature { bool BakePending { get; } void Bake(); }
interface IPreDrawFeature  : IRenderFeature { void PreDraw(in RenderView view); }
interface IPostDrawFeature : IRenderFeature { void PostDraw(CommandBuffer cmd, in RenderView view); }
interface IRenderCore      : IResizeFeature { RenderMode Mode { get; } void Activate(); void Render(in RenderFrame f); }
```

**`Initialize` takes no arguments, and neither does the factory.** The sketch above used to thread a
`FeatureContext` through both; it was a second dependency channel doing what the wiring pass already
does. By the time `Initialize` runs, phase 2 has set `Gpu` via `INeedsGpu` and every collaborator via
`INeedsFeature<T>` -- so a feature already holds everything it needs in fields, and a constructor has
nothing useful to receive. Construction is `() => new DeferredCore()`. One channel, not two.

The one thing the context was carrying that injection did not cover is the transitional `Renderer`
reach, and that now rides the same wiring pass as `INeedsHost`. Keeping it an *interface* is the
point: every feature still coupled to the host declares it in its own type header, so "what is left
to decouple" is a search for `INeedsHost` and the boot manifest tags those features `NEEDS-HOST`.
When the last implementer drops it the interface is deleted, and the compiler proves nothing reaches
the host any more.

**Registration is compiler-enforced.** A feature declares its own descriptor through a
`static abstract` member, so the build fails if a feature exists without one:

```csharp
internal sealed class ReStirDICore : IRenderCore, ISelfRegisteringFeature<ReStirDICore>, INeedsHost
{
    public static FeatureDesc Desc =>
        new(Order: 60, Gate: gpu => gpu.Gfx.RayTracePipelineSupported, Make: () => new ReStirDICore());

    [ModuleInitializer]
    internal static void _Reg() => FeatureCatalog.Register<ReStirDICore>();
}
```

It has to be `static abstract` on a self-referencing interface rather than a plain static member on
`IRenderFeature`: a plain static interface member is *one storage slot shared by every implementer*,
so six cores would overwrite each other's descriptor. The `_Reg` line stays per-file because
`[ModuleInitializer]` runs once per **assembly**, not once per type -- there is no per-type startup
hook in .NET short of an assembly-wide reflection scan. It cannot silently drift, though:
`Register<T>` only accepts a type satisfying the interface, so it will not compile without a `Desc`.

`BuildAll` runs three passes over the whole set. They are separate because each needs the previous
to have finished for *every* feature, not just the ones ahead of it in `Order` -- wiring before all
features exist would resolve a dependency to null purely because its `Order` was higher.

```csharp
foreach (var d in FeatureCatalog.Descriptors.OrderBy(d => d.Order))
    if (d.Gate(_gpu)) Add(d.Make());   // 1. construct (gated) -- fields still null
foreach (var f in _all) Wire(f);        // 2. INeedsGpu / INeedsHost / INeedsFeature<T> injection
foreach (var f in _all) f.Initialize(); // 3. build owned state; lower Order is fully built
```

`Renderer.Initialize` lost every feature name. Its whole feature section is now:

```csharp
_features = new FeatureHost(Gpu, this);
_features.BuildAll();
Console.WriteLine(_features.Dump());
_activeCore = RenderCores[_desiredCoreIndex];
_activeCore.Activate();
```

`DrawFrame` has fixed pump points (`ServiceBakes` at the top with the other pending-work flags, then
`PreDraw` -> active core `Render` -> `PostDraw`), plus `Resize`. Bake is request-driven, so a feature
with nothing stale costs one bool test. `Cleanup` collapses to `_features.Dispose()` -- reverse
`Order`, the exact inverse of the Initialize guarantee -- then `renderTargets`, then `gfx` last.
`IRenderCore` is a specialization of `IResizeFeature`: exactly-one-active, mode-switched, produces
the draw, and inherits registration/gating/ordering/wiring/dispose rather than having a parallel set.
Its `Order` doubles as its mode-combo index, so the lowest-Order core is the boot default.

The tradeoff of descriptor-scattered `Order`: you lose the single readable boot-order list. The
`_features.Dump()` manifest (the analog of the existing `descriptorRegistry.DumpBindings()` call)
replaces it with a runtime view that also reflects gating and flags remaining host coupling:

```
[features] 6 built, 0 gated out by device caps
  Deferred                     core:Deferred NEEDS-HOST
  PathTrace (Compute)          core:RayCompute NEEDS-HOST
  PathTrace (RT pipeline)      core:RayTrace NEEDS-HOST
  Forward+ (stub)              core:ForwardPlus
  PathTrace (Wavefront)        core:RayWavefront NEEDS-HOST
  ReSTIR DI (RT pipeline)      core:ReStirDI NEEDS-HOST
```

Follow-on: `RegisterSceneBindings` becomes feature-driven -- each feature registers its own scene-set
bindings in `Initialize()`, shrinking the central method to genuinely-core bindings only.

## Scene state and the draw loop

**GpuScene is the single source of scene GPU truth** (materials/instances/lights/emissive SSBOs, the
generation-checked `RenderableHandle` registry, world-transform cache). It is already the
best-factored subsystem; keep it a data owner and do not put Vulkan AS API in it. Its only cleanup
is A3 (constants -> `RenderConfig`) and A5 (`UpdateLights` -> a lights feature).

**Extraction is owned by the draw loop, runs once, and is consumed downstream.** *(Done.)* It used
to be smeared: `DeferredCore` drove `ExtractRenderables`, each of the four PT/RT cores drove its own
`UpdateLights` + `UpdateMaterials`, `PbrDeferredPipeline.UpdatePerFrame` drove a *third*
`UpdateLights`, and the draw loop drove `BeginTransforms` for all of them. Now `DrawFrame` has one
extraction phase --
`BeginTransforms -> UpdateMaterials -> UpdateLights -> ExtractRenderables` -- and builds the frame's
`RenderView` from its results. Cores read counts off the view; `DrawCullPipeline.Record` takes the
view and reads `RenderableCount` + `TransparentCandidates` from it instead of reaching into
`Renderer.gpuScene`. `Renderer.UpdateLights` / `UpdateMaterials` / `LightCount` /
`GetLightStorageBuffer` are deleted, so there is no longer a way to re-trigger a pack off the host
(which mattered: extraction is dirty-driven per frame slot, so a second call in the same frame
returns a *stale cached count*, not a fresh pack). Stable *queries* (`ResolveSlot`, `TryGetHandle`)
remain open to anyone.

Two consequences worth knowing. Extraction now runs in every render mode, not just deferred -- the
PT/RT modes pack a cull buffer they do not read. It is dirty-gated, so on a static scene that costs
nothing after the first `MAX_CONCURRENT_FRAMES` frames, and it is the price of the phase being
technique-independent. And `RebuildTlas` still opens its own `BeginTransforms` window, because the
asset / visibility paths call `OnSceneEntitiesChanged` directly from UI callbacks, outside
`DrawFrame`. That folds into the single window when `SceneAS` turns the rebuild into a
request-driven bake at a fixed pump point.

**`RenderView` is the per-frame snapshot, and it is the old `FrameContext`.** Rather than have both,
`Renderer.FrameContext` was renamed into the `RenderView` stub that had been sitting dead in
`GpuScene.cs` and given the extraction outputs (`RenderableCount`, `LightCount`, `MaterialCount`,
`TransparentCandidates`) alongside the frame index / camera / scene / extent it already carried. It
now flows unchanged through `RenderFrame` -> `core.Render` -> `FrameGraph.Execute` -> every
`PassExecute` body, so a pass that needs frame state has it in hand rather than reaching for a host.
Deliberately *not* in it: stable buffers, descriptor sets, pipelines -- those are not frame state.
Also deferred: precomputing `View`/`Proj`/`ViewProj` on the view. The call sites disagree today on
aspect, near/far and Y-flip convention, so unifying them changes pixels and wants to be its own
verified change, not a rider on a plumbing step.

## Acceleration structure: verb / policy split

The AS in `Renderer_Ray_Query.cs` was two altitudes bundled:

- **AS verbs** (create/destroy AS, device address, `CmdBuildAccelerationStructures`, compaction copy,
  the `khrAccelStruct` dispatcher + `PhysicalDeviceAccelerationStructurePropertiesKHR`) are thin,
  stateless RHI wrappers at the same altitude as `CreateBuffer`/`CreateImage`. **Moved to
  `GraphicsDevice`** as the grouped `gfx.As.*` facet (`AsDevice`), which avoids flattening a dozen
  methods onto the top-level surface. Test: "could a different renderer with a different BLAS
  strategy reuse this unchanged?" -> if yes, it is a device verb. *(Done -- see below.)*
- **AS policy** (BLAS clustering / scene-wide SAH strategy, rebuild-vs-refit, compaction scheduling,
  instance-record packing) is renderer logic. **It becomes a gated `SceneAS` feature.**

`SceneAS` owns only the Vulkan AS objects, **consumes GpuScene** as its source of truth (reads the
renderable set + world transforms + handle->slot map instead of re-deriving them; repacks the world
cache into `TransformMatrixKHR` rather than re-walking the scene), **publishes the TLAS** through
`DescriptorRegistry.RegisterTlas` so consumers bind by name, and reads `ShadowEntityInfo` from
GpuScene (that struct already lives in `GpuScene.cs` but is currently authored by the AS path --
move authorship home). Dirty tracking unifies: GpuScene owns "scene changed"; `SceneAS` consumes it
rather than keeping a parallel `tlasDirty`. `MarkTlasDirty` (today a `Renderer` method the UI calls)
becomes GpuScene's, driven by a `SceneMutated` event.

Verify before implementing: whether the BLAS build needs a geometry layout GpuScene does not expose
(it currently walks meshes for the cluster SAH build). If so, GpuScene gains a small geometry-view
accessor rather than `SceneAS` re-walking the scene -- still one source of truth, wider read surface.

### The `gfx.As.*` facet (done)

`AsDevice` holds the dispatch table, the cached scratch alignment, and eight verbs: `Create`,
`Destroy`, `DeviceAddress`, `GetBuildSizes`, `Build`, `QueryCompactedSize`, `CopyCompact`, plus
`Available`. Nothing more -- each one is only the Vulkan call plus the struct it needs filled, and
nothing in it decides *what* to build.

Three judgement calls worth recording:

- **`Build` submits, it does not just record.** Both call sites did
  `BeginSingleTimeCommands -> CmdBuildAccelerationStructures -> EndSingleTimeCommands` identically,
  so the submit is part of the verb. A batched or async builder would want a `CmdBuild` recording
  into a caller-supplied buffer instead; that is a policy change for the AS owner to ask for, and
  the facet grows the overload when something actually needs it. Same for `QueryCompactedSize`,
  which owns its query pool and blocks -- compaction *scheduling* stays policy, the size query is a
  verb.
- **The scratch buffer stayed behind.** It is state, not a verb, and "one persistent scratch sized
  to the largest build so far" is the AS owner's growth policy. Only `ScratchAlignment` -- a device
  property -- moved. It goes to `SceneAS` in step 8 along with the instance / shadow-info /
  cluster-transform buffers.
- **The load gate widened, deliberately.** The dispatch table now loads whenever
  VK_KHR_acceleration_structure is enabled, not only when ray *query* is. Whether the verbs are
  callable is a device fact; whether anything gets built stays `RayShadowsSupported`, checked
  unchanged in `InitRayQuery`. This costs nothing today and stops a latent trap where an
  RT-pipeline-capable device without ray query would have had no dispatch table.

Two duplicates fell out on the way. `Renderer.CreateBufferWithDeviceAddress` had become a
byte-for-byte copy of `gfx.CreateBuffer` (the allocator started putting the device-address flag on
every buffer block), and `GetBufferDeviceAddress` existed privately in two places
(`Renderer_Ray_Query` and `RTPipeline`, for the SBT). Both now resolve to one
`gfx.GetBufferDeviceAddress`.

`Renderer_Ray_Query.cs` lost ~100 lines and every `khrAccelStruct` / `vk` / `device` reach in its
build paths. What remains is exactly the policy the doc says belongs to `SceneAS`: clustering, the
capacity grow-by-doubling helpers, instance and `ShadowEntityInfo` packing, emissive collection, and
the rebuild cadence.

## UI via events

The UI emits intent and owns no renderer state. Boundary: **events are fire-and-forget
intents/notifications only** (`TonemapFilterChangedEvent`, `SceneDirtyEvent`,
`PathTracingAccumulatorInvalidatedEvent`, `SceneEntitySelectedEvent`) -- never for request-response
data (that stays a direct call to a narrow interface). Do not over-eventify: when there is one clear
owner and synchronous ordering matters, an intent method (the `RequestCoreIndex` style) is clearer.
Events earn their keep exactly when the publisher should not know who handles it, which is the UI's
situation.

This retires A1 and G1 (settings-panel flag flips + duplicate pending flags ->
`TonemapFilterChangedEvent`, owner holds one flag) and G4 (editor render-settings ->
events/`RenderConfig` data).

### The bus (done)

`EventBus` was input/window-oriented and had three properties that would not survive renderer
traffic. All three are fixed; the rewrite is in `VulkanEngine/Events/`.

**Deleted.** `EventDispatcher` resolved an event's type and then called `handler.OnEvent(evt)` --
the same untyped entry point -- so the type parameter bought nothing and the handler re-switched
anyway; it also compared types exactly, silently missing subclasses. `EventSystem` was a second
competing bus reachable only from the commented-out `PhysicsComponent` (its "uses stackalloc" doc
comment was untrue). `Event.GetEventType()` was redundant with `GetType()`. `Event.Clone()` existed
only to copy into the deferred queue, defending against a mutation that never happens -- publishers
build an event and drop it -- and three of the renderer events implemented it as
`throw new NotImplementedException()`, so the only existing renderer event hard-crashed the moment
anything queued it. Events are now `sealed` with `readonly` payload instead, and `Clone` is gone.

**Fixed.**

- *Reentrancy.* Dispatch iterated the live listener dictionary, so any handler that subscribed or
  unsubscribed threw `InvalidOperationException`. Subscription state is now copy-on-write: writers
  rebuild an array under a lock and publish it by reference, the dispatch loop iterates a snapshot
  that cannot change under it. A late unsubscribe still takes effect mid-event through a volatile
  `Alive` flag rather than waiting for the next publish.
- *Category matching was subset, not intersection.* `(flags & mask) == mask` meant a listener's mask
  had to be a **subset** of the event's bits, so a panel subscribing to `Renderer | Editor` would
  have received nothing. Now `!= 0`. `AddListener` also used `Dictionary.Add`, so a second subscribe
  threw and one listener could not hold two masks; subscriptions are independent entries now.
- *Half-present thread safety.* `Monitor.Pulse` with no `Wait` anywhere was a no-op, publish did not
  lock while drain did, and the whole drain ran inside the lock so a handler that published fed the
  loop it was inside. See threading below.

**Delivery is per category, not one global `immediateMode` flag.** The old bool defaulted to true,
which made `ProcessEvents` dead code and left the queued path untested. Input/window/editor stay
immediate (latency, and it is the previous behaviour); `Renderer` is queued, so a UI callback cannot
rebuild a pipeline part-way through a frame. An event carrying several bits queues if any bit is
queued -- delivery is a safety property, the stricter bit wins. `SetDelivery` overrides per category;
resolution is one masked compare against a `uint`, no lock and no allocation, because publish is on
the mouse-move path.

**Threading / async.** Publishing is safe from any thread; delivery is not, since every handler
touches Vulkan, ImGui or scene state. `BindToCurrentThread` (called from `Engine.Start`) names the
delivery thread, and anything published from off it is queued regardless of category and drained in
the next `ProcessEvents`. That is the channel `AsyncResourceManager`'s worker needs -- its load
callbacks currently fire *on the worker thread*, so today they cannot touch the renderer at all.
`ProcessEvents` double-buffers the queue and dispatches outside the lock; a handler that publishes
lands in the next pass, capped at 8 passes so a publish cycle is a logged warning rather than a hung
frame. `NextAsync<T>(ct)` completes on the next `T` for async flows that want to wait on a point in
the frame; its continuation resumes on the thread pool, never inline on the drain, so getting back
onto the bus thread means publishing an event.

**Two ways to subscribe, both returning a disposable token.** `AddListener(listener, category)` is
the broadcast path, for consumers that want a whole stream in order and switch on it themselves
(ImGui, `Camera`) -- the C# `switch` pattern-match they already use is what `EventDispatcher` should
have been. `Subscribe<T>(handler)` is one type, one delegate, for single-owner intents where a
category switch is noise; it walks the base chain, so a future grouping base type still fires. The
token matters more than the typing: features are created and disposed by `FeatureHost` (step 7), and
a disposed feature left on the listener list would be handed an event and dereference GPU handles it
has already destroyed. Features hold tokens and dispose them with themselves.

**Ordering + `Handled`.** Subscriptions carry an `order` and an optional `skipHandled`; broadcast
runs before typed, since broadcast is where input capture decides consumption. This is the mechanism
for step 4's G4 work: `Camera.OnEvent` currently gates on `EditorState.ViewportFocused` /
`ViewportHovered`, which is a hand-rolled substitute for event consumption and a live instance of
render code reading editor globals. With ImGui ordered ahead of `Camera` and setting `Handled` when
`io.WantCaptureKeyboard`/`WantCaptureMouse`, `Camera` stops reading `EditorState` entirely. The
rewiring itself is still open.

Note the category enum already carried `Editor` and `Renderer` bits before this work -- the "gains a
render/editor category" framing was wrong. The categories existed; the mechanics above were what was
missing.

### Wiring (done)

| Event                                    | Published by                                                        | Handled by       | Effect                                        |
|------------------------------------------|---------------------------------------------------------------------|------------------|-----------------------------------------------|
| `SceneDirtyEvent`                        | `FileBrowserPanel` (visibility toggle)                              | `Renderer`       | `MarkTlasDirty` + `MarkAccumulatorDirty`      |
| `PathTracingAccumulatorInvalidatedEvent` | `RendererSettingsPanel` (mode switch, restart, FOV, lens, bounce)   | `Renderer`       | `MarkAccumulatorDirty`                        |
| `TonemapFilterChangedEvent`              | `RendererSettingsPanel` (Apply)                                     | `Renderer`       | stores operator, sets `pendingTonemapRebuild` |
| `PbrSoftShadowingChangedEvent`           | `RendererSettingsPanel` (Apply)                                     | `Renderer`       | stores flag, sets `pendingPbrRebuild`         |
| `SceneEntitySelectedEvent`               | `SceneOutlinerPanel`, `RendererSettingsPanel`, `SelectionSystem`    | `SelectionSystem`| writes `EditorState.SelectedEntity`, restarts accumulation |

`Renderer` subscribes in `SubscribeToEvents` (last thing in `Initialize`, so no handler can fire
against half-built pipelines) and disposes its tokens first thing in `Cleanup` -- the bus outlives
the renderer, and a queued intent drained afterwards would run against destroyed pipelines.
`SelectionSystem` holds its own token and disposes it with itself.

**Timing is unchanged.** `ProcessEvents` runs in Engine's Update pass, ahead of `DrawFrame`, which
is where `pendingPbrRebuild` / `pendingTonemapRebuild` / the accumulator flag are consumed. A panel
publishing while drawing frame N is applied at the top of frame N+1 -- exactly where the panel's
direct field writes landed before.

**On G1.** The panel keeps `_pbrRebuildPending` / `_tonemapRebuildPending`, but they now mean
something different and genuinely panel-local: *the user moved this control and has not hit Apply*,
which is what decides whether the Apply button shows and which value the widget displays. The staged
value rides in the event; the renderer holds the one rebuild flag. What is gone is the panel writing
`renderer.softShadowsEnabled`, `renderer.tonemapOperator`, `renderer.pendingPbrRebuild` and
`renderer.pendingTonemapRebuild` -- four of A1's raw-field flips, and the duplicated rebuild intent.

**Selection is now single-writer.** Nothing assigns `EditorState.SelectedEntity` except
`SelectionSystem.OnEntitySelected`; the outliner, the probe spawn and the viewport pick all publish.
Because `Editor` is an immediate category, the store updates before the publishing panel's next line
runs, so the outliner highlight and the Inspector binding still land on the click's own frame. The
handler no-ops when the selection did not actually change, so re-clicking the selected row no longer
costs an accumulator restart.

`InspectorPanel`'s ~35 `MarkAccumulatorDirty` / `MarkTlasDirty` calls are deliberately untouched:
the audit already rates that panel as fine, those are intent methods with one clear owner, and
turning them into events would be churn for no decoupling. The events exist for publishers that
should not be holding a `Renderer` at all.

## Pipeline ownership

- **Technique-private pipeline** (one core: `geometry`, `drawCull`, `lightCull` -> Deferred; the
  PT/RT/wavefront pipelines -> their cores): the core owns it.
- **Shared pipeline** (>1 core: tonemap, likely skybox): owned by the `FeatureHost` as a shared
  service; cores compose their own graph module around the one shared instance.

Tonemap is the canonical shared case, already factored correctly: one shared `TonemapPipeline`
(stateless, expensive-to-compile) wrapped by a per-core `TonemapModule` that graph-bakes the
HDR-input to that core's scene-color image. Do not duplicate the pipeline per core -- it clones the
wrong part (N boot compiles, N descriptor pools), fragments the single `pendingTonemapRebuild` path,
and splits global operator/exposure settings, while the per-core bit (HDR input) is already the
module's job. Duplicate instances only when a pipeline is technique-private *and* cheap *and*
carries no shared settings.

*(Done. Five deferred pipelines belong to `DeferredCore`, one tracer each to the four PT/RT cores.
The shared pair is a feature -- `SharedPipelines`, Order 5 -- rather than raw fields on the host,
because "the tonemap must exist and be initialized before any core builds the graph that composes
it" is an ordering guarantee, and `Order` is the machinery that already states ordering guarantees.
Cores take it as `INeedsFeature<ISharedPipelines>`. Both spec-constant rebuilds became bakes: the
tone curve on `SharedPipelines`, soft shadows on `DeferredCore`, each subscribing to its own event.
That deleted `RebuildPbrPipelines` / `RebuildTonemapPipeline` / `pendingPbrRebuild` /
`pendingTonemapRebuild` / `softShadowsEnabled` / `tonemapOperator` from the host, and left
`DrawFrame` with one `ServiceBakes()` where three rebuild paths used to be spelled out.)*

## Step 8 as built

**Selection** implements `IPreDrawFeature` + `IPostDrawFeature` + `IResizeFeature`, which is the
whole of what the host used to call by name (`ProcessPickRequest` / `RecordOutline` / `RebindMask`).
Order 100 puts it after every core, so "FinalColor exists by now" holds for the outline composite
without the host sequencing it. Pick moved from before the command buffer opens to the PreDraw pump,
which is later but strictly safer -- it now runs after the in-flight fence wait rather than before.

**IBL is a provider.** `IIblProvider` is deliberately three members: consumers of *bindable*
resources go through the registry and never name the producer, so the split-sum images are absent
from it. What is left is one scalar the shaders need as a uniform (`PrefilteredCubeMipLevels`) and
the two handles the probes reuse rather than duplicate (the prefilter kernel and the cube sampler).
Probes take it as `INeedsFeature<IIblProvider>` at Order 2, which is what dissolved the old
`new ReflectionProbeSystem(gpu, renderer, iblSystem)` argument -- `Order` now carries the "IBL is
fully built first" guarantee that the constructor parameter was standing in for.

Both features register their own descriptor bindings during `Initialize`, so `RegisterFeatureBindings`
is deleted outright and `RegisterSceneBindings` keeps only what the ResourceManager and GpuScene own.
This is why `FeatureHost.BuildAll` had to move *ahead* of the scene setup and the registry
cross-check in `Initialize`: features are providers now, and the check cannot run before they have
published. The five pipelines that read `Renderer.Ibl.prefilteredCubeMipLevels` read one narrow
`Renderer.PrefilteredCubeMipLevels` accessor instead, resolved through the interface at record time
so a rebake can never leave a stale copy.

**`SceneAS`** is the AS policy half, gated on ray query + acceleration structure, so on a device
without them it is never constructed and none of its buffers are allocated. It publishes `sceneTlas`
/ `sceneEntityInfo` / `sceneEmissiveTris` / `sceneEmissiveAlias` itself after every rebuild, which
is what let the host stop knowing they exist. Rebuilds are an `IBakeFeature`: the `tlasDirty` flag
and the `if (tlasDirty) OnSceneEntitiesChanged()` block at the top of `DrawFrame` are gone, replaced
by the bake pump, which gives the same "at most one rebuild per frame however many slider ticks
landed" guarantee with none of the host-side bookkeeping. `Renderer_Ray_Query.cs` became
`Renderer_SceneAsBridge.cs`: ~980 lines down to a handful of null-tolerant forwarders for the editor
panels and the path-trace pipelines that still name the Renderer.

Two things step 8 did *not* do, on purpose:

- **`SceneAS` still walks the scene rather than consuming GpuScene's extracted renderable set.** It
  already consumes GpuScene for the parts that matter to identity and correctness (`SyncRenderables`,
  `BeginTransforms`, `WorldOf`, `Register`), but the cluster gather re-walks entities because
  `RenderableInputGpu` does not carry the handle slot or the mesh pointer the emissive collection
  needs. Closing that means widening GpuScene's read surface, which changes what geometry ends up in
  the BVH -- a pixel-affecting change that should not ride along with a file move. Moving the code
  and changing what it reads in one step would make any regression unbisectable.
- **`INeedsHost` is still on nine of the eleven features.** What remains behind it is a small,
  legible set: `renderTargets`, `gpuScene`, `Scene`, `MarkAccumulatorDirty`/`AccumulatorDirty`, and
  `gfx`. Those are the next channels to grow, not more feature migrations. The manifest tags each
  remaining one `NEEDS-HOST`, so the debt reports itself at every boot.

The boot-time registry cross-check needed one change to survive this step: `ReflectedPrograms` used
to walk only the Renderer's own `PipelineBase` fields, so migrating a pipeline into its core would
have silently dropped it out of the check. It now walks the host *and* every built feature. The
proof it still works is that the consumed-parameter count is unchanged across the whole step.

## What already exists (not starting cold)

- `IGraphModule<In,Out>` modules (`DeferredModule` et al.): the target discipline realized -- typed
  `Inputs`/`Outputs` records, collaborators injected by type, `Func<>` for per-frame params, zero
  `Renderer` reach. The north star; propagate it outward.
- `DescriptorRegistry` named broker (incl. `RegisterTlas`): the resource channel, built; IBL, the
  probes and SceneAS all publish through it as of step 8.
- Cores self-register + list-dispose: the lifecycle model, ready to generalize.
- `MarkAccumulatorDirty` / `MarkTlasDirty` / `RequestCoreIndex` and `InspectorPanel`: the intent
  style the whole UI should adopt.

## Migration order

Ordered so early steps shrink later ones. Each step ships green.

0. **Trivial independent fixes** (near-zero risk, no dependencies): G3 (`Scene.Cam` and
   `Renderer_Ray_Query.PreviousCount` static -> instance); G2 (`Engine.renderer` ->
   `{ get; private set; }`). Removes a latent multi-scene bug now.
1. **`RenderConfig` static** (A3). Extract MAX_*/TILE_*. Deletes ~14 pipeline reaches + all of
   GpuScene's constant reaches, shrinking step 2.
2. **`GpuContext` + RHI consumers** (A2). Define the record; `PipelineBase` takes it as a single ctor
   arg; migrate `ResourceManager` and `BcEncoder` off `Renderer` (and move `BcEncoder` *ownership*
   off `GraphicsDevice` into the texture path -- device enables the BC capability, the resource layer
   owns the tooling). Pipelines stop taking `Renderer`.
3. **Events channel.** *(done)* `EventBus` rewrite -- dead `EventDispatcher`/`EventSystem`/`Clone`
   deleted, reentrancy + category-matching + threading fixed, per-category delivery, thread-safe
   publish with main-thread drain, typed `Subscribe<T>` with disposable tokens, `NextAsync<T>`,
   ordering + `Handled`. Renderer event types declared (`SceneDirtyEvent`,
   `TonemapFilterChangedEvent`, `PathTracingAccumulatorInvalidatedEvent`,
   `SceneEntitySelectedEvent`, `PbrSoftShadowingChangedEvent`). Foundational: makes steps 4 and 9
   collapse into publish/subscribe instead of interim hacks.
4. **A/D quick wins on events** (G1, A1-partial, G4). *(mostly done -- see Wiring above.)* The
   rebuild-flag duplication is gone (panel stages, event carries the value, renderer owns the one
   flag) and selection flows as an event with a single writer. Still open: render code reading the
   remaining `EditorState` globals (skybox / IBL intensities via `RenderConfig` or frame settings),
   and `Camera` gating on `ViewportFocused` / `ViewportHovered` instead of ImGui setting `Handled`.
5. **GpuScene single source + `RenderView` extraction.** *(Done, except `ShadowEntityInfo`.)* One
   extraction phase in `DrawFrame`; `FrameContext` promoted to `RenderView` carrying the extraction
   outputs; cores + `DrawCullPipeline` consume it; the `UpdateLights`/`UpdateMaterials`/`LightCount`/
   `GetLightStorageBuffer` forwarders on `Renderer` are gone (A5 is now only `SyncRenderables` and
   the AS path). **`ShadowEntityInfo` authorship deliberately stayed put:** the packing is
   cluster-block layout (`baseSlot`, contiguous per-cluster geometry ranges), which is AS *policy*,
   not scene data, and its buffer + capacity growth live in the AS path. Moving it now would move it
   twice -- it belongs in step 8 where `SceneAS` is built. What step 5 owed step 8 is delivered
   anyway: the world cache is populated before the AS gather and `RebuildTlas` already reads it.
6. **AS verbs -> `GraphicsDevice`.** *(Done.)* `AsDevice` facet reached as `gfx.As.*`; the dispatch
   table + scratch alignment load with the device; `Renderer_Ray_Query` keeps only policy. Dropped
   two duplicated buffer-address helpers on the way. What is left there is now the `SceneAS`
   feature's body, waiting on step 7's lifecycle.
7. **Feature lifecycle** (A4-enabling). *(Done, except the bridge accessors -- see below.)*
   `IRenderFeature` + phase interfaces + `INeedsGpu` / `INeedsFeature<T>` / `INeedsHost` +
   `FeatureHost` (3-phase `BuildAll`, phase pumps, reverse dispose, `Dump`) + `FeatureCatalog` +
   compiler-enforced `[ModuleInitializer]` descriptors. All six cores migrated onto it;
   `Renderer.Initialize` names no feature, `Cleanup` is one `_features.Dispose()`, and
   `RegisterCore` / `_renderCores` are gone. Verified by boot: the manifest above is real output.

   Bridge accessors were deliberately left for step 8, since until ownership actually moved an
   accessor would have forwarded a field to itself. They landed with each subsystem as it became a
   feature; they are the null-tolerant properties on `Renderer` and in `Renderer_SceneAsBridge.cs`.

   Two pieces of the machinery had no implementer at the end of step 7 and were therefore
   unexercised: `INeedsFeature<T>` resolution and the `Bake` / `PreDraw` / `PostDraw` pumps (the
   calls were in `DrawFrame`, walking empty lists). Step 8 filled all four -- two contracts injected
   across six consumers, three bakes, one pre-draw and one post-draw -- and none of them needed a
   change to the machinery, which is the result the split was betting on.
8. **Migrate features onto the lifecycle** (A4, A5, IBL, SceneAS). **Done** -- eleven features now
   build, in Order: IBL(1), probes(2), SceneAS(3), SharedPipelines(5), the six cores(10-60),
   Selection(100). `Renderer` owns no pipeline and no subsystem field; `Initialize` names no
   feature; `DrawFrame` has no per-subsystem call left. See "Step 8 as built" below for what each
   piece cost and the two carve-outs.
9. **Settings-panel command/event facade** (A1-finish). Last, highest surface; by now most targets
   are already events/intents.
10. **(Optional, orthogonal)** G5 consistency pass: `VkCheck` / `VulkanException` in place of raw
    `throw new Exception`.