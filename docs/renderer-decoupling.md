# Renderer Decoupling

Design research for the ongoing effort to strip the `Renderer` god object down to a thin
composition root, so that adding a feature does not require editing `Renderer`, `DrawFrame`, or
`Cleanup`, and so that systems depend only on what they actually use.

This consolidates the target architecture. It is a plan, not yet fully implemented; see the
migration order at the end for current status.

## The core problem

`Renderer` is passed wholesale into nearly everything (pipelines via `PipelineBase(Renderer)`,
cores via `Core(Renderer host)`, `ResourceManager`, UI panels). A consumer that takes `Renderer`
depends on the entire type to reach one member, and the field/method surface it pokes is
unbounded. The fix is to route each *kind* of need through its own narrow channel, and to give
features a lifecycle that the frame skeleton drives generically.

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

Five anti-patterns, each with a fix:

- **A. UI mutating renderer internals** (`RendererSettingsPanel`). Flips raw fields
  (`pendingTonemapRebuild`, `tonemapOperator`, `softShadowsEnabled`, `renderMode`) and pokes
  individual pipelines. Fix: an intent/command facade (`RequestTonemapRebuild()`,
  `SetSoftShadows(bool)`). `InspectorPanel` is the model to copy: only `MarkAccumulatorDirty` /
  `MarkTlasDirty`.
- **B. Systems that only need the RHI** (`ResourceManager`). Every reach is device territory
  (`CreateBuffer`, `UploadBufferData`, `GpuMemoryAllocator`, `device`, `descriptorPool`,
  `WriteBindlessTexture`). Fix: inject `GpuContext`; drop `Renderer`.
- **C. Constants sourced off Renderer** (`GpuScene`, every pipeline). `MAX_CONCURRENT_FRAMES`,
  `MAX_INSTANCES/MATERIALS/LIGHTS`, `TILE_SIZE`, `MAX_TILE_COUNT`. Fix: `static RenderConfig`.
- **D. Renderer as an ad-hoc pipeline registry** (`DeferredCore`). Fishes `host.transparentPipeline`,
  `host.skyboxPipeline`, `host.geometryPipeline` etc., yet its own `DeferredModule` already takes
  those by explicit injection. Fix: the core owns/constructs its pipelines and hands them to the
  module; never re-reads them from `host`.
- **E. Behavior reach-through** (`UpdateLights`, `UpdateMaterials`). A subsystem's behavior parked
  on Renderer. Fix: a feature with a contract interface, resolved like IBL.

## Target: four dependency channels, none of them `Renderer`

Everything a system might grab off `Renderer` sorts into exactly one channel. Keeping them
separate is what stops the god object reforming.

| Need                                             | Channel             | Mechanism                                    |
|--------------------------------------------------|---------------------|----------------------------------------------|
| Device verbs (buffers, images, single-time cmd)  | `GpuContext`        | injected; one small fixed record             |
| Bindable GPU resources (IBL cubes, g-buffer)     | `DescriptorRegistry`| named provider/consumer broker (exists)      |
| Feature collaborators (IBL scalars, probes)      | `FeatureHost`       | `Features.Get<IIblProvider>()`               |
| Constants (MAX_*, TILE_SIZE)                      | `RenderConfig`      | static; no injection                         |
| Per-frame state (extent, camera, frame index)    | `RenderFrame`       | arrives at record time; never a ctor dep     |

`GpuContext` is deliberately capped and never grows:

```csharp
public readonly record struct GpuContext(
    GraphicsDevice Gfx, DescriptorRegistry Registry, ShaderLibrary Shaders);
```

The discipline: nothing frame-scoped or feature-scoped goes in it. A pipeline ctor takes only
`GpuContext` (plus `FeatureHost` if it has collaborators). Any temptation to add a fifth thing is
frame state (to record time), a constant (to `RenderConfig`), or a feature (to `Features.Get<T>()`).

### Feature-to-feature access

Consumers of a feature's *bindable resources* go through `DescriptorRegistry` and never name the
producer: IBL calls `RegisterImage("iblIrradiance", ...)` on bake (re-register on rebake is a
free hot-swap via the per-frame `BeginFrame` rewrite); a consumer declares the binding name. For
non-bindable data (e.g. `prefilteredCubeMipLevels`), resolve a contract interface from the
`FeatureHost` and cache the *provider*, reading the value at record time so rebakes never leave a
stale copy. Avoid `Renderer.X`; if an ambient accessor is wanted, put a static holder on a
dedicated `Gpu` type (never on `Renderer`) restricted to the RHI trio.

## Feature lifecycle

Cores already prove two thirds of the pattern: they self-register (`RegisterCore`) and are
list-disposed. Generalize that to all ambient features so `DrawFrame`/`Cleanup` stop changing per
feature.

```csharp
interface IRenderFeature : IDisposable { string Name { get; } void Initialize(in FeatureContext ctx); }
interface IBakeFeature     { void Bake(in BakeContext ctx); }      // IBL, probes
interface IResizeFeature   { void Resize(Extent2D extent); }
interface IPreDrawFeature  { void PreDraw(in FrameContext frame); }// selection pick
interface IPostDrawFeature { void PostDraw(in RenderFrame frame); }// selection outline
```

A `FeatureHost` owns the master list, pre-buckets by capability at registration (no per-frame
casts), and drives the phases from fixed pump points in `DrawFrame`
(`PreDraw` -> active core `Render` -> `PostDraw`) plus `Resize` and reverse-order `Dispose`. Bake
is request-driven, not per-frame: model it like the existing `pendingPbrRebuild` / `tlasDirty`
servicing at the top of `DrawFrame`.

Registration lives in a single `FeatureCatalog` file, not in `Renderer`. It stays explicit and
greppable and keeps device-capability gates (`if (rtPipeline != null) ...`). Adding a feature
edits the catalog only. Prefer this over `[ModuleInitializer]`/assembly-scan auto-discovery, whose
implicit init order and attribute-encoded gating fight this engine's explicit ethos.

`IRenderCore` stays a specialization (exactly-one-active, mode-switched, produces the draw); it can
implement `IRenderFeature` so it shares the dispose/registration path.

## Pipeline ownership

- **Technique-private pipeline** (one core uses it: `geometry`, `drawCull`, `lightCull` -> Deferred;
  the PT/RT/wavefront pipelines -> their cores): the core owns it.
- **Shared pipeline** (>1 core: tonemap, likely skybox): owned by the `FeatureHost` as a shared
  service, not by any single core. Cores resolve it and compose their own graph module around it.

Tonemap is the canonical shared case and is already factored correctly: one shared
`TonemapPipeline` (the stateless, expensive-to-compile part) wrapped by a per-core `TonemapModule`
that graph-bakes the HDR-input to that core's scene-color image. Do **not** duplicate the pipeline
per core: it would clone the wrong part (N boot compiles, N descriptor pools) and fragment the
single `pendingTonemapRebuild` path and the global operator/exposure settings, while the genuinely
per-core bit (HDR input) is already the module's job. Duplicate instances only when a pipeline is
technique-private *and* cheap *and* carries no shared settings.

## What already exists

- `IGraphModule<In,Out>` modules (`DeferredModule` et al.): the target discipline realized -- typed
  `Inputs`/`Outputs` records, collaborators injected by type, `Func<>` for per-frame params, zero
  `Renderer` reach. The north star; propagate it outward.
- `DescriptorRegistry` named broker: channel 2, built; IBL not yet publishing through it.
- Cores self-register + list-dispose: the lifecycle model, ready to generalize.
- `MarkAccumulatorDirty` / `MarkTlasDirty` / `RequestCoreIndex`: the intent-command style the UI
  and settings facade should adopt everywhere.

## Migration order

Smallest blast radius first; each step ships green.

1. `RenderConfig` static -- extract MAX_*/TILE_*. Deletes ~14 reaches across every pipeline and all
   of `GpuScene`'s constant reaches. Zero risk.
2. `GpuContext` record -- convert `PipelineBase` and `ResourceManager` off `Renderer`.
3. `BcEncoder` ownership move -- off `GraphicsDevice` into the texture path (device enables the
   capability via the BC feature flag; the resource layer owns the encoder tooling).
4. IBL-as-provider -- publish images through `DescriptorRegistry`, expose scalars via
   `IIblProvider`; delete `Renderer.Ibl.*` in both PBR pipelines.
5. `IRenderFeature` + phase interfaces + `FeatureCatalog` -- lift `SelectionSystem` first (it owns
   the pre/post hooks, so it exercises the phase design end-to-end), then IBL/probes onto
   `IBakeFeature`.
6. `DeferredCore` owns its pipelines -- stop reading siblings off `host`; feed `DeferredModule`
   directly.
7. `RendererSettingsPanel` -> command facade -- last, highest surface; by then most targets are
   already intent methods.