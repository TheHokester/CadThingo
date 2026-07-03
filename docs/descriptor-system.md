# Descriptor System — Reflection-Driven Sets & Registry

A redesign of descriptor management around three pillars: **Slang runtime reflection**
(compile + reflect `.slang` in-process, disk-cached), a **DescriptorRegistry** owning a
unified scene set and feature-persistent sets matched to shaders **by parameter name**,
and **graph-owned pass sets** baked at `FrameGraph.Compile`. Companion to
`render-graph.md` (the FrameGraph this integrates with) and `renderer-refactor.md`
(L1 `GraphicsDevice` / L2 `GpuScene` / L3 cores vocabulary used throughout).

## Contents

1. [Problem statement](#1-problem-statement)
2. [Target model](#2-target-model)
3. [Slang runtime integration](#3-slang-runtime-integration)
4. [Shader cache](#4-shader-cache)
5. [SceneBindings.slang and the DescriptorRegistry](#5-scenebindingsslang-and-the-descriptorregistry)
6. [FrameGraph integration - graph-owned pass sets](#6-framegraph-integration--graph-owned-pass-sets)
7. [The (0,0) constant slot - dynamic-UBO arena](#7-the-00-constant-slot--dynamic-ubo-arena)
8. [Bindless / UpdateAfterBind details](#8-bindless--updateafterbind-details)
9. [Pipeline layouts and the PipelineBase diet](#9-pipeline-layouts-and-the-pipelinebase-diet)
10. [Spec constants and push constants](#10-spec-constants-and-push-constants)
11. [Validation and debug](#11-validation-and-debug)
12. [Migration phasing](#12-migration-phasing)
13. [Risks and open questions](#13-risks-and-open-questions)

---

## 1. Problem statement

Where descriptor handling stands today:

- **Every pipeline hand-writes everything.** `PipelineBase`
  (`Renderer/Pipelines/Pipelines.cs`) forces each concrete pipeline through
  `CreateDescriptorSetLayouts` / `CreateDescriptorSets` / `WriteDescriptors`, with
  `DescriptorSets[setIdx][frame]` and `OwnedDescriptorSetLayoutIndices` bookkeeping.
- **The scene set is copy-pasted.** A ~12-binding lighting/scene set 0 (frame UBO, lights
  SSBO, TLAS, tile buffers, IBL cubes, probe buffers) is duplicated across
  `PbrDeferredPipeline`, `TransparentPipeline`, `PTComputePipeline`, `WavefrontPTPipeline` -
  the wavefront pipeline literally documents its sets as "verbatim from PTComputePipeline".
- **Cross-pipeline wiring is manual and ordering-sensitive.** A TLAS rebuild fans out to
  seven `WriteTlasDescriptor` calls from `Renderer_Core` / `Renderer_Ray_Query`; every
  resize re-wires g-buffer samplers via `WriteGBufferDescriptors` from `DeferredCore`;
  tile buffers, shadow-alpha buffers, and probe buffers each have their own phase-separated
  `WriteXxx` method that must be called at exactly the right point in init order.
- **Bindless lives at inconsistent set indices.** `ResourceManager` owns the bindless set
  (materials, instances, `Texture2D[]`, samplers) but shaders borrow it at set 1, 2, or 3
  depending on the pass.
- **Nothing is reflected.** Bindings are explicit `[[vk::binding(b,s)]]` in Slang, and the
  matching C# layouts are hand-transcribed; the two can silently drift, and set/binding
  conventions differ per shader (IBL cubes at set 0 in PBR, set 3 in the path tracer, etc.).
- **The FrameGraph stops at physical handles.** Passes get `ImageView`/`Buffer` via
  `PassResources`, but descriptor sets for those resources are still pipeline-owned and
  rewired by hand on every graph rebuild.

## 2. Target model

| Set | Contents | Lifetime / bind frequency | Owner |
|---|---|---|---|
| **0** | Scene set: materials, instances, bindless `textures[]`, samplers, TLAS, lights, entityInfo, global VB/IB, emissive sampling tables. Binding **(0,0) reserved**: per-pass constant slot (dynamic UBO into a per-frame arena, §7) | bound once per command buffer per bind point; most stable | `DescriptorRegistry` |
| **1** | Pass-local graph resources (g-buffer views, tile buffers, accumulator, SoA queues declared as graph buffers, ...) | baked at `FrameGraph.Compile`; rewritten only on recompile/resize | FrameGraph |
| **2+** | Feature-persistent sets: `ibl` at set 2 (IBL cubes + probe buffers), `wavefrontPT` at set 3 (SoA working set), future groups | feature lifetime | `DescriptorRegistry` (named groups) |

IBL deliberately stays out of set 0 - it is renderer-technique data, not scene data, and
several passes never touch it.

Matching is **name-based**: Slang reflection yields parameter name -> (set, binding, type);
the registry and the graph resolve those names to real resources; a startup validation pass
fails fast on anything unresolved or type-mismatched. The shader parameter name is the
contract.

Frequency-ordered sets minimize rebinds: `vkCmdBindDescriptorSets` disturbs compatibility
only for sets *above* the changed index, so the most stable set sits lowest.

Out of scope (stay manual): IBL bake kernels and `BcEncoder` (offline-ish one-shot
dispatches with private sets) and ImGui (own tiny set). Revisit if they become graph passes.

## 3. Slang runtime integration

### 3.1 What we need from Slang

At startup (cold path only - §4): create a session with search paths, `#define`s
(`USE_SER`), and capabilities (`spvRayQueryKHR` / `spvRayTracingKHR` /
`spvShaderInvocationReorderNV`); load modules; compile entry points to SPIR-V; walk
reflection (parameters -> name/set/binding/type, push constants, spec constants, compute
thread-group sizes); and reflect the `SceneBindings` **module without an entry point**.

All of this exists in the modern COM-lite API (verified against
`%VULKAN_SDK%\Include\slang\slang.h`, Slang 2026.1 - the Vulkan SDK has bundled Slang
since 1.3.296):

- `slang_createGlobalSession2(&desc, &globalSession)` - a plain C export.
- `IGlobalSession::createSession(SessionDesc, ISession**)`. `SessionDesc` carries
  `TargetDesc[]` (format `SLANG_SPIRV`, `profile = findProfile("spirv_1_6")`),
  `searchPaths`, `preprocessorMacros` (`PreprocessorMacroDesc[]`), and
  `compilerOptionEntries`. Capabilities go in as
  `CompilerOptionEntry { name = CompilerOptionName.Capability, intValue0 = globalSession.findCapability("spvRayQueryKHR") }`.
- `ISession::loadModule("PBR")` -> `IModule`; `IModule::findEntryPointByName` ->
  `IEntryPoint`; `ISession::createCompositeComponentType([module, entryPoints...])` ->
  `IComponentType`; `link()`; `getEntryPointCode(i, 0)` -> SPIR-V blob per entry point.
- Reflection: `IComponentType::getLayout(0)` -> `slang::ProgramLayout`. Because `IModule`
  **is** an `IComponentType`, `sceneModule.getLayout(0)` reflects the module's global
  parameters with no entry point - exactly what the canonical scene layout needs. Walk
  `getParameterCount()/getParameterByIndex()` ->
  `VariableLayoutReflection::getName()/getBindingIndex()/getBindingSpace()` plus the
  `TypeLayoutReflection` descriptor-range APIs (`SLANG_BINDING_TYPE_*` -> `VkDescriptorType`
  mapping table).
- Per-entry used-binding info for validation: post-compile `getEntryPointMetadata()` ->
  `IMetadata::isParameterLocationUsed()` - feeds the "who consumes what" debug dump (§11).
- Sessions are cached compile contexts; one session per `#define` combination (today only
  `USE_SER=1` needs a second session). SER becomes a *runtime* decision - device supports
  it, so compile/load the SER variant - instead of two build outputs.

### 3.2 Interop route

Three candidates were evaluated:

| Route | What it is | Verdict |
|---|---|---|
| **Hand-rolled COM-lite interop** | ~600 lines of `unsafe` C# in one file pair | **Adopt.** Details below. |
| Slang.Sdk (NuGet, alpha) | C++/CLI wrapper with a Session/Module/Reflection object model | Reject for production: explicitly experimental, C++/CLI (fragile toolchain coupling), bundles its own Slang so we cannot ride the Vulkan SDK's dll, third-party release cadence. Useful as a reference for vtable layouts. |
| Slangc.NET (NuGet) | slangc-CLI-shaped: `byte[] Compile(args)` | Reject as primary: compiler-invocation shape, no in-proc reflection object model. Second-choice fallback. |

**The hand-rolled route**, in `VulkanEngine/Renderer/Shaders/SlangInterop.cs` (L1):

1. **Loading:** `NativeLibrary.Load(Path.Combine(VULKAN_SDK, "Bin", "slang.dll"))` via a
   `DllImportResolver` registered for the `"slang"` module name.
2. **COM-lite calls:** Slang interfaces derive from `ISlangUnknown`
   (IUnknown-compatible; `SlangResult` is HRESULT-compatible int32). Virtual methods are
   called by indexing the vtable with C# function pointers - the same pattern Silk.NET
   uses for D3D:

   ```csharp
   readonly unsafe struct Session(void* p)
   {
       void** Vtbl => *(void***)p;
       // slot order = declaration order in slang.h; slots 0-2 are ISlangUnknown
       public void* LoadModule(byte* name, out void* diagnostics)
           => ((delegate* unmanaged[MemberFunction]<void*, byte*, void**, void*>)Vtbl[4])(p, name, ...);
   }
   ```

   .NET's `unmanaged[MemberFunction]` calling convention matches the MSVC virtual-call
   ABI. Only ~6 interfaces and ~20 slots are needed: `IGlobalSession` (createSession,
   findProfile, findCapability), `ISession` (loadModule, createCompositeComponentType),
   `IModule` (findEntryPointByName, getDependencyFileCount/Path, + IComponentType slots),
   `IComponentType` (getLayout, link, getEntryPointCode, getEntryPointMetadata),
   `ISlangBlob` (getBufferPointer/getBufferSize).
3. **Reflection walking avoids vtables entirely:** every reflection query is a flat C
   export (`spReflection_GetParameterCount`, `spReflectionVariableLayout_GetName`,
   `spReflectionTypeLayout_GetDescriptorSetCount`, ...) reachable with plain
   `[DllImport("slang")]`. Verified: `slang.h`'s reflection "classes" are inline wrappers
   calling exactly these exports, so they are the load-bearing ABI for every C++ client.
   They are listed in `slang-deprecated.h`, which is a real (low) risk - see §13.

Risks and mitigations:

- *Vtable drift across Slang versions.* Slang appends rather than reorders, but pin
  anyway: record the dll's `FileVersionInfo` and refuse versions outside a tested range;
  a startup smoke test compiles a 5-line shader and checks the SPIR-V magic plus one
  reflected binding before trusting anything.
- *Struct marshalling.* `SessionDesc` / `TargetDesc` / `CompilerOptionEntry` all begin
  with a `size_t structureSize` for versioning - mirror them as blittable structs with a
  `nuint` first field.
- *Thread safety.* The global session is documented not thread-safe. Compile on one
  dedicated thread; if cold-start time demands parallelism later, one global session per
  worker thread.

**Fallback if interop proves fragile:** shell out to `slangc.exe` per cache miss with
`-reflection-json` and parse the JSON into the same `ShaderReflectionData` (§4.2). Slower
cold starts, zero ABI risk, identical downstream design - the interop is quarantined
behind `IShaderCompiler` precisely so this swap is a one-file change.

### 3.3 API sketch (L1 service)

```csharp
// GraphicsDevice-owned. slang.dll is lazy-loaded: warm starts never touch it.
public sealed class ShaderLibrary : IDisposable
{
    public ShaderProgram GetProgram(in ShaderProgramDesc desc);   // cache-or-compile
    public ShaderReflectionData ReflectModule(string module);     // SceneBindings, IblBindings, ...
    public event Action<ShaderProgram>? ProgramReloaded;          // hot-reload hook (future)
}

public readonly record struct ShaderProgramDesc(
    string   Module,        // "Deferred/PBR" -> Features/Deferred/Kernels/PBR.slang
    string[] EntryPoints,   // ["main"], or the RT entry list (raygen/miss/chit/ahit)
    string[] Defines,       // ["USE_SER=1"]
    string[] Capabilities); // ["spvRayQueryKHR"]

public sealed class ShaderProgram
{
    public ReadOnlyMemory<byte>  Spirv(int entryIndex);
    public ShaderReflectionData  Reflection { get; }      // merged over entry points
    public PipelineLayout        Layout { get; }          // via PipelineLayoutCache (section 9)
    public DescriptorSetLayout   PassSetLayout { get; }   // set 1, from reflection (section 6)
}
```

## 4. Shader cache

### 4.1 On-disk layout and key

```
Assets/ShaderCache/                  (gitignored)
    PBR-a41f09c3d2e488b1.spv         one per entry point (PBR.0-<hash>.spv when multi-entry)
    PBR-a41f09c3d2e488b1.refl        engine-format reflection + dependency manifest
```

**Key** = SHA-256 (truncated to 16 hex chars) over:

1. `slang.dll` identity - `FileVersionInfo` + file size + mtime, readable **without
   loading the dll** (this keeps warm starts slang-free);
2. source bytes of the module;
3. bytes of every transitive import (dependency closure);
4. `ShaderProgramDesc` fields (entry points, defines, capabilities), optimization level,
   target profile string;
5. a `CacheFormatVersion` constant for the `.refl` schema.

**Dependency closure bootstrap:** imports are only knowable post-compile
(`IModule::getDependencyFileCount/getDependencyFilePath`), so the `.refl` stores a *dep
manifest* - the path list from the last compile. Warm start hashes source +
manifest-listed files -> key -> hit or miss. If any listed file changed the key changes,
which is a miss, which produces a fresh manifest (the classic ninja-depfile scheme). The
first-ever run is always a miss, so there is no chicken-and-egg.

**Warm start does not load slang.dll at all.** SPIR-V plus the engine-format reflection
blob is sufficient to build every DSL, pipeline layout, and pipeline. slang.dll loads
lazily on the first cache miss or hot-reload request.

Stale entries are garbage-collected dumbly on startup (delete files untouched for 30 days).

### 4.2 Serialized reflection form (`.refl`)

Compact little-endian binary: one versioned header + tables, mirrored by blittable record
structs read with `MemoryMarshal`:

```csharp
public sealed class ShaderReflectionData
{
    public BindingDesc[]    Bindings;      // name, set, binding, VkDescriptorType,
                                           // count (0 = unbounded), ShaderStageFlags
    public PushRangeDesc[]  PushConstants; // stageMask, offset, size, structName
    public SpecConstDesc[]  SpecConstants; // name, constantId, byteSize, default
    public EntryPointDesc[] EntryPoints;   // name, stage, threadGroupSize (compute)
    public string[]         Dependencies;  // dep manifest, paths relative to shader root
}
```

Names are the Slang global-parameter names - the registry's matching currency.
`VkDescriptorType` is resolved at compile time from `SLANG_BINDING_TYPE_*`, so the loader
needs no Slang enum knowledge.

### 4.3 Coexistence with `Shaders.targets`

During migration both paths run: `Shaders.targets` keeps compiling *unmigrated* shaders to
`Assets/Shaders/**.spv`; migrated shaders are removed from the `Shader`/`Kernel` item
groups one by one. The runtime session's search paths are the same directories the target
reads today (`VulkanEngine/Shaders`, `Renderer/Features/**/Kernels`), and `.slang` sources
are copied to output (`CopyToOutputDirectory=PreserveNewest`) so published builds can
cold-compile. Phase D removes the target for entry shaders entirely; shared modules remain
build inputs only for as long as any unmigrated shader imports them.

## 5. SceneBindings.slang and the DescriptorRegistry

### 5.1 The canonical scene module (source of truth)

`VulkanEngine/Shaders/SceneBindings.slang` - plain globals, explicit bindings, imported by
every shader. One module, reflected once by C#, is the single source of truth for the
scene `VkDescriptorSetLayout`; shaders are consistent by construction because they all
import the same declarations.

```slang
module SceneBindings;
import CommonTypes;

// (0,0) is deliberately NOT declared here. It is the reserved per-pass constant slot:
// each entry shader declares its own
//     [[vk::binding(0, 0)]] ConstantBuffer<MyPassParams> gPass;
// and the canonical VkDescriptorSetLayout declares binding 0 as
// UNIFORM_BUFFER_DYNAMIC. SPIR-V does not distinguish dynamic UBOs, so the shader's
// plain ConstantBuffer stays layout-compatible. See section 7.

public [[vk::binding( 1, 0)]] StructuredBuffer<PbrMaterial>      sceneMaterials;
public [[vk::binding( 2, 0)]] StructuredBuffer<InstanceData>     sceneInstances;
public [[vk::binding( 3, 0)]] StructuredBuffer<PbrLight>         sceneLights;
public [[vk::binding( 4, 0)]] StructuredBuffer<ShadowEntityInfo> sceneEntityInfo;
public [[vk::binding( 5, 0)]] RaytracingAccelerationStructure    sceneTlas;
public [[vk::binding( 6, 0)]] ByteAddressBuffer                  sceneVertices;
public [[vk::binding( 7, 0)]] ByteAddressBuffer                  sceneIndices;
public [[vk::binding( 8, 0)]] StructuredBuffer<EmissiveTri>      sceneEmissiveTris;
public [[vk::binding( 9, 0)]] StructuredBuffer<AliasEntry>       sceneEmissiveAlias;
public [[vk::binding(10, 0)]] SamplerState                       sceneSamplers[16];
// Unbounded array LAST (highest binding) -- required for variable-count.
public [[vk::binding(11, 0)]] Texture2D                          sceneTextures[];
```

C# reflects this module (`ShaderLibrary.ReflectModule("SceneBindings")`, no entry point)
and injects binding 0 (dynamic UBO, `ShaderStageFlags.All`) as the single hand-authored
binding. Stage flags for everything: `All` - per-binding stage narrowing buys nothing
measurable and complicates sharing.

Feature modules follow the same pattern at pinned set indices:

```slang
module IblBindings;                       // registered as group "ibl", set 2
public [[vk::binding(0, 2)]] SamplerCube                    iblIrradiance;
public [[vk::binding(1, 2)]] SamplerCube                    iblPrefiltered;
public [[vk::binding(2, 2)]] Sampler2D                      iblBrdfLut;
public [[vk::binding(3, 2)]] SamplerCube                    iblEnvCube;
public [[vk::binding(4, 2)]] SamplerCubeArray               probeCubes;
public [[vk::binding(5, 2)]] StructuredBuffer<ProbeRecord>  probeRecords;
public [[vk::binding(6, 2)]] StructuredBuffer<ProbeClusterRange> probeClusters;
public [[vk::binding(7, 2)]] StructuredBuffer<uint>         probeIndexList;
```

`WavefrontBindings.slang` (today's 25-binding SoA working set at set 4) moves verbatim to
group `"wavefrontPT"` at **set 3**. A pipeline may consume multiple feature groups as long
as their pinned set indices do not collide (wavefront kernels use `ibl`@2 +
`wavefrontPT`@3); the registry validates collisions at startup. Set-index constants live in
one C# class: `ShaderSets.Scene = 0, Pass = 1`, features >= 2.

### 5.2 DescriptorRegistry (L2)

```csharp
public sealed unsafe class DescriptorRegistry : IDisposable
{
    // ---- canonical layouts + set instances ---------------------------------
    public DescriptorSetLayout SceneSetLayout { get; }        // from SceneBindings reflection
    public DescriptorSet       SceneSet(uint frame);          // MAX_CONCURRENT_FRAMES copies
    public DescriptorSetLayout FeatureLayout(string group);   // "ibl", "wavefrontPT", ...
    public DescriptorSet       FeatureSet(string group, uint frame);
    public void RegisterFeatureGroup(string group, ShaderReflectionData moduleReflection);

    // ---- name -> resource registration (name = Slang global param name) ----
    // Re-registering an existing name = handle change: queues a rewrite applied to each
    // per-frame set at the top of that frame (fence-safe).
    public void RegisterBuffer(string name, Buffer buf, ulong offset = 0, ulong range = Vk.WholeSize);
    public void RegisterBufferPerFrame(string name, Buffer[] perFrame, ulong range = Vk.WholeSize);
    public void RegisterImage(string name, ImageView view, ImageLayout layout, Sampler? sampler = null);
    public void RegisterImagePerFrame(string name, ImageView[] perFrame, ImageLayout layout);
    public void RegisterSampler(string name, Sampler s, int arrayIndex = 0);
    public void RegisterTlas(string name, AccelerationStructureKHR tlas);

    // ---- bindless texture table (absorbed from ResourceManager) ------------
    public int  RegisterBindlessTexture(Texture tex);          // stable index, UpdateAfterBind write
    public void UnregisterBindlessTexture(int index, Texture fallback);

    // ---- per-pass constant arena (section 7) --------------------------------
    public PassConstantArena ConstantArena { get; }

    // ---- lifecycle ----------------------------------------------------------
    public void BeginFrame(uint frame);   // applies queued rewrites for this frame's sets
    public void Validate(IEnumerable<ShaderProgram> allPrograms);   // section 11
    public string DumpBindings();         // provider/consumer report
}
```

Key behaviors:

- **Handle change = one call.** A TLAS rebuild becomes
  `registry.RegisterTlas("sceneTlas", tlas)` - replacing the seven-call
  `WriteTlasDescriptor` fan-out. Global VB/IB reallocation on mesh-pool growth is the same
  path (`RegisterBuffer("sceneVertices", newVb)`).
- **Fence-safe rewrites.** Non-UpdateAfterBind bindings (TLAS, SSBOs) cannot be written
  while their set may be referenced by in-flight work. Rewrites are queued and applied per
  frame slot in `BeginFrame(frame)` - after the frame fence wait, before recording - so
  each of the two sets is patched exactly when it is provably idle; both copies converge
  within two frames. (Rejected alternative: enabling update-after-bind for every binding -
  more feature bits, no simplification, and the frame-start queue is needed for per-frame
  buffers anyway.)
- **Pool strategy.** The registry owns one `UpdateAfterBindPoolBit` pool sized for exactly
  the scene sets (2 x full scene layout including `MAX_BINDLESS_TEXTURES` sampled images)
  plus the feature sets. Pass-set pooling belongs to the FrameGraph (§6). The hand-tuned
  global pool in `Renderer_Resources` shrinks to legacy-only during migration and is
  deleted at the end; `FreeDescriptorSetBit` disappears - sets are never individually
  freed, only pools reset or destroyed.
- **Registration sources:** GpuScene/ResourceManager registers
  materials/instances/lights/entityInfo/VB/IB (per-frame arrays where double-buffered) at
  init; the ray-query system registers `sceneTlas`; the IBL feature registers its group and
  images; sampler table slots via `RegisterSampler("sceneSamplers", s, i)`.

## 6. FrameGraph integration — graph-owned pass sets

### 6.1 Authoring: a pass declares its program and bind names

`GraphBuilder` gains:

```csharp
// existing Read/Write stay for accesses that are not shader-visible
// (attachments, indirect args, transfer)
void UseProgram(ShaderProgram program, int defaultPipelineIndex = 0);
GraphImage  Read (GraphImage h,  ResourceUsage u, string bindAs);
GraphImage  Write(GraphImage h,  ResourceUsage u, string bindAs);
GraphBuffer Read (GraphBuffer h, ResourceUsage u, string bindAs);
GraphBuffer Write(GraphBuffer h, ResourceUsage u, string bindAs);
```

`bindAs` is the Slang parameter name in the pass's set 1, and `ResourceAccess` gains a
`string? BindName`. Matching is **explicit-name-first**: no fuzzy leaf-name inference
against scoped resource names like `"Deferred/HDRColor"` - the author writes
`b.Read(hdr, ResourceUsage.SampledCompute, "srcColor")`. Attachments are unchanged (the
graph already knows `ColorTargets`/`DepthTarget` and drives `BeginRendering`).

### 6.2 Compile: bake pass sets (new step 5.5)

After resource allocation (step 5) and before sync baking, for each live pass with a
program:

1. Take `program.Reflection.Bindings` where `set == 1`; for each, find the pass access with
   a matching `BindName`. **Error** on: a set-1 parameter with no matching declared access;
   a `BindName` not present in the shader; a descriptor-type/usage mismatch (parameter is
   `RWTexture2D` but the access says `SampledCompute`).
2. Allocate the pass set(s) from a **graph-owned plain pool** created at Compile, sized
   from the sum of live passes' set-1 layouts x `MAX_CONCURRENT_FRAMES`. Per-frame copies
   are allocated only when a bound resource is itself per-frame
   (`PhysBufferFrames != null` or a per-frame imported image); otherwise one set serves
   both frames. Passes that `UseProgram` the same `ShaderProgram` with the same resolved
   bind tuple share one set - the wavefront module's Generate/Extend/Shade/Connect passes
   collapse onto one set.
3. Write descriptors from resolved physical handles
   (`GraphResource.PhysView` / `PhysBuffer` / `PhysBufferFrames[f]`).
4. Store on `GraphPass`: `PipelineLayout`, default `Pipeline`, `BindPoint`,
   `DescriptorSet[] PassSets`, and which feature groups the program references (any
   reflected binding with set >= 2, looked up by set index).

The graph is rebuilt on resize, so recompile-rewrites come for free: a new Compile bakes a
fresh pool and fresh sets from the new transients. This is what deletes
`WriteGBufferDescriptors` / `WriteTileBufferDescriptors` and the resize re-wiring: g-buffer
views and tile buffers become named set-1 reads.

### 6.3 Execute: the graph binds, not the pipeline

Before invoking `pass.Execute` (both plain and chunked paths):

```csharp
if (pass.Program != null)
{
    uint dynOffset = pass.PrepareConstants?.Invoke(in frame, registry.ConstantArena) ?? 0;
    vk.CmdBindPipeline(cmd, pass.BindPoint, pass.DefaultPipeline);
    // scene set always; pass set if present; feature sets at their pinned indices
    vk.CmdBindDescriptorSets(cmd, pass.BindPoint, pass.PipelineLayout,
        firstSet: 0, setCount, sets, dynamicOffsetCount: 1, &dynOffset);
}
pass.Execute(cmd, resources, in frame);
```

Contract changes:

- **Record methods stop binding sets.** Pass bodies only push constants, bind
  vertex/index buffers, switch sibling PSOs, and issue draws/dispatches.
- **Multi-kernel passes** (wavefront PT: many PSOs, one shared layout): the graph binds
  sets plus the declared default PSO; the body may `CmdBindPipeline` siblings *sharing the
  same `PipelineLayout`* - bound sets stay valid under Vulkan compatibility rules.
- **Per-frame imported buffers** need no per-execute patching: per-frame set copies handle
  them statically (the barrier-side `BufferBarrierRes` patching stays as is).
- **RT passes**: the RT `ShaderProgram` is multi-entry (raygen/miss/chit/ahit compiled per
  entry via `getEntryPointCode(i)`); the layout derives identically; `RtPipeline` keeps SBT
  packing and `CmdTraceRays`, with reflection's entry-point order defining group indices.
- **Chunked mode** binds per chunk command buffer; "skip when layout+sets equal the
  previous pass in the chunk" is a trivial internal optimization.

## 7. The (0,0) constant slot — dynamic-UBO arena

Today most pipelines own a small per-frame frame UBO at (0,0) (`LightingFrameUBO`,
`GeometryUBO`, `PathFrameUBO`, ...). With one shared scene set those per-pipeline UBOs
cannot live there. Options considered:

| Option | Mechanics | Pros | Cons |
|---|---|---|---|
| **A. Dynamic-UBO arena (adopted)** | scene binding 0 = `UNIFORM_BUFFER_DYNAMIC` into a per-frame host-visible bump buffer; dynamic offset supplied at bind | one buffer, zero per-pass sets/writes; per-draw extensible (rebind with a new offset is 4 bytes of cmd stream); RenderDoc shows a typed cbuffer; hot path is memcpy + aligned bump | descriptor range fixed at write time -> enforce a max per-pass constant size; offsets align to `minUniformBufferOffsetAlignment` (<= 256 by spec) |
| B. Pass-set UBO | each pass declares its UBO as a named set-1 resource | no reserved binding, no dynamic offsets | a per-frame UBO buffer + descriptor per pass; per-draw variation needs multiple sets; strictly more allocations for no benefit |
| C. BDA push constant | 8-byte device address pushed; shader loads through a pointer | fewest descriptors; unlimited size | loses typed-UBO debuggability (raw address in RenderDoc); burns shared push-constant space; ConstantBuffer -> pointer syntax rewrite in every shader; loses UBO-path hardware caching on some IHVs |

Sketch:

```csharp
public sealed class PassConstantArena     // owned by DescriptorRegistry
{
    // MAX_CONCURRENT_FRAMES x 256 KiB HOST_VISIBLE|COHERENT
    public uint Push<T>(uint frame, in T data) where T : unmanaged;
    // returns the dynamic offset, aligned up to minUniformBufferOffsetAlignment
    public void Reset(uint frame);        // called from registry.BeginFrame
}

// AddPass gains an optional prepare delegate, invoked by the graph before binding:
public delegate uint PassPrepare(in Renderer.FrameContext frame, PassConstantArena arena);
// usage: (in f, arena) => arena.Push(f.FrameIndex, new PbrPassParams { ... })
```

Validation: if a program reflects a (0,0) parameter it must be a constant buffer with
`size <= maxPassUniformSize` (4 KiB); the canonical DSL's binding 0 covers it. Passes with
no (0,0) parameter bind offset 0 harmlessly. Shared camera constants can later graduate to
a scene-set `frameConstants` binding without any design change.

## 8. Bindless / UpdateAfterBind details

- **Two per-frame scene sets, not one fully-update-after-bind set.** Materials, instances,
  and lights SSBOs are double-buffered per frame today; a single set would force
  update-after-bind onto every binding (feature-bit sprawl) and still could not express
  "frame f reads buffer f". Two sets plus the §5.2 frame-start rewrite queue is simpler and
  matches `MAX_CONCURRENT_FRAMES = 2`.
- **Texture array** (`sceneTextures[]`, binding 11):
  `UpdateAfterBindBit | PartiallyBoundBit | VariableDescriptorCountBit`, allocated at
  `MAX_BINDLESS_TEXTURES`; layout and pool carry the UpdateAfterBind flags. Slot writes go
  to both frame sets immediately (as today) - safe because a newly registered slot is
  unreferenced by any in-flight frame (materials pointing at it upload with the next
  frame's material SSBO), and unregistration parks a fallback texture in freed slots before
  reuse.
- **Samplers** (binding 10): fixed array of 16, written once. Immutable samplers in the DSL
  are a possible later refinement, not required.
- Reflection maps an unbounded array (`count == 0` in `.refl`) to the variable-count
  binding at `MAX_BINDLESS_TEXTURES`. Validation enforces it is the highest binding.
- Everything else in the scene set is a plain descriptor, updated only through the
  fence-safe queue.

## 9. Pipeline layouts and the PipelineBase diet

```csharp
// L1, GraphicsDevice-owned.
public sealed class PipelineLayoutCache : IDisposable
{
    // Key: (DSL handle tuple, push-constant ranges). Value: VkPipelineLayout.
    public PipelineLayout Get(ReadOnlySpan<DescriptorSetLayout> sets,
                              ReadOnlySpan<PushConstantRange> push);
    // Dedupes identical reflected set-1 layouts across programs too.
    public DescriptorSetLayout GetSetLayout(ReadOnlySpan<BindingDesc> bindings);
}
```

`ShaderProgram.Layout` assembles
`[SceneSetLayout, PassSetLayout (possibly empty), FeatureLayout(group) per reflected set >= 2]`
plus reflected push ranges, then calls `PipelineLayoutCache.Get`. Gaps between occupied set
indices get a cached empty DSL.

`PipelineBase` slims to: shader program + fixed-function state + push-constant helpers.

- **Deleted:** `CreateDescriptorSetLayouts`, `CreateDescriptorSets`, `WriteDescriptors`,
  `DescriptorSetLayouts[]`, `OwnedDescriptorSetLayoutIndices`, `DescriptorSets[][]`,
  `GetDescriptorSet`, `CreatePipelineLayout` (replaced by the cache), and every
  `WriteXxxDescriptor(s)` method across the features.
- **Kept:** `GraphicsPipeline`'s fixed-function hooks (rasterizer, depth-stencil, blend,
  vertex input, dynamic-rendering formats, view mask) - genuinely per-pipeline.
  `CreatePipeline` swaps `File.ReadAllBytes(ShaderPath)` for
  `ShaderLibrary.GetProgram(desc).Spirv(i)`.
- `Initialize()` becomes `CreatePipeline()` + `CreateResources()` (owned buffers, which
  shrink dramatically - most existed as descriptor-set backing).
- Per-pipeline `PipelineCacheHandle`s consolidate into one device-level `VkPipelineCache`
  persisted next to the shader cache (small adjacent win, Phase A).

## 10. Spec constants and push constants

- **Push constants:** reflected ranges become the `PushConstantRange[]` fed to the layout
  cache. Pipelines keep their C# push structs; a debug assert compares `sizeof(T)` against
  the reflected size per stage.
- **Spec constants:** reflection lists `(name, constantId, size)`. The stage-indexed unsafe
  `FillSpecializationData` hook is replaced by a name-keyed map:

```csharp
public sealed class SpecValues
{
    public SpecValues Set(string name, int v);
    public SpecValues Set(string name, bool v);
    public SpecValues Set(string name, float v);
}
// VkSpecializationInfo is built generically by joining SpecValues with reflected
// constantIds; unknown names are a startup error, unset ones keep shader defaults.
```

The wavefront/PT per-PSO spec permutations (generate camera modes, shade lobe classes)
become `GetProgram` + one `SpecValues` per PSO against a single cached SPIR-V - no extra
compiles.

## 11. Validation and debug

**Startup validation** (`DescriptorRegistry.Validate`), run after feature registration and
the first graph Compile, fails fast with a full report:

```
[DescriptorRegistry] scene set: 12 bindings, 2 frame copies, 1 dynamic UBO
  OK   sceneMaterials      (1,0) StorageBuffer   <- GpuScene.MaterialsSSBO   [per-frame]
  OK   sceneTlas           (5,0) AccelStruct     <- RayQuerySystem.Tlas
  FAIL sceneEmissiveAlias  (9,0) StorageBuffer   <- UNREGISTERED (consumed by: PTCompute, WavefrontPT/Shade)
  FAIL iblEnvCube          (3,2) CombinedImage   <- registered as StorageImage: TYPE MISMATCH
```

Checks: every reflected name in every program resolves (scene/feature names via the
registry, set-1 names via the pass's declared `bindAs` map); descriptor types match; (0,0)
size cap; feature set-index collisions; unbounded-array-is-last.

**Consumer tracking:** per-program `IMetadata::isParameterLocationUsed` (cached into
`.refl`) lets the dump show who *actually reads* each binding, not just who could.

**`DumpBindings()`** emits the provider-to-consumer table; optionally a DOT export of
name-to-program edges in the spirit of `GraphDebug.ToDot`.

Name-mismatch errors at graph Compile carry pass + shader + a nearest-name suggestion.

## 12. Migration phasing

Each phase leaves the app buildable and rendering identically.

| Phase | Delivers | Notes |
|---|---|---|
| **A** | `SlangInterop` + `ShaderCompiler` + shader cache + `ShaderLibrary`; `PipelineBase.CreatePipeline` sources SPIR-V from the library instead of `File.ReadAllBytes`; **assert-mode reflection** compares reflected layouts against the existing hand-written DSLs (mismatches logged, not fatal); one persisted device-level `VkPipelineCache` | zero behavior change; `Shaders.targets` untouched; proves the interop + cache and audits today's drift |
| **B** | `SceneBindings.slang`; `DescriptorRegistry` (scene set, frame-start rewrite queue, bindless absorbed behind a narrow interface `ResourceManager` calls); `PassConstantArena` + the (0,0) dynamic slot (prerequisite of a shared set 0); shaders migrate to `import SceneBindings` one by one (unmigrated shaders keep their private set 0 - two layout shapes coexist); TLAS/lights/VB-IB `WriteXxx` fan-outs become `Register*` calls | the biggest shader diff; do it shader-by-shader |
| **C** | graph-baked pass sets: `UseProgram` + `bindAs` overloads, Compile step 5.5, Execute-side binding; delete `WriteGBufferDescriptors` / `WriteTileBufferDescriptors` and the resize re-wiring in `DeferredCore` / `Renderer_Core` | deletes the phase-separated init-order wiring; the graph owns binding |
| **D** | feature sets (`IblBindings`@2, `WavefrontBindings`@3, probe/ReSTIR groups); delete `PipelineBase` descriptor machinery, `ResourceManager` bindless internals, and the global descriptor pool; remove migrated entries from `Shaders.targets`; `SpecValues` replaces `FillSpecializationData` | end state; hot reload becomes a small follow-up (file watcher -> recompile closure -> PSO swap at a frame boundary) |

Recommended stopping point if time-boxed: **A + B** - that already kills the worst
duplication (the copy-pasted scene set and the TLAS fan-out).

## 13. Risks and open questions

- **COM-lite ABI drift** across Slang releases: pinned version range + startup smoke
  compile; the interop is quarantined behind `IShaderCompiler`, with the slangc-subprocess
  `-reflection-json` route as a drop-in fallback.
- **`spReflection_*` exports are listed in `slang-deprecated.h`:** they are the ABI beneath
  slang.h's inline C++ wrappers today, but Slang reserves the right to drop them. All
  reflection walking sits in one file; worst case it moves to JSON parsing.
- **Unused `RaytracingAccelerationStructure` in raster shaders:** importing `SceneBindings`
  into a shader compiled *without* ray capabilities must dead-strip the TLAS global.
  Verify in Phase A; fallback = a `SceneRT.slang` sub-module (still set 0, both modules
  reflected into the one canonical DSL).
- **Cold-start compile time:** ~35 kernels through the runtime compiler on first run.
  Measure in Phase A; mitigations: parallel sessions (one global session per worker),
  shipping a warm cache, keeping `Shaders.targets` as a cache-primer during development.
- **Global session thread-affinity** constrains hot-reload threading (dedicated compile
  thread + frame-boundary hand-off).
- Open: when materials/instances SSBOs become single-buffered device-local (graph Phase 2
  transfer-queue uploads), some per-frame set divergence collapses - revisit the two-set
  split then.
- Open: IBL bake kernels / `BcEncoder` / ImGui stay manual (decided for now); revisit if
  they become graph passes.