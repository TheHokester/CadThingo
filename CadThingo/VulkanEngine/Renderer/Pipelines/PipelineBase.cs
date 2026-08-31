using CadThingo.VulkanEngine.Renderer.Descriptors;
using CadThingo.VulkanEngine.Renderer.Slang;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Pipelines;


//  Pipeline wrapper layout
//
//  Three layers:
//    1. PipelineBase      - owns the handle, layout, cache, descriptor set
//                           layouts, push-constant ranges, and the lifecycle.
//    2. GraphicsPipeline  - assembles GraphicsPipelineCreateInfo from a set of
//                           protected virtual hooks (vertex input, raster,
//                           blend, depth, dynamic rendering formats, …) so
//                           concrete pipelines override only what differs.
//    3. ComputePipeline   - single-shader-stage compute equivalent.
//
//  Concrete pipelines (Geometry, PbrDeferred, DrawCull, LightCull, …)
//  inherit from layer 2/3 and own their own SSBOs/UBOs, push-constant
//  structs, descriptor sets, and Record(...) entry points.

public abstract unsafe class PipelineBase : IDisposable
{
    

    // The injected device-services channel. Anything RHI, descriptor, or shader related goes
    // through these three handles. Technique and scene data arrive from the owning feature, either
    // at construction or at record time.
    protected readonly GpuContext Gpu;

    protected GraphicsDevice     Gfx      => Gpu.Gfx;
    protected DescriptorRegistry Registry => Gpu.Registry;
    protected ShaderLibrary      Shaders  => Gpu.Shaders;

    // Convenience accessors so subclass bodies stay short — forward to the device.
    protected Vk     Vk     => Gfx.Vk;
    protected Device Device => Gfx.Device;

    protected Pipeline       PipelineHandle;
    protected PipelineLayout PipelineLayoutHandle;

    /// The shared device-level pipeline cache every PSO build feeds, warm across runs. Was a
    /// per-pipeline field that nothing ever created, so every PSO compiled cold on every launch.
    protected PipelineCache PipelineCacheHandle => Gfx.LayoutCache.PipelineCache;

    // Subclasses populate these in CreateDescriptorSetLayouts(). The default
    // CreatePipelineLayout() reads them to build the VkPipelineLayout.
    protected DescriptorSetLayout[] DescriptorSetLayouts = Array.Empty<DescriptorSetLayout>();

    // Indices into DescriptorSetLayouts that this pipeline OWNS and should
    // destroy on Dispose. Subclasses that borrow a layout from elsewhere
    // (e.g. ResourceManager's bindless layout) leave that index out.
    protected int[] OwnedDescriptorSetLayoutIndices = Array.Empty<int>();

    /// <summary>
    /// Descriptor sets for the pipeline <br/>
    /// DescriptorSets[layoutNum][frame]
    /// </summary>
    protected DescriptorSet[][] DescriptorSets = Array.Empty<DescriptorSet[]>();
    protected PushConstantRange[]   PushConstantRanges   = Array.Empty<PushConstantRange>();

    public Pipeline                  Handle    => PipelineHandle;
    public PipelineLayout            Layout    => PipelineLayoutHandle;
    
    
    protected PipelineBase(in GpuContext gpu)
    {
        Gpu = gpu;
    }

    public DescriptorSet GetDescriptorSet(int layoutNum, uint frame) => DescriptorSets[layoutNum][frame];
    public abstract PipelineBindPoint BindPoint { get; }

    // ---- program ---------------------------------------------------------------------------
    // The .slang module this pipeline is built from. ShaderLibrary compiles it at startup (disk
    // cached) and reflection supplies SPIR-V, push-constant ranges and spec-constant ids; a
    // pipeline may still hand-write its descriptor set layouts alongside. Every concrete
    // pipeline overrides this -- CreatePipeline throws without it.
    protected virtual ShaderCompileRequest? Program => null;

    /// The resolved program. Valid from the start of Initialize.
    protected ShaderProgram? Reflected { get; private set; }

    /// The resolved program, for whole-renderer cross-checks (DescriptorRegistry.Validate).
    public ShaderProgram? ReflectedProgram => Reflected;

    /// The set indices this pipeline builds its own layout for, and so owns privately. Set 0 is the
    /// registry's scene set only for pipelines that opted into it - a pass with no scene dependency
    /// (Tonemap, DrawCull, Skybox) puts its own pass set there instead. Registry validation must
    /// skip these, or it would try to resolve private pass names against SceneBindings.
    public IReadOnlyList<int> PrivateSetIndices => OwnedDescriptorSetLayoutIndices;

    /// Values for the program's reflected spec constants, keyed by name. Read at each build.
    protected virtual SpecValues? Specialization => null;

    /// The immutable sampler to bake into a reflected layout binding, or null for none.
    /// Reflection reports that a binding is a combined image sampler but not which sampler the
    /// layout pins, so the pipeline still answers that.
    protected virtual Sampler? ImmutableSamplerFor(in BindingDesc binding) => null;

    /// Stages to OR onto a reflected binding's visibility. Reflection only knows the stages of the
    /// program it walked, so a pipeline that builds SIBLING PSOs of a different kind against the
    /// same layout (ReSTIR's compute passes on the RT layout) must name those stages here, or the
    /// sibling cannot bind the set.
    protected virtual ShaderStageFlags ExtraStagesFor(in BindingDesc binding) => 0;

    /// Stages to OR onto every reflected push-constant range, for the same reason as
    /// <see cref="ExtraStagesFor"/>.
    protected virtual ShaderStageFlags ExtraPushStages => 0;

    /// The program's reflected bindings for one descriptor set, ordered by binding index.
    protected BindingDesc[] ReflectedBindings(uint set)
    {
        var program = Reflected ?? throw new InvalidOperationException(
            $"{GetType().Name}: ReflectedBindings requires a Program.");
        return program.Reflection.Bindings.Where(b => b.Set == set).OrderBy(b => b.Binding).ToArray();
    }

    /// Builds a VkDescriptorSetLayout for one reflected set. The shader declaration is the whole
    /// contract: binding index, descriptor type, and count all come from reflection.
    protected DescriptorSetLayout CreateReflectedSetLayout(uint set)
    {
        var reflected = ReflectedBindings(set);
        if (reflected.Length == 0)
            throw new InvalidOperationException($"{GetType().Name}: program reflects no bindings in set {set}.");

        var bindings = new DescriptorSetLayoutBinding[reflected.Length];
        var samplers = new Sampler?[reflected.Length];
        for (int i = 0; i < reflected.Length; i++)
        {
            var b = reflected[i];
            bindings[i] = new DescriptorSetLayoutBinding
            {
                Binding         = b.Binding,
                DescriptorType  = b.Type,
                DescriptorCount = b.Count,
                StageFlags      = b.Stages | ExtraStagesFor(in b),
            };
            samplers[i] = ImmutableSamplerFor(in b);
        }

        // Through the cache so it owns the handle; pass sets rarely collide, so expect a miss.
        return Gfx.LayoutCache.GetSetLayout(bindings, samplers);
    }

    // Builds the VkSpecializationInfo payload for one stage: joins the pipeline's named values
    // with the program's reflected constant ids, keeping only ids this stage's SPIR-V declares
    // (Slang strips constants an entry point never reads). Returns the entry count.
    private int FillReflectedSpecialization(
        int entryIndex, SpecializationMapEntry* entries, byte* data, out uint dataSize)
    {
        dataSize = 0;
        var program = Reflected!;
        if (Specialization is not { } values || values.Bits.Count == 0) return 0;

        var declared = SpirvUtil.SpecConstantIds(program.Spirv(entryIndex).Span);
        int filled = 0;
        foreach (var (name, bits) in values.Bits)
        {
            var match = program.Reflection.SpecConstants.FirstOrDefault(c => c.Name == name);
            if (match.Name != name)
                throw new InvalidOperationException(
                    $"{GetType().Name}: spec constant '{name}' is not declared by {program.Desc.Module} " +
                    $"(declared: {string.Join(", ", program.Reflection.SpecConstants.Select(c => c.Name))})");
            if (!declared.Contains(match.ConstantId)) continue;

            entries[filled] = new SpecializationMapEntry
            {
                ConstantID = match.ConstantId,
                Offset     = dataSize,
                Size       = sizeof(uint),
            };
            *(uint*)(data + dataSize) = bits;
            dataSize += sizeof(uint);
            filled++;
        }
        return filled;
    }

    // Creates the shader module for one entry point of the reflected program.
    protected ShaderModule CreateReflectedModule(int entryIndex)
        => Gfx.CreateShaderModule(Reflected!.Spirv(entryIndex).ToArray());

    // Shared by GraphicsPipeline / ComputePipeline: build a stage's spec info on whichever route
    // is active. Scratch must outlive the vkCreate*Pipelines call.
    private protected int FillStageSpecialization(
        int stageIdx, SpecializationMapEntry* entries, byte* data, out uint dataSize)
        => Reflected != null
            ? FillReflectedSpecialization(stageIdx, entries, data, out dataSize)
            : FillSpecializationData(stageIdx, entries, data, out dataSize);

    // Legacy specialization hook - stage-indexed and unsafe. Pipelines on the reflected route
    // override Specialization instead.
    protected const int SpecScratchEntries = 8;   // max spec entries per stage
    protected const int SpecScratchBytes   = 64;  // max spec data bytes per stage

    protected virtual int FillSpecializationData(
        int stageIdx,
        SpecializationMapEntry* entries,
        byte* data,
        out uint dataSize)
    {
        dataSize = 0;
        return 0;
    }

    

    // Called once by the owner (Renderer) after construction. Each step is a
    // virtual hook so concrete pipelines slot in their own logic without
    // re-implementing the whole flow.
    public void Initialize()
    {
        // Resolve first: CreateDescriptorSetLayouts reads reflection on the new route, and the
        // push ranges reflection yields feed CreatePipelineLayout below.
        Reflected = Program is { } request ? Shaders.GetProgram(request) : null;
        if (Reflected != null)
            PushConstantRanges = Reflected.Reflection.PushConstants
                .Select(p => new PushConstantRange
                {
                    StageFlags = p.Stages | ExtraPushStages, Offset = 0, Size = p.Size,
                })
                .ToArray();

        CreateDescriptorSetLayouts();
        CreatePipelineLayout();
        CreatePipeline();
        CreateResources();
        CreateDescriptorSets();
        WriteDescriptors();
    }

    /// <summary>Rebuilds this pipeline's GPU objects in place, preserving object identity so
    /// any held reference (render-graph module fields, cross-pipeline bindings) stays valid:
    /// tears everything down via <see cref="ReleaseGpuResources"/> and recreates it via
    /// <see cref="Initialize"/> on the same instance. Deliberately does NOT route through
    /// <see cref="Dispose"/> -- reusing an object after Dispose violates the IDisposable
    /// contract and would break the moment a Dispose override adds a _disposed guard /
    /// finalizer / SuppressFinalize. The caller must have idled the device (e.g. after
    /// toggling a spec-constant setting) and is responsible for re-applying any external
    /// descriptor writes Initialize does not perform (cross-pipeline / graph-transient binds).
    /// Replaces the old "Dispose + new + Initialize" pattern, which orphaned references the
    /// owner had handed out.</summary>
    public void Rebuild()
    {
        ReleaseGpuResources();
        Initialize();
    }

    // Required: populate descriptorSetLayouts and (optionally) pushConstantRanges.
    protected abstract void CreateDescriptorSetLayouts();

    // Required: build the VkPipeline itself. GraphicsPipeline / ComputePipeline
    // seal this and drive it from their hooks; only override directly if a
    // pipeline doesn't fit either category.
    protected abstract void CreatePipeline();

    // Cache-owned, so teardown skips it (see ReleaseGpuResources).
    protected virtual void CreatePipelineLayout()
        => PipelineLayoutHandle = Gfx.LayoutCache.Get(DescriptorSetLayouts, PushConstantRanges);

    // Concrete pipelines override these to allocate their owned SSBOs/UBOs,
    // allocate descriptor sets from the pool, and write the initial bindings.
    protected virtual void CreateResources()          { }
    protected virtual void CreateDescriptorSets()     { }
    protected virtual void WriteDescriptors()         { }

    public virtual void Dispose() => ReleaseGpuResources();

    /// <summary>The single GPU-teardown path, shared by <see cref="Dispose"/> (terminal) and
    /// <see cref="Rebuild"/> (followed by a fresh <see cref="Initialize"/>). Concrete pipelines
    /// override THIS -- not Dispose -- to free their own owned resources (UBOs/SSBOs/samplers),
    /// then call <c>base.ReleaseGpuResources()</c>. Routing Rebuild through here rather than
    /// Dispose keeps re-init off the IDisposable path.</summary>
    protected virtual void ReleaseGpuResources()
    {
        if (PipelineHandle.Handle != 0) Vk.DestroyPipeline(Device, PipelineHandle, null);

        // Cache-owned layouts are shared with other pipelines and outlive this one - destroying
        // them here would be a double free and a use-after-free for the pipelines still holding
        // them. Anything built by hand is still ours to release.
        var cache = Gfx.LayoutCache;
        if (PipelineLayoutHandle.Handle != 0 && !cache.Owns(PipelineLayoutHandle))
            Vk.DestroyPipelineLayout(Device, PipelineLayoutHandle, null);

        // Only destroy DSLs we own 
        foreach (var idx in OwnedDescriptorSetLayoutIndices)
        {
            if (idx < DescriptorSetLayouts.Length && DescriptorSetLayouts[idx].Handle != 0
                && !cache.Owns(DescriptorSetLayouts[idx]))
                Vk.DestroyDescriptorSetLayout(Device, DescriptorSetLayouts[idx], null);
        }
    }
}
