using CadThingo.VulkanEngine.Renderer.Shaders;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Descriptors;

// Owner of the unified scene set (docs/descriptor-system.md section 5.2): reflects
// SceneBindings.slang into the canonical set-0 layout (binding 0 injected as the dynamic
// constant-arena slot), owns the per-frame set instances and their pool, and matches
// resources to bindings BY PARAMETER NAME.
//
// Handle changes are one call: re-registering a name queues a rewrite that BeginFrame
// applies to each frame's set right after that frame's fence wait - the only moment the
// set is provably not in flight. Bindless texture slots ride the same queue: Vulkan
// forbids dynamic UBOs (the (0,0) arena slot) in an UPDATE_AFTER_BIND_POOL layout
// (VUID-VkDescriptorSetLayoutCreateInfo-descriptorType-03001), so the array is
// PartiallyBound-only and a fresh slot becomes visible when the frame slot rewrites -
// which is also when material rows referencing it upload.
/// A consuming program plus the set indices its pipeline owns privately. Validation needs both:
/// the same set index means "registry-owned" for one pipeline and "my own pass set" for another,
/// and only the pipeline knows which.
public readonly record struct ProgramUse(ShaderProgram Program, IReadOnlyList<int> PrivateSets);

public sealed unsafe class DescriptorRegistry : IDisposable
{
    private readonly GraphicsDevice _gfx;
    private readonly uint _frames;
    private Vk Vk => _gfx.Vk;
    private Device Device => _gfx.Device;

    private DescriptorSetLayout _sceneLayout;
    private DescriptorPool _pool;
    private readonly DescriptorSet[] _sceneSets;

    // Reflected scene bindings by parameter name (set 0, bindings >= 1).
    private readonly Dictionary<string, BindingDesc> _sceneBindings = new();
    private string _texturesName = ""; // the unbounded array (bindless slots)

    private sealed class BufferReg { public Buffer[] PerFrame = []; public ulong Offset, Range; }
    private sealed class ImageReg { public ImageView[] PerFrame = []; public ImageLayout Layout; public Sampler Sampler; }
    private sealed class TlasReg { public AccelerationStructureKHR Tlas; }
    private sealed class SamplerReg { public Sampler[] Slots = []; public bool[] Valid = []; }
    private readonly Dictionary<string, object> _registered = new();
    private readonly HashSet<string>[] _dirty;

    // A feature descriptor set: a fixed-index global set,
    // pinned at its own >= FirstFeature index by its shader module (e.g. FeatureIBL @ set 2),
    // owned here like the scene set but simpler - no dynamic-UBO arena slot, no bindless array.
    // Resources register by name (routed here from the same RegisterBuffer/Image/... calls) and
    // land per frame slot through the same fence-safe BeginFrame queue.
    private sealed class FeatureGroup
    {
        public string Name = "";
        public uint SetIndex;
        public DescriptorSetLayout Layout;
        public DescriptorSet[] Sets = [];
        public readonly Dictionary<string, BindingDesc> Bindings = new();
        public readonly Dictionary<string, object> Registered = new();
        public HashSet<string>[] Dirty = [];
    }
    private readonly List<FeatureGroup> _features = new();

    // Zero-binding layout used to plug the gaps a sparse feature-set consumer leaves in its
    // pipeline layout array (a shader using set 0 + set 3 needs valid layouts at 1 and 2). Never
    // has a set bound to it - gaps are simply not bound at record time.
    private DescriptorSetLayout _emptyLayout;
    private DescriptorPool _featurePool;

    private int _nextBindlessSlot;
    private readonly Stack<int> _freeBindlessSlots = new();
    private readonly List<(int Slot, ImageView View, ImageLayout Layout)>[] _pendingBindless;

    public PassConstantArena ConstantArena { get; }
    public DescriptorSetLayout SceneSetLayout => _sceneLayout;
    public DescriptorSet SceneSet(uint frame) => _sceneSets[frame];

    /// The zero-binding placeholder layout for unused set slots (see <see cref="BuildPipelineSetLayouts"/>).
    public DescriptorSetLayout EmptySetLayout => _emptyLayout;

    /// The layout of a feature set, by module name (e.g. "FeatureIBL"). For pipelines that build
    /// their own layout array; most should use <see cref="BuildPipelineSetLayouts"/>.
    public DescriptorSetLayout FeatureSetLayout(string feature) => Feature(feature).Layout;

    /// This frame's instance of a feature set, for the pipeline's CmdBindDescriptorSets.
    public DescriptorSet FeatureSet(string feature, uint frame) => Feature(feature).Sets[frame];

    /// The pinned global set index of a feature set (reflected from its module). Consumers bind
    /// their feature set at this index instead of hardcoding the literal, so renumbering a feature
    /// is a one-line edit in its shader module - no C# bind sites to chase.
    public uint FeatureSetIndex(string feature) => Feature(feature).SetIndex;

    /// <summary>Assembles a pipeline's full descriptor-set-layout array: scene at set 0, the
    /// pipeline's own pass-set layout at set 1 (pass null if it has none), each named feature at
    /// its pinned global index, and the shared empty layout in every gap. The array is sized to
    /// the highest set index actually used. Callers bind only the sets their shader uses - gaps
    /// are never bound.</summary>
    public DescriptorSetLayout[] BuildPipelineSetLayouts(DescriptorSetLayout? passSet, params string[] features)
    {
        uint max = ShaderSets.Scene;
        if (passSet is not null) max = Math.Max(max, ShaderSets.Pass);
        foreach (var f in features) max = Math.Max(max, Feature(f).SetIndex);

        var arr = new DescriptorSetLayout[max + 1];
        for (int i = 0; i < arr.Length; i++) arr[i] = _emptyLayout;
        arr[ShaderSets.Scene] = _sceneLayout;
        if (passSet is not null) arr[ShaderSets.Pass] = passSet.Value;
        foreach (var f in features) { var g = Feature(f); arr[g.SetIndex] = g.Layout; }
        return arr;
    }

    private FeatureGroup Feature(string name)
        => _features.FirstOrDefault(f => f.Name == name)
           ?? throw new ArgumentException($"unknown feature set '{name}' (known: {string.Join(", ", _features.Select(f => f.Name))})");

    public DescriptorRegistry(GraphicsDevice gfx, ShaderLibrary shaders, uint framesInFlight)
    {
        _gfx = gfx;
        _frames = framesInFlight;
        _sceneSets = new DescriptorSet[framesInFlight];
        _dirty = new HashSet<string>[framesInFlight];
        _pendingBindless = new List<(int, ImageView, ImageLayout)>[framesInFlight];
        for (int i = 0; i < framesInFlight; i++)
        {
            _dirty[i] = [];
            _pendingBindless[i] = [];
        }

        var reflected = shaders.ReflectModule("SceneBindings").Bindings;
        ValidateSceneModule(reflected);
        foreach (var b in reflected)
        {
            _sceneBindings[b.Name] = b;
            if (b.Count == 0) _texturesName = b.Name;
        }

        CreateLayoutPoolAndSets(reflected);
        ConstantArena = new PassConstantArena(gfx, framesInFlight);

        // Feature sets: each is a shader module pinning its own global set index. The
        // resource owners (IblSystem / ReflectionProbeSystem) register into them by name after
        // construction. Add a module name here to introduce a new feature set.
        CreateEmptyLayout();
        CreateFeatureSets(shaders, "FeatureIBL", "FeatureEnv");

        // Binding 0 (dynamic constant slot) is stable for the registry's lifetime; the sets
        // are idle at construction, so write it directly.
        for (uint f = 0; f < framesInFlight; f++)
        {
            var info = new DescriptorBufferInfo
            {
                Buffer = ConstantArena.Buffer(f),
                Offset = 0,
                Range = PassConstantArena.MaxPassUniformSize,
            };
            var write = new WriteDescriptorSet
            {
                SType = StructureType.WriteDescriptorSet,
                DstSet = _sceneSets[f],
                DstBinding = 0,
                DescriptorType = DescriptorType.UniformBufferDynamic,
                DescriptorCount = 1,
                PBufferInfo = &info,
            };
            Vk.UpdateDescriptorSets(Device, 1, &write, 0, null);
        }
    }

    private static void ValidateSceneModule(BindingDesc[] bindings)
    {
        if (bindings.Length == 0)
            throw new Exception("SceneBindings reflected no bindings");
        uint maxBinding = 0;
        int unbounded = 0;
        foreach (var b in bindings)
        {
            if (b.Set != ShaderSets.Scene)
                throw new Exception($"SceneBindings: '{b.Name}' is in set {b.Set}, expected {ShaderSets.Scene}");
            if (b.Binding == 0)
                throw new Exception($"SceneBindings: '{b.Name}' occupies binding 0, which is reserved for the pass constant slot");
            maxBinding = Math.Max(maxBinding, b.Binding);
            if (b.Count == 0) unbounded++;
        }
        if (unbounded != 1)
            throw new Exception($"SceneBindings: expected exactly 1 unbounded array, found {unbounded}");
        foreach (var b in bindings)
            if (b.Count == 0 && b.Binding != maxBinding)
                throw new Exception($"SceneBindings: unbounded array '{b.Name}' must be the highest binding");
    }

    private void CreateLayoutPoolAndSets(BindingDesc[] reflected)
    {
        var bindings = new DescriptorSetLayoutBinding[reflected.Length + 1];
        var flags = new DescriptorBindingFlags[reflected.Length + 1];
        bindings[0] = new DescriptorSetLayoutBinding
        {
            Binding = 0,
            DescriptorType = DescriptorType.UniformBufferDynamic,
            DescriptorCount = 1,
            StageFlags = ShaderStageFlags.All,
        };
        for (int i = 0; i < reflected.Length; i++)
        {
            var b = reflected[i];
            bool isBindless = b.Count == 0;
            bindings[i + 1] = new DescriptorSetLayoutBinding
            {
                Binding = b.Binding,
                DescriptorType = b.Type,
                DescriptorCount = isBindless ? ResourceManager.MAX_BINDLESS_TEXTURES : b.Count,
                StageFlags = ShaderStageFlags.All,
            };
            // Fixed-count + PartiallyBound only: VariableDescriptorCount is not an enabled
            // device feature, and UpdateAfterBind is illegal alongside the dynamic UBO at
            // binding 0 (see the class comment).
            flags[i + 1] = isBindless ? DescriptorBindingFlags.PartiallyBoundBit : 0;
        }

        fixed (DescriptorSetLayoutBinding* pBindings = bindings)
        fixed (DescriptorBindingFlags* pFlags = flags)
        {
            var flagsInfo = new DescriptorSetLayoutBindingFlagsCreateInfo
            {
                SType = StructureType.DescriptorSetLayoutBindingFlagsCreateInfo,
                BindingCount = (uint)flags.Length,
                PBindingFlags = pFlags,
            };
            var layoutInfo = new DescriptorSetLayoutCreateInfo
            {
                SType = StructureType.DescriptorSetLayoutCreateInfo,
                BindingCount = (uint)bindings.Length,
                PBindings = pBindings,
                PNext = &flagsInfo,
            };
            if (Vk.CreateDescriptorSetLayout(Device, &layoutInfo, null, out _sceneLayout) != Result.Success)
                throw new Exception("DescriptorRegistry: failed to create scene set layout");
        }

        var poolSizes = new Dictionary<DescriptorType, uint>();
        foreach (var b in bindings)
            poolSizes[b.DescriptorType] = poolSizes.GetValueOrDefault(b.DescriptorType) + b.DescriptorCount * _frames;
        var sizes = poolSizes.Select(kv => new DescriptorPoolSize { Type = kv.Key, DescriptorCount = kv.Value }).ToArray();

        fixed (DescriptorPoolSize* pSizes = sizes)
        {
            var poolInfo = new DescriptorPoolCreateInfo
            {
                SType = StructureType.DescriptorPoolCreateInfo,
                PoolSizeCount = (uint)sizes.Length,
                PPoolSizes = pSizes,
                MaxSets = _frames,
            };
            if (Vk.CreateDescriptorPool(Device, &poolInfo, null, out _pool) != Result.Success)
                throw new Exception("DescriptorRegistry: failed to create scene descriptor pool");
        }

        var layouts = stackalloc DescriptorSetLayout[(int)_frames];
        for (int i = 0; i < _frames; i++) layouts[i] = _sceneLayout;
        var alloc = new DescriptorSetAllocateInfo
        {
            SType = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool = _pool,
            DescriptorSetCount = _frames,
            PSetLayouts = layouts,
        };
        fixed (DescriptorSet* pSets = _sceneSets)
        {
            if (Vk.AllocateDescriptorSets(Device, &alloc, pSets) != Result.Success)
                throw new Exception("DescriptorRegistry: failed to allocate scene sets");
        }
    }

    private void CreateEmptyLayout()
    {
        var info = new DescriptorSetLayoutCreateInfo
        {
            SType = StructureType.DescriptorSetLayoutCreateInfo,
            BindingCount = 0,
        };
        if (Vk.CreateDescriptorSetLayout(Device, &info, null, out _emptyLayout) != Result.Success)
            throw new Exception("DescriptorRegistry: failed to create empty placeholder layout");
    }

    // Reflects each feature module into its own fixed-index set, then allocates one set per frame
    // from a shared feature pool. Feature sets are plain: no dynamic-UBO slot, no bindless array,
    // all-stages visibility. Idle at construction; resources land later via the register + BeginFrame path.
    private void CreateFeatureSets(ShaderLibrary shaders, params string[] moduleNames)
    {
        var poolSizes = new Dictionary<DescriptorType, uint>();
        uint totalSets = 0;

        foreach (var moduleName in moduleNames)
        {
            var reflected = shaders.ReflectModule(moduleName).Bindings;
            var group = new FeatureGroup { Name = moduleName, Sets = new DescriptorSet[_frames], Dirty = new HashSet<string>[_frames] };
            for (int i = 0; i < _frames; i++) group.Dirty[i] = [];
            ValidateFeatureModule(moduleName, reflected, out group.SetIndex);

            var layoutBindings = new DescriptorSetLayoutBinding[reflected.Length];
            for (int i = 0; i < reflected.Length; i++)
            {
                var b = reflected[i];
                group.Bindings[b.Name] = b;
                layoutBindings[i] = new DescriptorSetLayoutBinding
                {
                    Binding = b.Binding,
                    DescriptorType = b.Type,
                    DescriptorCount = b.Count,
                    StageFlags = ShaderStageFlags.All,
                };
                poolSizes[b.Type] = poolSizes.GetValueOrDefault(b.Type) + b.Count * _frames;
            }
            fixed (DescriptorSetLayoutBinding* pB = layoutBindings)
            {
                var info = new DescriptorSetLayoutCreateInfo
                {
                    SType = StructureType.DescriptorSetLayoutCreateInfo,
                    BindingCount = (uint)layoutBindings.Length,
                    PBindings = pB,
                };
                if (Vk.CreateDescriptorSetLayout(Device, &info, null, out group.Layout) != Result.Success)
                    throw new Exception($"DescriptorRegistry: failed to create '{moduleName}' layout");
            }
            _features.Add(group);
            totalSets += _frames;
        }

        if (_features.Count == 0) return;

        var sizes = poolSizes.Select(kv => new DescriptorPoolSize { Type = kv.Key, DescriptorCount = kv.Value }).ToArray();
        fixed (DescriptorPoolSize* pSizes = sizes)
        {
            var poolInfo = new DescriptorPoolCreateInfo
            {
                SType = StructureType.DescriptorPoolCreateInfo,
                PoolSizeCount = (uint)sizes.Length,
                PPoolSizes = pSizes,
                MaxSets = totalSets,
            };
            if (Vk.CreateDescriptorPool(Device, &poolInfo, null, out _featurePool) != Result.Success)
                throw new Exception("DescriptorRegistry: failed to create feature descriptor pool");
        }

        foreach (var group in _features)
        {
            var layouts = stackalloc DescriptorSetLayout[(int)_frames];
            for (int i = 0; i < _frames; i++) layouts[i] = group.Layout;
            var alloc = new DescriptorSetAllocateInfo
            {
                SType = StructureType.DescriptorSetAllocateInfo,
                DescriptorPool = _featurePool,
                DescriptorSetCount = _frames,
                PSetLayouts = layouts,
            };
            fixed (DescriptorSet* pSets = group.Sets)
                if (Vk.AllocateDescriptorSets(Device, &alloc, pSets) != Result.Success)
                    throw new Exception($"DescriptorRegistry: failed to allocate '{group.Name}' sets");
        }
    }

    // Feature modules pin one global set index (>= FirstFeature); no bindless arrays (v1).
    private static void ValidateFeatureModule(string name, BindingDesc[] bindings, out uint setIndex)
    {
        if (bindings.Length == 0) throw new Exception($"{name} reflected no bindings");
        setIndex = bindings[0].Set;
        if (setIndex < ShaderSets.FirstFeature)
            throw new Exception($"{name}: set {setIndex} is below FirstFeature ({ShaderSets.FirstFeature})");
        foreach (var b in bindings)
        {
            if (b.Set != setIndex)
                throw new Exception($"{name}: '{b.Name}' is in set {b.Set}, expected a single feature set {setIndex}");
            if (b.Count == 0)
                throw new Exception($"{name}: '{b.Name}' is an unbounded array; feature sets are fixed-count (v1)");
        }
    }

    // ---- name -> resource registration --------------------------------------------------

    public void RegisterBuffer(string name, Buffer buf, ulong offset = 0, ulong range = Vk.WholeSize)
        => RegisterBufferInternal(name, [buf], offset, range);

    public void RegisterBufferPerFrame(string name, Buffer[] perFrame, ulong range = Vk.WholeSize)
    {
        if (perFrame.Length != _frames)
            throw new ArgumentException($"'{name}': expected {_frames} per-frame buffers, got {perFrame.Length}");
        RegisterBufferInternal(name, (Buffer[])perFrame.Clone(), 0, range);
    }

    private void RegisterBufferInternal(string name, Buffer[] perFrame, ulong offset, ulong range)
    {
        var (group, _) = Expect(name, DescriptorType.StorageBuffer, DescriptorType.UniformBuffer);
        StoreReg(group, name, new BufferReg { PerFrame = perFrame, Offset = offset, Range = range });
    }

    public void RegisterImage(string name, ImageView view, ImageLayout layout, Sampler? sampler = null)
        => RegisterImageInternal(name, [view], layout, sampler);

    public void RegisterImagePerFrame(string name, ImageView[] perFrame, ImageLayout layout, Sampler? sampler = null)
    {
        if (perFrame.Length != _frames)
            throw new ArgumentException($"'{name}': expected {_frames} per-frame views, got {perFrame.Length}");
        RegisterImageInternal(name, (ImageView[])perFrame.Clone(), layout, sampler);
    }

    private void RegisterImageInternal(string name, ImageView[] perFrame, ImageLayout layout, Sampler? sampler)
    {
        var (group, b) = Expect(name, DescriptorType.SampledImage, DescriptorType.CombinedImageSampler, DescriptorType.StorageImage);
        if (b.Type == DescriptorType.CombinedImageSampler && sampler is null)
            throw new ArgumentException($"'{name}' is a combined image sampler; a sampler is required");
        StoreReg(group, name, new ImageReg { PerFrame = perFrame, Layout = layout, Sampler = sampler ?? default });
    }

    public void RegisterTlas(string name, AccelerationStructureKHR tlas)
    {
        var (group, _) = Expect(name, DescriptorType.AccelerationStructureKhr);
        StoreReg(group, name, new TlasReg { Tlas = tlas });
    }

    public void RegisterSampler(string name, Sampler sampler, int arrayIndex = 0)
    {
        var (group, b) = Expect(name, DescriptorType.Sampler);
        if (arrayIndex < 0 || arrayIndex >= b.Count)
            throw new ArgumentException($"'{name}'[{arrayIndex}] out of range (count {b.Count})");
        var store = group?.Registered ?? _registered;
        if (store.TryGetValue(name, out var existing) && existing is SamplerReg reg)
        {
            reg.Slots[arrayIndex] = sampler;
            reg.Valid[arrayIndex] = true;
        }
        else
        {
            var fresh = new SamplerReg { Slots = new Sampler[b.Count], Valid = new bool[b.Count] };
            fresh.Slots[arrayIndex] = sampler;
            fresh.Valid[arrayIndex] = true;
            store[name] = fresh;
        }
        MarkDirtyOn(group, name);
    }

    // Resolves a name to its owning set (null group == scene) + BindingDesc, validating the type.
    private (FeatureGroup? group, BindingDesc b) Expect(string name, params DescriptorType[] allowed)
    {
        if (_sceneBindings.TryGetValue(name, out var sb))
        {
            if (name == _texturesName)
                throw new ArgumentException($"'{name}' is the bindless array; use RegisterBindlessTexture");
            if (!allowed.Contains(sb.Type))
                throw new ArgumentException($"'{name}' is {sb.Type}, not {string.Join("/", allowed)}");
            return (null, sb);
        }
        foreach (var g in _features)
            if (g.Bindings.TryGetValue(name, out var fb))
            {
                if (!allowed.Contains(fb.Type))
                    throw new ArgumentException($"'{name}' is {fb.Type}, not {string.Join("/", allowed)}");
                return (g, fb);
            }
        throw new ArgumentException($"'{name}' is not a scene or feature-set parameter");
    }

    // Stores a resource reg into its owning set + marks that set dirty on every frame slot.
    private void StoreReg(FeatureGroup? group, string name, object reg)
    {
        (group?.Registered ?? _registered)[name] = reg;
        MarkDirtyOn(group, name);
    }

    private void MarkDirtyOn(FeatureGroup? group, string name)
    {
        foreach (var set in group?.Dirty ?? _dirty) set.Add(name);
    }

    // ---- bindless texture table ----------------------------------------------------------

    /// Returns the slot immediately; the descriptor write lands per frame slot via the
    /// fence-safe queue (see the class comment for why not UpdateAfterBind). Callers must
    /// keep a replaced/removed view alive for MAX_CONCURRENT_FRAMES.
    public int RegisterBindlessTexture(ImageView view, ImageLayout layout = ImageLayout.ShaderReadOnlyOptimal)
    {
        int slot = _freeBindlessSlots.Count > 0 ? _freeBindlessSlots.Pop() : _nextBindlessSlot++;
        if (slot >= ResourceManager.MAX_BINDLESS_TEXTURES)
            throw new InvalidOperationException("bindless texture table exhausted");
        QueueBindlessWrite(slot, view, layout);
        return slot;
    }

    /// Parks a fallback view in the slot before recycling it, matching the existing
    /// ResourceManager behavior (stale materials sample the fallback, never a freed view).
    public void UnregisterBindlessTexture(int slot, ImageView fallback)
    {
        QueueBindlessWrite(slot, fallback, ImageLayout.ShaderReadOnlyOptimal);
        _freeBindlessSlots.Push(slot);
    }

    /// Mirrors an externally-allocated bindless slot. ResourceManager owns slot indices
    /// today (material rows store them), so during migration the registry table follows
    /// its allocator instead of using RegisterBindlessTexture's.
    public void SetBindlessSlot(int slot, ImageView view, ImageLayout layout = ImageLayout.ShaderReadOnlyOptimal)
    {
        if (slot < 0 || slot >= ResourceManager.MAX_BINDLESS_TEXTURES)
            throw new ArgumentOutOfRangeException(nameof(slot));
        QueueBindlessWrite(slot, view, layout);
    }

    private void QueueBindlessWrite(int slot, ImageView view, ImageLayout layout)
    {
        foreach (var pending in _pendingBindless) pending.Add((slot, view, layout));
    }

    // ---- lifecycle -------------------------------------------------------------------------

    /// Call once per frame after the frame fence wait, before recording: applies queued
    /// rewrites to this frame's set (it is provably idle here) and resets its arena slice.
    public void BeginFrame(uint frame)
    {
        ConstantArena.Reset(frame);

        foreach (var name in _dirty[frame])
            WriteOne(frame, name);
        _dirty[frame].Clear();

        var texturesBinding = _sceneBindings[_texturesName].Binding;
        foreach (var (slot, view, layout) in _pendingBindless[frame])
        {
            var info = new DescriptorImageInfo { ImageView = view, ImageLayout = layout };
            var write = new WriteDescriptorSet
            {
                SType = StructureType.WriteDescriptorSet,
                DstSet = _sceneSets[frame],
                DstBinding = texturesBinding,
                DstArrayElement = (uint)slot,
                DescriptorType = DescriptorType.SampledImage,
                DescriptorCount = 1,
                PImageInfo = &info,
            };
            Vk.UpdateDescriptorSets(Device, 1, &write, 0, null);
        }
        _pendingBindless[frame].Clear();

        // Feature sets ride the same fence-safe queue: flush each one's dirty names into this
        // frame slot (provably idle here). Handles are usually stable, so this is a one-shot per
        // frame slot at registration and again after a resource realloc.
        foreach (var g in _features)
        {
            foreach (var name in g.Dirty[frame])
                WriteBinding(g.Sets[frame], g.Bindings[name], g.Registered[name], frame);
            g.Dirty[frame].Clear();
        }
    }

    private void WriteOne(uint frame, string name)
        => WriteBinding(_sceneSets[frame], _sceneBindings[name], _registered[name], frame);

    // Emits the descriptor write(s) for one registered resource into a specific set. Shared by the
    // scene set and every feature set - the only differences (which set, which binding) are args.
    private void WriteBinding(DescriptorSet set, BindingDesc b, object resource, uint frame)
    {
        var write = new WriteDescriptorSet
        {
            SType = StructureType.WriteDescriptorSet,
            DstSet = set,
            DstBinding = b.Binding,
            DescriptorType = b.Type,
            DescriptorCount = 1,
        };
        switch (resource)
        {
            case BufferReg reg:
            {
                var info = new DescriptorBufferInfo
                {
                    Buffer = reg.PerFrame[reg.PerFrame.Length == 1 ? 0 : frame],
                    Offset = reg.Offset,
                    Range = reg.Range,
                };
                write.PBufferInfo = &info;
                Vk.UpdateDescriptorSets(Device, 1, &write, 0, null);
                break;
            }
            case ImageReg reg:
            {
                var info = new DescriptorImageInfo
                {
                    ImageView = reg.PerFrame[reg.PerFrame.Length == 1 ? 0 : frame],
                    ImageLayout = reg.Layout,
                    Sampler = reg.Sampler,
                };
                write.PImageInfo = &info;
                Vk.UpdateDescriptorSets(Device, 1, &write, 0, null);
                break;
            }
            case TlasReg reg:
            {
                var tlas = reg.Tlas;
                var asInfo = new WriteDescriptorSetAccelerationStructureKHR
                {
                    SType = StructureType.WriteDescriptorSetAccelerationStructureKhr,
                    AccelerationStructureCount = 1,
                    PAccelerationStructures = &tlas,
                };
                write.PNext = &asInfo;
                Vk.UpdateDescriptorSets(Device, 1, &write, 0, null);
                break;
            }
            case SamplerReg reg:
            {
                for (int i = 0; i < reg.Slots.Length; i++)
                {
                    if (!reg.Valid[i]) continue;
                    var info = new DescriptorImageInfo { Sampler = reg.Slots[i] };
                    write.DstArrayElement = (uint)i;
                    write.PImageInfo = &info;
                    Vk.UpdateDescriptorSets(Device, 1, &write, 0, null);
                }
                break;
            }
        }
    }

    // ---- startup validation ----------------------------------------------------------------

    /// <summary>Cross-checks every consuming program's reflected bindings against what the registry
    /// owns and has actually been given . This catches the
    /// class of bug the Vulkan validation layer is blind to: a scene parameter nothing ever
    /// registered, a name whose type or binding index drifted between the owning module and a
    /// consumer, a set index no owner claims. Sets 1 (pass) and 2 (graph-shared) are skipped - they
    /// are graph/pipeline-owned, not registry-owned.
    ///
    /// A binding with no provider only FAILS if something consumes it; an unconsumed hole is a
    /// warning, since a resource nobody reads is merely unfinished, not broken. Throws with the
    /// full report on any failure, otherwise returns it for logging.</summary>
    public string Validate(IEnumerable<ProgramUse> programs)
    {
        var consumers = new Dictionary<string, List<string>>();   // registry-owned name -> program labels
        var lines = new List<string>();
        var failures = new List<string>();

        void Fail(string line) { failures.Add(line); lines.Add("  FAIL " + line); }

        foreach (var (program, privateSets) in programs)
        {
            var desc = program.Desc;
            string label = desc.Module + (desc.Defines.Length > 0 ? $" [{string.Join(",", desc.Defines)}]" : "");

            foreach (var b in program.Reflection.Bindings)
            {
                // Pipeline- and graph-owned sets are validated by their own owners. Note set 0 is
                // NOT automatically the scene set: a pass with no scene dependency puts its own
                // private pass set there, and declares that by owning the layout.
                if (b.Set == ShaderSets.Pass || b.Set == ShaderSets.GraphShared) continue;
                if (privateSets.Contains((int)b.Set)) continue;

                // (0,0) is the arena slot the registry injects; it is deliberately absent from
                // SceneBindings.slang, so resolve it here rather than failing the name lookup.
                if (b.Set == ShaderSets.Scene && b.Binding == 0)
                {
                    if (b.Type is not (DescriptorType.UniformBuffer or DescriptorType.UniformBufferDynamic))
                        Fail($"{label}: '{b.Name}' at the reserved (0,0) arena slot is {b.Type}, expected a uniform buffer");
                    continue;
                }

                Dictionary<string, BindingDesc> declared;
                string owner;
                if (b.Set == ShaderSets.Scene) { declared = _sceneBindings; owner = "SceneBindings"; }
                else
                {
                    var group = _features.FirstOrDefault(f => f.SetIndex == b.Set);
                    if (group is null)
                    {
                        Fail($"{label}: '{b.Name}' is in set {b.Set}, which no registry owner claims " +
                             $"(scene={ShaderSets.Scene}, pass={ShaderSets.Pass}, graph-shared={ShaderSets.GraphShared}, " +
                             $"features: {string.Join(", ", _features.Select(f => $"{f.Name}@{f.SetIndex}"))})");
                        continue;
                    }
                    declared = group.Bindings; owner = group.Name;
                }

                if (!declared.TryGetValue(b.Name, out var d))
                {
                    Fail($"{label}: '{b.Name}' ({b.Set},{b.Binding}) is not declared in {owner}{Nearest(b.Name, declared.Keys)}");
                    continue;
                }
                if (d.Binding != b.Binding)
                    Fail($"{label}: '{b.Name}' is at binding {b.Binding}, but {owner} declares it at {d.Binding}");
                else if (d.Type != b.Type)
                    Fail($"{label}: '{b.Name}' is {b.Type}, but {owner} declares it {d.Type}");
                else if (d.Count != b.Count)
                    Fail($"{label}: '{b.Name}' has count {b.Count}, but {owner} declares count {d.Count}");

                if (!consumers.TryGetValue(b.Name, out var list)) consumers[b.Name] = list = [];
                list.Add(label);
            }
        }

        // Provider side: every owned binding, whether it has a resource, and who reads it.
        void Report(string owner, uint set, IEnumerable<BindingDesc> bindings, Dictionary<string, object> registered)
        {
            lines.Add($"  {owner} (set {set}):");
            foreach (var b in bindings.OrderBy(x => x.Binding))
            {
                var who = consumers.GetValueOrDefault(b.Name);
                string read = who is null ? "unread" : $"read by {who.Count}: {string.Join(", ", who.Distinct())}";

                // The bindless array is filled slot-wise, never through the name registry.
                bool provided = registered.ContainsKey(b.Name) || (set == ShaderSets.Scene && b.Name == _texturesName);
                string entry = $"{b.Name,-22} ({set},{b.Binding,2}) {b.Type,-26} count={b.Count,-4}";

                if (provided) lines.Add($"    OK   {entry} <- {read}");
                else if (who is not null) Fail($"{entry} <- UNREGISTERED, {read}");
                else lines.Add($"    warn {entry} <- UNREGISTERED (unread, no consumer yet)");
            }
        }

        Report("SceneBindings", ShaderSets.Scene, _sceneBindings.Values, _registered);
        foreach (var g in _features) Report(g.Name, g.SetIndex, g.Bindings.Values, g.Registered);

        string report = $"[registry] validate: {failures.Count} failure(s) across " +
                        $"{consumers.Count} consumed parameter(s)" + Environment.NewLine +
                        string.Join(Environment.NewLine, lines);

        if (failures.Count > 0)
            throw new Exception($"DescriptorRegistry.Validate found {failures.Count} problem(s):" +
                                Environment.NewLine + report);
        return report;
    }

    // Cheap typo hint for a failed name lookup: any candidate within edit distance 3.
    private static string Nearest(string name, IEnumerable<string> candidates)
    {
        var best = candidates
            .Select(c => (c, d: Distance(name, c)))
            .Where(t => t.d <= 3)
            .OrderBy(t => t.d)
            .Select(t => t.c)
            .FirstOrDefault();
        return best is null ? "" : $" (did you mean '{best}'?)";
    }

    private static int Distance(string a, string b)
    {
        var prev = new int[b.Length + 1];
        var cur = new int[b.Length + 1];
        for (int j = 0; j <= b.Length; j++) prev[j] = j;
        for (int i = 1; i <= a.Length; i++)
        {
            cur[0] = i;
            for (int j = 1; j <= b.Length; j++)
                cur[j] = Math.Min(Math.Min(prev[j] + 1, cur[j - 1] + 1),
                                  prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1));
            (prev, cur) = (cur, prev);
        }
        return prev[b.Length];
    }

    /// Startup diagnostics: which scene parameters have a provider, which are still holes.
    public string DumpBindings()
    {
        var lines = new List<string> { $"[registry] scene set: {_sceneBindings.Count + 1} bindings, {_frames} frame copies" };
        lines.Add("  arena (0,0) UniformBufferDynamic <- PassConstantArena");
        foreach (var b in _sceneBindings.Values.OrderBy(x => x.Binding))
        {
            string status = b.Name == _texturesName
                ? $"bindless table ({_nextBindlessSlot - _freeBindlessSlots.Count} live slots)"
                : _registered.ContainsKey(b.Name) ? "registered" : "UNREGISTERED";
            lines.Add($"  {b.Name,-20} (0,{b.Binding,2}) {b.Type,-26} count={b.Count} <- {status}");
        }
        return string.Join(Environment.NewLine, lines);
    }

    public void Dispose()
    {
        ConstantArena.Dispose();
        if (_pool.Handle != 0) Vk.DestroyDescriptorPool(Device, _pool, null);
        if (_sceneLayout.Handle != 0) Vk.DestroyDescriptorSetLayout(Device, _sceneLayout, null);

        if (_featurePool.Handle != 0) Vk.DestroyDescriptorPool(Device, _featurePool, null);
        foreach (var g in _features)
            if (g.Layout.Handle != 0) Vk.DestroyDescriptorSetLayout(Device, g.Layout, null);
        if (_emptyLayout.Handle != 0) Vk.DestroyDescriptorSetLayout(Device, _emptyLayout, null);
    }
}