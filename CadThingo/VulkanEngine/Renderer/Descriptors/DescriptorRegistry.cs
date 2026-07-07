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

    private int _nextBindlessSlot;
    private readonly Stack<int> _freeBindlessSlots = new();
    private readonly List<(int Slot, ImageView View, ImageLayout Layout)>[] _pendingBindless;

    public PassConstantArena ConstantArena { get; }
    public DescriptorSetLayout SceneSetLayout => _sceneLayout;
    public DescriptorSet SceneSet(uint frame) => _sceneSets[frame];

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
        var b = Expect(name, DescriptorType.StorageBuffer, DescriptorType.UniformBuffer);
        _registered[name] = new BufferReg { PerFrame = perFrame, Offset = offset, Range = range };
        MarkDirty(name);
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
        var b = Expect(name, DescriptorType.SampledImage, DescriptorType.CombinedImageSampler, DescriptorType.StorageImage);
        if (b.Type == DescriptorType.CombinedImageSampler && sampler is null)
            throw new ArgumentException($"'{name}' is a combined image sampler; a sampler is required");
        _registered[name] = new ImageReg { PerFrame = perFrame, Layout = layout, Sampler = sampler ?? default };
        MarkDirty(name);
    }

    public void RegisterTlas(string name, AccelerationStructureKHR tlas)
    {
        Expect(name, DescriptorType.AccelerationStructureKhr);
        _registered[name] = new TlasReg { Tlas = tlas };
        MarkDirty(name);
    }

    public void RegisterSampler(string name, Sampler sampler, int arrayIndex = 0)
    {
        var b = Expect(name, DescriptorType.Sampler);
        if (arrayIndex < 0 || arrayIndex >= b.Count)
            throw new ArgumentException($"'{name}'[{arrayIndex}] out of range (count {b.Count})");
        if (_registered.TryGetValue(name, out var existing) && existing is SamplerReg reg)
        {
            reg.Slots[arrayIndex] = sampler;
            reg.Valid[arrayIndex] = true;
        }
        else
        {
            var fresh = new SamplerReg { Slots = new Sampler[b.Count], Valid = new bool[b.Count] };
            fresh.Slots[arrayIndex] = sampler;
            fresh.Valid[arrayIndex] = true;
            _registered[name] = fresh;
        }
        MarkDirty(name);
    }

    private BindingDesc Expect(string name, params DescriptorType[] allowed)
    {
        if (!_sceneBindings.TryGetValue(name, out var b))
            throw new ArgumentException($"'{name}' is not a SceneBindings parameter (known: {string.Join(", ", _sceneBindings.Keys)})");
        if (name == _texturesName)
            throw new ArgumentException($"'{name}' is the bindless array; use RegisterBindlessTexture");
        if (!allowed.Contains(b.Type))
            throw new ArgumentException($"'{name}' is {b.Type}, not {string.Join("/", allowed)}");
        return b;
    }

    private void MarkDirty(string name)
    {
        foreach (var set in _dirty) set.Add(name);
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
    }

    private void WriteOne(uint frame, string name)
    {
        var b = _sceneBindings[name];
        var write = new WriteDescriptorSet
        {
            SType = StructureType.WriteDescriptorSet,
            DstSet = _sceneSets[frame],
            DstBinding = b.Binding,
            DescriptorType = b.Type,
            DescriptorCount = 1,
        };
        switch (_registered[name])
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
    }
}