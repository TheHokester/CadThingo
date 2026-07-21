using System.Text;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Pipelines;

// Central owner of pipeline/descriptor-set layouts and of the one device-level VkPipelineCache.
//
// The pipeline cache is what pays: pipelines used to pass a null cache handle, so every PSO
// recompiled cold each launch. Persisting one shared cache saves ~0.5s on warm start.
//
// The layout dedupe measures 14/15 and 19/19 distinct - one hit total. Scene and feature
// layouts, the ones that actually recur, are owned by DescriptorRegistry and never come here;
// what does is per-pipeline pass sets, unique by construction. Kept for the centralised
// ownership (teardown no longer rides on per-pipeline owned-index bookkeeping), not for speed.
//
// Anything handed out is destroyed by the cache, never by its caller; PipelineBase asks Owns()
// first, so a hand-built layout stays the pipeline's.
public sealed unsafe class PipelineLayoutCache : IDisposable
{
    private readonly Vk _vk;
    private readonly Device _device;

    private readonly Dictionary<string, DescriptorSetLayout> _setLayouts = new();
    private readonly Dictionary<string, PipelineLayout> _pipelineLayouts = new();
    private readonly HashSet<ulong> _ownedSetLayouts = new();
    private readonly HashSet<ulong> _ownedPipelineLayouts = new();

    private PipelineCache _pipelineCache;
    private readonly string _cachePath;

    private int _setLayoutRequests;
    private int _pipelineLayoutRequests;

    public int SetLayoutCount => _setLayouts.Count;
    public int PipelineLayoutCount => _pipelineLayouts.Count;
    public int SetLayoutRequests => _setLayoutRequests;
    public int PipelineLayoutRequests => _pipelineLayoutRequests;

    /// The shared VkPipelineCache every PSO build feeds. Warm across runs (persisted on dispose),
    /// so a second launch skips most driver-side shader compilation.
    public PipelineCache PipelineCache => _pipelineCache;

    public PipelineLayoutCache(Vk vk, Device device, PhysicalDevice physicalDevice, string cachePath)
    {
        _vk = vk;
        _device = device;
        _cachePath = cachePath;
        CreatePipelineCache(physicalDevice);
    }

    // ---- descriptor set layouts ---------------------------------------------------------

    /// <summary>Returns a layout matching the given bindings, creating it on first request.
    /// <paramref name="immutableSamplers"/> is parallel to <paramref name="bindings"/>; a non-null
    /// entry is baked into that binding. The samplers are part of the key - two layouts that agree
    /// on every binding but pin different samplers are genuinely different layouts. Callers must
    /// leave PImmutableSamplers unset; this method fills it, so the pointers stay valid for exactly
    /// the create call.</summary>
    public DescriptorSetLayout GetSetLayout(DescriptorSetLayoutBinding[] bindings, Sampler?[] immutableSamplers)
    {
        _setLayoutRequests++;
        string key = SetLayoutKey(bindings, immutableSamplers);
        if (_setLayouts.TryGetValue(key, out var cached)) return cached;

        // Flat array pinned for the whole create call: taking the address of an element requires
        // it to stay put, and Vulkan reads through PImmutableSamplers during vkCreate.
        var samplers = new Sampler[bindings.Length];
        fixed (Sampler* pSamplers = samplers)
        {
            for (int i = 0; i < bindings.Length; i++)
            {
                if (immutableSamplers[i] is not { } s) continue;
                samplers[i] = s;
                bindings[i].PImmutableSamplers = &pSamplers[i];
            }

            fixed (DescriptorSetLayoutBinding* pBindings = bindings)
            {
                var info = new DescriptorSetLayoutCreateInfo
                {
                    SType = StructureType.DescriptorSetLayoutCreateInfo,
                    BindingCount = (uint)bindings.Length,
                    PBindings = pBindings,
                };
                if (_vk.CreateDescriptorSetLayout(_device, &info, null, out var layout) != Result.Success)
                    throw new Exception("PipelineLayoutCache: failed to create descriptor set layout");

                _setLayouts[key] = layout;
                _ownedSetLayouts.Add(layout.Handle);
                return layout;
            }
        }
    }

    private static string SetLayoutKey(DescriptorSetLayoutBinding[] bindings, Sampler?[] samplers)
    {
        var sb = new StringBuilder();
        for (int i = 0; i < bindings.Length; i++)
        {
            var b = bindings[i];
            sb.Append(b.Binding).Append(':')
              .Append((int)b.DescriptorType).Append(':')
              .Append(b.DescriptorCount).Append(':')
              .Append((uint)b.StageFlags).Append(':')
              .Append(samplers[i]?.Handle ?? 0).Append('|');
        }
        return sb.ToString();
    }

    // ---- pipeline layouts ---------------------------------------------------------------

    /// Returns a pipeline layout for the given set layouts + push ranges, creating it on first
    /// request. Keying on set-layout HANDLES is only sound because identical descriptions already
    /// collapsed upstream - here and in DescriptorRegistry, which hands out one shared handle per
    /// scene/feature layout. Two matching-but-distinct handles would key apart and just miss.
    public PipelineLayout Get(ReadOnlySpan<DescriptorSetLayout> sets, ReadOnlySpan<PushConstantRange> push)
    {
        _pipelineLayoutRequests++;
        var sb = new StringBuilder();
        foreach (var s in sets) sb.Append(s.Handle).Append(',');
        sb.Append(';');
        foreach (var p in push) sb.Append((uint)p.StageFlags).Append(':').Append(p.Offset).Append(':').Append(p.Size).Append(',');
        string key = sb.ToString();

        if (_pipelineLayouts.TryGetValue(key, out var cached)) return cached;

        fixed (DescriptorSetLayout* pSets = sets)
        fixed (PushConstantRange* pPush = push)
        {
            var info = new PipelineLayoutCreateInfo
            {
                SType = StructureType.PipelineLayoutCreateInfo,
                SetLayoutCount = (uint)sets.Length,
                PSetLayouts = pSets,
                PushConstantRangeCount = (uint)push.Length,
                PPushConstantRanges = pPush,
            };
            if (_vk.CreatePipelineLayout(_device, &info, null, out var layout) != Result.Success)
                throw new Exception("PipelineLayoutCache: failed to create pipeline layout");

            _pipelineLayouts[key] = layout;
            _ownedPipelineLayouts.Add(layout.Handle);
            return layout;
        }
    }

    /// True if this cache created the handle, and so is responsible for destroying it. Pipelines
    /// consult this before tearing down a layout, since a cached one is shared.
    public bool Owns(DescriptorSetLayout layout) => _ownedSetLayouts.Contains(layout.Handle);
    public bool Owns(PipelineLayout layout) => _ownedPipelineLayouts.Contains(layout.Handle);

    // ---- device pipeline cache ------------------------------------------------------------

    // Seeds the cache from disk when the blob matches this driver + device, so PSO creation can
    // reuse the previous run's compilation. A mismatched or corrupt blob is dropped rather than
    // handed to the driver: the spec says implementations validate the header and ignore bad
    // data, but not every driver is graceful about a truncated file.
    private void CreatePipelineCache(PhysicalDevice physicalDevice)
    {
        byte[]? initial = null;
        try
        {
            if (File.Exists(_cachePath))
            {
                var blob = File.ReadAllBytes(_cachePath);
                if (IsCompatible(blob, physicalDevice)) initial = blob;
                else Console.WriteLine("[pipeline-cache] on-disk blob is for a different driver/device; starting cold");
            }
        }
        catch (IOException) { /* unreadable cache is not fatal - start cold */ }

        fixed (byte* pInitial = initial)
        {
            var info = new PipelineCacheCreateInfo
            {
                SType = StructureType.PipelineCacheCreateInfo,
                InitialDataSize = (nuint)(initial?.Length ?? 0),
                PInitialData = pInitial,
            };
            if (_vk.CreatePipelineCache(_device, &info, null, out _pipelineCache) != Result.Success)
                throw new Exception("PipelineLayoutCache: failed to create pipeline cache");
        }

        Console.WriteLine(initial != null
            ? $"[pipeline-cache] warm start from {initial.Length / 1024}KB blob"
            : "[pipeline-cache] cold start");
    }

    // VkPipelineCacheHeaderVersionOne: headerSize, headerVersion, vendorID, deviceID, then a
    // 16-byte cache UUID. All must match the current device or the payload is useless.
    private bool IsCompatible(byte[] blob, PhysicalDevice physicalDevice)
    {
        const int headerBytes = 32;
        if (blob.Length < headerBytes) return false;

        PhysicalDeviceProperties props;
        _vk.GetPhysicalDeviceProperties(physicalDevice, &props);

        uint headerVersion = BitConverter.ToUInt32(blob, 4);
        uint vendorId = BitConverter.ToUInt32(blob, 8);
        uint deviceId = BitConverter.ToUInt32(blob, 12);
        if (headerVersion != (uint)PipelineCacheHeaderVersion.One) return false;
        if (vendorId != props.VendorID || deviceId != props.DeviceID) return false;

        for (int i = 0; i < 16; i++)
            if (blob[16 + i] != props.PipelineCacheUuid[i]) return false;
        return true;
    }

    // Writes the driver's accumulated compilation back out. Best-effort: a failure here costs the
    // next run some warm-up, nothing more, so it never propagates out of teardown.
    private void SavePipelineCache()
    {
        try
        {
            nuint size = 0;
            if (_vk.GetPipelineCacheData(_device, _pipelineCache, ref size, null) != Result.Success || size == 0) return;

            var blob = new byte[size];
            fixed (byte* pBlob = blob)
                if (_vk.GetPipelineCacheData(_device, _pipelineCache, ref size, pBlob) != Result.Success) return;

            Directory.CreateDirectory(Path.GetDirectoryName(_cachePath)!);
            File.WriteAllBytes(_cachePath, blob);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException) { }
    }

    public void Dispose()
    {
        SavePipelineCache();
        if (_pipelineCache.Handle != 0) _vk.DestroyPipelineCache(_device, _pipelineCache, null);

        foreach (var layout in _pipelineLayouts.Values) _vk.DestroyPipelineLayout(_device, layout, null);
        foreach (var layout in _setLayouts.Values) _vk.DestroyDescriptorSetLayout(_device, layout, null);
        _pipelineLayouts.Clear();
        _setLayouts.Clear();
        _ownedPipelineLayouts.Clear();
        _ownedSetLayouts.Clear();
    }
}