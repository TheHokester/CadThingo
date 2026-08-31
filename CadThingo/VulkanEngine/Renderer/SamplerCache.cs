using System.Text;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer;

// Central owner of VkSampler. Samplers are immutable, cheap, and drawn from a small fixed set of
// configurations, so every caller that wants the same filtering gets the same handle.
//
// Two of them are named below because they are the engine defaults: PointClamp for texel-aligned
// fullscreen reads (g-buffer resolve, tonemap input) and LinearRepeat for material textures. Anything
// else asks Get() with its own create-info.
//
// Anything handed out is destroyed by the cache, never by its caller.
public sealed unsafe class SamplerCache : IDisposable
{
    private readonly Vk _vk;
    private readonly Device _device;
    private readonly float _maxAnisotropy;

    private readonly Dictionary<string, Sampler> _samplers = new();

    private Sampler _pointClamp;
    private Sampler _linearRepeat;

    public int Count => _samplers.Count;

    public SamplerCache(Vk vk, Device device, PhysicalDevice physicalDevice)
    {
        _vk = vk;
        _device = device;
        _vk.GetPhysicalDeviceProperties(physicalDevice, out var props);
        _maxAnisotropy = props.Limits.MaxSamplerAnisotropy;
    }

    /// <summary>Nearest/clamp-to-edge point sampler for texel-aligned fullscreen reads. Every
    /// deferred g-buffer binding and the tonemap HDR input pin it as an immutable sampler.</summary>
    public Sampler PointClamp
    {
        get
        {
            if (_pointClamp.Handle == 0) _pointClamp = Get(new SamplerCreateInfo
            {
                SType                   = StructureType.SamplerCreateInfo,
                MagFilter               = Filter.Nearest,
                MinFilter               = Filter.Nearest,
                AddressModeU            = SamplerAddressMode.ClampToEdge,
                AddressModeV            = SamplerAddressMode.ClampToEdge,
                AddressModeW            = SamplerAddressMode.ClampToEdge,
                AnisotropyEnable        = true,
                MaxAnisotropy           = MathF.Min(16f, _maxAnisotropy),
                BorderColor             = BorderColor.FloatOpaqueBlack,
                UnnormalizedCoordinates = false,
                CompareEnable           = false,
                CompareOp               = CompareOp.Always,
                MipmapMode              = SamplerMipmapMode.Nearest,
                MinLod                  = 0.0f,
                MaxLod                  = 1.0f,
                MipLodBias              = 0.0f,
            });
            return _pointClamp;
        }
    }

    /// <summary>Trilinear/repeat sampler with anisotropy, for material textures. The bindless
    /// sampler array's default entry.</summary>
    public Sampler LinearRepeat
    {
        get
        {
            if (_linearRepeat.Handle == 0) _linearRepeat = Get(new SamplerCreateInfo
            {
                SType                   = StructureType.SamplerCreateInfo,
                MagFilter               = Filter.Linear,
                MinFilter               = Filter.Linear,
                AddressModeU            = SamplerAddressMode.Repeat,
                AddressModeV            = SamplerAddressMode.Repeat,
                AddressModeW            = SamplerAddressMode.Repeat,
                AnisotropyEnable        = true,
                MaxAnisotropy           = MathF.Min(16f, _maxAnisotropy),
                BorderColor             = BorderColor.FloatOpaqueBlack,
                UnnormalizedCoordinates = false,
                CompareEnable           = false,
                CompareOp               = CompareOp.Always,
                MipmapMode              = SamplerMipmapMode.Linear,
                MinLod                  = 0.0f,
                MaxLod                  = Vk.LodClampNone,
                MipLodBias              = 0.0f,
            });
            return _linearRepeat;
        }
    }

    /// <summary>Returns a sampler matching the given create-info, creating it on first request.</summary>
    /// <exception cref="ArgumentException">The info chains a pNext struct, which the key cannot
    /// describe and so could alias two different samplers onto one handle.</exception>
    public Sampler Get(in SamplerCreateInfo info)
    {
        if (info.PNext != null)
            throw new ArgumentException("SamplerCache cannot key a chained SamplerCreateInfo", nameof(info));

        string key = Key(in info);
        if (_samplers.TryGetValue(key, out var cached)) return cached;

        Sampler sampler;
        fixed (SamplerCreateInfo* pInfo = &info)
            if (_vk.CreateSampler(_device, pInfo, null, out sampler) != Result.Success)
                throw new Exception("SamplerCache: failed to create sampler");

        _samplers[key] = sampler;
        return sampler;
    }

    private static string Key(in SamplerCreateInfo i)
    {
        var sb = new StringBuilder();
        sb.Append((uint)i.Flags).Append(':')
          .Append((int)i.MagFilter).Append(':').Append((int)i.MinFilter).Append(':')
          .Append((int)i.MipmapMode).Append(':')
          .Append((int)i.AddressModeU).Append(':').Append((int)i.AddressModeV).Append(':')
          .Append((int)i.AddressModeW).Append(':')
          .Append(i.MipLodBias).Append(':')
          .Append((bool)i.AnisotropyEnable).Append(':').Append(i.MaxAnisotropy).Append(':')
          .Append((bool)i.CompareEnable).Append(':').Append((int)i.CompareOp).Append(':')
          .Append(i.MinLod).Append(':').Append(i.MaxLod).Append(':')
          .Append((int)i.BorderColor).Append(':')
          .Append((bool)i.UnnormalizedCoordinates);
        return sb.ToString();
    }

    public void Dispose()
    {
        foreach (var s in _samplers.Values) _vk.DestroySampler(_device, s, null);
        _samplers.Clear();
        _pointClamp   = default;
        _linearRepeat = default;
    }
}