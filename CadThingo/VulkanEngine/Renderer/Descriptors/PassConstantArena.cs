using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Descriptors;

// Per-frame bump allocator for per-pass constants (docs/descriptor-system.md section 7).
// Backs scene-set binding (0,0): one UNIFORM_BUFFER_DYNAMIC descriptor per frame points at
// this buffer with range MaxPassUniformSize; each pass Pushes its constants and binds the
// returned dynamic offset. Reset per frame from DescriptorRegistry.BeginFrame.
public sealed unsafe class PassConstantArena : IDisposable
{
    public const int Capacity = 256 * 1024;
    public const uint MaxPassUniformSize = 4096; // descriptor range; reflected (0,0) structs must fit

    private readonly GraphicsDevice _gfx;
    private readonly UboBuffer[] _buffers;
    private readonly uint[] _cursor;
    private readonly uint _alignment; // minUniformBufferOffsetAlignment

    internal PassConstantArena(GraphicsDevice gfx, uint framesInFlight)
    {
        _gfx = gfx;
        _buffers = new UboBuffer[framesInFlight];
        _cursor = new uint[framesInFlight];
        for (int i = 0; i < framesInFlight; i++)
            gfx.CreateMappedUniformBuffer(Capacity, ref _buffers[i]);

        gfx.Vk.GetPhysicalDeviceProperties(gfx.PhysicalDevice, out var props);
        _alignment = Math.Max(1, (uint)props.Limits.MinUniformBufferOffsetAlignment);
    }

    internal Silk.NET.Vulkan.Buffer Buffer(uint frame) => _buffers[frame].buffer;

    /// Copies the pass constants into this frame's arena and returns the dynamic offset to
    /// supply when binding the scene set. Only valid until Reset(frame).
    public uint Push<T>(uint frame, in T data) where T : unmanaged
    {
        uint size = (uint)sizeof(T);
        if (size > MaxPassUniformSize)
            throw new ArgumentException($"pass constants {typeof(T).Name} ({size}B) exceed MaxPassUniformSize ({MaxPassUniformSize}B)");

        uint offset = (_cursor[frame] + (_alignment - 1)) & ~(_alignment - 1);
        if (offset + size > Capacity)
            throw new InvalidOperationException($"pass constant arena exhausted (frame {frame}, {offset + size}B > {Capacity}B)");

        *(T*)((byte*)_buffers[frame].mapped + offset) = data;
        _cursor[frame] = offset + size;
        return offset;
    }

    /// Called after the frame fence wait; everything pushed for this slot is consumed.
    public void Reset(uint frame) => _cursor[frame] = 0;

    public void Dispose()
    {
        foreach (var b in _buffers) _gfx.DestroyBuffer(b.buffer, b.alloc);
    }
}