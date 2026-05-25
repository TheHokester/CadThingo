using Silk.NET.Vulkan;
using VkBuffer = Silk.NET.Vulkan.Buffer;

namespace CadThingo.VulkanEngine.Renderer;

// Block-based suballocator. Replaces direct vkAllocateMemory calls so the engine
// stops burning one slot of maxMemoryAllocationCount (4096 on most drivers) per
// VkBuffer / VkImage. Blocks are bucketed by (memoryTypeIndex, ResourceKind);
// buffer blocks unconditionally carry MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT so any
// suballocated buffer with ShaderDeviceAddressBit usage works without re-bucketing.
//
// Single-threaded — the renderer's main loop owns it. Add a lock if it grows callers.
public readonly struct SubAlloc
{
    public readonly DeviceMemory Memory;
    public readonly ulong Offset;
    public readonly ulong Size;
    internal readonly int BlockId;   // -1 → dedicated (encoded as ~dedicatedIdx)
    internal readonly int NodeId;    // stable index into Block.Nodes; ignored when dedicated

    internal SubAlloc(DeviceMemory mem, ulong offset, ulong size, int blockId, int nodeId)
    {
        Memory  = mem;
        Offset  = offset;
        Size    = size;
        BlockId = blockId;
        NodeId  = nodeId;
    }

    public bool IsValid => Memory.Handle != 0;
}

public unsafe sealed class GpuMemoryAllocator : IDisposable
{
    private enum BucketKind { Buffer, ImageOptimal, ImageLinear }

    // Node slot: lives at a stable index in Block.Nodes for the slot's lifetime.
    // Removed nodes become tombstones (Alive=false) and their index goes onto the
    // block's tombstone stack for reuse — keeps indices stable for SubAlloc.NodeId.
    private struct Node
    {
        public ulong Offset;
        public ulong Size;
        public bool  Free;
        public bool  Alive;
        public int   Prev;   // doubly-linked list ordered by Offset; -1 at ends
        public int   Next;
    }

    private sealed class Block
    {
        public DeviceMemory Memory;
        public ulong        Size;
        public uint         MemoryTypeIndex;
        public bool         HostVisible;
        public void*        Mapped;          // null on device-local
        public List<Node>   Nodes = new();
        public List<int>    Tombstones = new();
        public int          Head;            // index of leftmost node; -1 if block fully torn down
    }

    private readonly Vk             _vk;
    private readonly Device         _device;
    private readonly PhysicalDevice _physical;

    private readonly Dictionary<int, List<Block>> _buckets = new();
    private readonly List<Block?> _allBlocks = new();
    private readonly List<Block?> _dedicated = new();

    private const ulong BUFFER_BLOCK_DEVICE_LOCAL = 64UL  * 1024 * 1024;
    private const ulong IMAGE_BLOCK_DEVICE_LOCAL  = 256UL * 1024 * 1024;
    private const ulong HOST_VISIBLE_BLOCK        = 16UL  * 1024 * 1024;

    public GpuMemoryAllocator(Vk vk, Device device, PhysicalDevice physical)
    {
        _vk = vk;
        _device = device;
        _physical = physical;
    }

    public SubAlloc AllocateForBuffer(VkBuffer buffer, MemoryPropertyFlags props)
    {
        _vk.GetBufferMemoryRequirements(_device, buffer, out var reqs);
        var alloc = Allocate(reqs, props, BucketKind.Buffer);
        _vk.BindBufferMemory(_device, buffer, alloc.Memory, alloc.Offset);
        return alloc;
    }

    public SubAlloc AllocateForImage(Image image, MemoryPropertyFlags props, ImageTiling tiling = ImageTiling.Optimal)
    {
        _vk.GetImageMemoryRequirements(_device, image, out var reqs);
        var kind = tiling == ImageTiling.Linear ? BucketKind.ImageLinear : BucketKind.ImageOptimal;
        var alloc = Allocate(reqs, props, kind);
        _vk.BindImageMemory(_device, image, alloc.Memory, alloc.Offset);
        return alloc;
    }

    /// <summary>
    /// Pointer to host-visible suballocation (mapped slice of the parent block).
    /// Caller must not unmap.
    /// </summary>
    public void* GetMapped(SubAlloc alloc)
    {
        Block? block = alloc.BlockId < 0 ? _dedicated[~alloc.BlockId] : _allBlocks[alloc.BlockId];
        if (block == null || block.Mapped == null)
            throw new InvalidOperationException("GetMapped called on non-host-visible alloc");
        return (byte*)block.Mapped + (long)alloc.Offset;
    }

    public void Free(SubAlloc alloc)
    {
        if (!alloc.IsValid) return;

        if (alloc.BlockId < 0)
        {
            int idx = ~alloc.BlockId;
            var dedicated = _dedicated[idx];
            if (dedicated == null) return;
            DestroyBlock(dedicated);
            _dedicated[idx] = null;
            return;
        }

        var b = _allBlocks[alloc.BlockId];
        if (b == null) return;

        int id = alloc.NodeId;
        var nodes = b.Nodes;
        var n = nodes[id];
        n.Free = true;
        nodes[id] = n;

        // Coalesce with previous.
        if (n.Prev >= 0)
        {
            var prev = nodes[n.Prev];
            if (prev.Alive && prev.Free)
            {
                prev.Size += n.Size;
                prev.Next  = n.Next;
                nodes[n.Prev] = prev;
                if (n.Next >= 0)
                {
                    var nx = nodes[n.Next];
                    nx.Prev = n.Prev;
                    nodes[n.Next] = nx;
                }
                Tombstone(b, id);
                id = n.Prev;
                n  = prev;
            }
        }
        // Coalesce with next.
        if (n.Next >= 0)
        {
            var next = nodes[n.Next];
            if (next.Alive && next.Free)
            {
                int killed = n.Next;
                n.Size += next.Size;
                n.Next  = next.Next;
                if (next.Next >= 0)
                {
                    var nn = nodes[next.Next];
                    nn.Prev = id;
                    nodes[next.Next] = nn;
                }
                Tombstone(b, killed);
                nodes[id] = n;
            }
        }
    }

    public void Dispose()
    {
        for (int i = 0; i < _allBlocks.Count; i++)
        {
            var b = _allBlocks[i];
            if (b != null) DestroyBlock(b);
        }
        for (int i = 0; i < _dedicated.Count; i++)
        {
            var b = _dedicated[i];
            if (b != null) DestroyBlock(b);
        }
        _allBlocks.Clear();
        _dedicated.Clear();
        _buckets.Clear();
    }

    // internals

    private SubAlloc Allocate(MemoryRequirements reqs, MemoryPropertyFlags props, BucketKind kind)
    {
        uint typeIndex = FindMemoryType(reqs.MemoryTypeBits, props);
        bool hostVisible = (props & MemoryPropertyFlags.HostVisibleBit) != 0;
        ulong blockSize  = TargetBlockSize(kind, hostVisible);

        if (reqs.Size + reqs.Alignment >= blockSize / 2)
            return AllocateDedicated(reqs.Size, typeIndex, hostVisible, kind == BucketKind.Buffer);

        int bucketKey = ((int)typeIndex << 8) | (int)kind;
        if (!_buckets.TryGetValue(bucketKey, out var blockList))
        {
            blockList = new List<Block>();
            _buckets[bucketKey] = blockList;
        }

        foreach (var block in blockList)
            if (TrySuballocate(block, reqs.Size, reqs.Alignment, out var sub))
                return sub;

        ulong newBlockSize = Math.Max(blockSize, reqs.Size + reqs.Alignment);
        var fresh = CreateBlock(newBlockSize, typeIndex, hostVisible, kind == BucketKind.Buffer);
        blockList.Add(fresh);

        if (!TrySuballocate(fresh, reqs.Size, reqs.Alignment, out var freshSub))
            throw new Exception("Suballocation failed in freshly-created block — alignment math is wrong");
        return freshSub;
    }

    private bool TrySuballocate(Block block, ulong size, ulong alignment, out SubAlloc result)
    {
        int id = block.Head;
        while (id >= 0)
        {
            var node = block.Nodes[id];
            if (!node.Alive)
            {
                // Shouldn't happen — Head/Next chain skips tombstones — but be defensive.
                break;
            }
            if (!node.Free)
            {
                id = node.Next;
                continue;
            }

            ulong aligned = AlignUp(node.Offset, alignment);
            ulong padding = aligned - node.Offset;
            if (node.Size < padding + size)
            {
                id = node.Next;
                continue;
            }

            int currId = id;

            // Front padding split → insert a new free node before currId.
            if (padding > 0)
            {
                int frontId = NewNode(block, node.Offset, padding, free: true, prev: node.Prev, next: currId);
                if (node.Prev >= 0)
                {
                    var p = block.Nodes[node.Prev];
                    p.Next = frontId;
                    block.Nodes[node.Prev] = p;
                }
                else
                {
                    block.Head = frontId;
                }
                node.Prev   = frontId;
                node.Offset = aligned;
                node.Size  -= padding;
            }

            // Back remainder split → insert a new free node after currId.
            if (node.Size > size)
            {
                ulong remainderSize = node.Size - size;
                int backId = NewNode(block, aligned + size, remainderSize, free: true, prev: currId, next: node.Next);
                if (node.Next >= 0)
                {
                    var nx = block.Nodes[node.Next];
                    nx.Prev = backId;
                    block.Nodes[node.Next] = nx;
                }
                node.Next = backId;
                node.Size = size;
            }

            node.Free = false;
            block.Nodes[currId] = node;
            result = new SubAlloc(block.Memory, aligned, size, AllBlockId(block), currId);
            return true;
        }

        result = default;
        return false;
    }

    private Block CreateBlock(ulong size, uint memoryTypeIndex, bool hostVisible, bool bufferBlock)
    {
        var info = new MemoryAllocateInfo
        {
            SType           = StructureType.MemoryAllocateInfo,
            AllocationSize  = size,
            MemoryTypeIndex = memoryTypeIndex,
        };
        MemoryAllocateFlagsInfo flagsInfo = default;
        if (bufferBlock)
        {
            // BufferDeviceAddress is enabled engine-wide; flagging every buffer block
            // lets buffers with/without ShaderDeviceAddressBit usage share blocks.
            flagsInfo = new MemoryAllocateFlagsInfo
            {
                SType = StructureType.MemoryAllocateFlagsInfo,
                Flags = MemoryAllocateFlags.DeviceAddressBit,
            };
            info.PNext = &flagsInfo;
        }

        if (_vk.AllocateMemory(_device, &info, null, out var memory) != Result.Success)
            throw new Exception("GpuMemoryAllocator: vkAllocateMemory failed");

        void* mapped = null;
        if (hostVisible)
        {
            void* p = null;
            _vk.MapMemory(_device, memory, 0, size, 0, ref p);
            mapped = p;
        }

        var block = new Block
        {
            Memory          = memory,
            Size            = size,
            MemoryTypeIndex = memoryTypeIndex,
            HostVisible     = hostVisible,
            Mapped          = mapped,
            Head            = 0,
        };
        block.Nodes.Add(new Node { Offset = 0, Size = size, Free = true, Alive = true, Prev = -1, Next = -1 });
        _allBlocks.Add(block);
        return block;
    }

    private SubAlloc AllocateDedicated(ulong size, uint memoryTypeIndex, bool hostVisible, bool bufferBlock)
    {
        var block = CreateBlock(size, memoryTypeIndex, hostVisible, bufferBlock);
        _allBlocks.RemoveAt(_allBlocks.Count - 1);  // detach from sub-alloc tracking
        _dedicated.Add(block);
        int dedicatedId = _dedicated.Count - 1;

        // Mark the lone node as used.
        var n = block.Nodes[0];
        n.Free = false;
        block.Nodes[0] = n;

        return new SubAlloc(block.Memory, 0, size, ~dedicatedId, 0);
    }

    private void DestroyBlock(Block block)
    {
        if (block.HostVisible && block.Mapped != null) _vk.UnmapMemory(_device, block.Memory);
        _vk.FreeMemory(_device, block.Memory, null);
    }


    private int NewNode(Block block, ulong offset, ulong size, bool free, int prev, int next)
    {
        if (block.Tombstones.Count > 0)
        {
            int id = block.Tombstones[^1];
            block.Tombstones.RemoveAt(block.Tombstones.Count - 1);
            block.Nodes[id] = new Node
            {
                Offset = offset, Size = size, Free = free, Alive = true, Prev = prev, Next = next,
            };
            return id;
        }
        block.Nodes.Add(new Node
        {
            Offset = offset, Size = size, Free = free, Alive = true, Prev = prev, Next = next,
        });
        return block.Nodes.Count - 1;
    }

    private void Tombstone(Block block, int id)
    {
        var n = block.Nodes[id];
        n.Alive = false;
        block.Nodes[id] = n;
        block.Tombstones.Add(id);
    }

    // O(n) over _allBlocks. Called only when a fresh allocation lands — not on the hot path.
    private int AllBlockId(Block block)
    {
        for (int i = 0; i < _allBlocks.Count; i++)
            if (ReferenceEquals(_allBlocks[i], block)) return i;
        throw new Exception("Block not tracked by allocator");
    }

    private uint FindMemoryType(uint typeBits, MemoryPropertyFlags props)
    {
        _vk.GetPhysicalDeviceMemoryProperties(_physical, out var memProps);
        for (uint i = 0; i < memProps.MemoryTypeCount; i++)
            if ((typeBits & (1u << (int)i)) != 0 &&
                (memProps.MemoryTypes[(int)i].PropertyFlags & props) == props)
                return i;
        throw new Exception("GpuMemoryAllocator: no memory type matches required properties");
    }

    private static ulong AlignUp(ulong v, ulong a) => a <= 1 ? v : (v + a - 1) & ~(a - 1);

    private static ulong TargetBlockSize(BucketKind kind, bool hostVisible)
    {
        if (hostVisible) return HOST_VISIBLE_BLOCK;
        return kind == BucketKind.Buffer ? BUFFER_BLOCK_DEVICE_LOCAL : IMAGE_BLOCK_DEVICE_LOCAL;
    }
}
