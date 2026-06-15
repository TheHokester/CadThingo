using Silk.NET.Vulkan;
using VkBuffer = Silk.NET.Vulkan.Buffer;

namespace CadThingo.VulkanEngine.Renderer;

// Block-based suballocator. Replaces direct vkAllocateMemory calls so the engine
// stops burning one slot of maxMemoryAllocationCount (4096 on most drivers) per
// VkBuffer / VkImage. Blocks are bucketed by (memoryTypeIndex, ResourceKind);
// buffer blocks unconditionally carry MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT so any
// suballocated buffer with ShaderDeviceAddressBit usage works without re-bucketing.
//
// Single-threaded - the renderer's main loop owns it. Add a lock if it grows callers.
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

// Snapshot of allocator occupancy. ReservedBytes is what we hold from the driver
// (sum of every live VkDeviceMemory block); UsedBytes is the suballocated portion
// actually backing resources. A growing Reserved-Used gap that never shrinks is the
// signature of retained empty blocks (Free never releases a fully-free pooled block).
public readonly struct AllocatorStats
{
    public readonly int   PooledBlocks;
    public readonly int   DedicatedBlocks;
    public readonly ulong ReservedBytes;
    public readonly ulong UsedBytes;

    public AllocatorStats(int pooled, int dedicated, ulong reserved, ulong used)
    {
        PooledBlocks    = pooled;
        DedicatedBlocks = dedicated;
        ReservedBytes   = reserved;
        UsedBytes       = used;
    }
}

public unsafe sealed class GpuMemoryAllocator : IDisposable
{
    // Residency-priority hints (VK_EXT_memory_priority). Range 0..1; higher = the driver
    // fights harder to keep the block resident when the process is over its WDDM/VidMm
    // budget. PriorityDefault matches Vulkan's implicit 0.5 when no priority struct is
    // chained, so anything left untagged behaves exactly as before. Folded into the bucket
    // key (quantized) so a block only ever holds one priority tier — the driver can then
    // evict a whole cold block instead of one whose priority is whatever its first tenant
    // happened to request. No-op unless memory-priority was enabled at device creation.
    public const float PriorityDefault = 0.5f;
    public const float PriorityHigh    = 0.9f;   // hot working set: render targets, PT images, BVH/TLAS, live geometry
    public const float PriorityLow     = 0.25f;  // cold / transient: staging, precompute scratch

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
        public int          BucketKey = -1;  // owning bucket (pooled only); lets Free remove it on release
        public float        Priority;        // VK_EXT_memory_priority value baked into this block
    }

    private readonly Vk             _vk;
    private readonly Device         _device;
    private readonly PhysicalDevice _physical;
    private readonly bool           _memoryPriorityEnabled;

    private readonly Dictionary<int, List<Block>> _buckets = new();
    private readonly List<Block?> _allBlocks = new();
    private readonly List<Block?> _dedicated = new();

    private const ulong BUFFER_BLOCK_DEVICE_LOCAL = 64UL  * 1024 * 1024;
    private const ulong IMAGE_BLOCK_DEVICE_LOCAL  = 256UL * 1024 * 1024;
    private const ulong HOST_VISIBLE_BLOCK        = 16UL  * 1024 * 1024;

    public GpuMemoryAllocator(Vk vk, Device device, PhysicalDevice physical, bool memoryPriorityEnabled = false)
    {
        _vk = vk;
        _device = device;
        _physical = physical;
        _memoryPriorityEnabled = memoryPriorityEnabled;
    }

    public SubAlloc AllocateForBuffer(VkBuffer buffer, MemoryPropertyFlags props, float priority = PriorityDefault)
    {
        _vk.GetBufferMemoryRequirements(_device, buffer, out var reqs);
        var alloc = Allocate(reqs, props, BucketKind.Buffer, priority);
        _vk.BindBufferMemory(_device, buffer, alloc.Memory, alloc.Offset);
        return alloc;
    }

    public SubAlloc AllocateForImage(Image image, MemoryPropertyFlags props,
        ImageTiling tiling = ImageTiling.Optimal, float priority = PriorityDefault)
    {
        _vk.GetImageMemoryRequirements(_device, image, out var reqs);
        var kind = tiling == ImageTiling.Linear ? BucketKind.ImageLinear : BucketKind.ImageOptimal;
        var alloc = Allocate(reqs, props, kind, priority);
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

        // After coalescing, a single node with no neighbours == the block is entirely
        // free. Hand it back to the driver so reverting a resize actually lowers committed
        // VRAM. Without this, the high-water reservation from a large-extent excursion
        // persists forever (Free used to only ever coalesce), which on a VRAM-pressured
        // system keeps the path-tracer working set demand-paged - a fixed per-frame
        // migration cost that survives shrinking back down. Dedicated blocks already
        // free on their own path above.
        if (n.Prev < 0 && n.Next < 0)
            ReleaseEmptyBlock(b, alloc.BlockId);
    }

    // Unlinks a fully-free pooled block from its bucket + the BlockId table and frees its
    // VkDeviceMemory. The slot in _allBlocks is nulled (not removed) so existing
    // SubAlloc.BlockId indices stay valid; a fully-free block has no live suballocations,
    // so nothing references it.
    private void ReleaseEmptyBlock(Block block, int blockId)
    {
        if (block.BucketKey >= 0 && _buckets.TryGetValue(block.BucketKey, out var list))
            list.Remove(block);
        if (blockId >= 0 && blockId < _allBlocks.Count) _allBlocks[blockId] = null;
        DestroyBlock(block);
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

    /// <summary>Walks live blocks to report reserved vs. actually-suballocated bytes +
    /// block counts. Diagnostic only (not on any hot path) — call it from the Stats panel
    /// to watch whether resize churn ratchets ReservedBytes up without releasing it.</summary>
    public AllocatorStats GetStats()
    {
        ulong reserved = 0, used = 0;
        int pooled = 0, dedicated = 0;

        foreach (var b in _allBlocks)
        {
            if (b == null) continue;
            pooled++;
            reserved += b.Size;
            foreach (var n in b.Nodes)
                if (n.Alive && !n.Free) used += n.Size;
        }
        foreach (var b in _dedicated)
        {
            if (b == null) continue;
            dedicated++;
            reserved += b.Size;
            used     += b.Size;   // a dedicated block is wholly used by its one resource
        }
        return new AllocatorStats(pooled, dedicated, reserved, used);
    }

    // internals

    private SubAlloc Allocate(MemoryRequirements reqs, MemoryPropertyFlags props, BucketKind kind, float priority)
    {
        uint typeIndex = FindMemoryType(reqs.MemoryTypeBits, props);
        bool hostVisible = (props & MemoryPropertyFlags.HostVisibleBit) != 0;
        ulong blockSize  = TargetBlockSize(kind, hostVisible);

        if (reqs.Size + reqs.Alignment >= blockSize / 2)
            return AllocateDedicated(reqs.Size, typeIndex, hostVisible, kind == BucketKind.Buffer, priority);

        // Priority tier is part of the bucket key so a block holds only same-priority
        // resources - otherwise a hot resource sharing a block with cold ones would be
        // pulled down to the block's (first-tenant) priority. typeIndex < 32 (bits 16+),
        // tier 0..10 (bits 8-15), kind 0..2 (bits 0-7) never collide.
        int tier = PriorityTier(priority);
        int bucketKey = ((int)typeIndex << 16) | (tier << 8) | (int)kind;
        if (!_buckets.TryGetValue(bucketKey, out var blockList))
        {
            blockList = new List<Block>();
            _buckets[bucketKey] = blockList;
        }

        foreach (var block in blockList)
            if (TrySuballocate(block, reqs.Size, reqs.Alignment, out var sub))
                return sub;

        ulong newBlockSize = Math.Max(blockSize, reqs.Size + reqs.Alignment);
        var fresh = CreateBlock(newBlockSize, typeIndex, hostVisible, kind == BucketKind.Buffer, priority);
        fresh.BucketKey = bucketKey;   // so Free can unlink it from this bucket when it empties
        blockList.Add(fresh);

        if (!TrySuballocate(fresh, reqs.Size, reqs.Alignment, out var freshSub))
            throw new Exception("Suballocation failed in freshly-created block — alignment math is wrong");
        return freshSub;
    }

    // Quantize 0..1 priority to 0..10 so near-equal requests share a bucket instead of
    // fragmenting into one block each. The exact (clamped) float still flows to the block.
    private static int PriorityTier(float priority) =>
        (int)MathF.Round(Math.Clamp(priority, 0f, 1f) * 10f);

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

    private Block CreateBlock(ulong size, uint memoryTypeIndex, bool hostVisible, bool bufferBlock, float priority)
    {
        var info = new MemoryAllocateInfo
        {
            SType           = StructureType.MemoryAllocateInfo,
            AllocationSize  = size,
            MemoryTypeIndex = memoryTypeIndex,
        };
        void** tail = (void**)&info.PNext;

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
            *tail = &flagsInfo;
            tail  = (void**)&flagsInfo.PNext;
        }

        // VK_EXT_memory_priority: hint how hard the driver should fight to keep this block
        // resident under WDDM budget pressure. Skipped (block gets the implicit 0.5) when
        // the extension wasn't enabled at device creation.
        MemoryPriorityAllocateInfoEXT priorityInfo = default;
        if (_memoryPriorityEnabled)
        {
            priorityInfo = new MemoryPriorityAllocateInfoEXT
            {
                SType    = StructureType.MemoryPriorityAllocateInfoExt,
                Priority = Math.Clamp(priority, 0f, 1f),
            };
            *tail = &priorityInfo;
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
            Priority        = priority,
        };
        block.Nodes.Add(new Node { Offset = 0, Size = size, Free = true, Alive = true, Prev = -1, Next = -1 });
        _allBlocks.Add(block);
        return block;
    }

    private SubAlloc AllocateDedicated(ulong size, uint memoryTypeIndex, bool hostVisible, bool bufferBlock, float priority)
    {
        var block = CreateBlock(size, memoryTypeIndex, hostVisible, bufferBlock, priority);
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

    private bool TryFindMemoryType(uint typeBits, MemoryPropertyFlags props, out uint index)
    {
        _vk.GetPhysicalDeviceMemoryProperties(_physical, out var memProps);
        for (uint i = 0; i < memProps.MemoryTypeCount; i++)
            if ((typeBits & (1u << (int)i)) != 0 &&
                (memProps.MemoryTypes[(int)i].PropertyFlags & props) == props)
            { index = i; return true; }
        index = 0;
        return false;
    }

    private uint FindMemoryType(uint typeBits, MemoryPropertyFlags props)
    {
        if (TryFindMemoryType(typeBits, props, out var idx)) return idx;
        throw new Exception("GpuMemoryAllocator: no memory type matches required properties");
    }

    /// <summary>
    /// Allocates + binds, preferring <paramref name="preferred"/> memory properties but
    /// falling back to <paramref name="fallback"/> when no memory type satisfies the
    /// preferred set (e.g. no resizable-BAR / device-local-host-visible heap). Used to
    /// place GPU-hot, host-written SSBOs in VRAM so shader reads stay off PCIe.
    /// <paramref name="usedPreferred"/> reports which set won, for diagnostics.
    /// </summary>
    public SubAlloc AllocateForBufferPreferred(VkBuffer buffer,
        MemoryPropertyFlags preferred, MemoryPropertyFlags fallback,
        out bool usedPreferred, float priority = PriorityDefault)
    {
        _vk.GetBufferMemoryRequirements(_device, buffer, out var reqs);
        usedPreferred = TryFindMemoryType(reqs.MemoryTypeBits, preferred, out _);
        var props = usedPreferred ? preferred : fallback;
        var alloc = Allocate(reqs, props, BucketKind.Buffer, priority);
        _vk.BindBufferMemory(_device, buffer, alloc.Memory, alloc.Offset);
        return alloc;
    }

    private static ulong AlignUp(ulong v, ulong a) => a <= 1 ? v : (v + a - 1) & ~(a - 1);

    private static ulong TargetBlockSize(BucketKind kind, bool hostVisible)
    {
        if (hostVisible) return HOST_VISIBLE_BLOCK;
        return kind == BucketKind.Buffer ? BUFFER_BLOCK_DEVICE_LOCAL : IMAGE_BLOCK_DEVICE_LOCAL;
    }
}
