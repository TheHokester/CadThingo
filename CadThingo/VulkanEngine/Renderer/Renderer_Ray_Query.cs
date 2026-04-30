using Silk.NET.Vulkan;
using Silk.NET.Vulkan.Extensions.KHR;

namespace CadThingo.VulkanEngine.Renderer;

public unsafe partial class Renderer
{
    // ──────────────────────────────────────────────────────────
    //  State
    // ──────────────────────────────────────────────────────────

    // Silk.NET dispatch table for VK_KHR_acceleration_structure. Loaded once via
    // TryGetDeviceExtension after the logical device is created. Null when the
    // extension wasn't enabled — every method below should early-out on null.
    private KhrAccelerationStructure? khrAccelStruct;

    // Pulled from PhysicalDeviceAccelerationStructurePropertiesKHR at startup.
    // Every scratch buffer offset passed to Cmd*BuildAccelerationStructures must be
    // a multiple of this value, otherwise validation errors with "scratchData not
    // properly aligned." Default 1 keeps math safe before LoadRayQueryExtensions runs.
    private uint asScratchAlignment = 1;

    // BLAS cache keyed by the underlying Mesh* (Mesh lives in NativeMemory and never
    // moves, so the raw pointer makes a stable key). nint instead of Mesh* because
    // Dictionary keys can't be pointers.
    private readonly Dictionary<nint, BlasEntry> blasCache = new();

    private struct BlasEntry
    {
        public AccelerationStructureKHR Handle;
        public Buffer        Storage;          // usage = AccelerationStructureStorageBitKhr | ShaderDeviceAddressBit
        public DeviceMemory  StorageMem;
        public ulong         DeviceAddress;    // from GetAccelerationStructureDeviceAddress (NOT GetBufferDeviceAddress)
    }

    // Single scene-wide TLAS. Rebuild on entity-set / transform changes; flag
    // tlasDirty so DrawFrame can pick it up at the top of a frame.
    private AccelerationStructureKHR tlas;
    private Buffer       tlasStorage;
    private DeviceMemory tlasStorageMem;

    // Instance buffer feeds Cmd*BuildAccelerationStructures with the per-instance
    // AccelerationStructureInstanceKHR records. Host-visible + coherent so we can
    // memcpy each frame; usage must include AccelerationStructureBuildInputReadOnlyBitKhr
    // and ShaderDeviceAddressBit (the build reads it via device address).
    private Buffer       tlasInstanceBuffer;
    private DeviceMemory tlasInstanceMem;
    private void*        tlasInstanceMapped;
    private uint         tlasInstanceCapacity;     // number of slots allocated, not bytes

    // Persistent scratch buffer reused across builds. Sized to the largest
    // BuildScratchSize seen so far; reallocated if a bigger build comes along.
    private Buffer       asScratchBuffer;
    private DeviceMemory asScratchMem;
    private ulong        asScratchSize;

    private bool tlasDirty = true;


    // ──────────────────────────────────────────────────────────
    //  Helpers — finished
    // ──────────────────────────────────────────────────────────

    private ulong GetBufferDeviceAddress(Buffer buffer)
    {
        BufferDeviceAddressInfo deviceAddressInfo = new()
        {
            SType = StructureType.BufferDeviceAddressInfo,
            Buffer = buffer,
        };
        return vk!.GetBufferDeviceAddress(device, &deviceAddressInfo);
    }


    // ──────────────────────────────────────────────────────────
    //  Helpers — TODO
    // ──────────────────────────────────────────────────────────

    /// <summary>
    /// Same shape as the existing CreateBuffer helper but chains
    /// MemoryAllocateFlagsInfo { flags = AddressBitKhr } into MemoryAllocateInfo.PNext.
    /// REQUIRED for any buffer you'll call vkGetBufferDeviceAddress on
    /// (BLAS storage, TLAS storage, scratch, instance). Without it the address
    /// returned is undefined and validation will yell on first use.
    ///
    /// Caller still passes ShaderDeviceAddressBit in `usage` — both the buffer
    /// usage bit AND the alloc flag are needed.
    /// </summary>
    private void CreateBufferWithDeviceAddress(
        ulong size, BufferUsageFlags usage, MemoryPropertyFlags memProps,
        out Buffer buffer, out DeviceMemory memory)
    {
        BufferCreateInfo bufferInfo = new()
        {
            SType = StructureType.BufferCreateInfo,
            Size = size,
            Usage = usage,
            SharingMode = SharingMode.Exclusive,
        };
        if (vk!.CreateBuffer(device, &bufferInfo, null, out buffer) != Result.Success)
            throw new Exception("Failed to create device-address buffer");

        vk!.GetBufferMemoryRequirements(device, buffer, out var memReqs);

        // Address bit must be on the alloc flags AND the buffer usage; without
        // it vkGetBufferDeviceAddress returns garbage.
        var flagsInfo = new MemoryAllocateFlagsInfo
        {
            SType = StructureType.MemoryAllocateFlagsInfo,
            Flags = MemoryAllocateFlags.DeviceAddressBit,
        };
        MemoryAllocateInfo allocInfo = new()
        {
            SType = StructureType.MemoryAllocateInfo,
            PNext = &flagsInfo,
            AllocationSize = memReqs.Size,
            MemoryTypeIndex = FindMemoryType(vk, physicalDevice, memReqs.MemoryTypeBits, memProps),
        };
        if (vk!.AllocateMemory(device, &allocInfo, null, out memory) != Result.Success)
            throw new Exception("Failed to allocate device-address buffer memory");

        vk!.BindBufferMemory(device, buffer, memory, 0);
    }

    /// <summary>
    /// Packs System.Numerics.Matrix4x4 into Vulkan's TransformMatrixKHR (row-major 3×4,
    /// 12 floats; 4th row implicit [0,0,0,1]).
    ///
    /// Convention mismatch: System.Numerics is row-vector (translation in M41/M42/M43);
    /// Vulkan TransformMatrixKHR is column-vector (translation at slots 3/7/11). So we
    /// TRANSPOSE while packing — writing System.Numerics columns as Vulkan rows.
    ///
    /// The geometry shader gets away with no explicit transpose only because HLSL/Slang
    /// defaults to column-major matrix layout in constant buffers, which silently
    /// re-interprets the row-major upload. The AS builder doesn't do that — it reads
    /// the 12 floats verbatim per the spec.
    /// </summary>
    private static TransformMatrixKHR ToTransformMatrixKHR(in System.Numerics.Matrix4x4 m)
    {
        TransformMatrixKHR t = default;
        t.Matrix[0]  = m.M11; t.Matrix[1]  = m.M21; t.Matrix[2]  = m.M31; t.Matrix[3]  = m.M41;
        t.Matrix[4]  = m.M12; t.Matrix[5]  = m.M22; t.Matrix[6]  = m.M32; t.Matrix[7]  = m.M42;
        t.Matrix[8]  = m.M13; t.Matrix[9]  = m.M23; t.Matrix[10] = m.M33; t.Matrix[11] = m.M43;
        return t;
    }


    // ──────────────────────────────────────────────────────────
    //  Allocator helpers (used by both BuildBlas and RebuildTlas)
    // ──────────────────────────────────────────────────────────

    /// <summary>
    /// Grows the persistent scratch buffer if `required` exceeds current size.
    /// Padded up to asScratchAlignment so any offset into the buffer satisfies
    /// the scratchData alignment rule.
    /// </summary>
    private void EnsureScratchCapacity(ulong required)
    {
        ulong padded = ((required + asScratchAlignment - 1) / asScratchAlignment) * asScratchAlignment;
        if (asScratchBuffer.Handle != 0 && asScratchSize >= padded) return;

        if (asScratchBuffer.Handle != 0) DestroyBuffer(asScratchBuffer, asScratchMem);

        CreateBufferWithDeviceAddress(padded,
            BufferUsageFlags.StorageBufferBit | BufferUsageFlags.ShaderDeviceAddressBit,
            MemoryPropertyFlags.DeviceLocalBit,
            out asScratchBuffer, out asScratchMem);
        asScratchSize = padded;
    }

    /// <summary>
    /// Grows the persistently-mapped instance buffer to hold at least
    /// `requiredInstances` AccelerationStructureInstanceKHR records. Doubles
    /// capacity (min 8) so frequent small scenes don't re-allocate every frame.
    /// </summary>
    private void EnsureInstanceCapacity(uint requiredInstances)
    {
        if (tlasInstanceCapacity >= requiredInstances) return;

        if (tlasInstanceMem.Handle != 0)
        {
            vk!.UnmapMemory(device, tlasInstanceMem);
            DestroyBuffer(tlasInstanceBuffer, tlasInstanceMem);
            tlasInstanceMapped = null;
        }

        uint capacity = 8;
        while (capacity < requiredInstances) capacity <<= 1;

        ulong sizeBytes = (ulong)capacity * (ulong)sizeof(AccelerationStructureInstanceKHR);
        CreateBufferWithDeviceAddress(sizeBytes,
            BufferUsageFlags.AccelerationStructureBuildInputReadOnlyBitKhr | BufferUsageFlags.ShaderDeviceAddressBit,
            MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit,
            out tlasInstanceBuffer, out tlasInstanceMem);

        void* mapped = null;
        vk!.MapMemory(device, tlasInstanceMem, 0, sizeBytes, 0, ref mapped);
        tlasInstanceMapped = mapped;
        tlasInstanceCapacity = capacity;
    }


    // ──────────────────────────────────────────────────────────
    //  Building blocks
    // ──────────────────────────────────────────────────────────

    private BlasEntry BuildBlas(Mesh* mesh)
    {
        // so RebuildTlas can read .DeviceAddress.

        ulong vertexAddress = GetBufferDeviceAddress(Engine.ResourceManager.GlobalVertexBuffer);
        ulong indexAddress = GetBufferDeviceAddress(Engine.ResourceManager.GlobalIndexBuffer) + (ulong)(4 * mesh->offset);

        var geo = new AccelerationStructureGeometryKHR()
        {
            SType = StructureType.AccelerationStructureGeometryKhr,
            GeometryType = GeometryTypeKHR.TrianglesKhr,
            Flags = GeometryFlagsKHR.OpaqueBitKhr,

        };
        //triangle needs SType set or else triangles will default to garbage values
        geo.Geometry.Triangles.SType = StructureType.AccelerationStructureGeometryTrianglesDataKhr;
        geo.Geometry.Triangles.VertexFormat = Format.R32G32B32Sfloat;
        geo.Geometry.Triangles.VertexStride = (ulong)sizeof(Vertex);
        geo.Geometry.Triangles.VertexData.DeviceAddress = vertexAddress;

        // MaxVertex must be >= the highest vertex INDEX referenced by this mesh's
        // index range. Indices are rebased into [0, VertexHighWater) at upload time,
        // so the global high-water mark is a conservative upper bound that's always
        // valid (just a bit wasteful — revisit when meshes need a per-mesh range).
        geo.Geometry.Triangles.MaxVertex = (uint)Engine.ResourceManager.VertexHighWater;
        geo.Geometry.Triangles.IndexType = IndexType.Uint32;
        geo.Geometry.Triangles.IndexData.DeviceAddress = indexAddress;

        var rangeInfo = new AccelerationStructureBuildRangeInfoKHR {
              PrimitiveCount = (uint)(mesh->count / 3),
              PrimitiveOffset = 0, FirstVertex = 0, TransformOffset = 0,
          };
        var buildInfo = new AccelerationStructureBuildGeometryInfoKHR {
              SType = StructureType.AccelerationStructureBuildGeometryInfoKhr,
              Type = AccelerationStructureTypeKHR.BottomLevelKhr,
              Mode = BuildAccelerationStructureModeKHR.BuildKhr,
              Flags = BuildAccelerationStructureFlagsKHR.PreferFastTraceBitKhr,
              GeometryCount = 1,
              PGeometries = &geo,
          };
        //   - Query sizes:
        uint primitiveCount = (uint)(mesh->count / 3);
        var sizes = new AccelerationStructureBuildSizesInfoKHR { SType = StructureType.AccelerationStructureBuildSizesInfoKhr };
        khrAccelStruct!.GetAccelerationStructureBuildSizes(
            device, AccelerationStructureBuildTypeKHR.DeviceKhr,
            &buildInfo, &primitiveCount, &sizes);
         
        //   - Allocate storage (CreateBufferWithDeviceAddress, AccelerationStructureStorageBitKhr | ShaderDeviceAddressBit)
        CreateBufferWithDeviceAddress(sizes.AccelerationStructureSize, 
            BufferUsageFlags.AccelerationStructureStorageBitKhr | BufferUsageFlags.ShaderDeviceAddressBit,
            MemoryPropertyFlags.DeviceLocalBit, out var storage, out var storageMem);
        
        // Grow scratch buffer (allocates if first call). Without this, the line
        // below that reads GetBufferDeviceAddress(asScratchBuffer) returns 0 and
        // the build crashes — the original `if` only updated the size variable.
        EnsureScratchCapacity(sizes.BuildScratchSize);
        var createInfo = new AccelerationStructureCreateInfoKHR {
              SType = StructureType.AccelerationStructureCreateInfoKhr,
              Buffer = storage,
              Size = sizes.AccelerationStructureSize,
              Type = AccelerationStructureTypeKHR.BottomLevelKhr,
          };
        khrAccelStruct.CreateAccelerationStructure(device, &createInfo, null, out var handle);
        buildInfo.DstAccelerationStructure = handle;
        buildInfo.ScratchData.DeviceAddress = GetBufferDeviceAddress(asScratchBuffer);
        
        //   - Single-time-command:
        var cmd = BeginSingleTimeCommands();
        var pRange = &rangeInfo;
        khrAccelStruct.CmdBuildAccelerationStructures(cmd, 1, &buildInfo, &pRange);
        EndSingleTimeCommands(cmd);
        var addrInfo = new AccelerationStructureDeviceAddressInfoKHR {
        SType = StructureType.AccelerationStructureDeviceAddressInfoKhr,
        AccelerationStructure = handle,
        };
        ulong devAddr = khrAccelStruct.GetAccelerationStructureDeviceAddress(device, &addrInfo);
        blasCache[(nint)mesh] = new BlasEntry
        {
            Handle = handle,
            Storage = storage,
            StorageMem = storageMem,
            DeviceAddress = devAddr,
        };
        return blasCache[(nint)mesh];
    }

    private void RebuildTlas()
    {
        // 1. Make sure the persistently-mapped instance buffer can hold a
        //    worst-case fill (one record per entity).
        EnsureInstanceCapacity((uint)scene.EntityCount);
        var dst = (AccelerationStructureInstanceKHR*)tlasInstanceMapped;
        uint count = 0;

        // 2. Walk entities. Pack one record per (transform + mesh) pair. Entity
        //    only has GetComponent<T> (singular) — multi-mesh entities aren't a
        //    thing yet, so one record per entity is correct.
        for (int i = 0; i < scene.EntityCount; i++)
        {
            Entity* e = scene.GetEntity(i);
            if (e == null) continue;
            var transform = e->GetComponent<TransformComponent>();
            var meshComp  = e->GetComponent<MeshComponent>();
            if (transform == null || meshComp == null || meshComp.mesh == null) continue;

            // Cache lookup; build on miss. BuildBlas writes the cache itself, so
            // a subsequent call for the same mesh would rebuild + leak — this
            // guard keeps it one-shot per mesh.
            if (!blasCache.TryGetValue((nint)meshComp.mesh, out var blas))
                blas = BuildBlas(meshComp.mesh);

            dst[count++] = new AccelerationStructureInstanceKHR
            {
                Transform                              = ToTransformMatrixKHR(*transform.GetModelMatrix()),
                InstanceCustomIndex                    = (uint)i,
                Mask                                   = 0xFF,
                InstanceShaderBindingTableRecordOffset = 0,
                Flags                                  = GeometryInstanceFlagsKHR.TriangleFacingCullDisableBitKhr,
                AccelerationStructureReference         = blas.DeviceAddress,
            };
        }

        if (count == 0)
        {
            tlasDirty = false;
            return;
        }

        // 3. Geometry — instance data lives at tlasInstanceBuffer's device address.
        uint instanceCount = count;
        var geo = new AccelerationStructureGeometryKHR
        {
            SType        = StructureType.AccelerationStructureGeometryKhr,
            GeometryType = GeometryTypeKHR.InstancesKhr,
            Flags        = GeometryFlagsKHR.OpaqueBitKhr,
        };
        geo.Geometry.Instances.SType              = StructureType.AccelerationStructureGeometryInstancesDataKhr;
        geo.Geometry.Instances.ArrayOfPointers    = false;
        geo.Geometry.Instances.Data.DeviceAddress = GetBufferDeviceAddress(tlasInstanceBuffer);

        // 4. Build info — full rebuild for now. AllowUpdateBitKhr is set so a future
        //    transform-only path can use Mode = UpdateKhr + SrcAccelerationStructure = tlas.
        var buildInfo = new AccelerationStructureBuildGeometryInfoKHR
        {
            SType         = StructureType.AccelerationStructureBuildGeometryInfoKhr,
            Type          = AccelerationStructureTypeKHR.TopLevelKhr,
            Mode          = BuildAccelerationStructureModeKHR.BuildKhr,
            Flags         = BuildAccelerationStructureFlagsKHR.PreferFastTraceBitKhr
                          | BuildAccelerationStructureFlagsKHR.AllowUpdateBitKhr,
            GeometryCount = 1,
            PGeometries   = &geo,
        };

        // 5. Size query. For TLAS, the "primitive count" is the instance count.
        var sizes = new AccelerationStructureBuildSizesInfoKHR
        {
            SType = StructureType.AccelerationStructureBuildSizesInfoKhr,
        };
        khrAccelStruct!.GetAccelerationStructureBuildSizes(
            device, AccelerationStructureBuildTypeKHR.DeviceKhr,
            &buildInfo, &instanceCount, &sizes);

        // 6. (Re)allocate TLAS storage. Free + reallocate on every rebuild until
        //    the update-mode path lands.
        if (tlas.Handle != 0)
        {
            khrAccelStruct.DestroyAccelerationStructure(device, tlas, null);
            DestroyBuffer(tlasStorage, tlasStorageMem);
        }
        CreateBufferWithDeviceAddress(sizes.AccelerationStructureSize,
            BufferUsageFlags.AccelerationStructureStorageBitKhr | BufferUsageFlags.ShaderDeviceAddressBit,
            MemoryPropertyFlags.DeviceLocalBit,
            out tlasStorage, out tlasStorageMem);

        var createInfo = new AccelerationStructureCreateInfoKHR
        {
            SType  = StructureType.AccelerationStructureCreateInfoKhr,
            Buffer = tlasStorage,
            Size   = sizes.AccelerationStructureSize,
            Type   = AccelerationStructureTypeKHR.TopLevelKhr,
        };
        khrAccelStruct.CreateAccelerationStructure(device, &createInfo, null, out tlas);

        // 7. Wire scratch + dst into buildInfo, record + submit.
        EnsureScratchCapacity(sizes.BuildScratchSize);
        buildInfo.DstAccelerationStructure  = tlas;
        buildInfo.ScratchData.DeviceAddress = GetBufferDeviceAddress(asScratchBuffer);

        var range = new AccelerationStructureBuildRangeInfoKHR
        {
            PrimitiveCount  = instanceCount,
            PrimitiveOffset = 0,
            FirstVertex     = 0,
            TransformOffset = 0,
        };
        var pRange = &range;

        var cmd = BeginSingleTimeCommands();
        khrAccelStruct.CmdBuildAccelerationStructures(cmd, 1, &buildInfo, &pRange);
        EndSingleTimeCommands(cmd);

        tlasDirty = false;
    }


    // ──────────────────────────────────────────────────────────
    //  Orchestrators
    // ──────────────────────────────────────────────────────────

    private void InitRayQuery()
    {
        if (!RayShadowsSupported) return;

        if (!vk!.TryGetDeviceExtension(instance, device, out khrAccelStruct))
        {
            Console.Error.WriteLine("[RayQuery] KhrAccelerationStructure dispatch table failed to load");
            khrAccelStruct = null;
            return;
        }

        // Pull MinAccelerationStructureScratchOffsetAlignment via the properties2 chain.
        var asProps = new PhysicalDeviceAccelerationStructurePropertiesKHR
        {
            SType = StructureType.PhysicalDeviceAccelerationStructurePropertiesKhr,
        };
        var props2 = new PhysicalDeviceProperties2
        {
            SType = StructureType.PhysicalDeviceProperties2,
            PNext = &asProps,
        };
        vk!.GetPhysicalDeviceProperties2(physicalDevice, &props2);
        asScratchAlignment = Math.Max(1, asProps.MinAccelerationStructureScratchOffsetAlignment);

        // Build BLAS for every mesh referenced by an entity. Cache lookup means
        // duplicate meshes only build once.
        for (int i = 0; i < scene.EntityCount; i++)
        {
            Entity* e = scene.GetEntity(i);
            if (e == null) continue;
            var meshComp = e->GetComponent<MeshComponent>();
            if (meshComp == null || meshComp.mesh == null) continue;
            if (blasCache.ContainsKey((nint)meshComp.mesh)) continue;
            BuildBlas(meshComp.mesh);
        }

        RebuildTlas();
    }

    private void CleanupRayQuery()
    {
        if (khrAccelStruct == null) return;

        // Unmap before freeing the host-visible instance buffer.
        if (tlasInstanceMapped != null)
        {
            vk!.UnmapMemory(device, tlasInstanceMem);
            tlasInstanceMapped = null;
        }

        if (tlas.Handle != 0)
        {
            khrAccelStruct.DestroyAccelerationStructure(device, tlas, null);
            tlas = default;
        }
        if (tlasStorage.Handle != 0)        DestroyBuffer(tlasStorage,        tlasStorageMem);
        if (tlasInstanceBuffer.Handle != 0) DestroyBuffer(tlasInstanceBuffer, tlasInstanceMem);
        if (asScratchBuffer.Handle != 0)    DestroyBuffer(asScratchBuffer,    asScratchMem);

        foreach (var entry in blasCache.Values)
        {
            khrAccelStruct.DestroyAccelerationStructure(device, entry.Handle, null);
            DestroyBuffer(entry.Storage, entry.StorageMem);
        }
        blasCache.Clear();

        khrAccelStruct.Dispose();
        khrAccelStruct = null;
    }
}