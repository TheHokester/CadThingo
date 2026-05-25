using System.Runtime.InteropServices;
using Silk.NET.Vulkan;
using Silk.NET.Vulkan.Extensions.KHR;

namespace CadThingo.VulkanEngine.Renderer;

// Per-entity record consumed by the PBR lighting shader's shadow-ray alpha-test
// path. One slot per TLAS instance, keyed by AccelerationStructureInstanceKHR.
// InstanceCustomIndex. Layout matches PbrShader.slang::ShadowEntityInfo.
[StructLayout(LayoutKind.Sequential)]
public struct ShadowEntityInfo
{
    public uint IndexOffset;     // uint elements into GlobalIndexBuffer for this mesh's first triangle
    public uint MaterialIndex;   // index into the materials SSBO
    public uint Flags;           // copy of PbrMaterial.Flags — bit 0 = MASK, bit 2 = BLEND
    public uint _pad0;
}

public unsafe partial class Renderer
{
    
    //  State

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
        public SubAlloc      StorageAlloc;
        public ulong         DeviceAddress;    // from GetAccelerationStructureDeviceAddress (NOT GetBufferDeviceAddress)
    }

    // Single scene-wide TLAS. Rebuild on entity-set / transform changes; flag
    // tlasDirty so DrawFrame can pick it up at the top of a frame.
    private AccelerationStructureKHR tlas;
    private Buffer    tlasStorage;
    private SubAlloc  tlasStorageAlloc;

    // Instance buffer feeds Cmd*BuildAccelerationStructures with the per-instance
    // AccelerationStructureInstanceKHR records. Host-visible + coherent so we can
    // memcpy each frame; usage must include AccelerationStructureBuildInputReadOnlyBitKhr
    // and ShaderDeviceAddressBit (the build reads it via device address).
    private Buffer    tlasInstanceBuffer;
    private SubAlloc  tlasInstanceAlloc;
    private void*     tlasInstanceMapped;
    private uint      tlasInstanceCapacity;     // number of slots allocated, not bytes

    // Persistent scratch buffer reused across builds. Sized to the largest
    // BuildScratchSize seen so far; reallocated if a bigger build comes along.
    private Buffer    asScratchBuffer;
    private SubAlloc  asScratchAlloc;
    private ulong     asScratchSize;

    // Per-entity shadow-alpha info. One ShadowEntityInfo per TLAS instance,
    // indexed by InstanceCustomIndex. Host-visible + coherent so RebuildTlas
    // writes them inline with the instance buffer. Grows alongside the instance
    // buffer; capacity is tracked separately because zero-entity scenes still
    // need a valid binding.
    private Buffer    shadowInfoBuffer;
    private SubAlloc  shadowInfoAlloc;
    private void*     shadowInfoMapped;
    private uint      shadowInfoCapacity;     // number of slots allocated, not bytes

    /// <summary>Buffer holding ShadowEntityInfo records, indexed by InstanceCustomIndex.
    /// PbrDeferredPipeline binds this on its shadow-alpha descriptor set. Returns
    /// a zero handle until InitRayQuery has run.</summary>
    public Buffer ShadowInfoBuffer => shadowInfoBuffer;
    public ulong ShadowInfoBufferSize =>
        (ulong)shadowInfoCapacity * (ulong)sizeof(ShadowEntityInfo);

    /// <summary>Set by RebuildTlas whenever EnsureShadowInfoCapacity reallocates
    /// the underlying VkBuffer — the renderer reads this after RebuildTlas and
    /// re-writes the PBR pipeline's shadow-alpha descriptor when true.</summary>
    private bool shadowInfoBufferResized;

    private bool tlasDirty = true;

    /// <summary>
    /// Flags the TLAS as stale. Consumed at the top of DrawFrame which runs a
    /// single <see cref="OnSceneEntitiesChanged"/> per frame regardless of how
    /// many edits accumulated the previous frame. Use this from the editor side
    /// (InspectorPanel transforms, FileBrowserPanel visibility) — direct
    /// per-mutation RebuildTlas calls would stall the device on every slider tick.
    /// </summary>
    public void MarkTlasDirty() => tlasDirty = true;
    public bool IsTlasDirty => tlasDirty;


    //  Helpers — finished

    private ulong GetBufferDeviceAddress(Buffer buffer)
    {
        BufferDeviceAddressInfo deviceAddressInfo = new()
        {
            SType = StructureType.BufferDeviceAddressInfo,
            Buffer = buffer,
        };
        return vk!.GetBufferDeviceAddress(device, &deviceAddressInfo);
    }


    // 
    //  Helpers — TODO
    // 

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
        out Buffer buffer, out SubAlloc alloc)
    {
        // Allocator buffer blocks unconditionally carry MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
        // so this is now just a buffer create + bind. Caller still owns the
        // ShaderDeviceAddressBit usage flag — the buffer needs both halves.
        BufferCreateInfo bufferInfo = new()
        {
            SType = StructureType.BufferCreateInfo,
            Size = size,
            Usage = usage,
            SharingMode = SharingMode.Exclusive,
        };
        if (vk!.CreateBuffer(device, &bufferInfo, null, out buffer) != Result.Success)
            throw new Exception("Failed to create device-address buffer");
        alloc = memAllocator.AllocateForBuffer(buffer, memProps);
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


    //
    //  Allocator helpers (used by both BuildBlas and RebuildTlas)
    //

    /// <summary>
    /// Grows the persistent scratch buffer if `required` exceeds current size.
    /// Padded up to asScratchAlignment so any offset into the buffer satisfies
    /// the scratchData alignment rule.
    /// </summary>
    private void EnsureScratchCapacity(ulong required)
    {
        ulong padded = ((required + asScratchAlignment - 1) / asScratchAlignment) * asScratchAlignment;
        if (asScratchBuffer.Handle != 0 && asScratchSize >= padded) return;

        if (asScratchBuffer.Handle != 0) DestroyBuffer(asScratchBuffer, asScratchAlloc);

        CreateBufferWithDeviceAddress(padded,
            BufferUsageFlags.StorageBufferBit | BufferUsageFlags.ShaderDeviceAddressBit,
            MemoryPropertyFlags.DeviceLocalBit,
            out asScratchBuffer, out asScratchAlloc);
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

        if (tlasInstanceAlloc.IsValid)
        {
            DestroyBuffer(tlasInstanceBuffer, tlasInstanceAlloc);
            tlasInstanceMapped = null;
        }

        uint capacity = 8;
        while (capacity < requiredInstances) capacity <<= 1;

        ulong sizeBytes = (ulong)capacity * (ulong)sizeof(AccelerationStructureInstanceKHR);
        CreateBufferWithDeviceAddress(sizeBytes,
            BufferUsageFlags.AccelerationStructureBuildInputReadOnlyBitKhr | BufferUsageFlags.ShaderDeviceAddressBit,
            MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit,
            out tlasInstanceBuffer, out tlasInstanceAlloc);

        tlasInstanceMapped = memAllocator.GetMapped(tlasInstanceAlloc);
        tlasInstanceCapacity = capacity;
    }

    /// <summary>
    /// Mirror of EnsureInstanceCapacity for the ShadowEntityInfo SSBO. Returns
    /// true iff the underlying VkBuffer was (re-)allocated — the caller must
    /// re-write the PBR pipeline's shadow-alpha descriptor set in that case.
    /// </summary>
    private bool EnsureShadowInfoCapacity(uint requiredInstances)
    {
        if (shadowInfoCapacity >= requiredInstances && shadowInfoBuffer.Handle != 0)
            return false;

        if (shadowInfoAlloc.IsValid)
        {
            DestroyBuffer(shadowInfoBuffer, shadowInfoAlloc);
            shadowInfoMapped = null;
        }

        uint capacity = 8;
        while (capacity < requiredInstances) capacity <<= 1;

        ulong sizeBytes = (ulong)capacity * (ulong)sizeof(ShadowEntityInfo);
        CreateBuffer(sizeBytes, BufferUsageFlags.StorageBufferBit,
            MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit,
            out shadowInfoBuffer, out shadowInfoAlloc);

        shadowInfoMapped = memAllocator.GetMapped(shadowInfoAlloc);
        shadowInfoCapacity = capacity;
        return true;
    }


    //
    //  Building blocks
    //

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
            MemoryPropertyFlags.DeviceLocalBit, out var storage, out var storageAlloc);
        
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
            StorageAlloc = storageAlloc,
            DeviceAddress = devAddr,
        };
        return blasCache[(nint)mesh];
    }
    //tlas previous entry count(ensures that if all are removed old tlas isnt used)
    private static uint PreviousCount = 0;
    private void RebuildTlas()
    {
        // 1. Make sure the persistently-mapped instance buffer can hold a
        //    worst-case fill (one record per entity). Mirror that capacity into
        //    the ShadowEntityInfo SSBO so InstanceCustomIndex stays a direct index.
        EnsureInstanceCapacity((uint)scene.EntityCount);
        shadowInfoBufferResized = EnsureShadowInfoCapacity((uint)scene.EntityCount);
        var dst    = (AccelerationStructureInstanceKHR*)tlasInstanceMapped;
        var sDst   = (ShadowEntityInfo*)shadowInfoMapped;
        uint count = 0;

        // 2. Walk entities. Pack one record per (transform + mesh) pair. Entity
        //    only has GetComponent<T> (singular) — multi-mesh entities aren't a
        //    thing yet, so one record per entity is correct.
        //
        //    TLAS instances are compacted (`count` grows only for renderables),
        //    but ShadowEntityInfo MUST be keyed by entity index `i` because the
        //    shadow ray reads it via CandidateInstanceID() which returns the
        //    InstanceCustomIndex we set below (= `i`). Writing it by `count`
        //    instead silently misaligns the lookup whenever a non-renderable
        //    entity (light-only, transform-only, etc.) sits earlier in the list —
        //    so we default-init the i-th slot up front and overwrite when valid.
        for (int i = 0; i < scene.EntityCount; i++)
        {
            sDst[i] = default;

            Entity* e = scene.GetEntity(i);
            
            if(!e->IsActive) continue;
            
            if (e == null) continue;
            var transform = e->GetComponent<TransformComponent>();
            var meshComp  = e->GetComponent<MeshComponent>();
            if (transform == null || meshComp == null || meshComp.mesh == null) continue;

            // Cache lookup; build on miss. BuildBlas writes the cache itself, so
            // a subsequent call for the same mesh would rebuild + leak — this
            // guard keeps it one-shot per mesh.
            if (!blasCache.TryGetValue((nint)meshComp.mesh, out var blas))
                blas = BuildBlas(meshComp.mesh);

            // Lookup the entity's material so we can flag non-opaque instances
            // (MASK + BLEND + KHR_materials_transmission) as ForceNoOpaque —
            // that's what gives the ray query a chance to alpha-test / fresnel-
            // test in the Proceed loop on the GPU. Materials are scene-owned so
            // this is a cheap host lookup, no GPU readback.
            uint  matFlags        = 0u;
            float matTransmission = 0f;
            int   matIdx          = meshComp.materialIndex;
            if (matIdx >= 0)
            {
                var mat = scene.GetMaterial(matIdx);
                matFlags        = mat.Flags;
                matTransmission = mat.TransmissionFactor;
            }

            var instFlags = GeometryInstanceFlagsKHR.TriangleFacingCullDisableBitKhr;
            // Mask=0x1, Blend=0x4 → either alpha mode flips the instance to
            // non-opaque so the Proceed loop can stochastic-test alpha.
            // Transmission > 0 likewise — the PT shadow ray + traceRay both
            // need to see the candidate so transmissive surfaces can let the
            // ray pass through (or refract on a hit-commit).
            if ((matFlags & 5u) != 0u || matTransmission > 0f)
                instFlags |= GeometryInstanceFlagsKHR.ForceNoOpaqueBitKhr;

            sDst[i] = new ShadowEntityInfo
            {
                IndexOffset   = (uint)meshComp.mesh->offset,
                MaterialIndex = matIdx >= 0 ? (uint)matIdx : 0u,
                Flags         = matFlags,
            };

            dst[count++] = new AccelerationStructureInstanceKHR
            {
                // World matrix, not local — must match the per-instance ModelMatrix
                // the geometry pass uploads (Renderer_Compute.cs:235 and the gbuffer
                // shader). With only local here, child entities in the scenegraph
                // get their BLAS placed at the wrong world position and rays from
                // the correct world-space gbuffer surface miss them entirely.
                Transform                              = ToTransformMatrixKHR(*transform.GetWorldMatrix()),
                InstanceCustomIndex                    = (uint)i,
                Mask                                   = 0xFF,
                InstanceShaderBindingTableRecordOffset = 0,
                Flags                                  = instFlags,
                AccelerationStructureReference         = blas.DeviceAddress,
            };
        }
        //if 
        if (count == 0 && PreviousCount == 0)
        {
            tlasDirty = false;
            return;
        }
        PreviousCount = count;
        
        
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
            DestroyBuffer(tlasStorage, tlasStorageAlloc);
        }
        CreateBufferWithDeviceAddress(sizes.AccelerationStructureSize,
            BufferUsageFlags.AccelerationStructureStorageBitKhr | BufferUsageFlags.ShaderDeviceAddressBit,
            MemoryPropertyFlags.DeviceLocalBit,
            out tlasStorage, out tlasStorageAlloc);

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


    //  Orchestrators

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

    /// <summary>
    /// Destroys cached BLAS entries for every mesh pointer in
    /// <paramref name="meshPtrs"/>. Pairs with file destroy in the editor.
    /// Caller is responsible for DeviceWaitIdle before calling (typically via
    /// the editor's destroy path, which also frees mesh/texture resources).
    /// </summary>
    public void DestroyBlasFor(IEnumerable<nint> meshPtrs)
    {
        if (khrAccelStruct == null) return;
        foreach (var ptr in meshPtrs)
        {
            if (!blasCache.TryGetValue(ptr, out var entry)) continue;
            khrAccelStruct.DestroyAccelerationStructure(device, entry.Handle, null);
            DestroyBuffer(entry.Storage, entry.StorageAlloc);
            blasCache.Remove(ptr);
        }
    }

    /// <summary>
    /// Rebuilds BLAS for any newly-seen meshes and re-runs RebuildTlas. Re-writes
    /// the TLAS and (if its underlying VkBuffer reallocated) the ShadowEntityInfo
    /// descriptor on every consumer pipeline. Safe to call after a scene-edit
    /// from the editor; performs a DeviceWaitIdle internally so it doesn't race
    /// in-flight command buffers.
    /// </summary>
    public void OnSceneEntitiesChanged()
    {
        if (!initialized) return;
        // Pathtracer always cares because changing the scene invalidates the
        // accumulator regardless of whether ray shadows are supported.
        MarkAccumulatorDirty();

        if (!RayShadowsSupported || khrAccelStruct == null) return;

        vk!.DeviceWaitIdle(device);

        // Build BLAS for any meshes that joined the scene since last build.
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

        // TLAS handle changes on every rebuild — re-bind it on every consumer
        // descriptor set even when the shadow-info buffer didn't grow.
        if (tlas.Handle != 0)
        {
            PbrDeferredPipeline?.WriteTlasDescriptor(tlas);
            transparentPipeline?.WriteTlasDescriptor(tlas);
            ptComputePipeline?.WriteTlasDescriptor(tlas);
        }

        if (shadowInfoBufferResized)
        {
            PbrDeferredPipeline?.WriteShadowAlphaDescriptors();
            ptComputePipeline?.WriteShadowInfoDescriptor();
            shadowInfoBufferResized = false;
        }
    }

    private void CleanupRayQuery()
    {
        if (khrAccelStruct == null) return;

        // Host-visible mappings live for the lifetime of the parent block (the
        // allocator owns the map/unmap). Just null the pointers — Free below
        // releases the suballocation; the block stays mapped until allocator dispose.
        tlasInstanceMapped = null;
        shadowInfoMapped   = null;

        if (shadowInfoBuffer.Handle != 0) DestroyBuffer(shadowInfoBuffer, shadowInfoAlloc);

        if (tlas.Handle != 0)
        {
            khrAccelStruct.DestroyAccelerationStructure(device, tlas, null);
            tlas = default;
        }
        if (tlasStorage.Handle != 0)        DestroyBuffer(tlasStorage,        tlasStorageAlloc);
        if (tlasInstanceBuffer.Handle != 0) DestroyBuffer(tlasInstanceBuffer, tlasInstanceAlloc);
        if (asScratchBuffer.Handle != 0)    DestroyBuffer(asScratchBuffer,    asScratchAlloc);

        foreach (var entry in blasCache.Values)
        {
            khrAccelStruct.DestroyAccelerationStructure(device, entry.Handle, null);
            DestroyBuffer(entry.Storage, entry.StorageAlloc);
        }
        blasCache.Clear();

        khrAccelStruct.Dispose();
        khrAccelStruct = null;
    }
}