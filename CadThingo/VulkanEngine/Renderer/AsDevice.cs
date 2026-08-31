using Silk.NET.Vulkan;
using Silk.NET.Vulkan.Extensions.KHR;

namespace CadThingo.VulkanEngine.Renderer;

/// <summary>
/// Acceleration-structure verbs, reached as <c>gfx.As.*</c>. Everything here is a thin, stateless
/// wrapper over VK_KHR_acceleration_structure at the same altitude as <c>CreateBuffer</c> /
/// <c>CreateImage</c>: create/destroy an AS object, ask its device address, ask the builder how big
/// a build will be, run a build, compact one. The test each of these passes is "a renderer with a
/// completely different BLAS strategy would reuse this unchanged".
///
/// What is deliberately NOT here is everything that decides WHAT to build: clustering, rebuild vs
/// refit, compaction scheduling, instance-record packing, and the buffers those need (scratch,
/// instances, per-geometry transforms). That is renderer policy and lives with the AS owner.
///
/// A grouped facet rather than a dozen more methods on GraphicsDevice: AS work is a coherent
/// cluster used by one subsystem, and <c>gfx.As.Build(...)</c> reads better than
/// <c>gfx.BuildAccelerationStructures(...)</c> at every call site.
///
/// Owns no Vulkan objects beyond the dispatch table, so there is nothing to tear down but that.
/// </summary>
public sealed unsafe class AsDevice : IDisposable
{
    private readonly GraphicsDevice _gfx;
    private readonly Vk _vk;

    // Silk.NET dispatch table for VK_KHR_acceleration_structure, loaded once after the logical
    // device exists. Null when the extension was not enabled - every verb below early-outs or
    // throws rather than dereferencing it, and callers gate on Available.
    private KhrAccelerationStructure? _khr;

    private uint _scratchAlignment = 1;

    internal AsDevice(GraphicsDevice gfx, Vk vk)
    {
        _gfx = gfx;
        _vk  = vk;
    }

    /// <summary>True once the dispatch table is loaded. False means the device never enabled
    /// VK_KHR_acceleration_structure (or the table failed to load) and no AS verb is callable.</summary>
    public bool Available => _khr != null;

    /// <summary>Every scratch-buffer offset handed to a build must be a multiple of this
    /// (<c>minAccelerationStructureScratchOffsetAlignment</c>), or the build fails validation with
    /// "scratchData not properly aligned". 1 until <see cref="Load"/> runs, which keeps the
    /// padding math safe before then.</summary>
    public uint ScratchAlignment => _scratchAlignment;

    /// <summary>Loads the dispatch table and caches the scratch alignment. Called by
    /// GraphicsDevice.Initialize once the logical device exists; a no-op when the extension is not
    /// enabled, which leaves <see cref="Available"/> false.</summary>
    internal void Load(Instance instance, Device device, PhysicalDevice physicalDevice)
    {
        if (!_vk.TryGetDeviceExtension(instance, device, out _khr))
        {
            Console.Error.WriteLine("[AsDevice] KhrAccelerationStructure dispatch table failed to load");
            _khr = null;
            return;
        }

        var asProps = new PhysicalDeviceAccelerationStructurePropertiesKHR
        {
            SType = StructureType.PhysicalDeviceAccelerationStructurePropertiesKhr,
        };
        var props2 = new PhysicalDeviceProperties2
        {
            SType = StructureType.PhysicalDeviceProperties2,
            PNext = &asProps,
        };
        _vk.GetPhysicalDeviceProperties2(physicalDevice, &props2);
        _scratchAlignment = Math.Max(1, asProps.MinAccelerationStructureScratchOffsetAlignment);
    }

    /// <summary>Creates an AS object of <paramref name="size"/> bytes backed by
    /// <paramref name="storage"/>. The storage buffer must carry
    /// AccelerationStructureStorageBitKhr | ShaderDeviceAddressBit and be device-local; sizing it
    /// is <see cref="GetBuildSizes"/>'s job.</summary>
    public AccelerationStructureKHR Create(Buffer storage, ulong size, AccelerationStructureTypeKHR type)
    {
        var createInfo = new AccelerationStructureCreateInfoKHR
        {
            SType  = StructureType.AccelerationStructureCreateInfoKhr,
            Buffer = storage,
            Size   = size,
            Type   = type,
        };
        _khr!.CreateAccelerationStructure(_gfx.Device, &createInfo, null, out var handle);
        return handle;
    }

    /// <summary>Destroys an AS object. Does NOT free its storage buffer - the caller owns that and
    /// frees it separately. Safe on a zero handle.</summary>
    public void Destroy(AccelerationStructureKHR handle)
    {
        if (_khr == null || handle.Handle == 0) return;
        _khr.DestroyAccelerationStructure(_gfx.Device, handle, null);
    }

    /// <summary>The AS's own device address, for a TLAS instance's
    /// AccelerationStructureReference. This is NOT the storage buffer's address -
    /// vkGetBufferDeviceAddress on the storage gives a different (wrong) value.</summary>
    public ulong DeviceAddress(AccelerationStructureKHR handle)
    {
        var addrInfo = new AccelerationStructureDeviceAddressInfoKHR
        {
            SType                 = StructureType.AccelerationStructureDeviceAddressInfoKhr,
            AccelerationStructure = handle,
        };
        return _khr!.GetAccelerationStructureDeviceAddress(_gfx.Device, &addrInfo);
    }

    /// <summary>Asks the driver how much AS storage and build scratch
    /// <paramref name="buildInfo"/> needs. <paramref name="maxPrimitiveCounts"/> is one count per
    /// geometry in the build (for a TLAS: one entry, the instance count).</summary>
    public AccelerationStructureBuildSizesInfoKHR GetBuildSizes(
        ref AccelerationStructureBuildGeometryInfoKHR buildInfo, uint* maxPrimitiveCounts)
    {
        var sizes = new AccelerationStructureBuildSizesInfoKHR
        {
            SType = StructureType.AccelerationStructureBuildSizesInfoKhr,
        };
        fixed (AccelerationStructureBuildGeometryInfoKHR* pInfo = &buildInfo)
        {
            _khr!.GetAccelerationStructureBuildSizes(_gfx.Device,
                AccelerationStructureBuildTypeKHR.DeviceKhr, pInfo, maxPrimitiveCounts, &sizes);
        }
        return sizes;
    }

    /// <summary>Records and submits one AS build, blocking until it completes.
    /// <paramref name="buildInfo"/> must already carry its destination AS and scratch address;
    /// <paramref name="ranges"/> is the per-geometry range array. Discrete single-time submit -
    /// fine at edit/load cadence, which is the only time builds run; batching several builds into
    /// one submit is a policy change for the AS owner to make, not this layer.</summary>
    public void Build(ref AccelerationStructureBuildGeometryInfoKHR buildInfo,
        AccelerationStructureBuildRangeInfoKHR* ranges)
    {
        fixed (AccelerationStructureBuildGeometryInfoKHR* pInfo = &buildInfo)
        {
            var pRanges = ranges;
            var cmd = _gfx.BeginSingleTimeCommands();
            _khr!.CmdBuildAccelerationStructures(cmd, 1, pInfo, &pRanges);
            _gfx.EndSingleTimeCommands(cmd);
        }
    }

    /// <summary>Post-build compacted size of <paramref name="src"/>, which must have been built
    /// with AllowCompactionBitKhr. Blocks: the query result is only available after the build
    /// completes, so this runs its own single-time submit and waits on the result.
    /// Returns 0 if the driver reports nothing useful.</summary>
    public ulong QueryCompactedSize(AccelerationStructureKHR src)
    {
        var qpInfo = new QueryPoolCreateInfo
        {
            SType      = StructureType.QueryPoolCreateInfo,
            QueryType  = QueryType.AccelerationStructureCompactedSizeKhr,
            QueryCount = 1,
        };
        _vk.CreateQueryPool(_gfx.Device, &qpInfo, null, out var pool);

        var handle = src;
        var cmd = _gfx.BeginSingleTimeCommands();
        _vk.CmdResetQueryPool(cmd, pool, 0, 1);
        _khr!.CmdWriteAccelerationStructuresProperties(cmd, 1, &handle,
            QueryType.AccelerationStructureCompactedSizeKhr, pool, 0);
        _gfx.EndSingleTimeCommands(cmd);

        ulong compactedSize = 0;
        _vk.GetQueryPoolResults(_gfx.Device, pool, 0, 1, (nuint)sizeof(ulong), &compactedSize,
            (ulong)sizeof(ulong), QueryResultFlags.Result64Bit | QueryResultFlags.ResultWaitBit);
        _vk.DestroyQueryPool(_gfx.Device, pool, null);

        return compactedSize;
    }

    /// <summary>Copies <paramref name="src"/> into <paramref name="dst"/> in compacting mode.
    /// <paramref name="dst"/> must already exist at the size <see cref="QueryCompactedSize"/>
    /// reported. Neither AS is destroyed here - the caller decides what happens to the source.</summary>
    public void CopyCompact(AccelerationStructureKHR src, AccelerationStructureKHR dst)
    {
        var copyInfo = new CopyAccelerationStructureInfoKHR
        {
            SType = StructureType.CopyAccelerationStructureInfoKhr,
            Src   = src,
            Dst   = dst,
            Mode  = CopyAccelerationStructureModeKHR.CompactKhr,
        };
        var cmd = _gfx.BeginSingleTimeCommands();
        _khr!.CmdCopyAccelerationStructure(cmd, &copyInfo);
        _gfx.EndSingleTimeCommands(cmd);
    }

    public void Dispose()
    {
        _khr?.Dispose();
        _khr = null;
    }
}
