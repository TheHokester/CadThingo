using System.Diagnostics;
using Silk.NET.Core.Native;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.FrameGraph;

/// <summary>Pipeline-statistics for one pass (only the four counters GraphDebug records).</summary>
public struct PipelineStats
{
    public ulong VertexInvocations;
    public ulong Primitives;          // clipping invocations (≈ primitives entering the clipper)
    public ulong FragmentInvocations;
    public ulong ComputeInvocations;
}

/// <summary>Per-pass timing + statistics, surfaced to the debug overlay.</summary>
public struct PassTiming
{
    public string Name;
    public QueueClass Queue;
    public PassType Type;
    public double GpuMs;          // resolved frames-in-flight late; 0 until available
    public double CpuRecordMs;    // CPU cost to record this pass (barriers + body) last frame
    public PipelineStats Stats;   // zeroed unless pipeline-stats collection is on
}

/// <summary>Whole-graph snapshot for the ImGui "Render Graph" panel.</summary>
public struct GraphStats
{
    public double FrameGpuMs;        // Σ per-pass GPU ms (last resolved frame)
    public double CompileMs;         // cost of the last Compile()
    public int    LivePassCount;
    public int    CulledPassCount;
    public int    BarrierCount;      // image+buffer barriers across all passes + closing batch
    public bool   GpuTimingAvailable;
    public bool   PipelineStatsOn;
    public PassTiming[] Passes;      // live, in schedule order
}

/// <summary>
/// Per-pass GPU/CPU instrumentation + debug naming for a compiled <see cref="FrameGraph"/>.
/// Owns a timestamp query pool (2 per pass per frame-in-flight, resolved a frame late to
/// avoid stalls) and, when the device supports it, a pipeline-statistics pool. Wraps each
/// pass in a CmdBeginDebugUtilsLabel region and names the graph's VkImages/VkBuffers so
/// captures (RenderDoc / Nsight) read cleanly. Every capability degrades to a no-op when
/// the device or debug-utils extension is missing.
/// </summary>
internal sealed unsafe class GraphDebug : IDisposable
{
    // Statistics recorded per eligible pass, in ascending bit order (== the order
    // GetQueryPoolResults returns them, so the read indices below line up).
    private static readonly QueryPipelineStatisticFlags StatsMask =
        QueryPipelineStatisticFlags.VertexShaderInvocationsBit
      | QueryPipelineStatisticFlags.ClippingInvocationsBit
      | QueryPipelineStatisticFlags.FragmentShaderInvocationsBit
      | QueryPipelineStatisticFlags.ComputeShaderInvocationsBit;
    private const int StatsValues = 4;

    /// <summary>Runtime toggle for pipeline-statistics collection — the pool still exists,
    /// this only gates the per-frame Begin/EndQuery. Default on when supported; flip off if
    /// a driver rejects a stats query spanning a dynamic-rendering region.</summary>
    public bool CollectPipelineStats = true;

    private readonly GraphicsDevice _gfx;
    private readonly Vk _vk;
    private readonly Device _device;

    private int  _passCount;
    private uint _frames;

    private bool   _ts;          // timestamp pool live
    private bool   _ps;          // pipeline-stats pool live
    private double _periodNs;
    private ulong  _validMask;

    private QueryPool _tsPool;
    private QueryPool _psPool;

    private string[]     _names    = [];
    private QueueClass[] _queues   = [];
    private PassType[]   _types    = [];
    private bool[]       _eligible = [];   // stats only meaningful for draw/dispatch passes

    private double[]     _cpuMs   = [];
    private long         _cpuStart;
    private PassTiming[] _timings = [];
    private double       _frameGpuMs;

    // Per-frame-slot guard: a slot's queries are only read once it has been recorded at
    // least once. Replaces a one-time full-pool reset (which cost a QueueWaitIdle drain on
    // every framegraph rebuild) — the per-frame CmdResetQueryPool below already defines the
    // queries before they're written.
    private bool[] _frameWritten = [];

    public GraphDebug(GraphicsDevice gfx)
    {
        _gfx    = gfx;
        _vk     = gfx.Vk;
        _device = gfx.Device;
    }

    public void Initialize(string[] names, QueueClass[] queues, PassType[] types, bool[] eligible, uint frames)
    {
        _passCount = names.Length;
        _frames    = frames;
        _names     = names;
        _queues    = queues;
        _types     = types;
        _eligible  = eligible;
        _cpuMs        = new double[_passCount];
        _timings      = new PassTiming[_passCount];
        _frameWritten = new bool[frames];
        if (_passCount == 0) return;

        // Timestamp pool: begin+end per pass per frame-in-flight.
        if (_gfx.TimestampsSupported)
        {
            uint validBits = _gfx.TimestampValidBits;
            _validMask = validBits >= 64 ? ulong.MaxValue : (1UL << (int)validBits) - 1;
            _periodNs  = _gfx.TimestampPeriod;

            var info = new QueryPoolCreateInfo
            {
                SType = StructureType.QueryPoolCreateInfo,
                QueryType = QueryType.Timestamp,
                QueryCount = (uint)(2 * _passCount) * _frames,
            };
            if (_vk.CreateQueryPool(_device, in info, null, out _tsPool) == Result.Success)
            {
                _ts = true;
                NameObject(ObjectType.QueryPool, _tsPool.Handle, "FrameGraph/Timestamps");
            }
        }

        // Pipeline-statistics pool: one query per pass per frame.
        if (_gfx.PipelineStatisticsSupported)
        {
            var info = new QueryPoolCreateInfo
            {
                SType = StructureType.QueryPoolCreateInfo,
                QueryType = QueryType.PipelineStatistics,
                QueryCount = (uint)_passCount * _frames,
                PipelineStatistics = StatsMask,
            };
            if (_vk.CreateQueryPool(_device, in info, null, out _psPool) == Result.Success)
            {
                _ps = true;
                NameObject(ObjectType.QueryPool, _psPool.Handle, "FrameGraph/PipelineStats");
            }
        }

        // No up-front pool reset: the per-frame CmdResetQueryPool in BeginFrame defines a
        // slot's queries before they're written, and _frameWritten gates the read until
        // then. This avoids a full GPU drain on every framegraph rebuild (every resize).
    }

    // ---- per-frame replay hooks ----------------------------------------------
    public void BeginFrame(CommandBuffer cmd, uint frame)
    {
        if (_passCount == 0) return;
        // Skip the resolve until this slot has been recorded once — otherwise we'd read
        // queries that were never reset/written (no up-front pool reset anymore).
        if (_frameWritten[frame]) ResolvePrevious(frame);
        if (_ts) _vk.CmdResetQueryPool(cmd, _tsPool, frame * (uint)(2 * _passCount), (uint)(2 * _passCount));
        if (_ps) _vk.CmdResetQueryPool(cmd, _psPool, frame * (uint)_passCount,        (uint)_passCount);
        _frameWritten[frame] = true;
    }

    public void BeginPass(CommandBuffer cmd, uint frame, int pass)
    {
        BeginLabel(cmd, _names[pass], LabelColor(_types[pass]));
        if (_ts) _vk.CmdWriteTimestamp2(cmd, PipelineStageFlags2.TopOfPipeBit, _tsPool, TsIndex(frame, pass, 0));
        if (_ps && CollectPipelineStats && _eligible[pass])
            _vk.CmdBeginQuery(cmd, _psPool, PsIndex(frame, pass), 0);
        _cpuStart = Stopwatch.GetTimestamp();
    }

    public void EndPass(CommandBuffer cmd, uint frame, int pass)
    {
        _cpuMs[pass] = (Stopwatch.GetTimestamp() - _cpuStart) * 1000.0 / Stopwatch.Frequency;
        if (_ps && CollectPipelineStats && _eligible[pass])
            _vk.CmdEndQuery(cmd, _psPool, PsIndex(frame, pass));
        if (_ts) _vk.CmdWriteTimestamp2(cmd, PipelineStageFlags2.BottomOfPipeBit, _tsPool, TsIndex(frame, pass, 1));
        EndLabel(cmd);
    }

    private uint TsIndex(uint frame, int pass, int half) => (frame * (uint)_passCount + (uint)pass) * 2 + (uint)half;
    private uint PsIndex(uint frame, int pass)           =>  frame * (uint)_passCount + (uint)pass;

    private void ResolvePrevious(uint frame)
    {
        if (_ts)
        {
            int n = 2 * _passCount;
            ulong* data = stackalloc ulong[n];
            var res = _vk.GetQueryPoolResults(_device, _tsPool, frame * (uint)n, (uint)n,
                (nuint)(n * sizeof(ulong)), data, (ulong)sizeof(ulong), QueryResultFlags.Result64Bit);
            if (res == Result.Success)
            {
                for (int p = 0; p < _passCount; p++)
                {
                    ulong b = data[p * 2 + 0] & _validMask;
                    ulong e = data[p * 2 + 1] & _validMask;
                    _timings[p].GpuMs = e >= b ? (e - b) * _periodNs / 1_000_000.0 : 0;
                }
                // Wall-clock graph time = last-pass end − first-pass begin. (Summing the
                // per-pass deltas overcounts: on one queue the passes pipeline and overlap.)
                ulong first = data[0] & _validMask;
                ulong lastE = data[2 * _passCount - 1] & _validMask;
                _frameGpuMs = lastE >= first ? (lastE - first) * _periodNs / 1_000_000.0 : 0;
            }
        }

        if (_ps && CollectPipelineStats)
        {
            int n = _passCount * StatsValues;
            ulong* data = stackalloc ulong[n];
            var res = _vk.GetQueryPoolResults(_device, _psPool, frame * (uint)_passCount, (uint)_passCount,
                (nuint)(n * sizeof(ulong)), data, (ulong)(StatsValues * sizeof(ulong)), QueryResultFlags.Result64Bit);
            if (res == Result.Success)
                for (int p = 0; p < _passCount; p++)
                    _timings[p].Stats = new PipelineStats
                    {
                        VertexInvocations   = data[p * StatsValues + 0],
                        Primitives          = data[p * StatsValues + 1],
                        FragmentInvocations = data[p * StatsValues + 2],
                        ComputeInvocations  = data[p * StatsValues + 3],
                    };
        }
    }

    public GraphStats Snapshot(double compileMs, int culled, int barrierCount)
    {
        var passes = new PassTiming[_passCount];
        for (int p = 0; p < _passCount; p++)
            passes[p] = new PassTiming
            {
                Name = _names[p], Queue = _queues[p], Type = _types[p],
                GpuMs = _timings[p].GpuMs, CpuRecordMs = _cpuMs[p], Stats = _timings[p].Stats,
            };
        return new GraphStats
        {
            FrameGpuMs = _frameGpuMs,
            CompileMs = compileMs,
            LivePassCount = _passCount,
            CulledPassCount = culled,
            BarrierCount = barrierCount,
            GpuTimingAvailable = _ts,
            PipelineStatsOn = _ps && CollectPipelineStats,
            Passes = passes,
        };
    }

    // ---- debug-utils naming + labels -----------------------------------------
    public void NameObject(ObjectType type, ulong handle, string name)
    {
        if (_gfx.DebugUtils is not { } du || handle == 0) return;
        nint p = SilkMarshal.StringToPtr(name);
        var info = new DebugUtilsObjectNameInfoEXT
        {
            SType = StructureType.DebugUtilsObjectNameInfoExt,
            ObjectType = type,
            ObjectHandle = handle,
            PObjectName = (byte*)p,
        };
        du.SetDebugUtilsObjectName(_device, &info);
        SilkMarshal.Free(p);
    }

    private void BeginLabel(CommandBuffer cmd, string name, (float r, float g, float b) c)
    {
        if (_gfx.DebugUtils is not { } du) return;
        nint p = SilkMarshal.StringToPtr(name);
        var label = new DebugUtilsLabelEXT { SType = StructureType.DebugUtilsLabelExt, PLabelName = (byte*)p };
        label.Color[0] = c.r; label.Color[1] = c.g; label.Color[2] = c.b; label.Color[3] = 1f;
        du.CmdBeginDebugUtilsLabel(cmd, &label);
        SilkMarshal.Free(p);
    }

    private void EndLabel(CommandBuffer cmd) => _gfx.DebugUtils?.CmdEndDebugUtilsLabel(cmd);

    private static (float, float, float) LabelColor(PassType t) => t switch
    {
        PassType.Graphics => (0.40f, 0.70f, 0.40f),
        PassType.Compute  => (0.40f, 0.55f, 0.85f),
        PassType.Transfer => (0.85f, 0.65f, 0.35f),
        PassType.RayTrace => (0.70f, 0.45f, 0.80f),
        _ => (0.6f, 0.6f, 0.6f),
    };

    public void Dispose()
    {
        if (_ts) _vk.DestroyQueryPool(_device, _tsPool, null);
        if (_ps) _vk.DestroyQueryPool(_device, _psPool, null);
        _ts = _ps = false;
    }
}