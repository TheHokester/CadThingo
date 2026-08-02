namespace CadThingo.VulkanEngine.Renderer.Features.PathTracer;

/// <summary>
/// Everything the tracers share: the progressive IO pair, the restart flag that rides with it, and
/// the acceleration-structure scalars their NEE loop needs as uniforms. Every PT core takes this as
/// <c>INeedsFeature&lt;IPathTracingProvider&gt;</c> and reads through it at build / record time.
///
/// The two images appear here because the cores import them into their graphs by handle. Their
/// descriptor side travels through the registry as "accumulator" / "outColor", so nothing binds
/// them through this contract.
/// </summary>
public interface IPathTracingProvider
{
    /// <summary>R32G32B32A32 progressive accumulator, shared by every tracer. Replaced on resize.</summary>
    ImageResource Accumulator { get; }

    /// <summary>R32G32B32A32 resolved output the tonemap modules take as their HDR input.</summary>
    ImageResource OutColor { get; }

    /// <summary>Set whenever the image the tracers are integrating towards changes: a camera move,
    /// a scene edit, fresh images after a resize. The active core consumes and clears it once per
    /// frame, which restarts the sample count.</summary>
    bool AccumulatorDirty { get; }

    void MarkAccumulatorDirty();
    void ClearAccumulatorDirty();

    /// <summary>True once a TLAS exists and is traceable this frame. False on a device with no ray
    /// infrastructure, where the tracers still run and hit nothing.</summary>
    bool RayInfraReady { get; }

    /// <summary>Emissive triangles packed for next-event estimation. Zero makes the tracers skip
    /// their NEE loop.</summary>
    uint EmissiveTriangleCount { get; }

    /// <summary>Summed emissive power over those triangles, for the light-sampling pdf.</summary>
    float TotalEmissivePower { get; }
}
