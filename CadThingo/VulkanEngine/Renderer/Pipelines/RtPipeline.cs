using Silk.NET.Vulkan;
using Silk.NET.Vulkan.Extensions.KHR;

namespace CadThingo.VulkanEngine.Renderer.Pipelines;
// Base for the (additive, opt-in) ray-tracing-pipeline path. Owns the
// VK_KHR_ray_tracing_pipeline dispatch table and the SBT-layout properties every
// RT pipeline needs to pack its shader binding table — deliberately kept off the
// Renderer so the RT path stays self-contained and the compute path tracer is
// untouched. The concrete RTPipeline (the path tracer itself: shader groups, SBT,
// CmdTraceRays) inherits from this in a later phase. Construction is gated by the
// Renderer on RayTracePipelineSupported.
public abstract unsafe class RtPipeline : PipelineBase
{
    public override PipelineBindPoint BindPoint => PipelineBindPoint.RayTracingKhr;

    // Build-time .spv for the legacy route; null on the reflected route (see PipelineBase.Program).
    protected virtual string? ShaderPath => null;

    // VK_KHR_ray_tracing_pipeline dispatch table + SBT-layout properties, loaded
    // once per instance from the device. Null / zero if the extension failed to
    // load (defensive — the Renderer gate should prevent that).
    protected KhrRayTracingPipeline? KhrRtPipeline;
    protected uint ShaderGroupHandleSize;        // bytes per shader-group handle
    protected uint ShaderGroupBaseAlignment;     // raygen/miss/hit SBT region base alignment
    protected uint ShaderGroupHandleAlignment;   // per-handle stride alignment within a region
    protected uint MaxRayRecursionDepth;         // device cap; the path tracer targets depth 1

    protected RtPipeline(in GpuContext gpu, Renderer renderer) : base(gpu, renderer)
    {
        LoadDispatchAndProperties();
    }

    // Loads the dispatch table and queries PhysicalDeviceRayTracingPipelinePropertiesKHR
    // for the SBT alignment/handle sizes. Safe to call even if unsupported — it
    // just leaves KhrRtPipeline null and the properties zero.
    private void LoadDispatchAndProperties()
    {
        if (!Vk.TryGetDeviceExtension(Gfx.GetVkInstance(), Device, out KhrRtPipeline))
        {
            Console.Error.WriteLine("[RtPipeline] KhrRayTracingPipeline dispatch table failed to load");
            KhrRtPipeline = null;
            return;
        }

        var rtProps = new PhysicalDeviceRayTracingPipelinePropertiesKHR
        {
            SType = StructureType.PhysicalDeviceRayTracingPipelinePropertiesKhr,
        };
        var props2 = new PhysicalDeviceProperties2
        {
            SType = StructureType.PhysicalDeviceProperties2,
            PNext = &rtProps,
        };
        Vk.GetPhysicalDeviceProperties2(Gfx.PhysicalDevice, &props2);

        ShaderGroupHandleSize      = rtProps.ShaderGroupHandleSize;
        ShaderGroupBaseAlignment   = rtProps.ShaderGroupBaseAlignment;
        ShaderGroupHandleAlignment = rtProps.ShaderGroupHandleAlignment;
        MaxRayRecursionDepth       = rtProps.MaxRayRecursionDepth;
    }
}