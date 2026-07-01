namespace CadThingo.VulkanEngine.Renderer.Features.PathTracer;

/// <summary>
/// The camera / depth-of-field controls shared by every path-tracer pipeline
/// (the megakernel <see cref="PTComputePipeline"/> and the wavefront pipeline).
/// Lets the Renderer Settings panel drive whichever tracer is active through one
/// code path -- the camera-mode combo + lens sliders are identical and functional
/// for both, since both Generate kernels bake the same four CameraMode PSOs and
/// read the same DoF fields out of PathFrameUBO. Per-tracer differences (the
/// sample counter, bounce-cap semantics) stay outside this interface.
/// </summary>
public interface IPathTracerCamera
{
    PTComputePipeline.CameraMode Mode { get; set; }
    float Aperture            { get; set; }
    float FocusDistance       { get; set; }
    float PaniniDistance      { get; set; }
    float VerticalCompression { get; set; }
}