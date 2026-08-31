using System.Text;

namespace CadThingo.VulkanEngine.Renderer.Slang;

// Headless shader check (`dotnet run --project CadThingo -- --shader-audit`). Compiles every
// engine kernel through ShaderLibrary and reports, without bringing up a device.
// Three outputs:
//   1. per-kernel compile + reflection status - this is the fast syntax check. Shaders are no
//      longer compiled by the build, so a broken kernel would otherwise only surface when the
//      pipeline that owns it is created at runtime;
//   2. the matrix-layout guard (see the col-major count below) - silent to the validation layer
//      and no longer catchable by diffing against a build-time .spv;
//   3. a cross-kernel drift table - parameter names declared at DIFFERENT (set, binding) or
//      with different types across kernels, i.e. the inconsistency catalog SceneBindings unifies.
// Nothing throws; a broken kernel or a matrix-layout regression comes back as exit code 1 so CI
// can gate on it.
//
// The manifest is hand-kept: a new kernel needs an entry here (with its entry points,
// capabilities and any -D variants) or the audit will not cover it.
public static class ShaderAudit
{
    private static readonly string[] None = [];
    private static readonly string[] RayQuery = ["spvRayQueryKHR"];
    private static readonly string[] RayTracing = ["spvRayTracingKHR"];
    private static readonly string[] RayTracingSer = ["spvRayTracingKHR", "spvShaderInvocationReorderNV"];
    private static readonly string[] Graphics = ["VSMain", "PSMain"];
    private static readonly string[] Compute = ["main"];
    private static readonly string[] ComputeMain = ["Main"];
    private static readonly string[] RtEntries = ["rayGenMain", "missMain", "closestHitMain", "anyHitMain"];
    private static readonly string[] SerDefine = ["USE_SER=1"];

    private static readonly ShaderCompileRequest[] Manifest =
    [
        // flat shaders (VulkanEngine/Shaders)
        new("ImGui", Graphics, None, None),

        // Deferred
        new("Deferred/CullDraws", ComputeMain, None, None),
        new("Deferred/Geometry", Graphics, None, None),
        new("Deferred/PBR", Graphics, None, RayQuery),

        // Forward
        new("Forward/LightCulling", ComputeMain, None, None),
        new("Forward/Transparent", Graphics, None, RayQuery),

        // IBL
        new("IBL/BrdfLutGen", Compute, None, None),
        new("IBL/EquirectToCube", Compute, None, None),
        new("IBL/IrradianceConvolve", Compute, None, None),
        new("IBL/PrefilterEnv", Compute, None, None),
        new("IBL/ProbeCapture", Graphics, None, None),
        new("IBL/Skybox", Graphics, None, None),

        // PathTracer (+ SER variant, mirroring the ShaderVariant items)
        new("PathTracer/PTCompute", Compute, None, RayQuery),
        new("PathTracer/PathTraceRT", RtEntries, None, RayTracing),
        new("PathTracer/PathTraceRT", RtEntries, SerDefine, RayTracingSer),

        // ReSTIR (+ SER variant)
        new("ReSTIR/BuildTemporal", Compute, None, None),
        new("ReSTIR/ReStirDI", RtEntries, None, RayTracing),
        new("ReSTIR/ReStirDI", RtEntries, SerDefine, RayTracingSer),
        new("ReSTIR/SpatialShade", Compute, None, RayQuery),

        // Selection
        new("Selection/Outline", Graphics, None, None),
        new("Selection/PickCompute", Compute, None, RayQuery),
        new("Selection/SelectionMask", Compute, None, RayQuery),

        // TextureCompression
        new("TextureCompression/BcEncode", Compute, None, None),

        // Tonemapping
        new("Tonemapping/Tonemap", Graphics, None, None),

        // WavefrontPathTracer
        new("WavefrontPathTracer/Connect", Compute, None, RayQuery),
        new("WavefrontPathTracer/Extend", Compute, None, RayQuery),
        new("WavefrontPathTracer/Finalize", Compute, None, RayQuery),
        new("WavefrontPathTracer/Generate", Compute, None, RayQuery),
        new("WavefrontPathTracer/Shade", Compute, None, RayQuery),
        new("WavefrontPathTracer/TailMegakernel", Compute, None, RayQuery),
    ];

    /// <summary>Runs the audit and returns a process exit code: 0 when every kernel compiled with
    /// row-major matrices, 1 otherwise. Drift is reported but does not fail, because the same name
    /// at different bindings across pipelines is legal.</summary>
    public static int Run()
    {
        using var library = ShaderLibrary.CreateDefault();
        var log = new StringBuilder();
        int failed = 0, cached = 0, colMajorKernels = 0;
        var declarations = new Dictionary<string, List<(string Module, BindingDesc B)>>();

        foreach (var request in Manifest)
        {
            string label = request.Module + (request.Defines.Length > 0 ? " [" + string.Join(",", request.Defines) + "]" : "");
            try
            {
                var program = library.GetProgram(request);
                if (program.FromCache) cached++;

                // Matrix-layout guard: sum RowMajor/ColMajor member decorations across every emitted
                // stage. ColMajor > 0 means the compiler regressed to transposed matrices (see
                // slang-matrix-layout-inversion); flag it loudly, it is silent to the validation layer.
                int row = 0, col = 0;
                for (int e = 0; e < request.EntryPoints.Length; e++)
                {
                    var (r, c) = SpirvUtil.MatrixMajorness(program.Spirv(e).Span);
                    row += r; col += c;
                }
                if (col > 0) colMajorKernels++;

                log.AppendLine($"  {(col > 0 ? "COLMAJ" : "OK  ")} {label,-42} entries={request.EntryPoints.Length} " +
                               $"bindings={program.Reflection.Bindings.Length} push={program.Reflection.PushConstants.Length} " +
                               $"spec={program.Reflection.SpecConstants.Length} mtx=R{row}/C{col}{(program.FromCache ? " (cache)" : "")}");
                foreach (var b in program.Reflection.Bindings)
                {
                    if (!declarations.TryGetValue(b.Name, out var list))
                        declarations[b.Name] = list = [];
                    list.Add((label, b));
                }
            }
            catch (Exception e)
            {
                failed++;
                string reason = e.Message.ReplaceLineEndings(" ");
                log.AppendLine($"  FAIL {label,-42} {reason[..Math.Min(reason.Length, 180)]}");
            }
        }

        log.AppendLine();
        log.AppendLine("  drift: parameter names declared inconsistently across kernels");
        int drifting = 0;
        foreach (var (name, uses) in declarations.OrderBy(kv => kv.Key))
        {
            bool consistent = uses.All(u =>
                u.B.Set == uses[0].B.Set && u.B.Binding == uses[0].B.Binding &&
                u.B.Type == uses[0].B.Type && u.B.Count == uses[0].B.Count);
            if (consistent) continue;
            drifting++;
            log.AppendLine($"    {name}:");
            foreach (var (module, b) in uses)
                log.AppendLine($"      ({b.Set},{b.Binding}) {b.Type} count={b.Count,-4} in {module}");
        }
        if (drifting == 0) log.AppendLine("    none");

        Console.WriteLine($"[slang] shader audit: {Manifest.Length - failed}/{Manifest.Length} compiled " +
                          $"({cached} from cache), {failed} failed, {drifting} drifting parameter names, " +
                          $"{colMajorKernels} col-major (matrix-layout regression if >0)");
        Console.Write(log.ToString());

        return failed > 0 || colMajorKernels > 0 ? 1 : 0;
    }
}