using System.Text;

namespace CadThingo.VulkanEngine.Renderer.Shaders;

// Phase A assert-mode audit (docs/descriptor-system.md section 12): compiles every engine
// kernel through ShaderLibrary and reports, without changing what the renderer runs.
// Two outputs:
//   1. per-kernel compile + reflection status - a failure means the runtime compiler cannot
//      yet replace Shaders.targets for that kernel (search path, capability, or interop gap);
//   2. a cross-kernel drift table - parameter names declared at DIFFERENT (set, binding) or
//      with different types across kernels, i.e. the inconsistency catalog that SceneBindings
//      (Phase B) has to unify.
// Log-only by design: nothing throws, the renderer keeps running on build-time .spv.
//
// The manifest mirrors Shaders.targets (entries, capabilities, SER variants) and, like it,
// must be updated when kernels are added - until Phase B/C retire both.
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
        // legacy flat shaders (VulkanEngine/Shaders)
        new("ImGui", Graphics, None, None),
        new("TexturedMesh", ["VSMain", "FSMain"], None, None),

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

    public static void Run()
    {
        using var library = ShaderLibrary.CreateDefault();
        var log = new StringBuilder();
        int failed = 0, cached = 0;
        var declarations = new Dictionary<string, List<(string Module, BindingDesc B)>>();

        foreach (var request in Manifest)
        {
            string label = request.Module + (request.Defines.Length > 0 ? " [" + string.Join(",", request.Defines) + "]" : "");
            try
            {
                var program = library.GetProgram(request);
                if (program.FromCache) cached++;
                log.AppendLine($"  OK   {label,-42} entries={request.EntryPoints.Length} " +
                               $"bindings={program.Reflection.Bindings.Length} push={program.Reflection.PushConstants.Length} " +
                               $"spec={program.Reflection.SpecConstants.Length}{(program.FromCache ? " (cache)" : "")}");
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
                          $"({cached} from cache), {failed} failed, {drifting} drifting parameter names");
        Console.Write(log.ToString());
    }
}