using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Slang;

// Startup smoke test for the interop : compiles a tiny
// inline shader and sanity-checks the output before anything trusts the vtable math. Call once
// from engine init behind a debug flag while the interop is being built; it becomes the
// version-pin gate for ShaderLibrary later.
public static unsafe class SlangSmokeTest
{
    // Resources are module-scope globals (how all engine shaders declare them - the walker
    // reads global params, not entry-point params). Coverage: implicit binding (outBuf ->
    // set 0 binding 0), explicit [[vk::binding]] round-trip (inBuf), unbounded array
    // (textures, count 0), and a push constant (size path through the element type layout).
    private const string Source = """
        RWStructuredBuffer<float> outBuf;

        [[vk::binding(1, 2)]]
        StructuredBuffer<float> inBuf;

        [[vk::binding(0, 3)]]
        Texture2D textures[];

        struct SmokeParams { float scale; }
        [[vk::push_constant]] ConstantBuffer<SmokeParams> params;

        [shader("compute")]
        [numthreads(8, 8, 1)]
        void main(uint3 tid: SV_DispatchThreadID)
        {
            outBuf[tid.x] = inBuf[tid.x] * params.scale + textures[0].Load(int3(0, 0, 0)).x;
        }
        """;

    public static void Run()
    {
        // Step 0 (works now): prove the dll loads and report the version.
        Console.WriteLine($"[slang] build tag: {SlangNative.BuildTag()}");

        // Step 1 (works now): prove vtable dispatch - must print the same tag as step 0.
        var global = SlangGlobalSession.Create();
        Console.WriteLine($"[slang] vtable check: {global.BuildTag()}");

        // step 2: global.FindProfile("spirv_1_6") -> nonzero uint.
        uint profile = global.FindProfile("spirv_1_6");
        
        // step 3: build a SessionDesc (TargetDesc.Create() with Format = Spirv and the
        //   profile from step 2; no search paths needed) and global.CreateSession(desc).
        var target = TargetDesc.Create();
        target.Format = SlangCompileTarget.Spirv;
        target.Profile = profile;
        var desc = SessionDesc.Create();
        desc.Targets = &target;
        desc.TargetCount = 1;
        var session = global.CreateSession(desc);
        
        //step 4: session.LoadModuleFromSourceString("smoke", "smoke.slang", Source, out var diag)
        //   - on null module, throw with diag.AsString().
        var module = session.LoadModuleFromSourceString("smoke", "smoke.slang", Source);
        
        // step 5: module.FindEntryPointByName("main"), CreateCompositeComponentType(
        //   [module, entryPoint]), Link, GetEntryPointCode(0, 0).
        var entryPoint = module.FindEntryPointByName("main");
        var composite = session.CreateComposite([module.Handle, entryPoint.Handle]);
        var linked = composite.Link();
        byte[] spirv = linked.GetEntryPointCode(0);
        
        // step 6: assert the blob starts with the SPIR-V magic 0x07230203 and is
        //   nonempty; print byte count. Release everything created (blobs, components,
        //   entry point, module, session).
        Console.WriteLine($"[slang] SPIR-V byte count: {spirv.Length}");
        if (BitConverter.ToUInt32(spirv, 0) == 0x07230203)
        {
            Console.WriteLine("[slang] SPIR-V test, length OK");
        }

        // step 7: walk reflection off the linked program (must happen BEFORE linked is
        // released - the layout is owned by the component) and assert what the shader declares.
        var refl = SlangReflectionWalker.Walk(linked.GetLayout());
        foreach (var b in refl.Bindings)
            Console.WriteLine($"[slang] binding '{b.Name}' set={b.Set} binding={b.Binding} type={b.Type} count={b.Count}");
        foreach (var p in refl.PushConstants)
            Console.WriteLine($"[slang] push '{p.Name}' size={p.Size}");
        foreach (var sc in refl.SpecConstants)
            Console.WriteLine($"[slang] spec '{sc.Name}' id={sc.ConstantId}");
        foreach (var ep in refl.EntryPoints)
            Console.WriteLine($"[slang] entry '{ep.Name}' stage={ep.Stage} group={ep.GroupSizeX}x{ep.GroupSizeY}x{ep.GroupSizeZ}");

        Check(refl.Bindings.Length == 3, "expected 3 bindings");
        var outBuf = refl.Bindings.Single(b => b.Name == "outBuf");
        Check(outBuf is { Set: 0, Binding: 0, Type: DescriptorType.StorageBuffer, Count: 1 },
            $"outBuf reflected wrong: {outBuf}");
        var inBuf = refl.Bindings.Single(b => b.Name == "inBuf");
        Check(inBuf is { Set: 2, Binding: 1, Type: DescriptorType.StorageBuffer, Count: 1 },
            $"inBuf reflected wrong: {inBuf}");
        var textures = refl.Bindings.Single(b => b.Name == "textures");
        Check(textures is { Set: 3, Binding: 0, Type: DescriptorType.SampledImage, Count: 0 },
            $"textures reflected wrong (Count 0 = unbounded): {textures}");
        var push = refl.PushConstants.Single();
        Check(push is { Name: "params", Size: 4 }, $"push constant reflected wrong: {push}");
        var main = refl.EntryPoints.Single(e => e.Name == "main");
        Check(main is { Stage: ShaderStageFlags.ComputeBit, GroupSizeX: 8, GroupSizeY: 8, GroupSizeZ: 1 },
            $"entry point reflected wrong: {main}");
        Console.WriteLine("[slang] reflection test OK");

        // step 8: release unmanaged resources - out-param objects only. The module is
        // session-owned (borrowed); releasing it underflows the refcount and corrupts the
        // heap in ways only visible OUTSIDE the debugger.
        linked.Release();
        composite.Release();
        entryPoint.Release();
        session.Release();
        global.Release();

        // step 9: ShaderLibrary + disk cache round-trip, self-contained in a temp dir:
        // library A compiles cold and stores; library B must hit the disk cache without
        // compiling (FromCache) and hand back byte-identical SPIR-V.
        string tempRoot = Path.Combine(Path.GetTempPath(), "CadThingoSlangSmoke");
        Directory.CreateDirectory(tempRoot);
        File.WriteAllText(Path.Combine(tempRoot, "Smoke.slang"), Source);
        string cacheDir = Path.Combine(tempRoot, "ShaderCache");
        if (Directory.Exists(cacheDir)) Directory.Delete(cacheDir, recursive: true);

        var request = new ShaderCompileRequest("Smoke", ["main"], [], []);
        using (var libA = new ShaderLibrary(tempRoot, tempRoot, cacheDir))
        {
            var cold = libA.GetProgram(request);
            Check(!cold.FromCache, "first GetProgram should compile, not hit the cache");
            Check(cold.Reflection.Bindings.Length == 3, "library compile reflected wrong binding count");
            Check(BitConverter.ToUInt32(cold.Spirv(0).Span) == 0x07230203, "library compile produced bad SPIR-V");
            Check(ReferenceEquals(cold, libA.GetProgram(request)), "same library should memoize the program");

            using var libB = new ShaderLibrary(tempRoot, tempRoot, cacheDir);
            var warm = libB.GetProgram(request);
            Check(warm.FromCache, "second library should load from the disk cache");
            Check(warm.Spirv(0).Span.SequenceEqual(cold.Spirv(0).Span), "cached SPIR-V differs from compiled");
            Check(warm.Reflection.PushConstants.Single() == cold.Reflection.PushConstants.Single(),
                "cached reflection differs from compiled");

            // touching a dependency must invalidate: append a comment, expect a recompile
            File.AppendAllText(Path.Combine(tempRoot, "Smoke.slang"), "\n// cache-buster\n");
            using var libC = new ShaderLibrary(tempRoot, tempRoot, cacheDir);
            Check(!libC.GetProgram(request).FromCache, "edited source should miss the cache");
        }
        Console.WriteLine("[slang] shader library + disk cache OK");

        // step 10: SceneBindings foundations (Phase B). Sandbox library: the real
        // SceneBindings module copied to the temp lib dir; its imports (CommonTypes,
        // PTUtils) resolve via the real shader dir as an extra include path.
        string engineShaders = Path.Combine(AppContext.BaseDirectory, "VulkanEngine", "Shaders");
        if (!Directory.Exists(engineShaders))
            engineShaders = @"C:\Users\jamie\RiderProjects\CadThingo\CadThingo\VulkanEngine\Shaders";
        File.Copy(Path.Combine(engineShaders, "SceneBindings.slang"),
                  Path.Combine(tempRoot, "SceneBindings.slang"), overwrite: true);
        File.WriteAllText(Path.Combine(tempRoot, "DeadStripProbe.slang"), """
            import SceneBindings;

            [[vk::binding(0, 1)]]
            RWStructuredBuffer<uint> probeOut;

            [shader("compute")]
            [numthreads(64, 1, 1)]
            void main(uint3 tid : SV_DispatchThreadID)
            {
                probeOut[tid.x] = sceneVertices.Load(tid.x * 4) + sceneIndices.Load(tid.x * 4);
            }
            """);
        using var libScene = new ShaderLibrary(tempRoot, tempRoot, cacheDir, [engineShaders]);

        // 10a: module-only reflection - the canonical scene layout source.
        var scene = SortByBinding(libScene.ReflectModule("SceneBindings").Bindings);
        Check(scene.Length == 11, $"SceneBindings expected 11 bindings, got {scene.Length}");
        Check(scene.All(b => b.Set == 0 && b.Binding != 0), "SceneBindings must sit in set 0 with binding 0 reserved");
        Check(scene[^1] is { Name: "sceneTextures", Binding: 11, Type: DescriptorType.SampledImage, Count: 0 },
            $"sceneTextures reflected wrong: {scene[^1]}");
        Check(scene.Single(b => b.Name == "sceneSamplers") is { Binding: 10, Count: 16 }, "sceneSamplers reflected wrong");
        Check(scene.Single(b => b.Name == "sceneTlas").Type == DescriptorType.AccelerationStructureKhr, "sceneTlas reflected wrong");

        // 10b: THE dead-strip probe (design-doc risk): a shader importing SceneBindings -
        // which declares a TLAS - compiled WITHOUT any ray capability must still compile.
        var probe = libScene.GetProgram(new ShaderCompileRequest("DeadStripProbe", ["main"], [], []));
        Check(BitConverter.ToUInt32(probe.Spirv(0).Span) == 0x07230203, "dead-strip probe produced bad SPIR-V");
        Check(probe.Reflection.Bindings.Any(b => b is { Name: "probeOut", Set: 1, Binding: 0 }),
            "probe's own pass-local binding missing from reflection");
        Console.WriteLine("[slang] SceneBindings reflection + dead-strip probe OK");
    }

    private static BindingDesc[] SortByBinding(BindingDesc[] bindings)
    {
        var sorted = (BindingDesc[])bindings.Clone();
        Array.Sort(sorted, (a, b) => a.Binding.CompareTo(b.Binding));
        return sorted;
    }

    private static void Check(bool condition, string message)
    {
        if (!condition) throw new Exception($"[slang] smoke test failed: {message}");
    }
}