using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Shaders;

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

        // step 8: release unmanaged resources
        linked.Release();
        composite.Release();
        entryPoint.Release();
        module.Release();
        session.Release();
        global.Release();
    }

    private static void Check(bool condition, string message)
    {
        if (!condition) throw new Exception($"[slang] smoke test failed: {message}");
    }
}