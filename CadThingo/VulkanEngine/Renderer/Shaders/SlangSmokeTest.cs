namespace CadThingo.VulkanEngine.Renderer.Shaders;

// Startup smoke test for the interop : compiles a tiny
// inline shader and sanity-checks the output before anything trusts the vtable math. Call once
// from engine init behind a debug flag while the interop is being built; it becomes the
// version-pin gate for ShaderLibrary later.
public static unsafe class SlangSmokeTest
{
    private const string Source = """
        [shader("compute")]
        [numthreads(8, 8, 1)]
        void main(uint3 tid: SV_DispatchThreadID, RWStructuredBuffer<float> outBuf)
        {
            outBuf[tid.x] = float(tid.x) * 2.0;
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
        
        //step 7: release unmanaged resources
        global.Release();
    }
}