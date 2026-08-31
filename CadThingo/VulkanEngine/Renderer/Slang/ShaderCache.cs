using System.Security.Cryptography;
using System.Text;

namespace CadThingo.VulkanEngine.Renderer.Slang;

// Disk cache for compiled shaders (docs/descriptor-system.md section 4): per request, a
// .refl file (reflection + dependency manifest) plus one .spv per entry point, named
// {module}-{requestKey}. The request key covers everything EXCEPT source contents:
// request fields, slang.dll identity (file size + mtime - the dll ships without version
// metadata and must not be loaded on the warm path), and the format version. Source
// changes are caught by the dep manifest instead: the .refl stores (path, content hash)
// for the module and every transitive import, and TryLoad re-hashes them - any edit is a
// miss, which recompiles and rewrites the entry. First compile is always a miss, so the
// manifest bootstraps itself.
internal sealed class ShaderCache
{
    private const uint Magic = 0x4C464552; // "REFL"
    // Also covers the fixed compiler options (optimization level, entry-point naming, matrix
    // layout): they are not part of the request, so a change to them only invalidates cached
    // blobs via this bump.
    private const uint FormatVersion = 5;
    private static readonly TimeSpan GcAge = TimeSpan.FromDays(30);

    private readonly string _dir;
    private readonly string _slangIdentity;

    public ShaderCache(string dir)
    {
        _dir = dir;
        Directory.CreateDirectory(_dir);

        var sdk = Environment.GetEnvironmentVariable("VULKAN_SDK");
        var dll = sdk != null ? new FileInfo(Path.Combine(sdk, "Bin", "slang.dll")) : null;
        _slangIdentity = dll is { Exists: true } ? $"{dll.Length}:{dll.LastWriteTimeUtc.Ticks}" : "no-slang";

        CollectGarbage();
    }

    public string RequestKey(in ShaderCompileRequest r) => Hash16(string.Join("\n",
        r.Module,
        string.Join(";", r.EntryPoints),
        string.Join(";", r.Defines),
        string.Join(";", r.Capabilities),
        _slangIdentity,
        FormatVersion.ToString()));

    public bool TryLoad(in ShaderCompileRequest request, string key, out ShaderCompileResult result)
    {
        result = null!;
        string reflPath = ReflPath(request.Module, key);
        if (!File.Exists(reflPath)) return false;

        try
        {
            using var reader = new BinaryReader(File.OpenRead(reflPath), Encoding.UTF8);
            if (reader.ReadUInt32() != Magic || reader.ReadUInt32() != FormatVersion) return false;

            int depCount = reader.ReadInt32();
            var deps = new string[depCount];
            for (int i = 0; i < depCount; i++)
            {
                deps[i] = reader.ReadString();
                string storedHash = reader.ReadString();
                if (!File.Exists(deps[i]) || Hash16File(deps[i]) != storedHash) return false;
            }

            var bindings = new BindingDesc[reader.ReadInt32()];
            for (int i = 0; i < bindings.Length; i++)
                bindings[i] = new BindingDesc(reader.ReadString(), reader.ReadUInt32(), reader.ReadUInt32(),
                    (Silk.NET.Vulkan.DescriptorType)reader.ReadInt32(), reader.ReadUInt32(),
                    (Silk.NET.Vulkan.ShaderStageFlags)reader.ReadUInt32());

            var pushes = new PushRangeDesc[reader.ReadInt32()];
            for (int i = 0; i < pushes.Length; i++)
                pushes[i] = new PushRangeDesc((Silk.NET.Vulkan.ShaderStageFlags)reader.ReadUInt32(),
                    reader.ReadUInt32(), reader.ReadString());

            var specs = new SpecConstDesc[reader.ReadInt32()];
            for (int i = 0; i < specs.Length; i++)
                specs[i] = new SpecConstDesc(reader.ReadString(), reader.ReadUInt32());

            var entries = new EntryPointDesc[reader.ReadInt32()];
            for (int i = 0; i < entries.Length; i++)
                entries[i] = new EntryPointDesc(reader.ReadString(),
                    (Silk.NET.Vulkan.ShaderStageFlags)reader.ReadUInt32(),
                    reader.ReadUInt32(), reader.ReadUInt32(), reader.ReadUInt32());

            int spirvCount = reader.ReadInt32();
            var spirv = new byte[spirvCount][];
            for (int i = 0; i < spirvCount; i++)
            {
                string spvPath = SpvPath(request.Module, key, i);
                if (!File.Exists(spvPath)) return false;
                spirv[i] = File.ReadAllBytes(spvPath);
            }

            Touch(request.Module, key, spirvCount);
            result = new ShaderCompileResult
            {
                SpirvPerEntryPoint = spirv,
                Reflection = new ShaderReflectionData
                {
                    Bindings = bindings, PushConstants = pushes,
                    SpecConstants = specs, EntryPoints = entries,
                    Dependencies = deps,
                },
                Dependencies = deps,
            };
            return true;
        }
        catch (Exception e) when (e is IOException or EndOfStreamException)
        {
            return false; // corrupt/truncated entry: recompile overwrites it
        }
    }

    public void Store(in ShaderCompileRequest request, string key, ShaderCompileResult result)
    {
        for (int i = 0; i < result.SpirvPerEntryPoint.Length; i++)
            File.WriteAllBytes(SpvPath(request.Module, key, i), result.SpirvPerEntryPoint[i]);

        using var writer = new BinaryWriter(File.Create(ReflPath(request.Module, key)), Encoding.UTF8);
        writer.Write(Magic);
        writer.Write(FormatVersion);

        writer.Write(result.Dependencies.Length);
        foreach (var dep in result.Dependencies)
        {
            writer.Write(dep);
            writer.Write(Hash16File(dep));
        }

        var r = result.Reflection;
        writer.Write(r.Bindings.Length);
        foreach (var b in r.Bindings)
        {
            writer.Write(b.Name); writer.Write(b.Set); writer.Write(b.Binding);
            writer.Write((int)b.Type); writer.Write(b.Count); writer.Write((uint)b.Stages);
        }
        writer.Write(r.PushConstants.Length);
        foreach (var p in r.PushConstants)
        {
            writer.Write((uint)p.Stages); writer.Write(p.Size); writer.Write(p.Name);
        }
        writer.Write(r.SpecConstants.Length);
        foreach (var s in r.SpecConstants)
        {
            writer.Write(s.Name); writer.Write(s.ConstantId);
        }
        writer.Write(r.EntryPoints.Length);
        foreach (var e in r.EntryPoints)
        {
            writer.Write(e.Name); writer.Write((uint)e.Stage);
            writer.Write(e.GroupSizeX); writer.Write(e.GroupSizeY); writer.Write(e.GroupSizeZ);
        }
        writer.Write(result.SpirvPerEntryPoint.Length);
    }

    private string BaseName(string module, string key) => $"{module.Replace('/', '_')}-{key}";
    private string ReflPath(string module, string key) => Path.Combine(_dir, BaseName(module, key) + ".refl");
    private string SpvPath(string module, string key, int entry) => Path.Combine(_dir, $"{BaseName(module, key)}.e{entry}.spv");

    // Cache hits refresh mtime so live entries survive the age-based GC.
    private void Touch(string module, string key, int spirvCount)
    {
        var now = DateTime.UtcNow;
        try
        {
            File.SetLastWriteTimeUtc(ReflPath(module, key), now);
            for (int i = 0; i < spirvCount; i++) File.SetLastWriteTimeUtc(SpvPath(module, key, i), now);
        }
        catch (IOException) { }
    }

    private void CollectGarbage()
    {
        var cutoff = DateTime.UtcNow - GcAge;
        foreach (var file in Directory.EnumerateFiles(_dir))
        {
            try
            {
                if (File.GetLastWriteTimeUtc(file) < cutoff) File.Delete(file);
            }
            catch (IOException) { }
        }
    }

    private static string Hash16(string material) =>
        Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(material)))[..16].ToLowerInvariant();

    private static string Hash16File(string path)
    {
        using var stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream))[..16].ToLowerInvariant();
    }
}