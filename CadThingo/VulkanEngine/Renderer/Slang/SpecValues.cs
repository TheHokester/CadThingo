using System.Runtime.InteropServices;

namespace CadThingo.VulkanEngine.Renderer.Slang;

/// <summary>Specialization-constant values keyed by the Slang constant's NAME. The constant ids
/// come from reflection, so a pipeline never transcribes a constant_id; renaming or renumbering in
/// the shader surfaces as a startup error instead of a silently mis-specialized pipeline. All
/// Vulkan spec constants are 4 bytes, so every value is stored as raw bits.</summary>
public sealed class SpecValues
{
    private readonly Dictionary<string, uint> _bits = new();

    public SpecValues Set(string name, uint v)  { _bits[name] = v; return this; }
    public SpecValues Set(string name, int v)   => Set(name, unchecked((uint)v));
    public SpecValues Set(string name, bool v)  => Set(name, v ? 1u : 0u);
    public SpecValues Set(string name, float v) => Set(name, BitConverter.SingleToUInt32Bits(v));

    public IReadOnlyDictionary<string, uint> Bits => _bits;
}

public static class SpirvUtil
{
    private const uint Magic = 0x07230203;
    private const uint OpDecorate = 71;
    private const uint OpMemberDecorate = 72;
    private const uint DecorationSpecId = 1;
    private const uint DecorationRowMajor = 4;
    private const uint DecorationColMajor = 5;

    /// <summary>The specialization-constant ids a SPIR-V module actually declares. Slang emits one
    /// module per entry point and strips constants the entry point never reads, so a program-wide
    /// reflected constant may be absent from a given stage; attaching a VkSpecializationInfo entry
    /// for an id the stage does not declare is invalid. Scanning the module is authoritative where
    /// program-level reflection is not.</summary>
    public static HashSet<uint> SpecConstantIds(ReadOnlySpan<byte> spirv)
    {
        var ids = new HashSet<uint>();
        var words = MemoryMarshal.Cast<byte, uint>(spirv);
        if (words.Length < 5 || words[0] != Magic) return ids;

        int i = 5; // skip the fixed-size header
        while (i < words.Length)
        {
            uint count = words[i] >> 16;
            uint op = words[i] & 0xFFFF;
            if (count == 0) break; // malformed; refuse to spin
            if (op == OpDecorate && count >= 4 && words[i + 2] == DecorationSpecId)
                ids.Add(words[i + 3]);
            i += (int)count;
        }
        return ids;
    }

    /// <summary>Counts the matrix-member majorness decorations in a SPIR-V module. Matrices in
    /// interface blocks (UBO/SSBO/push) carry an OpMemberDecorate RowMajor or ColMajor per member.
    /// This engine feeds the runtime compiler a COLUMN_MAJOR source request, which - counterintuitively
    /// - emits RowMajor-decorated members that match every build-time .spv and the System.Numerics
    /// mirrors. So the invariant is ColMajor == 0 across every emitted module; a single ColMajor is the
    /// exact signature of the matrix-layout regression (transposed matrices: RT camera pinned at the
    /// origin, matrix-using raster passes white). See slang-matrix-layout-inversion.</summary>
    public static (int RowMajor, int ColMajor) MatrixMajorness(ReadOnlySpan<byte> spirv)
    {
        int row = 0, col = 0;
        var words = MemoryMarshal.Cast<byte, uint>(spirv);
        if (words.Length < 5 || words[0] != Magic) return (0, 0);

        int i = 5; // skip the fixed-size header
        while (i < words.Length)
        {
            uint count = words[i] >> 16;
            uint op = words[i] & 0xFFFF;
            if (count == 0) break; // malformed; refuse to spin
            // OpMemberDecorate: [struct type, member index, decoration, ...] - decoration at word +3.
            if (op == OpMemberDecorate && count >= 4)
            {
                uint dec = words[i + 3];
                if (dec == DecorationRowMajor) row++;
                else if (dec == DecorationColMajor) col++;
            }
            i += (int)count;
        }
        return (row, col);
    }
}