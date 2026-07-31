using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer.Slang;

// Engine-facing reflection summary (docs/descriptor-system.md section 2.2): everything the
// descriptor registry and pipeline layout construction need, fully managed so it outlives
// the Slang objects it was walked from and can round-trip through the .refl cache later.
public sealed class ShaderReflectionData
{
    public required BindingDesc[]    Bindings { get; init; }
    public required PushRangeDesc[]  PushConstants { get; init; }
    public required SpecConstDesc[]  SpecConstants { get; init; }
    public required EntryPointDesc[] EntryPoints { get; init; }
    public string[] Dependencies { get; init; } = []; // set by the compiler service, not the walker
}

public readonly record struct BindingDesc(
    string Name,
    uint Set,
    uint Binding,
    DescriptorType Type,
    uint Count, // 0 = unbounded (variable-count descriptor array)
    ShaderStageFlags Stages);

public readonly record struct PushRangeDesc(ShaderStageFlags Stages, uint Size, string Name);

public readonly record struct SpecConstDesc(string Name, uint ConstantId);

public readonly record struct EntryPointDesc(
    string Name,
    ShaderStageFlags Stage,
    uint GroupSizeX,
    uint GroupSizeY,
    uint GroupSizeZ);