using System.Numerics;

namespace CadThingo.VulkanEngine;

public enum LightType : uint
{
    Directional = 0,
    Point       = 1,
    Spot        = 2,
}

/// <summary>
/// Per-entity light data. Scene gathers all enabled LightComponents each frame
/// into the per-frame StructuredBuffer&lt;PbrLight&gt; consumed by PbrShader.slang.
/// Color is linear and NOT pre-multiplied by intensity — the shader does
/// color * intensity so artists can tune intensity without re-saturating color.
/// </summary>
public unsafe class LightComponent : Component
{
    public LightType Type         = LightType.Point;
    public Vector3   Color        = new(1f, 1f, 1f);
    public float     Intensity    = 1f;
    public float     Range        = 10f;        // point/spot — ignored by directional
    public Vector3   Direction    = new(0f, -1f, 0f); // directional/spot, world-space
    public float     InnerConeCos = 0.95f;      // spot: cos of inner cone half-angle
    public float     OuterConeCos = 0.85f;      // spot: cos of outer cone half-angle
    public bool      CastShadows  = true;
    public bool      Enabled      = true;
}