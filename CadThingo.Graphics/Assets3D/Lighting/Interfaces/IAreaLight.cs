using System.Numerics;

namespace CadThingo.Graphics.Assets3D.Lighting.Interfaces;

/// <summary>
/// Defines geometric properties for area lights.
/// </summary>
public interface IAreaLight
{
    /// <summary>
    /// Gets or sets the rectangular size of the light.
    /// </summary>
    Vector2? AreaSize { get; set; }

    /// <summary>
    /// Gets or sets the radius for spherical or disc lights.
    /// </summary>
    float? Radius { get; set; }

    /// <summary>
    /// Gets or sets the length for tube lights.
    /// </summary>
    float? Length { get; set; }

    /// <summary>
    /// Gets or sets whether the light emits from both sides.
    /// </summary>
    bool TwoSided { get; set; }
}