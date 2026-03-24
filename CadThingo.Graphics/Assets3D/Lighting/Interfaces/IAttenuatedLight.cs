using System.Numerics;

namespace CadThingo.Graphics.Assets3D.Lighting.Interfaces;
/// <summary>
/// Enables attenuation of light based on distance.
/// </summary>
public interface IAttenuatedLight
{
    /// <summary>
    /// Gets or sets the maximum effective range of the light.
    /// Beyond this distance, the light has no influence.
    /// </summary>
    float Range { get; set; }

    /// <summary>
    /// Gets or sets the falloff model used to attenuate light over distance.
    /// </summary>
    LightFalloff Falloff { get; set; }

    /// <summary>
    /// Gets or sets the attenuation coefficients (constant, linear, quadratic).
    /// Used in the formula: 1 / (c + l * d + q * d²).
    /// </summary>
    Vector3 Attenuation { get; set; }

    /// <summary>
    /// Gets or sets the minimum distance clamp for attenuation calculations.
    /// Prevents excessively high intensity near the light source.
    /// </summary>
    float MinDistance { get; set; }

    /// <summary>
    /// Gets or sets the maximum distance clamp for attenuation calculations.
    /// Can be used in addition to Range for fine control.
    /// </summary>
    float MaxDistance { get; set; }
    
}
/// <summary>
/// Defines how light intensity attenuates over distance.
/// </summary>
public enum LightFalloff
{
    /// <summary>
    /// Physically correct inverse-square falloff (1 / r²).
    /// Standard for realistic lighting.
    /// </summary>
    InverseSquare,

    /// <summary>
    /// Linear falloff (1 / r).
    /// Less physically accurate but sometimes useful for artistic control.
    /// </summary>
    Linear,

    /// <summary>
    /// No falloff; intensity remains constant regardless of distance.
    /// Useful for stylized or debugging purposes.
    /// </summary>
    Constant
}