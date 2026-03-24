namespace CadThingo.Graphics.Assets3D.Lighting.Interfaces;
/// <summary>
/// Enables spotlight behavior for a light.
/// </summary>
public interface ISpotLight
{
    /// <summary>
    /// Gets or sets the inner cone angle (in degrees) for spot lights.
    /// Within this angle, the light is fully illuminated.
    /// </summary>
    float InnerConeAngle { get; set; }

    /// <summary>
    /// Gets or sets the outer cone angle (in degrees) for spot lights.
    /// Beyond this angle, the light has no effect.
    /// </summary>
    float OuterConeAngle { get; set; }

    /// <summary>
    /// Gets or sets the angular falloff controlling smoothness between inner and outer cones.
    /// Higher values produce softer edges.
    /// </summary>
    float AngularFallOff { get; set; }
}