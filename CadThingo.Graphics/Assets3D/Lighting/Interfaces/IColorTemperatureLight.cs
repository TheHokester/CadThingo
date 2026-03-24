namespace CadThingo.Graphics.Assets3D.Lighting.Interfaces;
/// <summary>
/// Enables color temperature-based lighting.
/// </summary>
public interface IColorTemperatureLight
{
    /// <summary>
    /// Gets or sets whether color temperature is used to determine light color.
    /// When enabled, TemperatureKelvin overrides the Color property.
    /// </summary>
    bool UseColorTemperature { get; set; }

    /// <summary>
    /// Gets or sets the color temperature in Kelvin.
    /// Only used if UseColorTemperature is true.
    /// </summary>
    float? TemperatureKelvin { get; set; }
}