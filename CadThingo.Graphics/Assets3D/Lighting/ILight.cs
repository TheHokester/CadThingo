using System.Numerics;
using CadThingo.Graphics.Assets3D.Material;

namespace CadThingo.Graphics.Assets3D.Lighting;
/// <summary>
/// Base interface for all light sources.
/// </summary>
public interface ILight : IIdentity, ITransform3D
{
    /// <summary>
    /// Gets or sets whether the light is enabled and contributes to the scene.
    /// Disabled lights are ignored during rendering.
    /// </summary>
    bool Enabled { get; set; }

    /// <summary>
    /// Gets or sets the type of the light (e.g., Directional, Point, Spot, Area).
    /// Determines how the light is evaluated in the lighting pipeline.
    /// </summary>
    LightType Type { get; set; }

    /// <summary>
    /// Gets or sets the rendering mode of the light (e.g., realtime, baked, mixed).
    /// Controls how and when the light contributes to lighting calculations.
    /// </summary>
    LightRenderMode RenderMode { get; set; }

    /// <summary>
    /// Gets or sets the priority of the light when resolving limits (e.g., max active lights).
    /// Higher priority lights are processed first.
    /// </summary>
    int Priority { get; set; }

    /// <summary>
    /// Gets or sets the layer mask used to determine which objects this light affects.
    /// </summary>
    uint LayerMask { get; set; }

    /// <summary>
    /// Gets or sets flags defining which lighting components are affected
    /// (e.g., diffuse, specular, volumetrics, shadows).
    /// </summary>
    LightAffectFlags AffectFlags { get; set; }

    /// <summary>
    /// Gets or sets the base color of the light in linear RGB space.
    /// This value is used before intensity and exposure adjustments.
    /// </summary>
    ColorRGB Color { get; set; }

    /// <summary>
    /// Gets or sets the intensity (brightness) of the light in the specified unit.
    /// </summary>
    float Intensity { get; set; }

    /// <summary>
    /// Gets or sets the unit used for intensity (e.g., lumens, candelas, lux).
    /// Determines how intensity is interpreted physically.
    /// </summary>
    LightIntensityUnit IntensityUnit { get; set; }

    /// <summary>
    /// Gets or sets an additional multiplier applied after photometric conversion.
    /// Useful for artistic control without breaking physical correctness.
    /// </summary>
    float ExposureWeight { get; set; }

    /// <summary>
    /// Gets or sets a multiplier applied only to the diffuse (Lambertian) lighting component.
    /// </summary>
    float DiffuseIntensity { get; set; }

    /// <summary>
    /// Gets or sets a multiplier applied only to the specular lighting component.
    /// </summary>
    float SpecularIntensity { get; set; }
}

public enum LightType
{
    Directional,
    Point,
    Spot, 
    AreaRect, 
    AreaDisc, 
    AreaSphere
}


/// <summary>
/// Defines the unit used to interpret light intensity.
/// Enables physically-based lighting workflows.
/// </summary>
public enum LightIntensityUnit
{
    /// <summary>
    /// Unitless scalar (typically 0–1 or arbitrary range).
    /// Not physically based.
    /// </summary>
    Unitless,

    /// <summary>
    /// Lumens (lm) — total luminous flux emitted by the light.
    /// Common for point and spot lights.
    /// </summary>
    Lumens,

    /// <summary>
    /// Candelas (cd) — luminous intensity in a given direction.
    /// Useful for directional emission.
    /// </summary>
    Candelas,

    /// <summary>
    /// Lux (lx) — illuminance (lumens per square meter) at a surface.
    /// Often defined at a reference distance (e.g., 1 meter).
    /// </summary>
    Lux,

    /// <summary>
    /// Nits (cd/m²) — luminance emitted from a surface.
    /// Primarily used for area lights and emissive surfaces.
    /// </summary>
    Nits
}

/// <summary>
/// Defines how a light contributes to the scene in terms of baking and realtime updates.
/// </summary>
public enum LightRenderMode
{
    /// <summary>
    /// Fully dynamic light evaluated every frame.
    /// Supports real-time shadows and movement.
    /// </summary>
    Realtime,

    /// <summary>
    /// Combination of baked and realtime lighting.
    /// Static objects receive baked lighting; dynamic objects use realtime.
    /// </summary>
    Mixed,

    /// <summary>
    /// Fully baked into lightmaps.
    /// No runtime cost but cannot change at runtime.
    /// </summary>
    Baked
}




/// <summary>
/// Flags controlling which lighting components are affected by the light.
/// Can be combined using bitwise operations.
/// </summary>
[Flags]
public enum LightAffectFlags
{
    /// <summary>
    /// The light does not affect any components.
    /// </summary>
    None = 0,

    /// <summary>
    /// Affects diffuse (Lambertian) lighting.
    /// </summary>
    Diffuse = 1 << 0,

    /// <summary>
    /// Affects specular highlights.
    /// </summary>
    Specular = 1 << 1,

    /// <summary>
    /// Contributes to reflection probes or screen-space reflections.
    /// </summary>
    Reflections = 1 << 2,

    /// <summary>
    /// Affects volumetric lighting (e.g., fog, light shafts).
    /// </summary>
    Volumetrics = 1 << 3,

    /// <summary>
    /// Contributes to global illumination systems.
    /// </summary>
    GI = 1 << 4,

    /// <summary>
    /// Affects all lighting components.
    /// </summary>
    All = ~0
}

