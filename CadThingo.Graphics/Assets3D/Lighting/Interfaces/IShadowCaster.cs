namespace CadThingo.Graphics.Assets3D.Lighting.Interfaces;
/// <summary>
/// Enables shadow casting for a light.
/// </summary>
public interface IShadowCaster
{
    /// <summary>
    /// Gets or sets the type of shadow casting (e.g., hard, soft, cascaded).
    /// </summary>
    ShadowType ShadowType { get; set; }

    /// <summary>
    /// Gets or sets the quality level of shadow rendering.
    /// Typically controls resolution, filtering, and sampling.
    /// </summary>
    ShadowQuality ShadowQuality { get; set; }

    /// <summary>
    /// Gets or sets whether the light casts shadows.
    /// </summary>
    bool CastShadows { get; set; }

    /// <summary>
    /// Gets or sets the strength (darkness) of the shadows.
    /// 0 = no shadow, 1 = fully dark.
    /// </summary>
    float ShadowStrength { get; set; }

    /// <summary>
    /// Gets or sets the depth bias applied to shadow mapping.
    /// Helps reduce shadow acne artifacts.
    /// </summary>
    float ShadowBias { get; set; }

    /// <summary>
    /// Gets or sets the normal-based bias offset.
    /// Reduces self-shadowing by offsetting along surface normals.
    /// </summary>
    float ShadowNormalBias { get; set; }

    /// <summary>
    /// Gets or sets the slope-scaled depth bias.
    /// Increases bias on surfaces at grazing angles to the light.
    /// </summary>
    float ShadowSlopeBias { get; set; }

    /// <summary>
    /// Gets or sets the near clipping plane of the shadow camera.
    /// </summary>
    float ShadowNearPlane { get; set; }

    /// <summary>
    /// Gets or sets the far clipping plane of the shadow camera.
    /// </summary>
    float ShadowFarPlane { get; set; }

    /// <summary>
    /// Gets or sets the resolution of the shadow map (e.g., 1024, 2048).
    /// Higher values improve quality but increase cost.
    /// </summary>
    int ShadowMapResolution { get; set; }

    /// <summary>
    /// Gets or sets the number of shadow maps used (e.g., cubemap faces for point lights).
    /// </summary>
    int ShadowMapCount { get; set; }

    /// <summary>
    /// Gets or sets the perceived softness of shadows.
    /// Typically implemented via filtering techniques.
    /// </summary>
    float ShadowSoftness { get; set; }

    /// <summary>
    /// Gets or sets the explicit filter radius used in shadow filtering
    /// (e.g., PCF, VSM, or other techniques).
    /// </summary>
    float ShadowFilterRadius { get; set; }

    ///<summary>
    /// Gets or sets the distance at which shadows begin to fade out.
    /// </summary>
    float ShadowFadeDistance { get; set; }

    /// <summary>
    /// Gets or sets the range over which shadows fade to zero.
    /// Controls smoothness of the fade-out.
    /// </summary>
    float ShadowFadeRange { get; set; }
}

/// <summary>
/// Defines preset quality levels for shadow rendering.
/// Typically maps to resolution, filtering, and sample count.
/// </summary>
public enum ShadowQuality
{
    /// <summary>
    /// Lowest quality with minimal performance cost.
    /// </summary>
    Low,

    /// <summary>
    /// Balanced quality and performance.
    /// </summary>
    Medium,

    /// <summary>
    /// Highest quality with increased performance cost.
    /// </summary>
    High
}

/// <summary>
/// Defines the technique used to generate and filter shadows.
/// </summary>
public enum ShadowType
{
    /// <summary>
    /// No shadows are rendered.
    /// </summary>
    None,

    /// <summary>
    /// Hard shadows with no filtering.
    /// Fast but aliased edges.
    /// </summary>
    Hard,

    /// <summary>
    /// Generic soft shadows (implementation-dependent).
    /// </summary>
    Soft,

    /// <summary>
    /// Percentage-Closer Filtering (PCF).
    /// Widely used technique for soft shadow edges.
    /// </summary>
    PCF,

    /// <summary>
    /// Variance Shadow Maps (VSM).
    /// Enables smooth filtering but may introduce light bleeding.
    /// </summary>
    VSM,

    /// <summary>
    /// Exponential Shadow Maps (ESM).
    /// Reduces light bleeding compared to VSM.
    /// </summary>
    ESM,

    /// <summary>
    /// Ray traced shadows.
    /// Physically accurate but computationally expensive.
    /// </summary>
    RayTraced
}