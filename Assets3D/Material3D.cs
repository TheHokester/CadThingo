using System.Drawing;

namespace Cadthingo.Assets3D;

public sealed class Material3D
{
    public Color Color { get; set; } = Color.LightGray;
    public bool RenderWireframe { get; set; } = false;
    public float Opacity { get; set; } = 1f;
    
}

