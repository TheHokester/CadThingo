using System.Numerics;

namespace CadThingo.Graphics.Assets3D.Material;
public struct ColorRGB
{
    public byte R { get; }
    public byte G { get; }
    public byte B { get; }
    public byte A { get; } = 255;
    /**
     * byte r = 0-255;
     * byte g = 0-255;
     * byte b = 0-255;
     * byte a = 0-255;
     */
    public ColorRGB(byte r, byte g, byte b, byte a)
    {
        R = r;
        G = g;
        B = b;
        A = a;
    }
    /**
     * byte r = 0-255;
     * byte g = 0-255;
     * byte b = 0-255;
     */
    public ColorRGB(byte r, byte g, byte b) : this(r, g, b, 255){}
    /**
     * float r = 0-1f;
     * float g = 0-1f;
     * float b = 0-1f;
     */
    public ColorRGB(float r, float g, float b) : this((byte)(r * 255), (byte)(g * 255), (byte)(b * 255), 255){}
    /**
     * int rgbHex = 0xFF0000;
     * float opacity = 0-1f;
     */
    public ColorRGB(int rgbHex, float opacity) 
        : this((byte)(rgbHex >> 16),
            (byte)(rgbHex >> 8 & 0xFF),
            (byte)(rgbHex & 0xFF),
            (byte)(opacity * 255)){
        
    }
    /**
     * uint rgbaHex = 0xFF0000FF;
     */
    public ColorRGB(uint rgbaHex) 
        : this((byte)(rgbaHex >> 24),
            (byte)(rgbaHex >> 16 & 0xFF),
            (byte)(rgbaHex >> 8 & 0xFF),
            (byte)(rgbaHex & 0xFF)){}
    /**
     * int rgbHex = 0xFF0000;
     */
    public ColorRGB(int rgbHex) : this(rgbHex, 1f){}
    
    public ColorRGB Invert() => new((byte)~R, (byte)~G, (byte)~B, A);
    
    public ColorRGB WithAlpha(byte alpha) => new(R, G, B, alpha);
    
    public ColorRGB Lighter(float factor) => new((byte)(R + ~R * factor), (byte)(G + ~G * factor), (byte)(B + ~B * factor), A);
    public ColorRGB Lighter() => Lighter(0.2f);
    public ColorRGB Darker(float factor) => new((byte)(R - ~R * factor), (byte)(G - ~G * factor), (byte)(B - ~B * factor), A);
    public ColorRGB Darker() => Darker(0.2f);

    public ColorRGB GreyScale() => new((byte)(R * 0.3f + G * 0.59f + B * 0.11f), (byte)(R * 0.3f + G * 0.59f + B * 0.11f), (byte)(R * 0.3f + G * 0.59f + B * 0.11f), A);
    
    public Vector3 ToVector3() => new(R / 255f, G / 255f, B / 255f);
    public Vector4 ToVector4() => new(R / 255f, G / 255f, B / 255f, A / 255f);
}