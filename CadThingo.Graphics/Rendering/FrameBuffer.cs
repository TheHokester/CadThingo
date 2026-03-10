using System.Drawing;

namespace CadThingo.Graphics.Rendering;

public class FrameBuffer
{
    public readonly int width;
    public readonly int height;
    
    public readonly Color[] color;
    public readonly float[] depth;

    public FrameBuffer(int width, int height)
    {
        this.width = width;
        this.height = height;
        color = new Color[width * height];
        depth = new float[width * height];
        
        
    }

    public void clear()
    {
        for (int i = 0; i < color.Length; i++)
        {
            color[i] = Color.Black;
            depth[i] = float.PositiveInfinity;
        }
    }
    
    public int Index(int x, int y) => y * width + x;
}