using CadThingo.Graphics.Assets3D;
using Color = System.Drawing.Color;
using CadThingo.Graphics.Assets3D.Geometry;
namespace CadThingo.Graphics.Rendering;
using System.Numerics;
public class Renderer
{
    private readonly Camera camera = new();
    private float angle;
    private  FrameBuffer buffer;
    private Matrix4x4 mvp;
     

    public Renderer(FrameBuffer buffer)
    {
        this.buffer = buffer;
        mvp = camera.ViewMatrix;
    }

    public void Render(IEnumerable<RenderTriangle> triangles)
    {
        foreach (var tri in triangles)
        {
            buffer.clear();
            mvp = camera.ViewMatrix;
            RasterizeTriangle(tri);


        }
    }
    //3D coordinates relative to camera, 
    //check if a pixel(x, y) falls within the area of the triangle. 
    //calculate the height of that pixel on that triangle
    //compare this to the current pixel in that position
    // update it in z buffer if closest
    private void RasterizeTriangle(RenderTriangle tri)
    {
        Vector4 a = Transform(tri.A.Position);
        Vector4 b = Transform(tri.B.Position);
        Vector4 c = Transform(tri.C.Position);
        
        //perpective divide 
        a /= a.W;
        b /= b.W;
        c /= c.W;
        
        Vector2 pa = MapToScreen(a);
        Vector2 pb = MapToScreen(b);
        Vector2 pc = MapToScreen(c);

        int minX = (int)MathF.Max(0, MathF.Min(pa.X, MathF.Min(pb.X, pc.X)));
        int maxX = (int)MathF.Min(buffer.width-1, MathF.Max(pa.X, MathF.Max(pb.X, pc.X)));
        int minY = (int)MathF.Max(0, MathF.Min(pa.Y, MathF.Min(pb.Y, pc.Y)));
        int maxY = (int)MathF.Min(buffer.height-1, MathF.Max(pa.Y, MathF.Max(pb.Y, pc.Y)));
        float area = Edge(pa, pb, pc);
        
        for(int y =  minY; y <= maxY; y++)
        for (int x = minX; x <= maxX; x++)
        {
            Vector2 p = new Vector2(x, y);
            var alpha = Edge(pb, pc, p)/area;
            var beta = Edge(pc, pa, p)/area;
            var gamma  = Edge(pa, pb, p)/area;
            
            if(alpha < 0 ||  beta < 0 || gamma < 0) continue;

            var depth = alpha * a.Z + beta * b.Z + gamma * c.Z;
            int idx = buffer.Index(x, y);
            if (depth < buffer.depth[idx])
            {
                buffer.depth[idx] = depth;
                buffer.color[idx] = tri.color;
            }
        }
    }
    //uses the cross product to find the area between points a,b,c 
    float Edge(Vector2 a, Vector2 b, Vector2 c)
    {
        return (c.X - a.X) * (b.Y - a.Y)
               - (c.Y - a.Y) * (b.X - a.X);
    }


    Vector4 Transform(Vector3 v)
    {
        return Vector4.Transform(new Vector4(v,1), mvp);
    }

    Vector2 MapToScreen(Vector4 v)
    {
        return new Vector2(
            (v.X * 0.5f + 0.5f) * buffer.width,
            (v.Y * 0.5f + 0.5f) * buffer.height);
        
    }
}