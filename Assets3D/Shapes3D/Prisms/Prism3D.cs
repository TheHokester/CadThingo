using System.Drawing;
using System.Numerics;
using System.Runtime.InteropServices.Java;
using CadThingo.Assets3D;
using Cadthingo.Assets3D.Geometry;
using CadThingo.Assets3D.Shapes3D;
using CadThingo.Rendering;

namespace Cadthingo.Assets3D.Shapes3D.Prisms;

public abstract class Prism3D : Shape3D
{
    public float Length { get;}

    private readonly IReadOnlyList<Vector2> _profile2D;

   
    public override Bounds3D GetLocalBounds()
    {
        var halfLength = Length / 2;
        
        var minX = _profile2D.Min(p => p.X);
        var maxX = _profile2D.Max(p => p.X);
        var minY = _profile2D.Min(p => p.Y);
        var maxY = _profile2D.Max(p => p.Y);
        return new Bounds3D(
            new Vector3(minX, minY, -halfLength),
            new Vector3(maxX, maxY, halfLength));
    }
    protected Prism3D(IEnumerable<Vector2> profile, float length, Vector3 position )
    {
        ArgumentNullException.ThrowIfNull(profile);

        var normalizedProfile = NormalizeAndValidateProfile(profile).ToArray();
        
        if(normalizedProfile.Length < 3) 
            throw new ArgumentException("Profile must have at least 3 vertices");
        if(length <= 0f) 
            throw new ArgumentOutOfRangeException(nameof(length), "Length must be greater than 0");
        
        _profile2D = normalizedProfile;
        Length = length;
        Position = position;
    }
    
    protected IReadOnlyList<Vector2> GetProfile2D() => _profile2D;

    protected IReadOnlyList<Vector3> GetFrontFaceVertices()
    {
        float z = -Length * 0.5f;
        return _profile2D.Select(p => new Vector3(p.X, p.Y, z)).ToArray();
    }
    protected IReadOnlyList<Vector3> GetBackFaceVertices()
    {
        float z = Length * 0.5f;
        return _profile2D.Select(p => new Vector3(p.X, p.Y, z)).ToArray();
    }

    protected IReadOnlyList<Vector3> GetLocalVertices()
    {
        var front = GetFrontFaceVertices();
        var back = GetBackFaceVertices();
        
        var all = new List<Vector3>(front.Count + back.Count);
        all.AddRange(front);
        all.AddRange(back);
        return all;
    }

    public override IEnumerable<RenderTriangle> GetTriangles()
    {
       var front = GetFrontFaceVertices();
       var back = GetBackFaceVertices();
       int count = _profile2D.Count;
       var color = Material.Color;

       foreach (var tri in BuildFrontCap(front, color))
       {
           yield return tri;
       }

       foreach (var tri in BuildBackCap(back, color))
       {
           yield return tri;
       }

       for (int i = 0; i < count; i++)
       {
           int next = (i + 1) % count;
           
           Vector3 frontA = front[i];
           Vector3 frontB = front[next];
           Vector3 backA = back[i];
           Vector3 backB = back[next];

           yield return new RenderTriangle(
               new Vertex(frontA),
               new Vertex(frontB),
               new Vertex(backB),
               color);
           yield return new RenderTriangle(
               new Vertex(frontA),
               new Vertex(backB),
               new Vertex(backA),
               color);
       }
    }



    public override IEnumerable<Edge3D> GetEdges()
    {
        var front = GetFrontFaceVertices();
        var back = GetBackFaceVertices();
        int count = _profile2D.Count;

        for (int i = 0; i < count; i++)
        {
            int next = (i + 1) % count;
            
            yield return new Edge3D(new Vertex(front[i]), new Vertex(front[next]));
            yield return new Edge3D(new Vertex(back[i]), new Vertex(back[next]));
            yield return new Edge3D(new Vertex(front[i]), new Vertex(back[i]));
        }

    }

    private static IEnumerable<RenderTriangle> BuildFrontCap(IReadOnlyList<Vector3> front, Color color)
    {
        for (int i = 1; i < front.Count - 1; i++)
        {
            yield return new RenderTriangle(
                new Vertex(front[0]),
                new Vertex(front[i + 1]),
                new Vertex(front[i]),
                color);
        }
    }

    private static IEnumerable<RenderTriangle> BuildBackCap(IReadOnlyList<Vector3> back, Color color)
    {
        for(int i = 0; i < back.Count - 1; i++)
        {
            yield return new RenderTriangle(
                new Vertex(back[0]),
                new Vertex(back[i]),
                new Vertex(back[i + 1]),
                color);
        }
    }

    private static IEnumerable<Vector2> NormalizeAndValidateProfile(IEnumerable<Vector2> profile)
    {
        var points = profile.Distinct().ToArray();

        if (points.Length < 3)
            throw new ArgumentException("Profile must have at least 3 vertices");
        
        float signedArea = GetSignedArea(points);
        if(MathF.Abs(signedArea) < 1e-6f) 
            throw new ArgumentException("Profile cannot be degenerate", nameof(profile));
        
        var centroid = GetPolygonCentroid(points);
        var centered = points.Select(p => p - centroid).ToArray();
        
        if(GetSignedArea(centered) < 0f)
            Array.Reverse(centered);
        if (!IsConvex(centered))
            throw new ArgumentException("Prism3D implementation doesn't support concave polygons");
        return centered;
    }

    private static float GetSignedArea(IReadOnlyList<Vector2> points)
    {
        float area = 0f;

        for (var i = 0; i < points.Count; i++)
        {
            var next = (i + 1) % points.Count;
            area += Vector2.Cross(points[i], points[next]); 
        }
        return area* 0.5f;
    }

    private static Vector2 GetPolygonCentroid(IReadOnlyList<Vector2> points)
    {
        var areaFactor = 0f;
        var cx = 0f;
        var cy = 0f;

        for (var i = 0; i < points.Count; i++)
        {
            int next = (i + 1) % points.Count;
            float cross = Vector2.Cross(points[i], points[next]);
            areaFactor += cross;
            cx += (points[i].X + points[next].X) * cross;
            cy += (points[i].Y + points[next].Y) * cross;
        }

        float area = areaFactor * 0.5f;
        if(MathF.Abs(area) < 1e-6f) 
            throw new ArgumentException("Cannot calculate centroid for a degenerate polygon");
        
        float scale = 1f / 6f * area;
        return new Vector2(cx * scale, cy * scale);
    }

    private static bool IsConvex(IReadOnlyList<Vector2> points)
    {
        if (points.Count < 3)
            return false;

        float? sign = null;

        for (var i = 0; i < points.Count; i++)
        {
            Vector2 a = points[i];
            Vector2 b = points[(i + 1) % points.Count];
            Vector2 c = points[(i + 2) % points.Count];
            
            float cross = Vector2.Cross(b - a, c - a);
            if (MathF.Abs(cross) < 1e-6f)
                continue;
            float currentSign = MathF.Sign(cross);

            if (sign is null)
            {
                sign = currentSign;
            }
            else if (currentSign != sign.Value)
            {
                return false;
            }
        }
        return true;
    }
}