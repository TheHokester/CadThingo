using System.Numerics;

namespace CadThingo.Graphics.Assets3D;

public readonly struct  Bounds3D
{
   public Vector3 Min { get; }
   public Vector3 Max { get; }
   
   public static Bounds3D Empty => new(Vector3.Zero, Vector3.Zero);
   public Bounds3D(Vector3 min, Vector3 max)
   {
       Min = min;
       Max = max;
   }
   public Vector3 Size => Max - Min;
   public Vector3 Center => (Min + Max) * 0.5f;

   public Bounds3D Transform(Matrix4x4 matrix)
   {
       var corners = new[]
       {
           new Vector3(Min.X, Min.Y, Min.Z),
           new Vector3(Max.X, Min.Y, Min.Z),
           new Vector3(Min.X, Max.Y, Min.Z),
           new Vector3(Max.X, Max.Y, Min.Z),
           new Vector3(Min.X, Min.Y, Max.Z),
           new Vector3(Max.X, Min.Y, Max.Z),
           new Vector3(Min.X, Max.Y, Max.Z),
           new Vector3(Max.X, Max.Y, Max.Z)

       };
       var transformed = corners
           .Select(c => Vector3.Transform(c, matrix))
           .ToArray();
       var min = transformed.Aggregate(Vector3.Min);
       var max = transformed.Aggregate(Vector3.Max);
       return new Bounds3D(min, max);
   }
   public static Bounds3D Union(Bounds3D a, Bounds3D b)
   {
       return new Bounds3D(Vector3.Min(a.Min, b.Min), Vector3.Max(a.Max, b.Max));
   }
}