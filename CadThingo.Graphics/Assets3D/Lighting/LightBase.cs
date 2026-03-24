using System.Numerics;
using CadThingo.Graphics.Assets3D;
using CadThingo.Graphics.Assets3D.Material;

namespace CadThingo.Graphics.Assets3D.Lighting;

// public abstract class LightBase : ILight
// {
//      public abstract string Description { get; }
//      public abstract string Name { get; }
//      public Guid Id { get; } = Guid.NewGuid();
//      
//      public Vector3 Scale { get; set; }
//      public Vector3 Position { get; set; }
//      public Vector3 Pivot { get; set; }
//      public Quaternion Rotation { get; set; }
//
//      public Matrix4x4 LocalMatrix
//      {
//           get
//           {
//                var moveToPivot = Matrix4x4.CreateTranslation(-Pivot);
//                var scale = Matrix4x4.CreateScale(Scale);
//                var rotate = Matrix4x4.CreateFromQuaternion(Rotation);
//                var moveBack = Matrix4x4.CreateTranslation(Pivot);
//                return moveToPivot * scale * rotate * moveBack;
//           }
//      }
//      public Matrix4x4 WorldMatrix { get; set;}
//      
// }