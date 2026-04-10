using System.Numerics;

namespace CadThingo.VulkanEngine;

public class Scene
{
    public bool RayCast(Ray ray, ref RayCastHit hit, float rayLength)
    {
        return false;
    }
}

public struct Ray
{
    public Vector3 origin;
    public Vector3 direction;
}
public struct RayCastHit
{
    
    public Vector3 point;
}