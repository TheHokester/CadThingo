using Silk.NET.Vulkan;

namespace CadThingo;

public static class Globals
{
    public static bool IsDebug => false;
    public static readonly Vk vk = Vk.GetApi();
    
    public static bool Initialize()
    {
        return true;
    }
}