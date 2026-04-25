using CadThingo.VulkanEngine;
using Silk.NET.Core;
using Silk.NET.Core.Native;

namespace CadThingo;

using System.Xml;
using Silk.NET.Maths;
using Silk.NET.Vulkan;
using Silk.NET.Windowing;


public class Program
{
    private static IWindow? window;
    private const bool IsTutorial = false;
    private static void Main()
    {
        if (IsTutorial)
        {
            var app = new App();
            app.Run();
        }
        else
        {
            var app = new Engine();
            app.Run();
            
        }
    }
}