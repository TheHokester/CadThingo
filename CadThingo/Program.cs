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
    private static void Main(string[] args)
    {
        // Headless developer tools: compile + reflect every kernel, or exercise the Slang
        // interop, without bringing up a device.
        if (args.Contains("--shader-audit")) { VulkanEngine.Renderer.Shaders.ShaderAudit.Run(); return; }
        if (args.Contains("--slang-smoke")) { VulkanEngine.Renderer.Shaders.SlangSmokeTest.Run(); return; }

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