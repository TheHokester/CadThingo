using CadThingo.VulkanEngine;
using CadThingo.VulkanEngine.Renderer.Slang;
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
    private static int Main(string[] args)
    {
        // Headless developer tools: compile + reflect every kernel, or exercise the Slang
        // interop, without bringing up a device. The audit's exit code is what CI gates on.
        if (args.Contains("--shader-audit")) return ShaderAudit.Run();
        if (args.Contains("--slang-smoke")) { SlangSmokeTest.Run(); return 0; }

        var app = new Engine();
        app.Run();
        return 0;
    }
}