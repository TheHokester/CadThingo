using Silk.NET.Maths;
using Silk.NET.Windowing;

namespace CadThingo;

public class App
{
    public static IWindow window;
    private VulkanRenderer renderer;

    public void Run()
    {
        var options = WindowOptions.Default;
        options.API = GraphicsAPI.DefaultVulkan;
        options.Size = new Vector2D<int>(1280, 720);
        options.Title = "Initial Test";
        window = Window.Create(options);

        renderer = new VulkanRenderer(window);

        window.Load += renderer.OnLoad;
        window.Render += renderer.OnRender;
        window.Closing += renderer.OnClose;

        window.Run();

    }
}