using Silk.NET.Maths;
using Silk.NET.Windowing;

namespace CadThingo;

public class App
{
    public static IWindow window;
    private VulkanRenderer renderer;
    
    const int width = 1280;
    const int height = 720;
    public const int MAX_FRAMES_IN_FLIGHT = 2;
    public void Run()
    {
        var options = WindowOptions.Default;
        options.API = GraphicsAPI.DefaultVulkan;
        options.Size = new Vector2D<int>(1280, 720);
        options.Title = "Initial Test";
        window = Window.Create(options);
        window.Initialize();
        
        renderer = new VulkanRenderer(window);

        renderer.InitVulkan();
        renderer.MainLoop();
        window.Closing += renderer.OnClose;

        

    }
}