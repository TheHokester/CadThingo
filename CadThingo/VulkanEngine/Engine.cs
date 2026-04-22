using System.Net.Sockets;
using Silk.NET.Input;
using Silk.NET.Maths;
using Silk.NET.Windowing;
using Renderer = CadThingo.VulkanEngine.Renderer.Renderer;
namespace CadThingo.VulkanEngine;

public class Engine
{
    //Window, Singleton? maybe
    public static IWindow? window;
    public static IInputContext? input;
    public static IKeyboard? keyboard;
    public static IMouse? mouse;
    
    
    //Renderer, Singleton?
    public static Renderer.Renderer renderer;
    //singleton event bus that all systems can subscribe to
    public static EventBus EventBus = new();
    
    //singleton resource manager
    public static AsyncResourceManager ResourceManager = new();


    public Engine()
    {
        
    }
    public void Start()
    {
        WindowOptions options = new()
        {
            API = GraphicsAPI.DefaultVulkan,
            Title = "CadThingo",
            Size = new Vector2D<int>(1280, 720),
            VSync = true,

        };
        
        window = Window.Create(options);
        window.Initialize();

        input = window.CreateInput();
        keyboard = input.Keyboards.First();
        mouse = input.Mice.First();

        mouse.MouseMove += (sender, e) => EventBus.PublishEvent(new MouseMoveEvent(e.X, e.Y));
        mouse.MouseDown += (sender, e) =>EventBus.PublishEvent(new MouseKeyDownEvent(e));
        mouse.MouseUp += (sender, e) => EventBus.PublishEvent(new MouseKeyReleaseEvent(e));
        // mouse.Scroll += (sender, e) => EventBus.PublishEvent(new )
        
        renderer = new( window);
        renderer.Initialize();
        
        window.Closing += Shutdown;
        
    }

    private void Shutdown()
    {
        renderer.Cleanup();
        window!.Dispose();
    }


    public void Run()
    {
        Start();
        MainLoop();
    }

    private void MainLoop()
    {
        while (true)
        {
            //do stuff part of regular operations
            //process events
            window!.DoEvents(); 
            EventBus.ProcessEvents();
            //render frame
            renderer.Update();
            
        }
    }
}