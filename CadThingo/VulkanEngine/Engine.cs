using System.Net.Sockets;
using System.Numerics;
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
    public static ResourceManager ResourceManager = new();

    // Mouse-delta state. GLFW/Silk.NET MouseMove gives absolute position per event,
    // so we convert to deltas here. _firstMouse prevents a huge jump on the first
    // event (where _lastMousePos is still default).
    private Vector2 _lastMousePos;
    private bool _firstMouse = true;

    public Engine()
    {

    }
    public void Start()
    {
        var options = WindowOptions.Default;

        options.API = GraphicsAPI.DefaultVulkan;
        options.Title = "CadThingo";
        options.Size = new Vector2D<int>(1280, 720);
        options.VSync = true;


        window = Window.Create(options);
        window.Initialize();

        input = window.CreateInput();
        keyboard = input.Keyboards.First();
        mouse = input.Mice.First();

        keyboard.KeyDown += (sender, e, keyCode) => EventBus.PublishEvent(new KeyPressEvent((int)e));
        keyboard.KeyUp += (sender, e, keyCode) => EventBus.PublishEvent(new KeyReleaseEvent((int)e));

        mouse.MouseMove += (sender, e) =>
        {
            if (_firstMouse)
            {
                _lastMousePos = e;
                _firstMouse = false;
                return;
            }
            var dx = e.X - _lastMousePos.X;
            var dy = e.Y - _lastMousePos.Y;
            _lastMousePos = e;
            EventBus.PublishEvent(new MouseMoveEvent(dx, dy));
        };
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
        // Silk.NET's IWindow.Run() is blocking and pumps events + fires
        // Update/Render events until the window closes. Hook once then Run once.
        window!.Update += delta =>
        {
            // Poll continuous inputs (keyboard-hold). Discrete inputs (presses,
            // mouse moves/clicks) already fire via the EventBus on the event thread.
            renderer.Camera.Tick(keyboard!, (float)delta);
            EventBus.ProcessEvents();
        };
        window!.Render += delta =>
        {
            renderer.Update(delta);
        };
        window.Run();
    }
}