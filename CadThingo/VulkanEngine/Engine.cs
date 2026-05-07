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

    // Frame timing, queryable from anywhere. DeltaTime is seconds since the previous
    // frame; TotalTime is seconds since the first frame. Driven by a monotonic
    // Stopwatch so it's independent of any caller-supplied delta and unaffected by
    // wall-clock changes. Updated once per frame at the top of MainLoop.
    private static readonly System.Diagnostics.Stopwatch _frameTimer = System.Diagnostics.Stopwatch.StartNew();
    private static double _lastTotalTime;
    public static float DeltaTime { get; private set; }
    public static float TotalTime { get; private set; }

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

        keyboard.KeyDown += (sender, e, keyCode) =>
        {
            EventBus.PublishEvent(new KeyPressEvent(sender, (int)e));
        };
        keyboard.KeyUp += (sender, e, keyCode) =>
        {
            EventBus.PublishEvent(new KeyReleaseEvent(sender, (int)e));
        };

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
            EventBus.PublishEvent(new MouseMoveEvent(sender, e.X, e.Y, dx, dy));
        };

        mouse.MouseDown += (sender, e) =>
        {
            EventBus.PublishEvent(new MouseKeyDownEvent(sender, e));
        };
        mouse.MouseUp += (sender, e) =>
        {
            EventBus.PublishEvent(new MouseKeyReleaseEvent(sender, e));
        };

        // Silk's IMouse.Scroll fires once per wheel tick with a ScrollWheel struct
        // (X = horizontal, Y = vertical). Forward both axes; ImGui consumes both.
        mouse.Scroll += (sender, e) =>
        {
            EventBus.PublishEvent(new MouseScrollEvent(sender, e.X, e.Y));
        };

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
            // Tick global frame timer first so any system reading Engine.DeltaTime
            // / Engine.TotalTime this frame sees fresh values.
            double now = _frameTimer.Elapsed.TotalSeconds;
            DeltaTime = (float)(now - _lastTotalTime);
            _lastTotalTime = now;
            TotalTime = (float)now;

            // Camera maintains its own held-key state from KeyPress/KeyRelease events;
            // Tick(delta) just applies movement using that state for framerate independence.
            renderer.Camera.Tick((float)delta);
            EventBus.ProcessEvents();
        };
        window!.Render += delta =>
        {
            renderer.Update(delta);
        };
        window.Run();
    }
}