using ImGuiNET;
using Silk.NET.Input;

namespace CadThingo.VulkanEngine.ImGui;

public static class ImGuiHelpers
{
    // GLFW modifier bitmask layout — matches the `mods` argument from GLFW's key callback.
    public const int GlfwModShift   = 0x0001;
    public const int GlfwModControl = 0x0002;
    public const int GlfwModAlt     = 0x0004;
    public const int GlfwModSuper   = 0x0008;
    public const int GlfwModCapsLock = 0x0010;
    public const int GlfwModNumLock  = 0x0020;

    // Silk.NET.Input.Key values are aligned with GLFW key codes by design,
    // so this overload just casts and forwards.
    public static bool TryMapGlfwKey(Key key, out ImGuiKey imKey)
        => TryMapGlfwKey((int)key, out imKey);

    public static bool TryMapGlfwKey(int key, out ImGuiKey imKey)
    {
        imKey = key switch
        {
            // Whitespace / control
            32  => ImGuiKey.Space,
            256 => ImGuiKey.Escape,
            257 => ImGuiKey.Enter,
            258 => ImGuiKey.Tab,
            259 => ImGuiKey.Backspace,
            260 => ImGuiKey.Insert,
            261 => ImGuiKey.Delete,

            // Arrows
            262 => ImGuiKey.RightArrow,
            263 => ImGuiKey.LeftArrow,
            264 => ImGuiKey.DownArrow,
            265 => ImGuiKey.UpArrow,

            // Navigation
            266 => ImGuiKey.PageUp,
            267 => ImGuiKey.PageDown,
            268 => ImGuiKey.Home,
            269 => ImGuiKey.End,

            // Locks / system
            280 => ImGuiKey.CapsLock,
            281 => ImGuiKey.ScrollLock,
            282 => ImGuiKey.NumLock,
            283 => ImGuiKey.PrintScreen,
            284 => ImGuiKey.Pause,

            // Punctuation (US layout)
            39  => ImGuiKey.Apostrophe,
            44  => ImGuiKey.Comma,
            45  => ImGuiKey.Minus,
            46  => ImGuiKey.Period,
            47  => ImGuiKey.Slash,
            59  => ImGuiKey.Semicolon,
            61  => ImGuiKey.Equal,
            91  => ImGuiKey.LeftBracket,
            92  => ImGuiKey.Backslash,
            93  => ImGuiKey.RightBracket,
            96  => ImGuiKey.GraveAccent,

            // Digits 0-9
            48 => ImGuiKey._0,
            49 => ImGuiKey._1,
            50 => ImGuiKey._2,
            51 => ImGuiKey._3,
            52 => ImGuiKey._4,
            53 => ImGuiKey._5,
            54 => ImGuiKey._6,
            55 => ImGuiKey._7,
            56 => ImGuiKey._8,
            57 => ImGuiKey._9,

            // Letters A-Z
            65 => ImGuiKey.A,
            66 => ImGuiKey.B,
            67 => ImGuiKey.C,
            68 => ImGuiKey.D,
            69 => ImGuiKey.E,
            70 => ImGuiKey.F,
            71 => ImGuiKey.G,
            72 => ImGuiKey.H,
            73 => ImGuiKey.I,
            74 => ImGuiKey.J,
            75 => ImGuiKey.K,
            76 => ImGuiKey.L,
            77 => ImGuiKey.M,
            78 => ImGuiKey.N,
            79 => ImGuiKey.O,
            80 => ImGuiKey.P,
            81 => ImGuiKey.Q,
            82 => ImGuiKey.R,
            83 => ImGuiKey.S,
            84 => ImGuiKey.T,
            85 => ImGuiKey.U,
            86 => ImGuiKey.V,
            87 => ImGuiKey.W,
            88 => ImGuiKey.X,
            89 => ImGuiKey.Y,
            90 => ImGuiKey.Z,

            // Function keys F1-F12 (ImGuiKey only defines through F12)
            290 => ImGuiKey.F1,
            291 => ImGuiKey.F2,
            292 => ImGuiKey.F3,
            293 => ImGuiKey.F4,
            294 => ImGuiKey.F5,
            295 => ImGuiKey.F6,
            296 => ImGuiKey.F7,
            297 => ImGuiKey.F8,
            298 => ImGuiKey.F9,
            299 => ImGuiKey.F10,
            300 => ImGuiKey.F11,
            301 => ImGuiKey.F12,

            // Keypad
            320 => ImGuiKey.Keypad0,
            321 => ImGuiKey.Keypad1,
            322 => ImGuiKey.Keypad2,
            323 => ImGuiKey.Keypad3,
            324 => ImGuiKey.Keypad4,
            325 => ImGuiKey.Keypad5,
            326 => ImGuiKey.Keypad6,
            327 => ImGuiKey.Keypad7,
            328 => ImGuiKey.Keypad8,
            329 => ImGuiKey.Keypad9,
            330 => ImGuiKey.KeypadDecimal,
            331 => ImGuiKey.KeypadDivide,
            332 => ImGuiKey.KeypadMultiply,
            333 => ImGuiKey.KeypadSubtract,
            334 => ImGuiKey.KeypadAdd,
            335 => ImGuiKey.KeypadEnter,
            336 => ImGuiKey.KeypadEqual,

            // Modifiers
            340 => ImGuiKey.LeftShift,
            341 => ImGuiKey.LeftCtrl,
            342 => ImGuiKey.LeftAlt,
            343 => ImGuiKey.LeftSuper,
            344 => ImGuiKey.RightShift,
            345 => ImGuiKey.RightCtrl,
            346 => ImGuiKey.RightAlt,
            347 => ImGuiKey.RightSuper,
            348 => ImGuiKey.Menu,

            _ => ImGuiKey.None,
        };
        return imKey != ImGuiKey.None;
    }

    // Reconstruct the GLFW-style mods bitmask from a Silk.NET keyboard's current state,
    // for cases where the callback didn't carry one (Silk's IKeyboard.KeyDown/Up don't).
    public static int GetMods(IKeyboard kb)
    {
        int mods = 0;
        if (kb.IsKeyPressed(Key.ShiftLeft) || kb.IsKeyPressed(Key.ShiftRight)) mods |= GlfwModShift;
        if (kb.IsKeyPressed(Key.ControlLeft) || kb.IsKeyPressed(Key.ControlRight)) mods |= GlfwModControl;
        if (kb.IsKeyPressed(Key.AltLeft) || kb.IsKeyPressed(Key.AltRight)) mods |= GlfwModAlt;
        if (kb.IsKeyPressed(Key.SuperLeft) || kb.IsKeyPressed(Key.SuperRight)) mods |= GlfwModSuper;
        return mods;
    }
}