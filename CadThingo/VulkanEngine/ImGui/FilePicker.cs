using System;
using System.Runtime.InteropServices;
using System.Text;

namespace CadThingo.VulkanEngine.ImGui;

/// <summary>
/// Thin P/Invoke wrapper around comdlg32!GetOpenFileNameW. Single-file open
/// dialog only — multi-select / save / folder pickers are out of scope.
///
/// Caller passes a filter list (label, "*.glb;*.gltf") and gets back the
/// absolute path the user picked, or null if the dialog was cancelled.
/// </summary>
internal static class FilePicker
{
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    private struct OpenFileName
    {
        public int      lStructSize;
        public IntPtr   hwndOwner;
        public IntPtr   hInstance;
        public string?  lpstrFilter;
        public string?  lpstrCustomFilter;
        public int      nMaxCustFilter;
        public int      nFilterIndex;
        public IntPtr   lpstrFile;          // wchar_t* buffer we own
        public int      nMaxFile;
        public IntPtr   lpstrFileTitle;
        public int      nMaxFileTitle;
        public string?  lpstrInitialDir;
        public string?  lpstrTitle;
        public int      Flags;
        public short    nFileOffset;
        public short    nFileExtension;
        public string?  lpstrDefExt;
        public IntPtr   lCustData;
        public IntPtr   lpfnHook;
        public string?  lpTemplateName;
        public IntPtr   pvReserved;
        public int      dwReserved;
        public int      flagsEx;
    }

    private const int OFN_FILEMUSTEXIST   = 0x00001000;
    private const int OFN_PATHMUSTEXIST   = 0x00000800;
    private const int OFN_NOCHANGEDIR     = 0x00000008;
    private const int OFN_EXPLORER        = 0x00080000;

    [DllImport("comdlg32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool GetOpenFileNameW(ref OpenFileName ofn);

    /// <summary>
    /// Shows a Windows open-file dialog. <paramref name="filterPairs"/> alternates
    /// label and pattern, e.g. ["glTF", "*.glb;*.gltf", "All files", "*.*"].
    /// Returns the chosen path or null if cancelled / not on Windows.
    /// </summary>
    public static string? Open(string title, string initialDir, params string[] filterPairs)
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return null;

        // GetOpenFileNameW filter is a double-null-terminated list of pairs:
        // "Label\0Pattern\0Label\0Pattern\0\0". String literals with embedded \0
        // marshal correctly because the struct field is CharSet.Unicode.
        var sb = new StringBuilder();
        for (int i = 0; i + 1 < filterPairs.Length; i += 2)
        {
            sb.Append(filterPairs[i]).Append('\0').Append(filterPairs[i + 1]).Append('\0');
        }
        sb.Append('\0');

        const int bufferChars = 4096;
        IntPtr buffer = Marshal.AllocHGlobal(bufferChars * sizeof(char));
        try
        {
            // Zero the buffer so the dialog sees an empty initial filename.
            for (int i = 0; i < bufferChars; i++)
                Marshal.WriteInt16(buffer, i * sizeof(char), 0);

            var ofn = new OpenFileName
            {
                lStructSize     = Marshal.SizeOf<OpenFileName>(),
                lpstrFilter     = sb.ToString(),
                nFilterIndex    = 1,
                lpstrFile       = buffer,
                nMaxFile        = bufferChars,
                lpstrInitialDir = initialDir,
                lpstrTitle      = title,
                Flags           = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR | OFN_EXPLORER,
            };

            if (!GetOpenFileNameW(ref ofn))
                return null;

            return Marshal.PtrToStringUni(buffer);
        }
        finally
        {
            Marshal.FreeHGlobal(buffer);
        }
    }
}