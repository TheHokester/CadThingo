using System.Globalization;
using System.Text;

namespace CadThingo.VulkanEngine;

/// <summary>
/// Minimal Radiance RGBE (.hdr) decoder. Returns linear RGBA float32 pixels in
/// row-major top-to-bottom order so the result blits directly into a Vulkan 2D
/// image without a vertical flip. Equirectangular layout — the IBL bake passes
/// unwrap it to a cubemap in compute.
/// </summary>
public static class HdrLoader
{
    public readonly record struct HdrImage(float[] Pixels, int Width, int Height);

    public static HdrImage Load(string path)
    {
        using var fs = File.OpenRead(path);

        // ── Header ──
        // The magic line is "#?RADIANCE" (modern) or "#?RGBE" (legacy). Everything
        // before the first blank line is metadata; we only validate FORMAT and
        // then look for the resolution line that follows the blank.
        var magic = ReadLine(fs)
            ?? throw new InvalidDataException($"Empty file: {path}");
        if (!magic.StartsWith("#?RADIANCE") && !magic.StartsWith("#?RGBE"))
            throw new InvalidDataException($"Not a Radiance .hdr file: {path}");

        string? line;
        while ((line = ReadLine(fs)) != null)
        {
            if (line.Length == 0) break;
            if (line.StartsWith("FORMAT=") && !line.Contains("32-bit_rle_rgbe"))
                throw new InvalidDataException($"Unsupported HDR format line '{line}' — only 32-bit_rle_rgbe is decoded.");
        }

        // Resolution line — typical form is "-Y <height> +X <width>" which means
        // the file is stored top-to-bottom, left-to-right. "+Y" means bottom-to-top;
        // we vertical-flip in that case so the output is always top-down.
        var resLine = ReadLine(fs) ?? throw new InvalidDataException("Missing HDR resolution line");
        var parts = resLine.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length < 4)
            throw new InvalidDataException($"Bad HDR resolution line '{resLine}'");
        if (parts[2] != "+X")
            throw new InvalidDataException($"Unsupported HDR scan order '{resLine}' — only +X width is decoded.");
        bool flipY = parts[0] == "+Y";
        if (!flipY && parts[0] != "-Y")
            throw new InvalidDataException($"Unsupported HDR scan order '{resLine}'");

        int height = int.Parse(parts[1], CultureInfo.InvariantCulture);
        int width  = int.Parse(parts[3], CultureInfo.InvariantCulture);

        var output = new float[width * height * 4];
        var scanline = new byte[width * 4];

        for (int y = 0; y < height; y++)
        {
            DecodeScanline(fs, scanline, width);

            int dstRow = (flipY ? (height - 1 - y) : y) * width * 4;
            for (int x = 0; x < width; x++)
            {
                int s = x * 4;
                byte r = scanline[s + 0];
                byte g = scanline[s + 1];
                byte b = scanline[s + 2];
                byte e = scanline[s + 3];

                if (e == 0)
                {
                    output[dstRow + x * 4 + 0] = 0f;
                    output[dstRow + x * 4 + 1] = 0f;
                    output[dstRow + x * 4 + 2] = 0f;
                }
                else
                {
                    // mantissa = m * 2^(e - 128) / 255
                    float f = MathF.Pow(2f, e - 128) / 255f;
                    output[dstRow + x * 4 + 0] = r * f;
                    output[dstRow + x * 4 + 1] = g * f;
                    output[dstRow + x * 4 + 2] = b * f;
                }
                output[dstRow + x * 4 + 3] = 1f;
            }
        }

        return new HdrImage(output, width, height);
    }

    /// <summary>
    /// Reads one scanline of width pixels into <paramref name="dst"/> (4 bytes
    /// per pixel, R/G/B/E interleaved). Handles both the new adaptive-RLE
    /// per-channel format and the rare uncompressed fallback.
    /// </summary>
    static void DecodeScanline(Stream s, byte[] dst, int width)
    {
        // Spec: adaptive RLE only kicks in for widths in [8, 32767].
        if (width < 8 || width > 0x7FFF)
        {
            ReadExact(s, dst, 0, width * 4);
            return;
        }

        long mark = s.Position;
        int b0 = s.ReadByte();
        int b1 = s.ReadByte();
        int b2 = s.ReadByte();
        int b3 = s.ReadByte();
        if (b0 < 0 || b1 < 0 || b2 < 0 || b3 < 0)
            throw new EndOfStreamException("Truncated HDR scanline header");

        // New-style header: 0x02 0x02 hi lo, with (hi<<8|lo) == width and hi's
        // high bit clear. Anything else means the file is stored uncompressed
        // or in the old per-pixel RLE format — rewind and fall back.
        bool newStyle = b0 == 2 && b1 == 2 && (b2 & 0x80) == 0 && ((b2 << 8) | b3) == width;
        if (!newStyle)
        {
            s.Position = mark;
            ReadExact(s, dst, 0, width * 4);
            return;
        }

        // R, G, B, E each RLE-compressed independently across the row.
        for (int channel = 0; channel < 4; channel++)
        {
            int x = 0;
            while (x < width)
            {
                int count = s.ReadByte();
                if (count < 0) throw new EndOfStreamException("Truncated HDR run");
                if (count > 128)
                {
                    int run = count - 128;
                    if (x + run > width) throw new InvalidDataException("HDR run overruns scanline");
                    int v = s.ReadByte();
                    if (v < 0) throw new EndOfStreamException("Truncated HDR run value");
                    for (int i = 0; i < run; i++)
                        dst[(x + i) * 4 + channel] = (byte)v;
                    x += run;
                }
                else
                {
                    if (x + count > width) throw new InvalidDataException("HDR literal overruns scanline");
                    for (int i = 0; i < count; i++)
                    {
                        int v = s.ReadByte();
                        if (v < 0) throw new EndOfStreamException("Truncated HDR literal");
                        dst[(x + i) * 4 + channel] = (byte)v;
                    }
                    x += count;
                }
            }
        }
    }

    static string? ReadLine(Stream s)
    {
        var bytes = new List<byte>(64);
        while (true)
        {
            int b = s.ReadByte();
            if (b < 0) return bytes.Count > 0 ? Encoding.ASCII.GetString(bytes.ToArray()) : null;
            if (b == '\n') return Encoding.ASCII.GetString(bytes.ToArray());
            bytes.Add((byte)b);
        }
    }

    static void ReadExact(Stream s, byte[] buf, int offset, int count)
    {
        while (count > 0)
        {
            int n = s.Read(buf, offset, count);
            if (n <= 0) throw new EndOfStreamException("Truncated HDR pixels");
            offset += n; count -= n;
        }
    }
}
