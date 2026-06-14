// BcEncode -- GPU block-compression encoder for material textures.
//
// One thread compresses one 4x4 texel block of a single source mip into a BC block,
// writing the raw block bytes into a tightly-packed output buffer (the host then
// vkCmdCopyBufferToImage's each mip into the BC image). Run at load (BcEncoder.cs),
// not per frame. Source is read as raw UNORM (no sRGB conversion on fetch) so the
// destination BC image can carry the sRGB-ness for the final sample without a double
// decode. Endpoint selection is the standard min/max bounding box -- fast, no iterative
// refinement; quality is fine for albedo/normal/data textures at 4x-8x.
//
// Modes (push constant): 0=BC1 (RGB, 8B/block), 1=BC3 (RGBA = BC4 alpha + BC1, 16B),
// 2=BC5 (RG = two BC4, 16B), 3=BC4 (R, 8B). BC3/BC5 reuse the BC1 + BC4 primitives.

[[vk::binding(0, 0)]] Texture2D<float4>   src;
[[vk::binding(1, 0)]] RWByteAddressBuffer  dst;

struct EncPush {
    uint mode;          // 0 BC1, 1 BC3, 2 BC5, 3 BC4
    uint srcMip;        // mip level of `src` to read
    uint mipW;          // this mip's width  (texels)
    uint mipH;          // this mip's height (texels)
    uint blockOffset;   // byte offset into dst where this mip's blocks start
};
[[vk::push_constant]] EncPush pc;

static const uint MODE_BC1 = 0u, MODE_BC3 = 1u, MODE_BC5 = 2u, MODE_BC4 = 3u;

// ---- RGB565 endpoint pack/unpack --------------------------------------------
uint pack565(float3 c) {
    uint r = uint(round(saturate(c.r) * 31.0));
    uint g = uint(round(saturate(c.g) * 63.0));
    uint b = uint(round(saturate(c.b) * 31.0));
    return (r << 11) | (g << 5) | b;
}
float3 unpack565(uint v) {
    return float3(float((v >> 11) & 31u) / 31.0,
                  float((v >> 5)  & 63u) / 63.0,
                  float( v        & 31u) / 31.0);
}

// ---- BC1 colour block (4-colour opaque mode) --------------------------------
// Endpoints from the per-channel bounding box, kept ep0 >= ep1 so the decoder uses
// the 4-colour (opaque) interpolation. Indices map 0->ep0, 1->ep1, 2->2/3, 3->1/3.
uint2 encodeBC1(float3 texels[16]) {
    float3 lo = texels[0], hi = texels[0];
    for (uint i = 1u; i < 16u; ++i) { lo = min(lo, texels[i]); hi = max(hi, texels[i]); }

    uint ep0 = pack565(hi);
    uint ep1 = pack565(lo);
    if (ep0 < ep1) { uint t = ep0; ep0 = ep1; ep1 = t; }

    float3 p0 = unpack565(ep0);
    float3 p1 = unpack565(ep1);
    float3 p2 = (2.0 * p0 + p1) / 3.0;
    float3 p3 = (p0 + 2.0 * p1) / 3.0;

    uint indices = 0u;
    for (uint i = 0u; i < 16u; ++i) {
        float3 c = texels[i];
        float d0 = dot(c - p0, c - p0);
        float d1 = dot(c - p1, c - p1);
        float d2 = dot(c - p2, c - p2);
        float d3 = dot(c - p3, c - p3);
        uint  idx = 0u; float best = d0;
        if (d1 < best) { best = d1; idx = 1u; }
        if (d2 < best) { best = d2; idx = 2u; }
        if (d3 < best) { best = d3; idx = 3u; }
        indices |= idx << (2u * i);
    }
    return uint2(ep0 | (ep1 << 16), indices);   // word0: ep0|ep1, word1: 2-bit indices
}

// ---- BC4 single-channel block (8-value interpolation mode) ------------------
// 8-bit endpoints kept a0 >= a1 (8-interpolated-values mode). Index field is 16 x
// 3 bits starting at bit 16 of the 64-bit block; the straddling texel is split
// across the two output words.
uint2 encodeBC4(float vals[16]) {
    float lo = vals[0], hi = vals[0];
    for (uint i = 1u; i < 16u; ++i) { lo = min(lo, vals[i]); hi = max(hi, vals[i]); }

    uint a0 = uint(round(saturate(hi) * 255.0));
    uint a1 = uint(round(saturate(lo) * 255.0));
    if (a0 < a1) { uint t = a0; a0 = a1; a1 = t; }

    float pal[8];
    pal[0] = float(a0);
    pal[1] = float(a1);
    float fa0 = float(a0), fa1 = float(a1);
    [unroll] for (uint k = 1u; k < 7u; ++k)
        pal[k + 1u] = ((7.0 - float(k)) * fa0 + float(k) * fa1) / 7.0;

    uint lo32 = a0 | (a1 << 8);
    uint hi32 = 0u;
    for (uint i = 0u; i < 16u; ++i) {
        float v   = saturate(vals[i]) * 255.0;
        uint  idx = 0u; float best = abs(v - pal[0]);
        [unroll] for (uint k = 1u; k < 8u; ++k) {
            float d = abs(v - pal[k]);
            if (d < best) { best = d; idx = k; }
        }
        uint bitpos = 16u + 3u * i;
        if (bitpos < 32u) {
            lo32 |= idx << bitpos;
            if (bitpos + 3u > 32u) hi32 |= idx >> (32u - bitpos);   // straddle
        } else {
            hi32 |= idx << (bitpos - 32u);
        }
    }
    return uint2(lo32, hi32);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint blocksX = (pc.mipW + 3u) / 4u;
    uint blocksY = (pc.mipH + 3u) / 4u;
    if (tid.x >= blocksX || tid.y >= blocksY) return;

    // Gather the 4x4 texel block, clamping reads to the mip edge for non-multiple-of-4 mips.
    float4 t[16];
    for (uint j = 0u; j < 4u; ++j)
        for (uint i = 0u; i < 4u; ++i) {
            uint x = min(tid.x * 4u + i, pc.mipW - 1u);
            uint y = min(tid.y * 4u + j, pc.mipH - 1u);
            t[j * 4u + i] = src.Load(int3(int(x), int(y), int(pc.srcMip)));
        }

    uint blockIndex = tid.y * blocksX + tid.x;

    if (pc.mode == MODE_BC1) {
        float3 rgb[16];
        for (uint i = 0u; i < 16u; ++i) rgb[i] = t[i].rgb;
        dst.Store2(pc.blockOffset + blockIndex * 8u, encodeBC1(rgb));
    } else if (pc.mode == MODE_BC4) {
        float r[16];
        for (uint i = 0u; i < 16u; ++i) r[i] = t[i].r;
        dst.Store2(pc.blockOffset + blockIndex * 8u, encodeBC4(r));
    } else if (pc.mode == MODE_BC3) {
        float3 rgb[16]; float a[16];
        for (uint i = 0u; i < 16u; ++i) { rgb[i] = t[i].rgb; a[i] = t[i].a; }
        uint2 alpha = encodeBC4(a);     // BC3 layout: alpha block first, then colour block
        uint2 color = encodeBC1(rgb);
        dst.Store4(pc.blockOffset + blockIndex * 16u, uint4(alpha, color));
    } else { // MODE_BC5
        float r[16]; float g[16];
        for (uint i = 0u; i < 16u; ++i) { r[i] = t[i].r; g[i] = t[i].g; }
        uint2 rb = encodeBC4(r);
        uint2 gb = encodeBC4(g);
        dst.Store4(pc.blockOffset + blockIndex * 16u, uint4(rb, gb));
    }
}