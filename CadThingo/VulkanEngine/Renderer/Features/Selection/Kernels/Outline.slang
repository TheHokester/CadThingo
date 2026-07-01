// Selection-outline overlay. Fullscreen pass run on FinalColor (LoadOp.Load)
// after the scene has been composited by whichever render mode is active. Reads
// the selection mask (SelectionMask.slang) and draws an outer outline: a pixel
// is an edge if it's *outside* the mask but within `thickness` pixels of a
// covered pixel. Non-edge pixels discard, leaving the underlying image intact.
//
// Mask is fetched with .Load (texelFetch) so no sampler / filtering is needed —
// the mask is binary coverage and the pixel grid lines up 1:1 with FinalColor.

struct VSOutput {
    float4 Position : SV_POSITION;
};

struct OutlineParams {
    float4 color;       // rgb = outline color (a unused; outline is opaque)
    int2   screenSize;  // render-target pixels
    int    thickness;   // outline radius in pixels
    int    _pad;
};

[[vk::binding(0, 0)]] Texture2D<float> mask;
[[vk::push_constant]] OutlineParams params;

[shader("vertex")]
VSOutput VSMain(uint vertexID : SV_VertexID) {
    float2 positions[3] = {
        float2(-1.0, -1.0),
        float2( 3.0, -1.0),
        float2(-1.0,  3.0),
    };
    VSOutput o;
    o.Position = float4(positions[vertexID], 0.0, 1.0);
    return o;
}

[shader("fragment")]
float4 PSMain(VSOutput input) : SV_TARGET {
    int2 px = int2(input.Position.xy);   // top-left-origin pixel, matches the mask grid

    float center = mask.Load(int3(px, 0));

    // Outer outline only: covered pixels keep the rendered surface; the ring is
    // drawn on the empty pixels bordering the silhouette.
    if (center > 0.5) discard;

    int t = params.thickness;
    float maxN = 0.0;
    for (int dy = -t; dy <= t; ++dy) {
        for (int dx = -t; dx <= t; ++dx) {
            int2 s = px + int2(dx, dy);
            if (s.x < 0 || s.y < 0 || s.x >= params.screenSize.x || s.y >= params.screenSize.y)
                continue;
            maxN = max(maxN, mask.Load(int3(s, 0)));
        }
    }

    if (maxN < 0.5) discard;   // not adjacent to the selection — leave pixel untouched
    return float4(params.color.rgb, 1.0);
}