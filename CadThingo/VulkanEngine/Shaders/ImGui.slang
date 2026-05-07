
//DUMMY SHADER, to be filled in later, stubbed out for now so that a pipeline can be built around this layout


struct VSInput {
    float2 position : POSITION;
    float2 uv : TEXCOORD0;
    float4 color : COLOR0;
}

struct VSOutput {
    float4 screenPos : SV_POSITION;
    float2 uv : TEXCOORD0;
    float4 color : COLOR0;
}


struct PushConstants
{
    float2 scale;
    float2 translation;
};

[[vk::binding(0, 0)]] Sampler2D fontTex;
[[vk::push_constant]] PushConstants drawData;

[[shader("vertex")]]
VSOutput VSMain(VSInput input){
    VSOutput output;
    //transform position
    output.screenPos = float4(input.position.xy * drawData.scale + drawData.translation,0,1);
    output.uv = input.uv;
    output.color = input.color;
    return output;
}
[[shader("fragment")]]
float4 PSMain(VSOutput input) : SV_Target0{
    //sample font tex and multiply by color
    float4 color = input.color * fontTex.Sample(input.uv);
    return color;
}