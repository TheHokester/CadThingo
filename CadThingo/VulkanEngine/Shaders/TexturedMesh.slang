//combined vertex + fragment shader


struct VertexInput{
    [[vk::location(0)]] float3 position;
    [[vk::location(1)]] float3 normal;
    [[vk::location(2)]] float2 uv;
    [[vk::location(3)]] float4 tangent;
    
    
    //per instance data
    [[vk::location(4)]] row_major float4x4 instanceModel;//model matrix;
    //3 columns of 3x4 matrix, xyz used w unused
    [[vk::location(8)]] float4 instanceNormal0;//normal mat col1
    [[vk::location(9)]] float4 instanceNormal1;//normal mat col2
    [[vk::location(10)]] float4 instanceNormal2;//normal mat col3

}
struct VertexOutput{
    float4 Position : SV_POSITION;
    float3 WorldPos;
    float3 Normal: NORMAL;
    float2 TexCoord: TEXCOORD0;
    float4 Tangent: TANGENT;
   
}
struct UniformBufferObject{
    row_major float4x4 model;
    row_major float4x4 view;
    row_major float4x4 proj;

}

//uniform buffer object
[[vk::binding(0,0)]] ConstantBuffer<UniformBufferObject> ubo;
[[vk::binding(1,0)]] Sampler2D texSampler;
   
//vertex shader entry
[[shader("vertex")]]
VertexOutput VSMain(VertexInput input){
    VertexOutput output;
    
    //use instance matrices directly(cpu uploads these) 
    float4x4 model = input.instanceModel;
    
    //transform position to world space
    float4 worldPos = mul(ubo.model, mul(model, float4(input.position,1.0)));
    
    //final clipspace position
    output.Position = mul(ubo.proj, mul(ubo.view, worldPos));
    
    //pass world position and transformed normal to frag shader
    float3x3 model3x3 = (float3x3)ubo.model;
    output.WorldPos = worldPos.xyz;
    
    float3 instanceNormal = input.instanceNormal0.xyz * input.normal.x +
                            input.instanceNormal1.xyz * input.normal.y +
                            input.instanceNormal2.xyz * input.normal.z;
      
    output.Normal = normalize(mul(model3x3,instanceNormal));
    output.TexCoord = input.uv;
    output.Tangent = input.tangent;
    return output;
   
}
//fragment shader entry
[[shader("fragment")]]
float4 FSMain(VertexOutput input) : SV_TARGET{
    //sample texture
    float2 uv = float2(input.TexCoord.x,1.0-input.TexCoord.y);
    float4 color = texSampler.Sample(uv);
    
    return color;
}