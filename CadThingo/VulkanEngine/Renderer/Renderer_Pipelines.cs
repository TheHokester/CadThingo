using CadThingo.GraphicsPipeline;
using Silk.NET.Core.Native;
using Silk.NET.Vulkan;

namespace CadThingo.VulkanEngine.Renderer;

public unsafe partial class Renderer
{
    private void CreateDescriptorSetLayout()
    {
        DescriptorSetLayoutBinding uboLayoutBinding = new()
        {
            Binding = 0,
            DescriptorCount = 1,
            DescriptorType = DescriptorType.UniformBuffer,

            StageFlags = ShaderStageFlags.VertexBit, 
            PImmutableSamplers = null // Optional
        };
        DescriptorSetLayoutBinding samplerLayoutBinding = new()
        {
            Binding = 1,
            DescriptorCount = 1,
            DescriptorType = DescriptorType.CombinedImageSampler,
            StageFlags = ShaderStageFlags.FragmentBit,
            PImmutableSamplers = null
        };
        
        var bindings = new [] {uboLayoutBinding, samplerLayoutBinding};

        DescriptorSetLayoutBindingFlagsCreateInfo flagsCreateInfo = new(){SType = StructureType.DescriptorSetLayoutBindingFlagsCreateInfo};
        var flags = stackalloc DescriptorBindingFlags[2];
        fixed (DescriptorSetLayoutBinding* bindingsPtr = bindings)
        {
            if (descriptorIndexEnabled)
            {
                flags[0] = DescriptorBindingFlags.UpdateAfterBindBit | DescriptorBindingFlags.UpdateUnusedWhilePendingBit;
                flags[1] = DescriptorBindingFlags.UpdateAfterBindBit | DescriptorBindingFlags.UpdateUnusedWhilePendingBit;
                flagsCreateInfo.BindingCount = 2;
                flagsCreateInfo.PBindingFlags = flags;
            }

            DescriptorSetLayoutCreateInfo layoutInfo = new()
            {
                BindingCount = (uint)(bindings).Length,
                PBindings = bindingsPtr,
                SType = StructureType.DescriptorSetLayoutCreateInfo,
            };
            if (descriptorIndexEnabled)
            {
                layoutInfo.Flags |= DescriptorSetLayoutCreateFlags.UpdateAfterBindPoolBit;
                layoutInfo.PNext = &flagsCreateInfo;
            }

            if (vk!.CreateDescriptorSetLayout(device, &layoutInfo, null, out descriptorSetLayout) != Result.Success)
                throw new Exception("Failed to create descriptor set layout");
        }
            
    }
    
    private void CreateGraphicsPipeline()
    {
        var shaderByteCode = File.ReadAllBytes("C:\\Users\\jamie\\RiderProjects\\CadThingo\\CadThingo\\Assets\\Shaders\\TexturedMesh.spv");//worry about filepath and shader compilation later
        
        ShaderModule shaderModule= CreateShaderModule(shaderByteCode);
        
        //create shader stage info
        PipelineShaderStageCreateInfo vertShaderStageInfo = new()
        {
            SType = StructureType.PipelineShaderStageCreateInfo,
            Stage = ShaderStageFlags.VertexBit,
            Module = shaderModule,
            PName = (byte*)SilkMarshal.StringToPtr("VSMain")
        };
        
        PipelineShaderStageCreateInfo fragShaderStageInfo = new()
        {
            SType = StructureType.PipelineShaderStageCreateInfo,
            Stage = ShaderStageFlags.FragmentBit,
            Module = shaderModule,
            PName = (byte*)SilkMarshal.StringToPtr("FSMain")
        };
        PipelineShaderStageCreateInfo fragGlassStageInfo = new()
        {
    
            Stage = ShaderStageFlags.FragmentBit,
            Module = shaderModule,
            PName = (byte*)SilkMarshal.StringToPtr("GlassFSMain")
        };
        var shaderStages = stackalloc PipelineShaderStageCreateInfo[]
        {
            vertShaderStageInfo,
            fragShaderStageInfo,
        };
        
        
        //create vertex input with instancing support
        var vertexBindingDescriptions = Vertex.GetBindingDescription();
        var instanceBindingDescriptions = InstanceData.GetBindingDescription();
        var vertexInputBindingDescriptions = new VertexInputBindingDescription[]
        {
            vertexBindingDescriptions,
            instanceBindingDescriptions
        };
        
        var vertexInputAttributeDescriptions = Vertex.GetAttributeDescriptions();
        var instanceInputAttributeDescriptions = InstanceData.GetAttributeDescriptions();
        
        //combine all attribute descriptions
        var allAttributeDescriptions = new VertexInputAttributeDescription[
            vertexInputAttributeDescriptions.Length + instanceInputAttributeDescriptions.Length];
        //copy vertex attributes into the array
        vertexInputAttributeDescriptions.CopyTo(allAttributeDescriptions, 0);
        instanceInputAttributeDescriptions.CopyTo(allAttributeDescriptions, vertexInputAttributeDescriptions.Length);
        
        
        //material index attribute loc11 is unused
        PipelineVertexInputStateCreateInfo* vertexInputStateInfo = stackalloc PipelineVertexInputStateCreateInfo[1];
        fixed(VertexInputAttributeDescription* pAttributesDesc = allAttributeDescriptions)
        fixed (VertexInputBindingDescription* pBindingDesc = vertexInputBindingDescriptions)
        { 
            *vertexInputStateInfo = new()
            {
                SType = StructureType.PipelineVertexInputStateCreateInfo,
                VertexBindingDescriptionCount = (uint)vertexInputBindingDescriptions.Length,
                PVertexBindingDescriptions = pBindingDesc,
                VertexAttributeDescriptionCount = (uint)allAttributeDescriptions.Length,
                PVertexAttributeDescriptions = pAttributesDesc
            };
        }
        //create input assembly info
        PipelineInputAssemblyStateCreateInfo inputAssemblyStateInfo = new()
        {
            SType = StructureType.PipelineInputAssemblyStateCreateInfo,
            Topology = PrimitiveTopology.TriangleList,
            PrimitiveRestartEnable = false
        };
        //create viewport state info
        PipelineViewportStateCreateInfo viewportStateInfo = new()
        {
            SType = StructureType.PipelineViewportStateCreateInfo,
            ScissorCount = 1,
            PScissors = null,
            ViewportCount = 1,
        };
        //create rasterization state info
        PipelineRasterizationStateCreateInfo rasterizer = new()
        {
            SType = StructureType.PipelineRasterizationStateCreateInfo,
            CullMode = CullModeFlags.None,
            FrontFace = FrontFace.CounterClockwise,
            DepthClampEnable = false,
            DepthBiasEnable = false,
            LineWidth = 1.0f,
            PolygonMode = PolygonMode.Fill,
            RasterizerDiscardEnable = false
        };
        //create MS state info
        PipelineMultisampleStateCreateInfo multisampleStateCreateInfo = new()
        {
            SType = StructureType.PipelineMultisampleStateCreateInfo,
            RasterizationSamples = SampleCountFlags.Count1Bit,
            SampleShadingEnable = false
        };
        //create depth stencil state info
        PipelineDepthStencilStateCreateInfo depthStencilStateCreateInfo = new()
        {
            SType = StructureType.PipelineDepthStencilStateCreateInfo,
            DepthTestEnable = true,
            DepthWriteEnable = true,
            DepthCompareOp = CompareOp.LessOrEqual,
            DepthBoundsTestEnable = false,
            StencilTestEnable = false,
            MinDepthBounds = 0.0f,
            MaxDepthBounds = 1.0f
        };
        //create color blend attachment
        PipelineColorBlendAttachmentState colorBlending = new()
        {
            
            BlendEnable = false,
            ColorWriteMask = ColorComponentFlags.RBit | ColorComponentFlags.GBit | ColorComponentFlags.BBit |
                             ColorComponentFlags.ABit
        };
        //create color blend state info
        PipelineColorBlendStateCreateInfo colorBlendingCreateInfo = new()
        {
            SType = StructureType.PipelineColorBlendStateCreateInfo,
            LogicOpEnable = false,
            LogicOp = LogicOp.Copy,
            AttachmentCount = 1,
            PAttachments = &colorBlending
        };
        //create dynamic state info
        var dynamicStateCount = 2;
        var dynamicStates = stackalloc DynamicState[2]
        {
            DynamicState.Scissor,
            DynamicState.Viewport
        };
        PipelineDynamicStateCreateInfo dynamicStateCreateInfo = new()
        {
            SType = StructureType.PipelineDynamicStateCreateInfo,
            DynamicStateCount = 2,
            PDynamicStates = dynamicStates
        };
        //create pipeline layout
        fixed (DescriptorSetLayout* pLayout = &descriptorSetLayout)
        {
            PipelineLayoutCreateInfo pipelineLayoutInfo = new()
            {
                SType = StructureType.PipelineLayoutCreateInfo,
                SetLayoutCount = 1,
                PSetLayouts = pLayout,
                PushConstantRangeCount = 0,
                PPushConstantRanges = null
            };
            if(vk!.CreatePipelineLayout(device, &pipelineLayoutInfo, null, out pipelineLayout) != Result.Success)
                throw new Exception("Failed to create pipeline layout");
        }

        Format depthFormat = FindDepthFormat();
        fixed (Format* pswapChainImageFormat = &swapChainImageFormat)
            mainPipelineRenderingCreateInfo = new PipelineRenderingCreateInfo()
            {
                SType = StructureType.PipelineRenderingCreateInfo,
                ColorAttachmentCount = 1,
                PColorAttachmentFormats = pswapChainImageFormat,
                DepthAttachmentFormat = depthFormat,
                StencilAttachmentFormat = Format.Undefined
            };
        
        //create the graphics pipeline
        PipelineRasterizationStateCreateInfo rasterizerBack = rasterizer;
        //display backface culling for opaque PBR to avoid vanishing geometry when
        //instance or model transforms flip winding
        rasterizerBack.CullMode = CullModeFlags.None;
        fixed (PipelineRenderingCreateInfo* pPipeLineRenderinfo = &mainPipelineRenderingCreateInfo)
        {
            GraphicsPipelineCreateInfo pipelineInfo = new()
            {
                SType = StructureType.GraphicsPipelineCreateInfo,
                PNext = pPipeLineRenderinfo,
                Flags = new PipelineCreateFlags() { },
                StageCount = 2,
                PStages = shaderStages,
                PVertexInputState = vertexInputStateInfo,
                PInputAssemblyState = &inputAssemblyStateInfo,
                PViewportState = &viewportStateInfo,
                PRasterizationState = &rasterizerBack,
                PMultisampleState = &multisampleStateCreateInfo,
                PDepthStencilState = &depthStencilStateCreateInfo,
                PColorBlendState = &colorBlendingCreateInfo,
                PDynamicState = &dynamicStateCreateInfo,
                Layout = pipelineLayout,
                RenderPass = default,
                Subpass = 0,
                BasePipelineHandle = default,
                BasePipelineIndex = -1,
            };
            if (vk!.CreateGraphicsPipelines(device, default, 1, &pipelineInfo, null, out graphicsPipeline) !=
                Result.Success)
            {
                throw new Exception("Failed to create graphics pipeline");
            }
        }
        
        
    }
}