using Silk.NET.Core.Native;
using Silk.NET.Vulkan;
using Silk.NET.Vulkan.Extensions.KHR;
using Silk.NET.Vulkan.Extensions.EXT;

namespace CadThingo.GraphicsPipeline;

public class VulkanPipeline
{
    Vk? vk = Globals.vk;
    VulkanContext ctx;
    public VulkanPipeline(VulkanContext context)
    {
        ctx = context;
    }
    public unsafe void CreateGraphicsPipeline()
    {
        var fragShaderCode = File.ReadAllBytes("C:\\Users\\jamie\\RiderProjects\\CadThingo\\CadThingo\\Assets\\Shaders\\frag.spv");
        var vertShaderCode = File.ReadAllBytes("C:\\Users\\jamie\\RiderProjects\\CadThingo\\CadThingo\\Assets\\Shaders\\vert.spv");
        
        var vertShaderModule = CreateShaderModule(vertShaderCode);
        var fragShaderModule = CreateShaderModule(fragShaderCode);

        PipelineShaderStageCreateInfo vertShaderStageInfo = new()
        {
            SType = StructureType.PipelineShaderStageCreateInfo,
            Stage = ShaderStageFlags.VertexBit,
            Module = vertShaderModule,
            PName = (byte*)SilkMarshal.StringToPtr("main")
        };
        PipelineShaderStageCreateInfo fragShaderStageInfo = new()
        {
            SType = StructureType.PipelineShaderStageCreateInfo,
            Stage = ShaderStageFlags.FragmentBit,
            Module = fragShaderModule,
            PName = (byte*)SilkMarshal.StringToPtr("main")
        };
        var shaderStages = stackalloc PipelineShaderStageCreateInfo[]
        {
            vertShaderStageInfo,
            fragShaderStageInfo
        };

        PipelineVertexInputStateCreateInfo vertexInputStateInfo = new()
        {
            SType = StructureType.PipelineVertexInputStateCreateInfo,
            VertexBindingDescriptionCount = 0,
            VertexAttributeDescriptionCount = 0
        };

        PipelineInputAssemblyStateCreateInfo inputAssemblyStateInfo = new()
        {
            SType = StructureType.PipelineInputAssemblyStateCreateInfo,
            Topology = PrimitiveTopology.TriangleList,
            PrimitiveRestartEnable = false
        };
        Viewport viewport = new()
        {
            X = 0,
            Y = 0,
            Width = ctx.SwapChainExtent.Width,
            Height = ctx.SwapChainExtent.Height,
            MinDepth = 0,
            MaxDepth = 1
        };

        Rect2D scissor = new()
        {
            Offset = new Offset2D(0, 0),
            Extent = ctx.SwapChainExtent
        };

        PipelineViewportStateCreateInfo viewportStateInfo = new()
        {
            SType = StructureType.PipelineViewportStateCreateInfo,
            ViewportCount = 1,
            PViewports = &viewport,
            ScissorCount = 1,
            PScissors = &scissor
        };

        PipelineRasterizationStateCreateInfo rasterizationStateInfo = new()
        {
            SType = StructureType.PipelineRasterizationStateCreateInfo,
            DepthClampEnable = false,
            RasterizerDiscardEnable = false,
            PolygonMode = PolygonMode.Fill,
            LineWidth = 1,
            CullMode = CullModeFlags.BackBit,
            FrontFace = FrontFace.Clockwise,
            DepthBiasEnable = false,
        };
        PipelineMultisampleStateCreateInfo multisampleStateInfo = new()
        {
            SType = StructureType.PipelineMultisampleStateCreateInfo,
            SampleShadingEnable = false,
            RasterizationSamples = SampleCountFlags.Count1Bit
        };

        PipelineColorBlendAttachmentState colorBlendAttachmentState = new()
        {
            ColorWriteMask = ColorComponentFlags.RBit | ColorComponentFlags.GBit | ColorComponentFlags.BBit |
                             ColorComponentFlags.ABit,
            BlendEnable = false
        };

        PipelineColorBlendStateCreateInfo colorBlending = new()
        {
            SType = StructureType.PipelineColorBlendStateCreateInfo,
            LogicOpEnable = false,
            LogicOp = LogicOp.Copy,
            AttachmentCount = 1,
            PAttachments = &colorBlendAttachmentState
        };
        
        colorBlending.BlendConstants[0] = 0.0f;
        colorBlending.BlendConstants[1] = 0.0f;
        colorBlending.BlendConstants[2] = 0.0f;
        colorBlending.BlendConstants[3] = 0.0f;

        PipelineLayoutCreateInfo pipelineLayoutInfo = new()
        {
            SType = StructureType.PipelineLayoutCreateInfo,
            SetLayoutCount = 0,
            PushConstantRangeCount = 0
        };
        
        if(vk!.CreatePipelineLayout(ctx.Device, &pipelineLayoutInfo, null, out ctx.PipelineLayout) != Result.Success)
        {
            throw new Exception("Failed to create pipeline layout");
        }

        GraphicsPipelineCreateInfo pipelineInfo = new()
        {
            SType = StructureType.GraphicsPipelineCreateInfo,
            StageCount = 2,
            PStages = shaderStages,
            PVertexInputState = &vertexInputStateInfo,
            PInputAssemblyState = &inputAssemblyStateInfo,
            PViewportState = &viewportStateInfo,
            PRasterizationState = &rasterizationStateInfo,
            PMultisampleState = &multisampleStateInfo,
            PColorBlendState = &colorBlending,
            Layout = ctx.PipelineLayout,
            RenderPass = ctx.RenderPass,
            Subpass = 0,
            BasePipelineHandle = default,
        };

        if (vk!.CreateGraphicsPipelines(ctx.Device, default, 1, &pipelineInfo, null, out ctx.Pipeline) != Result.Success)
        {
            throw new Exception("Failed to create graphics pipeline");
        }
        Console.WriteLine("Created graphics pipeline successfully");
        
        vk!.DestroyShaderModule(ctx.Device, vertShaderModule, null);
        vk!.DestroyShaderModule(ctx.Device, fragShaderModule, null);
        
        SilkMarshal.Free((nint)vertShaderStageInfo.PName);
        SilkMarshal.Free((nint)fragShaderStageInfo.PName);
    }
    public unsafe void CreateRenderPass()
    {
        AttachmentDescription attachment = new()
        {
            Format = ctx.SwapChainImageFormat,
            Samples = SampleCountFlags.Count1Bit,

            LoadOp = AttachmentLoadOp.Clear,
            StoreOp = AttachmentStoreOp.Store,

            StencilLoadOp = AttachmentLoadOp.DontCare,
            StencilStoreOp = AttachmentStoreOp.DontCare,

            InitialLayout = ImageLayout.Undefined,
            FinalLayout = ImageLayout.PresentSrcKhr,
        };

        AttachmentReference colorAttachmentRef = new()
        {
            Attachment = 0,
            Layout = ImageLayout.ColorAttachmentOptimal
        };
        SubpassDescription subpass = new()
        {
            PipelineBindPoint = PipelineBindPoint.Graphics,
            ColorAttachmentCount = 1,
            PColorAttachments = &colorAttachmentRef
        };

        RenderPassCreateInfo renderPassInfo = new()
        {
            SType = StructureType.RenderPassCreateInfo,
            AttachmentCount = 1,
            PAttachments = &attachment,
            SubpassCount = 1,
            PSubpasses = &subpass
        };
        if (vk!.CreateRenderPass(ctx.Device, &renderPassInfo, null, out ctx.RenderPass) != Result.Success)
        {
            throw new Exception("Failed to create render pass");
        }
        Console.WriteLine("Created render pass");
    }
    private unsafe ShaderModule CreateShaderModule(byte[] fragShaderCode)
    {
        ShaderModuleCreateInfo createInfo = new()
        {
            SType = StructureType.ShaderModuleCreateInfo,
            CodeSize = (nuint)fragShaderCode.Length,
        };
        ShaderModule ShaderModule;

        fixed (byte* pCode = fragShaderCode)
        {
            createInfo.PCode = (uint*)pCode;

            if (vk!.CreateShaderModule(ctx.Device, &createInfo, null, &ShaderModule) != Result.Success)
            {
                throw new Exception("Failed to create shader module");
            }
        }
        return ShaderModule;
    }

    
}