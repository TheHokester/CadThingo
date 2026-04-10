using System.Runtime.InteropServices.ComTypes;
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
        var bindingDescription = VkVertex.GetBindingDescription();
        var attributeDescriptions = VkVertex.GetAttributeDescriptions();
        fixed(DescriptorSetLayout* setLayoutPtr = &ctx.DescriptorSetLayout)
        fixed (VertexInputAttributeDescription* pAttributesDesc = attributeDescriptions)
        {
            
            PipelineVertexInputStateCreateInfo vertexInputStateInfo = new()
            {
                SType = StructureType.PipelineVertexInputStateCreateInfo,
                VertexBindingDescriptionCount = 1,
                PVertexBindingDescriptions = &bindingDescription,
                VertexAttributeDescriptionCount = (uint)attributeDescriptions.Length,
                PVertexAttributeDescriptions = pAttributesDesc
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
                FrontFace = FrontFace.CounterClockwise,
                DepthBiasEnable = false,
            };
            PipelineMultisampleStateCreateInfo multisampleStateInfo = new()
            {
                SType = StructureType.PipelineMultisampleStateCreateInfo,
                SampleShadingEnable = false,
                RasterizationSamples = ctx.MsaaSamples
            };
            PipelineDepthStencilStateCreateInfo depthStencilState = new()
            {
                SType = StructureType.PipelineDepthStencilStateCreateInfo,
                DepthTestEnable = true,
                DepthWriteEnable = true,
                
                DepthCompareOp = CompareOp.Less,
               
                DepthBoundsTestEnable = false,
                MinDepthBounds = 0,
                MaxDepthBounds = 1,

                StencilTestEnable = false,
                Front = new StencilOpState(),
                Back = new StencilOpState(),
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
                SetLayoutCount = 1,
                PSetLayouts = setLayoutPtr,
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
                PDepthStencilState = &depthStencilState,
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
        }

        Console.WriteLine("Created graphics pipeline successfully");
        
        vk!.DestroyShaderModule(ctx.Device, vertShaderModule, null);
        vk!.DestroyShaderModule(ctx.Device, fragShaderModule, null);
        
        SilkMarshal.Free((nint)vertShaderStageInfo.PName);
        SilkMarshal.Free((nint)fragShaderStageInfo.PName);
    }
    public unsafe void CreateRenderPass()
    {
        AttachmentDescription colorAttachment = new()
        {
            Format = ctx.SwapChainImageFormat,
            Samples = ctx.MsaaSamples,

            LoadOp = AttachmentLoadOp.Clear,
            StoreOp = AttachmentStoreOp.Store,
            
            StencilLoadOp = AttachmentLoadOp.DontCare,
            StencilStoreOp = AttachmentStoreOp.DontCare,

            InitialLayout = ImageLayout.Undefined,
            FinalLayout = ImageLayout.ColorAttachmentOptimal,
        };
        AttachmentReference colorAttachmentRef = new()
        {
            Attachment = 0,
            Layout = ImageLayout.ColorAttachmentOptimal
        };
        
        AttachmentDescription depthAttachment = new()
        {
            Format = VulkanRenderer._DepthResources.FindDepthFormat(),
            Samples = ctx.MsaaSamples,
            LoadOp = AttachmentLoadOp.Clear,
            StoreOp = AttachmentStoreOp.DontCare,
            StencilLoadOp = AttachmentLoadOp.DontCare,
            StencilStoreOp = AttachmentStoreOp.DontCare,
            InitialLayout = ImageLayout.Undefined,
            FinalLayout = ImageLayout.DepthStencilAttachmentOptimal,
        };
        AttachmentReference depthAttachmentRef = new()
        {
            Attachment = 1,
            Layout = ImageLayout.DepthStencilAttachmentOptimal,
        };
        
        AttachmentDescription colorAttachmentResolve = new()
        {
            Format = ctx.SwapChainImageFormat,
            Samples = ctx.MsaaSamples,

            LoadOp = AttachmentLoadOp.DontCare,
            StoreOp = AttachmentStoreOp.Store,

            StencilLoadOp = AttachmentLoadOp.DontCare,
            StencilStoreOp = AttachmentStoreOp.DontCare,

            InitialLayout = ImageLayout.Undefined,
            FinalLayout = ImageLayout.PresentSrcKhr,
        };
        AttachmentReference colorAttachmentResolveRef = new()
        {
            Attachment = 2,
            Layout = ImageLayout.ColorAttachmentOptimal,
        };
        
        SubpassDescription subpass = new()
        {
            PipelineBindPoint = PipelineBindPoint.Graphics,
            ColorAttachmentCount = 1,
            PColorAttachments = &colorAttachmentRef,
            PDepthStencilAttachment = &depthAttachmentRef,
            PResolveAttachments =  &colorAttachmentResolveRef,
        };
        SubpassDependency dependency = new()
        {
            SrcSubpass = Vk.SubpassExternal,
            DstSubpass = 0,
            SrcStageMask = PipelineStageFlags.ColorAttachmentOutputBit | PipelineStageFlags.EarlyFragmentTestsBit,
            SrcAccessMask = AccessFlags.ColorAttachmentWriteBit | AccessFlags.DepthStencilAttachmentWriteBit,
            DstStageMask = PipelineStageFlags.ColorAttachmentOutputBit | PipelineStageFlags.EarlyFragmentTestsBit,
            DstAccessMask = AccessFlags.ColorAttachmentReadBit | AccessFlags.DepthStencilAttachmentWriteBit,
            
        };
        
        var attachments = new[]{colorAttachment, depthAttachment, colorAttachmentResolve};
        fixed (AttachmentDescription* attachmentsPtr = attachments)
        {
            RenderPassCreateInfo renderPassInfo = new()
            {
                SType = StructureType.RenderPassCreateInfo,
                AttachmentCount = (uint)attachments.Length,
                PAttachments = attachmentsPtr,
                SubpassCount = 1,
                PSubpasses = &subpass,
                DependencyCount = 1,
                PDependencies = &dependency,
            };
            if (vk!.CreateRenderPass(ctx.Device, &renderPassInfo, null, out ctx.RenderPass) != Result.Success)
            {
                throw new Exception("Failed to create render pass");
            }
        }
        Console.WriteLine("Created render pass");
    }

    public unsafe void CreateDescriptorSetLayout()
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
        fixed (DescriptorSetLayoutBinding* bindingsPtr = bindings)
        {
            DescriptorSetLayoutCreateInfo layoutInfo = new()
            {
                SType = StructureType.DescriptorSetLayoutCreateInfo,
                BindingCount = (uint)(bindings).Length,
                PBindings = bindingsPtr
            };
            if (vk!.CreateDescriptorSetLayout(ctx.Device, &layoutInfo, null, out ctx.DescriptorSetLayout) != Result.Success)
            {
                throw new Exception("Failed to create descriptor set layout");
            }
            
        }
    }
    public unsafe void CreateDescriptorPool()
    {
        var poolSizes = new DescriptorPoolSize[]
        {
            new DescriptorPoolSize()
            {
                Type = DescriptorType.UniformBuffer,
                DescriptorCount = (uint)ctx.SwapChainImages!.Length,
            },
            new DescriptorPoolSize()
            {
                Type = DescriptorType.CombinedImageSampler,
                DescriptorCount = (uint)ctx.SwapChainImages!.Length,
            }
        };
        fixed (DescriptorPoolSize* poolSizesPtr = poolSizes)
        {
            DescriptorPoolCreateInfo poolInfo = new()
            {
                SType = StructureType.DescriptorPoolCreateInfo,
                MaxSets = (uint)ctx.SwapChainImages.Length,
                PoolSizeCount = 1,
                PPoolSizes = poolSizesPtr
            };
            if (vk!.CreateDescriptorPool(ctx.Device, &poolInfo, null, out ctx.DescriptorPool) != Result.Success)
            {
                throw new Exception("Failed to create descriptor pool");
            }
        }
    }

    public unsafe void CreateDescriptorSets()
    {
        var layouts = new DescriptorSetLayout[ctx.SwapChainImages.Length];
        Array.Fill(layouts, ctx.DescriptorSetLayout);

        fixed (DescriptorSetLayout* layoutsPtr = layouts)
        {
            DescriptorSetAllocateInfo allocateInfo = new()
            {
                SType = StructureType.DescriptorSetAllocateInfo,
                DescriptorPool = ctx.DescriptorPool,
                DescriptorSetCount = (uint)ctx.SwapChainImages.Length,
                PSetLayouts = layoutsPtr
            }; 
            ctx.DescriptorSets = new DescriptorSet[ctx.SwapChainImages.Length];
            fixed (DescriptorSet* descriptorSetsPtr = ctx.DescriptorSets)
            {
                if (vk!.AllocateDescriptorSets(ctx.Device, &allocateInfo, descriptorSetsPtr) != Result.Success)
                {
                    throw new Exception("Failed to allocate descriptor sets");
                }
            }
        }

        for (var i = 0; i < ctx.SwapChainImages.Length; i++)
        {
            DescriptorBufferInfo bufferInfo = new()
            {
                Buffer = ctx.UniformBuffers![i],
                Offset = 0,
                Range = (uint)sizeof(UniformBufferObject)
            };
            DescriptorImageInfo imageInfo = new()
            {
                ImageLayout = ImageLayout.ShaderReadOnlyOptimal,
                Sampler = ctx.TextureSampler,
                ImageView = ctx.TextureImageView,
            };
            var descriptorWrites = new WriteDescriptorSet[]
            {
                new()
                {
                    SType = StructureType.WriteDescriptorSet,
                    DstSet = ctx.DescriptorSets[i],
                    DstBinding = 0,
                    DstArrayElement = 0,

                    DescriptorType = DescriptorType.UniformBuffer,
                    DescriptorCount = 1,

                    PBufferInfo = &bufferInfo,
                    
                },
                new()
                {
                    SType = StructureType.WriteDescriptorSet,
                    DstSet = ctx.DescriptorSets[i],
                    DstBinding = 1,
                    DstArrayElement = 0,
                    DescriptorType = DescriptorType.CombinedImageSampler,
                    DescriptorCount = 1,
                    PImageInfo = &imageInfo,
                }
            };
            fixed(WriteDescriptorSet* writeDescriptorSetsPtr = descriptorWrites)
                vk!.UpdateDescriptorSets(ctx.Device, (uint)descriptorWrites.Length,
                    writeDescriptorSetsPtr, 0, null);
        }
        
        
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