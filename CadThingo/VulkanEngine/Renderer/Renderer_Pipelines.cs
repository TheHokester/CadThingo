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

        var bindings = new[] { uboLayoutBinding, samplerLayoutBinding };

        DescriptorSetLayoutBindingFlagsCreateInfo flagsCreateInfo = new()
            { SType = StructureType.DescriptorSetLayoutBindingFlagsCreateInfo };
        var flags = stackalloc DescriptorBindingFlags[2];
        fixed (DescriptorSetLayoutBinding* bindingsPtr = bindings)
        {
            if (descriptorIndexEnabled)
            {
                flags[0] = DescriptorBindingFlags.UpdateAfterBindBit |
                           DescriptorBindingFlags.UpdateUnusedWhilePendingBit;
                flags[1] = DescriptorBindingFlags.UpdateAfterBindBit |
                           DescriptorBindingFlags.UpdateUnusedWhilePendingBit;
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
        var shaderByteCode =
            File.ReadAllBytes(
                @"C:\Users\jamie\RiderProjects\CadThingo\CadThingo\Assets\Shaders\TexturedMesh.spv"); //worry about filepath and shader compilation later

        ShaderModule shaderModule = CreateShaderModule(shaderByteCode);

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
        fixed (VertexInputAttributeDescription* pAttributesDesc = allAttributeDescriptions)
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
            if (vk!.CreatePipelineLayout(device, &pipelineLayoutInfo, null, out pipelineLayout) != Result.Success)
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

    private void CreateGeometryDescriptorSetLayout()
    {
        // Binding 0: GBufferUBO — used by the vertex shader (model/view/proj transforms)
        // Binding 1-5: PBR material textures — used by the fragment shader only
        var bindings = new DescriptorSetLayoutBinding[]
        {
            new()
            {
                Binding = 0,
                DescriptorType = DescriptorType.UniformBuffer,
                DescriptorCount = 1,
                // GBufferUBO is read in VSMain for the MVP transform chain
                StageFlags = ShaderStageFlags.VertexBit,
                PImmutableSamplers = null
            },
            new()
            {
                Binding = 1,
                DescriptorType = DescriptorType.CombinedImageSampler,
                DescriptorCount = 1,
                StageFlags = ShaderStageFlags.FragmentBit,
                PImmutableSamplers = null
            },
            new()
            {
                Binding = 2,
                DescriptorType = DescriptorType.CombinedImageSampler,
                DescriptorCount = 1,
                StageFlags = ShaderStageFlags.FragmentBit,
                PImmutableSamplers = null
            },
            new()
            {
                Binding = 3,
                DescriptorType = DescriptorType.CombinedImageSampler,
                DescriptorCount = 1,
                StageFlags = ShaderStageFlags.FragmentBit,
                PImmutableSamplers = null
            },
            new()
            {
                Binding = 4,
                DescriptorType = DescriptorType.CombinedImageSampler,
                DescriptorCount = 1,
                StageFlags = ShaderStageFlags.FragmentBit,
                PImmutableSamplers = null
            },
            new()
            {
                Binding = 5,
                DescriptorType = DescriptorType.CombinedImageSampler,
                DescriptorCount = 1,
                StageFlags = ShaderStageFlags.FragmentBit,
                PImmutableSamplers = null
            },
        };

        DescriptorSetLayoutBindingFlagsCreateInfo flagsCreateInfo = new()
        {
            SType = StructureType.DescriptorSetLayoutBindingFlagsCreateInfo
        };
        // One flag entry per binding
        var flags = stackalloc DescriptorBindingFlags[bindings.Length];

        fixed (DescriptorSetLayoutBinding* pBindings = bindings)
        {
            if (descriptorIndexEnabled)
            {
                var updateFlags = DescriptorBindingFlags.UpdateAfterBindBit |
                                  DescriptorBindingFlags.UpdateUnusedWhilePendingBit;
                for (int i = 0; i < bindings.Length; i++)
                    flags[i] = updateFlags;

                flagsCreateInfo.BindingCount = (uint)bindings.Length;
                flagsCreateInfo.PBindingFlags = flags;
            }

            DescriptorSetLayoutCreateInfo layoutInfo = new()
            {
                SType = StructureType.DescriptorSetLayoutCreateInfo,
                BindingCount = (uint)bindings.Length,
                PBindings = pBindings,
            };
            if (descriptorIndexEnabled)
            {
                layoutInfo.Flags |= DescriptorSetLayoutCreateFlags.UpdateAfterBindPoolBit;
                layoutInfo.PNext = &flagsCreateInfo;
            }

            if (vk!.CreateDescriptorSetLayout(device, &layoutInfo, null, out geometryDescriptorSetLayout) !=
                Result.Success)
                throw new Exception("Failed to create geometry descriptor set layout");
        }
    }

    private void CreateGeometryPipeline()
    {
        byte[] shaderCode =
            File.ReadAllBytes(@"C:\Users\jamie\RiderProjects\CadThingo\CadThingo\Assets\Shaders\Geometry.spv");

        ShaderModule shader = CreateShaderModule(shaderCode);

        //config vertex stage 
        PipelineShaderStageCreateInfo vertShaderInfo = new()
        {
            SType = StructureType.PipelineShaderStageCreateInfo,
            Stage = ShaderStageFlags.VertexBit,
            Module = shader,
            PName = (byte*)SilkMarshal.StringToPtr("VSMain")
        };
        //config frag stage
        PipelineShaderStageCreateInfo fragShaderInfo = new()
        {
            SType = StructureType.PipelineShaderStageCreateInfo,
            Stage = ShaderStageFlags.FragmentBit,
            Module = shader,
            PName = (byte*)SilkMarshal.StringToPtr("PSMain")
        };

        var shaderStageCount = 2;
        var shaderStages = stackalloc PipelineShaderStageCreateInfo[]
        {
            vertShaderInfo,
            fragShaderInfo
        };
        var bindingDescription = Vertex.GetBindingDescription();
        var attributeDescriptions = Vertex.GetAttributeDescriptions();


        var vertexInputInfo = new PipelineVertexInputStateCreateInfo();
        fixed (VertexInputAttributeDescription* pAttributesDesc = attributeDescriptions)
        {
            vertexInputInfo = new()
            {
                SType = StructureType.PipelineVertexInputStateCreateInfo,
                VertexBindingDescriptionCount = 1,
                PVertexBindingDescriptions = &bindingDescription,
                VertexAttributeDescriptionCount = (uint)attributeDescriptions.Length,
                PVertexAttributeDescriptions = pAttributesDesc
            };
        }

        PipelineInputAssemblyStateCreateInfo inputAssemblyInfo = new()
        {
            SType = StructureType.PipelineInputAssemblyStateCreateInfo,
            Topology = PrimitiveTopology.TriangleList,
            PrimitiveRestartEnable = false
        };

        PipelineViewportStateCreateInfo viewportStateInfo = new()
        {
            SType = StructureType.PipelineViewportStateCreateInfo,
            ScissorCount = 1,
            ViewportCount = 1,
        };
        var dynamicStateCount = 2;
        var dynamicStates = stackalloc DynamicState[]
        {
            DynamicState.Viewport,
            DynamicState.Scissor
        };
        PipelineDynamicStateCreateInfo dynamicStateInfo = new()
        {
            SType = StructureType.PipelineDynamicStateCreateInfo,
            DynamicStateCount = (uint)dynamicStateCount,
            PDynamicStates = dynamicStates
        };
        PipelineRasterizationStateCreateInfo rasterizer = new()
        {
            SType = StructureType.PipelineRasterizationStateCreateInfo,
            DepthClampEnable = false, //dont clamp depth values
            RasterizerDiscardEnable = false, //dont discard primitives before rasterization
            PolygonMode = PolygonMode.Fill, //fill triangles
            LineWidth = 1, //line width (only relevant for wireframe)
            CullMode = CullModeFlags.BackBit, //Cull backfacing triangles
            FrontFace = FrontFace.CounterClockwise, //Counter clockwise winding
            DepthBiasEnable = false, //no depth bias
        };
        //config msaa settings
        PipelineMultisampleStateCreateInfo multisampleStateInfo = new()
        {
            SType = StructureType.PipelineMultisampleStateCreateInfo,
            SampleShadingEnable = false,
            RasterizationSamples = SampleCountFlags.Count1Bit //no msaa
        };
        //depth testing and z buffer config
        PipelineDepthStencilStateCreateInfo depthStencil = new()
        {
            DepthTestEnable = true,
            DepthWriteEnable = true,
            DepthCompareOp = CompareOp.Less,
            DepthBoundsTestEnable = false,
            StencilTestEnable = false,
        };
        //config color blending
        var blendAttachments = stackalloc PipelineColorBlendAttachmentState[4];
        for (int i = 0; i < 4; i++)
        {
            blendAttachments[i] = new()
            {
                BlendEnable = false,
                ColorWriteMask = ColorComponentFlags.RBit | ColorComponentFlags.GBit |
                                 ColorComponentFlags.BBit | ColorComponentFlags.ABit
            };
        }

        PipelineColorBlendStateCreateInfo colorBlendInfo = new()
        {
            SType = StructureType.PipelineColorBlendStateCreateInfo,
            AttachmentCount = 4,
            PAttachments = blendAttachments,
        };
        //configure pushconstants
        //TODO: create the push constant struct
        PushConstantRange pushConstantRange = new()
        {
            StageFlags = ShaderStageFlags.VertexBit,
            Offset = 0,
            Size = 128
        };
        //Create the pipeline layout - defunes resource organization
        fixed (DescriptorSetLayout* pDSLayout = &geometryDescriptorSetLayout)
        {
            PipelineLayoutCreateInfo layoutInfo = new()
            {
                SType = StructureType.PipelineLayoutCreateInfo,
                SetLayoutCount = 1,
                PSetLayouts = pDSLayout,
                PushConstantRangeCount = 1,
                PPushConstantRanges = &pushConstantRange
            };
            //create pipeline layout
            vk!.CreatePipelineLayout(device, &layoutInfo, null, out geometryPipelineLayout);
        }

        //assemble complete pipeline
        GraphicsPipelineCreateInfo pipelineInfo = new()
        {
            SType = StructureType.GraphicsPipelineCreateInfo,
            StageCount = (uint)shaderStageCount,
            PStages = shaderStages,
            PVertexInputState = &vertexInputInfo,
            PInputAssemblyState = &inputAssemblyInfo,
            PViewportState = &viewportStateInfo,
            PRasterizationState = &rasterizer,
            PMultisampleState = &multisampleStateInfo,
            PDepthStencilState = &depthStencil,
            PColorBlendState = &colorBlendInfo,
            PDynamicState = &dynamicStateInfo,
            Layout = pbrPipelineLayout,
            RenderPass = default,
            Subpass = 0,
            BasePipelineHandle = default,
        };
        //config for dynamic rendering
        var colorFormats = stackalloc Format[]
        {
            Format.R32G32B32A32Sfloat, // Position
            Format.R32G32B32A32Sfloat, // Normal
            Format.R8G8B8A8Unorm, // Albedo
            Format.R8G8B8A8Unorm, // Material
        };

        geometryPipelineRenderingCreateInfo = new()
        {
            ColorAttachmentCount = 4,
            PColorAttachmentFormats = colorFormats,
            DepthAttachmentFormat = FindDepthFormat(),
        };
        fixed (PipelineRenderingCreateInfo* pRenderingInfo = &geometryPipelineRenderingCreateInfo)
        {
            pipelineInfo.PNext = pRenderingInfo;
        }

        vk!.CreateGraphicsPipelines(device, default, 1, &pipelineInfo, null, out geometryPipeline);
    }


    private void CreatePBRDescriptorSetLayout()
    {
        // ── Set 0: LightingUBO ────────────────────────────────────────────────
        // Only the fragment shader reads lights, camPos, exposure, gamma etc.
        // The vertex shader is a procedural fullscreen triangle (SV_VertexID only).
        var set0Bindings = new DescriptorSetLayoutBinding[]
        {
            new()
            {
                Binding = 0,
                DescriptorType = DescriptorType.UniformBuffer,
                DescriptorCount = 1,
                StageFlags = ShaderStageFlags.FragmentBit,
                PImmutableSamplers = null
            },
        };

        // ── Set 1: G-Buffer inputs ────────────────────────────────────────────
        // Four samplers written by the geometry pass, read here for lighting.
        var set1Bindings = new DescriptorSetLayoutBinding[]
        {
            new() // gbufferPositionSampler
            {
                Binding = 0,
                DescriptorType = DescriptorType.CombinedImageSampler,
                DescriptorCount = 1,
                StageFlags = ShaderStageFlags.FragmentBit,
                PImmutableSamplers = null
            },
            new() // gbufferNormalSampler
            {
                Binding = 1,
                DescriptorType = DescriptorType.CombinedImageSampler,
                DescriptorCount = 1,
                StageFlags = ShaderStageFlags.FragmentBit,
                PImmutableSamplers = null
            },
            new() // gbufferAlbedoSampler  (rgb = baseColor, a = metallic)
            {
                Binding = 2,
                DescriptorType = DescriptorType.CombinedImageSampler,
                DescriptorCount = 1,
                StageFlags = ShaderStageFlags.FragmentBit,
                PImmutableSamplers = null
            },
            new() // gbufferMaterialSampler  (r = roughness, g = ao)
            {
                Binding = 3,
                DescriptorType = DescriptorType.CombinedImageSampler,
                DescriptorCount = 1,
                StageFlags = ShaderStageFlags.FragmentBit,
                PImmutableSamplers = null
            },
        };

        DescriptorSetLayoutBindingFlagsCreateInfo set0FlagsInfo = new()
            { SType = StructureType.DescriptorSetLayoutBindingFlagsCreateInfo };
        DescriptorSetLayoutBindingFlagsCreateInfo set1FlagsInfo = new()
            { SType = StructureType.DescriptorSetLayoutBindingFlagsCreateInfo };

        var set0Flags = stackalloc DescriptorBindingFlags[set0Bindings.Length];
        var set1Flags = stackalloc DescriptorBindingFlags[set1Bindings.Length];

        fixed (DescriptorSetLayoutBinding* pSet0 = set0Bindings)
        fixed (DescriptorSetLayoutBinding* pSet1 = set1Bindings)
        {
            if (descriptorIndexEnabled)
            {
                var updateFlags = DescriptorBindingFlags.UpdateAfterBindBit |
                                  DescriptorBindingFlags.UpdateUnusedWhilePendingBit;

                for (int i = 0; i < set0Bindings.Length; i++) set0Flags[i] = updateFlags;
                set0FlagsInfo.BindingCount = (uint)set0Bindings.Length;
                set0FlagsInfo.PBindingFlags = set0Flags;

                for (int i = 0; i < set1Bindings.Length; i++) set1Flags[i] = updateFlags;
                set1FlagsInfo.BindingCount = (uint)set1Bindings.Length;
                set1FlagsInfo.PBindingFlags = set1Flags;
            }

            // Create set 0 layout (LightingUBO) → stored in PBRDescriptorSetLayout
            DescriptorSetLayoutCreateInfo set0LayoutInfo = new()
            {
                SType = StructureType.DescriptorSetLayoutCreateInfo,
                BindingCount = (uint)set0Bindings.Length,
                PBindings = pSet0,
            };
            if (descriptorIndexEnabled)
            {
                set0LayoutInfo.Flags |= DescriptorSetLayoutCreateFlags.UpdateAfterBindPoolBit;
                set0LayoutInfo.PNext = &set0FlagsInfo;
            }

            if (vk!.CreateDescriptorSetLayout(device, &set0LayoutInfo, null, out PBRDescriptorSetLayout) !=
                Result.Success)
                throw new Exception("Failed to create PBR set 0 descriptor set layout");

            // Create set 1 layout (GBuffer samplers) → stored in PBRGBufferDescriptorSetLayout
            DescriptorSetLayoutCreateInfo set1LayoutInfo = new()
            {
                SType = StructureType.DescriptorSetLayoutCreateInfo,
                BindingCount = (uint)set1Bindings.Length,
                PBindings = pSet1,
            };
            if (descriptorIndexEnabled)
            {
                set1LayoutInfo.Flags |= DescriptorSetLayoutCreateFlags.UpdateAfterBindPoolBit;
                set1LayoutInfo.PNext = &set1FlagsInfo;
            }

            if (vk!.CreateDescriptorSetLayout(device, &set1LayoutInfo, null, out PBRGBufferDescriptorSetLayout) !=
                Result.Success)
                throw new Exception("Failed to create PBR set 1 (GBuffer) descriptor set layout");
        }
    }

    private void CreatePBRPipeline()
    {
        //load compiled shader
        byte[] shaderCode =
            File.ReadAllBytes(@"C:\Users\jamie\RiderProjects\CadThingo\CadThingo\Assets\Shaders\PBR.spv");

        //create shader module
        ShaderModule shader = CreateShaderModule(shaderCode);

        //config vertex stage 
        PipelineShaderStageCreateInfo vertShaderInfo = new()
        {
            SType = StructureType.PipelineShaderStageCreateInfo,
            Stage = ShaderStageFlags.VertexBit,
            Module = shader,
            PName = (byte*)SilkMarshal.StringToPtr("VSMain")
        };
        //config frag stage
        PipelineShaderStageCreateInfo fragShaderInfo = new()
        {
            SType = StructureType.PipelineShaderStageCreateInfo,
            Stage = ShaderStageFlags.FragmentBit,
            Module = shader,
            PName = (byte*)SilkMarshal.StringToPtr("PSMain")
        };
        var shaderStageCount = 2;
        var shaderStages = stackalloc PipelineShaderStageCreateInfo[]
        {
            vertShaderInfo,
            fragShaderInfo
        };


        var vertexInputInfo = new PipelineVertexInputStateCreateInfo()
        {
            SType = StructureType.PipelineVertexInputStateCreateInfo,
            VertexBindingDescriptionCount = 0,
            VertexAttributeDescriptionCount = 0,
        };


        PipelineInputAssemblyStateCreateInfo inputAssemblyInfo = new()
        {
            SType = StructureType.PipelineInputAssemblyStateCreateInfo,
            Topology = PrimitiveTopology.TriangleList,
            PrimitiveRestartEnable = false
        };

        PipelineViewportStateCreateInfo viewportStateInfo = new()
        {
            SType = StructureType.PipelineViewportStateCreateInfo,
            ScissorCount = 1,
            ViewportCount = 1,
        };
        var dynamicStateCount = 2;
        var dynamicStates = stackalloc DynamicState[]
        {
            DynamicState.Viewport,
            DynamicState.Scissor
        };
        PipelineDynamicStateCreateInfo dynamicStateInfo = new()
        {
            SType = StructureType.PipelineDynamicStateCreateInfo,
            DynamicStateCount = (uint)dynamicStateCount,
            PDynamicStates = dynamicStates
        };
        PipelineRasterizationStateCreateInfo rasterizer = new()
        {
            SType = StructureType.PipelineRasterizationStateCreateInfo,
            DepthClampEnable = false, //dont clamp depth values
            RasterizerDiscardEnable = false, //dont discard primitives before rasterization
            PolygonMode = PolygonMode.Fill, //fill triangles
            LineWidth = 1, //line width (only relevant for wireframe)
            CullMode = CullModeFlags.None, //Cull None
            FrontFace = FrontFace.CounterClockwise, //Counter clockwise winding
            DepthBiasEnable = false, //no depth bias
        };
        //config msaa settings
        PipelineMultisampleStateCreateInfo multisampleStateInfo = new()
        {
            SType = StructureType.PipelineMultisampleStateCreateInfo,
            SampleShadingEnable = false,
            RasterizationSamples = SampleCountFlags.Count1Bit //no msaa
        };
        //depth testing and z buffer config
        PipelineDepthStencilStateCreateInfo depthStencil = new()
        {
            DepthTestEnable = false,
            DepthWriteEnable = false,
            DepthCompareOp = CompareOp.Always,
            DepthBoundsTestEnable = false,
            StencilTestEnable = false,
        };
        //config color blending
        PipelineColorBlendAttachmentState colorBlendAttachment = new()
        {
            BlendEnable = false, //enable alpha blending
            ColorWriteMask = ColorComponentFlags.RBit | ColorComponentFlags.GBit |
                             ColorComponentFlags.BBit | ColorComponentFlags.ABit
        };
        PipelineColorBlendStateCreateInfo colorBlendStateInfo = new()
        {
            SType = StructureType.PipelineColorBlendStateCreateInfo,
            LogicOpEnable = false,
            AttachmentCount = 1,
            PAttachments = &colorBlendAttachment
        };
        //configure pushconstants
        //TODO: create the push constant struct
        PushConstantRange pushConstantRange = new()
        {
            StageFlags = ShaderStageFlags.VertexBit,
            Offset = 0,
            Size = 128
        };
        var pbrLayouts = stackalloc DescriptorSetLayout[]
        {
            PBRDescriptorSetLayout,
            PBRGBufferDescriptorSetLayout
        };
        //Create the pipeline layout - defunes resource organization

        PipelineLayoutCreateInfo layoutInfo = new()
        {
            SType = StructureType.PipelineLayoutCreateInfo,
            SetLayoutCount = 2,
            PSetLayouts = pbrLayouts,
            PushConstantRangeCount = 1,
            PPushConstantRanges = &pushConstantRange
        };
        //create pipeline layout
        vk!.CreatePipelineLayout(device, &layoutInfo, null, out pbrPipelineLayout);
        //assemble complete pipeline
        GraphicsPipelineCreateInfo pipelineInfo = new()
        {
            SType = StructureType.GraphicsPipelineCreateInfo,
            StageCount = (uint)shaderStageCount,
            PStages = shaderStages,
            PVertexInputState = &vertexInputInfo,
            PInputAssemblyState = &inputAssemblyInfo,
            PViewportState = &viewportStateInfo,
            PRasterizationState = &rasterizer,
            PMultisampleState = &multisampleStateInfo,
            PDepthStencilState = &depthStencil,
            PColorBlendState = &colorBlendStateInfo,
            PDynamicState = &dynamicStateInfo,
            Layout = pbrPipelineLayout,
            RenderPass = default,
            Subpass = 0,
            BasePipelineHandle = default,
        };
        //config for dynamic rendering
        fixed (Format* pColorAttcFormat = &swapChainImageFormat)
        {
            pbrPipelineRenderingCreateInfo = new()
            {
                ColorAttachmentCount = 1,
                PColorAttachmentFormats = pColorAttcFormat,
                DepthAttachmentFormat = FindDepthFormat(),
            };
        }

        fixed (PipelineRenderingCreateInfo* pRenderingInfo = &pbrPipelineRenderingCreateInfo)
        {
            pipelineInfo.PNext = pRenderingInfo;
        }

        //Create the final graphics pipeline
        vk!.CreateGraphicsPipelines(device, default, 1, &pipelineInfo, null, out pbrLightingPipeline);
    }
}