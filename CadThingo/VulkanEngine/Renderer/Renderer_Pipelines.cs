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
        
        vk!.DestroyShaderModule(device, shaderModule, null);
    }

    // Set 0 of the geometry pipeline. One binding (the per-frame FrameUBO with view+proj),
    // bound once at the start of the geometry pass and reused for every draw.
    private void CreateGeometryFrameDescriptorSetLayout()
    {
        var binding = new DescriptorSetLayoutBinding
        {
            Binding = 0,
            DescriptorType = DescriptorType.UniformBuffer,
            DescriptorCount = 1,
            StageFlags = ShaderStageFlags.VertexBit,
            PImmutableSamplers = null,
        };

        DescriptorSetLayoutBindingFlagsCreateInfo flagsCreateInfo = new()
            { SType = StructureType.DescriptorSetLayoutBindingFlagsCreateInfo };
        var flag = DescriptorBindingFlags.UpdateAfterBindBit |
                   DescriptorBindingFlags.UpdateUnusedWhilePendingBit;

        if (descriptorIndexEnabled)
        {
            flagsCreateInfo.BindingCount = 1;
            flagsCreateInfo.PBindingFlags = &flag;
        }

        DescriptorSetLayoutCreateInfo layoutInfo = new()
        {
            SType = StructureType.DescriptorSetLayoutCreateInfo,
            BindingCount = 1,
            PBindings = &binding,
        };
        if (descriptorIndexEnabled)
        {
            layoutInfo.Flags |= DescriptorSetLayoutCreateFlags.UpdateAfterBindPoolBit;
            layoutInfo.PNext = &flagsCreateInfo;
        }

        if (vk!.CreateDescriptorSetLayout(device, &layoutInfo, null, out geometryFrameDescriptorSetLayout) !=
            Result.Success)
            throw new Exception("Failed to create geometry frame descriptor set layout");
    }

    // Set 1 of the geometry pipeline. Five PBR textures (baseColor, metallicRoughness, normal,
    // occlusion, emissive). One set per material; bound per-draw.
    const int materialSetCount = 5;
    
    private void CreateGeometryMaterialDescriptorSetLayout()
    {
        var bindings = new DescriptorSetLayoutBinding[]
        {
            new()
            {
                Binding = 0,
                DescriptorType = DescriptorType.StorageBuffer,
                DescriptorCount = 1,
                StageFlags = ShaderStageFlags.FragmentBit,
            },
            new()
            {
                Binding = 1,
                DescriptorType = DescriptorType.StorageBuffer,
                DescriptorCount = 1,
                StageFlags = ShaderStageFlags.VertexBit|ShaderStageFlags.FragmentBit,
            },
            new()
            {
                Binding = 2,
                DescriptorType = DescriptorType.SampledImage,
                DescriptorCount = MAX_MATERIALS * 5,
                StageFlags = ShaderStageFlags.FragmentBit,
            }, 
            new()
            {
                Binding = 3,
                DescriptorType = DescriptorType.Sampler,
                DescriptorCount = 8,
                StageFlags = ShaderStageFlags.FragmentBit,
            }
        };
        
            

        DescriptorSetLayoutBindingFlagsCreateInfo flagsCreateInfo = new()
            { SType = StructureType.DescriptorSetLayoutBindingFlagsCreateInfo };
        var flags = stackalloc DescriptorBindingFlags[bindings.Length];

        fixed (DescriptorSetLayoutBinding* pBindings = bindings)
        {
            if (descriptorIndexEnabled)
            {
                // Only the bindless texture array (binding 2) needs UpdateAfterBind +
                // PartiallyBound — RegisterBindless writes new texture slots while the set
                // is live. The two storage buffers (bindings 0,1) and sampler (3) are
                // written once at setup, so they don't need UpdateAfterBind (which would
                // require descriptorBindingStorageBufferUpdateAfterBind / SamplerUpdateAfterBind
                // features that we don't request).
                flags[0] = DescriptorBindingFlags.UpdateUnusedWhilePendingBit;
                flags[1] = DescriptorBindingFlags.UpdateUnusedWhilePendingBit;
                flags[2] = DescriptorBindingFlags.UpdateAfterBindBit |
                           DescriptorBindingFlags.UpdateUnusedWhilePendingBit |
                           DescriptorBindingFlags.PartiallyBoundBit;
                flags[3] = DescriptorBindingFlags.UpdateUnusedWhilePendingBit;

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

            if (vk!.CreateDescriptorSetLayout(device, &layoutInfo, null, out geometryMaterialDescriptorSetLayout) !=
                Result.Success)
                throw new Exception("Failed to create geometry material descriptor set layout");
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
            CullMode = CullModeFlags.BackBit, // TEMP option A: disable culling to verify cube renders
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
        //config color blending — one entry per geometry attachment (5 total: position,
        //normal, albedo, material, emissive). Blending disabled (G-buffer = direct write).
        var blendAttachments = stackalloc PipelineColorBlendAttachmentState[5];
        for (int i = 0; i < 5; i++)
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
            AttachmentCount = 5,
            PAttachments = blendAttachments,
        };
        // Pipeline layout — set 0 is per-frame (FrameUBO), set 1 is bindless
        // (StructuredBuffer<PbrMaterial>, StructuredBuffer<InstanceData>, Texture2D[], SamplerState[]).
        // Per-draw model matrix + material index live in the instance SSBO; no push constants.
        var setLayouts = stackalloc DescriptorSetLayout[]
        {
            geometryFrameDescriptorSetLayout,
            geometryMaterialDescriptorSetLayout,
        };
        PipelineLayoutCreateInfo layoutInfo = new()
        {
            SType = StructureType.PipelineLayoutCreateInfo,
            SetLayoutCount = 2,
            PSetLayouts = setLayouts,
            PushConstantRangeCount = 0,
            PPushConstantRanges = null
        };
        vk!.CreatePipelineLayout(device, &layoutInfo, null, out geometryPipelineLayout);

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
            Layout = geometryPipelineLayout,
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
            Format.R8G8B8A8Unorm, // Emissive
        };

        geometryPipelineRenderingCreateInfo = new()
        {
            SType = StructureType.PipelineRenderingCreateInfo,
            ColorAttachmentCount = 5,
            PColorAttachmentFormats = colorFormats,
            DepthAttachmentFormat = FindDepthFormat(),
        };
        fixed (PipelineRenderingCreateInfo* pRenderingInfo = &geometryPipelineRenderingCreateInfo)
        {
            pipelineInfo.PNext = pRenderingInfo;
        }

        vk!.CreateGraphicsPipelines(device, default, 1, &pipelineInfo, null, out geometryPipeline);
        
        vk!.DestroyShaderModule(device, shader, null);
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
            new() // TLAS for ray-traced shadows. Bound once at startup; no per-frame update.
            {
                Binding = 1,
                DescriptorType = DescriptorType.AccelerationStructureKhr,
                DescriptorCount = 1,
                StageFlags = ShaderStageFlags.FragmentBit,
                PImmutableSamplers = null,
            },
        };

        // ── Set 1: G-Buffer inputs ────────────────────────────────────────────
        // Five samplers written by the geometry pass, read here for lighting.
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
            new() // gbufferEmissiveSampler  (rgb = emissive radiance)
            {
                Binding = 4,
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

                for (int i = 0; i < set0Bindings.Length; i++)
                {
                    // AccelerationStructureKhr bindings can't carry UpdateAfterBindBit
                    // unless descriptorBindingAccelerationStructureUpdateAfterBind is
                    // enabled (a separate feature we don't request). Leave default (0).
                    if (set0Bindings[i].DescriptorType == DescriptorType.AccelerationStructureKhr) continue;
                    set0Flags[i] = updateFlags;
                }
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
        var pbrLayouts = stackalloc DescriptorSetLayout[]
        {
            PBRDescriptorSetLayout,
            PBRGBufferDescriptorSetLayout
        };
        // PBR lighting pass uses no push constants — set 0 is per-frame light/cam UBO + TLAS,
        // set 1 is g-buffer samplers, no per-draw data.
        PipelineLayoutCreateInfo layoutInfo = new()
        {
            SType = StructureType.PipelineLayoutCreateInfo,
            SetLayoutCount = 2,
            PSetLayouts = pbrLayouts,
            PushConstantRangeCount = 0,
            PPushConstantRanges = null
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
        //config for dynamic rendering — lighting pass writes to FinalColor (R8G8B8A8Unorm), no depth.
        Format finalColorFormat = Format.R8G8B8A8Unorm;
        pbrPipelineRenderingCreateInfo = new()
        {
            SType = StructureType.PipelineRenderingCreateInfo,
            ColorAttachmentCount = 1,
            PColorAttachmentFormats = &finalColorFormat,
            DepthAttachmentFormat = Format.Undefined,
        };

        fixed (PipelineRenderingCreateInfo* pRenderingInfo = &pbrPipelineRenderingCreateInfo)
        {
            pipelineInfo.PNext = pRenderingInfo;
        }

        //Create the final graphics pipeline
        vk!.CreateGraphicsPipelines(device, default, 1, &pipelineInfo, null, out pbrLightingPipeline);
        
        vk!.DestroyShaderModule(device, shader, null);
    }
}