using System.Numerics;
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
        };
        PipelineLayoutCreateInfo layoutInfo = new()
        {
            SType = StructureType.PipelineLayoutCreateInfo,
            SetLayoutCount = 2,
            PSetLayouts = setLayouts,
            PushConstantRangeCount = 0,
            PPushConstantRanges = null
        };
        // vk!.CreatePipelineLayout(device, &layoutInfo, null, out geometryPipelineLayout);

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
            // Layout = geometryPipelineLayout,
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

        // vk!.CreateGraphicsPipelines(device, default, 1, &pipelineInfo, null, out geometryPipeline);
        
        vk!.DestroyShaderModule(device, shader, null);
    }


    private void CreatePBRDescriptorSetLayout()
    {
        // ── Set 0: LightingFrameUBO + Light SSBO + TLAS + Tile cull buffers ───
        // Only the fragment shader reads lights, camPos, exposure, gamma etc.
        // The vertex shader is a procedural fullscreen triangle (SV_VertexID only).
        // Phase 4 added bindings 3,4 for the tiled light cull output.
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
            new() // StructuredBuffer<PbrLight> — per-frame, rewritten in UpdateLightingBuffers.
            {
                Binding = 1,
                DescriptorType = DescriptorType.StorageBuffer,
                DescriptorCount = 1,
                StageFlags = ShaderStageFlags.FragmentBit,
                PImmutableSamplers = null,
            },
            new() // TLAS for ray-traced shadows. Bound once at startup; no per-frame update.
            {
                Binding = 2,
                DescriptorType = DescriptorType.AccelerationStructureKhr,
                DescriptorCount = 1,
                StageFlags = ShaderStageFlags.FragmentBit,
                PImmutableSamplers = null,
            },
            new() // tileLightCount[tileIdx] — produced by LightCulling.slang.
            {
                Binding = 3,
                DescriptorType = DescriptorType.StorageBuffer,
                DescriptorCount = 1,
                StageFlags = ShaderStageFlags.FragmentBit,
                PImmutableSamplers = null,
            },
            new() // tileLightIndices[tileIdx*MAX + slot] — produced by LightCulling.slang.
            {
                Binding = 4,
                DescriptorType = DescriptorType.StorageBuffer,
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
                    // AccelerationStructureKhr and StorageBuffer bindings can't carry
                    // UpdateAfterBindBit unless their respective UpdateAfterBind features
                    // (descriptorBindingAccelerationStructureUpdateAfterBind /
                    // descriptorBindingStorageBufferUpdateAfterBind) are enabled — we
                    // don't request either. The light SSBO is per-frame mapped memory,
                    // so per-frame vkUpdateDescriptorSets isn't needed anyway.
                    if (set0Bindings[i].DescriptorType == DescriptorType.AccelerationStructureKhr) continue;
                    if (set0Bindings[i].DescriptorType == DescriptorType.StorageBuffer) continue;
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

// ────────────────────────────────────────────────────────────────────────────
//  Pipeline wrapper layout
// ────────────────────────────────────────────────────────────────────────────
//  Three layers:
//    1. PipelineBase      — owns the handle, layout, cache, descriptor set
//                           layouts, push-constant ranges, and the lifecycle.
//    2. GraphicsPipeline  — assembles GraphicsPipelineCreateInfo from a set of
//                           protected virtual hooks (vertex input, raster,
//                           blend, depth, dynamic rendering formats, …) so
//                           concrete pipelines override only what differs.
//    3. ComputePipeline   — single-shader-stage compute equivalent.
//
//  Concrete pipelines (Geometry, PbrLighting, DrawCull, LightCull, …)
//  inherit from layer 2/3 and own their own SSBOs/UBOs, push-constant
//  structs, descriptor sets, and Record(...) entry points.
// ────────────────────────────────────────────────────────────────────────────

public abstract unsafe class PipelineBase : IDisposable
{
    // Single reference back to the renderer — pipelines call into things like
    // Renderer.CreateMappedStorageBuffer / CreateShaderModule / FindDepthFormat
    // directly. The fields and methods on Renderer that pipelines need are
    // 'internal' for same-assembly access.
    protected readonly Renderer Renderer;

    // Convenience accessors so subclass bodies stay short — these just forward
    // to the renderer's vk / device.
    protected Vk     Vk     => Renderer.vk!;
    protected Device Device => Renderer.device;

    protected Pipeline       PipelineHandle;
    protected PipelineLayout PipelineLayoutHandle;
    protected PipelineCache  PipelineCacheHandle;

    // Subclasses populate these in CreateDescriptorSetLayouts(). The default
    // CreatePipelineLayout() reads them to build the VkPipelineLayout.
    protected DescriptorSetLayout[] DescriptorSetLayouts = Array.Empty<DescriptorSetLayout>();

    /// <summary>
    /// Descriptor sets for the pipeline <br/>
    /// DescriptorSets[layoutNum][frame]
    /// </summary>
    protected DescriptorSet[][] DescriptorSets;
    protected PushConstantRange[]   PushConstantRanges   = Array.Empty<PushConstantRange>();

    public Pipeline                  Handle    => PipelineHandle;
    public PipelineLayout            Layout    => PipelineLayoutHandle;
    
    public DescriptorSet GetDescriptorSet(int layoutNum, uint frame) => DescriptorSets[layoutNum][frame];
    public abstract PipelineBindPoint BindPoint { get; }

    protected PipelineBase(Renderer renderer)
    {
        Renderer = renderer;
    }

    // Called once by the owner (Renderer) after construction. Each step is a
    // virtual hook so concrete pipelines slot in their own logic without
    // re-implementing the whole flow.
    public void Initialize()
    {
        CreateDescriptorSetLayouts();
        CreatePipelineLayout();
        CreatePipeline();
        CreateResources();
        CreateDescriptorSets();
        WriteDescriptors();
    }

    // Required: populate descriptorSetLayouts and (optionally) pushConstantRanges.
    protected abstract void CreateDescriptorSetLayouts();

    // Required: build the VkPipeline itself. GraphicsPipeline / ComputePipeline
    // seal this and drive it from their hooks; only override directly if a
    // pipeline doesn't fit either category.
    protected abstract void CreatePipeline();

    protected virtual void CreatePipelineLayout()
    {
        fixed (DescriptorSetLayout* pLayouts = DescriptorSetLayouts)
        fixed (PushConstantRange*   pRanges  = PushConstantRanges)
        {
            PipelineLayoutCreateInfo info = new()
            {
                SType                  = StructureType.PipelineLayoutCreateInfo,
                SetLayoutCount         = (uint)DescriptorSetLayouts.Length,
                PSetLayouts            = pLayouts,
                PushConstantRangeCount = (uint)PushConstantRanges.Length,
                PPushConstantRanges    = pRanges,
            };
            if (Vk.CreatePipelineLayout(Device, &info, null, out PipelineLayoutHandle) != Result.Success)
                throw new Exception($"Failed to create pipeline layout for {GetType().Name}");
        }
    }

    // Concrete pipelines override these to allocate their owned SSBOs/UBOs,
    // allocate descriptor sets from the pool, and write the initial bindings.
    protected virtual void CreateResources()          { }
    protected virtual void CreateDescriptorSets()     { }
    protected virtual void WriteDescriptors()  { }

    public virtual void Dispose()
    {
        if (PipelineHandle.Handle       != 0) Vk.DestroyPipeline(Device, PipelineHandle, null);
        if (PipelineLayoutHandle.Handle != 0) Vk.DestroyPipelineLayout(Device, PipelineLayoutHandle, null);
        foreach (var dsl in DescriptorSetLayouts)
            if (dsl.Handle != 0) Vk.DestroyDescriptorSetLayout(Device, dsl, null);
        if (PipelineCacheHandle.Handle  != 0) Vk.DestroyPipelineCache(Device, PipelineCacheHandle, null);
    }
}

public abstract unsafe class GraphicsPipeline : PipelineBase
{
    public override PipelineBindPoint BindPoint => PipelineBindPoint.Graphics;

    protected GraphicsPipeline(Renderer renderer) : base(renderer) { }

    // ── Required overrides ──────────────────────────────────────────────────
    protected abstract string   ShaderPath              { get; }
    protected abstract Format[] ColorAttachmentFormats  { get; }

    // ── Optional hooks (defaults match the common case) ─────────────────────
    protected virtual Format DepthAttachmentFormat { get; init; } = Format.Undefined;

    protected virtual (ShaderStageFlags Stage, string EntryPoint)[] ShaderStages => new[]
    {
        (ShaderStageFlags.VertexBit,   "VSMain"),
        (ShaderStageFlags.FragmentBit, "PSMain"),
    };

    protected virtual VertexInputBindingDescription[]   GetVertexInputBindings()   => Array.Empty<VertexInputBindingDescription>();
    protected virtual VertexInputAttributeDescription[] GetVertexInputAttributes() => Array.Empty<VertexInputAttributeDescription>();

    protected virtual PipelineInputAssemblyStateCreateInfo BuildInputAssembly() => new()
    {
        SType                  = StructureType.PipelineInputAssemblyStateCreateInfo,
        Topology               = PrimitiveTopology.TriangleList,
        PrimitiveRestartEnable = false,
    };

    protected virtual PipelineViewportStateCreateInfo BuildViewportState() => new()
    {
        SType         = StructureType.PipelineViewportStateCreateInfo,
        ViewportCount = 1,
        ScissorCount  = 1,
    };

    protected virtual PipelineRasterizationStateCreateInfo BuildRasterizer() => new()
    {
        SType                   = StructureType.PipelineRasterizationStateCreateInfo,
        DepthClampEnable        = false,
        RasterizerDiscardEnable = false,
        PolygonMode             = PolygonMode.Fill,
        LineWidth               = 1.0f,
        CullMode                = CullModeFlags.BackBit,
        FrontFace               = FrontFace.CounterClockwise,
        DepthBiasEnable         = false,
    };

    protected virtual PipelineMultisampleStateCreateInfo BuildMultisample() => new()
    {
        SType                = StructureType.PipelineMultisampleStateCreateInfo,
        SampleShadingEnable  = false,
        RasterizationSamples = SampleCountFlags.Count1Bit,
    };

    protected virtual PipelineDepthStencilStateCreateInfo BuildDepthStencil() => new()
    {
        SType                 = StructureType.PipelineDepthStencilStateCreateInfo,
        DepthTestEnable       = true,
        DepthWriteEnable      = true,
        DepthCompareOp        = CompareOp.Less,
        DepthBoundsTestEnable = false,
        StencilTestEnable     = false,
        MinDepthBounds        = 0.0f,
        MaxDepthBounds        = 1.0f,
    };

    // One no-blend attachment per color target by default. Override for
    // additive / alpha / G-buffer-with-mixed-formats cases.
    protected virtual PipelineColorBlendAttachmentState[] BuildColorBlendAttachments()
    {
        var att = new PipelineColorBlendAttachmentState[ColorAttachmentFormats.Length];
        for (int i = 0; i < att.Length; i++)
        {
            att[i] = new()
            {
                BlendEnable    = false,
                ColorWriteMask = ColorComponentFlags.RBit | ColorComponentFlags.GBit |
                                 ColorComponentFlags.BBit | ColorComponentFlags.ABit,
            };
        }
        return att;
    }

    protected virtual DynamicState[] DynamicStates => new[] { DynamicState.Viewport, DynamicState.Scissor };

    // Drives the pipeline build from the hooks above. Sealed because concrete
    // pipelines should configure via overrides rather than re-implementing
    // the whole assembly — that's the main thing keeping the sprawl out.
    protected sealed override void CreatePipeline()
    {
        byte[] code   = File.ReadAllBytes(ShaderPath);
        var    module = Renderer.CreateShaderModule(code);

        var stageDefs = ShaderStages;
        var stages    = stackalloc PipelineShaderStageCreateInfo[stageDefs.Length];
        var entryPtrs = stackalloc nint[stageDefs.Length];
        for (int i = 0; i < stageDefs.Length; i++)
        {
            entryPtrs[i] = SilkMarshal.StringToPtr(stageDefs[i].EntryPoint);
            stages[i] = new()
            {
                SType  = StructureType.PipelineShaderStageCreateInfo,
                Stage  = stageDefs[i].Stage,
                Module = module,
                PName  = (byte*)entryPtrs[i],
            };
        }

        var vertexInputBindings   = GetVertexInputBindings();
        var vertexInputAttributes = GetVertexInputAttributes();
        var inputAssembly         = BuildInputAssembly();
        var viewportState         = BuildViewportState();
        var rasterizer            = BuildRasterizer();
        var multisample           = BuildMultisample();
        var depthStencil          = BuildDepthStencil();
        var blendAttachments      = BuildColorBlendAttachments();
        var dynamicStates         = DynamicStates;
        var colorFormats          = ColorAttachmentFormats;

        fixed (VertexInputBindingDescription*     pBindings  = vertexInputBindings)
        fixed (VertexInputAttributeDescription*   pAttribs   = vertexInputAttributes)
        fixed (PipelineColorBlendAttachmentState* pBlend     = blendAttachments)
        fixed (DynamicState*                      pDyn       = dynamicStates)
        fixed (Format*                            pColorFmts = colorFormats)
        {
            var vertexInput = new PipelineVertexInputStateCreateInfo
            {
                SType                           = StructureType.PipelineVertexInputStateCreateInfo,
                VertexBindingDescriptionCount   = (uint)vertexInputBindings.Length,
                PVertexBindingDescriptions      = pBindings,
                VertexAttributeDescriptionCount = (uint)vertexInputAttributes.Length,
                PVertexAttributeDescriptions    = pAttribs,
            };

            var colorBlend = new PipelineColorBlendStateCreateInfo
            {
                SType           = StructureType.PipelineColorBlendStateCreateInfo,
                LogicOpEnable   = false,
                LogicOp         = LogicOp.Copy,
                AttachmentCount = (uint)blendAttachments.Length,
                PAttachments    = pBlend,
            };

            var dynamic = new PipelineDynamicStateCreateInfo
            {
                SType             = StructureType.PipelineDynamicStateCreateInfo,
                DynamicStateCount = (uint)dynamicStates.Length,
                PDynamicStates    = pDyn,
            };

            var renderingInfo = new PipelineRenderingCreateInfo
            {
                SType                   = StructureType.PipelineRenderingCreateInfo,
                ColorAttachmentCount    = (uint)colorFormats.Length,
                PColorAttachmentFormats = pColorFmts,
                DepthAttachmentFormat   = DepthAttachmentFormat,
                StencilAttachmentFormat = Format.Undefined,
            };

            var info = new GraphicsPipelineCreateInfo
            {
                SType               = StructureType.GraphicsPipelineCreateInfo,
                PNext               = &renderingInfo,
                StageCount          = (uint)stageDefs.Length,
                PStages             = stages,
                PVertexInputState   = &vertexInput,
                PInputAssemblyState = &inputAssembly,
                PViewportState      = &viewportState,
                PRasterizationState = &rasterizer,
                PMultisampleState   = &multisample,
                PDepthStencilState  = &depthStencil,
                PColorBlendState    = &colorBlend,
                PDynamicState       = &dynamic,
                Layout              = PipelineLayoutHandle,
                RenderPass          = default,
                Subpass             = 0,
                BasePipelineHandle  = default,
                BasePipelineIndex   = -1,
            };

            if (Vk.CreateGraphicsPipelines(Device, PipelineCacheHandle, 1, &info, null, out PipelineHandle) != Result.Success)
                throw new Exception($"Failed to create graphics pipeline for {GetType().Name}");
        }

        for (int i = 0; i < stageDefs.Length; i++) SilkMarshal.Free(entryPtrs[i]);
        Vk.DestroyShaderModule(Device, module, null);
    }
}

public abstract unsafe class ComputePipeline : PipelineBase
{
    public override PipelineBindPoint BindPoint => PipelineBindPoint.Compute;

    protected ComputePipeline(Renderer renderer) : base(renderer) { }

    protected abstract string ShaderPath { get; }

    // slangc emits the SPIR-V OpEntryPoint as "main" regardless of the source
    // function name. Override only if the .spv was produced by a toolchain
    // that preserves a different entry-point symbol.
    protected virtual string EntryPoint => "main";

    protected sealed override void CreatePipeline()
    {
        byte[] code   = File.ReadAllBytes(ShaderPath);
        var    module = Renderer.CreateShaderModule(code);
        var    entry  = SilkMarshal.StringToPtr(EntryPoint);

        var stage = new PipelineShaderStageCreateInfo
        {
            SType  = StructureType.PipelineShaderStageCreateInfo,
            Stage  = ShaderStageFlags.ComputeBit,
            Module = module,
            PName  = (byte*)entry,
        };

        var info = new ComputePipelineCreateInfo
        {
            SType  = StructureType.ComputePipelineCreateInfo,
            Stage  = stage,
            Layout = PipelineLayoutHandle,
        };

        if (Vk.CreateComputePipelines(Device, PipelineCacheHandle, 1, &info, null, out PipelineHandle) != Result.Success)
            throw new Exception($"Failed to create compute pipeline for {GetType().Name}");

        SilkMarshal.Free(entry);
        Vk.DestroyShaderModule(Device, module, null);
    }
}

public sealed unsafe class GeometryPipeline : GraphicsPipeline
{
    struct GeometryUBO
    {
        public Matrix4x4 view;
        public Matrix4x4 proj;
    }
    //Per frame uniform buffers for geometry pipeline
    private UboBuffer[] GeometryUniformBuffers = new UboBuffer[2];
    
    protected override string ShaderPath { get; } =
        @"C:\Users\jamie\RiderProjects\CadThingo\CadThingo\Assets\Shaders\Geometry.spv";
    protected override Format[] ColorAttachmentFormats { get; } =
    [
        Format.R32G32B32A32Sfloat, // Position
        Format.R32G32B32A32Sfloat, // Normal
        Format.R8G8B8A8Unorm, // Albedo
        Format.R8G8B8A8Unorm, // Material
        Format.R8G8B8A8Unorm // Emissive
    ];
    
    
    public GeometryPipeline(Renderer renderer) : base(renderer)
    {
        DepthAttachmentFormat = renderer.FindDepthFormat();
    }  

    protected override void CreateDescriptorSetLayouts()
    {
        CreateFrameDescriptorSetLayout(out var frameDSL);
        DescriptorSetLayouts = new[] { frameDSL,  Engine.ResourceManager.GetBindlessLayout()};
    }

    protected override void CreateResources()
    {
        base.CreateResources();
        
        
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            Renderer.CreateMappedUniformBuffer(sizeof(GeometryUBO), ref GeometryUniformBuffers[i]);
        }
    }

    protected override void CreateDescriptorSets()
    {
        // Per-frame "frame" descriptor sets (set 0): only binding 0 = FrameUBO (view+proj).
        // Material samplers live on set 1 and are allocated from materialDescriptorPool.
        var layouts = stackalloc DescriptorSetLayout[(int)Renderer.MAX_CONCURRENT_FRAMES];
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++) layouts[i] = DescriptorSetLayouts[0];

        DescriptorSetAllocateInfo allocateInfo = new()
        {
            SType = StructureType.DescriptorSetAllocateInfo,
            DescriptorPool = Renderer.descriptorPool,
            DescriptorSetCount = Renderer.MAX_CONCURRENT_FRAMES,
            PSetLayouts = layouts
        };
        DescriptorSets = new DescriptorSet[1][];
        DescriptorSets[0] = new DescriptorSet[Renderer.MAX_CONCURRENT_FRAMES];
        fixed (DescriptorSet* pDS = DescriptorSets[0])
        {
            if (Vk.AllocateDescriptorSets(Device, &allocateInfo, pDS) != Result.Success)
            {
                throw new Exception("Failed to allocate descriptor sets using layout 0 for geometry pipeline");
            }
        }
    }

    protected override void WriteDescriptors()
    {
        for (var i = 0; i < Renderer.MAX_CONCURRENT_FRAMES; i++)
        {
            DescriptorBufferInfo bufferInfo = new()
            {
                Buffer = GeometryUniformBuffers[i].buffer,
                Offset = 0,
                Range = (ulong)sizeof(GeometryUBO),
            };
            WriteDescriptorSet descriptorWrite = new()
            {
                SType = StructureType.WriteDescriptorSet,
                DstSet = DescriptorSets[0][i],
                DstBinding = 0,
                DstArrayElement = 0,
                DescriptorType = DescriptorType.UniformBuffer,
                DescriptorCount = 1,
                PBufferInfo = &bufferInfo
            };
            Vk!.UpdateDescriptorSets(Device, 1, &descriptorWrite, 0, null);
        }
    }

    protected override VertexInputBindingDescription[] GetVertexInputBindings()
    {
        return [Vertex.GetBindingDescription()];
    }

    protected override VertexInputAttributeDescription[] GetVertexInputAttributes()
    {
        return Vertex.GetAttributeDescriptions();
    }
    
    
    // Set 0 of the geometry pipeline. One binding (the per-frame FrameUBO with view+proj),
    // bound once at the start of the geometry pass and reused for every draw.
    private void CreateFrameDescriptorSetLayout(out DescriptorSetLayout layout)
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

        if (Renderer.descriptorIndexEnabled)
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
        if (Renderer.descriptorIndexEnabled)
        {
            layoutInfo.Flags |= DescriptorSetLayoutCreateFlags.UpdateAfterBindPoolBit;
            layoutInfo.PNext = &flagsCreateInfo;
        }

        if (Vk!.CreateDescriptorSetLayout(Device, &layoutInfo, null, out layout) !=
            Result.Success)
            throw new Exception("Failed to create geometry frame descriptor set layout");
    }

    // Set 1 of the geometry pipeline. Five PBR textures (baseColor, metallicRoughness, normal,
    // occlusion, emissive). One set per material; bound per-draw.
    const int materialSetCount = 5;
    
    // Writes the per-frame view+proj into GeometryUniformBuffers[frameIndex].
    // Called once per frame in DrawFrame; per-draw model matrix is pushed via PbrPushConstants.
    public void UpdateUbo(uint frameIndex, Camera camera)
    {
        GeometryUBO ubo = new();
        if (camera != null)
        {
            ubo.proj = camera.GetProjectionMatrix((float)Renderer.swapChainExtent.Width / Renderer.swapChainExtent.Height, 0.1f, 100.0f);
            ubo.view = camera.GetViewMatrix();
            ubo.proj.M22 *= -1; // Vulkan clip space has Y down
        }
        else
        {
            ubo.view = Matrix4x4.CreateLookAt(new Vector3(2, 2, 2), new Vector3(0, 0, 0), new Vector3(0, 0, 1));
            ubo.proj = Matrix4x4.CreatePerspectiveFieldOfView((float)(45 * Math.PI / 180),
                (float)Renderer.swapChainExtent.Width / Renderer.swapChainExtent.Height, 0.1f, 100.0f);
            ubo.proj.M22 *= -1; // flip Y for Vulkan clip space
        }

        void* data = GeometryUniformBuffers[frameIndex].mapped;
        new Span<GeometryUBO>(data, 1).Fill(ubo);
    }
}