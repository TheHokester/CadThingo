using Silk.NET.Vulkan;
using Silk.NET.Core.Native;
using Silk.NET.Core;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Numerics;


namespace CadThingo;

public class VulkanCommands(VulkanContext ctx)
{   
    Vk? vk = Globals.vk;
    
    
    public void CreateCommandPool()
    {
        unsafe
        {
            var queueFamilyIndices = VulkanDevice.FindQueueFamilies(ctx.PhysicalDevice, ctx.Surface, ctx.KhrSurface);

            CommandPoolCreateInfo commandPoolInfo = new()
            {
                SType = StructureType.CommandPoolCreateInfo,
                QueueFamilyIndex = queueFamilyIndices.graphicsFamily!.Value,
                Flags = CommandPoolCreateFlags.ResetCommandBufferBit
            };
            if(vk!.CreateCommandPool(ctx.Device, &commandPoolInfo, null, out ctx.CommandPool) != Result.Success)
                throw new Exception("Failed to create command pool");
        }
    }
    
    
    public unsafe void CreateCommandBuffers()
    {
        ctx.CommandBuffers = new CommandBuffer[ctx.SwapChainImages!.Length];
        CommandBufferAllocateInfo allocateInfo = new()
        {
            SType = StructureType.CommandBufferAllocateInfo,
            CommandPool = ctx.CommandPool,
            Level = CommandBufferLevel.Primary,
            CommandBufferCount = 1
        };
        

        for (int i = 0; i < ctx.CommandBuffers.Length; i++)
        {
            if (vk!.AllocateCommandBuffers(ctx.Device, &allocateInfo, out ctx.CommandBuffers[i]) != Result.Success)
            {
                throw new Exception("Failed to allocate command buffers");
            } 
        }

        for (var i = 0; i < ctx.CommandBuffers.Length; i++)
        {
            CommandBufferBeginInfo beginInfo = new()
            {
                SType = StructureType.CommandBufferBeginInfo,
            };
            if (vk!.BeginCommandBuffer(ctx.CommandBuffers[i], &beginInfo) != Result.Success)
            {
                throw new Exception("Failed to begin recording command buffer");
            }

            RenderPassBeginInfo renderPassInfo = new()
            {
                SType = StructureType.RenderPassBeginInfo,
                RenderPass = ctx.RenderPass,
                Framebuffer = ctx.SwapChainFramebuffers[i],
                RenderArea =
                {
                    Offset = { X = 0, Y = 0 },
                    Extent = ctx.SwapChainExtent
                }
            };
            var clearValues = new ClearValue[2];
            clearValues[0].Color = new() { Float32_0 = 1.0f, Float32_1 = 1.0f, Float32_2 = 1f, Float32_3 = 1.0f };
            clearValues[1].DepthStencil = new() { Depth = 1.0f, Stencil = 0 };

            fixed (ClearValue* clearValuesPtr = clearValues)
            {
                renderPassInfo.ClearValueCount = (uint)clearValues.Length;
                renderPassInfo.PClearValues = clearValuesPtr;
            }
            
            vk!.CmdBeginRenderPass(ctx.CommandBuffers![i], &renderPassInfo, SubpassContents.Inline);
            
            vk!.CmdBindPipeline(ctx.CommandBuffers[i], PipelineBindPoint.Graphics, ctx.Pipeline);
            var vertexBuffers = new Buffer[] { ctx.VertexBuffer };
            var offsets = new ulong[] { 0 };
            fixed(ulong* offsetsPtr = offsets)
            fixed (Buffer* vertexBuffersPtr = vertexBuffers)
            {
                vk!.CmdBindVertexBuffers(ctx.CommandBuffers[i], 0, 1, vertexBuffersPtr, offsetsPtr);
                vk!.CmdBindIndexBuffer(ctx.CommandBuffers[i], ctx.IndexBuffer, 0, IndexType.Uint32);
            }
            
            vk!.CmdBindDescriptorSets(ctx.CommandBuffers[i], PipelineBindPoint.Graphics, ctx.PipelineLayout, 0, 1, in ctx.DescriptorSets![i], 0, null);
            vk!.CmdDrawIndexed(ctx.CommandBuffers[i], (uint)ctx.Indices.Length, 1, 0, 0, 0);
            
            vk!.CmdEndRenderPass(ctx.CommandBuffers[i]);
            
            if(vk!.EndCommandBuffer(ctx.CommandBuffers[i]) != Result.Success)
                throw new Exception("Failed to end recording command buffer");
        }
    }

    public unsafe CommandBuffer BeginSingleTimeCommands()
    {
        CommandBufferAllocateInfo allocInfo = new()
        {
            SType = StructureType.CommandBufferAllocateInfo,
            Level = CommandBufferLevel.Primary,
            CommandPool = ctx.CommandPool,
            CommandBufferCount = 1
        };
        CommandBuffer commandBuffer;
        vk!.AllocateCommandBuffers(ctx.Device, &allocInfo, &commandBuffer);

        CommandBufferBeginInfo beginInfo = new()
        {
            SType = StructureType.CommandBufferBeginInfo,
            Flags = CommandBufferUsageFlags.OneTimeSubmitBit
        };
        
        vk!.BeginCommandBuffer(commandBuffer, &beginInfo);

        return commandBuffer;
    }

    public unsafe void EndSingleTimeCommands(CommandBuffer commandBuffer)
    {
        vk!.EndCommandBuffer(commandBuffer);
        SubmitInfo submitInfo = new()
        {
            SType = StructureType.SubmitInfo,
            CommandBufferCount = 1,
            PCommandBuffers = &commandBuffer

        };
        vk!.QueueSubmit(ctx.GraphicsQueue, 1, &submitInfo, default );
        vk!.QueueWaitIdle(ctx.GraphicsQueue);
        
        vk!.FreeCommandBuffers(ctx.Device, ctx.CommandPool, 1, &commandBuffer);

    }

    public unsafe void copyBuffer(Buffer srcBuffer, Buffer dstBuffer, ulong size)
    {
        CommandBuffer commandBuffer = BeginSingleTimeCommands();

        BufferCopy copyRegion = new()
        {
            Size = size
        };
        vk!.CmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
        
        EndSingleTimeCommands(commandBuffer);
    }
}