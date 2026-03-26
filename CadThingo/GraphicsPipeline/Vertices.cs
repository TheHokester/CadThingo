global using Buffer = Silk.NET.Vulkan.Buffer;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using CadThingo;

using Silk.NET.Vulkan;

namespace CadThingo.GraphicsPipeline;

public struct VkVertex
{
    
    public Vector2 pos;
    public Vector3 color;
    
    
    public static VertexInputBindingDescription GetBindingDescription()
    {
        VertexInputBindingDescription bindingDescription = new()
        {
            Binding = 0,
            Stride = (uint)Unsafe.SizeOf<VkVertex>(),
            InputRate = VertexInputRate.Vertex
        };
        return bindingDescription;
    }
    
    public static VertexInputAttributeDescription[] GetAttributeDescriptions()
    {
        var attributeDescriptions = new[]
        {
            new VertexInputAttributeDescription()
            {
                Binding = 0,
                Location = 0,
                Format = Format.R32G32Sfloat,
                Offset = (uint)Marshal.OffsetOf<VkVertex>(nameof(pos))
            },
            new VertexInputAttributeDescription()
            {
                Binding = 0,
                Location = 1,
                Format = Format.R32G32B32Sfloat,
                Offset = (uint)Marshal.OffsetOf<VkVertex>(nameof(color))
            }
        };
        
        return attributeDescriptions;
    }
}

public class VkBuffers(VulkanContext ctx)
{
    Vk? vk = Globals.vk;

    private unsafe void CreateBuffer(ulong size, BufferUsageFlags usage, MemoryPropertyFlags properties, Buffer* buffer,
        DeviceMemory* bufferMemory)
    {
        BufferCreateInfo bufferInfo = new BufferCreateInfo()
        {
            SType = StructureType.BufferCreateInfo,
            Size = size,
            Usage = usage,
            SharingMode = SharingMode.Exclusive,
        };
        
        if (vk!.CreateBuffer(ctx.Device, &bufferInfo, null, buffer) != Result.Success)
        {
            throw new Exception("Failed to create vertex buffer");
        }
        MemoryRequirements memReqs = new();
        vk!.GetBufferMemoryRequirements(ctx.Device, *buffer, &memReqs);
        
        MemoryAllocateInfo allocInfo = new()
        {
            SType = StructureType.MemoryAllocateInfo,
            AllocationSize = memReqs.Size,
            MemoryTypeIndex = FindMemoryType(memReqs.MemoryTypeBits, properties)
        };

        if (vk!.AllocateMemory(ctx.Device, &allocInfo, null, bufferMemory) != Result.Success)
        {
            throw new Exception("Failed to allocate vertex buffer memory");
        }
        
        vk!.BindBufferMemory(ctx.Device, *buffer, *bufferMemory, 0);
    }
    public unsafe void CreateVertexBuffers()
    {
        var size = (ulong)(sizeof(VkVertex) * ctx.Vertices.Length);
        
        Buffer stagingBuffer = new();
        DeviceMemory stagingBufferMemory = new();
            
        CreateBuffer(size, BufferUsageFlags.TransferSrcBit, MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit, &stagingBuffer, &stagingBufferMemory);
        void* data;
        vk!.MapMemory(ctx.Device, stagingBufferMemory, 0, size, 0, &data);
        ctx.Vertices.AsSpan().CopyTo(new Span<VkVertex>(data, ctx.Vertices.Length));
        vk!.UnmapMemory(ctx.Device, stagingBufferMemory);
        
        
        fixed(DeviceMemory* vertexBfrMemPtr = &ctx.VertexBufferMemory)
        fixed(Buffer* vertexBfrPtr = &ctx.VertexBuffer)
            CreateBuffer(size, BufferUsageFlags.VertexBufferBit, MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit, vertexBfrPtr, vertexBfrMemPtr);
        
        CopyBuffer(stagingBuffer, ctx.VertexBuffer, size);
        
        vk!.DestroyBuffer(ctx.Device, stagingBuffer, null);
        vk!.FreeMemory(ctx.Device, stagingBufferMemory, null);
        
    }

    public unsafe void CreateIndexBuffer()
    {
        ulong size = (ulong)(sizeof(uint) * ctx.Indices.Length);

        Buffer stagingBuffer = new();
        DeviceMemory stagingBufferMemory = new();
        
        CreateBuffer(size, BufferUsageFlags.TransferSrcBit, MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit, &stagingBuffer, &stagingBufferMemory);

        void* data;
        vk!.MapMemory(ctx.Device, stagingBufferMemory, 0, size, 0, &data);
        ctx.Indices.AsSpan().CopyTo(new Span<uint>(data, ctx.Indices.Length));
        vk!.UnmapMemory(ctx.Device, stagingBufferMemory);
        
        fixed(DeviceMemory* indexBfrMemPtr = &ctx.IndexBufferMemory)
        fixed(Buffer* indexBfrPtr = &ctx.IndexBuffer)
            CreateBuffer(size, BufferUsageFlags.IndexBufferBit, MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit, indexBfrPtr, indexBfrMemPtr);
        
        CopyBuffer(stagingBuffer, ctx.IndexBuffer, size);
        
        vk!.DestroyBuffer(ctx.Device, stagingBuffer, null);
        vk!.FreeMemory(ctx.Device, stagingBufferMemory, null);
    }

    private unsafe void CopyBuffer(Buffer srcBuffer, Buffer dstBuffer, ulong size)
    {
        CommandBufferAllocateInfo allocInfo = new()
        {
            SType = StructureType.CommandBufferAllocateInfo,
            CommandPool = ctx.CommandPool,
            Level = CommandBufferLevel.Primary,
            CommandBufferCount = 1
        };
        CommandBuffer commandBuffer = new();
        vk!.AllocateCommandBuffers(ctx.Device, &allocInfo, &commandBuffer);

        CommandBufferBeginInfo beginInfo = new()
        {
            SType = StructureType.CommandBufferBeginInfo,
            Flags = CommandBufferUsageFlags.OneTimeSubmitBit
        };
        vk!.BeginCommandBuffer(commandBuffer, &beginInfo);

        BufferCopy copyRegion = new BufferCopy()
        {
            SrcOffset = 0,
            DstOffset = 0,
            Size = size
        };
        vk!.CmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

        vk!.EndCommandBuffer(commandBuffer);
        SubmitInfo submitInfo = new()
        {
            SType = StructureType.SubmitInfo,
            CommandBufferCount = 1,
            PCommandBuffers = &commandBuffer
        };
        //TODO: potentially add fence to wait on instead to optimise, aswell as potentially find a transfer queue in future to save resources in ctx.GraphicsQueue
        vk!.QueueSubmit(ctx.GraphicsQueue, 1, &submitInfo, default);
        vk!.QueueWaitIdle(ctx.GraphicsQueue);
        
        vk!.FreeCommandBuffers(ctx.Device, ctx.CommandPool, 1, &commandBuffer);
    }
    
    private uint FindMemoryType(uint typeFilter, MemoryPropertyFlags properties)
    {
        vk!.GetPhysicalDeviceMemoryProperties(ctx.PhysicalDevice, out PhysicalDeviceMemoryProperties memProperties);

        for (var i = 0; i < memProperties.MemoryTypeCount; i++)
        {
            if ((typeFilter & (1 << i)) != 0 && (memProperties.MemoryTypes[i].PropertyFlags & properties) == properties)
            {
                return (uint)i;
            }
        }

        throw new Exception("failed to find suitable memory type!");
    }
}