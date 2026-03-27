global using Buffer = Silk.NET.Vulkan.Buffer;
global using VkImage = Silk.NET.Vulkan.Image;
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
    public Vector2 uv; //texCoord
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
            },
            new VertexInputAttributeDescription()
            {
                Binding = 0,
                Location = 2,
                Format = Format.R32G32Sfloat,
                Offset = (uint)Marshal.OffsetOf<VkVertex>(nameof(uv))
            }
        };
        
        return attributeDescriptions;
    }
    
}
public struct UniformBufferObject
{
    public Matrix4x4 Model;
    public Matrix4x4 View;  
    public Matrix4x4 Proj; 
}

public class VkBuffers(VulkanContext ctx)
{
    Vk? vk = Globals.vk;

    public unsafe void CreateBuffer(ulong size, BufferUsageFlags usage, MemoryPropertyFlags properties, Buffer* buffer,
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
            CreateBuffer(size, BufferUsageFlags.VertexBufferBit | BufferUsageFlags.TransferDstBit,
                MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit, vertexBfrPtr, vertexBfrMemPtr);
        
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
            CreateBuffer(size, BufferUsageFlags.TransferDstBit | BufferUsageFlags.IndexBufferBit, 
                MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit, indexBfrPtr, indexBfrMemPtr);
        
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

    public uint FindMemoryType(uint typeFilter, MemoryPropertyFlags properties)
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

    public unsafe void CreateUniformBuffers()
    {
        var bufferSize = (uint)sizeof(UniformBufferObject);

        ctx.UniformBuffers = new Buffer[ctx.SwapChainImages!.Length];
        ctx.UniformBuffersMemory = new DeviceMemory[ctx.SwapChainImages!.Length];
        ctx.UniformBuffersMemoryPtrs = new void*[ctx.SwapChainImages!.Length];
        for (int i = 0; i < ctx.SwapChainImages!.Length; i++)
        {
            fixed(Buffer* bufferPtr = &ctx.UniformBuffers[i])
            fixed(DeviceMemory* bufferMemoryPtr = &ctx.UniformBuffersMemory[i])
                CreateBuffer(bufferSize, BufferUsageFlags.UniformBufferBit, MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit,bufferPtr, bufferMemoryPtr);
                
            void* mappedMemory = null;
            vk!.MapMemory(ctx.Device, ctx.UniformBuffersMemory[i], 0, bufferSize, 0, &mappedMemory);
            ctx.UniformBuffersMemoryPtrs[i] = mappedMemory;
        }
    }

    public unsafe void UpdateUniformBuffers(uint currentFrame)
    {
        //silk window has time information so skipping this
        var time = (float)App.window!.Time;

        UniformBufferObject ubo = new()
        {
            Model = Matrix4x4.Identity * Matrix4x4.CreateFromAxisAngle(new Vector3(0, 0, 1), time * Radians(90f)),
            View = Matrix4x4.CreateLookAt(new Vector3(2, 2, 2), new Vector3(0, 0, 0), new Vector3(0, 0, 1)),
            Proj = Matrix4x4.CreatePerspectiveFieldOfView(Radians(45f),
                (float)ctx.SwapChainExtent.Width / ctx.SwapChainExtent.Height, 0.1f, 10.0f)
        };
        
        
        ubo.Proj.M22 *= -1;
        var data = ctx.UniformBuffersMemoryPtrs![currentFrame];
        new Span<UniformBufferObject>(data, 1).Fill(ubo);
        
        static float Radians(float angle) => angle * MathF.PI / 180f;
    }
}