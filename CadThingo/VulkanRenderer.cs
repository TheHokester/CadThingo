global using Semaphore = Silk.NET.Vulkan.Semaphore;
global using ImageSharp = SixLabors.ImageSharp;
using System.Numerics;
using CadThingo.GraphicsPipeline;
using Silk.NET.Windowing;
using Silk.NET.Vulkan;

using Silk.NET.Maths;
using Silk.NET.Vulkan.Extensions.KHR;

namespace CadThingo;

public class VulkanRenderer
{
    
    private static IWindow window;
    private bool FrameBufferResized = false;

    public VulkanRenderer(IWindow _window)
    {
        window = _window;
        vk = Globals.vk;
        ctx = new VulkanContext();
        _Instance = new VulkanInstance(ctx);
        _Device = new VulkanDevice(ctx);
        _SwapChain = new VulkanSwapchain(ctx);
        _Pipeline = new VulkanPipeline(ctx);
        _Commands = new VulkanCommands(ctx);
        _Buffers = new VkBuffers(ctx);
        _Texturing = new Texturing(ctx);
        ctx.Vertices = new VkVertex[]
        {
            new VkVertex { pos = new Vector2(-0.5f, -0.5f), color = new Vector3(1.0f, 0.0f, 0.0f), uv = new Vector2(1.0f, 0.0f)},
            new VkVertex { pos = new Vector2(0.5f, -0.5f), color = new Vector3(0.0f, 1.0f, 0.0f), uv = new Vector2(0.0f, 0.0f) },
            new VkVertex { pos = new Vector2(0.5f, 0.5f), color = new Vector3(0.0f, 0.0f, 1.0f), uv = new Vector2(0.0f, 1.0f)},
            new VkVertex { pos = new Vector2(-0.5f, 0.5f), color = new Vector3(1.0f, 1.0f, 1.0f), uv = new Vector2(1.0f, 1.0f) }
        };
        ctx.Indices = [0, 1, 2, 2, 3, 0];
    }
        
    
    
    private static Vk vk;
    private static VulkanContext ctx;
    public static VulkanInstance _Instance;
    public static VulkanDevice _Device;
    public static VulkanSwapchain _SwapChain;
    public static VulkanPipeline _Pipeline;
    public static VulkanCommands _Commands;
    public static VkBuffers _Buffers;
    public static Texturing _Texturing;
    
    
    public unsafe void InitVulkan()
    {
        _Instance.CreateInstance(out ctx.EnableValidation);
        _Instance.SetupDebugMessenger(ctx.EnableValidation);
        _Instance.CreateSurface();
        _Device.PickPhysicalDevice();
        _Device.CreateLogicalDevice();
        _SwapChain.CreateSwapChain();
        _SwapChain.CreateImageViews();
        _Pipeline.CreateRenderPass();
        _Pipeline.CreateDescriptorSetLayout();
        _Pipeline.CreateGraphicsPipeline();
        _SwapChain.CreateFrameBuffer();
        _Commands.CreateCommandPool();
        _Texturing.CreateTextureImage();
        _Texturing.CreateTextureImageView();
        _Texturing.CreateTextureSampler();
        _Buffers.CreateVertexBuffers();
        _Buffers.CreateIndexBuffer();
        _Buffers.CreateUniformBuffers();
        _Pipeline.CreateDescriptorPool();
        _Pipeline.CreateDescriptorSets();
        _Commands.CreateCommandBuffers();
        CreateSyncObjects();
        Console.WriteLine("Vulkan initialized");
    }
    public void MainLoop()
    {
        window!.Render += DrawFrame;
        window.Run();
        vk!.DeviceWaitIdle(ctx.Device);
    }
    public unsafe void OnClose()
    {
        _SwapChain.CleanupSwapChain();
        
        vk!.DestroySampler(ctx.Device, ctx.TextureSampler, null);
        vk!.DestroyImageView(ctx.Device, ctx.TextureImageView, null);
        vk!.DestroyImage(ctx.Device, ctx.TextureImage, null);
        vk!.FreeMemory(ctx.Device, ctx.TextureImageMemory, null);
        
        for (var i = 0; i < App.MAX_FRAMES_IN_FLIGHT; i++)
        {
            vk!.DestroyBuffer(ctx.Device, ctx.UniformBuffers![i], null);
            vk!.FreeMemory(ctx.Device, ctx.UniformBuffersMemory![i], null);
        }
        vk!.DestroyDescriptorPool(ctx.Device, ctx.DescriptorPool, null);
        vk!.DestroyDescriptorSetLayout(ctx.Device, ctx.DescriptorSetLayout, null);
        
        vk!.DestroyBuffer(ctx.Device, ctx.VertexBuffer, null);
        vk!.FreeMemory(ctx.Device, ctx.VertexBufferMemory, null);
        
        vk!.DestroyBuffer(ctx.Device, ctx.IndexBuffer, null);
        vk!.FreeMemory(ctx.Device, ctx.IndexBufferMemory, null);
        
        for (var i = 0; i < App.MAX_FRAMES_IN_FLIGHT; i++)
        {
            vk!.DestroySemaphore(ctx.Device, ctx.RenderFinishedSemaphores![i], null);
            vk!.DestroySemaphore(ctx.Device, ctx.ImageAvailableSemaphores![i], null);
            vk!.DestroyFence(ctx.Device, ctx.InFlightFences![i], null);
        }
        
        vk!.DestroyCommandPool(ctx.Device, ctx.CommandPool, null);
        vk.DestroyDevice(ctx.Device, null);
        
        if(ctx.EnableValidation)
            ctx.DebugUtils!.DestroyDebugUtilsMessenger(ctx.Instance, ctx.DebugMessenger, null);
        
        ctx.KhrSurface!.DestroySurface(ctx.Instance, ctx.Surface, null);
        vk.DestroyInstance(ctx.Instance, null);
        vk.Dispose();
        window?.Dispose();
        
    }
    
    
    
    
    private unsafe void CreateSyncObjects()
    {
        ctx.ImageAvailableSemaphores = new Semaphore[App.MAX_FRAMES_IN_FLIGHT];
        ctx.RenderFinishedSemaphores = new Semaphore[App.MAX_FRAMES_IN_FLIGHT];
        ctx.InFlightFences = new Fence[App.MAX_FRAMES_IN_FLIGHT];
        ctx.ImagesInFlight = new Fence[ctx.SwapChainImages!.Length];
        
        SemaphoreCreateInfo semaphoreCreateInfo = new()
        {
            SType = StructureType.SemaphoreCreateInfo,
        };
        FenceCreateInfo fenceCreateInfo = new()
        {
            SType = StructureType.FenceCreateInfo,
            Flags = FenceCreateFlags.SignaledBit
        };

        for (var i = 0; i < App.MAX_FRAMES_IN_FLIGHT; i++)
        {
            if (vk!.CreateSemaphore(ctx.Device, &semaphoreCreateInfo, null, out ctx.ImageAvailableSemaphores[i]) != Result.Success ||
                vk!.CreateSemaphore(ctx.Device, &semaphoreCreateInfo, null, out ctx.RenderFinishedSemaphores[i]) != Result.Success ||
                vk!.CreateFence(ctx.Device, &fenceCreateInfo, null, out ctx.InFlightFences[i]) != Result.Success)
            {
                throw new Exception("Failed to create synchronization objects for a frame");
            }
        }
    }

    private unsafe void DrawFrame(double dt)
    {
        vk!.WaitForFences(ctx.Device, 1, in ctx.InFlightFences![ctx.CurrentFrame], true, ulong.MaxValue);

        uint imageIndex = 0;
        // --- acquire next image ---
        //verify that the image has not changed
        var result = ctx.KhrSwapChain!.AcquireNextImage(ctx.Device, ctx.SwapChain, ulong.MaxValue,
            ctx.ImageAvailableSemaphores![ctx.CurrentFrame], default, &imageIndex);
        if (result == Result.ErrorOutOfDateKhr)//recreates swapchain if its out of date
        {
            _SwapChain.RecreateSwapChain(window, _Pipeline, this);
            return;
        } else if (result != Result.Success && result != Result.SuboptimalKhr)
        {
            throw new Exception("Failed to acquire swap chain image");
        }
        
        
        
        if (ctx.ImagesInFlight![imageIndex].Handle != 0)
        {
            vk!.WaitForFences(ctx.Device, 1, in ctx.ImagesInFlight[imageIndex], true, ulong.MaxValue);
        }

        ctx.ImagesInFlight[imageIndex] = ctx.InFlightFences[ctx.CurrentFrame];
        
        _Buffers.UpdateUniformBuffers(imageIndex);
        
        SubmitInfo submitInfo = new()
        {
            SType = StructureType.SubmitInfo,
        };
        var waitSemaphores = stackalloc[] { ctx.ImageAvailableSemaphores[ctx.CurrentFrame] };
        var waitStages = stackalloc[] { PipelineStageFlags.ColorAttachmentOutputBit };
        
        var buffer = ctx.CommandBuffers![imageIndex];

        submitInfo = submitInfo with
        {
            WaitSemaphoreCount = 1,
            PWaitSemaphores = waitSemaphores,
            PWaitDstStageMask = waitStages,

            CommandBufferCount = 1,
            PCommandBuffers = &buffer
        };

        var signalSemaphores = stackalloc[] { ctx.RenderFinishedSemaphores![ctx.CurrentFrame] };
        submitInfo = submitInfo with
        {
            SignalSemaphoreCount = 1,
            PSignalSemaphores = signalSemaphores
        };
        
        vk!.ResetFences(ctx.Device, 1, in ctx.InFlightFences![ctx.CurrentFrame]);

        if (vk!.QueueSubmit(ctx.GraphicsQueue, 1, &submitInfo, ctx.InFlightFences[ctx.CurrentFrame]) != Result.Success) 
        {
            throw new Exception("Failed to submit draw command buffer");
        }

        var swapChains = stackalloc[] { ctx.SwapChain };
        PresentInfoKHR presentInfo = new()
        {
            SType = StructureType.PresentInfoKhr,

            WaitSemaphoreCount = 1,
            PWaitSemaphores = signalSemaphores,

            SwapchainCount = 1,
            PSwapchains = swapChains,

            PImageIndices = &imageIndex,

        };
        //verify that the image has not changed if it has update the swapchain
        result = ctx.KhrSwapChain.QueuePresent(ctx.PresentQueue, &presentInfo);
        if (result == Result.ErrorOutOfDateKhr || result == Result.SuboptimalKhr || FrameBufferResized)
        {
            FrameBufferResized = false;
            _SwapChain.RecreateSwapChain(window, _Pipeline, this);
        }else if (result != Result.Success)
            throw new Exception("Failed to present swap chain image");
        
        
        ctx.CurrentFrame = (ctx.CurrentFrame + 1) % App.MAX_FRAMES_IN_FLIGHT;
    }

    public void FramebufferResizeCallback(Vector2D<int> obj)
    {
        FrameBufferResized = true;
    }
}
