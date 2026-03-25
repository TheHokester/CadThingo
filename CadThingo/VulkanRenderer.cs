global using Semaphore = Silk.NET.Vulkan.Semaphore;
using Silk.NET.Windowing;
using Silk.NET.Vulkan;

using CadThingo.GraphicsPipeline;
using Silk.NET.Vulkan.Extensions.KHR;

namespace CadThingo;


public class VulkanRenderer
{
    private static IWindow window;
    
    public VulkanRenderer(IWindow _window)
        => window = _window;
    
    
    private static Vk? vk;
    private VulkanContext ctx;
    private VulkanInstance VKInstance;
    private VulkanDevice VKDevice;
    private VulkanSwapchain VKSwapChain;
    private VulkanPipeline VKPipeline;
    
    
    public unsafe void InitVulkan()
    {
        vk = Globals.vk;
        ctx = new VulkanContext();
        VKInstance = new VulkanInstance(ctx);
        VKDevice = new VulkanDevice(ctx);
        VKSwapChain = new VulkanSwapchain(ctx);
        VKPipeline = new VulkanPipeline(ctx);
        
        VKInstance.CreateInstance(out ctx.EnableValidation);
        VKInstance.SetupDebugMessenger(ctx.EnableValidation);
        // --- surface ---
        VKInstance.CreateSurface();
        // --- physical device ---
        VKDevice.PickPhysicalDevice();
        // --- logical device ---
        VKDevice.CreateLogicalDevice();
        // --- swap chain ---
        VKSwapChain.CreateSwapChain();
        VKSwapChain.CreateImageViews();
        // --- create pipeline ---
        VKPipeline.CreateRenderPass();
        VKPipeline.CreateGraphicsPipeline();
        VKSwapChain.CreateFrameBuffer();
        // --- command rendering pool ---
        CreateCommandPool();
        CreateCommandBuffers();
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
        for (int i = 0; i < App.MAX_FRAMES_IN_FLIGHT; i++)
        {
            vk!.DestroySemaphore(ctx.Device, ctx.RenderFinishedSemaphores![i], null);
            vk!.DestroySemaphore(ctx.Device, ctx.ImageAvailableSemaphores![i], null);
            vk!.DestroyFence(ctx.Device, ctx.InFlightFences![i], null);
        }
        vk!.DestroyCommandPool(ctx.Device, ctx.CommandPool, null);
        VKSwapChain.DestroyFrameBuffers();
        vk!.DestroyPipeline(ctx.Device, ctx.Pipeline, null);
        vk.DestroyPipelineLayout(ctx.Device, ctx.PipelineLayout, null);
        vk.DestroyRenderPass(ctx.Device, ctx.RenderPass, null);
        VKSwapChain.DestroyImageViews();
        ctx.KhrSwapChain!.DestroySwapchain(ctx.Device, ctx.SwapChain, null);
        vk.DestroyDevice(ctx.Device, null);
        if(ctx.EnableValidation)
            ctx.DebugUtils!.DestroyDebugUtilsMessenger(ctx.Instance, ctx.DebugMessenger, null);
        
        ctx.KhrSurface!.DestroySurface(ctx.Instance, ctx.Surface, null);
        vk.DestroyInstance(ctx.Instance, null);
        vk.Dispose();
        window?.Dispose();
        
    }
    
    private void CreateCommandPool()
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

    private unsafe void CreateCommandBuffers()
    {
        ctx.CommandBuffers = new CommandBuffer[ctx.SwapChainImages!.Length];
        CommandBufferAllocateInfo allocateInfo = new()
        {
            SType = StructureType.CommandBufferAllocateInfo,
            CommandPool = ctx.CommandPool,
            Level = CommandBufferLevel.Primary,
            CommandBufferCount = 1
        };
        // fixed (CommandBuffer* commandBuffers = ctx.CommandBuffers)
        // {
        //     
        //     {
        //         if (vk!.AllocateCommandBuffers(ctx.Device, &allocateInfo, commandBuffers) != Result.Success)
        //         {
        //             throw new Exception("Failed to allocate command buffers");
        //         }
        //     }
        // }

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
            ClearValue clearColor = new()
            {
                Color = new() { Float32_0 = 0.0f, Float32_1 = 0.0f, Float32_2 = 0f, Float32_3 = 1.0f }
            };

            renderPassInfo.ClearValueCount = 1;
            renderPassInfo.PClearValues = &clearColor;
            
            vk!.CmdBeginRenderPass(ctx.CommandBuffers![i], &renderPassInfo, SubpassContents.Inline);
            
            vk!.CmdBindPipeline(ctx.CommandBuffers[i], PipelineBindPoint.Graphics, ctx.Pipeline);
            
            vk!.CmdDraw(ctx.CommandBuffers[i], 3, 1, 0, 0);
            
            vk!.CmdEndRenderPass(ctx.CommandBuffers[i]);
            
            if(vk!.EndCommandBuffer(ctx.CommandBuffers[i]) != Result.Success)
                throw new Exception("Failed to end recording command buffer");
        }
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
        ctx.KhrSwapChain!.AcquireNextImage(ctx.Device, ctx.SwapChain, ulong.MaxValue,
            ctx.ImageAvailableSemaphores![ctx.CurrentFrame], default, &imageIndex);

        if (ctx.ImagesInFlight![imageIndex].Handle != 0)
        {
            vk!.WaitForFences(ctx.Device, 1, in ctx.ImagesInFlight[imageIndex], true, ulong.MaxValue);
        }

        ctx.ImagesInFlight[imageIndex] = ctx.InFlightFences[ctx.CurrentFrame];

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
        ctx.KhrSwapChain.QueuePresent(ctx.PresentQueue, &presentInfo);
        
        ctx.CurrentFrame = (ctx.CurrentFrame + 1) % App.MAX_FRAMES_IN_FLIGHT;
    }
}
