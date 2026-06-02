using CadThingo.VulkanEngine.Renderer;
using Silk.NET.Vulkan;
// ReSharper disable InconsistentNaming

namespace CadThingo.VulkanEngine;

/// <summary>
/// Owns the per-frame command buffers and CPU/GPU sync primitives (image-available /
/// render-finished semaphores, in-flight fences) plus the frame counters. One slot per
/// frame-in-flight, cycled by <see cref="Advance"/> at end-of-frame.
///
/// L1.4 of the renderer refactor. Depends only on <see cref="GraphicsDevice"/>. The
/// frame *body* (mode dispatch, blit, outline, ImGui) stays on the Renderer; this owns
/// the ring state and its lifetime. Disposed before the GraphicsDevice.
/// </summary>
public sealed unsafe class FrameRing : IDisposable
{
    private readonly GraphicsDevice _gfx;
    private readonly Vk vk;
    private readonly uint _maxFrames;

    private CommandBuffer[] commandBuffers           = Array.Empty<CommandBuffer>();
    private Semaphore[]     imageAvailableSemaphores  = Array.Empty<Semaphore>();
    private Semaphore[]     renderFinishedSemaphores  = Array.Empty<Semaphore>();
    private Fence[]         inFlightFences            = Array.Empty<Fence>();
    private uint  currentFrame;
    // Monotonic frame counter — incremented once per frame. Distinct from
    // currentFrame which cycles 0..maxFrames-1. Used by the probe scheduler for
    // EveryNFrames bookkeeping and capture timing.
    private ulong frameCounter;

    // Reserved for async transfer-queue uploads (timeline semaphore). Declared but
    // not yet wired — carried over from the pre-refactor Renderer.
    private Semaphore uploadsTimeline;
    private volatile uint lastTimelineValue;

    public FrameRing(GraphicsDevice gfx, uint maxFrames)
    {
        _gfx = gfx;
        vk = gfx.Vk;
        _maxFrames = maxFrames;
    }

    public CommandBuffer[] CommandBuffers           => commandBuffers;
    public Semaphore[]     ImageAvailableSemaphores => imageAvailableSemaphores;
    public Semaphore[]     RenderFinishedSemaphores => renderFinishedSemaphores;
    public Fence[]         InFlightFences           => inFlightFences;
    public uint  CurrentFrame => currentFrame;
    public ulong FrameCounter => frameCounter;

    public void CreateCommandBuffers(int count)
    {
        var device = _gfx.Device;
        commandBuffers = new CommandBuffer[count];
        CommandBufferAllocateInfo allocateInfo = new()
        {
            SType = StructureType.CommandBufferAllocateInfo,
            CommandPool = _gfx.CommandPool,
            Level = CommandBufferLevel.Primary,
            CommandBufferCount = 1
        };

        for (int i = 0; i < commandBuffers.Length; i++)
        {
            if (vk.AllocateCommandBuffers(device, &allocateInfo, out commandBuffers[i]) != Result.Success)
            {
                throw new Exception("Failed to allocate command buffers");
            }
        }
    }

    public void CreateSyncObjects()
    {
        var device = _gfx.Device;
        imageAvailableSemaphores = new Semaphore[_maxFrames];
        renderFinishedSemaphores = new Semaphore[_maxFrames];
        inFlightFences = new Fence[_maxFrames];

        SemaphoreCreateInfo semaphoreCreateInfo = new()
        {
            SType = StructureType.SemaphoreCreateInfo,
        };
        FenceCreateInfo fenceCreateInfo = new()
        {
            SType = StructureType.FenceCreateInfo,
            Flags = FenceCreateFlags.SignaledBit
        };

        for (var i = 0; i < _maxFrames; i++)
        {
            if (vk.CreateSemaphore(device, &semaphoreCreateInfo, null, out imageAvailableSemaphores[i]) != Result.Success ||
                vk.CreateSemaphore(device, &semaphoreCreateInfo, null, out renderFinishedSemaphores[i]) != Result.Success ||
                vk.CreateFence(device, &fenceCreateInfo, null, out inFlightFences[i]) != Result.Success)
            {
                throw new Exception("Failed to create synchronization objects for a frame");
            }
        }
    }

    /// <summary>Advances to the next frame-in-flight slot and bumps the monotonic
    /// frame counter. Called once at end-of-frame.</summary>
    public void Advance()
    {
        currentFrame = (currentFrame + 1) % _maxFrames;
        frameCounter++;
    }

    public void Dispose()
    {
        var device = _gfx.Device;
        for (var i = 0; i < _maxFrames; i++)
        {
            if (renderFinishedSemaphores.Length > i && renderFinishedSemaphores[i].Handle != 0)
                vk.DestroySemaphore(device, renderFinishedSemaphores[i], null);
            if (imageAvailableSemaphores.Length > i && imageAvailableSemaphores[i].Handle != 0)
                vk.DestroySemaphore(device, imageAvailableSemaphores[i], null);
            if (inFlightFences.Length > i && inFlightFences[i].Handle != 0)
                vk.DestroyFence(device, inFlightFences[i], null);
        }
    }
}