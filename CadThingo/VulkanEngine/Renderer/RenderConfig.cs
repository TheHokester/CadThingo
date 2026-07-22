namespace CadThingo.VulkanEngine.Renderer;

public static class RenderConfig
{
    
    
    public const int MAX_CONCURRENT_FRAMES = 2;
    
    
    // Pipeline-wide size budgets. The actual GPU buffers these size are owned by
    // the pipelines that need them (DrawCullPipeline, LightCullPipeline,
    // PbrDeferredPipeline) or by ResourceManager (materials/instances/bindless
    // textures).
    public const uint MAX_MATERIALS         = 256;
    public const uint MAX_INSTANCES         = 4096;
    // Must match ResourceManager.MAX_BINDLESS_TEXTURES — both bound the same
    // descriptor array. 9 = worst-case textures per material post-extensions
    // (5 core + transmission + 3× clearcoat).
    public const uint MAX_BINDLESS_TEXTURES = MAX_MATERIALS * 9;

    // Lighting pipeline set 0 binding 1 — per-frame StructuredBuffer<PbrLight>.
    // Cap chosen to keep the SSBO at 64KB (1024 × 64B); raise if needed.
    public const uint MAX_LIGHTS = 1024;

    // Tiled light culling. Tile size matches the compute group
    // (16×16 threads collaborating on one tile). MAX_LIGHTS_PER_TILE bounds the
    // worst-case per-tile slot count; lights past this are dropped (the cull
    // shader saturates the count).
    public const uint TILE_SIZE           = 16;
    public const uint MAX_LIGHTS_PER_TILE = 64;
    // Hard cap on tile count — covers up to 3840×2160(4K) (240*135 tiles). The actual
    // per-frame tileCountX/Y depends on swapChainExtent and is uploaded via the
    // frame UBO each frame.
    public const uint MAX_TILE_COUNT      = 240 * 135;
        
}