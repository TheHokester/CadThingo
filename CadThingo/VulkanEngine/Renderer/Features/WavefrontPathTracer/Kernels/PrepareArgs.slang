import WavefrontBindings;
import PTUtils;

struct ArgsPc {
    uint countIndex;
    uint argsByteOffset;
    uint resetMask;
    uint copyToNextRay;   
}
[[vk::push_constant]] ArgsPc args;

[shader("compute")]
[numthreads(1,1,1)]
void main() {
    if(args.copyToNextRay != 0) {
        counters[RAY_COUNT] = counters[NEXT_RAY_COUNT];
        counters[NEXT_RAY_COUNT] = 0;
    }
    uint n = counters[args.countIndex];
    
    dispatchArgs[args.argsByteOffset/4 + 0] = (n + 63u) / 64u;
    dispatchArgs[args.argsByteOffset/4 + 1] = 1;
    dispatchArgs[args.argsByteOffset/4 + 2] = 1;
    
    if((args.resetMask & 1u) != 0u) counters[SHADE_COUNT_0] = 0u;
    if((args.resetMask & 2u) != 0u) {
        counters[SHADOW_COUNT] = 0u;
        counters[NEXT_RAY_COUNT] = 0u;
    }
}