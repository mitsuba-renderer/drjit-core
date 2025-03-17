#include "common.h"

// Payload for IR nodes that encode a ray tracing operation
struct TraceData {
    std::vector<uint32_t> indices;
    bool shadow_ray;

    ~TraceData() {
        for (uint32_t index: indices)
            jitc_var_dec_ref(index);
    }
};
