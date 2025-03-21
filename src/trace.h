#include "common.h"

// Payload for IR nodes that encode a ray tracing operation
struct TraceData {
    std::vector<uint32_t> indices;
#if defined(DRJIT_ENABLE_OPTIX)
    std::vector<uint32_t> hit_object_fields;
    bool invoke;
#endif

    ~TraceData() {
        for (uint32_t index: indices)
            jitc_var_dec_ref(index);
    }
};
