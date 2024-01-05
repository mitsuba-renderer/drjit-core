#include "common.h"

struct TraceData {
    std::vector<uint32_t> indices;

    ~TraceData() {
        for (uint32_t index: indices)
            jitc_var_dec_ref(index);
    }
};
