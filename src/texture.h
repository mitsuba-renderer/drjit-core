#pragma once

#include "internal.h"

/// Encodes information about a TexLookup node
struct TexLookupData {
    uint32_t active;

    TexLookupData(uint32_t active) : active(active) {
        jitc_var_inc_ref(active);
    }

    ~TexLookupData() {
        if (active)
            jitc_var_dec_ref(active);
    }
};
