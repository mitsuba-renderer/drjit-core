/*
    kernels/misc.cuh -- Miscellaneous CUDA kernels

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "common.h"

KERNEL void poke_u8(uint8_t *out, uint8_t value) {
    *out = value;
}

KERNEL void poke_u16(uint16_t *out, uint16_t value) {
    *out = value;
}

KERNEL void poke_u32(uint32_t *out, uint32_t value) {
    *out = value;
}

KERNEL void poke_u64(uint64_t *out, uint64_t value) {
    *out = value;
}

KERNEL void fill_64(uint64_t *out, uint32_t size, uint64_t value) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += blockDim.x * gridDim.x)
        out[i] = value;
}

struct VCallDataRecord {
    int32_t size;
    uint32_t offset;
    const void *src;
};

KERNEL void aggregate(void *out, const VCallDataRecord *rec_, uint32_t size) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    VCallDataRecord rec = rec_[idx];

    const void *src = rec.src;
    void *dst = (uint8_t *) out + rec.offset;

    switch (rec.size) {
        case  1: *(uint8_t *)  dst = (uint8_t)  (uintptr_t) src; break;
        case  2: *(uint16_t *) dst = (uint16_t) (uintptr_t) src; break;
        case  4: *(uint32_t *) dst = (uint32_t) (uintptr_t) src; break;
        case  8: *(uint64_t *) dst = (uint64_t) (uintptr_t) src; break;
        case -1: *(uint8_t *)  dst = *(uint8_t *)  src; break;
        case -2: *(uint16_t *) dst = *(uint16_t *) src; break;
        case -4: *(uint32_t *) dst = *(uint32_t *) src; break;
        case -8: *(uint64_t *) dst = *(uint64_t *) src; break;
    }
}
