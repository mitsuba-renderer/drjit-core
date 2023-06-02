/*
    src/io.h -- Disk cache for LLVM/CUDA kernels

    Copyright (c) 2023 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "hash.h"
#include "core.h"

// using LLVMKernelFunction = void (*)(uint64_t start, uint64_t end, void **ptr);


// LZ4 compression dictionary
static const int jitc_lz4_dict_size = 65536;
extern char jitc_lz4_dict[];

/// Initialize dictionary
extern void jitc_lz4_init();

extern bool jitc_kernel_load(const char *source, uint32_t source_size,
                             JitBackend backend, XXH128_hash_t hash,
                             Kernel &kernel);

extern bool jitc_kernel_write(const char *source, uint32_t source_size,
                              JitBackend backend, XXH128_hash_t hash,
                              const Kernel &kernel);

