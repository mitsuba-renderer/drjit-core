/*
    src/llvm.h -- LLVM backend functionality

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <stdlib.h>
#include <stdint.h>
#include <vector>

// Forward declarations
struct Task;
struct Kernel;

/// Current top-level task in the task queue
extern Task *jitc_task;

/// Attempt to dynamically load LLVM into the process
extern bool jitc_llvm_api_init();

/// Free any resources allocated by jitc_llvm_api_init()
extern void jitc_llvm_api_shutdown();

/// Are the core parts of LLVM available
extern bool jitc_llvm_api_has_core();

/// Is the MCJIT/ORCv2-based backend available
extern bool jitc_llvm_api_has_mcjit();
extern bool jitc_llvm_api_has_orcv2();

// Which pass builder interface is available?
extern bool jitc_llvm_api_has_pb_legacy();
extern bool jitc_llvm_api_has_pb_new();

/// String describing the LLVM target
extern char *jitc_llvm_target_triple;

/// Target CPU string used by the LLVM backend
extern char *jitc_llvm_target_cpu;

/// Target feature string used by the LLVM backend
extern char *jitc_llvm_target_features;

/// Vector width of code generated by the LLVM backend
extern uint32_t jitc_llvm_vector_width;

/// Maximum alignment needed for vector loads
extern uint32_t jitc_llvm_max_align;

/// Should the LLVM IR use typed (e.g., "i8*") or untyped ("ptr") pointers?
extern bool jitc_llvm_opaque_pointers;

/// LLVM version (parts can equal -1, which means: not sure)
extern int jitc_llvm_version_major;
extern int jitc_llvm_version_minor;
extern int jitc_llvm_version_patch;

/// Pre-generated strings for use by the template engine

/// String of all ones, for different variable types
extern char **jitc_llvm_ones_str;

/// <i32 0, i32 1, ... > (up to the current vector width)
extern char *jitc_llvm_u32_arange_str;

/// <i32 width, i32 width, ... > (up to the current vector width)
extern char *jitc_llvm_u32_width_str;

extern uint32_t jitc_llvm_block_size;

/// Various hardware capabilities
extern bool jitc_llvm_has_avx;
extern bool jitc_llvm_has_avx512;
extern bool jitc_llvm_has_neon;

/// Try to load initialize LLVM backend
extern bool jitc_llvm_init();

/// Shut down the LLVM backend
extern void jitc_llvm_shutdown();

/// Initialize the MCJIT/ORCv2-specific parts of the LLVM backend
extern bool jitc_llvm_mcjit_init();
extern bool jitc_llvm_orcv2_init();

/// Shut down the MCJIT/ORCv2-specific parts of the LLVM backend
extern void jitc_llvm_mcjit_shutdown();
extern void jitc_llvm_orcv2_shutdown();

/// Run the MCJIT/ORCv2-based compiler on the given module
extern void jitc_llvm_mcjit_compile(void *llvm_module,
                                    std::vector<uint8_t *> &symbols);
extern void jitc_llvm_orcv2_compile(void *llvm_module,
                                    std::vector<uint8_t *> &symbols);

/// Compile the current IR string and store the resulting kernel into `kernel`
extern void jitc_llvm_compile(Kernel &kernel);

/// Dump disassembly for the given kernel
extern void jitc_llvm_disasm(const Kernel &kernel);

/// Override the target architecture
extern void jitc_llvm_set_target(const char *target_cpu,
                                 const char *target_features,
                                 uint32_t vector_width);

/// Insert a ray tracing function call into the LLVM program
extern void jitc_llvm_ray_trace(uint32_t func, uint32_t scene, int shadow_ray,
                                const uint32_t *in, uint32_t *out);

/// Computes the workers and replication_per_worker factors for the
/// ``jitc_var_expand`` function, given the size and type size.
/// ``jitc_var_expand`` Expands a variable to a larger storage area to avoid
/// atomic scatter.
extern std::pair<uint32_t, uint32_t>
jitc_llvm_expand_replication_factor(uint32_t size, uint32_t tsize);
