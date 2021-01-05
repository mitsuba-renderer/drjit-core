/*
    src/eval.h -- Main computation graph evaluation routine

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "internal.h"
#include <tsl/robin_set.h>

/// A single variable that is scheduled to execute for a launch with 'size' entries
struct ScheduledVariable {
    uint32_t size;
    uint32_t index;

    ScheduledVariable(uint32_t size, uint32_t index)
        : size(size), index(index) { }
};

/// Start and end index of a group of variables that will be merged into the same kernel
struct ScheduledGroup {
    uint32_t size;
    uint32_t start;
    uint32_t end;

    ScheduledGroup(uint32_t size, uint32_t start, uint32_t end)
        : size(size), start(start), end(end) { }
};

/// Hashing helper for GlobalMap
struct XXH128Hash {
    size_t operator()(const XXH128_hash_t &h) const { return h.low64; }
};
struct XXH128Eq {
    size_t operator()(const XXH128_hash_t &h1, const XXH128_hash_t &h2) const {
        return h1.low64 == h2.low64 && h1.high64 == h2.high64;
    }
};

/// Cache data structure for global declarations
using GlobalsMap = tsl::robin_map<XXH128_hash_t, uint32_t, XXH128Hash, XXH128Eq>;

/// Name of the last generated kernel
extern char kernel_name[52];

/// Buffer containing global declarations
extern std::vector<std::string> globals;

/// Ensure uniqueness of global declarations (intrinsics, virtual functions)
extern GlobalsMap globals_map;

#if defined(ENOKI_JIT_ENABLE_OPTIX)
/// Are we recording an OptiX kernel?
extern bool uses_optix;

/// List of optix callable references in call sites, used to create the SBT
extern std::vector<uint32_t> optix_callables;
#endif

/// Does the program contain a %data register so far? (for branch-based vcalls)
extern bool data_reg_global;

/// Ordered list of variables that should be computed
extern std::vector<ScheduledVariable> schedule;

/// Groups of variables with the same size
extern std::vector<ScheduledGroup> schedule_groups;

/// Evaluate all computation that is queued on the current thread
extern void jitc_eval(ThreadState *ts);

/// Used by jitc_eval() to generate PTX source code
extern void jitc_assemble_cuda(ThreadState *ts, ScheduledGroup group,
                               uint32_t n_regs, uint32_t n_params);

/// Used by jitc_eval() to generate LLVM IR source code
extern void jitc_assemble_llvm(ThreadState *ts, ScheduledGroup group);

/// Used by jitc_vcall() to generate source code for vcalls
extern XXH128_hash_t jitc_assemble_func(ThreadState *ts, uint32_t in_size,
                                        uint32_t in_align, uint32_t out_size,
                                        uint32_t out_align, bool has_data_arg,
                                        uint32_t n_in, const uint32_t *in,
                                        uint32_t n_out, const uint32_t *out,
                                        const uint32_t *out_nested,
                                        uint32_t n_se, const uint32_t *se,
                                        const char *ret_label);

/// Used by jitc_vcall() to generate PTX source code for vcalls
extern void jitc_assemble_cuda_func(uint32_t n_regs, uint32_t in_size,
                                    uint32_t in_align, uint32_t out_size,
                                    uint32_t out_align, bool has_data_arg,
                                    uint32_t n_out, const uint32_t *out,
                                    const uint32_t *out_nested,
                                    const char *ret_label);
