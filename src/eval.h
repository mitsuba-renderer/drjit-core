/*
    src/eval.h -- Main computation graph evaluation routine

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "internal.h"
#include "strbuf.h"
#include <map>

/// A single variable that is scheduled to execute for a launch with 'size' entries
struct ScheduledVariable {
    uint32_t size;
    uint32_t index;
    uint32_t scope;
    void *data;

    ScheduledVariable(uint32_t size, uint32_t scope, uint32_t index)
        : size(size), index(index), scope(scope), data(nullptr) { }
};

/// Start and end index of a group of variables that will be merged into the same kernel
struct ScheduledGroup {
    uint32_t size;
    uint32_t start;
    uint32_t end;

    ScheduledGroup(uint32_t size, uint32_t start, uint32_t end)
        : size(size), start(start), end(end) { }
};

struct GlobalKey {
    XXH128_hash_t hash;
    bool callable;

    GlobalKey(XXH128_hash_t hash, bool callable)
        : hash(hash), callable(callable) { }

    /* Order so that callables are defined before other globals, but don't use
       the callable ID itself for ordering (it can be non-deterministic in
       programs that use Dr.Jit with parallelization) */
    bool operator<(const GlobalKey &v) const {
        int callable_key_t =   callable ? 0 : 1,
            callable_key_v = v.callable ? 0 : 1;
        return std::tie(callable_key_t, hash.high64, hash.low64) <
               std::tie(callable_key_v, v.hash.high64, v.hash.low64);
    }
};

struct GlobalValue {
    /// Offset and length for the 'globals' buffer
    size_t start, length;

    /// Index within the callable list, if applicable
    uint32_t callable_index;

    GlobalValue(size_t start, size_t length)
        : start(start), length(length), callable_index(0) { }
};

/// Cache data structure for global declarations
using GlobalsMap = std::map<GlobalKey, GlobalValue>;

/// StringBuffer for global definitions (intrinsics, callables, etc.)
extern StringBuffer globals;

/// Mapping that describes the contents of the 'globals' buffer
extern GlobalsMap globals_map;

/// Name of the last generated kernel
extern char kernel_name[52];

/// Are we recording an OptiX kernel?
extern bool uses_optix;

/// Size and alignment of auxiliary buffer needed by virtual function calls
extern int32_t alloca_size;
extern int32_t alloca_align;

/// Number of tentative callables that were assembled in the kernel being compiled
extern uint32_t callable_count;

/// Number of unique callables in the kernel being compiled
extern uint32_t callable_count_unique;

/// Specifies the nesting level of virtual calls being compiled
extern uint32_t callable_depth;

/// Ordered list of variables that should be computed
extern std::vector<ScheduledVariable> schedule;

/// Groups of variables with the same size
extern std::vector<ScheduledGroup> schedule_groups;

/// Evaluate all computation that is queued on the current thread
extern void jitc_eval(ThreadState *ts);

/// Used by jitc_eval() to generate PTX source code
extern void jitc_cuda_assemble(ThreadState *ts, ScheduledGroup group,
                               uint32_t n_regs, uint32_t n_params);

/// Used by jitc_eval() to generate LLVM IR source code
extern void jitc_llvm_assemble(ThreadState *ts, ScheduledGroup group);

/// Used by jitc_vcall() to generate source code for vcalls
extern XXH128_hash_t
jitc_assemble_func(ThreadState *ts, const char *name, uint32_t inst_id,
                   uint32_t in_size, uint32_t in_align, uint32_t out_size,
                   uint32_t out_align, uint32_t data_offset,
                   const tsl::robin_map<uint64_t, uint32_t, UInt64Hasher> &data_map,
                   uint32_t n_in, const uint32_t *in, uint32_t n_out,
                   const uint32_t *out_nested, uint32_t n_se,
                   const uint32_t *se, bool use_self);

/// Used by jitc_vcall() to generate PTX source code for vcalls
extern void
jitc_cuda_assemble_func(const char *name, uint32_t inst_id, uint32_t n_regs,
                        uint32_t in_size, uint32_t in_align, uint32_t out_size,
                        uint32_t out_align, uint32_t data_offset,
                        const tsl::robin_map<uint64_t, uint32_t, UInt64Hasher> &data_map,
                        uint32_t n_out, const uint32_t *out_nested,
                        bool use_self);

/// Used by jitc_vcall() to generate LLVM IR source code for vcalls
extern void
jitc_llvm_assemble_func(const char *name, uint32_t inst_id,
                        uint32_t in_size, uint32_t data_offset,
                        const tsl::robin_map<uint64_t, uint32_t, UInt64Hasher> &data_map,
                        uint32_t n_out, const uint32_t *out_nested,
                        bool use_self);

/// Register a global declaration that will be included in the final program
extern void jitc_register_global(const char *str);
