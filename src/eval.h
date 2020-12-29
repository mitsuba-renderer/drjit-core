/*
    src/eval.h -- Main computation graph evaluation routine

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

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

/// Hashing helper for GlobalSet
struct GlobalsHash {
    size_t operator()(const std::string &s) const {
        return hash_str(s.c_str(), s.length());
    }
};

/// Cache data structure for global declarations
using GlobalsSet =
    tsl::robin_set<std::string, GlobalsHash,
                   std::equal_to<std::string>,
                   std::allocator<std::string>,
                   /* StoreHash = */ true>;

/// Buffer containing global declarations
extern Buffer globals;

/// Ensure uniqueness of global declarations (intrinsics, virtual functions)
extern GlobalsSet globals_set;

/// Are we recording an OptiX kernel?
extern bool uses_optix;

/// Ordered list of variables that should be computed
extern std::vector<ScheduledVariable> schedule;

/// Groups of variables with the same size
extern std::vector<ScheduledGroup> schedule_groups;

/// Evaluate all computation that is queued on the current thread
extern void jitc_eval(ThreadState *ts);

/// Used by jitc_eval() to generate PTX source code
extern void jitc_assemble_cuda(ThreadState *ts, ScheduledGroup group,
                               uint32_t n_regs, uint32_t n_params,
                               int params_global);

/// Used by jitc_eval() to generate LLVM IR source code
extern void jitc_assemble_llvm(ThreadState *ts, ScheduledGroup group);
