/*
    src/llvm_orcv2.h -- ORCv2-based compilation and linking of LLVM units

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "llvm_api.h"
#include <stddef.h>
#include <stdint.h>

struct UnitArtifact;

/// A per-thread LLVM compiler instance. LLVMContext is not thread-safe, so
/// each in-flight unit compilation checks one out of a shared pool (see
/// llvm_orcv2.cpp). Instances are created lazily and reused across kernels.
struct LLVMCompiler {
    /// Parsing context
    LLVMContextRef context = nullptr;

    /// Target machine driving the optimization pipeline and code generation
    LLVMTargetMachineRef tm = nullptr;
};

/// Check a compiler instance out of / back into the shared pool
extern LLVMCompiler *jitc_llvm_compiler_acquire();
extern void jitc_llvm_compiler_release(LLVMCompiler *c);

/// Destroy the pooled compiler instances, e.g. following a target change.
/// Must not run while a compilation is in flight.
extern void jitc_llvm_compiler_pool_clear();

/// Link a relocatable object file into the process and resolve the entry
/// point 'symbol'. On return, 'artifact' holds the resource tracker that owns
/// the linked code (ptr[0]), the entry point address (value), and the object
/// size (size). 'source' is printed when linking fails.
extern void jitc_llvm_link(const char *symbol, const uint8_t *object,
                           size_t size, const char *source,
                           UnitArtifact &artifact);

/// Unlink a unit and release its memory
extern void jitc_llvm_unlink(UnitArtifact &artifact);
