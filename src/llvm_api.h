#pragma once

#include <enoki/jit.h>

/// Target CPU string used by the LLVM backend
extern char *jit_llvm_target_cpu;

/// Target feature string used by the LLVM backend
extern char *jit_llvm_target_features;

/// Vector width used by the LLVM backend
extern int jit_llvm_vector_width;

/// Try to load the LLVM backend
extern bool jit_llvm_init();

/// Compile an IR string
extern void jit_llvm_compile(const char *str, size_t size);

/// Fully unload LLVM
extern void jit_llvm_shutdown();
