#pragma once

#include <stdlib.h>
#include <stdint.h>

struct Kernel;

/// Target CPU string used by the LLVM backend
extern char *jit_llvm_target_cpu;

/// Target feature string used by the LLVM backend
extern char *jit_llvm_target_features;

/// Vector width used by the LLVM backend
extern uint32_t jit_llvm_vector_width;

/// Try to load the LLVM backend
extern bool jit_llvm_init();

/// Compile an IR string
extern void jit_llvm_compile(const char *str, size_t size, Kernel &kernel,
                             bool include_supplemental_kernels = false);

/// Dump disassembly for the given kernel
extern void jit_llvm_disasm(const Kernel &kernel);

/// Fully unload LLVM
extern void jit_llvm_shutdown();

/// Override the target architecture
extern void jit_llvm_set_target(const char *target_cpu,
                                const char *target_features,
                                uint32_t vector_width);

/// Convenience function for intrinsic function selection
extern int jit_llvm_if_at_least(uint32_t vector_width,
                                const char *feature);
