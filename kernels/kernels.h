#pragma once

#include <stddef.h>

#if defined(__cplusplus)
extern "C" {
#endif

extern const int    kernels_dict_size_uncompressed;
extern const int    kernels_dict_size_compressed;
extern const size_t kernels_dict_hash;
extern const char   kernels_dict[];

extern const int    kernels_50_size_uncompressed;
extern const int    kernels_50_size_compressed;
extern const size_t kernels_50_hash;
extern const char   kernels_50[];

extern const int    kernels_70_size_uncompressed;
extern const int    kernels_70_size_compressed;
extern const size_t kernels_70_hash;
extern const char   kernels_70[];

extern const char   *kernels_list;

extern const int    llvm_kernels_7_size_uncompressed;
extern const int    llvm_kernels_7_size_compressed;
extern const size_t llvm_kernels_7_hash;
extern const char   llvm_kernels_7[];

extern const int    llvm_kernels_9_size_uncompressed;
extern const int    llvm_kernels_9_size_compressed;
extern const size_t llvm_kernels_9_hash;
extern const char   llvm_kernels_9[];

#if defined(__cplusplus)
}
#endif