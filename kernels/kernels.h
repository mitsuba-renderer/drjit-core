#pragma once

#include <stddef.h>

#if defined(__cplusplus)
extern "C" {
#endif

extern int    kernels_dict_size_uncompressed;
extern int    kernels_dict_size_compressed;
extern size_t kernels_dict_hash;
extern char   kernels_dict[];

extern int    kernels_50_size_uncompressed;
extern int    kernels_50_size_compressed;
extern size_t kernels_50_hash;
extern char   kernels_50[];

extern int    kernels_70_size_uncompressed;
extern int    kernels_70_size_compressed;
extern size_t kernels_70_hash;
extern char   kernels_70[];

extern int    llvm_kernels_7_size_uncompressed;
extern int    llvm_kernels_7_size_compressed;
extern size_t llvm_kernels_7_hash;
extern char   llvm_kernels_7[];

extern int    llvm_kernels_9_size_uncompressed;
extern int    llvm_kernels_9_size_compressed;
extern size_t llvm_kernels_9_hash;
extern char   llvm_kernels_9[];

#if defined(__cplusplus)
}
#endif