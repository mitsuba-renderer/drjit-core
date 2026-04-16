# Target of the precompiled PTX files shipped with Dr.Jit-core:
#   - CUDA 12.2 (nvcc 12.2.140)
#   - PTX ISA version 8.2
#   - Minimum CUDA driver: 535 (the one bundled with CUDA 12.2)
#   - Compute Capability >= 7.5 (Turing and newer)

COMPUTE_CAPABILITY=compute_75
CUDA_VER=12.2
HOST_CC=/usr/bin/gcc-12
NVCC=/usr/local/cuda-$(CUDA_VER)/bin/nvcc -ccbin $(HOST_CC) -m64 --ptx --expt-relaxed-constexpr -std=c++14 -Xcudafe --diag_suppress=550 -Xcudafe --diag_suppress=177 -Xcudafe --diag_suppress=2912 -Xcudafe --display_error_number

all: kernels_75.lz4

kernels_75.ptx: common.h block_reduce.cuh block_prefix_reduce.cuh reduce_2.cuh compress.cuh mkperm.cuh misc.cuh gemm.cuh kernels.cu
	$(NVCC) -Wno-deprecated-gpu-targets -gencode arch=$(COMPUTE_CAPABILITY),code=$(COMPUTE_CAPABILITY) kernels.cu -o kernels_75.ptx

#kernels.dict:
#	zstd --train train/* --maxdict=65536 -o kernels.dict

pack: pack.c
	$(CC) pack.c ../ext/lz4/lz4hc.c ../ext/lz4/lz4.c ../ext/lz4/xxhash.c -o pack -I ../ext/lz4

kernels_75.lz4: pack kernels_75.ptx
	./pack

clean:
	rm -f kernels_75.ptx kernels.bin
