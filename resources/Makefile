COMPUTE_CAPABILITY=compute_70
CUDA_VER=10.2
NVCC=/usr/local/cuda-$(CUDA_VER)/bin/nvcc -m64 --ptx --expt-relaxed-constexpr -std=c++14

all: kernels.h

kernels_50.ptx: reduce.cuh reduce_2.cuh prefix_sum.cuh compress.cuh mkperm.cuh misc.cuh block_reduce.cuh kernels.cu
	$(NVCC) --Wno-deprecated-gpu-targets -gencode arch=compute_50,code=compute_50 kernels.cu -o kernels_50.ptx

kernels_70.ptx: reduce.cuh reduce_2.cuh prefix_sum.cuh compress.cuh mkperm.cuh misc.cuh block_reduce.cuh kernels.cu
	$(NVCC) -Wno-deprecated-gpu-targets -gencode arch=compute_70,code=compute_70 kernels.cu -o kernels_70.ptx

kernels.dict:
	zstd --train train/* --maxdict=65536 -o kernels.dict

generate: generate.c
	$(CC) generate.c ../ext/lz4/lz4hc.c ../ext/lz4/lz4.c ../ext/lz4/xxhash.c -o generate -I ../ext/lz4

kernels.h: kernels_50.ptx kernels_70.ptx generate
	./generate

clean:
	rm -f kernels.ptx kernels.h generate
