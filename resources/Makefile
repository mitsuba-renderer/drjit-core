COMPUTE_CAPABILITY=compute_70
CUDA_VER=10.2
NVCC=/usr/local/cuda-$(CUDA_VER)/bin/nvcc -m64 --ptx --expt-relaxed-constexpr -std=c++14 -Xcudafe --diag_suppress=550 -Xcudafe --diag_suppress=177 -Xcudafe --diag_suppress=2912 -Xcudafe --display_error_number

all: kernels_70.lz4

kernels_50.ptx: common.h block_reduce.cuh block_prefix_reduce.cuh reduce_2.cuh compress.cuh mkperm.cuh misc.cuh block_reduce.cuh kernels.cu
	$(NVCC) -Wno-deprecated-gpu-targets -gencode arch=compute_50,code=compute_50 kernels.cu -o kernels_50.ptx

kernels_70.ptx: common.h block_reduce.cuh block_prefix_reduce.cuh reduce_2.cuh compress.cuh mkperm.cuh misc.cuh block_reduce.cuh kernels.cu
	$(NVCC) -Wno-deprecated-gpu-targets -gencode arch=compute_70,code=compute_70 kernels.cu -o kernels_70.ptx

#kernels.dict:
#	zstd --train train/* --maxdict=65536 -o kernels.dict

pack: pack.c
	$(CC) pack.c ../ext/lz4/lz4hc.c ../ext/lz4/lz4.c ../ext/lz4/xxhash.c -o pack -I ../ext/lz4

kernels_70.lz4: pack kernels_50.ptx kernels_70.ptx
	./pack

clean:
	rm -f kernels_50.ptx kernels_70.ptx kernels.bin
