COMPUTE_CAPABILITY=compute_70
NVCC=nvcc -m64 --ptx --expt-relaxed-constexpr

all: kernels.h

kernels_50.ptx: reduce.cuh scan.cuh mkperm.cuh misc.cuh kernels.cu
	$(NVCC) -gencode arch=compute_50,code=compute_50 kernels.cu -o kernels_50.ptx

kernels_70.ptx: reduce.cuh scan.cuh mkperm.cuh misc.cuh kernels.cu
	$(NVCC) -gencode arch=compute_70,code=compute_70 kernels.cu -o kernels_70.ptx

kernels.dict:
	zstd --train train/* --maxdict=65536 -o kernels.dict

generate: generate.c
	$(CC) generate.c ../ext/lz4/lz4hc.c ../ext/lz4/lz4.c ../ext/lz4/xxhash.c -o generate -I ../ext/lz4

kernels.h: kernels_50.ptx kernels_70.ptx generate
	./generate

scatter_add.ll: scatter_add.cpp
	clang++-10 scatter_add.cpp -O3 -mavx512f -mavx512vl -mavx512cd -mavx512dq -fvisibility=hidden -fomit-frame-pointer -fno-asynchronous-unwind-tables -emit-llvm -S -o scatter_add.ll

clean:
	rm -f kernels.ptx kernels.h generate
