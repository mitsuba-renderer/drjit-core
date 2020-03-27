COMPUTE_CAPABILITY=compute_61
NVCC=nvcc -m64 -gencode arch=$(COMPUTE_CAPABILITY),code=$(COMPUTE_CAPABILITY) --ptx --expt-relaxed-constexpr

all: kernels.h

kernels.ptx: reduce.cuh scan.cuh mkperm.cuh misc.cuh
	$(NVCC) kernels.cu -o kernels.ptx

kernels.dict:
	zstd --train train/* --maxdict=65536 -o kernels.dict

generate: generate.c
	$(CC) generate.c ../ext/lz4/lz4hc.c ../ext/lz4/lz4.c ../ext/lz4/xxhash.c -o generate -I ../ext/lz4

kernels.h: kernels.ptx generate
	./generate

clean:
	rm -f kernels.ptx kernels.h generate
