COMPUTE_CAPABILITY=compute_61
NVCC=nvcc -gencode arch=$(COMPUTE_CAPABILITY),code=$(COMPUTE_CAPABILITY) --ptx

all: kernels.ptx.gz

kernels.ptx: kernels.cu
	$(NVCC) kernels.cu -o kernels.ptx

kernels.ptx.gz: kernels.ptx
	rm -f kernels.ptx.gz
	gzip -k -n -9 kernels.ptx

clean:
	rm -f kernels.ptx kernels.ptx.gz
