COMPUTE_CAPABILITY=compute_61
NVCC=nvcc -m64 -gencode arch=$(COMPUTE_CAPABILITY),code=$(COMPUTE_CAPABILITY) --ptx --expt-relaxed-constexpr

all: kernels.h

kernels.ptx: kernels.cu
	$(NVCC) kernels.cu -I cub -o kernels.ptx

kernels.ptx.gz: kernels.ptx
	rm -f kernels.ptx.gz
	gzip -k -n -9 kernels.ptx

kernels.h: kernels.ptx.gz
	$(eval SIZE := $(shell stat -c%s kernels.ptx))
	@echo "static size_t kernels_ptx_uncompressed_size = $(SIZE);" > kernels.h
	bin2c kernels.ptx.gz --static --name "kernels_ptx_compressed" | grep -v '^#\|^ext' | head -n -3 >> kernels.h

clean:
	rm -f kernels.ptx kernels.ptx.gz kernels.h
