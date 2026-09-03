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

# The LZ4 dictionaries of the kernel cache (lz4_dict_*) are raw 64 KiB
# blobs trained with zstd's "cover" dictionary builder on uncompressed cache
# entries, with the zstd header stripped off. The entries are LZ4 frames that
# the lz4 tool unpacks given the matching dictionary, e.g.
#
#   lz4 -d -D lz4_dict_macho ~/.drjit/<hash>.o.lz4 kernel.o
#
# The dictionaries are embedded as-is by CMake.

pack: pack.c
	$(CC) pack.c ../ext/lz4/lz4hc.c ../ext/lz4/lz4.c -o pack -I ../ext/lz4

kernels_75.lz4: pack kernels_75.ptx
	./pack

# Precompiled Metal kernels shipped with Dr.Jit-core (macOS only). Requires the
# full Metal toolchain (xcrun metal), unlike the build itself, which only embeds
# the committed compressed metallib.
metal: metal_kernels.lz4

metal_kernels.metallib: metal_kernels.metal
	xcrun -sdk macosx metal -mmacosx-version-min=13.0 -std=metal3.2 -fno-fast-math -c metal_kernels.metal -o metal_kernels.air
	xcrun -sdk macosx metallib metal_kernels.air -o metal_kernels.metallib
	rm -f metal_kernels.air

metal_kernels.lz4: pack metal_kernels.metallib
	./pack metal

clean:
	rm -f kernels_75.ptx kernels.bin metal_kernels.air metal_kernels.metallib
