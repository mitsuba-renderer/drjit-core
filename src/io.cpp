#include "io.h"
#include "log.h"
#include "internal.h"
#include "cuda_api.h"
#include <stdexcept>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <lz4.h>

/// Version number for cache files
#define ENOKI_LLVM_CACHE_VERSION 1

bool jit_kernel_load(const char *source, uint32_t source_size,
                     bool llvm, size_t hash, Kernel &kernel) {
    char filename[1024];
    snprintf(filename, sizeof(filename), "%s/.enoki/%016llx.%s.bin",
             getenv("HOME"), (unsigned long long) hash,
             llvm ? "llvm" : "cuda");
    int fd = open(filename, O_RDONLY);
    if (fd == -1)
        return false;

    auto read_retry = [&](uint8_t *data, size_t data_size) {
        while (data_size > 0) {
            ssize_t n_read = read(fd, data, data_size);
            if (n_read <= 0) {
                if (errno == EINTR) {
                    continue;
                } else {
                    jit_raise("jit_kernel_read(): I/O error while while "
                              "reading compiled kernel from cache "
                              "file \"%s\": %s",
                              filename, strerror(errno));
                }
            }
            data += n_read;
            data_size -= n_read;
        }
    };

    uint8_t version;
    uint32_t compressed_size, uncompressed_size, source_size_file, kernel_size,
        func_offset;
    uint8_t *compressed = nullptr, *uncompressed = nullptr;

    bool success = true;
    try {
        read_retry((uint8_t *) &version, sizeof(uint8_t));
        if (version != ENOKI_LLVM_CACHE_VERSION)
            jit_raise("jit_kernel_read(): cache file \"%s\" is from an "
                      "incompatible version of Enoki. You may want to wipe "
                      "your ~/.enoki directory.", filename);

        read_retry((uint8_t *) &compressed_size, sizeof(uint32_t));
        read_retry((uint8_t *) &source_size_file, sizeof(uint32_t));

        if (source_size != source_size_file) {
            close(fd);
            return false;
        }

        read_retry((uint8_t *) &kernel_size, sizeof(uint32_t));
        read_retry((uint8_t *) &func_offset, sizeof(uint32_t));

        uncompressed_size = source_size + kernel_size;

        compressed = (uint8_t *) malloc_check(compressed_size);
        uncompressed = (uint8_t *) malloc_check(uncompressed_size);

        read_retry(compressed, compressed_size);

        uint32_t rv = (uint32_t) LZ4_decompress_safe(
            (const char *) compressed, (char *) uncompressed,
            (int) compressed_size, (int) uncompressed_size);

        if (rv != uncompressed_size)
            jit_raise("jit_kernel_read(): cache file \"%s\" is malformed.",
                      filename);
    } catch (const std::exception &e) {
        jit_log(Warn, "%s", e.what());
        success = false;
    }

    if (success && memcmp(uncompressed, source, source_size) != 0)
        success = false; // cache collision

    if (success) {
        jit_trace("jit_kernel_load(\"%s\")", filename);
        if (llvm) {
            kernel.size = kernel_size;
            kernel.data = mmap(nullptr, kernel_size, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (kernel.data == MAP_FAILED)
                jit_fail("jit_llvm_load(): could not mmap() memory: %s",
                         strerror(errno));

            memcpy(kernel.data, uncompressed + source_size, kernel_size);

            if (mprotect(kernel.data, kernel_size, PROT_READ | PROT_EXEC) == -1)
                jit_fail("jit_llvm_load(): mprotect() failed: %s", strerror(errno));

            kernel.llvm.func = (LLVMKernelFunction) ((uint8_t *) kernel.data + func_offset);
        } else {
            kernel.data = malloc_check(kernel_size);
            memcpy(kernel.data, uncompressed + source_size, kernel_size);
        }
    }

    free(compressed);
    free(uncompressed);

    close(fd);
    return success;
}


bool jit_kernel_write(const char *source, uint32_t source_size,
                      bool llvm, size_t hash, const Kernel &kernel) {
    char filename[1024], filename_tmp[1024];
    snprintf(filename, sizeof(filename), "%s/.enoki/%016llx.%s.bin",
             getenv("HOME"), (unsigned long long) hash,
             llvm ? "llvm" : "cuda");
    snprintf(filename_tmp, sizeof(filename_tmp), "%s.tmp", filename);

    mode_t mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;
    int fd = open(filename_tmp, O_CREAT | O_EXCL | O_WRONLY, mode);
    if (fd == -1) {
        jit_log(Warn,
                "jit_kernel_write(): could not write compiled kernel "
                "to cache file \"%s\": %s", filename_tmp, strerror(errno));
        return false;
    }

    auto write_retry = [&](const uint8_t *data, size_t data_size) {
        while (data_size > 0) {
            ssize_t n_written = write(fd, data, data_size);
            if (n_written <= 0) {
                if (errno == EINTR) {
                    continue;
                } else {
                    jit_raise("jit_kernel_write(): I/O error while while "
                              "writing compiled kernel to cache "
                              "file \"%s\": %s",
                              filename_tmp, strerror(errno));
                }
            }
            data += n_written;
            data_size -= n_written;
        }
    };

    uint32_t in_size  = source_size + kernel.size,
             out_size = LZ4_compressBound(in_size);

    uint8_t *temp_in  = (uint8_t *) malloc_check(in_size),
            *temp_out = (uint8_t *) malloc_check(out_size);

    memcpy(temp_in, source, source_size);
    memcpy(temp_in + source_size, kernel.data, kernel.size);

    LZ4_stream_t stream;
    LZ4_resetStream_fast(&stream);

    uint32_t compressed_size = (uint32_t) LZ4_compress_default(
        (const char *) temp_in, (char *) temp_out, (int) in_size,
        (int) out_size);

    free(temp_in);

    bool success = true;
    try {
        uint8_t version = ENOKI_LLVM_CACHE_VERSION;

        write_retry((const uint8_t *) &version, sizeof(uint8_t));
        write_retry((const uint8_t *) &compressed_size, sizeof(uint32_t));
        write_retry((const uint8_t *) &source_size, sizeof(uint32_t));
        write_retry((const uint8_t *) &kernel.size, sizeof(uint32_t));

        uint32_t func_offset = 0;
        if (llvm)
            func_offset = (uint8_t *) kernel.llvm.func - (uint8_t *) kernel.data;

        write_retry((const uint8_t *) &func_offset, sizeof(uint32_t));

        write_retry(temp_out, compressed_size);

    } catch (const std::exception &e) {
        jit_log(Warn, "%s", e.what());
        success = false;
    }

    if (success)
        jit_trace("jit_kernel_write(\"%s\")", filename);

    free(temp_out);
    close(fd);

    if (link(filename_tmp, filename) != 0) {
        jit_raise("jit_kernel_write(): could not link cache "
                  "file \"%s\" into file system: %s",
                  filename, strerror(errno));
        success = false;
    }


    if (unlink(filename_tmp) != 0) {
        jit_raise("jit_kernel_write(): could not unlink temporary "
                  "file \"%s\": %s",
                  filename_tmp, strerror(errno));
        success = false;
    }

    return success;
}

void jit_kernel_free(int device_id, const Kernel kernel) {
    if (device_id == -1) {
        if (munmap((void *) kernel.data, kernel.size) == -1)
            jit_fail("jit_kernel_free(): munmap() failed!");
    } else {
        const Device &device = state.devices.at(device_id);
        cuda_check(cuCtxSetCurrent(device.context));
        cuda_check(cuModuleUnload(kernel.cuda.cu_module));
        free(kernel.data);
    }
}

