/*
    src/io.cpp -- Disk cache for LLVM/CUDA kernels

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "io.h"
#include "log.h"
#include "internal.h"
#include "cuda_api.h"
#include "../kernels/kernels.h"
#include <stdexcept>
#include <stdio.h>
#include <fcntl.h>
#include <errno.h>
#include <lz4.h>

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <unistd.h>
#  include <sys/mman.h>
#endif

/// Version number for cache files
#define ENOKI_CACHE_VERSION 1

// Uncomment to write out training data for creating a compression dictionary
// #define ENOKI_CACHE_TRAIN 1

#pragma pack(push)
#pragma pack(1)
struct CacheFileHeader {
    uint8_t version;
    uint32_t compressed_size;
    uint32_t source_size;
    uint32_t kernel_size;
    uint32_t func_offset_wide;
    uint32_t func_offset_scalar;
};
#pragma pack(pop)

char jit_lz4_dict[jit_lz4_dict_size];
static bool jit_lz4_dict_ready = false;

void jit_lz4_init() {
    if (jit_lz4_dict_ready)
        return;

    if (jit_lz4_dict_size != kernels_dict_size_uncompressed)
        jit_fail("jit_init_lz4(): dictionary has invalid size!");

    if (LZ4_decompress_safe(kernels_dict, jit_lz4_dict,
                            kernels_dict_size_compressed,
                            kernels_dict_size_uncompressed) !=
        (int) kernels_dict_size_uncompressed)
        jit_fail("jit_init_lz4(): decompression of dictionary failed!");

    jit_lz4_dict_ready = true;
}

bool jit_kernel_load(const char *source, uint32_t source_size,
                     bool cuda, size_t hash, Kernel &kernel) {
    jit_lz4_init();

#if !defined(_WIN32)
    char filename[512];
    if (unlikely(snprintf(filename, sizeof(filename), "%s/%016llx.%s.bin",
                          jit_temp_path, (unsigned long long) hash,
                          cuda ? "cuda" : "llvm") < 0))
        jit_fail("jit_kernel_load(): scratch space for filename insufficient!");

    int fd = open(filename, O_RDONLY);
    if (fd == -1)
        return false;

    auto read_retry = [&](uint8_t* data, size_t data_size) {
        while (data_size > 0) {
            ssize_t n_read = read(fd, data, data_size);
            if (n_read <= 0) {
                if (errno == EINTR) {
                    continue;
                } else {
                    jit_raise("jit_kernel_load(): I/O error while while "
                              "reading compiled kernel from cache "
                              "file \"%s\": %s",
                              filename, strerror(errno));
                }
            }
            data += n_read;
            data_size -= n_read;
        }
    };
#else
    wchar_t filename_w[512];
    char filename[512];

    int rv = _snwprintf(filename_w, sizeof(filename_w) / sizeof(wchar_t),
                        L"%s\\%016llx.%s.bin",
                        jit_temp_path, (unsigned long long) hash,
                        cuda ? L"cuda" : L"llvm");

    if (rv < 0 || rv == sizeof(filename) ||
        wcstombs(filename, filename_w, sizeof(filename)) == sizeof(filename))
        jit_fail("jit_kernel_load(): scratch space for filename insufficient!");
 
    HANDLE fd = CreateFileW(filename_w, GENERIC_READ,
        FILE_SHARE_READ, nullptr, OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL, nullptr);

    if (fd == INVALID_HANDLE_VALUE)
        return false;

    auto read_retry = [&](uint8_t* data, size_t data_size) {
        while (data_size > 0) {
            DWORD n_read = 0;
            if (!ReadFile(fd, data, (DWORD) data_size, &n_read, nullptr) || n_read == 0)
                jit_raise("jit_kernel_load(): I/O error while while "
                          "reading compiled kernel from cache "
                          "file \"%s\": %u", filename, GetLastError());

            data += n_read;
            data_size -= n_read;
        }
    };
#endif

    char *compressed = nullptr, *uncompressed = nullptr;

    CacheFileHeader header;
    bool success = true;

    try {
        read_retry((uint8_t *) &header, sizeof(CacheFileHeader));

        if (header.version != ENOKI_CACHE_VERSION)
            jit_raise("jit_kernel_load(): cache file \"%s\" is from an "
                      "incompatible version of Enoki. You may want to wipe "
                      "your ~/.enoki directory.", filename);

        if (header.source_size != source_size)
            jit_raise("jit_kernel_load(): cache collision in file \"%s\": size "
                      "mismatch (%u vs %u bytes).",
                      filename, header.source_size, source_size);

        uint32_t uncompressed_size = source_size + header.kernel_size;

        compressed = (char *) malloc_check(header.compressed_size);
        uncompressed = (char *) malloc_check(size_t(uncompressed_size) + jit_lz4_dict_size);
        memcpy(uncompressed, jit_lz4_dict, jit_lz4_dict_size);

        read_retry((uint8_t *) compressed, header.compressed_size);

        uint32_t rv = (uint32_t) LZ4_decompress_safe_usingDict(
            compressed, uncompressed + jit_lz4_dict_size,
            (int) header.compressed_size, (int) uncompressed_size,
            (char *) uncompressed, jit_lz4_dict_size);

        if (rv != uncompressed_size)
            jit_raise("jit_kernel_load(): cache file \"%s\" is malformed.",
                      filename);
    } catch (const std::exception &e) {
        jit_log(Warn, "%s", e.what());
        success = false;
    }

    char *uncompressed_data = uncompressed + jit_lz4_dict_size;

    if (success && memcmp(uncompressed_data, source, source_size) != 0) {
        jit_log(Warn, "jit_kernel_load(): cache collision in file \"%s\".", filename);
        success = false;
    }

    if (success) {
        jit_trace("jit_kernel_load(\"%s\")", filename);
        kernel.size = header.kernel_size;
        if (cuda) {
            kernel.data = malloc_check(header.kernel_size);
            memcpy(kernel.data, uncompressed_data + source_size, header.kernel_size);
        } else {
#if !defined(_WIN32)
            kernel.data = mmap(nullptr, header.kernel_size, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (kernel.data == MAP_FAILED)
                jit_fail("jit_llvm_load(): could not mmap() memory: %s",
                         strerror(errno));

            memcpy(kernel.data, uncompressed_data + source_size, header.kernel_size);

            if (mprotect(kernel.data, header.kernel_size, PROT_READ | PROT_EXEC) == -1)
                jit_fail("jit_llvm_load(): mprotect() failed: %s", strerror(errno));
#else
            kernel.data = VirtualAlloc(nullptr, header.kernel_size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
            if (!kernel.data)
                jit_fail("jit_llvm_load(): could not VirtualAlloc() memory: %u", GetLastError());
            memcpy(kernel.data, uncompressed_data + source_size, header.kernel_size);

            DWORD unused;
            if (VirtualProtect(kernel.data, header.kernel_size, PAGE_EXECUTE_READ, &unused) == 0)
                jit_fail("jit_llvm_load(): VirtualProtect() failed: %u", GetLastError()); 
#endif

            kernel.llvm.func = (LLVMKernelFunction)(
                (uint8_t *) kernel.data + header.func_offset_wide);
            kernel.llvm.func_scalar = (LLVMKernelFunction)(
                (uint8_t *) kernel.data + header.func_offset_scalar);
        }
    }

    free(compressed);
    free(uncompressed);

#if !defined(_WIN32)
    close(fd);
#else
    CloseHandle(fd);
#endif

    return success;
}

bool jit_kernel_write(const char *source, uint32_t source_size,
                      bool cuda, size_t hash, const Kernel &kernel) {
    jit_lz4_init();

#if !defined(_WIN32)
    char filename[512], filename_tmp[512];
    if (unlikely(snprintf(filename, sizeof(filename), "%s/%016llx.%s.bin",
                          jit_temp_path, (unsigned long long) hash,
                          cuda ? "cuda" : "llvm") < 0))
        jit_fail("jit_kernel_write(): scratch space for filename insufficient!");

    if (unlikely(snprintf(filename_tmp, sizeof(filename_tmp), "%s.tmp",
                          filename) < 0))
        jit_fail("jit_kernel_write(): scratch space for filename insufficient!");

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
#else
    wchar_t filename_w[512], filename_tmp_w[512];
    char filename[512], filename_tmp[512];

    int rv = _snwprintf(filename_w, sizeof(filename_w) / sizeof(wchar_t),
                        L"%s\\%016llx.%s.bin",
                        jit_temp_path, (unsigned long long) hash,
                        cuda ? L"cuda" : L"llvm");

    if (rv < 0 || rv == sizeof(filename) ||
        wcstombs(filename, filename_w, sizeof(filename)) == sizeof(filename))
        jit_fail("jit_kernel_write(): scratch space for filename insufficient!");

    rv = _snwprintf(filename_tmp_w, sizeof(filename_tmp_w) / sizeof(wchar_t),
                    L"%s.tmp", filename_w);

    if (rv < 0 || rv == sizeof(filename_tmp) ||
        wcstombs(filename_tmp, filename_tmp_w, sizeof(filename_tmp)) == sizeof(filename_tmp))
        jit_fail("jit_kernel_write(): scratch space for filename insufficient!");

    HANDLE fd = CreateFileW(filename_tmp_w, GENERIC_WRITE,
        0 /* exclusive */, nullptr, CREATE_NEW /* fail if creation fails */,
        FILE_ATTRIBUTE_NORMAL, nullptr);

    if (fd == INVALID_HANDLE_VALUE) {
        jit_log(Warn,
            "jit_kernel_write(): could not write compiled kernel "
            "to cache file \"%s\": %u", filename_tmp, GetLastError());
        return false;
    }

    auto write_retry = [&](const uint8_t* data, size_t data_size) {
        while (data_size > 0) {
            DWORD n_written = 0;
            if (!WriteFile(fd, data, (DWORD) data_size, &n_written, nullptr))
                jit_raise("jit_kernel_write(): I/O error while while "
                          "writing compiled kernel to cache "
                          "file \"%s\": %u",
                          filename_tmp, GetLastError());

            data += n_written;
            data_size -= n_written;
        }
    };
#endif

    uint32_t in_size  = source_size + kernel.size,
             out_size = LZ4_compressBound(in_size);

    uint8_t *temp_in  = (uint8_t *) malloc_check(in_size),
            *temp_out = (uint8_t *) malloc_check(out_size);

    memcpy(temp_in, source, source_size);
    memcpy(temp_in + source_size, kernel.data, kernel.size);

    LZ4_stream_t stream;
    memset(&stream, 0, sizeof(LZ4_stream_t));
    LZ4_resetStream_fast(&stream);
    LZ4_loadDict(&stream, jit_lz4_dict, jit_lz4_dict_size);

    CacheFileHeader header;
    header.version = ENOKI_CACHE_VERSION;
    header.source_size = source_size;
    header.kernel_size = kernel.size;
    header.compressed_size = (uint32_t) LZ4_compress_fast_continue(
        &stream, (const char *) temp_in, (char *) temp_out, (int) in_size,
        (int) out_size, 1);

    header.func_offset_wide = header.func_offset_scalar = 0;
    if (!cuda) {
        header.func_offset_wide = (uint32_t)
            ((uint8_t *) kernel.llvm.func - (uint8_t *) kernel.data);
        header.func_offset_scalar = (uint32_t)
            ((uint8_t *) kernel.llvm.func_scalar - (uint8_t *) kernel.data);
    }

    bool success = true;
    try {
        write_retry((const uint8_t *) &header, sizeof(CacheFileHeader));
        write_retry(temp_out, header.compressed_size);
    } catch (const std::exception &e) {
        jit_log(Warn, "%s", e.what());
        success = false;
    }

    bool trace_log = std::max(state.log_level_stderr,
                              state.log_level_callback) >= LogLevel::Trace;
    if (success && trace_log)
        jit_trace("jit_kernel_write(\"%s\"): compressed %s to %s", filename,
                  std::string(jit_mem_string(size_t(source_size) + kernel.size)).c_str(),
                  std::string(jit_mem_string(header.compressed_size)).c_str());

#if !defined(_WIN32)
    close(fd);

    if (link(filename_tmp, filename) != 0) {
        jit_log(Warn,
            "jit_kernel_write(): could not link cache "
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
#else
    CloseHandle(fd);

    if (MoveFileW(filename_tmp_w, filename_w) == 0)
        jit_log(Warn,
                "jit_kernel_write(): could not link cache "
                "file \"%s\" into file system: %u",
                filename, GetLastError());
#endif

#if ENOKI_CACHE_TRAIN == 1
    snprintf(filename, sizeof(filename), "%s/.enoki/%016llx.%s.trn",
             getenv("HOME"), (unsigned long long) hash,
             cuda ? "cuda" : "llvm");
    fd = open(filename, O_CREAT | O_WRONLY, mode);
    if (fd) {
        write_retry(temp_in, in_size);
        close(fd);
    }
#endif

    free(temp_out);
    free(temp_in);

    return success;
}

void jit_kernel_free(int device_id, const Kernel kernel) {
    if (device_id == -1) {
#if !defined(_WIN32)
        if (munmap((void *) kernel.data, kernel.size) == -1)
            jit_fail("jit_kernel_free(): munmap() failed!");
#else
        if (VirtualFree((void*)kernel.data, 0, MEM_RELEASE) == 0)
            jit_fail("jit_kernel_free(): VirtualFree() failed!");
#endif
    } else {
        const Device &device = state.devices.at(device_id);
        cuda_check(cuCtxSetCurrent(device.context));
        cuda_check(cuModuleUnload(kernel.cuda.cu_module));
        free(kernel.data);
    }
}

