/*
    src/io.cpp -- Disk cache for compiled kernel artifacts

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "io.h"
#include "log.h"
#include "internal.h"
#include "profile.h"
#include "cuda.h"
#include "optix.h"
#include "unit.h"
#include "resources/kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <lz4.h>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>
#include <mutex>

#if defined(_WIN32)
#  include <windows.h>
#  define jitc_getpid() GetCurrentProcessId()
#else
#  include <unistd.h>
#  define jitc_getpid() getpid()
#endif

namespace fs = std::filesystem;

/// Version number for cache files
#define DRJIT_CACHE_VERSION 2

/// Also write each payload in uncompressed form to a sibling ".trn" file
/// (set via DRJIT_CACHE_TRAIN) that can be used to train the LZ4 dictionary.
static bool jitc_cache_train = false;

/// Seed of the cache file checksum. Must differ from the (zero) seed of the
/// unit content hashes, so that the two cannot coincide.
#define DRJIT_CACHE_CHECKSUM_SEED 1

/// Records when the last sweep began. Must not end in ".bin".
#define DRJIT_CACHE_STAMP "sweep.stamp"

/// Kernel cache directory. Empty when the on-disk cache is unavailable.
static fs::path jitc_cache_path;

/// 'jitc_cache_path' as a char string, for log messages and jit_cache_dir()
static std::string jitc_cache_path_str;

/// Latched when a write fails for lack of permission. Reads keep working.
static bool jitc_cache_read_only = false;

/// Report what the sweep did? (set via DRJIT_CACHE_VERBOSE)
static bool jitc_cache_verbose = false;

/// Soft upper bound on the size of the kernel cache in bytes (0: no eviction)
static size_t jitc_cache_max_size = 0;

/// Default value of 'jitc_cache_max_size'
static constexpr size_t jitc_cache_default_max_size = (size_t) 1024 * 1024 * 1024;

/// Sweep the cache directory at most once per hour, per machine
static constexpr auto jitc_cache_sweep_interval = std::chrono::hours(1);

/// Only refresh the timestamp of entries that have been idle for a day
static constexpr auto jitc_cache_touch_interval = std::chrono::hours(24);

#pragma pack(push)
#pragma pack(1)
struct CacheFileHeader {
    uint8_t version;
    uint32_t compressed_size;
    uint32_t check;
    uint32_t payload_size;
    XXH128_hash_t checksum;
};
#pragma pack(pop)

static_assert(sizeof(CacheFileHeader) == 29, "CacheFileHeader is not packed!");

char jitc_lz4_dict[jitc_lz4_dict_size];
static std::once_flag jitc_lz4_dict_once;

void jitc_lz4_init() {
    std::call_once(jitc_lz4_dict_once, []() {
        if (jitc_lz4_dict_size != kernels_dict_size_uncompressed)
            jitc_fail("jit_init_lz4(): dictionary has invalid size!");

        if (LZ4_decompress_safe(kernels_dict, jitc_lz4_dict,
                                kernels_dict_size_compressed,
                                kernels_dict_size_uncompressed) !=
            (int) kernels_dict_size_uncompressed)
            jitc_fail("jit_init_lz4(): decompression of dictionary failed!");
    });
}

/// Checksum over the header (up to the checksum itself) and the payload in
/// compressed form. This serves as extra validation that a cache file has the
/// expected contents.
static XXH128_hash_t jitc_cache_checksum(const CacheFileHeader &header,
                                         const char *compressed) {
    XXH3_state_t xxh_state;
    XXH3_128bits_reset_withSeed(&xxh_state, DRJIT_CACHE_CHECKSUM_SEED);
    XXH3_128bits_update(&xxh_state, &header, offsetof(CacheFileHeader, checksum));
    XXH3_128bits_update(&xxh_state, compressed, header.compressed_size);
    return XXH3_128bits_digest(&xxh_state);
}

/// Path of the cache file holding an entry of the given kind
static fs::path jitc_cache_entry(XXH128_hash_t hash, const char *kind) {
    char name[80];
    snprintf(name, sizeof(name), "%016llx%016llx.%s.v%i.bin",
             (unsigned long long) hash.high64, (unsigned long long) hash.low64,
             kind, DRJIT_CACHE_VERSION);

    return jitc_cache_path / name;
}

// ============================================================================
//  Cache directory
// ============================================================================

/// Parse a DRJIT_CACHE_MAXSIZE value, returns SIZE_MAX when malformed
static size_t jitc_cache_parse_size(const char *value) {
    // strtoull() would quietly wrap a negative value into a huge budget
    if (*value == '-')
        return SIZE_MAX;

    char *end = nullptr;
    errno = 0;
    unsigned long long result = strtoull(value, &end, 10);

    if (errno != 0 || end == value)
        return SIZE_MAX;

    size_t scale = 1;
    switch (*end) {
        case '\0': break;
        case 'k': case 'K': scale = (size_t) 1 << 10; end++; break;
        case 'm': case 'M': scale = (size_t) 1 << 20; end++; break;
        case 'g': case 'G': scale = (size_t) 1 << 30; end++; break;
        default: return SIZE_MAX;
    }

    if (*end != '\0' || result > SIZE_MAX / scale)
        return SIZE_MAX;

    return (size_t) result * scale;
}

/// Default cache location. Returns false (with a warning) when there is none.
static bool jitc_cache_default_path(fs::path &path) {
#if !defined(_WIN32)
    const char *home = getenv("HOME");
    if (!home || !*home) {
        jitc_log(Warn, "jit_init(): the HOME environment variable is not set, "
                       "disabling the kernel cache. Set DRJIT_CACHE_DIR to "
                       "choose a cache directory explicitly.");
        return false;
    }
    path = fs::path(home) / ".drjit";
#else
    wchar_t temp[MAX_PATH + 1];
    DWORD len = GetTempPathW(MAX_PATH + 1, temp);
    if (len == 0 || len > MAX_PATH) {
        jitc_log(Warn, "jit_init(): could not determine the path of the "
                       "temporary directory, disabling the kernel cache.");
        return false;
    }
    path = fs::path(temp) / "drjit";
#endif
    return true;
}

/// Has the cache directory been located since the last shutdown?
static bool jitc_cache_initialized = false;

void jitc_cache_init() {
    if (jitc_cache_initialized)
        return;
    jitc_cache_initialized = true;

    // The sweep runs before the host can raise the log level, so it needs its
    // own switch to be able to report anything at all.
    const char *verbose_str = getenv("DRJIT_CACHE_VERBOSE");
    jitc_cache_verbose =
        verbose_str && *verbose_str && strcmp(verbose_str, "0") != 0;

    const char *train_str = getenv("DRJIT_CACHE_TRAIN");
    jitc_cache_train =
        train_str && *train_str && strcmp(train_str, "0") != 0;

    jitc_cache_max_size = jitc_cache_default_max_size;
    const char *max_size_str = getenv("DRJIT_CACHE_MAXSIZE");
    if (max_size_str && *max_size_str) {
        size_t value = jitc_cache_parse_size(max_size_str);
        if (value == SIZE_MAX)
            jitc_log(Warn,
                     "jit_init(): could not parse DRJIT_CACHE_MAXSIZE=\"%s\", "
                     "using the default of %zu bytes instead.",
                     max_size_str, jitc_cache_max_size);
        else
            jitc_cache_max_size = value;
    }

    fs::path path;
    if (const char *dir = getenv("DRJIT_CACHE_DIR"); dir && *dir)
        path = dir;
    else if (!jitc_cache_default_path(path))
        return;

    std::string path_str = path.string();
    std::error_code ec;

    // DRJIT_CACHE_DIR may well point somewhere without an existing parent
    fs::create_directories(path, ec);

    if (!fs::is_directory(path, ec)) {
        jitc_log(Warn, "jit_init(): could not use \"%s\" as the kernel cache "
                       "directory, the disk cache is disabled.",
                 path_str.c_str());
        return;
    }

    jitc_cache_path = std::move(path);
    jitc_cache_path_str = std::move(path_str);

    // To periodically prune the kernel cache without every process redoing that
    // work, Dr.Jit installs a stamp file in the cache directory. Its
    // modification time marks the start of the last sweep, which limits sweeps
    // to one per hour and machine. Its existence acts as a lock:
    // jitc_cache_sweep() claims a sweep by deleting the stamp, which can only
    // succeed in one process. It then recreates it to keep the others out.

    // Creating the file in append mode leaves an existing stamp untouched.
    // A newly created stamp is backdated so that a fresh cache directory
    // gets swept right away instead of an hour later.
    fs::path stamp = jitc_cache_path / DRJIT_CACHE_STAMP;
    if (!fs::exists(stamp, ec) && std::ofstream(stamp, std::ios::app).good())
        fs::last_write_time(stamp, fs::file_time_type::clock::now() -
                                       2 * jitc_cache_sweep_interval, ec);
}

void jitc_cache_shutdown() {
    jitc_cache_path.clear();
    jitc_cache_path_str.clear();
    jitc_cache_read_only = false;
    jitc_cache_initialized = false;
}

const char *jitc_cache_dir() {
    return jitc_cache_path.empty() ? nullptr : jitc_cache_path_str.c_str();
}

bool jitc_cache_writable() {
    return !jitc_cache_path.empty() && !jitc_cache_read_only;
}

// ============================================================================
//  Loading and storing kernels
// ============================================================================

/// Releases a malloc()-ed buffer when it goes out of scope
struct ScopedBuffer {
    char *ptr = nullptr;
    ~ScopedBuffer() { free(ptr); }
};

/// A content error means the entry is unusable, so delete it. An I/O error says
/// nothing about the content and must never delete anything.
static bool jitc_cache_reject(const fs::path &path, const char *reason) {
    jitc_log(Debug, "jit_cache_blob_load(): discarding cache file \"%s\": %s.",
             path.string().c_str(), reason);
    if (!jitc_cache_read_only) {
        std::error_code ec;
        fs::remove(path, ec);
    }
    return false;
}

bool jitc_cache_blob_load(const char *kind, XXH128_hash_t hash,
                          uint32_t check, std::vector<uint8_t> &data) {
    if (jitc_cache_path.empty())
        return false;

    fs::path path = jitc_cache_entry(hash, kind);
    std::error_code ec;

    // A missing entry is an ordinary miss; anything odd is left well alone
    if (!fs::is_regular_file(path, ec))
        return false;

    uintmax_t file_size = fs::file_size(path, ec);
    if (ec)
        return false;

    jitc_lz4_init();

    if (file_size < sizeof(CacheFileHeader))
        return jitc_cache_reject(path, "truncated");

    // A stream that failed to open reports every subsequent read as an error
    std::ifstream file(path, std::ios::binary);
    CacheFileHeader header;

    if (!file.read((char *) &header, sizeof(CacheFileHeader)))
        return false;

    if (header.version != DRJIT_CACHE_VERSION)
        return jitc_cache_reject(path, "incompatible format version");

    // Bounds the allocation below, which precedes the checksum test
    if (file_size != sizeof(CacheFileHeader) + (uintmax_t) header.compressed_size)
        return jitc_cache_reject(path, "size disagrees with the header");

    if (header.check != check)
        return jitc_cache_reject(path, "hash collision");

    ScopedBuffer compressed, buf;
    compressed.ptr = (char *) malloc_check(header.compressed_size);

    if (!file.read(compressed.ptr, header.compressed_size))
        return false;

    file.close();

    if (!XXH128_isEqual(header.checksum,
                        jitc_cache_checksum(header, compressed.ptr)))
        return jitc_cache_reject(path, "checksum mismatch");

    // Decompress with the shared dictionary, which must precede the payload
    // in memory
    buf.ptr = (char *) malloc_check(size_t(header.payload_size) +
                                    jitc_lz4_dict_size);
    memcpy(buf.ptr, jitc_lz4_dict, jitc_lz4_dict_size);

    uint32_t rv = (uint32_t) LZ4_decompress_safe_usingDict(
        compressed.ptr, buf.ptr + jitc_lz4_dict_size,
        (int) header.compressed_size, (int) header.payload_size,
        buf.ptr, jitc_lz4_dict_size);

    if (rv != header.payload_size)
        return jitc_cache_reject(path, "malformed");

    jitc_log(Trace, "jit_cache_blob_load(\"%s\")", path.string().c_str());

    // Refresh the timestamp so that the sweep evicts in least-recently-used
    // rather than first-in-first-out order. The day-long threshold is
    // relatime's own heuristic, and keeps this essentially free.
    if (!jitc_cache_read_only) {
        auto now = fs::file_time_type::clock::now();
        if (now - fs::last_write_time(path, ec) > jitc_cache_touch_interval)
            fs::last_write_time(path, now, ec);
    }

    const uint8_t *payload = (const uint8_t *) buf.ptr + jitc_lz4_dict_size;
    data.assign(payload, payload + header.payload_size);
    return true;
}

bool jitc_cache_blob_store(const char *kind, XXH128_hash_t hash,
                           uint32_t check, const uint8_t *data, size_t size,
                           bool replace) {
    if (!jitc_cache_writable())
        return false;

    // LZ4 tops out at INT32_MAX, well below the range of 'payload_size'
    if (size > (size_t) INT32_MAX) {
        jitc_log(Warn, "jit_cache_blob_store(): the compiled artifact is too "
                       "large to be cached (%zu bytes).", size);
        return false;
    }

    jitc_lz4_init();

    CacheFileHeader header;
    header.version = DRJIT_CACHE_VERSION;
    header.check = check;
    header.payload_size = (uint32_t) size;

    // The temporary name carries the pid: a process that dies before the
    // cleanup below must not poison this entry for every future process.
    char tmp_suffix[24];
    snprintf(tmp_suffix, sizeof(tmp_suffix), ".%u.tmp", (unsigned) jitc_getpid());

    fs::path path = jitc_cache_entry(hash, kind), path_tmp = path;
    path_tmp += tmp_suffix;

    std::string filename = path.string();

    std::ofstream file(path_tmp, std::ios::binary | std::ios::trunc);
    if (!file) {
        if (errno == EACCES || errno == EPERM || errno == EROFS)
            jitc_cache_read_only = true;
        jitc_log(Warn, "jit_cache_blob_store(): could not write compiled "
                       "artifact to cache file \"%s\": %s",
                 path_tmp.string().c_str(), strerror(errno));
        return false;
    }

    uint32_t out_size = LZ4_compressBound((int) size);
    uint8_t *temp_out = (uint8_t *) malloc_check(out_size);

    LZ4_stream_t stream;
    memset(&stream, 0, sizeof(LZ4_stream_t));
    LZ4_resetStream_fast(&stream);
    LZ4_loadDict(&stream, jitc_lz4_dict, jitc_lz4_dict_size);

    header.compressed_size = (uint32_t) LZ4_compress_fast_continue(
        &stream, (const char *) data, (char *) temp_out, (int) size,
        (int) out_size, 1);

    // Every other header field must be final by this point
    header.checksum = jitc_cache_checksum(header, (const char *) temp_out);

    file.write((const char *) &header, sizeof(CacheFileHeader));
    file.write((const char *) temp_out, header.compressed_size);
    file.close();

    // close() reports a failure to flush, so this also covers deferred errors
    bool success = file.good();

    if (!success)
        jitc_log(Warn, "jit_cache_blob_store(): I/O error while writing "
                       "cache file \"%s\".", path_tmp.string().c_str());
    else if (jitc_log_active(LogLevel::Trace))
        jitc_trace("jit_cache_blob_store(\"%s\"): compressed %s to %s",
                  filename.c_str(),
                  std::string(jitc_mem_string(size)).c_str(),
                  std::string(jitc_mem_string(header.compressed_size)).c_str());

    std::error_code ec;
    if (success) {
        // Prevent tampering with these files
        fs::permissions(path_tmp, fs::perms::owner_read | fs::perms::owner_write |
                                  fs::perms::group_read | fs::perms::others_read, ec);

        // Publish atomically
        if (replace) {
            fs::rename(path_tmp, path, ec);
        } else {
            fs::create_hard_link(path_tmp, path, ec);
            // A pre-existing entry is equivalent, so leave it in place
            if (ec == std::errc::file_exists)
                ec.clear();
        }

        if (ec) {
            jitc_log(Warn, "jit_cache_blob_store(): could not link the cache "
                           "file \"%s\" into the file system: %s",
                     filename.c_str(), ec.message().c_str());
            success = false;
        }
    }

    fs::remove(path_tmp, ec);

    if (unlikely(jitc_cache_train)) {
        // Uncompressed payload, for retraining the LZ4 dictionary
        fs::path path_trn = path;
        path_trn.replace_extension("trn");
        std::ofstream(path_trn, std::ios::binary).write((const char *) data, size);
    }

    free(temp_out);

    return success;
}

// ============================================================================
//  Cache sweep
// ============================================================================

// Prune the kernel cache on a separate thread
static void jitc_cache_sweep_thread(fs::path dir, size_t max_size, bool verbose) {
    struct Entry {
        fs::path path;
        uintmax_t size;
        fs::file_time_type mtime;
    };

    std::vector<Entry> entries;
    uintmax_t total = 0, removed = 0, removed_size = 0;
    std::error_code ec;

    try {
        // Collect the whole listing before removing anything
        for (const fs::directory_entry &entry : fs::directory_iterator(dir, ec)) {
            if (entry.path().extension() != ".bin")
                continue;

            std::error_code e_type, e_size, e_time;
            bool regular = entry.is_regular_file(e_type);
            uintmax_t size = entry.file_size(e_size);
            fs::file_time_type mtime = entry.last_write_time(e_time);

            // A peer that removes an entry mid-scan makes file_size() return -1
            if (!regular || e_type || e_size || e_time)
                continue;

            entries.push_back({ entry.path(), size, mtime });
            total += size;
        }

        if (max_size && total > max_size) {
            std::sort(entries.begin(), entries.end(),
                      [](const Entry &a, const Entry &b) { return a.mtime < b.mtime; });

            // Evict slightly past the budget
            uintmax_t target = max_size / 5 * 4;

            for (const Entry &e : entries) {
                if (total <= target)
                    break;
                if (fs::remove(e.path, ec)) {
                    total -= e.size;
                    removed_size += e.size;
                    removed++;
                }
            }
        }
    } catch (...) {
        // A partial sweep is harmless
    }

    if (verbose)
        fprintf(stderr,
                "jit_cache_sweep(): removed %ju of %zu file%s (%ju bytes), "
                "%ju bytes remaining.\n",
                removed, entries.size(), entries.size() == 1 ? "" : "s",
                removed_size, total);
}

void jitc_cache_sweep() {
    if (jitc_cache_path.empty() || jitc_cache_read_only)
        return;

    fs::path stamp = jitc_cache_path / DRJIT_CACHE_STAMP;
    std::error_code ec;

    auto now = fs::file_time_type::clock::now(),
         mtime = fs::last_write_time(stamp, ec);

    // A timestamp from the future indicates clock skew: sweep anyway
    if (!ec && mtime <= now && now - mtime < jitc_cache_sweep_interval)
        return;

    // This removal can only succeed for one process, which is then responsible for
    // the cache sweep pass.
    if (!fs::remove(stamp, ec))
        return;

    std::ofstream(stamp).close(); // Creating the file already stamps it with 'now'

    // The thread performs only idempotent filesystem operations and can be detached
    try {
        std::thread(jitc_cache_sweep_thread, jitc_cache_path, jitc_cache_max_size,
                    jitc_cache_verbose).detach();
    } catch (...) { }
}

// ============================================================================

void jitc_kernel_free(int device_id, const Kernel &kernel) {
    delete[] kernel.param_info;
    free(kernel.src);

    if (device_id == -1) {
        // The per-unit images referenced by 'reloc' are owned by the unit
        // cache and released in jitc_unit_cache_flush()
        free(kernel.llvm.reloc);
    } else {
#if defined(DRJIT_ENABLE_METAL)
        if ((state.backends & (1u << (uint32_t) JitBackend::CUDA)) == 0 &&
            (state.backends & (1u << (uint32_t) JitBackend::Metal)) != 0) {
            // Metal kernel.
            extern void jitc_metal_kernel_free(Kernel &);
            jitc_metal_kernel_free(const_cast<Kernel &>(kernel));
            return;
        }
#endif
#if defined(DRJIT_ENABLE_CUDA)
        const CUDADevice &device = state.devices.at(device_id);
        scoped_set_context guard(device.context);
        if (kernel.size) {
            cuda_check(cuModuleUnload(kernel.cuda.mod));
        } else {
#if defined(DRJIT_ENABLE_OPTIX)
            jitc_optix_free(kernel);
#endif
        }
#else
        (void) device_id;
        (void) kernel;
#endif
    }
}

void jitc_flush_kernel_cache() {
    // Drain in-flight work first
    do {
        jitc_sync_thread();
    } while (jitc_task);

    jitc_log(Info, "jit_flush_kernel_cache(): releasing %zu kernel%s ..",
            state.kernel_cache.size(),
            state.kernel_cache.size() > 1 ? "s" : "");

    for (auto &v : state.kernel_cache) {
        jitc_kernel_free(v.first.device, v.second);
    }

    state.kernel_cache.clear();
    state.kernel_cache_generation++;

    // Kernels referencing the per-unit artifacts are gone; now evict those
    jitc_unit_cache_flush();
}
