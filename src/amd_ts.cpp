#include "amd_ts.h"

#if defined(DRJIT_ENABLE_AMD)

#include "amd_api.h"
#include "amd_gemm.h"
#include "amd_misc.h"
#include "amd_reduce.h"
#include "amd_rt.h"
#include "eval.h"
#include "internal.h"
#include "log.h"
#include "malloc.h"
#include "util.h"
#include "var.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <vector>

static hipFunction_t jitc_amd_builtin_function(AMDThreadState *ts,
                                               const char *source,
                                               std::vector<hipModule_t> &modules,
                                               const char *name) {
    size_t device_count = state.amd_devices.size();
    if (modules.size() < device_count)
        modules.resize(device_count, nullptr);

    hipModule_t &module = modules[(size_t) ts->device];
    if (!module) {
        Kernel kernel;
        memset(&kernel, 0, sizeof(Kernel));

        jitc_amd_compile(ts, source, strlen(source), name, kernel);

        AMDDevice &dev = state.amd_devices[(size_t) ts->device];
        module = kernel.amd.mod;
        dev.modules.push_back(module);
        return kernel.amd.func;
    }

    hipFunction_t result = nullptr;
    hip_check(hipModuleGetFunction(&result, module, name));
    return result;
}

static std::vector<hipFunction_t>
    jitc_amd_gemm[(int) VarType::Count][4][3];
static std::vector<hipModule_t> jitc_amd_gemm_module;
static std::vector<hipModule_t> jitc_amd_misc_module;
static std::vector<hipModule_t> jitc_amd_reduce_module;

static hipFunction_t jitc_amd_gemm_function(AMDThreadState *ts, VarType vt,
                                            int tile, int transpose) {
    if (vt != VarType::Float16 && vt != VarType::Float32 &&
        vt != VarType::Float64)
        return nullptr;

    std::vector<hipFunction_t> &functions =
        jitc_amd_gemm[(int) vt][tile][transpose];
    size_t device_count = state.amd_devices.size();
    if (functions.size() < device_count)
        functions.resize(device_count, nullptr);

    hipFunction_t &slot = functions[(size_t) ts->device];
    if (slot)
        return slot;

    static const char *gemm_suffix[3] = { "nn", "nt", "tn" };
    char name[128];
    snprintf(name, sizeof(name), "gemm_%s_%u_%s",
             type_name_short[(int) vt], 8u << tile,
             gemm_suffix[transpose]);

    slot = jitc_amd_builtin_function(ts, jitc_amd_gemm_source,
                                     jitc_amd_gemm_module, name);

    return slot;
}

static VarType amd_make_int_type_unsigned(VarType type) {
    switch (type) {
        case VarType::Int8:  return VarType::UInt8;
        case VarType::Int16: return VarType::UInt16;
        case VarType::Int32: return VarType::UInt32;
        case VarType::Int64: return VarType::UInt64;
        default: return type;
    }
}

static hipFunction_t jitc_amd_reduce_function(AMDThreadState *ts,
                                              const char *name) {
    return jitc_amd_builtin_function(ts, jitc_amd_reduce_source,
                                     jitc_amd_reduce_module, name);
}

static hipFunction_t jitc_amd_block_reduce_function(AMDThreadState *ts,
                                                    ReduceOp op, VarType vt,
                                                    int kernel_id) {
    char name[128];
    snprintf(name, sizeof(name), "block_reduce_%s_%s_%u",
             red_name[(int) op], type_name_short[(int) vt],
             1u << (kernel_id + 1));
    return jitc_amd_reduce_function(ts, name);
}

static hipFunction_t jitc_amd_block_reduce_vec_function(AMDThreadState *ts,
                                                        ReduceOp op,
                                                        VarType vt) {
    char name[128];
    snprintf(name, sizeof(name), "block_reduce_%s_%s_vec_1024",
             red_name[(int) op], type_name_short[(int) vt]);
    return jitc_amd_reduce_function(ts, name);
}

static hipFunction_t jitc_amd_block_prefix_reduce_function(AMDThreadState *ts,
                                                           ReduceOp op,
                                                           VarType vt,
                                                           int kernel_id) {
    char name[128];
    snprintf(name, sizeof(name), "block_prefix_reduce_%s_%s_%u",
             red_name[(int) op], type_name_short[(int) vt],
             1u << (kernel_id + 1));
    return jitc_amd_reduce_function(ts, name);
}

static hipFunction_t jitc_amd_reduce_dot_function(AMDThreadState *ts,
                                                  VarType vt) {
    char name[128];
    snprintf(name, sizeof(name), "reduce_dot_%s", type_name_short[(int) vt]);
    return jitc_amd_reduce_function(ts, name);
}

static hipFunction_t jitc_amd_misc_function(AMDThreadState *ts,
                                            const char *name) {
    return jitc_amd_builtin_function(ts, jitc_amd_misc_source,
                                     jitc_amd_misc_module, name);
}

static hipFunction_t jitc_amd_poke_function(AMDThreadState *ts, VarType vt) {
    char name[128];
    snprintf(name, sizeof(name), "poke_%s", type_name_short[(int) vt]);
    return jitc_amd_misc_function(ts, name);
}

static void amd_submit_gpu(AMDThreadState *ts, KernelType type,
                           hipFunction_t kernel, uint32_t block_count_x,
                           uint32_t thread_count,
                           uint32_t shared_mem_bytes, void **args,
                           uint32_t width, uint32_t block_count_y = 1,
                           uint32_t block_count_z = 1) {
    KernelHistoryEntry entry = {};
    uint32_t flags = jit_flags();

    hip_check(hipCtxSetCurrent(ts->amd_context));

    if (unlikely(flags & (uint32_t) JitFlag::KernelHistory)) {
        hip_check(hipEventCreate((hipEvent_t *) &entry.event_start));
        hip_check(hipEventCreate((hipEvent_t *) &entry.event_end));
        hip_check(hipEventRecord((hipEvent_t) entry.event_start,
                                 ts->amd_stream));
    }

    hip_check(hipModuleLaunchKernel(kernel, block_count_x, block_count_y,
                                    block_count_z, thread_count, 1, 1,
                                    shared_mem_bytes, ts->amd_stream, args,
                                    nullptr));

    if (unlikely(flags & (uint32_t) JitFlag::LaunchBlocking))
        hip_check(hipStreamSynchronize(ts->amd_stream));

    if (unlikely(flags & (uint32_t) JitFlag::KernelHistory)) {
        entry.backend = JitBackend::AMD;
        entry.type = type;
        entry.recording_mode = ts->recording_mode;
        entry.size = width;
        entry.input_count = 1;
        entry.output_count = 1;
        hip_check(hipEventRecord((hipEvent_t) entry.event_end,
                                 ts->amd_stream));
        state.kernel_history.append(entry);
    }
}

Task *AMDThreadState::launch(Kernel kernel, KernelKey &, XXH128_hash_t,
                             uint32_t size, std::vector<void *> &kernel_params,
                             const std::vector<uint32_t> &,
                             KernelHistoryEntry *kernel_history_entry) {
    if (kernel_history_entry) {
        auto &e = *kernel_history_entry;
        hip_check(hipEventCreate((hipEvent_t *) &e.event_start));
        hip_check(hipEventCreate((hipEvent_t *) &e.event_end));
        hip_check(hipEventRecord((hipEvent_t) e.event_start, amd_stream));
    }

    uint32_t kernel_param_count = (uint32_t) kernel_params.size();
    size_t param_size = (size_t) kernel_param_count * sizeof(void *);

    const KernelParamInfo *info = kernel.param_info;
    if (info) {
        for (uint32_t i = 1; i < kernel_param_count; ++i) {
            AMDScene *scene = (AMDScene *) kernel_params[i];
            switch ((ResourceKind) info[i].kind) {
                case ResourceKind::Accel:
                    kernel_params[i] = scene ? scene->scene : nullptr;
                    break;

                case ResourceKind::IFT:
                    kernel_params[i] = scene ? scene->func_table : nullptr;
                    break;

                default:
                    break;
            }
        }
    }

    uint8_t *params_global =
        (uint8_t *) jitc_malloc(JitBackend::AMD, param_size);

    hip_check(hipCtxSetCurrent(amd_context));
    hip_check(hipMemcpy(params_global, kernel_params.data(), param_size,
                        hipMemcpyHostToDevice));

    uint32_t block_count = 0, thread_count = 0;
    const AMDDevice &device_ = state.amd_devices[this->device];
    device_.get_launch_config(&block_count, &thread_count, size,
                              kernel.amd.block_size
                                  ? kernel.amd.block_size
                                  : amd_max_threads);

    void *args[] = { &params_global };
    hip_check(hipModuleLaunchKernel(kernel.amd.func,
                                    block_count, 1, 1,
                                    thread_count, 1, 1,
                                    0, amd_stream, args, nullptr));

    jitc_trace("jit_run(): launching %u thread%s in %u block%s ..",
               thread_count, thread_count == 1 ? "" : "s", block_count,
               block_count == 1 ? "" : "s");

    if (unlikely(jit_flag(JitFlag::LaunchBlocking)))
        hip_check(hipStreamSynchronize(amd_stream));

    hip_check(hipLaunchHostFunc(
        amd_stream,
        [](void *payload) { jitc_free(payload); },
        params_global));

    if (kernel_history_entry) {
        hip_check(hipEventRecord((hipEvent_t) kernel_history_entry->event_end,
                                 amd_stream));
        state.kernel_history.append(*kernel_history_entry);
    }

    return nullptr;
}

void AMDThreadState::memset_async(void *ptr, uint32_t size_, uint32_t isize,
                                  const void *src) {
    if (isize != 1 && isize != 2 && isize != 4 && isize != 8)
        jitc_raise("jit_memset_async(): invalid element size (must be 1, 2, "
                   "4, or 8)!");

    jitc_trace("jit_memset_async(" DRJIT_PTR ", isize=%u, size=%u)",
               (uintptr_t) ptr, isize, size_);

    if (size_ == 0)
        return;

    size_t size = size_;
    uint64_t zero = 0;
    if (std::memcmp(src, &zero, isize) == 0) {
        hip_check(hipMemsetAsync(ptr, 0, size * isize, amd_stream));
        return;
    }

    hip_check(hipCtxSetCurrent(amd_context));

    switch (isize) {
        case 1:
            hip_check(hipMemsetD8Async((hipDeviceptr_t) ptr,
                                       ((const uint8_t *) src)[0], size,
                                       amd_stream));
            break;

        case 2:
            hip_check(hipMemsetD16Async((hipDeviceptr_t) ptr,
                                        ((const uint16_t *) src)[0], size,
                                        amd_stream));
            break;

        case 4:
            hip_check(hipMemsetD32Async((hipDeviceptr_t) ptr,
                                        (int) ((const uint32_t *) src)[0],
                                        size, amd_stream));
            break;

        case 8: {
            const AMDDevice &dev = state.amd_devices[(size_t) device];
            uint32_t block_count, thread_count;
            dev.get_launch_config(&block_count, &thread_count, size_);
            hipFunction_t func = jitc_amd_misc_function(this, "fill_64");
            void *args[] = { &ptr, &size_, (void *) src };
            amd_submit_gpu(this, KernelType::Memset, func, block_count,
                           thread_count, 0, args, size_);
            break;
        }
    }
}

void AMDThreadState::block_reduce(VarType vt, ReduceOp op, uint32_t size,
                                  uint32_t block_size, const void *in,
                                  void *out) {
    if (size == 0) {
        return;
    } else if (block_size == 0 || block_size > size) {
        jitc_raise("jit_block_reduce(): invalid block size "
                   "(size=%u, block_size=%u)!", size, block_size);
    }

    uint32_t tsize = type_size[(int) vt];
    if (block_size == 1) {
        memcpy_async(out, in, (size_t) size * tsize);
        return;
    }

    VarType vts = vt;
    if (op == ReduceOp::Add || op == ReduceOp::Mul ||
        op == ReduceOp::Or || op == ReduceOp::And) {
        vt = amd_make_int_type_unsigned(vt);
        if (vt == VarType::Float16)
            vts = VarType::Float32;
    }

    uint32_t block_count = ceil_div(size, block_size);
    uint32_t chunk_size = round_pow2(block_size);

    uint32_t thread_count, grid_dim_x, grid_dim_y, chunk_count;
    uint32_t chunks_per_block, chunks_per_thread_block;
    bool x_is_block_id = true;
    uint32_t vector_width = 1;

    if (chunk_size < 1024) {
        uint32_t min_threads = align_up(block_count * chunk_size, 32);
        thread_count = std::min(std::max(chunk_size, 128u), min_threads);

        chunk_count = block_count;
        chunks_per_block = 1;
        chunks_per_thread_block = thread_count / chunk_size;

        grid_dim_x = ceil_div(block_count, chunks_per_thread_block);
        grid_dim_y = 1;
    } else {
        if ((block_size * tsize) % 16 == 0 && (size * tsize) % 16 == 0 &&
            ((uintptr_t) in) % 16 == 0)
            vector_width = 16 / tsize;

        chunk_size = 1024;
        thread_count = chunk_size / vector_width;

        chunks_per_block = ceil_div(block_size, chunk_size);
        chunks_per_thread_block = 1;

        grid_dim_x = block_count;
        grid_dim_y = chunks_per_block;
        chunk_count = block_count * chunks_per_block;

        if (grid_dim_y > grid_dim_x) {
            std::swap(grid_dim_x, grid_dim_y);
            x_is_block_id = false;
        }
    }

    uint32_t after_stage_1 =
        chunk_size / (std::min(chunk_size, 32u) * vector_width);
    uint32_t smem_per_chunk =
        (after_stage_1 == 1 ? 0 : after_stage_1) * type_size[(int) vts],
        smem_bytes = smem_per_chunk * chunks_per_thread_block;

    jitc_log(Debug,
             "jit_amd_block_reduce(" DRJIT_PTR " -> " DRJIT_PTR
             ", type=%s, op=%s, size=%u, block_size=%u, block_count=%u, "
             "chunk_size=%u, chunks_per_block=%u, vector_width=%u): "
             "launching a %u x %u grid with %u threads and %u bytes of "
             "shared memory per thread block.",
             (uintptr_t) in, (uintptr_t) out, type_name[(int) vt],
             red_name[(int) op], size, block_size, block_count, chunk_size,
             chunks_per_block, vector_width, grid_dim_x, grid_dim_y,
             thread_count, smem_bytes);

    hipFunction_t func = nullptr;
    if (vector_width != 1) {
        func = jitc_amd_block_reduce_vec_function(this, op, vt);
    } else {
        int kernel_id = log2i_ceil(chunk_size) - 1;
        func = jitc_amd_block_reduce_function(this, op, vt, kernel_id);
    }

    if (!func)
        jitc_raise("jit_block_reduce(): no existing kernel for type=%s, "
                   "op=%s, vector_width=%u!",
                   type_name[(int) vt], red_name[(int) op], vector_width);

    struct {
        const void *in;
        void *out;
        uint32_t size;
        uint32_t block_size;
        uint32_t chunks_per_block;
        uint32_t chunk_count;
        uint8_t x_is_block_id;
    } params;

    params.in = in;
    params.size = size / vector_width;
    params.block_size = block_size / vector_width;
    params.chunks_per_block = chunks_per_block;
    params.chunk_count = chunk_count;
    params.x_is_block_id = (uint8_t) x_is_block_id;

    if (chunks_per_block == 1)
        params.out = out;
    else
        params.out = jitc_malloc(JitBackend::AMD, chunk_count * tsize);

    void *args[] = { &params };
    amd_submit_gpu(this, KernelType::BlockReduce, func, grid_dim_x,
                   thread_count, smem_bytes, args, size, grid_dim_y);

    if (chunks_per_block > 1) {
        block_reduce(vt, op, chunk_count, chunks_per_block, params.out, out);
        jitc_free(params.out);
    }
}

void AMDThreadState::block_prefix_reduce(VarType vt, ReduceOp op, uint32_t size,
                                         uint32_t block_size, bool exclusive,
                                         bool reverse, const void *in,
                                         void *out) {
    uint32_t tsize = type_size[(int) vt];
    if (size == 0) {
        return;
    } else if (block_size == 0 || block_size > size) {
        jitc_raise("jit_block_prefix_reduce(): invalid block size "
                   "(size=%u, block_size=%u)!", size, block_size);
    } else if (block_size == 1) {
        if (exclusive) {
            uint64_t ident = jitc_reduce_identity(vt, op);
            memset_async(out, size, tsize, &ident);
        } else if (in != out) {
            memcpy_async(out, in, (size_t) size * tsize);
        }
        return;
    }

    VarType vts = vt;
    if (op == ReduceOp::Add || op == ReduceOp::Mul ||
        op == ReduceOp::Or || op == ReduceOp::And) {
        vt = amd_make_int_type_unsigned(vt);
        if (vt == VarType::Float16)
            vts = VarType::Float32;
    }

    uint32_t block_count = ceil_div(size, block_size);
    uint32_t chunk_size = round_pow2(block_size);

    uint32_t thread_count, grid_dim_x, grid_dim_y, chunk_count;
    uint32_t chunks_per_block, chunks_per_thread_block;
    bool x_is_block_id = true;

    if (chunk_size < 1024) {
        uint32_t min_threads = align_up(block_count * chunk_size, 32);
        thread_count = std::min(std::max(chunk_size, 128u), min_threads);

        chunk_count = block_count;
        chunks_per_block = 1;
        chunks_per_thread_block = thread_count / chunk_size;

        grid_dim_x = ceil_div(block_count, chunks_per_thread_block);
        grid_dim_y = 1;
    } else {
        chunk_size = thread_count = 1024;
        chunks_per_block = ceil_div(block_size, chunk_size);
        chunks_per_thread_block = 1;

        grid_dim_x = block_count;
        grid_dim_y = chunks_per_block;
        chunk_count = block_count * chunks_per_block;

        if (grid_dim_y > grid_dim_x) {
            std::swap(grid_dim_x, grid_dim_y);
            x_is_block_id = false;
        }
    }

    uint32_t smem_bytes = thread_count * type_size[(int) vts];

    jitc_log(Debug,
             "jit_amd_block_prefix_reduce(" DRJIT_PTR " -> " DRJIT_PTR
             ", type=%s, op=%s, size=%u, block_size=%u, exclusive=%i, "
             "reverse=%i, block_count=%u, chunk_size=%u, "
             "chunks_per_block=%u): launching a %u x %u grid with %u "
             "threads and %u bytes of shared memory per thread block.",
             (uintptr_t) in, (uintptr_t) out, type_name[(int) vt],
             red_name[(int) op], size, block_size, exclusive, reverse,
             block_count, chunk_size, chunks_per_block, grid_dim_x,
             grid_dim_y, thread_count, smem_bytes);

    int kernel_id = log2i_ceil(chunk_size) - 1;
    hipFunction_t func =
        jitc_amd_block_prefix_reduce_function(this, op, vt, kernel_id);

    if (!func)
        jitc_raise("jit_block_prefix_reduce(): no existing kernel for "
                   "type=%s, op=%s!",
                   type_name[(int) vt], red_name[(int) op]);

    struct {
        const void *in;
        void *scratch;
        void *out;
        uint32_t size;
        uint32_t block_size;
        uint32_t chunks_per_block;
        uint8_t x_is_block_id;
        uint8_t exclusive;
        uint8_t reverse;
    } params;

    params.in = in;
    params.out = out;
    params.size = size;
    params.block_size = block_size;
    params.chunks_per_block = chunks_per_block;
    params.x_is_block_id = (uint8_t) x_is_block_id;
    params.exclusive = (uint8_t) exclusive;
    params.reverse = (uint8_t) reverse;

    if (chunks_per_block > 1) {
        uint32_t scratch_size = chunk_count * 2,
                 vsize = type_size[(int) vts];
        params.scratch = jitc_malloc(JitBackend::AMD, scratch_size * vsize);
        uint64_t z = 0;
        memset_async(params.scratch, scratch_size, vsize, &z);
    } else {
        params.scratch = nullptr;
    }

    void *args[] = { &params };
    amd_submit_gpu(this, KernelType::BlockPrefixReduce, func, grid_dim_x,
                   thread_count, smem_bytes, args, size, grid_dim_y);

    if (chunks_per_block > 1)
        jitc_free(params.scratch);
}

void AMDThreadState::reduce_dot(VarType vt, const void *ptr_1,
                                const void *ptr_2, uint32_t size, void *out) {
    if (size == 0)
        return;

    const AMDDevice &dev = state.amd_devices[(size_t) device];
    hipFunction_t red_dot = jitc_amd_reduce_dot_function(this, vt);

    if (!red_dot)
        jitc_raise("jit_reduce_dot(): no existing kernel for type=%s!",
                   type_name[(int) vt]);

    uint32_t thread_count = 1024,
             tsize = type_size[(int) vt],
             shared_size = thread_count * tsize,
             block_count = (size + thread_count * 2 - 1) / (thread_count * 2);

    block_count = std::min(dev.multi_processor_count * 4, block_count);

    jitc_log(Debug, "jit_amd_reduce_dot(" DRJIT_PTR ", " DRJIT_PTR
             ", type=%s, size=%u, smem=%u, blocks=%u)",
             (uintptr_t) ptr_1, (uintptr_t) ptr_2, type_name[(int) vt],
             size, shared_size, block_count);

    if (block_count == 1) {
        void *args[] = { (void *) &ptr_1, (void *) &ptr_2, &size, &out };
        amd_submit_gpu(this, KernelType::Dot, red_dot, 1, thread_count,
                       shared_size, args, size);
    } else {
        void *temp = jitc_malloc(JitBackend::AMD,
                                 block_count * (size_t) tsize);

        void *args[] = { (void *) &ptr_1, (void *) &ptr_2, &size, &temp };
        amd_submit_gpu(this, KernelType::Dot, red_dot, block_count,
                       thread_count, shared_size, args, size);

        block_reduce(vt, ReduceOp::Add, block_count, block_count, temp, out);
        jitc_free(temp);
    }
}

void AMDThreadState::batched_gemm(VarType vt, bool At, bool Bt, uint32_t M,
                                  uint32_t N, uint32_t K,
                                  const GemmBatch *batch, const void *A,
                                  const void *B, void *C) {
    if (At && Bt)
        jitc_raise("jit_batched_gemm(): internal error -- At=Bt=True "
                   "should have been rewritten by the caller.");

    uint32_t grid_count, reduce_count;
    if (!jitc_gemm_batch_counts(batch, grid_count, reduce_count))
        return;
    if (M == 0 || N == 0 || K == 0)
        return;

    GemmBatch batch_eff = batch ? *batch : GemmBatch{};
    uint32_t tsize = type_size[(int) vt];

    auto vec_width = [tsize](uint32_t bm) -> uint32_t {
        uint32_t tm   = bm / 8;
        uint32_t vmax = 16u / tsize;
        return tm < vmax ? tm : vmax;
    };

    uint32_t a_inner = At ? M : K,
             b_inner = Bt ? K : N;

    const AMDDevice &dev = state.amd_devices[(size_t) device];
    constexpr uint32_t grid_y_cap = 65535u;

    uint32_t small_dim = M < N ? M : N;
    int t_idx = At ? 2 : (Bt ? 1 : 0);
    uint32_t bm = 0;
    hipFunction_t func = nullptr;

    for (int l = 0; l <= 3; ++l) {
        uint32_t bm_try = 8u << l;
        uint32_t v      = vec_width(bm_try);

        if (a_inner % v || b_inner % v || (N % v))
            continue;
        if (ceil_div(M, bm_try) > grid_y_cap)
            continue;

        hipFunction_t f = jitc_amd_gemm_function(this, vt, l, t_idx);
        if (!f)
            continue;

        if (bm == 0) {
            bm = bm_try;
            func = f;
        }

        if (bm_try >= 2 * small_dim)
            continue;

        uint64_t grid = (uint64_t) ceil_div(M, bm_try) *
                        ceil_div(N, bm_try) * grid_count;
        uint64_t min_grid = (bm_try == 64)
                                ? 3u * dev.multi_processor_count
                                : dev.multi_processor_count;
        if (grid >= min_grid) {
            bm = bm_try;
            func = f;
        }
    }

    if (bm == 0)
        jitc_raise("jit_batched_gemm(): no compatible tile for M=%u, N=%u: "
                   "alignment or gridDim.y cap (%u) cannot be satisfied.",
                   M, N, grid_y_cap);

    uint32_t grid_x = ceil_div(N, bm),
             grid_y = ceil_div(M, bm);

    if (grid_count > 65535u)
        jitc_raise("jit_batched_gemm(): grid batch count %u exceeds the "
                   "current AMD GEMM gridDim.z limit of 65535.", grid_count);

    jitc_log(Debug,
             "jit_amd_batched_gemm(" DRJIT_PTR ", " DRJIT_PTR " -> "
             DRJIT_PTR ", type=%s, At=%i, Bt=%i, M=%u, N=%u, K=%u, "
             "grid=%u, reduce=%u, tile=%ux%u, launch=%ux%ux%u).",
             (uintptr_t) A, (uintptr_t) B, (uintptr_t) C,
             type_name[(int) vt], (int) At, (int) Bt, M, N, K, grid_count,
             reduce_count, bm, bm, grid_x, grid_y, grid_count);

    void *args[] = { (void *) &A, (void *) &B, &C,
                     &M, &N, &K, (void *) &batch_eff };

    amd_submit_gpu(this, KernelType::BatchedGemm, func, grid_x,
                   /* thread_count */ 64, /* shared_mem_bytes */ 0, args,
                   /* width */ grid_count * M * N, grid_y, grid_count);
}

uint32_t AMDThreadState::compress(const uint8_t *in, uint32_t size,
                                  uint32_t *out) {
    if (size == 0)
        return 0;

    const AMDDevice &dev = state.amd_devices[(size_t) device];
    hip_check(hipCtxSetCurrent(amd_context));

    uint32_t *count_out = (uint32_t *) jitc_malloc(
        JitBackend::AMD, sizeof(uint32_t), /*shared=*/true);

    if (size <= 4096) {
        uint32_t items_per_thread = 4,
                 thread_count =
                     round_pow2((size + items_per_thread - 1) /
                                items_per_thread),
                 shared_size = thread_count * 2 * sizeof(uint32_t),
                 trailer = thread_count * items_per_thread - size;

        jitc_log(Debug,
                 "jit_amd_compress(" DRJIT_PTR " -> " DRJIT_PTR
                 ", size=%u, type=small, threads=%u, shared=%u)",
                 (uintptr_t) in, (uintptr_t) out, size, thread_count,
                 shared_size);

        if (trailer > 0)
            hip_check(hipMemsetAsync((void *) (in + size), 0, trailer,
                                     amd_stream));

        hipFunction_t func = jitc_amd_misc_function(this, "compress_small");
        void *args[] = { (void *) &in, &out, &size, &count_out };
        amd_submit_gpu(this, KernelType::Compress, func, 1, thread_count,
                       shared_size, args, size);
    } else {
        uint32_t items_per_thread = 16,
                 thread_count = 128,
                 items_per_block = items_per_thread * thread_count,
                 block_count = (size + items_per_block - 1) / items_per_block,
                 shared_size = items_per_block * sizeof(uint32_t),
                 scratch_items = block_count + 32,
                 trailer = items_per_block * block_count - size;

        jitc_log(Debug,
                 "jit_amd_compress(" DRJIT_PTR " -> " DRJIT_PTR
                 ", size=%u, type=large, blocks=%u, threads=%u, shared=%u, "
                 "scratch=%u)",
                 (uintptr_t) in, (uintptr_t) out, size, block_count,
                 thread_count, shared_size, scratch_items * 4);

        uint64_t *scratch = (uint64_t *) jitc_malloc(
            JitBackend::AMD, scratch_items * sizeof(uint64_t));

        uint32_t block_count_init, thread_count_init;
        dev.get_launch_config(&block_count_init, &thread_count_init,
                              scratch_items);

        hipFunction_t init =
            jitc_amd_misc_function(this, "compress_large_init");
        void *args[] = { &scratch, &scratch_items };
        amd_submit_gpu(this, KernelType::Compress, init, block_count_init,
                       thread_count_init, 0, args, scratch_items);

        if (trailer > 0)
            hip_check(hipMemsetAsync((void *) (in + size), 0, trailer,
                                     amd_stream));

        scratch += 32;
        hipFunction_t func = jitc_amd_misc_function(this, "compress_large");
        void *args_2[] = { (void *) &in, &out, &scratch, &count_out };
        amd_submit_gpu(this, KernelType::Compress, func, block_count,
                       thread_count, shared_size, args_2, scratch_items);
        scratch -= 32;

        jitc_free(scratch);
    }

    jitc_sync_thread(this);
    uint32_t count_out_v = *count_out;
    jitc_free(count_out);
    return count_out_v;
}

static void amd_transpose(AMDThreadState *ts, const uint32_t *in,
                          uint32_t *out, uint32_t rows, uint32_t cols,
                          uint32_t num_batches = 1,
                          uint32_t batch_stride = 0) {
    uint16_t blocks_x = (uint16_t) ((cols + 15u) / 16u),
             blocks_y = (uint16_t) ((rows + 15u) / 16u);

    jitc_log(Debug,
             "jit_amd_transpose(" DRJIT_PTR " -> " DRJIT_PTR
             ", rows=%u, cols=%u, blocks=%ux%u, batches=%u)",
             (uintptr_t) in, (uintptr_t) out, rows, cols, blocks_x, blocks_y,
             num_batches);

    hipFunction_t func = jitc_amd_misc_function(ts, "transpose");
    void *args[] = { (void *) &in, &out, &rows, &cols, &batch_stride };
    hip_check(hipCtxSetCurrent(ts->amd_context));
    hip_check(hipModuleLaunchKernel(func, blocks_x, blocks_y, num_batches,
                                    16, 16, 1,
                                    16 * 17 * sizeof(uint32_t),
                                    ts->amd_stream, args, nullptr));
}

uint32_t AMDThreadState::block_mkperm(const uint32_t *values, uint32_t size,
                                      uint32_t block_size,
                                      uint32_t bucket_count, uint32_t *perm,
                                      uint32_t *offsets) {
    if (size == 0)
        return 0;
    else if (unlikely(bucket_count == 0))
        jitc_fail("jit_block_mkperm(): bucket_count cannot be zero!");

    hip_check(hipCtxSetCurrent(amd_context));
    const AMDDevice &dev = state.amd_devices[(size_t) device];

    const uint32_t warp_size = 32;

    uint32_t n_blocks = ceil_div(size, block_size);

    uint32_t gpu_blocks_per_group, thread_count;
    dev.get_launch_config(&gpu_blocks_per_group, &thread_count, block_size,
                          1024, 1);

    uint32_t warp_count = (thread_count + warp_size - 1) / warp_size;
    thread_count = warp_count * warp_size;

    uint32_t gpu_block_count = n_blocks * gpu_blocks_per_group;

    uint32_t bucket_size_1 = bucket_count * sizeof(uint32_t),
             bucket_size_all = bucket_size_1 * gpu_block_count;

    uint32_t shared_size = 0;
    const char *variant = nullptr;
    hipFunction_t phase_1 = nullptr, phase_4 = nullptr;
    bool initialize_buckets = false, is_tiny = false;

    if (bucket_size_1 * warp_count <= dev.shared_memory_bytes) {
        phase_1 = jitc_amd_misc_function(this, "block_mkperm_phase_1_tiny");
        phase_4 = jitc_amd_misc_function(this, "block_mkperm_phase_4_tiny");
        shared_size = bucket_size_1 * warp_count;
        bucket_size_all *= warp_count;
        variant = "tiny";
        is_tiny = true;
    } else if (bucket_size_1 <= dev.shared_memory_bytes) {
        phase_1 = jitc_amd_misc_function(this, "block_mkperm_phase_1_small");
        phase_4 = jitc_amd_misc_function(this, "block_mkperm_phase_4_small");
        shared_size = bucket_size_1;
        variant = "small";
    } else {
        phase_1 = jitc_amd_misc_function(this, "block_mkperm_phase_1_large");
        phase_4 = jitc_amd_misc_function(this, "block_mkperm_phase_4_large");
        variant = "large";
        initialize_buckets = true;
    }

    uint32_t rows_per_group =
        is_tiny ? gpu_blocks_per_group * warp_count : gpu_blocks_per_group;

    bool needs_transpose = rows_per_group > 1;
    uint32_t *buckets_1, *buckets_2, *counter = nullptr;
    buckets_1 = buckets_2 =
        (uint32_t *) jitc_malloc(JitBackend::AMD, bucket_size_all);

    if (needs_transpose)
        buckets_2 = (uint32_t *) jitc_malloc(JitBackend::AMD, bucket_size_all);

    if (offsets) {
        counter = (uint32_t *) jitc_malloc(JitBackend::AMD, sizeof(uint32_t));
        hip_check(hipMemsetAsync(counter, 0, sizeof(uint32_t), amd_stream));
    }

    if (initialize_buckets)
        hip_check(hipMemsetAsync(buckets_1, 0, bucket_size_all, amd_stream));

    uint32_t size_per_gpu_block =
        (block_size + gpu_blocks_per_group - 1) / gpu_blocks_per_group;

    jitc_log(Debug,
             "jit_amd_block_mkperm(" DRJIT_PTR
             ", size=%u, block_size=%u, bucket_count=%u, gpu_block_count=%u, "
             "thread_count=%u, size_per_gpu_block=%u, variant=%s, "
             "shared_size=%u)",
             (uintptr_t) values, size, block_size, bucket_count,
             gpu_block_count, thread_count, size_per_gpu_block, variant,
             shared_size);

    void *args_1[] = { (void *) &values, &buckets_1, &size,
                       &size_per_gpu_block, &bucket_count, &block_size };

    amd_submit_gpu(this, KernelType::MkPerm, phase_1, gpu_blocks_per_group,
                   thread_count, shared_size, args_1, size, n_blocks);

    if (needs_transpose) {
        uint32_t batch_stride = rows_per_group * bucket_count;
        amd_transpose(this, buckets_1, buckets_2, rows_per_group,
                      bucket_count, n_blocks, batch_stride);
    }

    uint32_t psum_count = bucket_size_all / sizeof(uint32_t);
    uint32_t psum_block_size = rows_per_group * bucket_count;
    block_prefix_reduce(VarType::UInt32, ReduceOp::Add, psum_count,
                        psum_block_size, true, false, buckets_2, buckets_2);

    if (needs_transpose) {
        uint32_t batch_stride = rows_per_group * bucket_count;
        amd_transpose(this, buckets_2, buckets_1, bucket_count,
                      rows_per_group, n_blocks, batch_stride);
    }

    if (likely(offsets) && n_blocks == 1) {
        uint32_t gpu_block_count_3, thread_count_3;
        dev.get_launch_config(&gpu_block_count_3, &thread_count_3,
                              bucket_count * gpu_block_count);

        uint32_t bucket_count_rounded =
            (bucket_count + thread_count_3 - 1) / thread_count_3 *
            thread_count_3;

        hipFunction_t phase_3 =
            jitc_amd_misc_function(this, "block_mkperm_phase_3");
        void *args_3[] = { &buckets_1, &bucket_count, &bucket_count_rounded,
                           &size, &counter, &offsets };

        amd_submit_gpu(this, KernelType::MkPerm, phase_3, gpu_block_count_3,
                       thread_count_3, sizeof(uint32_t) * thread_count_3,
                       args_3, size);

        hip_check(hipMemcpyAsync(offsets + 4 * (size_t) bucket_count, counter,
                                 sizeof(uint32_t), hipMemcpyDefault,
                                 amd_stream));

        hip_check(hipEventRecord(amd_event, amd_stream));
    }

    void *args_4[] = { (void *) &values, &buckets_1, &perm, &size,
                       &size_per_gpu_block, &bucket_count, &block_size };

    amd_submit_gpu(this, KernelType::MkPerm, phase_4, gpu_blocks_per_group,
                   thread_count, shared_size, args_4, size, n_blocks);

    if (likely(offsets) && n_blocks == 1) {
        unlock_guard guard(state.lock);
        hip_check(hipEventSynchronize(amd_event));
    }

    jitc_free(buckets_1);
    if (needs_transpose)
        jitc_free(buckets_2);
    jitc_free(counter);

    return (offsets && n_blocks == 1) ? offsets[4 * bucket_count] : 0u;
}

void AMDThreadState::memcpy(void *dst, const void *src, size_t size) {
    hip_check(hipCtxSetCurrent(amd_context));
    hip_check(hipStreamSynchronize(amd_stream));
    hip_check(hipMemcpy(dst, src, size, hipMemcpyDefault));
}

void AMDThreadState::memcpy_async(void *dst, const void *src, size_t size) {
    hip_check(hipCtxSetCurrent(amd_context));
    hip_check(hipMemcpyAsync(dst, src, size, hipMemcpyDefault, amd_stream));
}

void AMDThreadState::poke(void *dst, const void *src, uint32_t size) {
    jitc_log(Debug, "jit_poke(" DRJIT_PTR ", size=%u)", (uintptr_t) dst,
             size);

    VarType type;
    switch (size) {
        case 1: type = VarType::UInt8; break;
        case 2: type = VarType::UInt16; break;
        case 4: type = VarType::UInt32; break;
        case 8: type = VarType::UInt64; break;
        default:
            jitc_raise("jit_poke(): only size=1, 2, 4 or 8 are supported!");
    }

    hip_check(hipCtxSetCurrent(amd_context));
    hipFunction_t func = jitc_amd_poke_function(this, type);
    void *args[] = { &dst, (void *) src };
    amd_submit_gpu(this, KernelType::Poke, func, 1, 1, 0, args, 1);
}

void AMDThreadState::aggregate(void *dst, AggregationEntry *agg,
                               uint32_t size) {
    hip_check(hipCtxSetCurrent(amd_context));
    const AMDDevice &dev = state.amd_devices[(size_t) device];
    hipFunction_t func = jitc_amd_misc_function(this, "aggregate");

    for (uint32_t i = 0; i < size; ++i) {
        AggregationEntry &e = agg[i];
        if (e.size != 8)
            continue;

        AMDScene *scene = (AMDScene *) e.src;
        switch ((ResourceKind) e.resource_kind) {
            case ResourceKind::Accel:
                e.src = scene ? scene->scene : nullptr;
                break;

            case ResourceKind::IFT:
                e.src = scene ? scene->func_table : nullptr;
                break;

            default:
                break;
        }
    }

    void *args[] = { &dst, &agg, &size };

    uint32_t block_count, thread_count;
    dev.get_launch_config(&block_count, &thread_count, size);

    jitc_log(InfoSym,
             "jit_amd_aggregate(" DRJIT_PTR " -> " DRJIT_PTR
             ", size=%u, blocks=%u, threads=%u)",
             (uintptr_t) agg, (uintptr_t) dst, size, block_count,
             thread_count);

    amd_submit_gpu(this, KernelType::Aggregate, func, block_count,
                   thread_count, 0, args, 1);
}

void AMDThreadState::enqueue_host_func(void (*callback)(void *), void *payload) {
    hip_check(hipLaunchHostFunc(amd_stream, callback, payload));
}

void AMDThreadState::barrier() {
    if (!free_next.empty()) {
        free_later.insert(free_later.end(),
                          std::make_move_iterator(free_next.begin()),
                          std::make_move_iterator(free_next.end()));
        free_next.clear();
    }
    flush_deferred_free();
}

void AMDThreadState::flush_deferred_free() {
    if (void *batch = take_deferred_free())
        hip_check(hipLaunchHostFunc(
            amd_stream,
            [](void *payload) { jitc_malloc_release_batch(payload); },
            batch));
}

void AMDThreadState::coop_vec_pack(uint32_t count, const void *in_,
                                   const MatrixDescr *in_d, void *out_,
                                   const MatrixDescr *out_d) {
    (void) count; (void) in_; (void) in_d; (void) out_; (void) out_d;
    jitc_raise("AMDThreadState::coop_vec_pack(): cooperative vectors are not "
               "supported by the AMD/HIP backend without an OptiX-like HIPRT "
               "cooperative-vector facility.");
}

#endif
