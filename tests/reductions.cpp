#include "test.h"
#include <algorithm>

TEST_BOTH(01_all_any) {
    using Bool = Array<bool>;

    scoped_set_log_level ssll(LogLevel::Info);
    for (uint32_t i = 0; i < 100; ++i) {
        uint32_t size = 23*i*i*i + 1;

        Bool f = full<Bool>(false, size),
             t = full<Bool>(true, size);

        jit_var_eval(f.index());
        jit_var_eval(t.index());

        jit_assert(all(t) && any(t) && !all(f) && !any(f));

        for (int j = 0; j < 100; ++j) {
            uint32_t index = (uint32_t) rand() % size;
            f.write(index, true);
            t.write(index, false);

            if (size == 1)
                jit_assert(!all(t) && !any(t) && all(f) && any(f));
            else
                jit_assert(!all(t) && any(t) && !all(f) && any(f));

            f.write(index, false);
            t.write(index, true);
            jit_assert(all(t) && any(t) && !all(f) && !any(f));
        }
    }
}

TEST_BOTH(02_prefix_sum_exc_u32) {
    scoped_set_log_level ssll(LogLevel::Info);
    for (uint32_t i = 0; i < 100; ++i) {
        uint32_t size = 23*i*i*i + 1;

        UInt32 result, ref;

        if (i < 15) {
            result = arange<UInt32>(size);
            ref    = (result * (result - 1)) / 2;
        } else {
            result = full<UInt32>(1, size);
            ref    = arange<UInt32>(size);
        }
        jit_var_schedule(result.index());
        jit_var_schedule(ref.index());
        jit_eval();
        jit_prefix_sum(Float::Backend, VarType::UInt32, true, result.data(), size, result.data());
        jit_assert(result == ref);
    }
}

TEST_BOTH(03_prefix_sum_inc_u32) {
    scoped_set_log_level ssll(LogLevel::Info);
    for (uint32_t i = 0; i < 100; ++i) {
        uint32_t size = 23*i*i*i + 1;

        UInt32 result, ref;

        if (i < 15) {
            result = arange<UInt32>(size);
            ref    = ((result + 1) * result) / 2;
        } else {
            result = full<UInt32>(1, size);
            ref    = arange<UInt32>(size) + 1;
        }
        jit_var_schedule(result.index());
        jit_var_schedule(ref.index());
        jit_eval();
        jit_prefix_sum(Float::Backend, VarType::UInt32, false, result.data(), size, result.data());
        jit_assert(result == ref);
    }
}

TEST_BOTH(04_prefix_sum_exc_f32) {
    scoped_set_log_level ssll(LogLevel::Info);
    for (uint32_t i = 0; i < 80; ++i) {
        uint32_t size = 23*i*i*i + 1;

        Float result = full<Float>(1, size);
        Float ref    = arange<Float>(size);
        jit_var_schedule(result.index());
        jit_var_schedule(ref.index());
        jit_prefix_sum(Float::Backend, VarType::Float32, true, result.data(), size, result.data());
        float f = hsum(abs(result - ref)).read(0);
        jit_assert(f < 1e-6);
    }
}

TEST_BOTH(05_prefix_sum_inc_f32) {
    for (uint32_t i = 0; i < 80; ++i) {
        uint32_t size = 23*i*i*i + 1;

        Float result = full<Float>(1, size);
        Float ref    = arange<Float>(size) + 1.f;
        jit_var_schedule(result.index());
        jit_var_schedule(ref.index());
        jit_prefix_sum(Float::Backend, VarType::Float32, false, result.data(), size, result.data());
        float f = hsum(abs(result - ref)).read(0);
        jit_assert(f < 1e-6);
    }
}

TEST_BOTH(06_prefix_sum_exc_u64) {
    using UInt64 = typename UInt32::template ReplaceValue<uint64_t>;
    scoped_set_log_level ssll(LogLevel::Info);
    for (uint32_t i = 0; i < 100; ++i) {
        uint32_t size = 23*i*i*i + 1;

        UInt64 result, ref;

        if (i < 15) {
            result = arange<UInt64>(size);
            ref    = (result * (result - 1)) / 2;
        } else {
            result = full<UInt64>(1, size);
            ref    = arange<UInt64>(size);
        }
        jit_var_schedule(result.index());
        jit_var_schedule(ref.index());
        jit_eval();
        jit_prefix_sum(UInt64::Backend, VarType::UInt64, true, result.data(), size, result.data());

        jit_assert(result == ref);
    }
}

TEST_BOTH(07_prefix_sum_inc_u64) {
    using UInt64 = typename UInt32::template ReplaceValue<uint64_t>;
    scoped_set_log_level ssll(LogLevel::Info);
    for (uint32_t i = 0; i < 100; ++i) {
        uint32_t size = 23*i*i*i + 1;

        UInt64 result, ref;

        if (i < 15) {
            result = arange<UInt64>(size);
            ref    = (result * (result + 1)) / 2;
        } else {
            result = full<UInt64>(1, size);
            ref    = arange<UInt64>(size) + 1;
        }
        jit_var_schedule(result.index());
        jit_var_schedule(ref.index());
        jit_eval();
        jit_prefix_sum(UInt64::Backend, VarType::UInt64, false, result.data(), size, result.data());

        jit_assert(result == ref);
    }
}

TEST_BOTH(08_prefix_sum_exc_f64) {
    using Double = typename Float::template ReplaceValue<double>;
    scoped_set_log_level ssll(LogLevel::Info);
    for (uint32_t i = 0; i < 100; ++i) {
        uint32_t size = 23*i*i*i + 1;

        Double input  = full<Double>(1, size);
        Double output = empty<Double>(size);
        Double ref    = arange<Double>(size);
        jit_var_schedule(input.index());
        jit_var_schedule(ref.index());
        jit_eval();
        jit_prefix_sum(Double::Backend, VarType::Float64, true, input.data(), size, output.data());
        double f = hsum(abs(output - ref)).read(0);
        jit_assert(f < 1e-6);
    }
}

TEST_BOTH(09_prefix_sum_inc_f64) {
    using Double = typename Float::template ReplaceValue<double>;
    scoped_set_log_level ssll(LogLevel::Info);
    for (uint32_t i = 0; i < 100; ++i) {
        uint32_t size = 23*i*i*i + 1;

        Double input  = full<Double>(1, size);
        Double output = empty<Double>(size);
        Double ref    = arange<Double>(size)+1.0;
        jit_var_schedule(input.index());
        jit_var_schedule(ref.index());
        jit_eval();
        jit_prefix_sum(Double::Backend, VarType::Float64, false, input.data(), size, output.data());
        double f = hsum(abs(output - ref)).read(0);
        jit_assert(f < 1e-6);
    }
}


TEST_BOTH(10_compress) {
    scoped_set_log_level ssll(LogLevel::Info);
    for (uint32_t i = 0; i < 30; ++i) {
        uint32_t size = 23*i*i*i + 1;
        for (uint32_t j = 0; j <= i; ++j) {
            uint32_t n_ones = 23*j*j*j + 1;

            jit_log(LogLevel::Info, "===== size=%u, ones=%u =====", size, n_ones);
            uint8_t *data      = (uint8_t *) jit_malloc(AllocType::Host, size);
            uint32_t *perm     = (uint32_t *) jit_malloc(Float::Backend == JitBackend::CUDA ? AllocType::Device :
                                                                          AllocType::Host,
                                                          size * sizeof(uint32_t)),
                     *perm_ref = (uint32_t *) jit_malloc(AllocType::Host, size * sizeof(uint32_t));
            memset(data, 0, size);

            for (size_t k = 0; k < n_ones; ++k) {
                uint32_t index = rand() % size;
                data[index] = 1;
            }

            uint32_t ref_count = 0;
            for (size_t k = 0; k < size; ++k) {
                if (data[k])
                    perm_ref[ref_count++] = (uint32_t) k;
            }

            data = (uint8_t *) jit_malloc_migrate(
                data, Float::Backend == JitBackend::CUDA ? AllocType::Device : AllocType::Host);

            uint32_t count = jit_compress(Float::Backend, data, size, perm);
            perm = (uint32_t *) jit_malloc_migrate(perm, AllocType::Host);
            jit_sync_thread();

            jit_assert(count == ref_count);
            jit_assert(memcmp(perm, perm_ref, ref_count * sizeof(uint32_t)) == 0);

            jit_free(data);
            jit_free(perm);
            jit_free(perm_ref);
        }
    }
}

TEST_BOTH(11_mkperm) {
    scoped_set_log_level ssll(LogLevel::Info);
    srand(0);
    for (uint32_t i = 0; i < 30; ++i) {
        uint32_t size = 23*i*i*i + 1;
        for (uint32_t j = 0; j <= i; ++j) {
            uint32_t n_buckets = 23*j*j*j + 1;

            jit_log(LogLevel::Info, "===== size=%u, buckets=%u =====", size, n_buckets);
            uint32_t *data    = (uint32_t *) jit_malloc(AllocType::Host, size * sizeof(uint32_t)),
                     *perm    = (uint32_t *) jit_malloc(Float::Backend == JitBackend::CUDA ? AllocType::Device :
                                                                                             AllocType::Host,
                                                         size * sizeof(uint32_t)),
                     *offsets = (uint32_t *) jit_malloc(Float::Backend == JitBackend::CUDA ? AllocType::HostPinned :
                                                                                             AllocType::Host,
                                                         (n_buckets * 4 + 1) * sizeof(uint32_t));
            uint64_t *ref = new uint64_t[size];

            for (size_t k = 0; k < size; ++k) {
                uint32_t value = rand() % n_buckets;
                data[k] = value;
                ref[k] = (((uint64_t) value) << 32) | k;
            }

            data = (uint32_t *) jit_malloc_migrate(data, Float::Backend == JitBackend::CUDA ? AllocType::Device : AllocType::Host);
            uint32_t num_unique = jit_mkperm(Float::Backend, data, size, n_buckets, perm, offsets);

            perm = (uint32_t *) jit_malloc_migrate(perm, AllocType::Host);
            jit_sync_thread();

            struct Bucket {
                uint32_t id;
                uint32_t start;
                uint32_t size;
                uint32_t unused;
            };

            Bucket *buckets = (Bucket *) offsets;

            std::sort(ref, ref + size);
            std::sort(buckets, buckets + num_unique,
                [](const Bucket &a, const Bucket &b) {
                    return a.id < b.id;
                }
            );

            uint32_t total_size = 0;
            for (uint32_t k = 0; k < num_unique; ++k)
                total_size += buckets[k].size;
            if (total_size != size)
                jit_fail("Size mismatch: %u vs %u\n", total_size, size);

            const uint64_t *ref_ptr = ref;
            for (uint32_t l = 0; l < num_unique; ++l) {
                const Bucket &entry = buckets[l];
                uint32_t *perm_cur = perm + entry.start;

#if 0
                for (size_t k = 0; k < entry.size; ++k) {
                    uint64_t ref_value = ref_ptr[k];
                    uint32_t bucket_id = (uint32_t) (ref_value >> 32);
                    uint32_t perm_index = (uint32_t) ref_value;
                    fprintf(stderr, "id=%u/%u perm=%u/%u\n", entry.id, bucket_id, perm_cur[k], perm_index);
                }
#endif

                std::sort(perm_cur, perm_cur + entry.size);

                for (size_t k = 0; k < entry.size; ++k) {
                    uint64_t ref_value = *ref_ptr++;
                    uint32_t bucket_id = (uint32_t) (ref_value >> 32);
                    uint32_t perm_index = (uint32_t) ref_value;

                    if (bucket_id != entry.id)
                        jit_fail("Mismatched bucket ID");
                    if (perm_index != perm_cur[k])
                        jit_fail("Mismatched permutation index");
                }
            }
            jit_free(data);
            jit_free(perm);
            jit_free(offsets);
            delete[] ref;
        }
    }
}

#if 0
TEST_BOTH(12_block_ops) {
    Float a(0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f);

    jit_log(Info, "block_copy: %s\n", block_copy(a, 3).str());
    jit_log(Info, "block_sum:  %s\n", block_sum(a, 3).str());
}
#endif
