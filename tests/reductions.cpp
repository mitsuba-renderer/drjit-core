#include "test.h"
#include <algorithm>
#include <memory>

template <typename Value> inline Value fmix32(Value h) {
    h += 1;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

template <typename Value> inline Value block_sum_ref(uint32_t size, uint32_t block_size) {
    using Scalar = typename Value::Value;

    uint32_t blocks = (size + block_size - 1) / block_size;
    std::unique_ptr<Scalar[]> buf(new Scalar[blocks]);

    Scalar accum = 0;
    for (uint32_t i = 0, j = 0; i < size;) {
        accum += fmix32((Scalar) i);

        if (++i % block_size == 0) {
            buf[j++] = accum;
            accum = 0;
        }
    }

    if (size % block_size != 0)
        buf[blocks - 1] = accum;

    return Value::copy(buf.get(), blocks);
}

template <typename Value> inline Value block_sum_ref_const(uint32_t size, uint32_t block_size) {
    using Scalar = typename Value::Value;
    uint32_t blocks = (size + block_size - 1) / block_size;
    std::unique_ptr<Scalar[]> buf(new Scalar[blocks]);

    for (uint32_t i = 0; i < blocks; ++i)
        buf[i] = std::min(size - i*block_size, block_size);

    return Value::copy(buf.get(), blocks);
}

template <typename Value> inline Value block_prefix_sum_ref(uint32_t size, uint32_t block_size, bool exclusive, bool reverse) {
    using Scalar = typename Value::Value;
    std::unique_ptr<Scalar[]> buf(new Scalar[size]);

    Scalar accum = 0;
    for (uint32_t i = 0; i < size; ++i) {
        uint32_t j = reverse ? (size - 1 - i) : i;

        if (reverse) {
            if (j % block_size == block_size - 1)
                accum = 0;
        } else {
            if (i % block_size == 0)
                accum = 0;
        }

        Scalar prev = accum;
        accum += fmix32((Scalar) j);
        buf[j] = exclusive ? prev : accum;
    }

    return Value::copy(buf.get(), size);
}

// Try reductions over a range of arrays sizes
const uint32_t red_sizes[] {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 32, 60, 128, 250, 333,
    1024, 16384, 16388*10, 9973*17, 98973*17*3
};

TEST_BOTH_FLOAT_AGNOSTIC(01_all_any) {
    using Bool = Array<bool>;

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

TEST_BOTH_FLOAT_AGNOSTIC(02_block_reduce_u32_const) {
    for (uint32_t size : red_sizes) {
        for (uint32_t block_size : red_sizes) {
            if (block_size > size)
                continue;
            UInt32 v0 = full<UInt32>(1, size),
                   v1 = block_sum(v0, block_size);
            UInt32 vr = block_sum_ref_const<UInt32>(size, block_size);
            // printf("vr: %s\n", vr.str());
            // printf("v0: %s\n", v0.str());
            // printf("v1: %s\n", v1.str());
            jit_assert(v1 == vr);
        }
    }
}

TEST_BOTH_FLOAT_AGNOSTIC(02_block_reduce_u32) {
    for (uint32_t size : red_sizes) {
        for (uint32_t block_size : red_sizes) {
            if (block_size > size)
                continue;
            UInt32 v0 = fmix32(arange<UInt32>(size));
            UInt32 v1 = block_sum(v0, block_size);
            UInt32 vr = block_sum_ref<UInt32>(size, block_size);
            jit_assert(v1 == vr);
        }
    }
}

TEST_BOTH_FLOAT_AGNOSTIC(03_block_reduce_u64) {
    for (uint32_t size : red_sizes) {
        for (uint32_t block_size : red_sizes) {
            if (block_size > size)
                continue;

            using UInt64 = Array<uint64_t>;
            UInt64 v0 = fmix32(arange<UInt64>(size));
            UInt64 v1 = block_sum(v0, block_size);
            UInt64 vr = block_sum_ref<UInt64>(size, block_size);
            jit_assert(v1 == vr);
        }
    }
}

TEST_BOTH_FLOAT_AGNOSTIC(04_block_prefix_reduce_u32_inc_fwd) {
    for (uint32_t size : red_sizes) {
        for (uint32_t block_size : red_sizes) {
            if (block_size > size)
                continue;

            UInt32 v0 = fmix32(arange<UInt32>(size));
            UInt32 v1 = block_prefix_sum(v0, block_size, false, false);
            UInt32 vr = block_prefix_sum_ref<UInt32>(size, block_size, false, false);
            jit_assert(v1 == vr);
        }
    }
}

TEST_BOTH_FLOAT_AGNOSTIC(05_block_prefix_reduce_u32_exc_fwd) {
    for (uint32_t size : red_sizes) {
        for (uint32_t block_size : red_sizes) {
            if (block_size > size)
                continue;

            UInt32 v0 = fmix32(arange<UInt32>(size));
            UInt32 v1 = block_prefix_sum(v0, block_size, true, false);
            UInt32 vr = block_prefix_sum_ref<UInt32>(size, block_size, true, false);
            jit_assert(v1 == vr);
        }
    }
}

TEST_BOTH_FLOAT_AGNOSTIC(06_block_prefix_reduce_u32_inc_bwd) {
    for (uint32_t size : red_sizes) {
        for (uint32_t block_size : red_sizes) {
            if (block_size > size)
                continue;

            UInt32 v0 = fmix32(arange<UInt32>(size));
            UInt32 v1 = block_prefix_sum(v0, block_size, false, true);
            UInt32 vr = block_prefix_sum_ref<UInt32>(size, block_size, false, true);
            jit_assert(v1 == vr);
        }
    }
}

TEST_BOTH_FLOAT_AGNOSTIC(07_block_prefix_reduce_u32_exc_bwd) {
    for (uint32_t size : red_sizes) {
        for (uint32_t block_size : red_sizes) {
            if (block_size > size)
                continue;

            UInt32 v0 = fmix32(arange<UInt32>(size));
            UInt32 v1 = block_prefix_sum(v0, block_size, true, true);
            UInt32 vr = block_prefix_sum_ref<UInt32>(size, block_size, true, true);
            jit_assert(v1 == vr);
        }
    }
}

TEST_BOTH_FLOAT_AGNOSTIC(08_block_prefix_reduce_u64_inc_fwd) {
    for (uint32_t size : red_sizes) {
        for (uint32_t block_size : red_sizes) {
            if (block_size > size)
                continue;

            using UInt64 = Array<uint64_t>;
            UInt64 v0 = fmix32(arange<UInt64>(size));
            UInt64 v1 = block_prefix_sum(v0, block_size, false, false);
            UInt64 vr = block_prefix_sum_ref<UInt64>(size, block_size, false, false);
            jit_assert(v1 == vr);
        }
    }
}

TEST_BOTH_FLOAT_AGNOSTIC(09_block_prefix_reduce_u64_exc_fwd) {
    for (uint32_t size : red_sizes) {
        for (uint32_t block_size : red_sizes) {
            if (block_size > size)
                continue;
            using UInt64 = Array<uint64_t>;

            UInt64 v0 = fmix32(arange<UInt64>(size));
            UInt64 v1 = block_prefix_sum(v0, block_size, true, false);
            UInt64 vr = block_prefix_sum_ref<UInt64>(size, block_size, true, false);
            jit_assert(v1 == vr);
        }
    }
}

TEST_BOTH_FLOAT_AGNOSTIC(10_block_prefix_reduce_u64_inc_bwd) {
    for (uint32_t size : red_sizes) {
        for (uint32_t block_size : red_sizes) {
            if (block_size > size)
                continue;

            using UInt64 = Array<uint64_t>;
            UInt64 v0 = fmix32(arange<UInt64>(size));
            UInt64 v1 = block_prefix_sum(v0, block_size, false, true);
            UInt64 vr = block_prefix_sum_ref<UInt64>(size, block_size, false, true);
            jit_assert(v1 == vr);
        }
    }
}

TEST_BOTH_FLOAT_AGNOSTIC(11_block_prefix_reduce_u64_exc_bwd) {
    for (uint32_t size : red_sizes) {
        for (uint32_t block_size : red_sizes) {
            if (block_size > size)
                continue;

            using UInt64 = Array<uint64_t>;
            UInt64 v0 = fmix32(arange<UInt64>(size));
            UInt64 v1 = block_prefix_sum(v0, block_size, true, true);
            UInt64 vr = block_prefix_sum_ref<UInt64>(size, block_size, true, true);
            jit_assert(v1 == vr);
        }
    }
}

TEST_BOTH_FLOAT_AGNOSTIC(12_compress) {
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

TEST_BOTH_FLOAT_AGNOSTIC(13_mkperm) {
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
TEST_BOTH(14_block_ops) {
    Float a(0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f);

    jit_log(Info, "block_sum:  %s\n", block_sum(a, 3).str());
}
#endif
