#include "test.h"

#if 0

TEST_CUDA(01_all_any) {
    using Bool = Array<bool>;

    for (uint32_t i = 0; i < 100; ++i) {
        uint32_t size = 23*i*i*i + 1;

        Bool f = full<Bool>(false, size),
             t = full<Bool>(true, size);

        jitc_assert(all(t) && any(t) && !all(f) && !any(f));

        for (int j = 0; j < 100; ++j) {
            uint32_t index = (uint32_t) rand() % size;
            f.write(index, true);
            t.write(index, false);
            if (size == 1)
                jitc_assert(!all(t) && !any(t) && all(f) && any(f));
            else
                jitc_assert(!all(t) && any(t) && !all(f) && any(f));
            f.write(index, false);
            t.write(index, true);
            jitc_assert(all(t) && any(t) && !all(f) && !any(f));
        }
    }
}

TEST_CUDA(02_scan) {
    for (uint32_t i = 0; i < 100; ++i) {
        uint32_t size = 23*i*i*i + 1;

        UInt32 result, ref;

        if (i < 15) {
            result = arange<UInt32>(size);
            ref    = result * (result - 1) / 2;
        } else {
            result = full<UInt32>(1, size);
            ref    = arange<UInt32>(size);
        }
        jitc_eval();
        jitc_scan(result.data(), result.data(), size);
        jitc_assert(result == ref);
    }
}

#endif

TEST_CUDA(03_mkperm) {
    srand(0);
    for (uint32_t i = 0; i < 100; ++i) {
        uint32_t size = 23*i*i*i + 1;
        fprintf(stderr, "Iteration %i\n", i);
        for (uint32_t j = 0; j < 100; ++j) {
            uint32_t *buf = (uint32_t *) jitc_malloc(AllocType::HostPinned,
                                                     size * sizeof(uint32_t));
            uint32_t n_buckets = 23*i*i*i + 1;
            for (size_t i = 0; i < size; ++i)
                buf[i] = i % n_buckets;
            buf = (uint32_t *) jitc_malloc_migrate(buf, AllocType::Device);
            UInt32 buf_c = UInt32::map(buf, size);
            jitc_free(buf);
        }
    }
}
