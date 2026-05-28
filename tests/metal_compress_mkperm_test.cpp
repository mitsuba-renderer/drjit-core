// metal_compress_mkperm_test.cpp — Tests for compress and mkperm on Metal.

#include <drjit-core/array.h>
#include <cstdio>
#include <cstring>
#include <algorithm>

using namespace drjit;

int main() {
    jit_set_log_level_stderr(LogLevel::Info);
    jit_init(1u << (uint32_t) JitBackend::Metal);

    if (!jit_has_backend(JitBackend::Metal)) {
        fprintf(stderr, "Metal backend not available\n");
        return 1;
    }

    int failures = 0;

    // --- compress test: mask = [1,0,1,0,1,0,1,0] → [0,2,4,6] ---
    {
        uint8_t mask[] = { 1, 0, 1, 0, 1, 0, 1, 0 };
        uint32_t n = 8;

        void *mask_d = jit_malloc(JitBackend::Metal, n);
        jit_memcpy_async(JitBackend::Metal, mask_d, mask, n);

        void *out_d = jit_malloc(JitBackend::Metal, n * sizeof(uint32_t));
        uint32_t count = jit_compress(JitBackend::Metal,
                                       (const uint8_t *) mask_d, n,
                                       (uint32_t *) out_d);

        uint32_t result[8] = {};
        jit_memcpy(JitBackend::Metal, result, out_d, count * sizeof(uint32_t));

        printf("compress test: count=%u, indices=[", count);
        for (uint32_t i = 0; i < count; i++)
            printf("%u%s", result[i], i + 1 < count ? ", " : "");
        printf("]\n");
        fflush(stdout);

        if (count != 4 || result[0] != 0 || result[1] != 2 ||
            result[2] != 4 || result[3] != 6) {
            printf("  FAIL\n");
            failures++;
        }

        jit_free(mask_d);
        jit_free(out_d);
    }

    // --- mkperm test: values=[2,0,1,0,2,1] → groups {0:[1,3], 1:[2,5], 2:[0,4]} ---
    {
        uint32_t vals[] = { 2, 0, 1, 0, 2, 1 };
        uint32_t n = 6, bucket_count = 3;

        void *vals_d = jit_malloc(JitBackend::Metal, n * sizeof(uint32_t));
        jit_memcpy_async(JitBackend::Metal, vals_d, vals, n * sizeof(uint32_t));

        void *perm_d = jit_malloc(JitBackend::Metal, n * sizeof(uint32_t));

        // offsets array: 4 * bucket_count + 1 entries
        uint32_t offsets_size = (4 * bucket_count + 1) * sizeof(uint32_t);
        void *offsets_d = jit_malloc(JitBackend::Metal, offsets_size, /*shared=*/1);
        memset(offsets_d, 0, offsets_size);

        uint32_t unique = jit_block_mkperm(JitBackend::Metal,
                                      (const uint32_t *) vals_d, n, n,
                                      bucket_count,
                                      (uint32_t *) perm_d,
                                      (uint32_t *) offsets_d);

        uint32_t perm[6] = {};
        jit_memcpy(JitBackend::Metal, perm, perm_d, n * sizeof(uint32_t));

        printf("mkperm test: unique_buckets=%u, perm=[", unique);
        for (uint32_t i = 0; i < n; i++)
            printf("%u%s", perm[i], i + 1 < n ? ", " : "");
        printf("]\n");
        fflush(stdout);

        // Verify that the permutation groups values correctly:
        // values[perm[0]] and values[perm[1]] should be same bucket, etc.
        bool valid = true;

        // Check each bucket is contiguous in perm
        // Bucket 0 has 2 entries, bucket 1 has 2, bucket 2 has 2
        uint32_t expected_counts[3] = { 2, 2, 2 };
        uint32_t pos = 0;
        for (uint32_t b = 0; b < bucket_count; b++) {
            for (uint32_t i = 0; i < expected_counts[b]; i++) {
                if (pos >= n || vals[perm[pos]] != b) {
                    valid = false;
                    printf("  FAIL at pos=%u: perm[%u]=%u, vals[perm]=%u, expected bucket=%u\n",
                           pos, pos, perm[pos], pos < n ? vals[perm[pos]] : 999, b);
                }
                pos++;
            }
        }

        if (!valid || unique != 3) {
            failures++;
            printf("  FAIL\n");
        }

        jit_free(vals_d);
        jit_free(perm_d);
        jit_free(offsets_d);
    }

    printf("\n%d test(s) failed.\n", failures);
    fflush(stdout);

    jit_shutdown(0);
    return failures;
}
