// metal_prefix_test.cpp — Test block_prefix_reduce on Metal.

#include <drjit-core/array.h>
#include <cstdio>
#include <cstring>

using namespace drjit;

int main() {
    jit_set_log_level_stderr(LogLevel::Info);
    jit_init(1u << (uint32_t) JitBackend::Metal);
    if (!jit_has_backend(JitBackend::Metal)) { return 1; }

    int failures = 0;

    // Test 1: inclusive prefix sum of [1,2,3,4] = [1,3,6,10]
    {
        float data[] = { 1, 2, 3, 4 };
        uint32_t n = 4;
        void *in_d = jit_malloc(AllocType::Device, n * sizeof(float));
        void *out_d = jit_malloc(AllocType::Device, n * sizeof(float));
        jit_memcpy_async(JitBackend::Metal, in_d, data, n * sizeof(float));
        jit_block_prefix_reduce(JitBackend::Metal, VarType::Float32,
                                ReduceOp::Add, n, n, 0, 0, in_d, out_d);
        jit_sync_thread();
        float result[4];
        jit_memcpy(JitBackend::Metal, result, out_d, n * sizeof(float));
        printf("Test 1 — inclusive prefix sum: [%.0f, %.0f, %.0f, %.0f]\n",
               result[0], result[1], result[2], result[3]);
        fflush(stdout);
        if (result[0] != 1 || result[1] != 3 || result[2] != 6 || result[3] != 10) {
            printf("  FAIL (expected [1, 3, 6, 10])\n"); failures++;
        }
        jit_free(in_d); jit_free(out_d);
    }

    // Test 2: exclusive prefix sum of [1,2,3,4] = [0,1,3,6]
    {
        float data[] = { 1, 2, 3, 4 };
        uint32_t n = 4;
        void *in_d = jit_malloc(AllocType::Device, n * sizeof(float));
        void *out_d = jit_malloc(AllocType::Device, n * sizeof(float));
        jit_memcpy_async(JitBackend::Metal, in_d, data, n * sizeof(float));
        jit_block_prefix_reduce(JitBackend::Metal, VarType::Float32,
                                ReduceOp::Add, n, n, 1, 0, in_d, out_d);
        jit_sync_thread();
        float result[4];
        jit_memcpy(JitBackend::Metal, result, out_d, n * sizeof(float));
        printf("Test 2 — exclusive prefix sum: [%.0f, %.0f, %.0f, %.0f]\n",
               result[0], result[1], result[2], result[3]);
        fflush(stdout);
        if (result[0] != 0 || result[1] != 1 || result[2] != 3 || result[3] != 6) {
            printf("  FAIL (expected [0, 1, 3, 6])\n"); failures++;
        }
        jit_free(in_d); jit_free(out_d);
    }

    // Test 3: UInt32 exclusive prefix sum of [3,1,4,1,5] = [0,3,4,8,9]
    {
        uint32_t data[] = { 3, 1, 4, 1, 5 };
        uint32_t n = 5;
        void *in_d = jit_malloc(AllocType::Device, n * sizeof(uint32_t));
        void *out_d = jit_malloc(AllocType::Device, n * sizeof(uint32_t));
        jit_memcpy_async(JitBackend::Metal, in_d, data, n * sizeof(uint32_t));
        jit_block_prefix_reduce(JitBackend::Metal, VarType::UInt32,
                                ReduceOp::Add, n, n, 1, 0, in_d, out_d);
        jit_sync_thread();
        uint32_t result[5];
        jit_memcpy(JitBackend::Metal, result, out_d, n * sizeof(uint32_t));
        printf("Test 3 — u32 exclusive prefix: [%u, %u, %u, %u, %u]\n",
               result[0], result[1], result[2], result[3], result[4]);
        fflush(stdout);
        if (result[0] != 0 || result[1] != 3 || result[2] != 4 ||
            result[3] != 8 || result[4] != 9) {
            printf("  FAIL (expected [0, 3, 4, 8, 9])\n"); failures++;
        }
        jit_free(in_d); jit_free(out_d);
    }

    // Test 4: Pairwise prefix sum (block_size=2)
    //   [1,2, 3,4, 5,6] → inclusive: [1,3, 3,7, 5,11]
    {
        float data[] = { 1, 2, 3, 4, 5, 6 };
        uint32_t n = 6;
        void *in_d = jit_malloc(AllocType::Device, n * sizeof(float));
        void *out_d = jit_malloc(AllocType::Device, n * sizeof(float));
        jit_memcpy_async(JitBackend::Metal, in_d, data, n * sizeof(float));
        jit_block_prefix_reduce(JitBackend::Metal, VarType::Float32,
                                ReduceOp::Add, n, 2, 0, 0, in_d, out_d);
        jit_sync_thread();
        float result[6];
        jit_memcpy(JitBackend::Metal, result, out_d, n * sizeof(float));
        printf("Test 4 — block_size=2 inclusive: [%.0f, %.0f, %.0f, %.0f, %.0f, %.0f]\n",
               result[0], result[1], result[2], result[3], result[4], result[5]);
        fflush(stdout);
        if (result[0] != 1 || result[1] != 3 || result[2] != 3 ||
            result[3] != 7 || result[4] != 5 || result[5] != 11) {
            printf("  FAIL (expected [1, 3, 3, 7, 5, 11])\n"); failures++;
        }
        jit_free(in_d); jit_free(out_d);
    }

    printf("\n%d test(s) failed.\n", failures);
    fflush(stdout);
    jit_shutdown(0);
    return failures;
}
