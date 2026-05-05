// metal_gemm_test.cpp — Test batched_gemm via MPS on Metal.

#include <drjit-core/jit.h>
#include <cstdio>
#include <cstring>
#include <cmath>

int main() {
    jit_set_log_level_stderr(LogLevel::Info);
    jit_init((uint32_t) JitBackend::Metal);
    if (!jit_has_backend(JitBackend::Metal)) { return 1; }

    int failures = 0;

    // Test: 2x2 @ 2x2 matrix multiply
    // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    // C = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
    {
        float A[] = { 1, 2, 3, 4 };
        float B[] = { 5, 6, 7, 8 };
        float C[4] = {};

        void *A_d = jit_malloc(AllocType::Device, sizeof(A));
        void *B_d = jit_malloc(AllocType::Device, sizeof(B));
        void *C_d = jit_malloc(AllocType::Device, sizeof(C));

        jit_memcpy_async(JitBackend::Metal, A_d, A, sizeof(A));
        jit_memcpy_async(JitBackend::Metal, B_d, B, sizeof(B));
        jit_sync_thread();

        jit_batched_gemm(JitBackend::Metal, VarType::Float32,
                         0, 0, 2, 2, 2, nullptr, A_d, B_d, C_d);
        jit_sync_thread();

        jit_memcpy(JitBackend::Metal, C, C_d, sizeof(C));

        printf("GEMM test: C = [[%.0f, %.0f], [%.0f, %.0f]]\n",
               C[0], C[1], C[2], C[3]);
        fflush(stdout);

        if (C[0] != 19 || C[1] != 22 || C[2] != 43 || C[3] != 50) {
            printf("  FAIL (expected [[19,22],[43,50]])\n");
            failures++;
        }

        jit_free(A_d);
        jit_free(B_d);
        jit_free(C_d);
    }

    printf("\n%d test(s) failed.\n", failures);
    fflush(stdout);
    jit_shutdown(0);
    return failures;
}
