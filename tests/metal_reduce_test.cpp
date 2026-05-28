// metal_reduce_test.cpp -- Test block_reduce on the Metal backend.
//
// Tests sum and min reductions with various block sizes.

#include <drjit-core/array.h>
#include <cstdio>
#include <cstring>
#include <cmath>

using namespace drjit;

int main() {
    jit_set_log_level_stderr(LogLevel::Info);
    jit_init(1u << (uint32_t) JitBackend::Metal);

    if (!jit_has_backend(JitBackend::Metal)) {
        fprintf(stderr, "Metal backend not available\n");
        return 1;
    }

    using FloatM = MetalArray<float>;
    using UInt32M = MetalArray<uint32_t>;

    int failures = 0;

    // --- Test 1: sum of [1..8] = 36 ---
    {
        float data[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
        uint32_t src = jit_var_mem_copy(JitBackend::Metal, VarType::Float32,
                                        data, 8, /*from_host=*/1);
        uint32_t dst = jit_var_block_reduce(ReduceOp::Add, src, 8, 0);
        jit_var_eval(dst);
        const char *s = jit_var_str(dst);
        printf("Test 1 — sum([1..8]):  %s\n", s);
        fflush(stdout);
        if (strcmp(s, "[36]") != 0) { failures++; printf("  FAIL\n"); }
        jit_var_dec_ref(src);
        jit_var_dec_ref(dst);
    }

    // --- Test 2: pairwise sum of [1..8] with block_size=2 ---
    {
        float data[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
        uint32_t src = jit_var_mem_copy(JitBackend::Metal, VarType::Float32,
                                        data, 8, /*from_host=*/1);
        uint32_t dst = jit_var_block_reduce(ReduceOp::Add, src, 2, 0);
        jit_var_eval(dst);
        const char *s = jit_var_str(dst);
        printf("Test 2 — pairwise sum: %s\n", s);
        fflush(stdout);
        if (strcmp(s, "[3, 7, 11, 15]") != 0) { failures++; printf("  FAIL\n"); }
        jit_var_dec_ref(src);
        jit_var_dec_ref(dst);
    }

    // --- Test 3: min of [4, 2, 7, 1, 8, 3, 6, 5] = 1 ---
    {
        float data[] = { 4, 2, 7, 1, 8, 3, 6, 5 };
        uint32_t src = jit_var_mem_copy(JitBackend::Metal, VarType::Float32,
                                        data, 8, /*from_host=*/1);
        uint32_t dst = jit_var_block_reduce(ReduceOp::Min, src, 8, 0);
        jit_var_eval(dst);
        const char *s = jit_var_str(dst);
        printf("Test 3 — min:          %s\n", s);
        fflush(stdout);
        if (strcmp(s, "[1]") != 0) { failures++; printf("  FAIL\n"); }
        jit_var_dec_ref(src);
        jit_var_dec_ref(dst);
    }

    // --- Test 4: uint32 sum ---
    {
        uint32_t data[] = { 10, 20, 30, 40 };
        uint32_t src = jit_var_mem_copy(JitBackend::Metal, VarType::UInt32,
                                        data, 4, /*from_host=*/1);
        uint32_t dst = jit_var_block_reduce(ReduceOp::Add, src, 4, 0);
        jit_var_eval(dst);
        const char *s = jit_var_str(dst);
        printf("Test 4 — u32 sum:      %s\n", s);
        fflush(stdout);
        if (strcmp(s, "[100]") != 0) { failures++; printf("  FAIL\n"); }
        jit_var_dec_ref(src);
        jit_var_dec_ref(dst);
    }

    printf("\n%d test(s) failed.\n", failures);
    fflush(stdout);

    jit_shutdown(0);
    return failures;
}
