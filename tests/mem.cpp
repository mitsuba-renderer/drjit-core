#include "test.h"
#include <cstring>
#include <algorithm>

TEST_BOTH(01_gather) {
    Int32 r = arange<Int32>(100) + 100;
    r.eval();
    UInt32 index = UInt32(34, 62, 75, 2);
    Int32 ref = Int32(134, 162, 175, 102);
    Int32 value = gather<Int32>(r, index);
    jit_assert(all(eq(ref, value)));
}

TEST_BOTH(02_gather_mask) {
    Mask r = eq(arange<Int32>(100) & Int32(1), 1);
    r.eval();
    UInt32 index = UInt32(33, 62, 75, 2);
    Mask ref = UInt32(1, 0, 1, 0);
    Mask value = gather<Mask>(r, index);
    jit_assert(all(eq(ref, value)));
}

TEST_BOTH(03_gather_masked) {
    Int32 r = arange<Int32>(100) + 100;
    r.eval();
    UInt32 index = UInt32(34, 62, 75, 2);
    Mask mask = index > 50;
    UInt32 ref = UInt32(0, 162, 175, 0);
    UInt32 value = gather<UInt32>(r, index, mask);
    jit_assert(all(eq(ref, value)));
}

TEST_BOTH(04_gather_mask_masked) {
    Mask r = eq(arange<Int32>(100) & Int32(1), 1);
    r.eval();
    UInt32 index = UInt32(33, 62, 75, 2);
    Mask ref = UInt32(0, 0, 1, 0);
    Mask mask = index > 50;
    Mask value = gather<Mask>(r, index, mask);
    jit_assert(all(eq(ref, value)));
}

TEST_BOTH(05_gather_scalar) {
    /* unmasked, doesn't launch any kernels */ {
        Int32 r = 124;
        Array<uint64_t> index = Array<uint64_t>(34, 62, 75, 2);
        UInt32 ref = 124;
        UInt32 value = gather<UInt32>(r, index);
        jit_assert(all(eq(ref, value)));
    }
    /* masked */ {
        Int32 r = 124;
        Array<uint64_t> index = Array<uint64_t>(34, 62, 75, 2);
        Mask mask = index > 50;
        UInt32 ref = UInt32(0, 124, 124, 0);
        UInt32 value = gather<UInt32>(r, index, mask);
        jit_assert(all(eq(ref, value)));
    }
}

TEST_BOTH(06_gather_scalar_mask) {
    /* unmasked, doesn't launch any kernels */ {
        Mask r = true;
        Array<uint64_t> index = Array<uint64_t>(34, 62, 75, 2);
        Mask ref = true;
        Mask value = gather<Mask>(r, index);
        jit_assert(all(eq(ref, value)));
    }
    /* masked */ {
        Mask r = true;
        Array<uint64_t> index = Array<uint64_t>(34, 62, 75, 2);
        Mask mask = index > 50;
        Mask ref = Mask(false, true, true, false);
        Mask value = gather<Mask>(r, index, mask);
        jit_assert(all(eq(ref, value)));
    }
}

TEST_BOTH(07_scatter) {
    UInt32 r = arange<UInt32>(10);
    UInt32 index = UInt32(1, 7, 5);
    UInt32 value = UInt32(8, 2, 3);
    UInt32 ref = UInt32(0, 8, 2, 3, 4, 3, 6, 2, 8, 9);
    scatter(r, value, index);
    jit_assert(all(eq(ref, r)));
}

TEST_BOTH(08_scatter_mask) {
    UInt32 r = arange<UInt32>(10);
    UInt32 index = UInt32(1, 7, 5);
    UInt32 value = UInt32(8, 2, 3);
    Mask mask = Mask(true, false, true);
    UInt32 ref = UInt32(0, 8, 2, 3, 4, 3, 6, 7, 8, 9);
    scatter(r, value, index, mask);
    jit_assert(all(eq(ref, r)));
}

TEST_BOTH(09_safety) {
    /* Collapse adjacent scatters */ {
        Float a = arange<Float>(5);
        a.eval();
        uint32_t index = a.index();
        scatter(a, Float(0), UInt32(0));
        jit_assert(index == a.index());
        scatter(a, Float(1), UInt32(0));
        jit_assert(index == a.index());
        scatter(a, Float(2), UInt32(0));
        jit_assert(index == a.index());
    }

    /* Make safety copies with multiple ext. refs */ {
        Float a = arange<Float>(5), b = a;
        jit_assert(a.index() == b.index());
        a.eval();
        uint32_t index = a.index();
        jit_assert(index == b.index());
        scatter(a, Float(0), UInt32(0));
        jit_assert(index != a.index() && a.index() != b.index());
        index = a.index();
        scatter(a, Float(1), UInt32(0));
        jit_assert(index == a.index() && a.index() != b.index());
    }

    /* Make safety copies in the presence of gathers */ {
        Float a = arange<Float>(5);
        a.eval();
        uint32_t index = a.index();

        Float b = gather(a, UInt32(0));
        jit_assert(index == a.index());

        scatter(a, Float(0), UInt32(0));
        jit_assert(index != a.index());
    }
}

TEST_BOTH(10_scatter_atomic_rmw) {
    /* scatter 16 values */ {
        Float target = zeros<Float>(16);
        UInt32 index(0, 1, 2, 0, 4, 5, 6, 7, 8, 9, 10, 2, 3, 0, 0);

        scatter_reduce(ReduceOp::Add, target, Float(1), index);

        jit_assert(
            strcmp(target.str(),
                   "[4, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]") == 0);
    }

    /* scatter 17 values, tests LLVM masking */ {
        Float target = zeros<Float>(16);
        UInt32 index(0, 1, 2, 0, 4, 5, 6, 7, 8, 9, 10, 10, 2, 3, 0, 0);

        scatter_reduce(ReduceOp::Add, target, Float(1), index);

        jit_assert(
            strcmp(target.str(),
                   "[4, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0]") == 0);
    }

    /* masked scatter */ {
        Float target = zeros<Float>(16);
        UInt32 index(0, 1, 2, 0, 4, 5, 6, 7, 8, 9, 10, 10, 2, 3, 0, 0);
        Mask mask = neq(index, 7);

        scatter_reduce(ReduceOp::Add, target, Float(1), index, mask);

        jit_assert(
            strcmp(target.str(),
                   "[4, 1, 2, 1, 1, 1, 1, 0, 1, 1, 2, 0, 0, 0, 0, 0]") == 0);
    }
}

TEST_BOTH(11_reindex) {
    // Test that a gather expression can rewrite the original expression
    UInt32 i1 = arange<UInt32>(100) + 5,
           i2 = arange<UInt32>(10) * 3,
           i3 = gather<UInt32>(i1, i2),
           i4 = arange<UInt32>(10) * 3 + 5;

    jit_assert(!jit_var_is_evaluated(i1.index()) &&
               !jit_var_is_evaluated(i2.index()) &&
               !jit_var_is_evaluated(i3.index()) &&
               !jit_var_is_evaluated(i4.index()));

    jit_assert(i3.index() == i4.index());
}

TEST_BOTH(12_scatter_reduce_kahan) {
    Float buf_1 = zeros<Float>(1),
          buf_2 = zeros<Float>(1);

    scatter_reduce_kahan(buf_1, buf_2, Float(1e7 + 1), UInt32(0));
    jit_assert(all(eq(buf_1 - Float(1e7 + 1), Float(0))));
    jit_assert(all(eq(buf_2, Float(0))));

    scatter_reduce_kahan(buf_1, buf_2, Float(1e7), UInt32(0));
    jit_assert(all(eq(buf_1 - Float(2e7), Float(0))));
    jit_assert(all(eq(buf_2, Float(1))));

    scatter_reduce_kahan(buf_1, buf_2, Float(1), UInt32(0));
    jit_assert(all(eq(buf_1 - Float(2e7), Float(0))));
    jit_assert(all(eq(buf_2, Float(2))));
}

TEST_BOTH(13_gather_scalar_opaque) {
    Float buf_1 = opaque<Float>(1),
          buf_2 = gather<Float>(buf_1, arange<UInt32>(10));
    jit_assert(strcmp(buf_2.str(), "[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]") == 0);
}

TEST_LLVM(14_gather_symbolic_llvm_mask) {
    Float buf_1 = Float(1, 2, 3, 4, 5, 6, 7, 8);
    Float buf_2 = gather<Float>(buf_1, arange<UInt32>(0, 8, 1));
    Float buf_3 = gather<Float>(buf_2, arange<UInt32>(4, 8, 1));
    jit_assert(strcmp(buf_3.str(), "[5, 6, 7, 8]") == 0);
}

TEST_BOTH(15_gather_symbolic_multiple_mask) {
    /* A gather expression that is reindexed/rewritten should properly apply its
     * mask to any previous gather operations it depends on */
    Float buf_0 = Float(1, 2, 3, 4, 5, 6, 7, 8);

    // true, true, true, false, true
    Mask mask_1 = (arange<UInt32>(0, 5, 1) % 4) != 0;
    UInt32 index_1 = arange<UInt32>(0, 5, 1);
    Float buf_1 = gather<Float>(buf_0, index_1, mask_1);

    Mask mask_2 = Mask(true, true, false, false);
    UInt32 index_2 = UInt32(0, 1, -1, -1);

    // This gather will reindex, and should apply `mask_2` to the previous
    // gather, or else it will lookup invalid memory
    Float buf_2 = gather<Float>(buf_1, index_2, mask_2);
    jit_assert(strcmp(buf_2.str(), "[1, 2, 0, 0]") == 0);
}

TEST_BOTH(16_scatter_inc) {
    constexpr size_t n = 10000;
    uint32_t out_cpu[n];
    UInt32 counter(0);
    UInt32 index = arange<UInt32>(n);
    UInt32 out = zeros<UInt32>(n);
    UInt32 offset = scatter_inc(counter, UInt32(0), Mask(true));
    scatter(out, index, offset);
    jit_assert(all(eq(counter, n)));
    jit_memcpy(Backend, out_cpu, out.data(), n * sizeof(uint32_t));

    try {
        scatter(out, index, offset);
        jit_fail("16_scatter_inc(): Exception not raised!");
    } catch (...) { }

    try {
        offset.eval();
        jit_fail("16_scatter_inc(): Exception not raised!");
    } catch (...) { }

    std::sort(out_cpu, out_cpu + n);
    for (size_t i = 0; i < n; ++i) {
        if (i != out_cpu[i]) {
            printf("%zu - %u\n", i, out_cpu[i]);
            abort();
        }
    }
}

TEST_BOTH(17_scatter_inc_2) {
    constexpr size_t n = 10000;
    uint32_t out_cpu[n];
    UInt32 counter(0);
    UInt32 index = arange<UInt32>(n);
    UInt32 out = zeros<UInt32>(n);
    UInt32 offset = scatter_inc(counter, UInt32(0), full<Mask>(true, n));
    scatter(out, index, offset);
    jit_var_schedule(out.index());
    jit_var_schedule(offset.index());
    jit_eval();

    UInt32 out2 = zeros<UInt32>(n);
    scatter(out2, index, offset);
    out2.eval();
    jit_memcpy(Backend, out_cpu, out2.data(), n * sizeof(uint32_t));
    std::sort(out_cpu, out_cpu + n);
    for (size_t i = 0; i < n; ++i) {
        if (i != out_cpu[i]) {
            printf("%zu - %u\n", i, out_cpu[i]);
            abort();
        }
    }
}

TEST_BOTH(18_scatter_inc_mask) {
    constexpr size_t n = 100000;
    uint32_t out_cpu[n];
    UInt32 counter(0);
    UInt32 index = arange<UInt32>(n);
    UInt32 out = zeros<UInt32>(n);
    Mask active = eq(index & UInt32(1), 0);
    UInt32 offset = scatter_inc(counter, UInt32(0), active);
    scatter(out, index, offset, active);
    out.eval();
    jit_assert(all(eq(counter, n/2)));
    jit_memcpy(Backend, out_cpu, out.data(), n * sizeof(uint32_t));
    std::sort(out_cpu, out_cpu + n/2);
    for (size_t i = 0; i < n/2; ++i) {
        if (out_cpu[i] != i*2) {
            printf("%zu - %u\n", i, out_cpu[i]);
            abort();
        }
    }
    for (size_t i = n/2; i < n; ++i) {
        if (out_cpu[i] != 0) {
            printf("%zu - %u\n", i, out_cpu[i]);
            abort();
        }
    }
}
