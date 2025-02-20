#include "drjit-core/array.h"
#include "drjit-core/jit.h"
#include "test.h"

/**
 * Basic addition test.
 * Supplying a different input should replay the operation, with this input.
 * In this case, the input at replay is incremented and should result in an
 * incremented output.
 */
TEST_BOTH(01_basic_replay) {

    auto func = [](UInt32 input) { return input + 1; };

    FrozenFunction frozen(Backend, func);

    for (uint32_t i = 0; i < 3; i++) {
        auto input = arange<UInt32>(10 + i);

        auto result = frozen(input);

        auto reference = func(input);

        jit_assert(all(eq(result, reference)));
    }
}

/**
 * This tests a single kernel with multiple unique inputs and outputs.
 */
TEST_BOTH(02_MIMO) {

    auto func = [](UInt32 x, UInt32 y) {
        return std::make_tuple(x + y, x * y);
    };

    FrozenFunction frozen(Backend, func);

    for (uint32_t i = 0; i < 3; i++) {
        auto x = arange<UInt32>(10 + i);
        auto y = arange<UInt32>(10 + i) + 1;

        auto result = frozen(x, y);

        auto reference = func(x, y);

        jit_assert(all(eq(std::get<0>(result), std::get<0>(reference))));
        jit_assert(all(eq(std::get<1>(result), std::get<1>(reference))));
    }
}

/**
 * This tests if the recording feature works, when supplying the same variable
 * twice in the input. In the final implementation this test-case should never
 * occur, as variables would be deduplicated in beforehand.
 */
TEST_BOTH(03_deduplicating_input) {

    auto func = [](UInt32 x, UInt32 y) { return std::tuple(x + y, x * y); };

    FrozenFunction frozen(Backend, func);

    for (uint32_t i = 0; i < 3; i++) {
        auto x = arange<UInt32>(10 + i);

        auto result = frozen(x, x);

        auto reference = func(x, x);

        jit_assert(all(eq(std::get<0>(result), std::get<0>(reference))));
        jit_assert(all(eq(std::get<1>(result), std::get<1>(reference))));
    }
}

/**
 * This tests, Whether it is possible to record multiple kernels in sequence.
 * The input of the second kernel relies on the execution of the first.
 * On LLVM, the correctness of barrier operations is therefore tested.
 */
TEST_BOTH(04_sequential_kernels) {

    auto func = [](UInt32 x) {
        auto y = x + 1;
        y.eval();
        return y + x;
    };

    FrozenFunction frozen(Backend, func);

    for (uint32_t i = 0; i < 3; i++) {
        auto x = arange<UInt32>(10 + i);

        auto result = frozen(x);

        auto reference = func(x);

        jit_assert(all(eq(result, reference)));
    }
}

/**
 * This tests, Whether it is possible to record multiple independent kernels in
 * the same recording.
 * The variables of the kernels are of different size, therefore two kernels are
 * generated. At replay these can be executed in parallel (LLVM) or sequence
 * (CUDA).
 */
TEST_BOTH(05_parallel_kernels) {

    auto func = [](UInt32 x, UInt32 y) { return std::tuple(x + 1, y + 1); };

    FrozenFunction frozen(Backend, func);

    for (uint32_t i = 0; i < 3; i++) {
        auto x = arange<UInt32>(10 + i);
        auto y = arange<UInt32>(11 + i);

        auto result = frozen(x, y);

        auto reference = func(x, y);

        jit_assert(all(eq(std::get<0>(result), std::get<0>(reference))));
        jit_assert(all(eq(std::get<1>(result), std::get<1>(reference))));
    }
}

/**
 * This tests the recording and replay of a horizontal reduction operation
 * (hsum).
 */
TEST_BOTH(06_reduce_hsum) {

    auto func = [](UInt32 x) { return hsum(x + 1); };

    FrozenFunction frozen(Backend, func);

    for (uint32_t i = 0; i < 3; i++) {
        auto x = arange<UInt32>(10 + i);

        auto result = frozen(x);

        auto reference = func(x);

        jit_assert(all(eq(result, reference)));
    }
}

/**
 * Tests recording of a prefix sum operation with different inputs at replay.
 */
TEST_BOTH(07_prefix_sum) {

    auto func = [](UInt32 x) { return block_prefix_sum(x, x.size()); };

    FrozenFunction frozen(Backend, func);

    for (uint32_t i = 0; i < 3; i++) {
        auto x = arange<UInt32>(10 + i);

        auto result = frozen(x);

        auto reference = func(x);

        jit_assert(all(eq(result, reference)));
    }
}

/**
 * Tests that it is possible to pass a single input to multiple outputs
 * including directly in a frozen function without any use after free
 * conditions.
 */
TEST_BOTH(08_input_passthrough) {

    auto func = [](UInt32 x) {
        auto y = x + 1;
        return std::tuple(y, x);
    };

    FrozenFunction frozen(Backend, func);

    for (uint32_t i = 0; i < 3; i++) {
        auto x = arange<UInt32>(10 + i);
        x.make_opaque();

        auto result = frozen(x);

        auto reference = func(x);

        jit_assert(all(eq(std::get<0>(result), std::get<0>(reference))));
        jit_assert(all(eq(std::get<1>(result), std::get<1>(reference))));
    }
}

/**
 * Tests if the dry run mode catches the case where LLVM kernels have to be
 * replayed due to size changes in a scatter reduce operation.
 */
TEST_LLVM(09_dry_run) {
    auto func = [](UInt32 target, UInt32 src) {
        scatter_reduce(ReduceOp::Add, target, src,
                       arange<UInt32>(src.size()) % 2);
        return target;
    };

    FrozenFunction frozen(Backend, func);

    for (uint32_t i = 0; i < 4; i++) {
        auto src = full<UInt32>(1, 10 + i);
        src.make_opaque();

        auto result = full<UInt32>(0, (i + 2));
        result.make_opaque();
        result = frozen(result, src);

        auto reference = full<UInt32>(0, (i + 2));
        reference.make_opaque();
        reference = frozen(reference, src);

        jit_assert(all(eq(result, reference)));
    }
}

/**
 * Tests that scattering to a variable does not modify variables depending on
 * the scatter target. This is ensured by the borrowing reference to the inputs
 * in the FrozenFunction, which causes \c scatter to add a \c memcpy_async in
 * the recording.
 */
TEST_LLVM(10_scatter) {
    auto func = [](UInt32 x) {
        scatter(x, UInt32(0), arange<UInt32>(x.size()));
        // We have to return the input, since we do not perform input
        // re-assignment in the \c FrozenFunction for the tests.
        return x;
    };

    FrozenFunction frozen(Backend, func);

    for (uint32_t i = 0; i < 4; i++) {
        auto x = arange<UInt32>(10 + i);

        auto y = x + 1;

        x = frozen(x);

        jit_assert(all(eq(x, full<UInt32>(0, 10 + i))));
        jit_assert(all(eq(y, arange<UInt32>(10 + i) + 1)));
    }
}

TEST_BOTH(11_symbolic_width) {
    auto func = [](UInt32 x) {
        auto y = block_prefix_sum(x, x.size());
        y = y / x.symbolic_width();
        return y;
    };

    FrozenFunction frozen(Backend, func);

    for (uint32_t i = 0; i < 4; i++) {
        auto x = arange<UInt32>(10 + i);

        auto res = frozen(x);
        auto ref = func(x);

        jit_assert(all(eq(res, ref)));
    }
}
