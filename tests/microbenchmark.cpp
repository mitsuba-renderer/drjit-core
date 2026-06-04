// Dr.Jit data structure microbenchmark
//
// This program performs lots of pointless symbolic arithmetic (stepping through
// the Collatz sequence). However, in doing so, it exercises various internal
// data structures (locks, variable table, hash table for local value numbering)
// and can therefore be used to identify performance bottlenecks.
//
// By default the program only *traces* the computation (building a large
// expression graph) without ever evaluating it. Pass the '-e' flag to also
// evaluate the result of each outer iteration. This additionally exercises the
// scheduling, code generation, kernel hashing, and kernel cache lookup paths
// (the first evaluation compiles a kernel, the rest are cache hits).

#include <drjit-core/array.h>
#include <chrono>
#include <cstdio>
#include <cstring>

namespace dr = drjit;

using UInt32 = dr::LLVMArray<uint32_t>;

UInt32 collatz(UInt32 x) {
    return select(eq(x & UInt32(1), 0), x / 2, x*3 + 1);
}

int main(int argc, char **argv) {
    bool evaluate = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-e") == 0)
            evaluate = true;
        else {
            fprintf(stderr, "Usage: %s [-e]\n", argv[0]);
            return 1;
        }
    }

    jit_set_log_level_stderr(LogLevel::Warn);
    jit_llvm_set_thread_count(0);
    jit_init(1u << (uint32_t) JitBackend::LLVM);

    using namespace std::chrono;

    // When evaluating, the per-iteration kernel is much heavier, so use a
    // smaller chain to keep runtimes comparable.
    uint32_t outer = evaluate ? 1024 : 1024;
    uint32_t steps = evaluate ? 256 : 4096;

    printf("Mode: %s, outer=%u, steps=%u\n",
           evaluate ? "trace+eval" : "trace-only", outer, steps);

    for (int k = 0; k < 3; ++k) {
        auto start = high_resolution_clock::now();

        for (uint32_t j = 0; j < outer; ++j) {
            UInt32 i = dr::arange<UInt32>(1024);
            for (uint32_t l = 0; l < steps; ++l)
                i = collatz(i);
            if (evaluate)
                i.eval();
        }

        auto end = high_resolution_clock::now();

        printf("Iteration %i took %.2f ms\n", k,
               (int) duration_cast<microseconds>(end - start).count() / 1000.0);
    }

    return 0;
}
