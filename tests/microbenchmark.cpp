// Dr.Jit data structure microbenchmark
//
// This program performs lots of pointless symbolic arithmetic (stepping through
// the Collatz sequence). However, in doing so, it exercises various internal
// data structures (locks, variable table, hash table for local value numbering)
// and can therefore be used to identify performance bottlenecks.

#include <drjit-core/array.h>
#include <chrono>

namespace dr = drjit;

using UInt32 = dr::LLVMArray<uint32_t>;

UInt32 collatz(UInt32 x) {
    return select(eq(x & UInt32(1), 0), x / 2, x*3 + 1);
}

int main(int, char **) {
    jit_init((int) JitBackend::LLVM);

    using namespace std::chrono;

    for (int k = 0; k < 3; ++k) {
        auto start = high_resolution_clock::now();

        for (uint32_t j = 0; j < 1024; ++j) {
            UInt32 i = dr::arange<UInt32>(1024);
            for (uint32_t l = 0; l < 4096; ++l)
                i = collatz(i);
        }

        auto end = high_resolution_clock::now();

        printf("Iteration %i took %.2f ms\n", k,
               (int) duration_cast<microseconds>(end - start).count() / 1000.0);
    }

    return 0;
}
