# Dr.Jit data structure microbenchmark (Python port of microbenchmark.cpp)
#
# Mirrors tests/microbenchmark.cpp: pointless symbolic Collatz arithmetic that
# stresses the tracing hot path (locks, variable table, LVN hash table).
#
# By default it only *traces*. Pass '-e' to also evaluate each outer iteration
# (exercises scheduling, codegen, kernel hashing and cache-hit paths).

import sys
import time
import drjit as dr
from drjit.llvm import UInt32


def collatz(x):
    return dr.select((x & UInt32(1)) == 0, x // 2, x * 3 + 1)


def main():
    evaluate = "-e" in sys.argv[1:]

    dr.set_log_level(dr.LogLevel.Warn)
    dr.set_thread_count(0)

    # Match the C++ benchmark's sizes exactly.
    outer = 1024
    steps = 256 if evaluate else 4096

    print("Mode: %s, outer=%u, steps=%u"
          % ("trace+eval" if evaluate else "trace-only", outer, steps))

    for k in range(3):
        start = time.perf_counter()
        for j in range(outer):
            i = dr.arange(UInt32, 1024)
            for _ in range(steps):
                i = collatz(i)
            if evaluate:
                dr.eval(i)
        end = time.perf_counter()
        print("Iteration %i took %.2f ms" % (k, (end - start) * 1000.0))


if __name__ == "__main__":
    main()
