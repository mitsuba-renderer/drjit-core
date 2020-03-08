#include "api.h"
#include <stdio.h>
#include <iostream>

#include "hash.h"
#include <string.h>

enum EnokiType { Invalid = 0, Int8, UInt8, Int16, UInt16,
                 Int32, UInt32, Int64, UInt64, Float16,
                 Float32, Float64, Bool, Pointer };

void test_1() {
    // Scalar initialization
    uint32_t idx = jitc_trace_append(EnokiType::UInt32, "mov.%t0 %r0, 1234");
    // printf("%s\n", jitc_whos());
    jitc_eval();
    // printf("%s\n", jitc_whos());
    jitc_dec_ref_ext(idx);
}

void test_2() {
    // Caching
    uint32_t idx1 = jitc_trace_append(EnokiType::UInt32, "mov.%t0 %r0, 1234"),
             idx2 = jitc_trace_append(EnokiType::UInt32, "mov.%t0 %r0, 1234");
    // jit_eval();
    jitc_dec_ref_ext(idx1);
    jitc_dec_ref_ext(idx2);
}

int main(int argc, char **argv) {
    (void) argc;
    (void) argv;

    try {
        jitc_set_log_level(3);
        jitc_init_async();
        jitc_device_set(0, 0);

        test_1();
        test_1();

        jitc_shutdown();
    } catch (const std::exception &e) {
        std::cout << "Exception: "<< e.what() << std::endl;
    } catch (...) {
        std::cout << "Uncaught exception!" << std::endl;
    }
}
