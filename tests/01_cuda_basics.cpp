#include <enoki/jitvar.h>
#include <stdexcept>

void test_1() {
    CUDAArray x(1234);
    jitc_eval();
}

void test_2() {
    CUDAArray x(1234);
    CUDAArray y(1234);
    jitc_eval();
    // Caching
}

void test_3() {
    CUDAArray x(1234);
    jitc_eval();
    CUDAArray y(1234);
    // Caching
}

int main(int argc, char **argv) {
    (void) argc;
    (void) argv;

    try {
        jitc_log_level_set(4);
        jitc_log_buffer_enable(1);
        jitc_init_async();
        jitc_device_set(0, 0);

        test_1();
        test_1();

        jitc_shutdown();
    } catch (const std::exception &e) {
        fprintf(stderr, "Exception: %s!\n", e.what());
    }
}
