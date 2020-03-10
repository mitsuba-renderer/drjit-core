#include <enoki/jitvar.h>
#include <iostream>

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
        jitc_set_log_level(4);
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
