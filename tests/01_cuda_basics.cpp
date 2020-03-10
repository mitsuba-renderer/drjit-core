#include "test.h"

TEST_CUDA(01_creation_destruction) {
    CUDAArray x(1234);
}

TEST_CUDA(02_creation_destruction) {
    CUDAArray x(1234);
}

// void test_2() {
//     CUDAArray x(1234);
//     CUDAArray y(1234);
//     jitc_eval();
// }
//
// void test_3() {
//     CUDAArray x(1234);
//     jitc_eval();
//     CUDAArray y(1234);
// }

