#include "test.h"

using Float = CUDAArray<float>;

TEST_CUDA(01_creation_destruction) {
    Float value(1234);
    (void) value;
}

TEST_CUDA(02_eval_scalar) {
    Float value(1234);
    jitc_log(LogLevel::Debug, "value=%s", value.str());
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

