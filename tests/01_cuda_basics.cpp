#include "test.h"

using Float = CUDAArray<float>;

TEST_CUDA(01_creation_destruction) {
    Float value(1234);
    (void) value;
}

TEST_CUDA(02_fill_and_print) {
    jitc_log(Info, "  int8_t: %s", CUDAArray<  int8_t>::full(-111, 5).str());
    jitc_log(Info, " uint8_t: %s", CUDAArray< uint8_t>::full( 222, 5).str());
    jitc_log(Info, " int16_t: %s", CUDAArray< int16_t>::full(-1111, 5).str());
    jitc_log(Info, "uint16_t: %s", CUDAArray<uint16_t>::full( 2222, 5).str());
    jitc_log(Info, " int32_t: %s", CUDAArray< int32_t>::full(-1111111111, 5).str());
    jitc_log(Info, "uint32_t: %s", CUDAArray<uint32_t>::full( 2222222222, 5).str());
    jitc_log(Info, " int64_t: %s", CUDAArray< int64_t>::full(-1111111111111111111, 5).str());
    jitc_log(Info, "uint64_t: %s", CUDAArray<uint64_t>::full( 2222222222222222222, 5).str());
    jitc_log(Info, "   float: %s", CUDAArray<   float>::full(1.f/3.f, 5).str());
    jitc_log(Info, "  double: %s", CUDAArray<  double>::full(1.0/3.0, 5).str());
}

TEST_CUDA(03_eval_scalar) {
    Float value(1234);
    jitc_log(Info, "value=%s", value.str());
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


