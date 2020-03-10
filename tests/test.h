#pragma once

#include <enoki/jitvar.h>
#include <stdexcept>

extern int test_register(const char *name, void (*func)(), bool cuda);

#define TEST_CUDA(name)                                                        \
    void test##name();                                                         \
    int test##name##_s = test_register("test"#name, test##name, true);         \
    void test##name()                                                          \
