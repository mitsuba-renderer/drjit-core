#pragma once

#include <enoki/jitvar.h>
#include <stdexcept>

static constexpr LogLevel Error = LogLevel::Error;
static constexpr LogLevel Warn  = LogLevel::Warn;
static constexpr LogLevel Info  = LogLevel::Info;
static constexpr LogLevel Debug = LogLevel::Debug;
static constexpr LogLevel Trace = LogLevel::Trace;

extern int test_register(const char *name, void (*func)(), bool cuda);

#define TEST_CUDA(name)                                                        \
    void test##name();                                                         \
    int test##name##_s = test_register("test"#name, test##name, true);         \
    void test##name()                                                          \
