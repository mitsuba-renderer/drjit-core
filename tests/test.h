#pragma once

#include <enoki/cuda.h>
#include <enoki/llvm.h>
#include <stdexcept>
#include <algorithm>

static constexpr LogLevel Error = LogLevel::Error;
static constexpr LogLevel Warn  = LogLevel::Warn;
static constexpr LogLevel Info  = LogLevel::Info;
static constexpr LogLevel Debug = LogLevel::Debug;
static constexpr LogLevel Trace = LogLevel::Trace;

extern int test_register(const char *name, void (*func)(), bool cuda);
extern "C" void log_callback(LogLevel cb, const char *msg);

using FloatC = CUDAArray<float>;
using Int32C = CUDAArray<int32_t>;
using FloatL = LLVMArray<float>;
using Int32L = LLVMArray<int32_t>;

#define TEST_CUDA(name)                                                        \
    template <typename Float, typename Int32> void test##name();               \
    int test##name##_c =                                                       \
        test_register("test" #name, test##name<FloatC, Int32C>, true);         \
    template <typename Float, typename Int32>                                  \
    void test##name()

#define TEST_LLVM(name)                                                        \
    template <typename Float, typename Int32> void test##name();               \
    int test##name##_l =                                                       \
        test_register("test" #name, test##name<FloatL, Int32L>, false);        \
    template <typename Float, typename Int32>                                  \
    void test##name()

#define TEST_BOTH(name)                                                        \
    template <typename Float, typename Int32> void test##name();               \
    int test##name##_l = test_register("test" #name "_cuda",                   \
                                       test##name<FloatL, Int32L>, false);     \
    int test##name##_c = test_register("test" #name "_llvm",                   \
                                       test##name<FloatC, Int32C>, false);     \
    template <typename Float, typename Int32> void test##name()

/// RAII helper for temporarily decreasing the log level
struct scoped_set_log_level {
public:
    scoped_set_log_level(LogLevel level) {
        m_cb_level = jitc_log_callback();
        m_stderr_level = jitc_log_stderr();
        jitc_log_callback_set(std::min(level, m_cb_level), log_callback);
        jitc_log_stderr_set(std::min(level, m_stderr_level));
    }

    ~scoped_set_log_level() {
        jitc_log_callback_set(m_cb_level, log_callback);
        jitc_log_stderr_set(m_stderr_level);
    }

private:
    LogLevel m_cb_level;
    LogLevel m_stderr_level;
};
