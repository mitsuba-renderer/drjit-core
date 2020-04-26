#pragma once

#include <enoki-jit/cuda.h>
#include <enoki-jit/llvm.h>
#include <stdexcept>
#include <algorithm>

using namespace enoki;

static constexpr LogLevel Error = LogLevel::Error;
static constexpr LogLevel Warn  = LogLevel::Warn;
static constexpr LogLevel Info  = LogLevel::Info;
static constexpr LogLevel Debug = LogLevel::Debug;
static constexpr LogLevel Trace = LogLevel::Trace;

extern int test_register(const char *name, void (*func)(), bool cuda, const char *flags = nullptr);
extern "C" void log_level_callback(LogLevel cb, const char *msg);

using FloatC  = CUDAArray<float>;
using Int32C  = CUDAArray<int32_t>;
using UInt32C = CUDAArray<uint32_t>;
using FloatL  = LLVMArray<float>;
using Int32L  = LLVMArray<int32_t>;
using UInt32L = LLVMArray<uint32_t>;

#define TEST_CUDA(name, ...)                                                   \
    template <typename Float, typename Int32, typename UInt32,                 \
              template <class> class Array>                                    \
    void test##name();                                                         \
    int test##name##_c = test_register(                                        \
        "test" #name "_cuda", test##name<FloatC, Int32C, UInt32C, CUDAArray>,  \
        true, ##__VA_ARGS__);                                                  \
    template <typename Float, typename Int32, typename UInt32,                 \
              template <class> class Array>                                    \
    void test##name()

#define TEST_LLVM(name, ...)                                                   \
    template <typename Float, typename Int32, typename UInt32,                 \
              template <class> class Array>                                    \
    void test##name();                                                         \
    int test##name##_l = test_register(                                        \
        "test" #name "_llvm", test##name<FloatL, Int32L, UInt32L, LLVMArray>,  \
        false, ##__VA_ARGS__);                                                 \
    template <typename Float, typename Int32, typename UInt32,                 \
              template <class> class Array>                                    \
    void test##name()

#define TEST_BOTH(name, ...)                                                   \
    template <typename Float, typename Int32, typename UInt32,                 \
              template <class> class Array>                                    \
    void test##name();                                                         \
    int test##name##_c = test_register(                                        \
        "test" #name "_cuda", test##name<FloatC, Int32C, UInt32C, CUDAArray>,  \
        true, ##__VA_ARGS__);                                                  \
    int test##name##_l = test_register(                                        \
        "test" #name "_llvm", test##name<FloatL, Int32L, UInt32L, LLVMArray>,  \
        false, ##__VA_ARGS__);                                                 \
    template <typename Float, typename Int32, typename UInt32,                 \
              template <class> class Array>                                    \
    void test##name()

#define jitc_assert(cond)                                                      \
    do {                                                                       \
        if (!(cond))                                                           \
            jitc_fail("Assertion failure: %s in line %u.", #cond, __LINE__);   \
    } while (0)

/// RAII helper for temporarily decreasing the log level
struct scoped_set_log_level {
public:
    scoped_set_log_level(LogLevel level) {
        m_cb_level = jitc_log_level_callback();
        m_stderr_level = jitc_log_level_stderr();
        jitc_set_log_level_callback(std::min(level, m_cb_level), log_level_callback);
        jitc_set_log_level_stderr(std::min(level, m_stderr_level));
    }

    ~scoped_set_log_level() {
        jitc_set_log_level_callback(m_cb_level, log_level_callback);
        jitc_set_log_level_stderr(m_stderr_level);
    }

private:
    LogLevel m_cb_level;
    LogLevel m_stderr_level;
};
