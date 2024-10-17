#pragma once

#include <drjit-core/array.h>
#include <drjit-core/half.h>
#include <cstdio>
#include <stdexcept>
#include <cstring>

using namespace drjit;

static constexpr LogLevel Error = LogLevel::Error;
static constexpr LogLevel Warn  = LogLevel::Warn;
static constexpr LogLevel Info  = LogLevel::Info;
static constexpr LogLevel Debug = LogLevel::Debug;
static constexpr LogLevel Trace = LogLevel::Trace;

extern int test_register(const char *name, void (*func)(), const char *flags = nullptr);

using FloatC  = CUDAArray<float>;
using Int32C  = CUDAArray<int32_t>;
using UInt32C = CUDAArray<uint32_t>;
using MaskC   = CUDAArray<bool>;
using HalfC   = CUDAArray<drjit::half>;
using FloatL  = LLVMArray<float>;
using Int32L  = LLVMArray<int32_t>;
using UInt32L = LLVMArray<uint32_t>;
using MaskL   = LLVMArray<bool>;
using HalfL   = LLVMArray<drjit::half>;

#define TEST_REGISTER_CUDA(name, suffix, FloatType, ...)                       \
    int test##name##_##suffix =                                                \
        test_register("test" #name#suffix,                                     \
                      test##name<JitBackend::CUDA, FloatType, Int32C, UInt32C, \
                                 MaskC, CUDAArray>,                            \
                      ##__VA_ARGS__);

#define TEST_REGISTER_OPTIX(name, suffix, FloatType, ...)                      \
    int test##name##_##suffix =                                                \
        test_register("test" #name#suffix,                                     \
                      test##name<JitBackend::CUDA, FloatType, Int32C, UInt32C, \
                                 MaskC, CUDAArray>,                            \
                      ##__VA_ARGS__);

#define TEST_REGISTER_LLVM(name, suffix, FloatType, ...)                       \
    int test##name##_##suffix =                                                \
        test_register("test" #name#suffix,                                     \
                      test##name<JitBackend::LLVM, FloatType, Int32L, UInt32L, \
                                 MaskL, LLVMArray>,                            \
                      ##__VA_ARGS__);

#define TEST_CUDA(name, ...)                                                   \
    template <JitBackend Backend, typename Float, typename Int32,              \
              typename UInt32, typename Mask, template <class> class Array>    \
    void test##name();                                                         \
    TEST_REGISTER_CUDA(name,    _cuda_fp32,     FloatC)                        \
    TEST_REGISTER_OPTIX(name,   _optix_fp32,    FloatC)                        \
    TEST_REGISTER_CUDA(name,    _cuda_fp16,     HalfC)                         \
    TEST_REGISTER_OPTIX(name,   _optix_fp16,    HalfC)                         \
    template <JitBackend Backend, typename Float, typename Int32,              \
              typename UInt32, typename Mask, template <class> class Array>    \
    void test##name()

#define TEST_LLVM(name, ...)                                                   \
    template <JitBackend Backend, typename Float, typename Int32,              \
              typename UInt32, typename Mask, template <class> class Array>    \
    void test##name();                                                         \
    TEST_REGISTER_LLVM(name,    _llvm_fp32,     FloatL)                        \
    TEST_REGISTER_LLVM(name,    _llvm_fp16,     HalfL)                         \
    template <JitBackend Backend, typename Float, typename Int32,              \
              typename UInt32, typename Mask, template <class> class Array>    \
    void test##name()

#define TEST_BOTH(name, ...)                                                   \
    template <JitBackend Backend, typename Float, typename Int32,              \
              typename UInt32, typename Mask, template <class> class Array>    \
    void test##name();                                                         \
    TEST_REGISTER_CUDA(name,    _cuda_fp32,     FloatC)                        \
    TEST_REGISTER_OPTIX(name,   _optix_fp32,    FloatC)                        \
    TEST_REGISTER_CUDA(name,    _cuda_fp16,     HalfC)                         \
    TEST_REGISTER_OPTIX(name,   _optix_fp16,    HalfC)                         \
    TEST_REGISTER_LLVM(name,    _llvm_fp32,     FloatL)                        \
    TEST_REGISTER_LLVM(name,    _llvm_fp16,     HalfL)                         \
    template <JitBackend Backend, typename Float, typename Int32,              \
              typename UInt32, typename Mask, template <class> class Array>    \
    void test##name()

#define TEST_BOTH_FP32(name, ...)                                              \
    template <JitBackend Backend, typename Float, typename Int32,              \
              typename UInt32, typename Mask, template <class> class Array>    \
    void test##name();                                                         \
    TEST_REGISTER_CUDA(name,    _cuda,     FloatC)                             \
    TEST_REGISTER_OPTIX(name,   _optix,    FloatC)                             \
    TEST_REGISTER_LLVM(name,    _llvm,     FloatL)                             \
    template <JitBackend Backend, typename Float, typename Int32,              \
              typename UInt32, typename Mask, template <class> class Array>    \
    void test##name()

#define TEST_CUDA_FP32(name, ...)                                              \
    template <JitBackend Backend, typename Float, typename Int32,              \
              typename UInt32, typename Mask, template <class> class Array>    \
    void test##name();                                                         \
    TEST_REGISTER_CUDA(name,    _cuda,     FloatC)                             \
    TEST_REGISTER_OPTIX(name,   _optix,    FloatC)                             \
    template <JitBackend Backend, typename Float, typename Int32,              \
              typename UInt32, typename Mask, template <class> class Array>    \
    void test##name()

#define TEST_BOTH_FLOAT_AGNOSTIC(name, ...) TEST_BOTH_FP32(name, ##__VA_ARGS__)

#define TEST_REDUCE_UNSUPPORTED_SKIP(command)                                  \
    try {                                                                      \
        command;                                                               \
    } catch (const std::runtime_error &err) {                                  \
        if (strstr(err.what(),                                                 \
            "does not support the requested type of atomic reduction") == NULL)\
            throw err;                                                         \
                                                                               \
        jit_log(LogLevel::Warn, "Skipping test! %s", err.what());              \
        return;                                                                \
    }                                                                          \

#define jit_assert(cond)                                                       \
    do {                                                                       \
        if (!(cond))                                                           \
            jit_fail("Assertion failure: %s in line %u.", #cond, __LINE__);    \
    } while (0)

/// RAII helper for temporarily decreasing the log level
struct scoped_set_log_level {
public:
    scoped_set_log_level(LogLevel level) {
        m_cb_level = jit_log_level_callback();
        m_stderr_level = jit_log_level_stderr();
        jit_set_log_level_stderr(level < m_stderr_level ? level
                                                         : m_stderr_level);
    }

    ~scoped_set_log_level() {
        jit_set_log_level_stderr(m_stderr_level);
    }

private:
    LogLevel m_cb_level;
    LogLevel m_stderr_level;
};

