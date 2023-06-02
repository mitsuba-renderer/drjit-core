#pragma once

#include <drjit-core/array.h>
#include <cstdio>
#include <cstring>

using namespace drjit;

static constexpr LogLevel Error = LogLevel::Error;
static constexpr LogLevel Warn  = LogLevel::Warn;
static constexpr LogLevel Info  = LogLevel::Info;
static constexpr LogLevel Debug = LogLevel::Debug;
static constexpr LogLevel Trace = LogLevel::Trace;

extern int test_register(const char *name, void (*func)(), const char *flags = nullptr);
extern "C" void log_level_callback(LogLevel cb, const char *msg);

#define INST_TEST(Backend, Suffix, name_suffix, name, ...)                     \
    int test##name##_##S = test_register(                                      \
        "test" #name "_" name_suffix,                                          \
        test##name<JitBackend::Backend, Float##Suffix, Int32##Suffix,          \
                   UInt32##Suffix, Mask##Suffix, Backend##Array>,              \
        ##__VA_ARGS__);

#if defined(DRJIT_ENABLE_LLVM)
#  define INST_LLVM(name) INST_TEST(LLVM, L, "llvm", name, __VA_ARGS__)
   using FloatL  = LLVMArray<float>;
   using Int32L  = LLVMArray<int32_t>;
   using UInt32L = LLVMArray<uint32_t>;
   using MaskL   = LLVMArray<bool>;
#else
#  define INST_LLVM(...)
#endif

#if defined(DRJIT_ENABLE_CUDA)
#  define INST_CUDA(name) INST_TEST(CUDA, C, "cuda", name, __VA_ARGS__)
#  define INST_OPTIX(name) INST_TEST(CUDA, C, "optix", name, __VA_ARGS__)
   using FloatC  = CUDAArray<float>;
   using Int32C  = CUDAArray<int32_t>;
   using UInt32C = CUDAArray<uint32_t>;
   using MaskC   = CUDAArray<bool>;
#else
#  define INST_CUDA(...)
#  define INST_OPTIX(...)
#endif

#if defined(DRJIT_ENABLE_METAL)
#  define INST_METAL(name) INST_TEST(METAL, M, "metal", name, __VA_ARGS__)
   using FloatM  = MetalArray<float>;
   using Int32M  = MetalArray<int32_t>;
   using UInt32M = MetalArray<uint32_t>;
   using MaskM   = MetalArray<bool>;
#else
#  define INST_METAL(...)
#endif

#define DECL_TEST(name)                                                        \
    template <JitBackend Backend, typename Float, typename Int32,              \
              typename UInt32, typename Mask, template <class> class Array>    \
    void test##name()                                                          \

#define TEST_CUDA(name, ...)                                                   \
    DECL_TEST(name);                                                           \
    INST_CUDA(name, __VA_ARGS__)                                               \
    INST_OPTIX(name, __VA_ARGS__)                                              \
    DECL_TEST(name)

#define TEST_LLVM(name, ...)                                                   \
    DECL_TEST(name);                                                           \
    INST_LLVM(name, __VA_ARGS__)                                               \
    DECL_TEST(name)

#define TEST_METAL(name, ...)                                                  \
    DECL_TEST(name);                                                           \
    INST_METAL(name, __VA_ARGS__)                                              \
    DECL_TEST(name)

#define TEST_ALL(name, ...)                                                    \
    DECL_TEST(name);                                                           \
    INST_LLVM(name, __VA_ARGS__)                                               \
    INST_CUDA(name, __VA_ARGS__)                                               \
    INST_OPTIX(name, __VA_ARGS__)                                              \
    INST_METAL(name, __VA_ARGS__)                                              \
    DECL_TEST(name)

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
        jit_set_log_level_callback(level < m_cb_level ? level : m_cb_level,
                                    log_level_callback);
        jit_set_log_level_stderr(level < m_stderr_level ? level
                                                         : m_stderr_level);
    }

    ~scoped_set_log_level() {
        jit_set_log_level_callback(m_cb_level, log_level_callback);
        jit_set_log_level_stderr(m_stderr_level);
    }

private:
    LogLevel m_cb_level;
    LogLevel m_stderr_level;
};
