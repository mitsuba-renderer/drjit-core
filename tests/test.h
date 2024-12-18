#pragma once

#include <drjit-core/array.h>
#include <drjit-core/half.h>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <cstring>
#include <vector>
#include <tuple>
#include <type_traits>
#include <utility>

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

template <typename Array>
Array tile(const Array &source, uint32_t count) {
    return Array::steal(jit_var_tile(source.index(), count));
}

template <typename Array>
Array repeat(const Array &source, uint32_t count) {
    return Array::steal(jit_var_repeat(source.index(), count));
}

/// Operation, that can be applied to nested C++ traversable types.
/// The function receives a non-borrowing variable index from the \c JitArray it
/// is applied to and has to return an owned reference, transferring the
/// ownership back to the \c JitArray.
using apply_op = std::function<uint32_t(uint32_t)>;

/// Traversable type, used to traverse frozen function inputs and outputs.
template <typename T, typename = void> struct traversable {
    static constexpr bool value = false;
    /// Apply the operation \c op to the C++ value v
    static void apply(const apply_op &cb, T &v) {
        (void) v;
        (void) cb;
    }
};

template <typename T>
struct traversable<T, std::void_t<decltype(T::steal(0)),
                                  decltype(std::declval<T>().index())>> {
    static constexpr bool value = true;
    static void apply(const apply_op &cb, T &v) { v = T::steal(cb(v.index())); }
};

template <typename Tuple, std::size_t... I>
static void apply_tuple(const apply_op &cb, Tuple &t,
                        std::index_sequence<I...>) {
    // Expands in left-to-right order
    (traversable<std::decay_t<decltype(std::get<I>(t))>>::apply(cb,
                                                                std::get<I>(t)),
     ...);
}

template <typename... Ts> struct traversable<std::tuple<Ts...>> {
    static constexpr bool value = true;
    static void apply(const apply_op &cb, std::tuple<Ts...> &v) {
        apply_tuple(cb, v, std::index_sequence_for<Ts...>{});
    }
};

template <typename... Args>
static void apply_arguments(const apply_op &cb, Args &&...args) {
    (traversable<std::decay_t<Args>>::apply(cb, args), ...);
}

template <typename... Args> static void make_opaque(Args &&...args) {
    auto op = [](uint32_t index) {
        int rv;
        uint32_t new_index = jit_var_schedule_force(index, &rv);
        return new_index;
    };
    apply_arguments(op, args...);
    jit_eval();
}

/// Constructable type, used to construct frozen function outputs
template <typename T, typename = void> struct constructable {
    static constexpr bool value = false;
    static T construct(const std::function<uint32_t()> & /*cb*/) {
        static_assert(sizeof(T) == 0, "Could not construct type!");
    }
};

/// Construct any variable that has the \c borrow function
/// We have to use the \c borrow function instead of \c steal, since we would
/// split ownership between outputs if they appear twice in the type.
template <typename T>
struct constructable<T, std::void_t<decltype(T::borrow(0))>> {
    static constexpr bool value = true;
    static T construct(const std::function<uint32_t()> &cb) {
        return T::borrow(cb());
    }
};

template <typename... Ts> struct constructable<std::tuple<Ts...>> {
    static constexpr bool value = true;
    static std::tuple<Ts...> construct(const std::function<uint32_t()> &cb) {
        // NOTE: initializer list to guarantee order of construct evaluation
        return std::tuple{ constructable<Ts>::construct(cb)... };
    }
};

/**
 * \brief Minimal implementation of a FrozenFunction using the \c
 * RecordThreadState
 *
 * This struct contains a single recording, that will be recorded when the
 * function is first called.
 * There are no checks that validate that the input layout hasn't changed.
 * The registry is also not traversed, as this requires the registry pointers to
 * inherit from \c nanobind::intrusive_base.
 */
template <typename Func> class FrozenFunction {

    JitBackend m_backend;
    Func m_func;

    uint32_t m_outputs     = 0;
    Recording *m_recording = nullptr;

public:
    FrozenFunction(JitBackend backend, Func func)
        : m_backend(backend), m_func(func), m_outputs(0) {
        jit_log(LogLevel::Debug, "FrozenFunction()");
    }
    ~FrozenFunction() {
        if (m_recording)
            jit_freeze_destroy(m_recording);
        m_recording = nullptr;
    }

    void clear() {
        if (m_recording) {
            jit_freeze_destroy(m_recording);
            m_recording = nullptr;
            m_outputs   = 0;
        }
    }

    template <typename... Args>
    auto record(std::vector<uint32_t> &input_vector, Args &&...args) {
        using Output = typename std::invoke_result<Func, Args...>::type;
        Output output;

        jit_log(LogLevel::Debug, "record:");

        jit_freeze_start(m_backend, input_vector.data(), input_vector.size());

        // Record the function, including evaluation of all side effects on the
        // inputs and outputs
        {
            output = m_func(std::forward<Args>(args)...);

            make_opaque(output, args...);
        }

        // Traverse output for \c jit_freeze_stop
        // NOTE: in the implementation in drjit, we would also schedule the
        // input and re-assign it. Since we pass variables by value, modified
        // inputs have to be passed to the output explicitly.
        std::vector<uint32_t> output_vector;
        {
            auto op = [&output_vector](uint32_t index) {
                // Take non borrowing reference to the index
                output_vector.push_back(index);

                // Transfer ownership back to the \c JitArray
                jit_var_inc_ref(index);
                return index;
            };

            traversable<Output>::apply(op, output);
        }

        m_recording = jit_freeze_stop(m_backend, output_vector.data(),
                                      output_vector.size());
        m_outputs   = (uint32_t) output_vector.size();

        uint32_t counter = 0;

        // Construct output
        {
            output =
                constructable<Output>::construct([&counter, &output_vector] {
                    return output_vector[counter++];
                });
        }

        // Output does not have to be released, as it is not borrowed, just
        // referenced

        return output;
    }

    template <typename... Args>
    auto replay(std::vector<uint32_t> &input_vector, Args &&...args) {
        using Output = typename std::invoke_result<Func, Args...>::type;

        jit_log(LogLevel::Debug, "dry run:");

        int dryrun_success =
            jit_freeze_dry_run(m_recording, input_vector.data());

        if (!dryrun_success) {
            clear();

            return record(input_vector, args...);
        } else {
            std::vector<uint32_t> output_vector(m_outputs, 0);

            jit_log(LogLevel::Debug, "replay:");
            // replay adds borrowing references to the \c output_vector
            jit_freeze_replay(m_recording, input_vector.data(),
                              output_vector.data());

            // Construct output
            uint32_t counter = 0;
            Output output =
                constructable<Output>::construct([&counter, &output_vector] {
                    return output_vector[counter++];
                });

            // Release the borrowed indices
            for (uint32_t index : output_vector)
                jit_var_dec_ref(index);

            return output;
        }
    }

    template <typename... Args> auto operator()(Args &&...args) {
        using Output = typename std::invoke_result<Func, Args...>::type;

        make_opaque(args...);

        // Make input opaque and add it to \c input_vector, borrowing it
        std::vector<uint32_t> input_vector;
        auto op = [&input_vector](uint32_t index) {
            // Borrow from the index and add it to the input_vector
            jit_var_inc_ref(index);
            input_vector.push_back(index);

            // Transfer ownership back to the \c JitArray
            jit_var_inc_ref(index);
            return index;
        };
        apply_arguments(op, args...);

        Output output;
        if (!m_recording)
            output = record(input_vector, args...);
        else
            output = replay(input_vector, args...);

        // Release the borrowed indices
        for (uint32_t i = 0; i < input_vector.size(); i++)
            jit_var_dec_ref(input_vector[i]);

        return output;
    }
};
