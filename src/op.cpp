#include <drjit-core/nanostl.h>
#include <drjit-core/half.h>
#include "internal.h"
#include "llvm.h"
#include "var.h"
#include "log.h"
#include "eval.h"
#include "util.h"
#include "op.h"
#include "array.h"

#if defined(_MSC_VER)
#  pragma warning (disable: 4702) // unreachable code
#endif

template <bool Value> using enable_if_t = std::enable_if_t<Value, int>;

/// Various checks that can be requested from jitc_var_check()
enum CheckFlags {
    IsArithmetic = 1,
    IsInt = 2,
    IsIntOrBool = 4,
    IsNotVoid = 8,
    IsFloat = 16,
    IsCUDA = 32,
    ArrayAllowed = 64,
    IgnoreTypes = 128
};

/// Summary information about a set of variables returned by jitc_var_check()
struct OpInfo {
    /// Backend (assuming consistent operands)
    JitBackend backend;

    /// Type (assuming consistent operands)
    VarType type;

    /// Output size given the size of the operands
    uint32_t size;

    /// Should Dr.Jit try to simplify the operation? (if some operands are literals)
    bool simplify;

    /// Are *all* operands literals?
    bool literal;

    /// Did an operand have the 'symbolic' bit set?
    bool symbolic;
};

/**
 * \brief Check a set of variable indices and return summary information about them
 *
 * This function checks that the input arguments have a compatible size In
 * debug mode, it performs further checks (all of these are technically
 * "impossible" because the type system of the Dr.Jit C++ wrapper should have
 * caught them. But we still check just in case..):
 *
 * - It ensures that all arguments have a consistent backend
 * - It checks that the input types are consistent (if Check != Disabled)
 * - If Check == IsArithmetic, IsFloat, etc., it checks that the types make sense
 *
 * Note that the function does not check if any input arguments have pending
 * side effects (`is_dirty() == true`). This step is postponed to the
 * `jitc_var_new_node_x()` function to permit simplifications of operations
 * combining dirty and literal arguments (this is important for autodiff).
 *
 * Finally, the function returns a tuple containing a `OpInfo` record (see the
 * documentation of its fields for details) and a `const Variable *` pointer
 * per operand.
 */
template <int Flags, typename... Args, size_t... Is>
auto jitc_var_check_impl(const char *name, std::index_sequence<Is...>, Args... args) {
    constexpr size_t Size = sizeof...(Args);
    uint32_t dep[Size] = { args... };
    Variable *v[Size];

    bool symbolic = false,
         simplify = false,
         array = false,
         literal = true;

    JitBackend backend = JitBackend::None;
    VarType type = VarType::Void;
    uint32_t size = 0;
    const char *err = nullptr;

    for (uint32_t i = 0; i < Size; ++i) {
        if (!dep[i]) {
            v[i] = nullptr;
            continue;
        }
        Variable *vi = jitc_var(dep[i]);

#if !defined(NDEBUG)
        if constexpr (bool(Flags & IsArithmetic)) {
            if (unlikely(!jitc_is_arithmetic(vi))) {
                err = "expected arithmetic operand types";
                goto fail;
            }
        }

        if constexpr (bool(Flags & IsInt)) {
            if (unlikely(!jitc_is_int(vi))) {
                err = "expected integer operand types";
                goto fail;
            }
        }

        if constexpr (bool(Flags & IsIntOrBool)) {
            if (unlikely(!jitc_is_int(vi) && !jitc_is_bool(vi))) {
                err = "expected integer or boolean operand types";
                goto fail;
            }
        }

        if constexpr (bool(Flags & IsNotVoid)) {
            if (unlikely(jitc_is_void(vi))) {
                err = "operand cannot be void";
                goto fail;
            }
        }

        if constexpr (bool(Flags & IsFloat)) {
            if (unlikely(!jitc_is_float(vi))) {
                err = "expected floating point operand types";
                goto fail;
            }
        }

        if constexpr (bool(Flags & IsCUDA)) {
            if (unlikely((JitBackend) vi->backend != JitBackend::CUDA)) {
                err = "operation is only supported on the CUDA backend";
                goto fail;
            }
        }

        if constexpr (!bool(Flags & IgnoreTypes)) {
            if (unlikely(type != VarType::Void && (VarType) vi->type != type)) {
                err = "operands have incompatible types";
                goto fail;
            }
        }

        if (unlikely(backend != JitBackend::None && (JitBackend) vi->backend != backend)) {
            err = "operands have different backends";
            goto fail;
        }

#endif
        if (vi->consumed) {
            err = "operation references an operand that can only be evaluated once";
            goto fail;
        }

        size = std::max(size, vi->size);
        symbolic |= (bool) vi->symbolic;
        array |= (bool) vi->is_array();
        bool is_literal = vi->is_literal();
        literal &= is_literal;
        simplify |= is_literal;
        backend = (JitBackend) vi->backend;
        if (type == VarType::Void)
            type = (VarType) vi->type;
        v[i] = vi;
    }

    if (size > 0) {
        // Try simplifying binary expressions with matched arguments
        if constexpr (Size == 2)
            simplify |= dep[0] == dep[1];

        for (uint32_t i = 0; i < Size; ++i) {
            uint32_t vs = v[i] ? v[i]->size : 0;
            if (unlikely(vs != size && vs != 1)) {
                err = "operands have incompatible sizes";
                size = (uint32_t) -1;
                goto fail;
            }
        }

        if (simplify)
            simplify = jitc_flags() & (uint32_t) JitFlag::ConstantPropagation;
    }

#if !defined(NDEBUG)
    if constexpr (!bool(Flags & ArrayAllowed)) {
        if (unlikely(array)) {
            err = "array operands are not supported";
            goto fail;
        }
    }
#else
    (void) array;
#endif

    return drjit::tuple(
        OpInfo{ backend, type, size, simplify, literal, symbolic },
        v[Is]...
    );

fail:
    buffer.clear();
    buffer.fmt("%s(", name);
    for (uint32_t i = 0; i < Size; ++i)
        buffer.fmt("r%u%s", dep[i], i + 1 < Size ? ", " : "");
    buffer.fmt("): %s!", err);

    if (size == (uint32_t) -1) {
        buffer.put(" (sizes: ");
        for (uint32_t i = 0; i < Size; ++i)
            buffer.fmt("%u%s", dep[i] ? jitc_var(dep[i])->size : 0,
                       i + 1 < Size ? ", " : "");
        buffer.put(")");
    }

    throw std::runtime_error(buffer.get());
}

template <int Flags = IgnoreTypes, typename... Args>
JIT_INLINE auto jitc_var_check(const char *name, Args... args) {
    return jitc_var_check_impl<Flags>(
        name, std::make_index_sequence<sizeof...(Args)>(), args...);
}

// Convert from an 64-bit integer container to a literal type
template <typename Type> Type i2v(uint64_t value) {
    Type result;
    memcpy((void*)&result, &value, sizeof(Type));
    return result;
}

// Convert from a literal type to a 64-bit integer container
template <typename Type> uint64_t v2i(Type value) {
    uint64_t result;
    if constexpr (std::is_same_v<Type, bool>) {
        result = value ? 1 : 0;
    } else {
        result = 0;
        memcpy(&result, &value, sizeof(Type));
    }
    return result;
}

template <typename Dst, typename Src>
Dst memcpy_cast(const Src &src) {
    static_assert(sizeof(Src) == sizeof(Dst), "memcpy_cast: size mismatch!");
    Dst dst;
    memcpy((void*)&dst, &src, sizeof(Dst));
    return dst;
}

template <typename T, typename... Ts> T first(T arg, Ts...) { return arg; }

template <typename Func, typename... Args>
JIT_INLINE uint32_t jitc_eval_literal(const OpInfo &info, Func func,
                                      const Args *...args) {
    uint64_t r = 0;

    using half = drjit::half;

    switch ((VarType) first(args...)->type) {
        case VarType::Bool:    r = v2i(func(i2v<   bool> (args->literal)...)); break;
        case VarType::Int8:    r = v2i(func(i2v< int8_t> (args->literal)...)); break;
        case VarType::UInt8:   r = v2i(func(i2v<uint8_t> (args->literal)...)); break;
        case VarType::Int16:   r = v2i(func(i2v< int16_t>(args->literal)...)); break;
        case VarType::UInt16:  r = v2i(func(i2v<uint16_t>(args->literal)...)); break;
        case VarType::Int32:   r = v2i(func(i2v< int32_t>(args->literal)...)); break;
        case VarType::UInt32:  r = v2i(func(i2v<uint32_t>(args->literal)...)); break;
        case VarType::Int64:   r = v2i(func(i2v< int64_t>(args->literal)...)); break;
        case VarType::UInt64:  r = v2i(func(i2v<uint64_t>(args->literal)...)); break;
        case VarType::Float16: r = v2i(func(i2v<    half>(args->literal)...)); break;
        case VarType::Float32: r = v2i(func(i2v<   float>(args->literal)...)); break;
        case VarType::Float64: r = v2i(func(i2v<  double>(args->literal)...)); break;
        default: jitc_fail("jit_eval_literal(): unsupported variable type!");
    }

    return jitc_var_literal(info.backend, info.type, &r, info.size, 0);
}

// --------------------------------------------------------------------------
// Common constants
// --------------------------------------------------------------------------

uint32_t jitc_make_zero(OpInfo info) {
    uint64_t value = 0;
    return jitc_var_literal(info.backend, info.type, &value, info.size, 0);
}

uint32_t jitc_make_true(OpInfo info) {
    bool value = true;
    return jitc_var_literal(info.backend, VarType::Bool, &value, info.size, 0);
}

// --------------------------------------------------------------------------
// Helper routines for turning multiplies and divisions into shifts
// --------------------------------------------------------------------------

bool jitc_is_pow2(uint64_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

static int jitc_clz(uint64_t value) {
#if !defined(_MSC_VER)
    return __builtin_clzl(value);
#else
    int lz = 64;
    while (value) {
        value >>= 1;
        lz -= 1;
    }
    return lz;
#endif
}

template <bool Shl>
uint32_t jitc_var_shift(const OpInfo &info, uint32_t index, uint64_t amount) {
    amount = 63 - jitc_clz(amount);
    Ref shift = steal(jitc_var_literal(info.backend, info.type, &amount, info.size, 0));
    return Shl ? jitc_var_shl(index, shift)
               : jitc_var_shr(index, shift);
}

// --------------------------------------------------------------------------

template <typename T, enable_if_t<!drjit::detail::is_signed_v<T>> = 0>
T eval_neg(T v) { return T(-(std::make_signed_t<T>) v); }

template <typename T, enable_if_t<drjit::detail::is_signed_v<T>> = 0>
T eval_neg(T v) { return -v; }

static bool eval_neg(bool) { jitc_fail("eval_neg(): unsupported operands!"); }

uint32_t jitc_var_neg(uint32_t a0) {
    auto [info, v0] = jitc_var_check<IsArithmetic>("jit_var_neg", a0);

    uint32_t result = 0;
    if (info.simplify && info.literal)
        result = jitc_eval_literal(info, [](auto l0) { return eval_neg(l0); }, v0);

    if (!result && info.size)
        result = jitc_var_new_node_1(info.backend, VarKind::Neg, info.type,
                                     info.size, info.symbolic, a0, v0);

    jitc_trace("jit_var_neg(r%u <- r%u)", result, a0);
    return result;
}

// --------------------------------------------------------------------------

template <typename T> T eval_not(T v) {
    using U = uint_with_size_t<T>;
    return memcpy_cast<T>(U(~memcpy_cast<U>(v)));
}

static bool eval_not(bool v) { return !v; }

uint32_t jitc_var_not(uint32_t a0) {
    auto [info, v0] = jitc_var_check<IsIntOrBool>("jit_var_not", a0);

    uint32_t result = 0;
    if (info.simplify && info.literal)
        result = jitc_eval_literal(info, [](auto l0) { return eval_not(l0); }, v0);

    if (!result && info.size)
        result = jitc_var_new_node_1(info.backend, VarKind::Not, info.type,
                                     info.size, info.symbolic, a0, v0);

    jitc_trace("jit_var_not(r%u <- r%u)", result, a0);
    return result;
}

// --------------------------------------------------------------------------

template <typename T, enable_if_t<drjit::detail::is_floating_point_v<T>> = 0>
T eval_sqrt(T value) { using std::sqrt, drjit::half; return T(::sqrt(value)); }

template <typename T, enable_if_t<!drjit::detail::is_floating_point_v<T>> = 0>
T eval_sqrt(T) { jitc_fail("eval_sqrt(): unsupported operands!"); }

uint32_t jitc_var_sqrt(uint32_t a0) {
    auto [info, v0] = jitc_var_check<IsFloat>("jit_var_sqrt", a0);

    uint32_t result = 0;
    if (info.simplify && info.literal)
        result = jitc_eval_literal(info, [](auto l0) { return eval_sqrt(l0); }, v0);

    if (!result && info.size) {
        bool approx =
            info.backend == JitBackend::CUDA && info.type == VarType::Float32;
        result = jitc_var_new_node_1(
            info.backend, approx ? VarKind::SqrtApprox : VarKind::Sqrt,
            info.type, info.size, info.symbolic, a0, v0);
    }

    jitc_trace("jit_var_sqrt(r%u <- r%u)", result, a0);
    return result;
}

// --------------------------------------------------------------------------

template <typename T, enable_if_t<drjit::detail::is_signed_v<T>> = 0>
T eval_abs(T value) { return (T) std::abs(value); }

template <typename T, enable_if_t<!drjit::detail::is_signed_v<T>> = 0>
T eval_abs(T value) { return value; }

uint32_t jitc_var_abs(uint32_t a0) {
    auto [info, v0] = jitc_var_check<IsArithmetic>("jit_var_abs", a0);

    uint32_t result = 0;
    if (info.simplify && info.literal)
        result = jitc_eval_literal(info, [](auto l0) { return eval_abs(l0); }, v0);

    if (!result && jitc_is_uint(info.type))
        result = jitc_var_new_ref(a0);

    if (!result && info.size)
        result = jitc_var_new_node_1(info.backend, VarKind::Abs, info.type,
                                     info.size, info.symbolic, a0, v0);

    jitc_trace("jit_var_abs(r%u <- r%u)", result, a0);
    return result;
}

// --------------------------------------------------------------------------

uint32_t jitc_var_add(uint32_t a0, uint32_t a1) {
    auto [info, v0, v1] = jitc_var_check<IsArithmetic>("jit_var_add", a0, a1);

    uint32_t result = 0;
    if (info.simplify) {
        if (info.literal)
            result = jitc_eval_literal(
                info, [](auto l0, auto l1) { return l0 + l1; }, v0, v1);
        else if (jitc_is_any_zero(v0))
            result = jitc_var_resize(a1, info.size);
        else if (jitc_is_any_zero(v1))
            result = jitc_var_resize(a0, info.size);
    }

    if (!result && info.size)
        result = jitc_var_new_node_2(info.backend, VarKind::Add, info.type,
                                     info.size, info.symbolic, a0, v0, a1, v1);

    jitc_trace("jit_var_add(r%u <- r%u, r%u)", result, a0, a1);
    return result;
}

// --------------------------------------------------------------------------

uint32_t jitc_var_sub(uint32_t a0, uint32_t a1) {
    auto [info, v0, v1] = jitc_var_check<IsArithmetic>("jit_var_sub", a0, a1);

    uint32_t result = 0;
    if (info.simplify) {
        if (info.literal)
            result = jitc_eval_literal(
                info, [](auto l0, auto l1) { return l0 - l1; }, v0, v1);
        else if (jitc_is_any_zero(v1))
            result = jitc_var_resize(a0, info.size);
        else if (a0 == a1 && !jitc_is_float(v0))
            result = jitc_make_zero(info);
    }

    if (!result && info.size)
        result = jitc_var_new_node_2(info.backend, VarKind::Sub, info.type,
                                    info.size, info.symbolic, a0, v0, a1, v1);

    jitc_trace("jit_var_sub(r%u <- r%u, r%u)", result, a0, a1);
    return result;
}

// --------------------------------------------------------------------------

uint32_t jitc_var_mul(uint32_t a0, uint32_t a1) {
    auto [info, v0, v1] = jitc_var_check<IsArithmetic>("jit_var_mul", a0, a1);

    uint32_t result = 0;
    if (info.simplify) {
        bool fast_math = jit_flags() & (uint32_t) JitFlag::FastMath;

        if (info.literal)
            result = jitc_eval_literal(
                info, [](auto l0, auto l1) { return l0 * l1; }, v0, v1);
        else if (jitc_is_one(v0) || (jitc_is_any_zero(v1) && (jitc_is_int(v0) || fast_math)))
            result = jitc_var_resize(a1, info.size);
        else if (jitc_is_one(v1) || (jitc_is_any_zero(v0) && (jitc_is_int(v0) || fast_math)))
            result = jitc_var_resize(a0, info.size);
        else if (jitc_is_uint(info.type) && v0->is_literal() && jitc_is_pow2(v0->literal))
            result = jitc_var_shift<true>(info, a1, v0->literal);
        else if (jitc_is_uint(info.type) && v1->is_literal() && jitc_is_pow2(v1->literal))
            result = jitc_var_shift<true>(info, a0, v1->literal);
    }

    if (!result && info.size)
        result = jitc_var_new_node_2(info.backend, VarKind::Mul, info.type,
                                     info.size, info.symbolic, a0, v0, a1, v1);

    jitc_trace("jit_var_mul(r%u <- r%u, r%u)", result, a0, a1);
    return result;
}

// --------------------------------------------------------------------------

template <typename T> T eval_div(T v0, T v1) { return v0 / v1; }

static bool eval_div(bool, bool) { jitc_fail("eval_div(): unsupported operands!"); }

uint32_t jitc_var_div(uint32_t a0, uint32_t a1) {
    auto [info, v0, v1] = jitc_var_check<IsArithmetic>("jit_var_div", a0, a1);

    uint32_t result = 0;
    if (info.simplify) {
        if (info.literal) {
            result = jitc_eval_literal(
                info, [](auto l0, auto l1) { return eval_div(l0, l1); }, v0, v1);
        } else if (jitc_is_one(v1)) {
            result = jitc_var_resize(a0, info.size);
        } else if (jitc_is_uint(info.type) && v1->is_literal() && jitc_is_pow2(v1->literal)) {
            result = jitc_var_shift<false>(info, a0, v1->literal);
        } else if (jitc_is_float(info.type) && !jitc_is_half(info.type) && v1->is_literal()) {
            uint32_t recip = jitc_var_rcp(a1);
            result = jitc_var_mul(a0, recip);
            jitc_var_dec_ref(recip);
        } else if (a0 == a1 && !jitc_is_float(v0)) {
            uint64_t value = 1;
            result = jitc_var_literal(info.backend, info.type, &value, info.size, 0);
        }
    }


    if (!result && info.size) {
        bool approx =
            info.backend == JitBackend::CUDA && info.type == VarType::Float32;
        result = jitc_var_new_node_2(
            info.backend, approx ? VarKind::DivApprox : VarKind::Div, info.type,
            info.size, info.symbolic, a0, v0, a1, v1);
    }

    jitc_trace("jit_var_div(r%u <- r%u, r%u)", result, a0, a1);
    return result;
}

// --------------------------------------------------------------------------

template <typename T, enable_if_t<!std::is_integral_v<T> || std::is_same_v<T, bool>> = 0>
T eval_mod(T, T) { jitc_fail("eval_mod(): unsupported operands!"); }

template <typename T, enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>> = 0>
T eval_mod(T v0, T v1) { return v0 % v1; }

uint32_t jitc_var_mod(uint32_t a0, uint32_t a1) {
    auto [info, v0, v1] = jitc_var_check<IsIntOrBool>("jit_var_mod", a0, a1);

    uint32_t result = 0;
    if (info.simplify && info.literal) {
        result = jitc_eval_literal(
            info, [](auto l0, auto l1) { return eval_mod(l0, l1); }, v0, v1);
    } else if (v1->is_literal() && jitc_is_one(v1)) {
        uint64_t value = 0;
        result = jitc_var_literal(info.backend, info.type, &value, info.size, 0);
    }

    if (!result && info.size)
        result = jitc_var_new_node_2(info.backend, VarKind::Mod, info.type,
                                     info.size, info.symbolic, a0, v0, a1, v1);

    jitc_trace("jit_var_mod(r%u <- r%u, r%u)", result, a0, a1);
    return result;
}

// --------------------------------------------------------------------------

template <typename T, enable_if_t<!(std::is_integral_v<T> && (sizeof(T) == 4 || sizeof(T) == 8))> = 0>
T eval_mulhi(T, T) { jitc_fail("eval_mulhi(): unsupported operands!"); }

template <typename T, enable_if_t<std::is_integral_v<T> && (sizeof(T) == 4 || sizeof(T) == 8)> = 0>
T eval_mulhi(T a, T b) {
    if constexpr (sizeof(T) == 4) {
        using Wide = std::conditional_t<drjit::detail::is_signed_v<T>, int64_t, uint64_t>;
        return T(((Wide) a * (Wide) b) >> 32);
    } else {
#if defined(_MSC_VER)
        if constexpr (drjit::detail::is_signed_v<T>)
            return (T) __mulh((__int64) a, (__int64) b);
        else
            return (T) __umulh((unsigned __int64) a, (unsigned __int64) b);
#else
        using Wide = std::conditional_t<drjit::detail::is_signed_v<T>, __int128_t, __uint128_t>;
        return T(((Wide) a * (Wide) b) >> 64);
#endif
    }
}

uint32_t jitc_var_mulhi(uint32_t a0, uint32_t a1) {
    auto [info, v0, v1] = jitc_var_check<IsInt>("jit_var_mulhi", a0, a1);

    uint32_t result = 0;
    if (info.simplify) {
        if (info.literal)
            result = jitc_eval_literal(
                info, [](auto l0, auto l1) { return eval_mulhi(l0, l1); }, v0, v1);
        else if (jitc_is_zero(v0))
            result = jitc_var_resize(a0, info.size);
        else if (jitc_is_zero(v1))
            result = jitc_var_resize(a1, info.size);
    }

    if (!result && info.size)
        result = jitc_var_new_node_2(info.backend, VarKind::Mulhi, info.type,
                                     info.size, info.symbolic, a0, v0, a1, v1);

    jitc_trace("jit_var_mulhi(r%u <- r%u, r%u)", result, a0, a1);
    return result;
}

// --------------------------------------------------------------------------

template <typename T, enable_if_t<drjit::detail::is_floating_point_v<T>> = 0>
T eval_fma(T a, T b, T c) { return T(std::fma(a, b, c)); }

template <typename T, enable_if_t<!drjit::detail::is_floating_point_v<T> &&
                                  !std::is_same_v<T, bool>> = 0>
T eval_fma(T a, T b, T c) { return (T) (a * b + c); }

template <typename T, enable_if_t<std::is_same<T, bool>::value> = 0>
T eval_fma(T, T, T) { jitc_fail("eval_fma(): unsupported operands!"); }

uint32_t jitc_var_fma(uint32_t a0, uint32_t a1, uint32_t a2) {
    auto [info, v0, v1, v2] = jitc_var_check<IsArithmetic>("jit_var_fma", a0, a1, a2);

    uint32_t result = 0;
    if (info.simplify) {
        if (info.literal) {
            result = jitc_eval_literal(
                info,
                [](auto l0, auto l1, auto l2) { return eval_fma(l0, l1, l2); },
                v0, v1, v2);
        } else {
            bool fast_math = jit_flags() & (uint32_t) JitFlag::FastMath,
                 z0 = jitc_is_any_zero(v0),
                 z1 = jitc_is_any_zero(v1),
                 z2 = jitc_is_any_zero(v2);

            uint32_t tmp = 0;
            if (jitc_is_one(v0))
                tmp = jitc_var_add(a1, a2);
            else if (jitc_is_one(v1))
                tmp = jitc_var_add(a0, a2);
            else if (z2)
                tmp = jitc_var_mul(a0, a1);
            else if ((z0 && z1) || ((z0 || z1) && fast_math))
                tmp = jitc_var_new_ref(a2);

            if (tmp) {
                result = jitc_var_resize(tmp, info.size);
                jitc_var_dec_ref(tmp);
            }
        }
    }

    if (!result && info.size)
        result = jitc_var_new_node_3(info.backend, VarKind::Fma, info.type,
                                     info.size, info.symbolic,
                                     a0, v0, a1, v1, a2, v2);

    jitc_trace("jit_var_fma(r%u <- r%u, r%u, r%u)", result, a0, a1, a2);
    return result;
}

// --------------------------------------------------------------------------

uint32_t jitc_var_min(uint32_t a0, uint32_t a1) {
    auto [info, v0, v1] = jitc_var_check<IsArithmetic>("jit_var_min", a0, a1);

    uint32_t result = 0;
    if (info.simplify) {
        if (info.literal)
            result = jitc_eval_literal(
                info, [](auto l0, auto l1) { return std::min(l0, l1); }, v0, v1);
        else if (jitc_is_max(v0) || jitc_is_min(v1))
            result = jitc_var_resize(a1, info.size);
        else if (jitc_is_max(v1) || jitc_is_min(v0))
            result = jitc_var_resize(a0, info.size);
        else if (a0 == a1)
            result = jitc_var_resize(a0, info.size);
    }

    if (!result && info.size)
        result = jitc_var_new_node_2(info.backend, VarKind::Min, info.type,
                                     info.size, info.symbolic, a0, v0, a1, v1);

    jitc_trace("jit_var_min(r%u <- r%u, r%u)", result, a0, a1);
    return result;
}

// --------------------------------------------------------------------------

uint32_t jitc_var_max(uint32_t a0, uint32_t a1) {
    auto [info, v0, v1] = jitc_var_check<IsArithmetic>("jit_var_max", a0, a1);

    uint32_t result = 0;
    if (info.simplify) {
        if (info.literal)
            result = jitc_eval_literal(
                info, [](auto l0, auto l1) { return std::max(l0, l1); }, v0, v1);
        else if (jitc_is_min(v0) || jitc_is_max(v1))
            result = jitc_var_resize(a1, info.size);
        else if (jitc_is_min(v1) || jitc_is_max(v0))
            result = jitc_var_resize(a0, info.size);
        else if (a0 == a1)
            result = jitc_var_resize(a0, info.size);
    }

    if (!result && info.size)
        result = jitc_var_new_node_2(info.backend, VarKind::Max, info.type,
                                     info.size, info.symbolic, a0, v0, a1, v1);

    jitc_trace("jit_var_max(r%u <- r%u, r%u)", result, a0, a1);
    return result;
}

// --------------------------------------------------------------------------

template <typename T, enable_if_t<drjit::detail::is_floating_point_v<T>> = 0>
T eval_ceil(T value) { return T(std::ceil(value)); }

template <typename T, enable_if_t<!drjit::detail::is_floating_point_v<T>> = 0>
T eval_ceil(T) { jitc_fail("eval_ceil(): unsupported operands!"); }

uint32_t jitc_var_ceil(uint32_t a0) {
    auto [info, v0] = jitc_var_check<IsFloat>("jit_var_ceil", a0);

    uint32_t result = 0;
    if (info.simplify && info.literal)
        result = jitc_eval_literal(info, [](auto l0) { return eval_ceil(l0); }, v0);

    if (!result && info.size)
        result = jitc_var_new_node_1(info.backend, VarKind::Ceil, info.type,
                                     info.size, info.symbolic, a0, v0);

    jitc_trace("jit_var_ceil(r%u <- r%u)", result, a0);
    return result;
}

// --------------------------------------------------------------------------

template <typename T, enable_if_t<drjit::detail::is_floating_point_v<T>> = 0>
T eval_floor(T value) { return T(std::floor(value)); }

template <typename T, enable_if_t<!drjit::detail::is_floating_point_v<T>> = 0>
T eval_floor(T) { jitc_fail("eval_floor(): unsupported operands!"); }

uint32_t jitc_var_floor(uint32_t a0) {
    auto [info, v0] = jitc_var_check<IsFloat>("jit_var_floor", a0);

    uint32_t result = 0;
    if (info.simplify && info.literal)
        result = jitc_eval_literal(info, [](auto l0) { return eval_floor(l0); }, v0);

    if (!result && info.size)
        result = jitc_var_new_node_1(info.backend, VarKind::Floor, info.type,
                                     info.size, info.symbolic, a0, v0);

    jitc_trace("jit_var_floor(r%u <- r%u)", result, a0);
    return result;
}

// --------------------------------------------------------------------------

template <typename T, enable_if_t<drjit::detail::is_floating_point_v<T>> = 0>
T eval_round(T value) { return T(std::rint(value)); }

template <typename T, enable_if_t<!drjit::detail::is_floating_point_v<T>> = 0>
T eval_round(T) { jitc_fail("eval_round(): unsupported operands!"); }

uint32_t jitc_var_round(uint32_t a0) {
    auto [info, v0] = jitc_var_check<IsFloat>("jit_var_round", a0);

    uint32_t result = 0;
    if (info.simplify && info.literal)
        result = jitc_eval_literal(info, [](auto l0) { return eval_round(l0); }, v0);

    if (!result && info.size)
        result = jitc_var_new_node_1(info.backend, VarKind::Round, info.type,
                                     info.size, info.symbolic, a0, v0);

    jitc_trace("jit_var_round(r%u <- r%u)", result, a0);
    return result;
}

// --------------------------------------------------------------------------

template <typename T, enable_if_t<drjit::detail::is_floating_point<T>::value> = 0>
T eval_trunc(T value) { return T(std::trunc(value)); }

template <typename T, enable_if_t<!drjit::detail::is_floating_point<T>::value> = 0>
T eval_trunc(T) { jitc_fail("eval_trunc(): unsupported operands!"); }

uint32_t jitc_var_trunc(uint32_t a0) {
    auto [info, v0] = jitc_var_check<IsFloat>("jit_var_trunc", a0);

    uint32_t result = 0;
    if (info.simplify && info.literal)
        result = jitc_eval_literal(info, [](auto l0) { return eval_trunc(l0); }, v0);

    if (!result && info.size)
        result = jitc_var_new_node_1(info.backend, VarKind::Trunc, info.type,
                                     info.size, info.symbolic, a0, v0);

    jitc_trace("jit_var_trunc(r%u <- r%u)", result, a0);
    return result;
}

// --------------------------------------------------------------------------

uint32_t jitc_var_eq(uint32_t a0, uint32_t a1) {
    auto [info, v0, v1] = jitc_var_check<IsNotVoid>("jit_var_eq", a0, a1);
    info.type = VarType::Bool;

    uint32_t result = 0;
    if (info.simplify) {
        if (info.literal)
            result = jitc_eval_literal(
                info, [](auto l0, auto l1) { return l0 == l1; }, v0, v1);
        else if (a0 == a1 && !jitc_is_float(v0))
            result = jitc_make_true(info);
    }

    if (!result && info.size)
        result = jitc_var_new_node_2(info.backend, VarKind::Eq, info.type,
                                     info.size, info.symbolic, a0, v0, a1, v1);

    jitc_trace("jit_var_eq(r%u <- r%u, r%u)", result, a0, a1);
    return result;
}

// --------------------------------------------------------------------------

uint32_t jitc_var_neq(uint32_t a0, uint32_t a1) {
    auto [info, v0, v1] = jitc_var_check<IsNotVoid>("jit_var_neq", a0, a1);
    info.type = VarType::Bool;

    uint32_t result = 0;
    if (info.simplify) {
        if (info.literal)
            result = jitc_eval_literal(
                info, [](auto l0, auto l1) { return l0 != l1; }, v0, v1);
        else if (a0 == a1)
            result = jitc_make_zero(info);
    }

    if (!result && info.size)
        result = jitc_var_new_node_2(info.backend, VarKind::Neq, info.type,
                                     info.size, info.symbolic, a0, v0, a1, v1);

    jitc_trace("jit_var_neq(r%u <- r%u, r%u)", result, a0, a1);
    return result;
}

// --------------------------------------------------------------------------

uint32_t jitc_var_lt(uint32_t a0, uint32_t a1) {
    auto [info, v0, v1] = jitc_var_check<IsArithmetic>("jit_var_lt", a0, a1);
    info.type = VarType::Bool;

    uint32_t result = 0;
    if (info.simplify) {
        if (info.literal)
            result = jitc_eval_literal(
                info, [](auto l0, auto l1) { return l0 < l1; }, v0, v1);
        else if (a0 == a1)
            result = jitc_make_zero(info);
    }

    if (!result && info.size)
        result = jitc_var_new_node_2(info.backend, VarKind::Lt, info.type,
                                     info.size, info.symbolic, a0, v0, a1, v1);

    jitc_trace("jit_var_lt(r%u <- r%u, r%u)", result, a0, a1);
    return result;
}

// --------------------------------------------------------------------------

uint32_t jitc_var_le(uint32_t a0, uint32_t a1) {
    auto [info, v0, v1] = jitc_var_check<IsArithmetic>("jit_var_le", a0, a1);
    info.type = VarType::Bool;

    uint32_t result = 0;
    if (info.simplify) {
        if (info.literal)
            result = jitc_eval_literal(
                info, [](auto l0, auto l1) { return l0 <= l1; }, v0, v1);
        else if (a0 == a1 && !jitc_is_float(v0))
            result = jitc_make_true(info);
    }

    if (!result && info.size)
        result = jitc_var_new_node_2(info.backend, VarKind::Le, info.type,
                                     info.size, info.symbolic, a0, v0, a1, v1);

    jitc_trace("jit_var_le(r%u <- r%u, r%u)", result, a0, a1);
    return result;
}

// --------------------------------------------------------------------------

uint32_t jitc_var_gt(uint32_t a0, uint32_t a1) {
    auto [info, v0, v1] = jitc_var_check<IsArithmetic>("jit_var_gt", a0, a1);
    info.type = VarType::Bool;

    uint32_t result = 0;
    if (info.simplify) {
        if (info.literal)
            result = jitc_eval_literal(
                info, [](auto l0, auto l1) { return l0 > l1; }, v0, v1);
        else if (a0 == a1)
            result = jitc_make_zero(info);
    }

    if (!result && info.size)
        result = jitc_var_new_node_2(info.backend, VarKind::Gt, info.type,
                                     info.size, info.symbolic, a0, v0, a1, v1);

    jitc_trace("jit_var_gt(r%u <- r%u, r%u)", result, a0, a1);
    return result;
}

// --------------------------------------------------------------------------

uint32_t jitc_var_ge(uint32_t a0, uint32_t a1) {
    auto [info, v0, v1] = jitc_var_check<IsArithmetic>("jit_var_ge", a0, a1);
    info.type = VarType::Bool;

    uint32_t result = 0;
    if (info.simplify) {
        if (info.literal)
            result = jitc_eval_literal(
                info, [](auto l0, auto l1) { return l0 >= l1; }, v0, v1);
        else if (a0 == a1 && !jitc_is_float(v0))
            result = jitc_make_true(info);
    }

    if (!result && info.size)
        result = jitc_var_new_node_2(info.backend, VarKind::Ge, info.type,
                                     info.size, info.symbolic, a0, v0, a1, v1);

    jitc_trace("jit_var_ge(r%u <- r%u, r%u)", result, a0, a1);
    return result;
}

// --------------------------------------------------------------------------

uint32_t jitc_var_select(uint32_t a0, uint32_t a1, uint32_t a2) {
    auto [info, v0, v1, v2] = jitc_var_check<ArrayAllowed | IgnoreTypes>("jit_var_select", a0, a1, a2);

    if (info.size &&
        (!jitc_is_bool(v0) || v1->type != v2->type ||
         (v1->is_array() != v2->is_array())))
        jitc_raise("jitc_var_select(): invalid operands!");

    info.type = (VarType) v1->type;

    uint32_t result = 0;
    if (info.simplify || a1 == a2) {
        if (jitc_is_one(v0) || a1 == a2)
            return jitc_var_resize(a1, info.size);
        else if (jitc_is_zero(v0))
            return jitc_var_resize(a2, info.size);
    }

    if (!result && info.size) {
        if (!v1->is_array()) {
            result = jitc_var_new_node_3(info.backend, VarKind::Select,
                                         info.type, info.size, info.symbolic,
                                         a0, v0, a1, v1, a2, v2);
        } else {
            uint32_t array_length = v1->array_length;
            if (v2->array_length != array_length || v0->is_array())
                jitc_raise("jitc_var_select(): invalid operands!");

            Ref a3 = steal(jitc_array_create(info.backend, info.type, info.size, array_length));

            v0 = jitc_var(a0);
            v1 = jitc_var(a1);
            v2 = jitc_var(a2);
            Variable *v3 = jitc_var(a3);

            result = jitc_var_new_node_4(info.backend, VarKind::ArraySelect,
                                         info.type, info.size, info.symbolic,
                                         a0, v0, a1, v1, a2, v2, a3, v3);

            Variable *v = jitc_var(result);
            jitc_lvn_drop(result, v);

            v->array_length = array_length;
            v->array_state = (uint32_t) ArrayState::Clean;
        }
    }

    jitc_trace("jit_var_select(r%u <- r%u, r%u, r%u)", result, a0, a1, a2);
    return result;
}

// --------------------------------------------------------------------------

template <typename T, enable_if_t<!std::is_integral_v<T> || std::is_same_v<T, bool>> = 0>
T eval_popc(T) { jitc_fail("eval_popc(): unsupported operands!"); }

template <typename T, enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>> = 0>
T eval_popc(T value_) {
    auto value = (std::make_unsigned_t<T>) value_;
    T result = 0;

    while (value) {
        result += value & 1;
        value >>= 1;
    }

    return result;
}

uint32_t jitc_var_popc(uint32_t a0) {
    auto [info, v0] = jitc_var_check<IsInt>("jit_var_popc", a0);

    uint32_t result = 0;
    if (info.simplify && info.literal)
        result = jitc_eval_literal(info, [](auto l0) { return eval_popc(l0); }, v0);

    if (!result && info.size)
        result = jitc_var_new_node_1(info.backend, VarKind::Popc, info.type,
                                     info.size, info.symbolic, a0, v0);

    jitc_trace("jit_var_popc(r%u <- r%u)", result, a0);
    return result;
}

// --------------------------------------------------------------------------

template <typename T, enable_if_t<!std::is_integral_v<T> || std::is_same_v<T, bool>> = 0>
T eval_clz(T) { jitc_fail("eval_clz(): unsupported operands!"); }

template <typename T, enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>> = 0>
T eval_clz(T value_) {
    auto value = (std::make_unsigned_t<T>) value_;
    T result = sizeof(T) * 8;
    while (value) {
        value >>= 1;
        result -= 1;
    }
    return result;
}

uint32_t jitc_var_clz(uint32_t a0) {
    auto [info, v0] = jitc_var_check<IsInt>("jit_var_clz", a0);

    uint32_t result = 0;
    if (info.simplify && info.literal)
        result = jitc_eval_literal(info, [](auto l0) { return eval_clz(l0); }, v0);

    if (!result && info.size)
        result = jitc_var_new_node_1(info.backend, VarKind::Clz, info.type,
                                     info.size, info.symbolic, a0, v0);

    jitc_trace("jit_var_clz(r%u <- r%u)", result, a0);
    return result;
}

// --------------------------------------------------------------------------

template <typename T, enable_if_t<!std::is_integral_v<T> || std::is_same_v<T, bool>> = 0>
T eval_brev(T) { jitc_fail("eval_brev(): unsupported operands!"); }

template <typename T, enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>> = 0>
T eval_brev(T value_) {
    using Tu = std::make_unsigned_t<T>;
    Tu value = (Tu) value_;
    Tu result = 0;

    for (size_t i = 0; i < sizeof(Tu) * 8; ++i) {
        result = (result << 1) | (value & 1);
        value >>= 1;
    }

    return result;
}

uint32_t jitc_var_brev(uint32_t a0) {
    auto [info, v0] = jitc_var_check<IsInt>("jit_var_brev", a0);

    uint32_t result = 0;
    if (info.simplify && info.literal)
        result = jitc_eval_literal(info, [](auto l0) { return eval_brev(l0); }, v0);

    if (!result && info.size)
        result = jitc_var_new_node_1(info.backend, VarKind::Brev, info.type,
                                     info.size, info.symbolic, a0, v0);

    jitc_trace("jit_var_brev(r%u <- r%u)", result, a0);
    return result;
}

// --------------------------------------------------------------------------

template <typename T, enable_if_t<!std::is_integral_v<T> || std::is_same_v<T, bool>> = 0>
T eval_ctz(T) { jitc_fail("eval_ctz(): unsupported operands!"); }

template <typename T, enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>> = 0>
T eval_ctz(T value) {
    using U = uint_with_size_t<T>;
    U u_value = memcpy_cast<U>(value);

    T result = sizeof(T) * 8;
    while (u_value) {
        u_value <<= 1;
        result -= 1;
    }
    return result;
}

uint32_t jitc_var_ctz(uint32_t a0) {
    auto [info, v0] = jitc_var_check<IsInt>("jit_var_ctz", a0);

    uint32_t result = 0;
    if (info.simplify && info.literal)
        result = jitc_eval_literal(info, [](auto l0) { return eval_ctz(l0); }, v0);

    if (!result && info.size)
        result = jitc_var_new_node_1(info.backend, VarKind::Ctz, info.type,
                                     info.size, info.symbolic, a0, v0);

    jitc_trace("jit_var_ctz(r%u <- r%u)", result, a0);
    return result;
}

// --------------------------------------------------------------------------

template <typename T> T eval_and(T v0, T v1) {
    using U = uint_with_size_t<T>;
    return memcpy_cast<T>(U(memcpy_cast<U>(v0) & memcpy_cast<U>(v1)));
}

static bool eval_and(bool v0, bool v1) { return v0 && v1; }

uint32_t jitc_var_and(uint32_t a0, uint32_t a1) {
    auto [info, v0, v1] = jitc_var_check("jit_var_and", a0, a1);

    if (info.size && v0->type != v1->type && !jitc_is_bool(v1))
        jitc_raise("jitc_var_and(): invalid operands!");

    uint32_t result = 0;
    if (info.simplify) {
        if (info.literal && v0->type == v1->type)
            result = jitc_eval_literal(
                info, [](auto l0, auto l1) { return eval_and(l0, l1); }, v0, v1);
        else if (jitc_is_zero(v0) || jitc_is_zero(v1))
            result = jitc_make_zero(info);
        else if ((jitc_is_one(v1) && jitc_is_bool(v1)) || a0 == a1)
            result = jitc_var_resize(a0, info.size);
        else if (jitc_is_one(v0) && jitc_is_bool(v0))
            result = jitc_var_resize(a1, info.size);
    }

    if (!result && info.size)
        result = jitc_var_new_node_2(info.backend, VarKind::And, info.type,
                                     info.size, info.symbolic, a0, v0, a1, v1);

    jitc_trace("jit_var_and(r%u <- r%u, r%u)", result, a0, a1);
    return result;
}

// --------------------------------------------------------------------------

template <typename T> T eval_or(T v0, T v1) {
    using U = uint_with_size_t<T>;
    return memcpy_cast<T>(U(memcpy_cast<U>(v0) | memcpy_cast<U>(v1)));
}

static bool eval_or(bool v0, bool v1) { return v0 || v1; }

uint32_t jitc_var_or(uint32_t a0, uint32_t a1) {
    auto [info, v0, v1] = jitc_var_check("jit_var_or", a0, a1);

    if (info.size && v0->type != v1->type && !jitc_is_bool(v1))
        jitc_raise("jitc_var_or(): invalid operands!");

    uint32_t result = 0;
    if (info.simplify) {
        if (info.literal && v0->type == v1->type)
            result = jitc_eval_literal(
                info, [](auto l0, auto l1) { return eval_or(l0, l1); }, v0, v1);
        else if ((jitc_is_bool(v0) && jitc_is_one(v1)) ||
                 (jitc_is_zero(v0) && v0->type == v1->type) || a0 == a1)
            result = jitc_var_resize(a1, info.size);
        else if ((jitc_is_bool(v0) && jitc_is_one(v0)) || jitc_is_zero(v1))
            result = jitc_var_resize(a0, info.size);
    }

    if (!result && info.size)
        result = jitc_var_new_node_2(info.backend, VarKind::Or, info.type,
                                     info.size, info.symbolic, a0, v0, a1, v1);

    jitc_trace("jit_var_or(r%u <- r%u, r%u)", result, a0, a1);
    return result;
}

// --------------------------------------------------------------------------

template <typename T> T eval_xor(T v0, T v1) {
    using U = uint_with_size_t<T>;
    return memcpy_cast<T>(U(memcpy_cast<U>(v0) ^ memcpy_cast<U>(v1)));
}

static bool eval_xor(bool v0, bool v1) { return v0 != v1; }

uint32_t jitc_var_xor(uint32_t a0, uint32_t a1) {
    auto [info, v0, v1] = jitc_var_check<IsNotVoid>("jit_var_xor", a0, a1);

    uint32_t result = 0;
    if (info.simplify) {
        if (info.literal)
            result = jitc_eval_literal(
                info, [](auto l0, auto l1) { return eval_xor(l0, l1); }, v0, v1);
        else if (jitc_is_zero(v1))
            result = jitc_var_resize(a0, info.size);
        else if (jitc_is_zero(v0))
            result = jitc_var_resize(a1, info.size);
        else if (a0 == a1)
            result = jitc_make_zero(info);
    }

    if (!result && info.size)
        result = jitc_var_new_node_2(info.backend, VarKind::Xor, info.type,
                                     info.size, info.symbolic, a0, v0, a1, v1);

    jitc_trace("jit_var_xor(r%u <- r%u, r%u)", result, a0, a1);
    return result;
}

// --------------------------------------------------------------------------

template <typename T, enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>> = 0>
T eval_shl(T v0, T v1) {
    using UInt = std::make_unsigned_t<T>;
    return (T) (((UInt) v0) << ((UInt) v1));
}

template <typename T, enable_if_t<!std::is_integral_v<T> || std::is_same_v<T, bool>> = 0>
T eval_shl(T, T) { jitc_fail("eval_shl(): unsupported operands!"); }

uint32_t jitc_var_shl(uint32_t a0, uint32_t a1) {
    auto [info, v0, v1] = jitc_var_check<IsInt>("jit_var_shl", a0, a1);

    uint32_t result = 0;
    if (info.simplify) {
        if (info.literal)
            result = jitc_eval_literal(
                info, [](auto l0, auto l1) { return eval_shl(l0, l1); }, v0, v1);
        else if (jitc_is_zero(v0) || jitc_is_zero(v1))
            result = jitc_var_resize(a0, info.size);
    }

    if (!result && info.size)
        result = jitc_var_new_node_2(info.backend, VarKind::Shl, info.type,
                                     info.size, info.symbolic, a0, v0, a1, v1);

    jitc_trace("jit_var_shl(r%u <- r%u, r%u)", result, a0, a1);
    return result;
}

// --------------------------------------------------------------------------

template <typename T, enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>> = 0>
T eval_shr(T v0, T v1) { return v0 >> v1; }

template <typename T, enable_if_t<!std::is_integral_v<T> || std::is_same_v<T, bool>> = 0>
T eval_shr(T, T) { jitc_fail("eval_shr(): unsupported operands!"); }

uint32_t jitc_var_shr(uint32_t a0, uint32_t a1) {
    auto [info, v0, v1] = jitc_var_check<IsInt>("jit_var_shr", a0, a1);

    uint32_t result = 0;
    if (info.simplify) {
        if (info.literal)
            result = jitc_eval_literal(
                info, [](auto l0, auto l1) { return eval_shr(l0, l1); }, v0, v1);
        else if (jitc_is_zero(v0) || jitc_is_zero(v1))
            result = jitc_var_resize(a0, info.size);
    }

    if (!result && info.size)
        result = jitc_var_new_node_2(info.backend, VarKind::Shr, info.type,
                                     info.size, info.symbolic, a0, v0, a1, v1);

    jitc_trace("jit_var_shr(r%u <- r%u, r%u)", result, a0, a1);
    return result;
}

// --------------------------------------------------------------------------

template <typename T, enable_if_t<drjit::detail::is_floating_point_v<T>> = 0>
T eval_rcp(T value) { return T(1 / value); }

template <typename T, enable_if_t<!drjit::detail::is_floating_point_v<T>> = 0>
T eval_rcp(T) { jitc_fail("eval_rcp(): unsupported operands!"); }

uint32_t jitc_var_rcp(uint32_t a0) {
    auto [info, v0] = jitc_var_check<IsFloat>("jit_var_rcp", a0);

    uint32_t result = 0;
    if (info.simplify && info.literal)
        result = jitc_eval_literal(info, [](auto l0) { return eval_rcp(l0); }, v0);

    if (!result && info.backend == JitBackend::LLVM) {
        drjit::half h1(1.f); float f1 = 1.f; double d1 = 1.0;
        const void* num_ptr = nullptr;
        switch(info.type) {
            case VarType::Float16: num_ptr = &h1; break;
            case VarType::Float32: num_ptr = &f1; break;
            case VarType::Float64: num_ptr = &d1; break;
            default: jitc_fail("jitc_var_rcp(): Invalid variable type");
        }
        uint32_t one = jitc_var_literal(info.backend, info.type, num_ptr, 1, 0);
        result = jitc_var_div(one, a0);
        jitc_var_dec_ref(one);
    }

    if (!result && info.size) {
        bool fast_math = jit_flags() & (uint32_t) JitFlag::FastMath;
        result = jitc_var_new_node_1(
            info.backend, fast_math ? VarKind::RcpApprox : VarKind::Rcp,
            info.type, info.size, info.symbolic, a0, v0);
    }

    jitc_trace("jit_var_rcp(r%u <- r%u)", result, a0);
    return result;
}

// --------------------------------------------------------------------------

template <typename T, enable_if_t<drjit::detail::is_floating_point<T>::value> = 0>
T eval_rsqrt(T value) { return T(1 / std::sqrt(value)); }

template <typename T, enable_if_t<!drjit::detail::is_floating_point<T>::value> = 0>
T eval_rsqrt(T) { jitc_fail("eval_rsqrt(): unsupported operands!"); }

uint32_t jitc_var_rsqrt(uint32_t a0) {
    auto [info, v0] = jitc_var_check<IsFloat>("jit_var_rsqrt", a0);

    uint32_t result = 0;
    if (info.simplify && info.literal)
        result = jitc_eval_literal(info, [](auto l0) { return eval_rsqrt(l0); }, v0);

    bool fast_math = jit_flags() & (uint32_t) JitFlag::FastMath;

    if (!result && info.backend == JitBackend::CUDA &&
        info.type == VarType::Float32 && fast_math) {
        result =
            jitc_var_new_node_1(info.backend, VarKind::RSqrtApprox, info.type,
                                info.size, info.symbolic, a0, v0);
    }

    if (!result && info.size) {
        // Reciprocal, then square root (lower error than the other way around)
        uint32_t rcp = jitc_var_rcp(a0);
        result = jitc_var_sqrt(rcp);
        jitc_var_dec_ref(rcp);
    }

    jitc_trace("jit_var_rsqrt(r%u <- r%u)", result, a0);
    return result;
}

// --------------------------------------------------------------------------

template <typename T, enable_if_t<drjit::detail::is_floating_point_v<T>> = 0>
T eval_sin(T value) { return T(std::sin(value)); }

template <typename T, enable_if_t<!drjit::detail::is_floating_point_v<T>> = 0>
T eval_sin(T) { jitc_fail("eval_sin(): unsupported operands!"); }

uint32_t jitc_var_sin_intrinsic(uint32_t a0) {
    auto [info, v0] = jitc_var_check<IsFloat | IsCUDA>("jit_var_sin", a0);

    uint32_t result = 0;
    if (info.simplify && info.literal)
        result = jitc_eval_literal(info, [](auto l0) { return eval_sin(l0); }, v0);

    if (!result && info.size)
        result = jitc_var_new_node_1(info.backend, VarKind::Sin, info.type,
                                     info.size, info.symbolic, a0, v0);

    jitc_trace("jit_var_sin_intrinsic(r%u <- r%u)", result, a0);
    return result;
}

// --------------------------------------------------------------------------

template <typename T, enable_if_t<drjit::detail::is_floating_point_v<T>> = 0>
T eval_cos(T value) { return T(std::cos(value)); }

template <typename T, enable_if_t<!drjit::detail::is_floating_point_v<T>> = 0>
T eval_cos(T) { jitc_fail("eval_cos(): unsupported operands!"); }

uint32_t jitc_var_cos_intrinsic(uint32_t a0) {
    auto [info, v0] = jitc_var_check<IsFloat | IsCUDA>("jit_var_cos_intrinsic", a0);

    uint32_t result = 0;
    if (info.simplify && info.literal)
        result = jitc_eval_literal(info, [](auto l0) { return eval_cos(l0); }, v0);

    if (!result && info.size)
        result = jitc_var_new_node_1(info.backend, VarKind::Cos, info.type,
                                     info.size, info.symbolic, a0, v0);

    jitc_trace("jit_var_cos_intrinsic(r%u <- r%u)", result, a0);
    return result;
}

// --------------------------------------------------------------------------

template <typename T, enable_if_t<drjit::detail::is_floating_point_v<T>> = 0>
T eval_exp2(T value) { return T(std::exp2(value)); }

template <typename T, enable_if_t<!drjit::detail::is_floating_point_v<T>> = 0>
T eval_exp2(T) { jitc_fail("eval_exp2(): unsupported operands!"); }

uint32_t jitc_var_exp2_intrinsic(uint32_t a0) {
    auto [info, v0] = jitc_var_check<IsFloat | IsCUDA>("jit_var_exp2_intrinsic", a0);

    uint32_t result = 0;
    if (info.simplify && info.literal)
        result = jitc_eval_literal(info, [](auto l0) { return eval_exp2(l0); }, v0);

    if (!result && info.size)
        result = jitc_var_new_node_1(info.backend, VarKind::Exp2, info.type,
                                     info.size, info.symbolic, a0, v0);

    jitc_trace("jit_var_exp2_intrinsic(r%u <- r%u)", result, a0);
    return result;
}

// --------------------------------------------------------------------------

template <typename T, enable_if_t<drjit::detail::is_floating_point_v<T>> = 0>
T eval_log2(T value) { return T(std::log2(value)); }

template <typename T, enable_if_t<!drjit::detail::is_floating_point_v<T>> = 0>
T eval_log2(T) { jitc_fail("eval_log2(): unsupported operands!"); }

uint32_t jitc_var_log2_intrinsic(uint32_t a0) {
    auto [info, v0] = jitc_var_check<IsFloat | IsCUDA>("jit_var_log2_intrinsic", a0);

    uint32_t result = 0;
    if (info.simplify && info.literal)
        result = jitc_eval_literal(info, [](auto l0) { return eval_log2(l0); }, v0);

    if (!result && info.size)
        result = jitc_var_new_node_1(info.backend, VarKind::Log2, info.type,
                                     info.size, info.symbolic, a0, v0);

    jitc_trace("jit_var_log2_intrinsic(r%u <- r%u)", result, a0);
    return result;
}

// --------------------------------------------------------------------------

uint32_t jitc_var_cast(uint32_t a0, VarType target_type, int reinterpret) {
    if (a0 == 0)
        return 0;

    auto [info, v0] = jitc_var_check<IsNotVoid>("jit_var_cast", a0);
    info.type = target_type;

    const VarType source_type = (VarType) v0->type;

    bool source_bool = jitc_is_bool(source_type),
         target_bool = jitc_is_bool(target_type),
         source_float = jitc_is_float(source_type),
         target_float = jitc_is_float(target_type);

    uint32_t source_size = source_bool ? 0 : type_size[(uint32_t) source_type],
             target_size = target_bool ? 0 : type_size[(uint32_t) target_type];

    if (reinterpret && source_size != target_size) {
        jitc_raise("jit_var_cast(): cannot reinterpret-cast between types of "
                   "different size!");
    } else if (source_size == target_size && !source_float && !target_float) {
        reinterpret = 1;
    }

    uint32_t result = 0;
    if (source_type == target_type) {
        result = jitc_var_new_ref(a0);
    } else if (info.simplify && info.literal) {
        if (reinterpret) {
            uint64_t value = v0->literal;
            result = jitc_var_literal(info.backend, info.type, &value, info.size, 0);
        } else {
            result = jitc_eval_literal(info, [target_type](auto value) -> uint64_t {
                switch (target_type) {
                    case VarType::Bool:    return v2i((bool) value);
                    case VarType::Int8:    return v2i((int8_t) value);
                    case VarType::UInt8:   return v2i((uint8_t) value);
                    case VarType::Int16:   return v2i((int16_t) value);
                    case VarType::UInt16:  return v2i((uint16_t) value);
                    case VarType::Int32:   return v2i((int32_t) value);
                    case VarType::UInt32:  return v2i((uint32_t) value);
                    case VarType::Int64:   return v2i((int64_t) value);
                    case VarType::UInt64:  return v2i((uint64_t) value);
                    case VarType::Float16: return v2i((drjit::half) value);
                    case VarType::Float32: return v2i((float) value);
                    case VarType::Float64: return v2i((double) value);
                    default: jitc_fail("jit_var_cast(): unsupported variable type!");
                }
            }, v0);
        }
    }

    if (!result && info.size)
        result = jitc_var_new_node_1(
            info.backend, reinterpret ? VarKind::Bitcast : VarKind::Cast,
            info.type, info.size, info.symbolic, a0, v0);

    jitc_trace("jit_var_cast(r%u <- r%u)", result, a0);
    return result;
}

// --------------------------------------------------------------------------

static uint32_t jitc_scatter_gather_index(uint32_t source, uint32_t index) {
    const Variable *v_source = jitc_var(source),
                   *v_index = jitc_var(index);

    VarType source_type = (VarType) v_index->type;
    if (!jitc_is_int(source_type))
        jitc_raise("jit_scatter_gather_index(): expected an integer array as "
                   "scatter/gather index");

    VarType target_type = VarType::UInt32;
    // Need 64 bit indices for upper 2G entries (gather indices are signed in LLVM)
    if (v_source->size > 0x7fffffff &&
        (JitBackend) v_source->backend == JitBackend::LLVM)
        target_type = VarType::UInt64;

    return jitc_var_cast(index, target_type, 0);
}

/// Change all indices/counters in an expression tree to 'new_index'
static uint32_t jitc_var_reindex(uint32_t var_index, uint32_t new_index,
                                 uint32_t mask, uint32_t size) {
    Variable *v = jitc_var(var_index);

    if (v->is_evaluated() || (VarType) v->type == VarType::Void)
        return 0; // evaluated variable, give up

    if (v->extra) {
        VariableExtra *e = jitc_var_extra(v);
        if (e->callback)
            return 0; // "complicated" variable, give up
    }

    Ref dep[4];
    bool rebuild = v->size != size && v->size != 1;

    if (v->kind == (uint32_t) VarKind::DefaultMask && rebuild)
        // Do not re-index the mask, only resize it
        return jitc_var_mask_default((JitBackend) v->backend, size);

    if (!v->is_literal()) {
        for (uint32_t i = 0; i < 4; ++i) {
            uint32_t index_2 = v->dep[i];
            if (!index_2)
                continue;

            if (v->kind == (uint32_t) VarKind::Gather && i == 2) {
                // Gather nodes must have their masks replaced rather than reindexed
                JitBackend backend = (JitBackend) v->backend;
                Ref default_mask = steal(jitc_var_mask_default(backend, size));
                dep[i] = steal(jitc_var_and(mask, default_mask));
            } else {
                dep[i] = steal(jitc_var_reindex(index_2, new_index, mask, size));
            }

            v = jitc_var(var_index);
            if (!dep[i])
                return 0; // recursive call failed, give up
            rebuild |= dep[i] != index_2;
        }
    }


    if (v->kind == (uint32_t) VarKind::Counter) {
        return jitc_var_new_ref(new_index);
    } else if (rebuild) {
        Variable v2;
        v2.kind = v->kind;
        v2.backend = v->backend;
        v2.type = v->type;
        v2.size = size;
        v2.optix = v->optix;
        v2.symbolic = v->symbolic;
        v2.literal = v->literal;
        for (uint32_t i = 0; i < 4; ++i) {
            v2.dep[i] = dep[i];
            jitc_var_inc_ref(dep[i]);
        }
        return jitc_var_new(v2);
    } else {
        jitc_var_inc_ref(var_index, v);
        return var_index;
    }
}

uint32_t jitc_var_check_bounds(BoundsCheckType bct, uint32_t index,
                               uint32_t mask, uint32_t array_size) {
    auto [info, v_index, v_mask] =
        jitc_var_check("jit_var_check_bounds", index, mask);

    uint64_t zero = 0;
    Ref buf = steal(jitc_var_literal(info.backend, VarType::UInt32, &zero, 1, 1)),
        buffer_ptr =
            steal(jitc_var_pointer(info.backend, jitc_var(buf)->data, buf, 1));

    Ref result = steal(jitc_var_new_node_3(
        info.backend, VarKind::BoundsCheck, VarType::Bool, info.size,
        info.symbolic, index, jitc_var(index), mask, jitc_var(mask), buffer_ptr,
        jitc_var(buffer_ptr), array_size | (((uint64_t) bct) << 32)));

    jitc_var_set_callback(result,
         [](uint32_t index, int free, void *) {
             if (free)
                 return;

            Variable *v = jitc_var(index);
            uint32_t captured = 0;
            jitc_memcpy((JitBackend) v->backend, &captured,
                        jitc_var(v->dep[2])->data, sizeof(uint32_t));
            v = jitc_var(index);

            const char *label = jitc_var_label(index);
            if (captured) {
                const char *msg = nullptr;
                const char *msg2 = " in an array of size";
                uint32_t size = (uint32_t) v->literal;

                switch ((BoundsCheckType) (v->literal >> 32)) {
                    case BoundsCheckType::Gather:
                        msg = "drjit.gather(): out-of-bounds read from position";
                        break;

                    case BoundsCheckType::PacketGather:
                        msg = "drjit.gather(): out-of-bounds packet read from position";
                        break;

                    case BoundsCheckType::Scatter:
                        msg = "drjit.scatter(): out-of-bounds write to position";
                        break;

                    case BoundsCheckType::PacketScatter:
                        msg = "drjit.scatter(): out-of-bounds packet write to position";
                        break;

                    case BoundsCheckType::ScatterReduce:
                        msg = "drjit.scatter_reduce(): out-of-bounds write to position";
                        break;

                    case BoundsCheckType::PacketScatterReduce:
                        msg = "drjit.scatter_reduce(): out-of-bounds packet write to position";
                        break;

                    case BoundsCheckType::ScatterInc:
                        msg = "drjit.scatter_inc(): out-of-bounds write to position";
                        break;

                    case BoundsCheckType::ScatterAddKahan:
                        msg = "drjit.scatter_add_kahan(): out-of-bounds write to position";
                        break;

                    case BoundsCheckType::ArrayRead:
                        msg = "drjit.Local.read(): out-of-bounds read from position";
                        break;

                    case BoundsCheckType::ArrayWrite:
                        msg = "drjit.Local.write(): out-of-bounds write to position";
                        break;

                    case BoundsCheckType::Call:
                        msg = "Attempted to invoke callable with index";
                        msg2 = ", but this value must be strictly smaller than";
                        captured--;
                        size--;
                        break;

                    default:
                        jit_fail("jitc_var_check_bounds(): unhandled case!");
                }

                jitc_log(Warn, "%s %u%s %u. %s%s%s", msg, captured, msg2,
                         (uint32_t) size, label ? "(" : "", label ? label : "",
                         label ? ")" : "");
            }
        },
        nullptr, true);

    return result.release();
}

static void unwrap(Ref &index, Variable *&v) {
    while (true) {
        if (v->kind == (uint32_t) VarKind::LoopPhi) {
            index = borrow(v->dep[3]);
            v = jitc_var(index);
        } else {
            break;
        }
    }
}

uint32_t jitc_var_gather(uint32_t src_, uint32_t index, uint32_t mask) {
    if (index == 0)
        return 0;

    Ref src = borrow(src_);

    auto [src_info, src_v] =
        jitc_var_check("jit_var_gather", src_);
    auto [var_info, index_v, mask_v] =
        jitc_var_check("jit_var_gather", index, mask);

    // Go to the original if 'src' is wrapped into a loop state variable
    unwrap(src, src_v);

    uint32_t result = 0, ptr = 0;
    const char *msg = "";

    {
        /// Variables with _v subscript only inspected in this scope
        if (src_v->symbolic)
            jitc_raise("jit_var_gather(): cannot gather from a symbolic "
                       "variable (r%u, kind=%s)!",
                       src_, var_kind_name[(int) src_v->kind]);

        if (mask_v->is_literal() && mask_v->literal == 0) {
            var_info.type = src_info.type;
            result = jitc_make_zero(var_info);
            msg = " [elided, always masked]";
        }

        /// Make sure that the src and index variables doesn't have pending side effects
        if (!result && unlikely(index_v->is_dirty() || src_v->is_dirty())) {
            jitc_eval(thread_state(src_info.backend));
            if (jitc_var(index)->is_dirty())
                jitc_raise_dirty_error(index);
            if (jitc_var(src)->is_dirty())
                jitc_raise_dirty_error(src);

            src_v = jitc_var(src);
            index_v = jitc_var(index);
        }
    }

    // If the source is scalar we can just resize it
    if (!result && src_v->size == 1) {
        // Temporarily hold an extra reference to prevent 'jitc_var_resize' from changing 'src'
        Ref unused = borrow(src);
        Ref tmp = steal(jitc_var_resize(src, var_info.size));
        result = jitc_var_and(tmp, mask);
        msg = " [elided, scalar source]";
    }

    // Don't perform the gather operation if the inputs are trivial / can be re-indexed
    if (!result) {
        Ref index_2 = steal(jitc_var_cast(index, VarType::UInt32, 0));
        Ref src_reindexed = steal(jitc_var_reindex(src, index_2, mask, var_info.size));
        if (src_reindexed) {
            // Temporarily hold an extra reference to prevent 'jitc_var_resize' from changing 'src'
            Ref unused = borrow(src_reindexed);
            Ref tmp = steal(jitc_var_resize(src_reindexed, var_info.size));
            result = jitc_var_and(tmp, mask);
            msg = " [elided, reindexed]";
        }
    }

    // At this point, we *will* have to evalute the source, if not done already.
    if (!result)
        jitc_var_eval(src);

    /// Perform a memcpy when this is a size-1 literal load
    if (!result && var_info.size == 1 && var_info.literal) {
        size_t size = type_size[(int) src_info.type];
        size_t pos = (size_t) jitc_var(index)->literal;

        if (pos >= src_info.size)
            jitc_raise("jit_var_gather(): out-of-bounds read from position %zu "
                       "in an array of size %u.", pos, src_info.size);

        AllocType atype = (JitBackend) src_info.backend == JitBackend::CUDA
                              ? AllocType::Device
                              : AllocType::HostAsync;

        void *p = (uint8_t *) jitc_var(src)->data + size * pos,
             *p_out = jitc_malloc(atype, size);

        jitc_memcpy_async(src_info.backend, p_out, p, size);

        result = jitc_var_mem_map(src_info.backend, src_info.type, p_out, 1, 1);
        msg = " [elided, memcpy]";
    }

    // If 'result' is not set, none of the special cases above were triggered,
    // and we will need to either generate a Gather IR node.
    if (!result) {
        Ref mask_2 = steal(jitc_var_mask_apply(mask, var_info.size)),
            ptr_2  = steal(jitc_var_pointer(src_info.backend, jitc_var(src)->data, src, 0));

        ptr = (uint32_t) ptr_2;

        Ref index_2 = steal(jitc_scatter_gather_index(src, index));

        if (jit_flag(JitFlag::Debug))
            mask_2 = steal(jitc_var_check_bounds(
                BoundsCheckType::Gather, index, mask_2, src_info.size));

        var_info.size = std::max(var_info.size, jitc_var(mask_2)->size);

        result = jitc_var_new_node_3(
            src_info.backend, VarKind::Gather, src_info.type, var_info.size,
            var_info.symbolic, ptr_2, jitc_var(ptr_2), index_2,
            jitc_var(index_2), mask_2, jitc_var(mask_2));
    }

    jitc_log(Debug,
             "jit_var_gather(): %s r%u[%u] = r%u[r%u] (mask=r%u, ptr=r%u)%s",
             type_name[(int) src_info.type], result, var_info.size,
             (uint32_t) src, index, mask, ptr, msg);

    return result;
}

void jitc_var_gather_packet(size_t n, uint32_t src_, uint32_t index, uint32_t mask, uint32_t *out) {
    if (index == 0) {
        for (size_t i = 0; i < n; ++i)
            out[i] = 0;
        return;
    }
    Ref scale = steal(jitc_var_u32((JitBackend) jitc_var(src_)->backend, n));

    Ref src = borrow(src_);

    auto [src_info, src_v] =
        jitc_var_check("jit_var_gather_packet", src_);
    auto [var_info, index_v, mask_v] =
        jitc_var_check("jit_var_gather_packet", index, mask);

    if ((n & (n-1)) || n == 1)
        jitc_raise("jitc_var_gather_packet(): vector size must be a power of two "
                   "and >= 1 (got %zu)!", n);

    if ((src_info.size & (n-1)) != 0 && src_info.size != 1)
        jitc_raise("jitc_var_gather_packet(): source r%u has size %u, which is not "
                   "divisible by %zu!", index, src_info.size, n);

    // Go to the original if 'src' is wrapped into a loop state variable
    unwrap(src, src_v);

    uint32_t flags = jitc_flags();

    /// Revert to separate gathers in special various cases
    if (src_v->symbolic || // This will likely fail, let jitc_var_gather() generate an error
        !(flags & (uint32_t) JitFlag::PacketOps) ||      // Packet gathers are disabled
        (mask_v->is_literal() && mask_v->literal == 0) ||   // Masked load
        src_v->size == 1 ||                                 // Scalar load
        (var_info.size == 1 && var_info.literal) ||         // Memcpy
        src_v->unaligned) {                                 // Source must be aligned
        for (size_t i = 0; i < n; ++i) {
            Ref index2 = steal(jitc_var_u32(var_info.backend, (uint32_t) i));
            Ref index3 = steal(jitc_var_fma(index, scale, index2));
            out[i] = jitc_var_gather(src_, index3, mask);
        }
        return;
    }

    // Packet size 8 is the max. for the LLVM backend. Split larger requests.
    uint32_t max_width = std::min(8u, jitc_llvm_vector_width);
    if (n > max_width && var_info.backend == JitBackend::LLVM) {
        Ref step = steal(jitc_var_u32(var_info.backend, 1)),
            scale = steal(jitc_var_u32(var_info.backend, n/max_width)),
            index2 = steal(jitc_var_mul(index, scale));

        for (size_t i = 0; i < n; i += max_width) {
            jitc_var_gather_packet(max_width, src_, index2, mask, out+i);
            index2 = steal(jitc_var_add(index2, step));
        }
        return;
    }

    /// Make sure that the src and index variables doesn't have pending side effects
    if (unlikely(index_v->is_dirty() || src_v->is_dirty())) {
        jitc_eval(thread_state(src_info.backend));
        if (jitc_var(index)->is_dirty())
            jitc_raise_dirty_error(index);
        if (jitc_var(src)->is_dirty())
            jitc_raise_dirty_error(src);

        src_v = jitc_var(src);
        index_v = jitc_var(index);
    }

    // At this point, we *will* have to evalute the source, if not done already.
    jitc_var_eval(src);

    Ref mask_2 = steal(jitc_var_mask_apply(mask, var_info.size)),
        ptr_2  = steal(jitc_var_pointer(src_info.backend, jitc_var(src)->data, src, 0));

    Ref index_2 = steal(jitc_scatter_gather_index(src, index));

    if (jit_flag(JitFlag::Debug)) {
        Ref scaled_index = steal(jitc_var_mul(scale, index));
        mask_2 = steal(jitc_var_check_bounds(
            BoundsCheckType::PacketGather, scaled_index, mask_2, src_info.size));
    }

    size_t op_size = std::max(var_info.size, jitc_var(mask_2)->size);

    Ref gather_op = steal(jitc_var_new_node_3(
        src_info.backend, VarKind::PacketGather, src_info.type,
        op_size, var_info.symbolic, ptr_2,
        jitc_var(ptr_2), index_2, jitc_var(index_2), mask_2,
        jitc_var(mask_2), n));

    for (size_t i = 0; i < n; ++i)
        out[i] = jitc_var_new_node_1(
            src_info.backend, VarKind::Extract, src_info.type, var_info.size,
            var_info.symbolic, gather_op, jitc_var(gather_op), i);

    jitc_log(Debug,
             "jit_var_gather_packet(): %s <base r%u>[%u] = r%u[r%u] "
             "(mask=r%u, ptr=r%u)",
             type_name[(int) src_info.type], (uint32_t) gather_op, var_info.size,
             (uint32_t) src, index, mask, (uint32_t) ptr_2);
}

static const char *reduce_op_symbol[(int) ReduceOp::Count] = {
    "=", "+=", "*=", "= min", "= max", "&=", "|="
};

void jitc_var_scatter_add_kahan(uint32_t *target_1_p, uint32_t *target_2_p,
                                uint32_t value, uint32_t index, uint32_t mask) {
    if (value == 0 && index == 0)
        return;

    auto [var_info, value_v, index_v, mask_v] =
        jitc_var_check("jit_var_scatter_add_kahan", value, index, mask);

    Ref target_1 = borrow(*target_1_p),
        target_2 = borrow(*target_2_p);

    auto [target_info, target_1_v, target_2_v] =
        jitc_var_check("jit_var_scatter_add_kahan",
                       (uint32_t) target_1,
                       (uint32_t) target_2);

    // Go to the original if 'target' is wrapped into a loop state variable
    unwrap(target_1, target_1_v);
    unwrap(target_2, target_2_v);

    if (target_1_v->symbolic || target_2_v->symbolic)
        jitc_raise("jit_var_scatter_add_kahan(): cannot scatter to a symbolic "
                   "variable (r%u, r%u).",
                   (uint32_t) target_1, (uint32_t) target_2);

    if (target_1_v->type != value_v->type || target_2_v->type != value_v->type)
        jitc_raise("jit_var_scatter_add_kahan(): target/value type mismatch.");

    if (target_1_v->size != target_2_v->size)
        jitc_raise("jit_var_scatter_add_kahan(): target size mismatch.");

    if (value_v->is_literal() && value_v->literal == 0)
        return;

    if (mask_v->is_literal() && mask_v->literal == 0)
        return;

    uint32_t flags = jitc_flags();
    var_info.symbolic |= (flags & (uint32_t) JitFlag::SymbolicScope) != 0;

    // Copy-on-Write logic. See the same line in jitc_var_scatter() for details
    if (target_1_v->ref_count != 2 && target_1_v->ref_count_stashed != 1) {
        target_1 = steal(jitc_var_copy(target_1));

        // The above operation may have invalidated 'target_2_v' which is accessed below
        target_2_v = jitc_var(target_2);
    }

    // Copy-on-Write logic. See the same line in jitc_var_scatter() for details
    if ((target_2_v->ref_count != 2 && target_2_v->ref_count_stashed != 1) ||
        target_1 == target_2)
        target_2 = steal(jitc_var_copy(target_2));

    void *target_1_addr = nullptr, *target_2_addr = nullptr;
    target_1 = steal(jitc_var_data(target_1, false, &target_1_addr));
    target_2 = steal(jitc_var_data(target_2, false, &target_2_addr));

    if (target_1 != *target_1_p) {
        jitc_var_inc_ref(target_1);
        jitc_var_dec_ref(*target_1_p);
        *target_1_p = target_1;
    }

    if (target_2 != *target_2_p) {
        jitc_var_inc_ref(target_2);
        jitc_var_dec_ref(*target_2_p);
        *target_2_p = target_2;
    }

    Ref ptr_1 = steal(jitc_var_pointer(var_info.backend, target_1_addr, target_1, 1));
    Ref ptr_2 = steal(jitc_var_pointer(var_info.backend, target_2_addr, target_2, 1));

    Ref mask_2  = steal(jitc_var_mask_apply(mask, var_info.size)),
        index_2 = steal(jitc_scatter_gather_index(target_1, index));

    if (flags & (uint32_t) JitFlag::Debug)
        mask_2 = steal(jitc_var_check_bounds(BoundsCheckType::ScatterAddKahan,
                                             index, mask_2, target_info.size));

    var_info.size = std::max(var_info.size, jitc_var(mask_2)->size);

    Ref value_2 = steal(jitc_var_and(value, mask_2));

    bool symbolic = jit_flag(JitFlag::SymbolicScope);
    if (var_info.symbolic && !symbolic)
        jitc_raise(
            "jit_var_scatter_add_kahan(): input arrays are symbolic, but the "
            "operation was issued outside of a symbolic recording session.");

    uint32_t result = jitc_var_new_node_4(
        var_info.backend, VarKind::ScatterKahan, VarType::Void, var_info.size,
        symbolic, ptr_1, jitc_var(ptr_1), ptr_2, jitc_var(ptr_2), index_2,
        jitc_var(index_2), value_2, jitc_var(value_2));

    jitc_log(Debug,
             "jit_var_scatter_add_kahan(): (r%u[r%u], r%u[r%u]) += r%u "
             "(mask=r%u, ptrs=(r%u, r%u), se=r%u)",
             (uint32_t) target_1, (uint32_t) index_2, (uint32_t) target_2,
             (uint32_t) index_2, (uint32_t) value_2, (uint32_t) mask_2,
             (uint32_t) ptr_1, (uint32_t) ptr_2, result);

    jitc_var_mark_side_effect(result);
}

uint32_t jitc_var_scatter_inc(uint32_t *target_p, uint32_t index, uint32_t mask) {
    auto [var_info, index_v, mask_v] =
        jitc_var_check("jit_var_scatter_inc", index, mask);

    Ref target = borrow(*target_p);
    auto [target_info, target_v] =
        jitc_var_check("jit_var_scatter_inc", (uint32_t) target);

    // Go to the original if 'target' is wrapped into a loop state variable
    unwrap(target, target_v);

    if (target_v->symbolic)
        jitc_raise(
            "jit_var_scatter_inc(): cannot scatter to a symbolic variable (r%u).",
            (uint32_t) target);

    if ((VarType) target_v->type != VarType::UInt32)
        jitc_raise("jit_var_scatter_inc(): 'target' must be an unsigned 32-bit array.");

    if ((VarType) index_v->type != VarType::UInt32)
        jitc_raise("jit_var_scatter_inc(): 'index' must be an unsigned 32-bit array.");

    if (mask_v->is_literal() && mask_v->literal == 0)
        return 0;

    uint32_t flags = jitc_flags();
    var_info.symbolic |= (flags & (uint32_t) JitFlag::SymbolicScope) != 0;

    // Copy-on-Write logic. See the same line in jitc_var_scatter() for details
    if (target_v->ref_count != 2 && target_v->ref_count_stashed != 1)
        target = steal(jitc_var_copy(target));

    void *target_addr = nullptr;
    target = steal(jitc_var_data(target, false, &target_addr));

    if (target != *target_p) {
        jitc_var_inc_ref(target);
        jitc_var_dec_ref(*target_p);
        *target_p = target;
    }

    Ref ptr = steal(jitc_var_pointer(var_info.backend, target_addr, target, 0));

    Ref mask_2  = steal(jitc_var_mask_apply(mask, var_info.size)),
        index_2 = steal(jitc_scatter_gather_index(target, index));

    if (flags & (uint32_t) JitFlag::Debug)
        mask_2 = steal(jitc_var_check_bounds(BoundsCheckType::ScatterInc, index,
                                             mask_2, target_info.size));

    var_info.size = std::max(var_info.size, jitc_var(mask_2)->size);

    bool symbolic = jit_flag(JitFlag::SymbolicScope);
    if (var_info.symbolic && !symbolic)
        jitc_raise(
            "jit_var_scatter_inc(): input arrays are symbolic, but the "
            "operation was issued outside of a symbolic recording session.");

    uint32_t result = jitc_var_new_node_3(
        var_info.backend, VarKind::ScatterInc, VarType::UInt32, var_info.size,
        symbolic, ptr, jitc_var(ptr), index_2, jitc_var(index_2),
        mask_2, jitc_var(mask_2));

    jitc_log(Debug,
             "jit_var_scatter_inc(): r%u[r%u] += 1 (mask=r%u, ptr=r%u, se=r%u)",
             (uint32_t) target, (uint32_t) index_2, (uint32_t) mask_2,
             (uint32_t) ptr, result);

    // Create a dummy node that represents a side effect, and which holds a
    // write pointer to 'target' so that it will be marked dirty
    Ref write_ptr =
        steal(jitc_var_pointer(var_info.backend, target_addr, target, 1));

    Ref se = steal(jitc_var_new_node_1(var_info.backend, VarKind::Nop,
                                       VarType::Void, var_info.size, symbolic,
                                       result, jitc_var(result)));

    jitc_var(se)->dep[3] = write_ptr.release();
    jitc_var_mark_side_effect(se.release());

    return result;
}

extern size_t llvm_expand_threshold;

// Determine whether or not a particular type of reduction is supported
// by the backend. It's quite complicated!
bool jitc_can_scatter_reduce(JitBackend backend, VarType vt, ReduceOp op) {
    if (op == ReduceOp::Identity)
        return true;

    // Bitwise reductions of floating point values not permitted
    if ((vt == VarType::Float16 || vt == VarType::Float32 ||
         vt == VarType::Float64) &&
        (op == ReduceOp::Or || op == ReduceOp::And))
        return false;

    // No multiplication reduction atomics
    if (op == ReduceOp::Mul)
        return false;

    bool is_llvm = backend == JitBackend::LLVM;
    bool is_cuda = backend == JitBackend::CUDA;

    // LLVM prior to v15.0.0 lacks minimum/maximum atomic reduction intrinsics
    if (is_llvm && (op == ReduceOp::Min || op == ReduceOp::Max) &&
        jitc_llvm_version_major < 15)
        return false;

    size_t compute_capability = (size_t) -1;
    if (is_cuda)
        compute_capability = thread_state(backend)->compute_capability;

    switch (vt) {
        case VarType::Bool:
            // Scatter-reductions of masks not implemented
            return false;

        case VarType::Float16:
            // Half-precision sum reduction require sm_60
            if (op == ReduceOp::Add && compute_capability < 60)
                return false;

            // Half-precision min/max reductions require sm_90
            if ((op == ReduceOp::Min || op == ReduceOp::Max) &&
                compute_capability < 90)
                return false;

            // Half precision atomics too spotty on LLVM before v16.0.0
            if (is_llvm && jitc_llvm_version_major < 16)
                return false;

#if defined(__x86_64__)
            // FP16 min/max reduction requires a global offset table on x86_64,
            // which breaks the compilation
            if (is_llvm && (op == ReduceOp::Min || op == ReduceOp::Max))
                return false;
#endif
            break;

        case VarType::Float32:
            if (is_cuda && op != ReduceOp::Add)
                return false;
            break;

        case VarType::Float64:
            // Double precision reductions require sm_60
            if (is_cuda && (op != ReduceOp::Add || compute_capability < 60))
                return false;
            break;

        default:
            break;
    }

    return true;
}

static const char *mode_name[] = { "auto",        "direct", "local",
                                   "no_conflict", "expand", "permute" };

/// Logic related to choosing the 'ReduceMode' of a scatter(-reduction)
static std::pair<ReduceMode, bool>
jitc_var_infer_reduce_mode(const char *name, JitBackend backend, Ref &target,
                           Ref &index, ReduceOp op, ReduceMode mode) {
    bool reduce_expanded = false;

    // Raise an error if the target array is currently stored in an expanded
    // form that isn't compatible with the operation to be performed.
    uint32_t op_cur = jitc_var(target)->reduce_op;
    if (op_cur && op_cur != (uint32_t) op)
        jitc_raise(
            "%s(): it is not legal to mix different types of "
            "scatter-reductions when using dr.ReduceMode.Expand. Evaluate the "
            "array first before using a different kind of reduction.", name);

    if (op != ReduceOp::Identity) {
        if (mode == ReduceMode::Permute)
            jitc_raise("%s(): drjit.ReduceMode.Permute is not a "
                       "valid mode for scatter-reductions.", name);

        if (backend == JitBackend::LLVM) {
            uint32_t tsize = jitc_var(target)->size;
            if (mode == ReduceMode::Auto && tsize <= llvm_expand_threshold)
                mode = ReduceMode::Expand;
        } else {
            // ReduceMode::Expand is only supported on the LLVM backend
            if (mode == ReduceMode::Expand)
                mode = ReduceMode::Auto;
        }

        if (mode == ReduceMode::Auto)
            mode = jit_flag(JitFlag::ScatterReduceLocal)
                       ? ReduceMode::Local
                       : ReduceMode::Direct;

        if (mode == ReduceMode::Expand) {
            index = steal(jitc_var_cast(index, VarType::UInt32, 0));
            const Variable *index_v2 = jitc_var(index);
            uint32_t size = index_v2->size;

            // Track if this is the first time that 'jit_var_expand' is called
            reduce_expanded =
                jitc_var(target)->reduce_op == (uint32_t) ReduceOp::Identity;

            auto [target_i, expand_i] = jitc_var_expand(target, op);
            target = steal(target_i);

            Variable v{};
            v.kind = (uint32_t) VarKind::ThreadIndex;
            v.type = (uint32_t) VarType::UInt32;
            v.size = (uint32_t) size;
            v.backend = (uint32_t) backend;

            Ref expand = steal(jitc_var_u32(backend, expand_i)),
                thread_idx = steal(jitc_var_new(v));

            index = steal(jitc_var_fma(thread_idx, expand, index));

            mode = ReduceMode::NoConflicts;
        }
    } else {
        mode = ReduceMode::Auto;
    }

    return { mode, reduce_expanded };
}

uint32_t jitc_var_scatter(uint32_t target_, uint32_t value, uint32_t index_,
                          uint32_t mask, ReduceOp op, ReduceMode mode) {
    Ref target = borrow(target_), index = borrow(index_), ptr;

    auto print_log = [&](const char *msg, uint32_t result_node = 0) {
        int type_id = 0;
        if (value)
            type_id = jitc_var(value)->type;
        if (result_node)
            jitc_log(Debug,
                     "jit_var_scatter(): r%u[r%u] %s r%u (type=%s, mask=r%u, "
                     "ptr=r%u, se=r%u, out=r%u, mode=%s) [%s]",
                     target_, index_, reduce_op_symbol[(int) op], value,
                     type_name[type_id], mask, (uint32_t) ptr, result_node,
                     (uint32_t) target, mode_name[(int) mode], msg);
        else
            jitc_log(Debug,
                     "jit_var_scatter(): r%u[r%u] %s r%u (type=%s, mask=r%u, "
                     "ptr=r%u, mode=%s) [%s]",
                     target_, index_, reduce_op_symbol[(int) op], value,
                     type_name[type_id], mask, (uint32_t) ptr,
                     mode_name[(int) mode], msg);
    };

    if (value == 0 && index == 0) {
        print_log("empty scatter");
        return target.release();
    }

    if (target == 0)
        jitc_raise("jit_var_scatter(): attempted to scatter to an empty array!");

    auto [var_info, value_v, index_v, mask_v] =
        jitc_var_check("jit_var_scatter", value, index_, mask);

    Variable *target_v = jitc_var(target);
    const uint32_t target_size = target_v->size;
    const JitBackend backend = var_info.backend;
    VarType vt = (VarType) target_v->type;

    // Go to the original if 'target' is wrapped into a loop state variable
    unwrap(target, target_v);
    target_ = target;

    if (!jitc_can_scatter_reduce(backend, vt, op))
        jitc_raise(
            "jit_var_scatter(): the %s backend does not support the requested "
            "type of atomic reduction (%s) for variables of type (%s)",
            backend == JitBackend::CUDA ? "CUDA" : "LLVM", red_name[(int) op],
            type_name[(int) vt]);

    if (target_v->symbolic)
        jitc_raise(
            "jit_var_scatter(): cannot scatter to a symbolic variable (r%u)!",
            (uint32_t) target);

    uint32_t flags = jitc_flags();
    bool symbolic = flags & (uint32_t) JitFlag::SymbolicScope;
    if (var_info.symbolic && !symbolic)
        jitc_raise(
            "jit_var_scatter(): input arrays are symbolic, but the operation "
            "was issued outside of a symbolic recording session.");

    if (target_v->type != value_v->type)
        jitc_raise("jit_var_scatter(): target/value type mismatch!");

    if (target_v->is_literal() && value_v->is_literal() &&
        target_v->literal == value_v->literal && op == ReduceOp::Identity) {
        print_log("skipped, target/source are value variables with the "
                  "same value");
        return target.release();
    }

    if (mask_v->is_literal() && mask_v->literal == 0) {
        print_log("skipped, always masked");
        return target.release();
    }

    if (value_v->is_literal() && value_v->literal == 0 && op == ReduceOp::Add) {
        print_log("skipped, scatter-addition with zero-valued source variable");
        return target.release();
    }

    // The backends may employ various strategies to reduce the number of
    // atomic memory operations, or to avoid them altogether. Check if the user
    // requested this.
    // Warning: this operation might invalidate variable pointers
    bool reduce_expanded = false;
    std::tie(mode, reduce_expanded) = jitc_var_infer_reduce_mode(
        "jit_var_scatter", var_info.backend, target, index, op, mode);

    // Check if it is safe to directly write to ``target``. We borrowed
    // ``target`` above, and the caller also owns one reference. Therefore, the
    // target array can be directly modified when the reference count exactly
    // equals 2. Otherwise, we must make a copy first. If a reference count was
    // stashed via \ref jitc_var_stash_ref() (e.g., prior to a control flow
    // operation like dr.if_stmt()), then use that instead.
    target_v = jitc_var(target);

    if (target_v->is_dirty() && op == ReduceOp::Identity 
                             && mode != ReduceMode::Permute 
                             && mode != ReduceMode::NoConflicts) {
        jitc_var_eval(target);
        target_v = jitc_var(target);
    }

    if (target_v->ref_count > 2 && target_v->ref_count_stashed != 1)
        target = steal(jitc_var_copy(target));

    // Get a pointer to the array data. This evaluates the array if needed
    void *target_addr = nullptr;
    target = steal(jitc_var_data(target, false, &target_addr));
    ptr = steal(jitc_var_pointer(var_info.backend, target_addr, target, 1));

    // Apply default masks
    Ref mask_2 = steal(jitc_var_mask_apply(mask, var_info.size));
    index = steal(jitc_scatter_gather_index(target, index));

    // Insert a bounds check in debug mode
    if (flags & (uint32_t) JitFlag::Debug)
        mask_2 = steal(jitc_var_check_bounds(
            op == ReduceOp::Identity ? BoundsCheckType::Scatter
                                     : BoundsCheckType::ScatterReduce,
            index_ /* original index */, mask_2, target_size));

    // Encode the 'op' and 'mode' variables into variable's literal field
    uint64_t literal = (((uint64_t) mode) << 32) + ((uint64_t) op);

    uint32_t op_size = std::max(var_info.size, jitc_var(mask_2)->size);
    uint32_t scatter_op = jitc_var_new_node_4(
        var_info.backend, VarKind::Scatter, VarType::Void,
        op_size, symbolic, ptr, jitc_var(ptr),
        value, jitc_var(value), index, jitc_var(index),
        mask_2, jitc_var(mask_2), literal);

    print_log(((uint32_t) target == target_) ? "direct" : "copied target",
              scatter_op);

    if (reduce_expanded) {
        // Potentially call jitc_var_reduce_expanded following evaluation
        WeakRef wr(target, jitc_var(target)->counter);
        uintptr_t wr_i = memcpy_cast<uintptr_t>(wr);

        jitc_var_set_callback(
            scatter_op,
            [](uint32_t, int free, void *p) {
                if (free)
                    return;
                WeakRef wr = memcpy_cast<WeakRef>((uintptr_t) p);
                if (!jitc_var(wr))
                    return;
                jitc_var_reduce_expanded(wr.index);
            },
            (void*) wr_i, true
        );
    }

    jitc_var_mark_side_effect(scatter_op);

    return target.release();
}

uint32_t jitc_var_scatter_packet(size_t n, uint32_t target_,
                                 const uint32_t *values, uint32_t index_,
                                 uint32_t mask, ReduceOp op, ReduceMode mode) {
    Ref target = borrow(target_), index = borrow(index_), ptr;

    auto print_log = [&](const char *msg, uint32_t result_node = 0) {
        int type_id = 0;
        if (target_)
            type_id = jitc_var(target_)->type;
        jitc_log(Debug,
                 "jit_var_scatter_packet(): r%u[r%u] %s (...) (type=%s, mask=r%u, "
                 "ptr=r%u, se=r%u, out=r%u, mode=%s) [%s]",
                 target_, index_, reduce_op_symbol[(int) op],
                 type_name[type_id], mask, (uint32_t) ptr, result_node,
                 (uint32_t) target, mode_name[(int) mode], msg);
    };

    bool some_empty = false, all_empty = true;
    for (size_t i = 0; i < n; ++i) {
        uint32_t index = values[i];
        some_empty |= index == 0;
        all_empty &= index == 0;
    }

    if (index_ == 0 && all_empty) {
        print_log("empty scatter");
        return target.release();
    }

    if (target == 0)
        jitc_raise("jit_var_scatter_packet(): attempted to scatter to an empty array!");

    if (some_empty || index_ == 0 || mask == 0)
        jitc_raise("jit_var_scatter_packet(): index, value(s), and mask must "
                   "have a compatible size!");

    auto [target_info, target_v] =
        jitc_var_check("jit_var_scatter_packet", (uint32_t) target);
    auto [var_info, index_v, mask_v] =
        jitc_var_check("jit_var_scatter", index_, mask);
    const uint32_t target_size = target_v->size;
    const JitBackend backend = var_info.backend;

    if (target_v->is_dirty() && op == ReduceOp::Identity 
                             && mode != ReduceMode::Permute 
                             && mode != ReduceMode::NoConflicts) {
        jitc_var_eval(target);
        jitc_var_eval(index_);
        jitc_var_eval(mask);

        target_v = jitc_var(target);
        index_v = jitc_var(index_);
        mask_v = jitc_var(mask);
    }

    // Go to the original if 'target' is wrapped into a loop state variable
    unwrap(target, target_v);
    target_ = target;

    if (target_v->symbolic)
        jitc_raise(
            "jit_var_scatter_packet(): cannot scatter to a symbolic variable (r%u)!",
            (uint32_t) target);

    uint32_t flags = jitc_flags();
    bool symbolic = flags & (uint32_t) JitFlag::SymbolicScope;
    if (var_info.symbolic && !symbolic)
        jitc_raise(
            "jit_var_scatter(): input arrays are symbolic, but the operation "
            "was issued outside of a symbolic recording session.");

    if ((n & (n-1)) || n == 1)
        jitc_raise("jitc_var_scatter_packet(): vector size must be a power of two "
                   "and >= 1 (got %zu)!", n);

    if ((target_info.size & (n-1)) != 0 && target_info.size != 1)
        jitc_raise("jitc_var_scatter_packet(): target r%u has size %u, which is not "
                   "divisible by %zu!", index_, target_info.size, n);

    drjit::unique_ptr<PacketScatterData> psd(new PacketScatterData());

    bool all_zero = true, same_value = true;
    for (size_t i = 0; i < n; ++i) {
        const Variable *v = jitc_var(values[i]);

        if (v->type != target_v->type)
            jitc_raise("jit_var_scatter_packet(): target/value type mismatch!");

        if (v->size != var_info.size && v->size != 1 && var_info.size != 1)
            jitc_raise("jit_var_scatter_packet(): argument size mismatch (%u vs %u)!",
                       v->size, var_info.size);

        all_zero &= v->is_literal() && v->literal == 0;
        same_value &= v->is_literal() && target_v->is_literal() &&
                      v->literal == target_v->literal;
        jitc_var_inc_ref(values[i]);
        psd->values.push_back(values[i]);
        var_info.size = std::max(v->size, var_info.size);
    }

    if (mask_v->is_literal() && mask_v->literal == 0) {
        print_log("skipped, always masked");
        return target.release();
    }

    if (all_zero && op == ReduceOp::Add) {
        print_log("skipped, scatter-addition with zero-valued source variable");
        return target.release();
    }

    if (same_value && op == ReduceOp::Identity) {
        print_log("skipped, target/source are value variables with the "
                  "same value");
        return target.release();
    }

    bool use_packet_op = false;

    if (flags & (uint32_t) JitFlag::PacketOps) {
        if (op == ReduceOp::Identity) {
            use_packet_op = mode == ReduceMode::Auto ||
                            mode == ReduceMode::Expand ||
                            mode == ReduceMode::Permute ||
                            mode == ReduceMode::NoConflicts;
        } else if (op == ReduceOp::Add && backend == JitBackend::LLVM) {
            use_packet_op = (mode == ReduceMode::Expand ||
                             mode == ReduceMode::Permute ||
                             mode == ReduceMode::NoConflicts ||
                             (mode == ReduceMode::Auto &&
                              target_info.size <= llvm_expand_threshold));
        }
    }

    Ref scale = steal(jitc_var_u32(backend, n));

    // Potentially reduce to a sequence of scatters
    if (!use_packet_op) {
        for (size_t i = 0; i < n; ++i) {
            Ref index2 = steal(jitc_var_u32(backend, (uint32_t) i));
            Ref index3 = steal(jitc_var_fma(index, scale, index2));

            uint32_t index_t = target;
            if (index_t == target_)
                target.reset();

            target = steal(jitc_var_scatter(index_t, values[i], index3, mask, op, mode));
        }
        return target.release();
    }

    // Packet size 8 is the max. for the LLVM backend. Split larger requests.
    uint32_t max_width = std::min(8u, jitc_llvm_vector_width);
    if (n > max_width && var_info.backend == JitBackend::LLVM) {
        Ref step = steal(jitc_var_u32(var_info.backend, 1)),
            scale = steal(jitc_var_u32(var_info.backend, n/max_width)),
            index2 = steal(jitc_var_mul(index, scale));

        for (size_t i = 0; i < n; i += max_width) {

            uint32_t index_t = target;
            if (index_t == target_)
                target.reset();

            target = steal(jitc_var_scatter_packet(
                max_width, index_t, values + i, index2, mask, op, mode));
            index2 = steal(jitc_var_add(index2, step));
        }

        return target.release();
    }

    // Must compute final index before potentially expanding below
    index = steal(jitc_var_mul(index, scale));

    // Infer the ReduceOp parameter (if it is set to ReduceOp::Auto)
    // This operation may invalidate variable pointers.
    bool reduce_expanded = false;
    std::tie(mode, reduce_expanded) = jitc_var_infer_reduce_mode(
        "jit_var_scatter_packet()", backend, target, index, op, mode);

    // Check if it is safe to directly write to ``target``.
    // See the original scatter operation for details.
    target_v = jitc_var(target);
    if (target_v->ref_count > 2 && target_v->ref_count_stashed != 1)
        target = steal(jitc_var_copy(target));

    // Get a pointer to the array data. This evaluates the array if needed
    void *target_addr = nullptr;
    target = steal(jitc_var_data(target, false, &target_addr));
    ptr = steal(jitc_var_pointer(backend, target_addr, target, 1));

    // Apply default masks
    Ref mask_2 = steal(jitc_var_mask_apply(mask, var_info.size));
    Ref index_2 = steal(jitc_scatter_gather_index(target, index));

    // Insert a bounds check in debug mode
    if (flags & (uint32_t) JitFlag::Debug) {
        mask_2 = steal(jitc_var_check_bounds(
            op == ReduceOp::Identity ? BoundsCheckType::PacketScatter
                                     : BoundsCheckType::PacketScatterReduce,
            /* original index */ index, mask_2, target_size));
    }

    uint32_t op_size = std::max(var_info.size, jitc_var(mask_2)->size);
    uint32_t scatter_op = jitc_var_new_node_3(
        backend, VarKind::PacketScatter, VarType::Void,
        op_size, symbolic, ptr, jitc_var(ptr), index_2,
        jitc_var(index_2), mask_2, jitc_var(mask_2), (uint64_t) (uintptr_t) psd.get());

    psd->mode = mode;
    psd->op = op;
    if (reduce_expanded)
        psd->to_reduce = WeakRef(target, jitc_var(target)->counter);

    print_log(((uint32_t) target == target_) ? "direct" : "copied target",
              scatter_op);

    jitc_var_set_callback(
        scatter_op,
        [](uint32_t, int free, void *p) {
            PacketScatterData *psd = (PacketScatterData *) p;
            if (free) {
                delete psd;
            } else {
                if (jitc_var(psd->to_reduce))
                    jitc_var_reduce_expanded(psd->to_reduce.index);
            }
        },
        psd.release(), true
    );

    jitc_var_mark_side_effect(scatter_op);

    return target.release();
}


// --------------------------------------------------------------------------
// Dynamic interface to operations
// --------------------------------------------------------------------------

uint32_t jitc_var_op(JitOp op, const uint32_t *dep) {
    switch (op) {
        case JitOp::Add:    return jitc_var_add(dep[0], dep[1]);
        case JitOp::Sub:    return jitc_var_sub(dep[0], dep[1]);
        case JitOp::Mul:    return jitc_var_mul(dep[0], dep[1]);
        case JitOp::Mulhi:  return jitc_var_mulhi(dep[0], dep[1]);
        case JitOp::Div:    return jitc_var_div(dep[0], dep[1]);
        case JitOp::Mod:    return jitc_var_mod(dep[0], dep[1]);
        case JitOp::Min:    return jitc_var_min(dep[0], dep[1]);
        case JitOp::Max:    return jitc_var_max(dep[0], dep[1]);
        case JitOp::Neg:    return jitc_var_neg(dep[0]);
        case JitOp::Not:    return jitc_var_not(dep[0]);
        case JitOp::Sqrt:   return jitc_var_sqrt(dep[0]);
        case JitOp::Rcp:    return jitc_var_rcp(dep[0]);
        case JitOp::Rsqrt:  return jitc_var_rsqrt(dep[0]);
        case JitOp::Abs:    return jitc_var_abs(dep[0]);
        case JitOp::Round:  return jitc_var_round(dep[0]);
        case JitOp::Trunc:  return jitc_var_trunc(dep[0]);
        case JitOp::Floor:  return jitc_var_floor(dep[0]);
        case JitOp::Ceil:   return jitc_var_ceil(dep[0]);
        case JitOp::Fma:    return jitc_var_fma(dep[0], dep[1], dep[2]);
        case JitOp::Select: return jitc_var_select(dep[0], dep[1], dep[2]);
        case JitOp::Sin:    return jitc_var_sin_intrinsic(dep[0]);
        case JitOp::Cos:    return jitc_var_cos_intrinsic(dep[0]);
        case JitOp::Exp2:   return jitc_var_exp2_intrinsic(dep[0]);
        case JitOp::Log2:   return jitc_var_log2_intrinsic(dep[0]);
        case JitOp::Eq:     return jitc_var_eq(dep[0], dep[1]);
        case JitOp::Neq:    return jitc_var_neq(dep[0], dep[1]);
        case JitOp::Lt:     return jitc_var_lt(dep[0], dep[1]);
        case JitOp::Le:     return jitc_var_le(dep[0], dep[1]);
        case JitOp::Gt:     return jitc_var_gt(dep[0], dep[1]);
        case JitOp::Ge:     return jitc_var_ge(dep[0], dep[1]);
        case JitOp::Popc:   return jitc_var_popc(dep[0]);
        case JitOp::Clz:    return jitc_var_clz(dep[0]);
        case JitOp::Brev:   return jitc_var_brev(dep[0]);
        case JitOp::Ctz:    return jitc_var_ctz(dep[0]);
        case JitOp::Shr:    return jitc_var_shr(dep[0], dep[1]);
        case JitOp::Shl:    return jitc_var_shl(dep[0], dep[1]);
        case JitOp::And:    return jitc_var_and(dep[0], dep[1]);
        case JitOp::Or:     return jitc_var_or(dep[0], dep[1]);
        case JitOp::Xor:    return jitc_var_xor(dep[0], dep[1]);
        default: jitc_raise("jit_var_new_op(): unsupported operation!");
    }
}
