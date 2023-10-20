#include <drjit-core/containers.h>
#include "internal.h"
#include "var.h"
#include "log.h"
#include "eval.h"
#include "op.h"

#if defined(_MSC_VER)
#  pragma warning (disable: 4702) // unreachable code
#endif

template <bool Value> using enable_if_t = std::enable_if_t<Value, int>;

/// Various checks that can be requested from jitc_var_check()
enum CheckFlags {
    Disabled = 0,
    IsArithmetic = 1,
    IsInt = 2,
    IsIntOrBool = 4,
    IsNotVoid = 8,
    IsFloat = 16,
    IsCUDA = 32
};

/// Summary information about a set of variables returned by jitc_var_check()
struct VarInfo {
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

    /// Did an operand have the 'placeholder' bit set?
    bool placeholder;
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
 * Finally, the function returns a tuple containing a `VarInfo` record (see the
 * documentation of its fields for details) and a `const Variable *` pointer
 * per operand.
 */
template <int Flags, typename... Args, size_t... Is>
auto jitc_var_check_impl(const char *name, std::index_sequence<Is...>, Args... args) {
    constexpr size_t Size = sizeof...(Args);
    uint32_t dep[Size] = { args... };
    Variable *v[Size];

    bool placeholder = false,
         simplify = false,
         literal = true;

    JitBackend backend = JitBackend::Invalid;
    VarType type = VarType::Void;
    uint32_t size = 0;
    const char *err = nullptr;

    for (uint32_t i = 0; i < Size; ++i) {
        if (!dep[i])
            continue;
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

        if constexpr (Flags != Disabled) {
            if (unlikely(type != VarType::Void && (VarType) vi->type != type)) {
                err = "operands have incompatible types";
                goto fail;
            }
        }

        if (unlikely(backend != JitBackend::Invalid && (JitBackend) vi->backend != backend)) {
            err = "operands have different backends";
            goto fail;
        }

#endif
        if (vi->consumed) {
            err = "operation references an operand that can only be evaluated once";
            goto fail;
        }

        size = std::max(size, vi->size);
        placeholder |= (bool) vi->placeholder;
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
            if (unlikely(v[i]->size != size && v[i]->size != 1)) {
                err = "operands have incompatible sizes";
                size = (uint32_t) -1;
                goto fail;
            }
        }

        if (simplify)
            simplify = jitc_flags() & (uint32_t) JitFlag::ConstProp;
    }

    return drjit::dr_tuple(
        VarInfo{ backend, type, size, simplify, literal, placeholder },
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

template <int Flags = Disabled, typename... Args>
JIT_INLINE auto jitc_var_check(const char *name, Args... args) {
    return jitc_var_check_impl<Flags>(
        name, std::make_index_sequence<sizeof...(Args)>(), args...);
}

// Convert from an 64-bit integer container to a literal type
template <typename Type> Type i2v(uint64_t value) {
    Type result;
    memcpy(&result, &value, sizeof(Type));
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
    memcpy(&dst, &src, sizeof(Dst));
    return dst;
}

template <typename T, typename... Ts> T first(T arg, Ts...) { return arg; }

template <typename Func, typename... Args>
JIT_INLINE uint32_t jitc_eval_literal(const VarInfo &info, Func func,
                                      const Args *...args) {
    uint64_t r = 0;

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
        case VarType::Float32: r = v2i(func(i2v<   float>(args->literal)...)); break;
        case VarType::Float64: r = v2i(func(i2v<  double>(args->literal)...)); break;
        default: jitc_fail("jit_eval_literal(): unsupported variable type!");
    }

    return jitc_var_literal(info.backend, info.type, &r, info.size, 0);
}

// --------------------------------------------------------------------------
// Common constants
// --------------------------------------------------------------------------

uint32_t jitc_make_zero(VarInfo info) {
    uint64_t value = 0;
    return jitc_var_literal(info.backend, info.type, &value, info.size, 0);
}

uint32_t jitc_make_true(VarInfo info) {
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
uint32_t jitc_var_shift(const VarInfo &info, uint32_t index, uint64_t amount) {
    amount = 63 - jitc_clz(amount);
    Ref shift = steal(jitc_var_literal(info.backend, info.type, &amount, info.size, 0));
    return Shl ? jitc_var_shl(index, shift)
               : jitc_var_shr(index, shift);
}

// --------------------------------------------------------------------------

template <typename T, enable_if_t<!std::is_signed_v<T>> = 0>
T eval_neg(T v) { return T(-(std::make_signed_t<T>) v); }

template <typename T, enable_if_t<std::is_signed_v<T>> = 0>
T eval_neg(T v) { return -v; }

static bool eval_neg(bool) { jitc_fail("eval_neg(): unsupported operands!"); }

uint32_t jitc_var_neg(uint32_t a0) {
    auto [info, v0] = jitc_var_check<IsArithmetic>("jit_var_neg", a0);

    uint32_t result = 0;
    if (info.simplify && info.literal)
        result = jitc_eval_literal(info, [](auto l0) { return eval_neg(l0); }, v0);

    if (!result && info.size)
        result = jitc_var_new_node_1(info.backend, VarKind::Neg, info.type,
                                     info.size, info.placeholder, a0, v0);

    jitc_log(Debug, "jit_var_neg(r%u <- r%u)", result, a0);
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
                                     info.size, info.placeholder, a0, v0);

    jitc_log(Debug, "jit_var_not(r%u <- r%u)", result, a0);
    return result;
}

// --------------------------------------------------------------------------

template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
T eval_sqrt(T value) { return std::sqrt(value); }

template <typename T, enable_if_t<!std::is_floating_point_v<T>> = 0>
T eval_sqrt(T) { jitc_fail("eval_sqrt(): unsupported operands!"); }

uint32_t jitc_var_sqrt(uint32_t a0) {
    auto [info, v0] = jitc_var_check<IsFloat>("jit_var_sqrt", a0);

    uint32_t result = 0;
    if (info.simplify && info.literal)
        result = jitc_eval_literal(info, [](auto l0) { return eval_sqrt(l0); }, v0);

    if (!result && info.size)
        result = jitc_var_new_node_1(info.backend, VarKind::Sqrt, info.type,
                                     info.size, info.placeholder, a0, v0);

    jitc_log(Debug, "jit_var_sqrt(r%u <- r%u)", result, a0);
    return result;
}

// --------------------------------------------------------------------------

template <typename T, enable_if_t<std::is_signed_v<T>> = 0>
T eval_abs(T value) { return (T) std::abs(value); }

template <typename T, enable_if_t<!std::is_signed_v<T>> = 0>
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
                                     info.size, info.placeholder, a0, v0);

    jitc_log(Debug, "jit_var_abs(r%u <- r%u)", result, a0);
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
        else if (jitc_is_zero(v0))
            result = jitc_var_resize(a1, info.size);
        else if (jitc_is_zero(v1))
            result = jitc_var_resize(a0, info.size);
    }

    if (!result && info.size)
        result = jitc_var_new_node_2(info.backend, VarKind::Add, info.type,
                                     info.size, info.placeholder, a0, v0, a1, v1);

    jitc_log(Debug, "jit_var_add(r%u <- r%u, r%u)", result, a0, a1);
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
        else if (jitc_is_zero(v1))
            result = jitc_var_resize(a0, info.size);
        else if (a0 == a1 && !jitc_is_float(v0))
            result = jitc_make_zero(info);
    }

    if (!result && info.size)
        result = jitc_var_new_node_2(info.backend, VarKind::Sub, info.type,
                                    info.size, info.placeholder, a0, v0, a1, v1);

    jitc_log(Debug, "jit_var_sub(r%u <- r%u, r%u)", result, a0, a1);
    return result;
}

// --------------------------------------------------------------------------

uint32_t jitc_var_mul(uint32_t a0, uint32_t a1) {
    auto [info, v0, v1] = jitc_var_check<IsArithmetic>("jit_var_mul", a0, a1);

    uint32_t result = 0;
    if (info.simplify) {
        if (info.literal)
            result = jitc_eval_literal(
                info, [](auto l0, auto l1) { return l0 * l1; }, v0, v1);
        else if (jitc_is_one(v0) || (jitc_is_zero(v1) && jitc_is_int(v0)))
            result = jitc_var_resize(a1, info.size);
        else if (jitc_is_one(v1) || (jitc_is_zero(v0) && jitc_is_int(v0)))
            result = jitc_var_resize(a0, info.size);
        else if (jitc_is_uint(info.type) && v0->is_literal() && jitc_is_pow2(v0->literal))
            result = jitc_var_shift<true>(info, a1, v0->literal);
        else if (jitc_is_uint(info.type) && v1->is_literal() && jitc_is_pow2(v1->literal))
            result = jitc_var_shift<true>(info, a0, v1->literal);
    }

    if (!result && info.size)
        result = jitc_var_new_node_2(info.backend, VarKind::Mul, info.type,
                                     info.size, info.placeholder, a0, v0, a1, v1);

    jitc_log(Debug, "jit_var_mul(r%u <- r%u, r%u)", result, a0, a1);
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
        } else if (jitc_is_float(info.type) && v1->is_literal()) {
            uint32_t recip = jitc_var_rcp(a1);
            result = jitc_var_mul(a0, recip);
            jitc_var_dec_ref(recip);
        } else if (a0 == a1 && !jitc_is_float(v0)) {
            uint64_t value = 1;
            result = jitc_var_literal(info.backend, info.type, &value, info.size, 0);
        }
    }


    if (!result && info.size)
        result = jitc_var_new_node_2(info.backend, VarKind::Div, info.type,
                                     info.size, info.placeholder, a0, v0, a1, v1);

    jitc_log(Debug, "jit_var_div(r%u <- r%u, r%u)", result, a0, a1);
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
    if (info.simplify && info.literal)
        result = jitc_eval_literal(
            info, [](auto l0, auto l1) { return eval_mod(l0, l1); }, v0, v1);

    if (!result && info.size)
        result = jitc_var_new_node_2(info.backend, VarKind::Mod, info.type,
                                     info.size, info.placeholder, a0, v0, a1, v1);

    jitc_log(Debug, "jit_var_mod(r%u <- r%u, r%u)", result, a0, a1);
    return result;
}

// --------------------------------------------------------------------------

template <typename T, enable_if_t<!(std::is_integral_v<T> && (sizeof(T) == 4 || sizeof(T) == 8))> = 0>
T eval_mulhi(T, T) { jitc_fail("eval_mulhi(): unsupported operands!"); }

template <typename T, enable_if_t<std::is_integral_v<T> && (sizeof(T) == 4 || sizeof(T) == 8)> = 0>
T eval_mulhi(T a, T b) {
    if constexpr (sizeof(T) == 4) {
        using Wide = std::conditional_t<std::is_signed_v<T>, int64_t, uint64_t>;
        return T(((Wide) a * (Wide) b) >> 32);
    } else {
#if defined(_MSC_VER)
        if constexpr (std::is_signed_v<T>)
            return (T) __mulh((__int64) a, (__int64) b);
        else
            return (T) __umulh((unsigned __int64) a, (unsigned __int64) b);
#else
        using Wide = std::conditional_t<std::is_signed_v<T>, __int128_t, __uint128_t>;
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
                                     info.size, info.placeholder, a0, v0, a1, v1);

    jitc_log(Debug, "jit_var_mulhi(r%u <- r%u, r%u)", result, a0, a1);
    return result;
}

// --------------------------------------------------------------------------

template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
T eval_fma(T a, T b, T c) { return std::fma(a, b, c); }

template <typename T, enable_if_t<!std::is_floating_point_v<T> &&
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
            uint32_t tmp = 0;
            if (jitc_is_one(v0))
                tmp = jitc_var_add(a1, a2);
            else if (jitc_is_one(v1))
                tmp = jitc_var_add(a0, a2);
            else if (jitc_is_zero(v2))
                tmp = jitc_var_mul(a0, a1);
            else if (jitc_is_zero(v0) && jitc_is_zero(v1))
                tmp = jitc_var_new_ref(a2);

            if (tmp) {
                result = jitc_var_resize(tmp, info.size);
                jitc_var_dec_ref(tmp);
            }
        }
    }

    if (!result && info.size)
        result = jitc_var_new_node_3(info.backend, VarKind::Fma, info.type,
                                     info.size, info.placeholder,
                                     a0, v0, a1, v1, a2, v2);

    jitc_log(Debug, "jit_var_fma(r%u <- r%u, r%u, r%u)", result, a0, a1, a2);
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
        else if (a0 == a1)
            result = jitc_var_resize(a0, info.size);
    }

    if (!result && info.size)
        result = jitc_var_new_node_2(info.backend, VarKind::Min, info.type,
                                     info.size, info.placeholder, a0, v0, a1, v1);

    jitc_log(Debug, "jit_var_min(r%u <- r%u, r%u)", result, a0, a1);
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
        else if (a0 == a1)
            result = jitc_var_resize(a0, info.size);
    }


    if (!result && info.size)
        result = jitc_var_new_node_2(info.backend, VarKind::Max, info.type,
                                     info.size, info.placeholder, a0, v0, a1, v1);

    jitc_log(Debug, "jit_var_max(r%u <- r%u, r%u)", result, a0, a1);
    return result;
}

// --------------------------------------------------------------------------

template <typename T, enable_if_t<std::is_floating_point<T>::value> = 0>
T eval_ceil(T value) { return std::ceil(value); }

template <typename T, enable_if_t<!std::is_floating_point<T>::value> = 0>
T eval_ceil(T) { jitc_fail("eval_ceil(): unsupported operands!"); }

uint32_t jitc_var_ceil(uint32_t a0) {
    auto [info, v0] = jitc_var_check<IsFloat>("jit_var_ceil", a0);

    uint32_t result = 0;
    if (info.simplify && info.literal)
        result = jitc_eval_literal(info, [](auto l0) { return eval_ceil(l0); }, v0);

    if (!result && info.size)
        result = jitc_var_new_node_1(info.backend, VarKind::Ceil, info.type,
                                     info.size, info.placeholder, a0, v0);

    jitc_log(Debug, "jit_var_ceil(r%u <- r%u)", result, a0);
    return result;
}

// --------------------------------------------------------------------------

template <typename T, enable_if_t<std::is_floating_point<T>::value> = 0>
T eval_floor(T value) { return std::floor(value); }

template <typename T, enable_if_t<!std::is_floating_point<T>::value> = 0>
T eval_floor(T) { jitc_fail("eval_floor(): unsupported operands!"); }

uint32_t jitc_var_floor(uint32_t a0) {
    auto [info, v0] = jitc_var_check<IsFloat>("jit_var_floor", a0);

    uint32_t result = 0;
    if (info.simplify && info.literal)
        result = jitc_eval_literal(info, [](auto l0) { return eval_floor(l0); }, v0);

    if (!result && info.size)
        result = jitc_var_new_node_1(info.backend, VarKind::Floor, info.type,
                                     info.size, info.placeholder, a0, v0);

    jitc_log(Debug, "jit_var_floor(r%u <- r%u)", result, a0);
    return result;
}

// --------------------------------------------------------------------------

template <typename T, enable_if_t<std::is_floating_point<T>::value> = 0>
T eval_round(T value) { return std::rint(value); }

template <typename T, enable_if_t<!std::is_floating_point<T>::value> = 0>
T eval_round(T) { jitc_fail("eval_round(): unsupported operands!"); }

uint32_t jitc_var_round(uint32_t a0) {
    auto [info, v0] = jitc_var_check<IsFloat>("jit_var_round", a0);

    uint32_t result = 0;
    if (info.simplify && info.literal)
        result = jitc_eval_literal(info, [](auto l0) { return eval_round(l0); }, v0);

    if (!result && info.size)
        result = jitc_var_new_node_1(info.backend, VarKind::Round, info.type,
                                     info.size, info.placeholder, a0, v0);

    jitc_log(Debug, "jit_var_round(r%u <- r%u)", result, a0);
    return result;
}

// --------------------------------------------------------------------------

template <typename T, enable_if_t<std::is_floating_point<T>::value> = 0>
T eval_trunc(T value) { return std::trunc(value); }

template <typename T, enable_if_t<!std::is_floating_point<T>::value> = 0>
T eval_trunc(T) { jitc_fail("eval_trunc(): unsupported operands!"); }

uint32_t jitc_var_trunc(uint32_t a0) {
    auto [info, v0] = jitc_var_check<IsFloat>("jit_var_trunc", a0);

    uint32_t result = 0;
    if (info.simplify && info.literal)
        result = jitc_eval_literal(info, [](auto l0) { return eval_trunc(l0); }, v0);

    if (!result && info.size)
        result = jitc_var_new_node_1(info.backend, VarKind::Trunc, info.type,
                                     info.size, info.placeholder, a0, v0);

    jitc_log(Debug, "jit_var_trunc(r%u <- r%u)", result, a0);
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
                                     info.size, info.placeholder, a0, v0, a1, v1);

    jitc_log(Debug, "jit_var_eq(r%u <- r%u, r%u)", result, a0, a1);
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
                                     info.size, info.placeholder, a0, v0, a1, v1);

    jitc_log(Debug, "jit_var_neq(r%u <- r%u, r%u)", result, a0, a1);
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
                                     info.size, info.placeholder, a0, v0, a1, v1);

    jitc_log(Debug, "jit_var_lt(r%u <- r%u, r%u)", result, a0, a1);
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
                                     info.size, info.placeholder, a0, v0, a1, v1);

    jitc_log(Debug, "jit_var_le(r%u <- r%u, r%u)", result, a0, a1);
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
                                     info.size, info.placeholder, a0, v0, a1, v1);

    jitc_log(Debug, "jit_var_gt(r%u <- r%u, r%u)", result, a0, a1);
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
                                     info.size, info.placeholder, a0, v0, a1, v1);

    jitc_log(Debug, "jit_var_ge(r%u <- r%u, r%u)", result, a0, a1);
    return result;
}

// --------------------------------------------------------------------------

uint32_t jitc_var_select(uint32_t a0, uint32_t a1, uint32_t a2) {
    auto [info, v0, v1, v2] = jitc_var_check("jit_var_select", a0, a1, a2);

    if (info.size && (!jitc_is_bool(v0) || v1->type != v2->type))
        jitc_raise("jitc_var_select(): invalid operands!");

    info.type = (VarType) v1->type;

    uint32_t result = 0;
    if (info.simplify) {
        if (jitc_is_one(v0) || a1 == a2)
            return jitc_var_resize(a1, info.size);
        else if (jitc_is_zero(v0))
            return jitc_var_resize(a2, info.size);
    }

    if (!result && info.size)
        result = jitc_var_new_node_3(info.backend, VarKind::Select, info.type,
                                    info.size, info.placeholder,
                                    a0, v0, a1, v1, a2, v2);

    jitc_log(Debug, "jit_var_select(r%u <- r%u, r%u, r%u)", result, a0, a1, a2);
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
                                     info.size, info.placeholder, a0, v0);

    jitc_log(Debug, "jit_var_popc(r%u <- r%u)", result, a0);
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
                                     info.size, info.placeholder, a0, v0);

    jitc_log(Debug, "jit_var_clz(r%u <- r%u)", result, a0);
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
                                     info.size, info.placeholder, a0, v0);

    jitc_log(Debug, "jit_var_ctz(r%u <- r%u)", result, a0);
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
                                     info.size, info.placeholder, a0, v0, a1, v1);

    jitc_log(Debug, "jit_var_and(r%u <- r%u, r%u)", result, a0, a1);
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
                                     info.size, info.placeholder, a0, v0, a1, v1);

    jitc_log(Debug, "jit_var_or(r%u <- r%u, r%u)", result, a0, a1);
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
                                     info.size, info.placeholder, a0, v0, a1, v1);

    jitc_log(Debug, "jit_var_xor(r%u <- r%u, r%u)", result, a0, a1);
    return result;
}

// --------------------------------------------------------------------------

template <typename T, enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>> = 0>
T eval_shl(T v0, T v1) { return v0 << v1; }

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
                                     info.size, info.placeholder, a0, v0, a1, v1);

    jitc_log(Debug, "jit_var_shl(r%u <- r%u, r%u)", result, a0, a1);
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
                                     info.size, info.placeholder, a0, v0, a1, v1);

    jitc_log(Debug, "jit_var_shr(r%u <- r%u, r%u)", result, a0, a1);
    return result;
}

// --------------------------------------------------------------------------

template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
T eval_rcp(T value) { return 1 / value; }

template <typename T, enable_if_t<!std::is_floating_point_v<T>> = 0>
T eval_rcp(T) { jitc_fail("eval_rcp(): unsupported operands!"); }

uint32_t jitc_var_rcp(uint32_t a0) {
    auto [info, v0] = jitc_var_check<IsFloat>("jit_var_rcp", a0);

    uint32_t result = 0;
    if (info.simplify && info.literal)
        result = jitc_eval_literal(info, [](auto l0) { return eval_rcp(l0); }, v0);

    if (!result && info.backend == JitBackend::LLVM) {
        float f1 = 1.f; double d1 = 1.0;
        uint32_t one = jitc_var_literal(info.backend, info.type,
                                            info.type == VarType::Float32
                                                ? (const void *) &f1
                                                : (const void *) &d1, 1, 0);
        result = jitc_var_div(one, a0);
        jitc_var_dec_ref(one);
    }

    if (!result && info.size)
        result = jitc_var_new_node_1(info.backend, VarKind::Rcp, info.type,
                                     info.size, info.placeholder, a0, v0);

    jitc_log(Debug, "jit_var_rcp(r%u <- r%u)", result, a0);
    return result;
}

// --------------------------------------------------------------------------

template <typename T, enable_if_t<std::is_floating_point<T>::value> = 0>
T eval_rsqrt(T value) { return 1 / std::sqrt(value); }

template <typename T, enable_if_t<!std::is_floating_point<T>::value> = 0>
T eval_rsqrt(T) { jitc_fail("eval_rsqrt(): unsupported operands!"); }

uint32_t jitc_var_rsqrt(uint32_t a0) {
    auto [info, v0] = jitc_var_check<IsFloat>("jit_var_rsqrt", a0);

    uint32_t result = 0;
    if (info.simplify && info.literal)
        result = jitc_eval_literal(info, [](auto l0) { return eval_rsqrt(l0); }, v0);

    if (!result && info.backend == JitBackend::LLVM) {
        // Reciprocal, then square root (lower error than the other way around)
        uint32_t rcp = jitc_var_rcp(a0);
        result = jitc_var_sqrt(rcp);
        jitc_var_dec_ref(rcp);
    }

    if (!result && info.size)
        result = jitc_var_new_node_1(info.backend, VarKind::Rsqrt, info.type,
                                    info.size, info.placeholder, a0, v0);

    jitc_log(Debug, "jit_var_rsqrt(r%u <- r%u)", result, a0);
    return result;
}

// --------------------------------------------------------------------------

template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
T eval_sin(T value) { return std::sin(value); }

template <typename T, enable_if_t<!std::is_floating_point_v<T>> = 0>
T eval_sin(T) { jitc_fail("eval_sin(): unsupported operands!"); }

uint32_t jitc_var_sin(uint32_t a0) {
    auto [info, v0] = jitc_var_check<IsFloat | IsCUDA>("jit_var_sin", a0);

    uint32_t result = 0;
    if (info.simplify && info.literal)
        result = jitc_eval_literal(info, [](auto l0) { return eval_sin(l0); }, v0);

    if (!result && info.size)
        result = jitc_var_new_node_1(info.backend, VarKind::Sin, info.type,
                                     info.size, info.placeholder, a0, v0);

    jitc_log(Debug, "jit_var_sin(r%u <- r%u)", result, a0);
    return result;
}

// --------------------------------------------------------------------------

template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
T eval_cos(T value) { return std::cos(value); }

template <typename T, enable_if_t<!std::is_floating_point_v<T>> = 0>
T eval_cos(T) { jitc_fail("eval_cos(): unsupported operands!"); }

uint32_t jitc_var_cos(uint32_t a0) {
    auto [info, v0] = jitc_var_check<IsFloat | IsCUDA>("jit_var_cos", a0);

    uint32_t result = 0;
    if (info.simplify && info.literal)
        result = jitc_eval_literal(info, [](auto l0) { return eval_cos(l0); }, v0);

    if (!result && info.size)
        result = jitc_var_new_node_1(info.backend, VarKind::Cos, info.type,
                                     info.size, info.placeholder, a0, v0);

    jitc_log(Debug, "jit_var_cos(r%u <- r%u)", result, a0);
    return result;
}

// --------------------------------------------------------------------------

template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
T eval_exp2(T value) { return std::exp2(value); }

template <typename T, enable_if_t<!std::is_floating_point_v<T>> = 0>
T eval_exp2(T) { jitc_fail("eval_exp2(): unsupported operands!"); }

uint32_t jitc_var_exp2(uint32_t a0) {
    auto [info, v0] = jitc_var_check<IsFloat | IsCUDA>("jit_var_exp2", a0);

    uint32_t result = 0;
    if (info.simplify && info.literal)
        result = jitc_eval_literal(info, [](auto l0) { return eval_exp2(l0); }, v0);

    if (!result && info.size)
        result = jitc_var_new_node_1(info.backend, VarKind::Exp2, info.type,
                                     info.size, info.placeholder, a0, v0);

    jitc_log(Debug, "jit_var_exp2(r%u <- r%u)", result, a0);
    return result;
}

// --------------------------------------------------------------------------

template <typename T, enable_if_t<std::is_floating_point_v<T>> = 0>
T eval_log2(T value) { return std::log2(value); }

template <typename T, enable_if_t<!std::is_floating_point_v<T>> = 0>
T eval_log2(T) { jitc_fail("eval_log2(): unsupported operands!"); }

uint32_t jitc_var_log2(uint32_t a0) {
    auto [info, v0] = jitc_var_check<IsFloat | IsCUDA>("jit_var_log2", a0);

    uint32_t result = 0;
    if (info.simplify && info.literal)
        result = jitc_eval_literal(info, [](auto l0) { return eval_log2(l0); }, v0);

    if (!result && info.size)
        result = jitc_var_new_node_1(info.backend, VarKind::Log2, info.type,
                                     info.size, info.placeholder, a0, v0);

    jitc_log(Debug, "jit_var_log2(r%u <- r%u)", result, a0);
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
            info.type, info.size, info.placeholder, a0, v0);

    jitc_log(Debug, "jit_var_cast(r%u <- r%u)", result, a0);
    return result;
}

// --------------------------------------------------------------------------

static uint32_t jitc_scatter_gather_index(uint32_t source, uint32_t index) {
    const Variable *v_source = jitc_var(source),
                   *v_index = jitc_var(index);

    VarType source_type = (VarType) v_index->type;
    if (!jitc_is_int(source_type))
        jitc_raise("jit_scatter_gather_index(): expected an integer array as scatter/gather index");

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

    if (v->is_data() || (VarType) v->type == VarType::Void)
        return 0; // evaluated variable, give up

    if (v->extra) {
        Extra &e = state.extra[var_index];
        if (e.n_dep || e.callback || e.vcall_buckets || e.assemble)
            return 0; // "complicated" variable, give up
    }

    Ref dep[4];
    bool rebuild = v->size != size && v->size != 1;

    if (v->kind == VarKind::DefaultMask && rebuild)
        // Do not re-index the mask, only resize it
        return jitc_var_mask_default((JitBackend) v->backend, size);

    if (!v->is_literal()) {
        for (uint32_t i = 0; i < 4; ++i) {
            uint32_t index_2 = v->dep[i];
            if (!index_2)
                continue;

            if (v->kind == VarKind::Gather && i == 2) {
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


    if (v->kind == VarKind::Counter) {
        return jitc_var_new_ref(new_index);
    } else if (rebuild) {
        Variable v2;
        v2.kind = v->kind;
        v2.backend = v->backend;
        v2.type = v->type;
        v2.size = size;
        v2.optix = v->optix;
        v2.placeholder = v->placeholder;
        if (v->is_stmt()) {
            if (!v->free_stmt) {
                v2.stmt = v->stmt;
            } else {
                v2.stmt = strdup(v->stmt);
                v2.free_stmt = 1;
            }
        } else {
            v2.literal = v->literal;
        }
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


uint32_t jitc_var_gather(uint32_t src, uint32_t index, uint32_t mask) {
    if (index == 0)
        return 0;

    auto [src_info, src_v] =
        jitc_var_check("jit_var_gather", src);
    auto [var_info, index_v, mask_v] =
        jitc_var_check("jit_var_gather", index, mask);

    uint32_t result = 0, ptr = 0;
    const char *msg = "";

    {
        /// Variables with _v subscript only inspected in this scope
        if (src_info.placeholder)
            jitc_raise("jit_var_gather(): cannot gather from a placeholder variable!");

        if (mask_v->is_literal() && mask_v->literal == 0) {
            var_info.type = src_info.type;
            result = jitc_make_zero(var_info);
            msg = ": elided (always masked)";
        }

        /// Make sure that the src and index variables doesn't have pending side effects
        if (!result && unlikely(index_v->is_dirty() || src_v->is_dirty())) {
            jitc_eval(thread_state(src_info.backend));
            if (jitc_var(index)->is_dirty())
                jitc_fail("jit_var_gather(): operand r%u remains dirty following evaluation!", index);
            if (jitc_var(src)->is_dirty())
                jitc_fail("jit_var_gather(): operand r%u remains dirty following evaluation!", src);

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
            msg = ": elided (reindexed)";
        }
    }

    if (!result) {
        Ref ptr_2   = steal(jitc_var_pointer(src_info.backend, jitc_var_ptr(src), src, 0)),
            index_2 = steal(jitc_scatter_gather_index(src, index)),
            mask_2  = steal(jitc_var_mask_apply(mask, var_info.size));

        var_info.size = std::max(var_info.size, jitc_var(mask_2)->size);

        result = jitc_var_new_node_3(
            src_info.backend, VarKind::Gather, src_info.type, var_info.size,
            var_info.placeholder, ptr_2, jitc_var(ptr_2), index_2,
            jitc_var(index_2), mask_2, jitc_var(mask_2));
        ptr = (uint32_t) ptr_2;
    }

    jitc_log(Debug,
             "jit_var_gather(r%u <- r%u[r%u] if r%u, via ptr r%u)%s",
             result, src, index, mask, ptr, msg);

    return result;
}

static const char *reduce_op_name[(int) ReduceOp::Count] = {
    "none", "add", "mul", "min", "max", "and", "or"
};

void jitc_var_scatter_reduce_kahan(uint32_t *target_1, uint32_t *target_2,
                                   uint32_t value, uint32_t index, uint32_t mask) {
    if (value == 0 && index == 0)
        return;

    auto [var_info, value_v, index_v, mask_v] =
        jitc_var_check("jit_var_scatter_reduce_kahan", value, index, mask);

    auto [target_info, target_1_v, target_2_v] =
        jitc_var_check("jit_var_scatter_reduce_kahan", *target_1, *target_2);

    if (target_1 == target_2)
        jitc_raise("jit_var_scatter_reduce_kahan(): the destination arrays cannot be the same!");

    if (target_1_v->placeholder || target_2_v->placeholder)
        jitc_raise("jit_var_scatter_reduce_kahan(): cannot scatter to a placeholder variable!");

    if (target_1_v->type != value_v->type || target_2_v->type != value_v->type)
        jitc_raise("jit_var_scatter_reduce_kahan(): target/value type mismatch!");

    if (target_1_v->size != target_2_v->size)
        jitc_raise("jit_var_scatter_reduce_kahan(): target size mismatch!");

    if (value_v->is_literal() && value_v->literal == 0)
        return;

    if (mask_v->is_literal() && mask_v->literal == 0)
        return;

    var_info.placeholder |= (bool) (jitc_flags() & (uint32_t) JitFlag::Recording);

    // Check if it is safe to write directly
    if (jitc_var(*target_1)->ref_count > 1) {
        uint32_t tmp = jitc_var_copy(*target_1);
        jitc_var_dec_ref(*target_1);
        *target_1 = tmp;
    }

    if (jitc_var(*target_2)->ref_count > 1 || *target_1 == *target_2) {
        uint32_t tmp = jitc_var_copy(*target_2);
        jitc_var_dec_ref(*target_2);
        *target_2 = tmp;
    }

    Ref ptr_1 = steal(jitc_var_pointer(var_info.backend, jitc_var_ptr(*target_1), *target_1, 1));
    Ref ptr_2 = steal(jitc_var_pointer(var_info.backend, jitc_var_ptr(*target_2), *target_2, 1));

    Ref mask_2  = steal(jitc_var_mask_apply(mask, var_info.size)),
        index_2 = steal(jitc_scatter_gather_index(*target_1, index));

    var_info.size = std::max(var_info.size, jitc_var(mask_2)->size);

    uint32_t result = jitc_var_new_node_0(
        var_info.backend, VarKind::ScatterKahan, VarType::Void,
        var_info.size, var_info.placeholder);

    uint32_t *dep = (uint32_t *) malloc_check(sizeof(uint32_t) * 5);
    dep[0] = ptr_1;
    dep[1] = ptr_2;
    dep[2] = index_2;
    dep[3] = mask_2;
    dep[4] = value;

    jitc_var_inc_ref(ptr_1);
    jitc_var_inc_ref(ptr_2);
    jitc_var_inc_ref(index_2);
    jitc_var_inc_ref(mask_2);
    jitc_var_inc_ref(value);

    Variable *result_v = jitc_var(result);
    result_v->extra = 1;

    Extra &e = state.extra[result];
    e.n_dep = 5;
    e.dep = dep;

    jitc_log(Debug,
             "jit_var_scatter_reduce_kahan((r%u[r%u], r%u[r%u]) += r%u if r%u, via "
             "ptrs (r%u, r%u)): r%u",
             *target_1, (uint32_t) index_2, *target_2, (uint32_t) index_2, value, (uint32_t) mask_2,
             (uint32_t) ptr_1, (uint32_t) ptr_2, result);

    jitc_var_mark_side_effect(result);
}

uint32_t jitc_var_scatter_inc(uint32_t *target, uint32_t index, uint32_t mask) {
    auto [var_info, index_v, mask_v] =
        jitc_var_check("jit_var_scatter_inc", index, mask);

    auto [target_info, target_v] =
        jitc_var_check("jit_var_scatter_inc", *target);

    if (target_v->placeholder)
        jitc_raise("jit_var_scatter_inc(): cannot scatter to a placeholder variable!");

    if ((VarType) target_v->type != VarType::UInt32)
        jitc_raise("jit_var_scatter_inc(): target must be an unsigned 32-bit array!");

    if ((VarType) index_v->type != VarType::UInt32)
        jitc_raise("jit_var_scatter_inc(): index must be an unsigned 32-bit array!");

    if (index_v->size != 1)
        jitc_raise("jit_var_scatter_inc(): index must be a scalar! (this is a "
                   "limitation of the current implementation that enables a "
                   "particularly simple and efficient implementation)");

    if (mask_v->is_literal() && mask_v->literal == 0)
        return 0;

    var_info.placeholder |= (bool) (jitc_flags() & (uint32_t) JitFlag::Recording);

    // Check if it is safe to write directly
    if (jitc_var(*target)->ref_count > 1) { // 1 from original array, 1 from borrow above
        uint32_t tmp = jitc_var_copy(*target);
        jitc_var_dec_ref(*target);
        *target = tmp;
    }

    Ref ptr = steal(jitc_var_pointer(var_info.backend, jitc_var_ptr(*target), *target, 0));

    Ref mask_2  = steal(jitc_var_mask_apply(mask, var_info.size)),
        index_2 = steal(jitc_scatter_gather_index(*target, index));

    var_info.size = std::max(var_info.size, jitc_var(mask_2)->size);

    uint32_t result = jitc_var_new_node_3(
        var_info.backend, VarKind::ScatterInc, VarType::UInt32, var_info.size,
        var_info.placeholder, ptr, jitc_var(ptr), index_2, jitc_var(index_2),
        mask_2, jitc_var(mask_2));

    jitc_log(Debug,
             "jit_var_scatter_inc(r%u[r%u] += 1 if r%u, via "
             "ptr r%u): r%u",
             *target, (uint32_t) index_2, (uint32_t) mask_2, (uint32_t) ptr,
             result);

    return result;
}

uint32_t jitc_var_scatter(uint32_t target_, uint32_t value, uint32_t index,
                          uint32_t mask, ReduceOp reduce_op) {
    Ref target = borrow(target_), ptr;

    auto print_log = [&](const char *reason, uint32_t result_node = 0) {
        if (result_node)
            jitc_log(Debug,
                     "jit_var_scatter(r%u[r%u] <- r%u if r%u, via "
                     "ptr r%u, reduce_op=%s): r%u (output=r%u, %s)",
                     target_, index, value, mask, (uint32_t) ptr,
                     reduce_op_name[(int) reduce_op], result_node,
                     (uint32_t) target, reason);
        else
            jitc_log(Debug,
                     "jit_var_scatter(r%u[r%u] <- r%u if r%u, via "
                     "ptr r%u, reduce_op=%s) (%s)",
                     target_, index, value, mask, (uint32_t) ptr,
                     reduce_op_name[(int) reduce_op], reason);
    };

    if (value == 0 && index == 0) {
        print_log("empty scatter");
        return target.release();
    }

    auto [var_info, value_v, index_v, mask_v] =
        jitc_var_check("jit_var_scatter", value, index, mask);

    const Variable *target_v = jitc_var(target);

    if (target_v->placeholder)
        jitc_raise("jit_var_scatter(): cannot scatter to a placeholder variable!");

    var_info.placeholder |= (bool) (jitc_flags() & (uint32_t) JitFlag::Recording);

    if (target_v->type != value_v->type)
        jitc_raise("jit_var_scatter(): target/value type mismatch!");

    if (target_v->is_literal() && value_v->is_literal() &&
        target_v->literal == value_v->literal && reduce_op == ReduceOp::None) {
        print_log("skipped, target/source are value variables with the "
                  "same value");
        return target.release();
    }

    if (mask_v->is_literal() && mask_v->literal == 0) {
        print_log("skipped, always masked");
        return target.release();
    }

    if (value_v->is_literal() && value_v->literal == 0 && reduce_op == ReduceOp::Add) {
        print_log("skipped, scatter_reduce(ScatterOp.Add) with zero-valued "
                  "source variable");
        return target.release();
    }

    // Check if it is safe to write directly
    if (target_v->ref_count > 2) /// 1 from original array, 1 from borrow above
        target = steal(jitc_var_copy(target));

    ptr = steal(jitc_var_pointer(var_info.backend, jitc_var_ptr(target), target, 1));

    Ref mask_2  = steal(jitc_var_mask_apply(mask, var_info.size)),
        index_2 = steal(jitc_scatter_gather_index(target, index));

    var_info.size = std::max(var_info.size, jitc_var(mask_2)->size);

    uint32_t result = jitc_var_new_node_4(
        var_info.backend, VarKind::Scatter, VarType::Void,
        var_info.size, var_info.placeholder, ptr,
        jitc_var(ptr), value, jitc_var(value), index_2, jitc_var(index_2),
        mask_2, jitc_var(mask_2), (uint64_t) reduce_op);

    print_log(((uint32_t) target == target_) ? "direct" : "copy", result);

    jitc_var_mark_side_effect(result);

    return target.release();
}

// --------------------------------------------------------------------------

void jitc_var_printf(JitBackend backend, uint32_t mask, const char *fmt,
                     uint32_t narg, const uint32_t *arg) {
    ThreadState *ts = thread_state(backend);
    bool dirty, placeholder;
    uint32_t size;

    {
        Variable *mask_v = jitc_var(mask);
        placeholder = mask_v->placeholder;
        dirty = mask_v->ref_count_se != 0;
        size = mask_v->size;
    }

    for (uint32_t i = 0; i < narg; ++i) {
        const Variable *v = jitc_var(arg[i]);
        if (unlikely(size != v->size && v->size != 1 && size != 1))
            jitc_raise("jit_var_printf(): arrays have incompatible size!");
        dirty |= v->ref_count_se != 0;
        placeholder |= v->placeholder;
        size = std::max(size, v->size);
    }

    placeholder |= (bool) (jitc_flags() & (uint32_t) JitFlag::Recording);

    if (dirty) {
        jitc_eval(ts);
        dirty = false;
        if (mask)
            dirty = jitc_var(mask)->ref_count_se != 0;
        for (uint32_t i = 0; i < narg; ++i)
            dirty |= jitc_var(arg[i])->ref_count_se != 0;
        jitc_fail("jit_var_printf(): variable remains dirty after evaluation!");
    }

    Ref mask_2 = steal(jitc_var_mask_apply(mask, size));
    std::unique_ptr<char[], decltype(&std::free)>
        fmt_copy { strdup(fmt), &std::free };

    Ref result;
    if (backend == JitBackend::LLVM) {
        Ref target = steal(jitc_var_pointer(backend, (const void *) &printf, 0, 0));
        result = steal(
            jitc_var_new_node_2(backend, VarKind::Printf, VarType::Void, size,
                                placeholder, mask_2, jitc_var(mask_2), target,
                                jitc_var(target), (uintptr_t) fmt_copy.get()));
    } else {
        result = steal(
            jitc_var_new_node_1(backend, VarKind::Printf, VarType::Void, size,
                                placeholder, mask_2, jitc_var(mask_2),
                                (uintptr_t) fmt_copy.get()));
    }

    Variable *v = jitc_var(result);
    v->extra = 1;

    size_t dep_size = narg * sizeof(uint32_t);
    Extra &e = state.extra[result];
    e.n_dep = narg;
    e.dep = (uint32_t *) malloc_check(dep_size);
    memcpy(e.dep, arg, dep_size);
    for (uint32_t i = 0; i < narg; ++i)
        jitc_var_inc_ref(arg[i]);

    e.callback_data = fmt_copy.release();
    e.callback = [](uint32_t, int free_var, void *ptr) {
        if (free_var)
            free(ptr);
    };
    e.callback_internal = true;

    jitc_log(Debug, "jit_var_printf(void r%u, fmt=\"%s\")", (uint32_t) result, fmt);
    jitc_var_mark_side_effect(result.release());
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
        case JitOp::Sin:    return jitc_var_sin(dep[0]);
        case JitOp::Cos:    return jitc_var_cos(dep[0]);
        case JitOp::Exp2:   return jitc_var_exp2(dep[0]);
        case JitOp::Log2:   return jitc_var_log2(dep[0]);
        case JitOp::Eq:     return jitc_var_eq(dep[0], dep[1]);
        case JitOp::Neq:    return jitc_var_neq(dep[0], dep[1]);
        case JitOp::Lt:     return jitc_var_lt(dep[0], dep[1]);
        case JitOp::Le:     return jitc_var_le(dep[0], dep[1]);
        case JitOp::Gt:     return jitc_var_gt(dep[0], dep[1]);
        case JitOp::Ge:     return jitc_var_ge(dep[0], dep[1]);
        case JitOp::Popc:   return jitc_var_popc(dep[0]);
        case JitOp::Clz:    return jitc_var_clz(dep[0]);
        case JitOp::Ctz:    return jitc_var_ctz(dep[0]);
        case JitOp::Shr:    return jitc_var_shr(dep[0], dep[1]);
        case JitOp::Shl:    return jitc_var_shl(dep[0], dep[1]);
        case JitOp::And:    return jitc_var_and(dep[0], dep[1]);
        case JitOp::Or:     return jitc_var_or(dep[0], dep[1]);
        case JitOp::Xor:    return jitc_var_xor(dep[0], dep[1]);
        default: jitc_raise("jit_var_new_op(): unsupported operation!");
    }
}

