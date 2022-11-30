/*
    src/op.cpp -- Conversion of standard operations into PTX and LLVM IR

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "internal.h"
#include "var.h"
#include "eval.h"
#include "log.h"
#include "op.h"
#include "printf.h"

/// Temporary string buffer for miscellaneous variable-related tasks
extern Buffer var_buffer;

// ===========================================================================
// Helper functions to classify different variable types
// ===========================================================================

static bool jitc_is_arithmetic(VarType type) {
    return type != VarType::Void && type != VarType::Bool;
}

static bool jitc_is_float(VarType type) {
    return type == VarType::Float16 ||
           type == VarType::Float32 ||
           type == VarType::Float64;
}

static bool jitc_is_sint(VarType type) {
    return type == VarType::Int8 ||
           type == VarType::Int16 ||
           type == VarType::Int32 ||
           type == VarType::Int64;
}

static bool jitc_is_uint(VarType type) {
    return type == VarType::UInt8 ||
           type == VarType::UInt16 ||
           type == VarType::UInt32 ||
           type == VarType::UInt64;
}

static bool jitc_is_int(VarType type) {
    return jitc_is_sint(type) || jitc_is_uint(type);
}

static bool jitc_is_not_void(VarType type) {
    return type != VarType::Void;
}

// ===========================================================================
// Evaluation helper routines for value constant values
// ===========================================================================

template <bool Value>
using enable_if_t = typename std::enable_if<Value, int>::type;

template <typename Dst, typename Src>
Dst memcpy_cast(const Src &src) {
    static_assert(sizeof(Src) == sizeof(Dst), "memcpy_cast: size mismatch!");
    Dst dst;
    memcpy(&dst, &src, sizeof(Dst));
    return dst;
}

template <typename T> T eval_not(T v) {
    using U = uint_with_size_t<T>;
    return memcpy_cast<T>(U(~memcpy_cast<U>(v)));
}

template <typename T> T eval_and(T v0, T v1) {
    using U = uint_with_size_t<T>;
    return memcpy_cast<T>(U(memcpy_cast<U>(v0) & memcpy_cast<U>(v1)));
}

template <typename T> T eval_or(T v0, T v1) {
    using U = uint_with_size_t<T>;
    return memcpy_cast<T>(U(memcpy_cast<U>(v0) | memcpy_cast<U>(v1)));
}

template <typename T> T eval_xor(T v0, T v1) {
    using U = uint_with_size_t<T>;
    return memcpy_cast<T>(U(memcpy_cast<U>(v0) ^ memcpy_cast<U>(v1)));
}

template <typename T, enable_if_t<std::is_integral<T>::value &&
                                 !std::is_same<T, bool>::value> = 0>
T eval_shl(T v0, T v1) {
    return v0 << v1;
}

template <typename T, enable_if_t<!std::is_integral<T>::value ||
                                  std::is_same<T, bool>::value> = 0>
T eval_shl(T, T) {
    jitc_raise("eval_shl(): unsupported operands!");
}

template <typename T, enable_if_t<std::is_integral<T>::value &&
                                 !std::is_same<T, bool>::value> = 0>
T eval_shr(T v0, T v1) {
    return v0 >> v1;
}

template <typename T, enable_if_t<!std::is_integral<T>::value ||
                                   std::is_same<T, bool>::value> = 0>
T eval_shr(T, T) {
    jitc_raise("eval_shr(): unsupported operands!");
}

inline bool eval_and(bool v0, bool v1) { return v0 && v1; }
inline bool eval_or(bool v0, bool v1) { return v0 || v1; }
inline bool eval_xor(bool v0, bool v1) { return v0 != v1; }
inline bool eval_not(bool v) { return !v; }

template <typename T, enable_if_t<std::is_unsigned<T>::value> = 0>
T eval_neg(T v) {
    using TS = typename std::make_signed<T>::type;
    return T(-(TS) v);
}

template <typename T, enable_if_t<std::is_signed<T>::value> = 0>
T eval_neg(T v) {
    return -v;
}

inline bool eval_neg(bool) {
    jitc_raise("eval_neg(): unsupported operands!");
}

template <typename T> T eval_div(T v0, T v1) { return v0 / v1; }
inline bool eval_div(bool, bool) {
    jitc_raise("eval_div(): unsupported operands!");
}

template <typename T, enable_if_t<std::is_signed<T>::value> = 0>
T eval_abs(T value) { return (T) std::abs(value); }

template <typename T, enable_if_t<!std::is_signed<T>::value> = 0>
T eval_abs(T value) { return value; }

template <typename T, enable_if_t<!std::is_integral<T>::value ||
                                  std::is_same<T, bool>::value> = 0>
T eval_mod(T, T) {
    jitc_raise("eval_mod(): unsupported operands!");
}

template <typename T, enable_if_t<std::is_integral<T>::value &&
                                  !std::is_same<T, bool>::value> = 0>
T eval_mod(T v0, T v1) {
    return v0 % v1;
}

template <typename T, enable_if_t<std::is_floating_point<T>::value> = 0>
T eval_rcp(T value) { return 1 / value; }

template <typename T, enable_if_t<!std::is_floating_point<T>::value> = 0>
T eval_rcp(T) {
    jitc_raise("eval_rcp(): unsupported operands!");
}

template <typename T, enable_if_t<std::is_floating_point<T>::value> = 0>
T eval_sqrt(T value) { return std::sqrt(value); }

template <typename T, enable_if_t<!std::is_floating_point<T>::value> = 0>
T eval_sqrt(T) {
    jitc_raise("eval_sqrt(): unsupported operands!");
}

template <typename T, enable_if_t<std::is_floating_point<T>::value> = 0>
T eval_rsqrt(T value) { return 1 / std::sqrt(value); }

template <typename T, enable_if_t<!std::is_floating_point<T>::value> = 0>
T eval_rsqrt(T) {
    jitc_raise("eval_rsqrt(): unsupported operands!");
}

template <typename T, enable_if_t<std::is_floating_point<T>::value> = 0>
T eval_ceil(T value) { return std::ceil(value); }

template <typename T, enable_if_t<!std::is_floating_point<T>::value> = 0>
T eval_ceil(T) {
    jitc_raise("eval_ceil(): unsupported operands!");
}

template <typename T, enable_if_t<std::is_floating_point<T>::value> = 0>
T eval_floor(T value) { return std::floor(value); }

template <typename T, enable_if_t<!std::is_floating_point<T>::value> = 0>
T eval_floor(T) {
    jitc_raise("eval_floor(): unsupported operands!");
}

template <typename T, enable_if_t<std::is_floating_point<T>::value> = 0>
T eval_round(T value) { return std::rint(value); }

template <typename T, enable_if_t<!std::is_floating_point<T>::value> = 0>
T eval_round(T) {
    jitc_raise("eval_round(): unsupported operands!");
}

template <typename T, enable_if_t<std::is_floating_point<T>::value> = 0>
T eval_trunc(T value) { return std::trunc(value); }

template <typename T, enable_if_t<!std::is_floating_point<T>::value> = 0>
T eval_trunc(T) {
    jitc_raise("eval_trunc(): unsupported operands!");
}

template <typename T, enable_if_t<std::is_floating_point<T>::value> = 0>
T eval_exp2(T value) { return std::exp2(value); }

template <typename T, enable_if_t<!std::is_floating_point<T>::value> = 0>
T eval_exp2(T) {
    jitc_raise("eval_exp2(): unsupported operands!");
}

template <typename T, enable_if_t<std::is_floating_point<T>::value> = 0>
T eval_log2(T value) { return std::log2(value); }

template <typename T, enable_if_t<!std::is_floating_point<T>::value> = 0>
T eval_log2(T) {
    jitc_raise("eval_log2(): unsupported operands!");
}

template <typename T, enable_if_t<std::is_floating_point<T>::value> = 0>
T eval_sin(T value) { return std::sin(value); }

template <typename T, enable_if_t<!std::is_floating_point<T>::value> = 0>
T eval_sin(T) {
    jitc_raise("eval_sin(): unsupported operands!");
}

template <typename T, enable_if_t<std::is_floating_point<T>::value> = 0>
T eval_cos(T value) { return std::cos(value); }

template <typename T, enable_if_t<!std::is_floating_point<T>::value> = 0>
T eval_cos(T) {
    jitc_raise("eval_cos(): unsupported operands!");
}

template <typename T, enable_if_t<!std::is_integral<T>::value ||
                                  std::is_same<T, bool>::value> = 0>
T eval_popc(T) {
    jitc_raise("eval_popc(): unsupported operands!");
}

template <typename T, enable_if_t<std::is_integral<T>::value &&
                                 !std::is_same<T, bool>::value> = 0>
T eval_popc(T value_) {
    using T2 = typename std::make_unsigned<T>::type;
    T2 value = (T2) value_;
    T result = 0;

    while (value) {
        result += value & 1;
        value >>= 1;
    }

    return result;
}

template <typename T, enable_if_t<!std::is_integral<T>::value ||
                                  std::is_same<T, bool>::value> = 0>
T eval_clz(T) {
    jitc_raise("eval_clz(): unsupported operands!");
}

template <typename T, enable_if_t<std::is_integral<T>::value &&
                                 !std::is_same<T, bool>::value> = 0>
T eval_clz(T value_) {
    using T2 = typename std::make_unsigned<T>::type;
    T2 value = (T2) value_;
    T result = sizeof(T) * 8;
    while (value) {
        value >>= 1;
        result -= 1;
    }
    return result;
}

template <typename T, enable_if_t<!std::is_integral<T>::value ||
                                  std::is_same<T, bool>::value> = 0>
T eval_ctz(T) {
    jitc_raise("eval_ctz(): unsupported operands!");
}

template <typename T, enable_if_t<std::is_integral<T>::value &&
                                 !std::is_same<T, bool>::value> = 0>
T eval_ctz(T value) {
    T result = sizeof(T) * 8;
    while (value) {
        value <<= 1;
        result -= 1;
    }
    return result;
}

template <typename T, enable_if_t<std::is_floating_point<T>::value> = 0>
T eval_fma(T a, T b, T c) { return std::fma(a, b, c); }

template <typename T, enable_if_t<!std::is_floating_point<T>::value &&
                                  !std::is_same<T, bool>::value> = 0>
T eval_fma(T a, T b, T c) {
    return (T) (a * b + c);
}

template <typename T, enable_if_t<std::is_same<T, bool>::value> = 0>
T eval_fma(T, T, T) {
    jitc_raise("eval_fma(): unsupported operands!");
}

template <typename T, enable_if_t<!(std::is_integral<T>::value && (sizeof(T) == 4 || sizeof(T) == 8))> = 0>
T eval_mulhi(T, T) {
    jitc_raise("eval_mulhi(): unsupported operands!");
}

template <typename T, enable_if_t<std::is_integral<T>::value && (sizeof(T) == 4 || sizeof(T) == 8)> = 0>
T eval_mulhi(T a, T b) {
    if (sizeof(T) == 4) {
        using Wide = std::conditional_t<std::is_signed<T>::value, int64_t, uint64_t>;
        return T(((Wide) a * (Wide) b) >> 32);
    } else {
#if defined(_MSC_VER)
        if (std::is_signed<T>::value)
            return (T) __mulh((__int64) a, (__int64) b);
        else
            return (T) __umulh((unsigned __int64) a, (unsigned __int64) b);
#else
        using Wide = std::conditional_t<std::is_signed<T>::value, __int128_t, __uint128_t>;
        return T(((Wide) a * (Wide) b) >> 64);
#endif
    }
}


// ===========================================================================
// Infrastructure to apply a given expression (lambda) to value varaibles
// ===========================================================================

#if defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable: 4702) // unreachable code
#endif

template <typename Type> Type i2v(uint64_t value) {
    Type result;
    memcpy(&result, &value, sizeof(Type));
    return result;
}

template <typename Type, enable_if_t<!std::is_same<Type, bool>::value> = 0>
uint64_t v2i(Type value) {
    uint64_t result = 0;
    memcpy(&result, &value, sizeof(Type));
    return result;
}

template <typename Type, enable_if_t<std::is_same<Type, bool>::value> = 0>
uint64_t v2i(Type value) {
    return value ? 1 : 0;
}

template <typename Arg, typename... Args> Arg first(Arg arg, Args...) { return arg; }

template <bool Cond = false, typename Func, typename... Args>
uint64_t jitc_eval_literal(Func func, const Args *... args) {
    switch ((VarType) first(args->type...)) {
        case VarType::Bool:    return v2i(func(i2v<   bool> (args->literal)...));
        case VarType::Int8:    return v2i(func(i2v< int8_t> (args->literal)...));
        case VarType::UInt8:   return v2i(func(i2v<uint8_t> (args->literal)...));
        case VarType::Int16:   return v2i(func(i2v< int16_t>(args->literal)...));
        case VarType::UInt16:  return v2i(func(i2v<uint16_t>(args->literal)...));
        case VarType::Int32:   return v2i(func(i2v< int32_t>(args->literal)...));
        case VarType::UInt32:  return v2i(func(i2v<uint32_t>(args->literal)...));
        case VarType::Int64:   return v2i(func(i2v< int64_t>(args->literal)...));
        case VarType::UInt64:  return v2i(func(i2v<uint64_t>(args->literal)...));
        case VarType::Float32: return v2i(func(i2v<   float>(args->literal)...));
        case VarType::Float64: return v2i(func(i2v<  double>(args->literal)...));
        default: jitc_fail("jit_eval_value(): unsupported variable type!");
    }
}

#if defined(_MSC_VER)
#  pragma warning(pop)
#endif

// ===========================================================================
// Helper routines for turning multiplies and divisions into shifts
// ===========================================================================

static bool jitc_is_pow2(uint64_t value) {
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

uint32_t jitc_var_shift(JitBackend backend, VarType vt, JitOp op,
                        uint32_t index, uint64_t amount) {
    amount = 63 - jitc_clz(amount);
    uint32_t shift = jitc_var_new_literal(backend, vt, &amount, 1, 0);
    uint32_t deps[2] = { index, shift };
    uint32_t result = jitc_var_new_op(op, 2, deps);
    jitc_var_dec_ref(shift);
    return result;
}

// ===========================================================================
// jitc_var_new_op(): various standard operations for DrJit variables
// ===========================================================================

const char *op_name[(int) JitOp::Count] {
    // ---- Unary ----
    "not", "neg", "abs", "sqrt", "rcp", "rsqrt", "ceil", "floor", "round", "trunc", "exp2", "log2", "sin", "cos",
    "popc", "clz", "ctz",

    // ---- Binary ----
    "add", "sub", "mul", "mulhi", "div", "mod", "min", "max", "and", "or",
    "xor", "shl", "shr",

    // ---- Comparisons ----
    "eq", "neq", "lt", "le", "gt", "ge",

    // ---- Ternary ----
    "fmadd", "select"
};

static_assert(sizeof(op_name) / sizeof(const char *) == (size_t) JitOp::Count,
              "op_name and JitOp enum are out of sync!");

// Error handler
JIT_NOINLINE uint32_t jitc_var_new_op_fail(const char *error, JitOp op,
                                           uint32_t n_dep, const uint32_t *dep);

uint32_t jitc_var_new_op(JitOp op, uint32_t n_dep, const uint32_t *dep) {
    uint32_t size = 0;
    bool dirty = false, literal = true, uninitialized = false, placeholder = false;
    uint32_t vti = 0;
    bool literal_zero[4] { }, literal_one[4] { };
    uint32_t backend_i = 0;
    Variable *v[4] { };
    bool const_prop = jitc_flags() & (uint32_t) JitFlag::ConstProp;

    if (unlikely(n_dep == 0 || n_dep > 4))
        jitc_fail("jit_var_new_op(): 1-4 dependent variables supported!");

    for (uint32_t i = 0; i < n_dep; ++i) {
        if (likely(dep[i])) {
            Variable *vi = jitc_var(dep[i]);
            vti = std::max(vti, vi->type);
            size = std::max(size, vi->size);
            dirty |= vi->is_dirty();
            placeholder |= (bool) vi->placeholder;
            backend_i |= (uint32_t) vi->backend;
            v[i] = vi;

            if (vi->is_literal() && const_prop) {
                uint64_t one;
                switch ((VarType) vi->type) {
                    case VarType::Float16: one = 0x3c00ull; break;
                    case VarType::Float32: one = 0x3f800000ull; break;
                    case VarType::Float64: one = 0x3ff0000000000000ull; break;
                    default: one = 1; break;
                }
                literal_zero[i] = vi->literal == 0;
                literal_one[i] = vi->literal == one;
            } else {
                literal = false;
            }
        } else {
            uninitialized = true;
        }
    }

    JitBackend backend = (JitBackend) backend_i;
    VarType vt  = (VarType) vti,
            vtr = vt;

    // Some sanity checks
    const char *error = nullptr;
    if (unlikely(size == 0))
        return 0;
    else if (unlikely(uninitialized))
        error = "arithmetic involving an uninitialized variable!";

    for (uint32_t i = 0; i < n_dep; ++i) {
        if (error)
            break;

        else if (unlikely(v[i]->size != size && v[i]->size != 1)) {
            error = "arithmetic involving arrays of incompatible size!";
        } else if (unlikely(v[i]->type != vti)) {
            // Two special cases in which mixed mask/value arguments are OK
            bool exception_1 = op == JitOp::Select && i == 0 &&
                               (VarType) v[i]->type == VarType::Bool;
            bool exception_2 = (op == JitOp::And || op == JitOp::Or) && i == 1 &&
                               (VarType) v[i]->type == VarType::Bool;
            if (!exception_1 && !exception_2)
                error = "arithmetic involving arrays of incompatible type!";
        } else if (unlikely(v[i]->backend != backend_i)) {
            error = "mixed CUDA and LLVM arrays!";
        }
    }

    if (unlikely(error))
        jitc_var_new_op_fail(error, op, n_dep, dep);

    bool is_float  = jitc_is_float(vt),
         is_uint = jitc_is_uint(vt),
         is_single = vt == VarType::Float32,
         is_valid = jitc_is_arithmetic(vt);

    const char *stmt = nullptr;

    // Used if the result produces a value value
    uint64_t lv = 0;

    /* Used if the result is simply the index of an input variable,
       or when the operation-specific implementation has created its own
       variable (in that case, it must set li_created=true below) */
    uint32_t li = 0;
    bool li_created = false;

    switch (op) {
        case JitOp::Not:
            is_valid = jitc_is_not_void(vt) && !is_float;
            if (literal) {
                lv = jitc_eval_literal([](auto value) { return eval_not(value); }, v[0]);
            } else if (backend == JitBackend::CUDA) {
                stmt = "not.$b0 $r0, $r1";
            } else {
                stmt = !jitc_is_float(vt)
                           ? "$r0 = xor <$w x $t1> $r1, $o0"
                           : "$r0_0 = bitcast <$w x $t1> $r1 to <$w x $b0>$n"
                             "$r0_1 = xor <$w x $b0> $r0_0, $o0$n"
                             "$r0 = bitcast <$w x $b0> $r0_1 to <$w x $t0>";
            }
            break;

        case JitOp::Neg:
            is_valid = jitc_is_arithmetic(vt);
            if (literal) {
                lv = jitc_eval_literal([](auto value) { return eval_neg(value); }, v[0]);
            } else if (backend == JitBackend::CUDA) {
                if (is_uint) {
                    stmt = type_size[vti] == 4 ?
                           "neg.s32 $r0, $r1" :
                           "neg.s64 $r0, $r1";
                } else {
                    stmt = is_single ? "neg.ftz.$t0 $r0, $r1"
                                     : "neg.$t0 $r0, $r1";
                }
            } else if (is_float) {
                if (jitc_llvm_version_major > 7)
                    stmt = "$r0 = fneg <$w x $t0> $r1";
                else
                    stmt = "$r0 = fsub <$w x $t0> zeroinitializer, $r1";
            } else {
                stmt = "$r0 = sub <$w x $t0> zeroinitializer, $r1";
            }
            break;

        case JitOp::Abs:
            if (is_uint) {
                li = dep[0];
            } else if (literal) {
                lv = jitc_eval_literal([](auto value) { return eval_abs(value); }, v[0]);
            } else if (backend == JitBackend::CUDA) {
                stmt = "abs.$t0 $r0, $r1";
            } else {
                if (is_float) {
                    uint64_t mask_value = ((uint64_t) 1 << (type_size[vti] * 8 - 1)) - 1;
                    uint32_t mask = jitc_var_new_literal(backend, vt, &mask_value, 1, 0);
                    uint32_t deps[2] = { dep[0], mask };
                    li = jitc_var_new_op(JitOp::And, 2, deps);
                    li_created = true;
                    jitc_var_dec_ref(mask);
                } else {
                    stmt = "$r0_0 = icmp slt <$w x $t0> $r1, zeroinitializer$n"
                           "$r0_1 = sub <$w x $t0> zeroinitializer, $r1$n"
                           "$r0 = select <$w x i1> $r0_0, <$w x $t1> $r0_1, <$w x $t1> $r1";
                }
            }
            break;

        case JitOp::Sqrt:
            is_valid = jitc_is_float(vt);
            if (literal) {
                lv = jitc_eval_literal([](auto value) { return eval_sqrt(value); }, v[0]);
            } else if (backend == JitBackend::CUDA) {
                stmt = is_single ? "sqrt.approx.ftz.$t0 $r0, $r1"
                                 : "sqrt.rn.$t0 $r0, $r1";
            } else {
                stmt = "$r0 = call <$w x $t0> @llvm.sqrt.v$w$a1(<$w x $t1> $r1)"
                       "$[declare <$w x $t0> @llvm.sqrt.v$w$a1(<$w x $t1>)$]";
            }
            break;

        case JitOp::Rcp:
            is_valid = jitc_is_float(vt);
            if (literal) {
                lv = jitc_eval_literal([](auto value) { return eval_rcp(value); }, v[0]);
            } else if (backend == JitBackend::CUDA) {
                stmt = is_single ? "rcp.approx.ftz.$t0 $r0, $r1"
                                 : "rcp.rn.$t0 $r0, $r1";
            } else {
                // TODO can do better here..
                float f1 = 1.f; double d1 = 1.0;
                uint32_t one = jitc_var_new_literal(backend, vt,
                                                    vt == VarType::Float32
                                                        ? (const void *) &f1
                                                        : (const void *) &d1,
                                                    1, 0);
                uint32_t deps[2] = { one, dep[0] };
                li = jitc_var_new_op(JitOp::Div, 2, deps);
                jitc_var_dec_ref(one);
                li_created = true;
            }
            break;

        case JitOp::Rsqrt:
            is_valid = jitc_is_float(vt);
            if (literal) {
                lv = jitc_eval_literal([](auto value) { return eval_rsqrt(value); }, v[0]);
            } else if (backend == JitBackend::CUDA) {
                stmt = is_single ? "rsqrt.approx.ftz.$t0 $r0, $r1"
                                 : "rcp.rn.$t0 $r0, $r1$n"
                                   "sqrt.rn.$t0 $r0, $r0";
            } else {
                // TODO can do better here..
                float f1 = 1.f; double d1 = 1.0;
                uint32_t one = jitc_var_new_literal(backend, vt,
                                                    vt == VarType::Float32
                                                        ? (const void *) &f1
                                                        : (const void *) &d1,
                                                    1, 0);
                uint32_t deps[2] = { one, dep[0] };
                uint32_t result_1 = jitc_var_new_op(JitOp::Div, 2, deps);
                li = jitc_var_new_op(JitOp::Sqrt, 1, &result_1);
                li_created = true;
                jitc_var_dec_ref(one);
                jitc_var_dec_ref(result_1);
            }
            break;

        case JitOp::Ceil:
            is_valid = jitc_is_float(vt);
            if (literal) {
                lv = jitc_eval_literal([](auto value) { return eval_ceil(value); }, v[0]);
            } else if (backend == JitBackend::CUDA) {
                stmt = "cvt.rpi.$t0.$t0 $r0, $r1";
            } else {
                stmt = "$r0 = call <$w x $t0> @llvm.ceil.v$w$a1(<$w x $t1> $r1)"
                       "$[declare <$w x $t0> @llvm.ceil.v$w$a1(<$w x $t1>)$]";
            }
            break;

        case JitOp::Floor:
            is_valid = jitc_is_float(vt);
            if (literal) {
                lv = jitc_eval_literal([](auto value) { return eval_floor(value); }, v[0]);
            } else if (backend == JitBackend::CUDA) {
                stmt = "cvt.rmi.$t0.$t0 $r0, $r1";
            } else {
                stmt = "$r0 = call <$w x $t0> @llvm.floor.v$w$a1(<$w x $t1> $r1)"
                       "$[declare <$w x $t0> @llvm.floor.v$w$a1(<$w x $t1>)$]";
            }
            break;

        case JitOp::Round:
            is_valid = jitc_is_float(vt);
            if (literal) {
                lv = jitc_eval_literal([](auto value) { return eval_round(value); }, v[0]);
            } else if (backend == JitBackend::CUDA) {
                stmt = "cvt.rni.$t0.$t0 $r0, $r1";
            } else {
                stmt = "$r0 = call <$w x $t0> @llvm.nearbyint.v$w$a1(<$w x $t1> $r1)"
                       "$[declare <$w x $t0> @llvm.nearbyint.v$w$a1(<$w x $t1>)$]";
            }
            break;

        case JitOp::Trunc:
            is_valid = jitc_is_float(vt);
            if (literal) {
                lv = jitc_eval_literal([](auto value) { return eval_trunc(value); }, v[0]);
            } else if (backend == JitBackend::CUDA) {
                stmt = "cvt.rzi.$t0.$t0 $r0, $r1";
            } else {
                stmt = "$r0 = call <$w x $t0> @llvm.trunc.v$w$a1(<$w x $t1> $r1)"
                       "$[declare <$w x $t0> @llvm.trunc.v$w$a1(<$w x $t1>)$]";
            }
            break;

        case JitOp::Exp2:
            is_valid = jitc_is_float(vt) && backend == JitBackend::CUDA;
            if (literal) {
                lv = jitc_eval_literal([](auto value) { return eval_exp2(value); }, v[0]);
            } else if (backend == JitBackend::CUDA) {
                stmt = "ex2.approx.ftz.$t0 $r0, $r1";
            }
            break;

        case JitOp::Log2:
            is_valid = jitc_is_float(vt) && backend == JitBackend::CUDA;
            if (literal) {
                lv = jitc_eval_literal([](auto value) { return eval_log2(value); }, v[0]);
            } else if (backend == JitBackend::CUDA) {
                stmt = "lg2.approx.ftz.$t1 $r0, $r1";
            }
            break;

        case JitOp::Sin:
            is_valid = jitc_is_float(vt) && backend == JitBackend::CUDA;
            if (literal) {
                lv = jitc_eval_literal([](auto value) { return eval_sin(value); }, v[0]);
            } else if (backend == JitBackend::CUDA) {
                stmt = "sin.approx.ftz.$t1 $r0, $r1";
            }
            break;

        case JitOp::Cos:
            is_valid = jitc_is_float(vt) && backend == JitBackend::CUDA;
            if (literal) {
                lv = jitc_eval_literal([](auto value) { return eval_cos(value); }, v[0]);
            } else if (backend == JitBackend::CUDA) {
                stmt = "cos.approx.ftz.$t1 $r0, $r1";
            }
            break;

        case JitOp::Popc:
            is_valid = jitc_is_arithmetic(vt) && !is_float;
            if (literal) {
                lv = jitc_eval_literal([](auto value) { return eval_popc(value); }, v[0]);
            } else if (backend == JitBackend::CUDA) {
                stmt = (vt == VarType::UInt32 || vt == VarType::Int32)
                           ? "popc.$b0 $r0, $r1"
                           : "popc.$b0 %r3, $r1$n"
                             "cvt.$t0.u32 $r0, %r3";
            } else {
                stmt = "$r0 = call <$w x $t0> @llvm.ctpop.v$w$a1(<$w x $t1> $r1)"
                       "$[declare <$w x $t0> @llvm.ctpop.v$w$a1(<$w x $t1>)$]";
            }
            break;

        case JitOp::Clz:
            is_valid = jitc_is_arithmetic(vt) && !is_float;
            if (literal) {
                lv = jitc_eval_literal([](auto value) { return eval_clz(value); }, v[0]);
            } else if (backend == JitBackend::CUDA) {
                stmt = (vt == VarType::UInt32 || vt == VarType::Int32)
                           ? "clz.$b0 $r0, $r1"
                           : "clz.$b0 %r3, $r1$n"
                             "cvt.$t0.u32 $r0, %r3";
            } else {
                stmt = "$r0 = call <$w x $t0> @llvm.ctlz.v$w$a1(<$w x $t1> $r1, i1 0)"
                       "$[declare <$w x $t0> @llvm.ctlz.v$w$a1(<$w x $t1>, i1)$]";
            }
            break;

        case JitOp::Ctz:
            is_valid = jitc_is_arithmetic(vt) && !is_float;
            if (literal) {
                lv = jitc_eval_literal([](auto value) { return eval_ctz(value); }, v[0]);
            } else if (backend == JitBackend::CUDA) {
                stmt = (vt == VarType::UInt32 || vt == VarType::Int32)
                           ? "brev.$b0 %r3, $r1$n"
                             "clz.$b0 $r0, %r3"
                           : "brev.$b0 %rd3, $r1$n"
                             "clz.$b0 %r3, %rd3$n"
                             "cvt.$t0.u32 $r0, %r3";
            } else {
                stmt = "$r0 = call <$w x $t0> @llvm.cttz.v$w$a1(<$w x $t1> $r1, i1 0)"
                       "$[declare <$w x $t0> @llvm.cttz.v$w$a1(<$w x $t1>, i1)$]";
            }
            break;

        case JitOp::Add:
            if (literal) {
                lv = jitc_eval_literal([](auto v0, auto v1) { return v0 + v1; },
                                     v[0], v[1]);
            } else if (literal_zero[0]) {
                li = dep[1];
            } else if (literal_zero[1]) {
                li = dep[0];
            } else if (backend == JitBackend::CUDA) {
                stmt = is_single ? "add.ftz.$t0 $r0, $r1, $r2"
                                 : "add.$t0 $r0, $r1, $r2";
            } else {
                stmt = is_float ? "$r0 = fadd <$w x $t0> $r1, $r2"
                                : "$r0 = add <$w x $t0> $r1, $r2";
            }
            break;

        case JitOp::Sub:
            if (literal) {
                lv = jitc_eval_literal([](auto v0, auto v1) { return v0 - v1; },
                                      v[0], v[1]);
            } else if (literal_zero[1]) {
                li = dep[0];
            } else if (backend == JitBackend::CUDA) {
                stmt = is_single ? "sub.ftz.$t0 $r0, $r1, $r2"
                                 : "sub.$t0 $r0, $r1, $r2";
            } else {
                stmt = is_float ? "$r0 = fsub <$w x $t0> $r1, $r2"
                                : "$r0 = sub <$w x $t0> $r1, $r2";
            }
            break;

        case JitOp::Mulhi:
            is_valid = jitc_is_arithmetic(vt) && !jitc_is_float(vt);
            if (literal) {
                lv = jitc_eval_literal([](auto v0, auto v1) { return eval_mulhi(v0, v1); },
                                       v[0], v[1]);
            } else if (literal_zero[0]) {
                li = dep[0];
            } else if (literal_zero[1]) {
                li = dep[1];
            } else if (backend == JitBackend::CUDA) {
                stmt = "mul.hi.$t0 $r0, $r1, $r2";
            } else {
                if (jitc_is_sint(vt))
                    stmt = "$r0_0 = sext <$w x $t1> $r1 to <$w x $T1>$n"
                           "$r0_1 = sext <$w x $t2> $r2 to <$w x $T2>$n"
                           "$r0_2 = sext <$w x $t3> $r3 to <$w x $T3>$n"
                           "$r0_3 = mul <$w x $T1> $r0_0, $r0_1$n"
                           "$r0_4 = lshr <$w x $T1> $r0_3, $r0_2$n"
                           "$r0 = trunc <$w x $T1> $r0_4 to <$w x $t1>";
                else
                    stmt = "$r0_0 = zext <$w x $t1> $r1 to <$w x $T1>$n"
                           "$r0_1 = zext <$w x $t2> $r2 to <$w x $T2>$n"
                           "$r0_2 = zext <$w x $t3> $r3 to <$w x $T3>$n"
                           "$r0_3 = mul <$w x $T1> $r0_0, $r0_1$n"
                           "$r0_4 = lshr <$w x $T1> $r0_3, $r0_2$n"
                           "$r0 = trunc <$w x $T1> $r0_4 to <$w x $t1>";

                uint64_t shift_amount = type_size[vti] * 8;

                uint32_t shift =
                    jitc_var_new_literal(backend, vt, &shift_amount, 1, 0);
                uint32_t deps[3] = { dep[0], dep[1], shift };
                li = jitc_var_new_stmt(backend, vt, stmt, 1, 3, deps);
                jitc_var_dec_ref(shift);
                li_created = true;
            }
            break;

        case JitOp::Mul:
            if (literal) {
                lv = jitc_eval_literal([](auto v0, auto v1) { return v0 * v1; },
                                       v[0], v[1]);
            } else if (literal_one[0] || (literal_zero[1] && jitc_is_int(vt))) {
                li = dep[1];
            } else if (literal_one[1] || (literal_zero[0] && jitc_is_int(vt))) {
                li = dep[0];
            } else if (is_uint && v[0]->is_literal() && jitc_is_pow2(v[0]->literal) && const_prop) {
                li = jitc_var_shift(backend, vt, JitOp::Shl, dep[1], v[0]->literal);
                li_created = true;
            } else if (is_uint && v[1]->is_literal() && jitc_is_pow2(v[1]->literal) && const_prop) {
                li = jitc_var_shift(backend, vt, JitOp::Shl, dep[0], v[1]->literal);
                li_created = true;
            } else if (backend == JitBackend::CUDA) {
                if (is_single)
                    stmt = "mul.ftz.$t0 $r0, $r1, $r2";
                else if (is_float)
                    stmt = "mul.$t0 $r0, $r1, $r2";
                else
                    stmt = "mul.lo.$t0 $r0, $r1, $r2";
            } else {
                stmt = is_float ? "$r0 = fmul <$w x $t0> $r1, $r2"
                                : "$r0 = mul <$w x $t0> $r1, $r2";
            }
            break;

        case JitOp::Div:
            if (literal) {
                lv = jitc_eval_literal([](auto v0, auto v1) { return eval_div(v0, v1); },
                                      v[0], v[1]);
            } else if (literal_one[1]) {
                li = dep[0];
            } else if (is_uint && v[1]->is_literal() && jitc_is_pow2(v[1]->literal) && const_prop) {
                li = jitc_var_shift(backend, vt, JitOp::Shr, dep[0], v[1]->literal);
                li_created = true;
            } else if (jitc_is_float(vt) && v[1]->is_literal() && const_prop) {
                uint32_t recip = jitc_var_new_op(JitOp::Rcp, 1, &dep[1]);
                uint32_t deps[2] = { dep[0], recip };
                li = jitc_var_new_op(JitOp::Mul, 2, deps);
                li_created = 1;
                jitc_var_dec_ref(recip);
            } else if (backend == JitBackend::CUDA) {
                if (is_single)
                    stmt = "div.approx.ftz.$t0 $r0, $r1, $r2";
                else if (is_float)
                    stmt = "div.rn.$t0 $r0, $r1, $r2";
                else
                    stmt = "div.$t0 $r0, $r1, $r2";
            } else {
                if (is_float)
                    stmt = "$r0 = fdiv <$w x $t0> $r1, $r2";
                else if (is_uint)
                    stmt = "$r0 = udiv <$w x $t0> $r1, $r2";
                else
                    stmt = "$r0 = sdiv <$w x $t0> $r1, $r2";
            }
            break;

        case JitOp::Mod:
            is_valid = jitc_is_arithmetic(vt) && !jitc_is_float(vt);
            if (literal) {
                lv = jitc_eval_literal([](auto v0, auto v1) { return eval_mod(v0, v1); },
                                      v[0], v[1]);
            } else if (backend == JitBackend::CUDA) {
                stmt = "rem.$t0 $r0, $r1, $r2";
            } else {
                if (is_uint)
                    stmt = "$r0 = urem <$w x $t0> $r1, $r2";
                else
                    stmt = "$r0 = srem <$w x $t0> $r1, $r2";
            }
            break;

        case JitOp::Min:
            if (literal) {
                lv = jitc_eval_literal([](auto v0, auto v1) { return std::min(v0, v1); },
                                      v[0], v[1]);
            } else if (backend == JitBackend::CUDA) {
                stmt = is_single ? "min.ftz.$t0 $r0, $r1, $r2"
                                 : "min.$t0 $r0, $r1, $r2";
            } else {
                if (is_float)
                    stmt = "$r0 = call <$w x $t0> @llvm.minnum.v$w$a1(<$w x $t1> $r1, <$w x $t2> $r2)"
                           "$[declare <$w x $t0> @llvm.minnum.v$w$a1(<$w x $t1>, <$w x $t2>)$]";
                else if (is_uint)
                    stmt = "$r0_0 = icmp ult <$w x $t0> $r1, $r2$n"
                           "$r0 = select <$w x i1> $r0_0, <$w x $t1> $r1, <$w x $t1> $r2";
                else
                    stmt = "$r0_0 = icmp slt <$w x $t0> $r1, $r2$n"
                           "$r0 = select <$w x i1> $r0_0, <$w x $t1> $r1, <$w x $t1> $r2";
            }
            break;

        case JitOp::Max:
            if (literal) {
                lv = jitc_eval_literal([](auto v0, auto v1) { return std::max(v0, v1); },
                                      v[0], v[1]);
            } else if (backend == JitBackend::CUDA) {
                stmt = is_single ? "max.ftz.$t0 $r0, $r1, $r2"
                                 : "max.$t0 $r0, $r1, $r2";
            } else {
                if (is_float)
                    stmt = "$r0 = call <$w x $t0> @llvm.maxnum.v$w$a1(<$w x $t1> $r1, <$w x $t2> $r2)"
                           "$[declare <$w x $t0> @llvm.maxnum.v$w$a1(<$w x $t1>, <$w x $t2>)$]";
                else if (is_uint)
                    stmt = "$r0_0 = icmp ugt <$w x $t0> $r1, $r2$n"
                           "$r0 = select <$w x i1> $r0_0, <$w x $t1> $r1, <$w x $t1> $r2";
                else
                    stmt = "$r0_0 = icmp sgt <$w x $t0> $r1, $r2$n"
                           "$r0 = select <$w x i1> $r0_0, <$w x $t1> $r1, <$w x $t1> $r2";
            }
            break;

        case JitOp::Shr:
            is_valid = jitc_is_arithmetic(vt) && !jitc_is_float(vt);
            if (literal) {
                lv = jitc_eval_literal([](auto v0, auto v1) { return eval_shr(v0, v1); },
                                      v[0], v[1]);
            } else if (literal_zero[0] || literal_zero[1]) {
                li = dep[0];
            } else if (backend == JitBackend::CUDA) {
                if (is_uint)
                    stmt = (vt == VarType::UInt32)
                               ? "shr.$b0 $r0, $r1, $r2"
                               : "cvt.u32.$t2 %r3, $r2$n"
                                 "shr.$b0 $r0, $r1, %r3";
                else
                    stmt = (vt == VarType::Int32)
                               ? "shr.$t0 $r0, $r1, $r2"
                               : "cvt.u32.$t2 %r3, $r2$n"
                                 "shr.$t0 $r0, $r1, %r3";
            } else {
                stmt = is_uint ? "$r0 = lshr <$w x $t0> $r1, $r2"
                               : "$r0 = ashr <$w x $t0> $r1, $r2";
            }
            break;

        case JitOp::Shl:
            is_valid = jitc_is_arithmetic(vt) && !jitc_is_float(vt);
            if (literal) {
                lv = jitc_eval_literal([](auto v0, auto v1) { return eval_shl(v0, v1); },
                                      v[0], v[1]);
            } else if (literal_zero[0] || literal_zero[1]) {
                li = dep[0];
            } else if (backend == JitBackend::CUDA) {
                stmt = (vt == VarType::UInt32 || vt == VarType::Int32)
                           ? "shl.$b0 $r0, $r1, $r2"
                           : "cvt.u32.$t2 %r3, $r2$n"
                             "shl.$b0 $r0, $r1, %r3";
            } else {
                stmt = "$r0 = shl <$w x $t0> $r1, $r2";
            }
            break;

        case JitOp::And:
            is_valid = jitc_is_not_void(vt);
            literal &= v[0]->type == v[1]->type;
            if (literal) {
                lv = jitc_eval_literal([](auto v0, auto v1) { return eval_and(v0, v1); },
                                       v[0], v[1]);
            } else if (((VarType) v[0]->type == VarType::Bool && literal_one[0]) ||
                        (literal_zero[1] && v[0]->type == v[1]->type)) {
                li = dep[1];
            } else if (((VarType) v[1]->type == VarType::Bool && literal_one[1]) || literal_zero[0]) {
                li = dep[0];
            } else if (dep[0] == dep[1]) {
                li = dep[0];
            } else if (backend == JitBackend::CUDA) {
                stmt = ((VarType) v[1]->type == VarType::Bool && vt != VarType::Bool)
                           ? "selp.$b0 $r0, $r1, 0, $r2"
                           : "and.$b0 $r0, $r1, $r2";
            } else {
                if ((VarType) v[1]->type == VarType::Bool && vt != VarType::Bool) {
                    stmt = "$r0 = select <$w x $t2> $r2, <$w x $t1> $r1, <$w x $t1> zeroinitializer";
                } else {
                    stmt = !is_float
                               ? "$r0 = and <$w x $t1> $r1, $r2"
                               : "$r0_0 = bitcast <$w x $t1> $r1 to <$w x $b0>$n"
                                 "$r0_1 = bitcast <$w x $t2> $r2 to <$w x $b0>$n"
                                 "$r0_2 = and <$w x $b0> $r0_0, $r0_1$n"
                                 "$r0 = bitcast <$w x $b0> $r0_2 to <$w x $t0>";
                }
            }
            break;

        case JitOp::Or:
            is_valid = jitc_is_not_void(vt);
            literal &= v[0]->type == v[1]->type;
            if (literal) {
                lv = jitc_eval_literal([](auto v0, auto v1) { return eval_or(v0, v1); },
                                      v[0], v[1]);
            } else if ((vt == VarType::Bool && literal_one[0]) || literal_zero[1]) {
                li = dep[0];
            } else if ((vt == VarType::Bool && literal_one[1]) ||
                       (literal_zero[0] && v[0]->type == v[1]->type)) {
                li = dep[1];
            } else if (dep[0] == dep[1]) {
                li = dep[0];
            } else if (backend == JitBackend::CUDA) {
                stmt = ((VarType) v[1]->type == VarType::Bool && vt != VarType::Bool)
                           ? "selp.$b0 $r0, -1, $r1, $r2"
                           : "or.$b0 $r0, $r1, $r2";
            } else {
                if ((VarType) v[1]->type == VarType::Bool && vt != VarType::Bool) {
                    stmt = "$r0_0 = sext <$w x $t2> $r2 to <$w x $b0>$n"
                           "$r0_1 = bitcast <$w x $t1> $r1 to <$w x $b0>$n"
                           "$r0_2 = or <$w x $b0> $r0_0, $r0_1$n"
                           "$r0 = bitcast <$w x $b0> $r0_2 to <$w x $t0>";
                } else {
                    stmt = !is_float
                               ? "$r0 = or <$w x $t1> $r1, $r2"
                               : "$r0_0 = bitcast <$w x $t1> $r1 to <$w x $b0>$n"
                                 "$r0_1 = bitcast <$w x $t2> $r2 to <$w x $b0>$n"
                                 "$r0_2 = or <$w x $b0> $r0_0, $r0_1$n"
                                 "$r0 = bitcast <$w x $b0> $r0_2 to <$w x $t0>";
                }
            }
            break;

        case JitOp::Xor:
            is_valid = jitc_is_not_void(vt);
            if (literal) {
                lv = jitc_eval_literal([](auto v0, auto v1) { return eval_xor(v0, v1); },
                                      v[0], v[1]);
            } else if (literal_zero[0]) {
                li = dep[1];
            } else if (literal_zero[1]) {
                li = dep[0];
            } else if (backend == JitBackend::CUDA) {
                stmt = "xor.$b0 $r0, $r1, $r2";
            } else {
                stmt = !is_float
                           ? "$r0 = xor <$w x $t1> $r1, $r2"
                           : "$r0_0 = bitcast <$w x $t1> $r1 to <$w x $b0>$n"
                             "$r0_1 = bitcast <$w x $t2> $r2 to <$w x $b0>$n"
                             "$r0_2 = xor <$w x $b0> $r0_0, $r0_1$n"
                             "$r0 = bitcast <$w x $b0> $r0_2 to <$w x $t0>";
            }
            break;

        case JitOp::Eq:
            is_valid = jitc_is_not_void(vt);
            vtr = VarType::Bool;
            if (literal) {
                lv = jitc_eval_literal([](auto v0, auto v1) { return v0 == v1; },
                                      v[0], v[1]);
            } else if (backend == JitBackend::CUDA) {
                if (vt == VarType::Bool)
                    stmt = "xor.$t1 $r0, $r1, $r2$n"
                           "not.$t1 $r0, $r0";
                else
                    stmt = "setp.eq.$t1 $r0, $r1, $r2";
            } else {
                stmt = is_float ? "$r0 = fcmp oeq <$w x $t1> $r1, $r2"
                                : "$r0 = icmp eq <$w x $t1> $r1, $r2";
            }
            break;

        case JitOp::Neq:
            is_valid = jitc_is_not_void(vt);
            vtr = VarType::Bool;
            if (literal) {
                lv = jitc_eval_literal([](auto v0, auto v1) { return v0 != v1; },
                                      v[0], v[1]);
            } else if (backend == JitBackend::CUDA) {
                if (vt == VarType::Bool)
                    stmt = "xor.$t1 $r0, $r1, $r2";
                else
                    stmt = "setp.ne.$t1 $r0, $r1, $r2";
            } else {
                stmt = is_float ? "$r0 = fcmp one <$w x $t1> $r1, $r2"
                                : "$r0 = icmp ne <$w x $t1> $r1, $r2";
            }
            break;

        case JitOp::Lt:
            vtr = VarType::Bool;
            if (literal) {
                lv = jitc_eval_literal([](auto v0, auto v1) { return v0 < v1; },
                                      v[0], v[1]);
            } else if (backend == JitBackend::CUDA) {
                stmt = is_uint ? "setp.lo.$t1 $r0, $r1, $r2"
                               : "setp.lt.$t1 $r0, $r1, $r2";
            } else {
                if (is_float)
                    stmt = "$r0 = fcmp olt <$w x $t1> $r1, $r2";
                else if (is_uint)
                    stmt = "$r0 = icmp ult <$w x $t1> $r1, $r2";
                else
                    stmt = "$r0 = icmp slt <$w x $t1> $r1, $r2";
            }
            break;

        case JitOp::Le:
            vtr = VarType::Bool;
            if (literal) {
                lv = jitc_eval_literal([](auto v0, auto v1) { return v0 <= v1; },
                                      v[0], v[1]);
            } else if (backend == JitBackend::CUDA) {
                stmt = is_uint ? "setp.ls.$t1 $r0, $r1, $r2"
                               : "setp.le.$t1 $r0, $r1, $r2";
            } else {
                if (is_float)
                    stmt = "$r0 = fcmp ole <$w x $t1> $r1, $r2";
                else if (is_uint)
                    stmt = "$r0 = icmp ule <$w x $t1> $r1, $r2";
                else
                    stmt = "$r0 = icmp sle <$w x $t1> $r1, $r2";
            }
            break;

        case JitOp::Gt:
            vtr = VarType::Bool;
            if (literal) {
                lv = jitc_eval_literal([](auto v0, auto v1) { return v0 > v1; },
                                      v[0], v[1]);
            } else if (backend == JitBackend::CUDA) {
                stmt = is_uint ? "setp.hi.$t1 $r0, $r1, $r2"
                               : "setp.gt.$t1 $r0, $r1, $r2";
            } else {
                if (is_float)
                    stmt = "$r0 = fcmp ogt <$w x $t1> $r1, $r2";
                else if (is_uint)
                    stmt = "$r0 = icmp ugt <$w x $t1> $r1, $r2";
                else
                    stmt = "$r0 = icmp sgt <$w x $t1> $r1, $r2";
            }
            break;

        case JitOp::Ge:
            vtr = VarType::Bool;
            if (literal) {
                lv = jitc_eval_literal([](auto v0, auto v1) { return v0 >= v1; },
                                      v[0], v[1]);
            } else if (backend == JitBackend::CUDA) {
                stmt = is_uint ? "setp.hs.$t1 $r0, $r1, $r2"
                               : "setp.ge.$t1 $r0, $r1, $r2";
            } else {
                if (is_float)
                    stmt = "$r0 = fcmp oge <$w x $t1> $r1, $r2";
                else if (is_uint)
                    stmt = "$r0 = icmp uge <$w x $t1> $r1, $r2";
                else
                    stmt = "$r0 = icmp sge <$w x $t1> $r1, $r2";
            }
            break;

        case JitOp::Fmadd:
            if (literal) {
                lv = jitc_eval_literal(
                    [](auto v0, auto v1, auto v2) {
                        return eval_fma(v0, v1, v2);
                    },
                    v[0], v[1], v[2]);
            } else if (literal_one[0]) {
                uint32_t deps[2] = { dep[1], dep[2] };
                li = jitc_var_new_op(JitOp::Add, 2, deps);
                li_created = true;
            } else if (literal_one[1]) {
                uint32_t deps[2] = { dep[0], dep[2] };
                li = jitc_var_new_op(JitOp::Add, 2, deps);
                li_created = true;
            } else if (literal_zero[2]) {
                uint32_t deps[2] = { dep[0], dep[1] };
                li = jitc_var_new_op(JitOp::Mul, 2, deps);
                li_created = true;
            } else if (literal_zero[0] && literal_zero[1]) {
                li = dep[2];
            } else if (backend == JitBackend::CUDA) {
                if (is_float) {
                    stmt = is_single ? "fma.rn.ftz.$t0 $r0, $r1, $r2, $r3"
                                     : "fma.rn.$t0 $r0, $r1, $r2, $r3";
                } else {
                    stmt = "mad.lo.$t0 $r0, $r1, $r2, $r3";
                }
            } else {
                if (is_float) {
                    stmt = "$r0 = call <$w x $t0> @llvm.fma.v$w$a1(<$w x $t1> $r1, <$w x $t2> $r2, <$w x $t3> $r3)"
                           "$[declare <$w x $t0> @llvm.fma.v$w$a1(<$w x $t1>, <$w x $t2>, <$w x $t3>)$]";
                } else {
                    stmt = "$r0_0 = mul <$w x $t0> $r1, $r2$n"
                           "$r0 = add <$w x $t0> $r0_0, $r3";
                }
            }
            break;

        case JitOp::Select:
            is_valid = (VarType) v[0]->type == VarType::Bool &&
                       v[1]->type == v[2]->type;
            if (literal_one[0]) {
                li = dep[1];
            } else if (literal_zero[0]) {
                li = dep[2];
            } else if (literal) {
                jitc_fail("jit_var_new_op(): select: internal error!");
            } else if (dep[1] == dep[2]) {
                li = dep[1];
            } else if (backend == JitBackend::CUDA) {
                if ((VarType) v[1]->type != VarType::Bool) {
                    stmt = literal_zero[2] ? "selp.$b0 $r0, $r2, 0, $r1"
                                           : "selp.$t0 $r0, $r2, $r3, $r1";
                } else {
                    // r0 = (r1 && r2) || (!r1 && r3)
                    stmt = "and.pred %p3, $r1, $r2$n"
                           "and.pred %p2, !$r1, $r3$n"
                           "or.pred $r0, %p2, %p3";
                }
            } else {
                stmt = literal_zero[2] ? "$r0 = select <$w x $t1> $r1, <$w x "
                                         "$t2> $r2, <$w x $t2> zeroinitializer"
                                       : "$r0 = select <$w x $t1> $r1, <$w x "
                                         "$t2> $r2, <$w x $t3> $r3";
            }
            break;

        default:
            error = "operation not supported!";
    }

    if (unlikely(error || !is_valid)) {
        if (!error)
            error = "invalid input operands";
        jitc_var_new_op_fail(error, op, n_dep, dep);
    }

    uint32_t result;
    if (li) {
        result = jitc_var_resize(li, size);
        if (li_created)
            jitc_var_dec_ref(li);
    } else {
        if (dirty) {
            jitc_eval(thread_state(backend));
            for (uint32_t i = 0; i < n_dep; ++i) {
                v[i] = jitc_var(dep[i]);
                if (v[i]->is_dirty())
                    error = "variable remains dirty following evaluation!";
            }
        }

        Variable v2;
        v2.size = size;
        v2.type = (uint32_t) vtr;
        v2.backend = (uint32_t) backend;
        v2.placeholder = placeholder;

        if (literal) {
            v2.kind = (uint32_t) VarKind::Literal;
            v2.literal = lv;
        } else {
            v2.kind = (uint32_t) VarKind::Stmt;
            v2.stmt = (char *) stmt;
            for (uint32_t i = 0; i < n_dep; ++i) {
                v2.dep[i] = dep[i];
                jitc_var_inc_ref(dep[i], v[i]);
            }
        }

        result = jitc_var_new(v2);
    }

    if (unlikely(std::max(state.log_level_stderr, state.log_level_callback) >=
                 LogLevel::Debug)) {
        var_buffer.clear();
        var_buffer.fmt("jit_var_new_op(%s r%u <- %s ",
                       type_name[(uint32_t) vtr], result, op_name[(int) op]);
        for (uint32_t i = 0; i < n_dep; ++i)
            var_buffer.fmt("r%u%s", dep[i], i + 1 < n_dep ? ", " : ")");
        if (literal)
            var_buffer.put(": literal");
        else if (li)
            var_buffer.put(": simplified");
        jitc_log(Debug, "%s", var_buffer.get());
    }

    return result;
}

JIT_NOINLINE uint32_t jitc_var_new_op_fail(const char *error, JitOp op, uint32_t n_dep, const uint32_t *dep) {
    switch (n_dep) {
        case 1:
            jitc_raise("jit_var_new_op(%s, r%u): %s", op_name[(int) op], dep[0],
                      error);
        case 2:
            jitc_raise("jit_var_new_op(%s, r%u, r%u): %s", op_name[(int) op],
                      dep[0], dep[1], error);
        case 3:
            jitc_raise("jit_var_new_op(%s, r%u, r%u, r%u): %s", op_name[(int) op],
                      dep[0], dep[1], dep[2], error);
        case 4:
            jitc_raise("jit_var_new_op(%s, r%u, r%u, r%u, r%u): %s",
                      op_name[(int) op], dep[0], dep[1], dep[2], dep[3], error);
        default:
            jitc_fail("jit_var_new_op(): invalid number of arguments!");
    }
}

uint32_t jitc_var_new_cast(uint32_t index, VarType target_type, int reinterpret) {
    if (index == 0)
        return 0;

    Variable *v = jitc_var(index);
    const JitBackend backend = (JitBackend) v->backend;
    const VarType source_type = (VarType) v->type;

    if (source_type == target_type) {
        jitc_var_inc_ref(index);
        return index;
    }

    bool source_bool = source_type == VarType::Bool,
         target_bool = target_type == VarType::Bool,
         source_float = jitc_is_float(source_type),
         target_float = jitc_is_float(target_type);

    uint32_t source_size =
                 source_bool ? 0 : type_size[(uint32_t) source_type],
             target_size =
                 target_bool ? 0 : type_size[(uint32_t) target_type];

    if (reinterpret && source_size != target_size) {
        jitc_raise("jit_var_new_cast(): reinterpret cast between types of "
                   "different size!");
    } else if (source_size == target_size && !source_float && !target_float) {
        reinterpret = 1;
    }

    if (v->is_dirty()) {
        jitc_eval(thread_state(backend));
        v = jitc_var(index);
        if (unlikely(v->is_dirty()))
            jitc_fail("jit_var_new_cast(): variable remains dirty after evaluation!");
    }

    if (v->is_literal() && (jitc_flags() & (uint32_t) JitFlag::ConstProp)) {
        uint64_t value;
        if (reinterpret) {
            value = v->literal;
        } else {
            value = jitc_eval_literal([target_type](auto value) -> uint64_t {
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
                    default: jitc_fail("jit_var_new_cast(): unsupported variable type!");
                }
            }, v);
        }
        return jitc_var_new_literal((JitBackend) v->backend, target_type,
                                    &value, v->size, 0);
    } else {
        bool source_uint   = jitc_is_uint(source_type),
             target_uint   = jitc_is_uint(target_type);

        const char *stmt = nullptr;
        if (reinterpret) {
            stmt = backend == JitBackend::CUDA
                       ? "mov.$b0 $r0, $r1"
                       : "$r0 = bitcast <$w x $t1> $r1 to <$w x $t0>";
        } else if (target_bool) {
            if (backend == JitBackend::CUDA) {
                stmt = source_float ? "setp.ne.$t1 $r0, $r1, 0.0"
                                    : "setp.ne.$t1 $r0, $r1, 0";
            } else {
                stmt = source_float ? "$r0 = fcmp one <$w x $t1> $r1, zeroinitializer"
                                    : "$r0 = icmp ne <$w x $t1> $r1, zeroinitializer";
            }
        } else if (source_bool) {
            if (backend == JitBackend::CUDA) {
                stmt = target_float ? "selp.$t0 $r0, 1.0, 0.0, $r1"
                                    : "selp.$t0 $r0, 1, 0, $r1";
            } else {
                if (target_float)
                    stmt = "$r0_1 = insertelement <$w x $t0> undef, $t0 1.0, i32 0$n"
                           "$r0_2 = shufflevector <$w x $t0> $r0_1, <$w x $t0> undef, <$w x i32> zeroinitializer$n"
                           "$r0 = select <$w x $t1> $r1, <$w x $t0> $r0_2, <$w x $t0> zeroinitializer";
                else
                    stmt = "$r0_1 = insertelement <$w x $t0> undef, $t0 1, i32 0$n"
                           "$r0_2 = shufflevector <$w x $t0> $r0_1, <$w x $t0> undef, <$w x i32> zeroinitializer$n"
                           "$r0 = select <$w x $t1> $r1, <$w x $t0> $r0_2, <$w x $t0> zeroinitializer";
            }
        } else if (!source_float && target_float) {
            if (backend == JitBackend::CUDA) {
                stmt = "cvt.rn.$t0.$t1 $r0, $r1";
            } else {
                stmt = source_uint ? "$r0 = uitofp <$w x $t1> $r1 to <$w x $t0>"
                                   : "$r0 = sitofp <$w x $t1> $r1 to <$w x $t0>";
            }
        } else if (source_float && !target_float) {
            if (backend == JitBackend::CUDA) {
                stmt = "cvt.rzi.$t0.$t1 $r0, $r1";
            } else {
                stmt = target_uint ? "$r0 = fptoui <$w x $t1> $r1 to <$w x $t0>"
                                   : "$r0 = fptosi <$w x $t1> $r1 to <$w x $t0>";
            }
        } else if (source_float && target_float) {
            if (target_size < source_size) {
                stmt = backend == JitBackend::CUDA
                           ? "cvt.rn.$t0.$t1 $r0, $r1"
                           : "$r0 = fptrunc <$w x $t1> $r1 to <$w x $t0>";
            } else {
                stmt = backend == JitBackend::CUDA
                           ? "cvt.$t0.$t1 $r0, $r1"
                           : "$r0 = fpext <$w x $t1> $r1 to <$w x $t0>";

            }
        } else if (!source_float && !target_float) {
            if (backend == JitBackend::CUDA) {
                stmt = "cvt.$t0.$t1 $r0, $r1";
            } else {
                stmt = target_size < source_size
                         ? "$r0 = trunc <$w x $t1> $r1 to <$w x $t0>"
                         : (source_uint
                                ? "$r0 = zext <$w x $t1> $r1 to <$w x $t0>"
                                : "$r0 = sext <$w x $t1> $r1 to <$w x $t0>");
            }
        } else {
            jitc_raise("Unsupported conversion!");
        }

        Variable v2;
        v2.kind = (uint32_t) VarKind::Stmt;
        v2.size = v->size;
        v2.type = (uint32_t) target_type;
        v2.backend = (uint32_t) backend;
        v2.stmt = (char *) stmt;
        v2.dep[0] = index;
        v2.placeholder = v->placeholder;
        jitc_var_inc_ref(index, v);
        uint32_t result = jitc_var_new(v2);

        jitc_log(Debug, "jit_var_new_cast(%s r%u <- %s r%u)",
                 type_name[(int) target_type], result,
                 type_name[(int) source_type], index);

        return result;
    }
}

static uint32_t jitc_scatter_gather_index(uint32_t source, uint32_t index) {
    const Variable *v_source = jitc_var(source),
                   *v_index = jitc_var(index);

    VarType source_type = (VarType) v_index->type;
    if (!jitc_is_uint(source_type) && !jitc_is_sint(source_type))
        jitc_raise("jit_scatter_gather_index(): expected an integer array as scatter/gather index");

    VarType target_type = VarType::UInt32;
    // Need 64 bit indices for upper 2G entries (gather indices are signed in LLVM)
    if (v_source->size > 0x7fffffff && (JitBackend) v_source->backend == JitBackend::LLVM)
        target_type = VarType::UInt64;

    return jitc_var_new_cast(index, target_type, 0);
}

/// Change all indices/counters in an expression tree to 'new_index'
static uint32_t jitc_var_reindex(uint32_t var_index, uint32_t new_index,
                                 uint32_t size) {
    Variable *v = jitc_var(var_index);

    if (v->is_data())
        return 0; // evaluated variable, give up

    if (v->extra) {
        Extra &e = state.extra[var_index];
        if (e.n_dep || e.callback || e.vcall_buckets || e.assemble)
            return 0; // "complicated" variable, give up
    }

    if (v->is_literal()) {
        jitc_var_inc_ref(var_index, v);
        return var_index;
    }

    Ref dep[4];
    bool rebuild = false;
    for (uint32_t i = 0; i < 4; ++i) {
        uint32_t index_2 = v->dep[i];
        if (!index_2)
            continue;
        dep[i] = steal(jitc_var_reindex(index_2, new_index, size));
        if (!dep[i])
            return 0; // recursive call failed, give up
        rebuild |= dep[i] != index_2;
        if (rebuild)
            v = jitc_var(var_index);
    }

    const char *counter_str = (JitBackend) v->backend == JitBackend::CUDA
                                  ? (char *) "mov.u32 $r0, %r0"
                                  : jitc_llvm_counter_str;

    if (rebuild) {
        Variable v2;
        v2.kind = v->kind;
        v2.size = size;
        v2.type = v->type;
        v2.backend = v->backend;
        v2.placeholder = v->placeholder;
        if (!v->free_stmt) {
            v2.stmt = v->stmt;
        } else {
            v2.stmt = strdup(v->stmt);
            v2.free_stmt = 1;
        }
        for (uint32_t i = 0; i < 4; ++i) {
            v2.dep[i] = dep[i];
            jitc_var_inc_ref(dep[i]);
        }
        return jitc_var_new(v2);
    } else if (v->is_stmt() && strcmp(v->stmt, counter_str) == 0) {
        jitc_var_inc_ref(new_index);
        return new_index;
    } else {
        jitc_var_inc_ref(var_index, v);
        return var_index;
    }
}

uint32_t jitc_var_new_gather(uint32_t source, uint32_t index_, uint32_t mask_) {
    if (index_ == 0)
        return 0;
    else if (unlikely(source == 0 || mask_ == 0))
        jitc_raise("jit_var_new_gather(source=%u, index=%u, mask=%u): uninitialized arguments!", source, index_, mask_);

    const Variable *v_source = jitc_var(source),
                   *v_index = jitc_var(index_),
                   *v_mask = jitc_var(mask_);

    uint32_t vti = v_source->type;
    uint32_t size = std::max(v_index->size, v_mask->size);
    JitBackend backend = (JitBackend) v_source->backend;
    bool v_source_value = v_source->is_literal();

    if (v_source->placeholder)
        jitc_raise("jit_var_new_gather(): cannot gather from a placeholder variable!");

    // Don't perform the gather operation if it is always masked
    if (v_mask->is_literal() && v_mask->literal == 0) {
        uint64_t value = 0;
        uint32_t result = jitc_var_new_literal(backend, (VarType) vti, &value, size, 0);
        jitc_log(Debug,
                 "jit_var_new_gather(r%u <- r%u[r%u] if r%u): elided, always masked",
                 result, source, index_, mask_);
        return result;
    }

    // Don't perform the gather operation if the inputs are trivial / can be re-indexed
    Ref trivial_source = steal(jitc_var_reindex(source, index_, size));
    if (trivial_source) {
        // Temporarily hold an extra reference to prevent 'jitc_var_resize' from changing 'source'
        Ref unused = borrow(trivial_source);
        Ref tmp = steal(jitc_var_resize(trivial_source, size));
        uint32_t deps[2] = { (uint32_t) tmp, mask_ };
        uint32_t result = jitc_var_new_op(JitOp::And, 2, deps);

        jitc_log(Debug, "jit_var_new_gather(%s r%u <- r%u[r%u] if r%u): elided, %s source",
                 type_name[vti], result, source, index_, mask_,
                 v_source_value ? "value" : "scalar");

        return result;
    }

    Ref mask  = steal(jitc_var_mask_apply(mask_, size)),
        index = steal(jitc_scatter_gather_index(source, index_));

    // Location of variables may have changed
    v_source = jitc_var(source);
    v_index = jitc_var(index);
    v_mask = jitc_var(mask);

    if (v_source->is_dirty() || v_index->is_dirty() || v_mask->is_dirty()) {
        jitc_eval(thread_state(backend));

        // Location of variables may have changed
        v_source = jitc_var(source);
        v_index = jitc_var(index);
        v_mask = jitc_var(mask);

        if (unlikely(v_source->is_dirty() || v_index->is_dirty() || v_mask->is_dirty()))
            jitc_fail("jit_var_new_gather(): variable remains dirty after evaluation!");
    }

    bool unmasked = v_mask->is_literal() && v_mask->literal == 1;
    bool index_zero = v_index->is_literal() && v_index->literal == 0;

    VarType vt = (VarType) v_source->type;

    // Create a pointer + reference, invalidates the v_* variables
    Ref ptr = steal(jitc_var_new_pointer(backend, jitc_var_ptr(source), source, 0));

    uint32_t dep[4] = { ptr, index, mask, 0 };
    uint32_t n_dep = 3;

    uint32_t debug_print = 0;

#if 0
    // Debug out-of-bounds issues
    if (backend == JitBackend::CUDA) {
        char tmp[128];
        snprintf(tmp, sizeof(tmp),
                 "setp.eq.u64 $r0, $r1, 0$n" // base pointer issue
                 "setp.ge.or.$t2 $r0, $r2, %u, $r0$n" // overflow
                 "and.pred $r0, $r0, $r3",
                 v_source->size);
        uint32_t is_null =
            jitc_var_new_stmt(backend, VarType::Bool, tmp, 0, 3, dep);
        snprintf(
            tmp, sizeof(tmp),
            "Issue with gather: r%u <- r%u[r%u] if r%u: ptr=%%p, index=%%u, max_size=%u\n",
            state.variable_index, source, index_, mask_, v_source->size);
        debug_print = jitc_var_printf(backend, is_null, tmp, 2, dep);
        jitc_var_dec_ref(is_null);
    }
#endif

    const char *stmt;
    if (backend == JitBackend::CUDA) {
        if (!index_zero) {
            if (vt != VarType::Bool) {
                if (unmasked)
                    stmt = "mad.wide.$t2 %rd3, $r2, $s0, $r1$n"
                           "ld.global.nc.$t0 $r0, [%rd3]";
                else
                    stmt = ""; // Will use assemble callback set below
            } else {
                if (unmasked)
                    stmt = "mad.wide.$t2 %rd3, $r2, $s0, $r1$n"
                           "ld.global.nc.u8 %w0, [%rd3]$n"
                           "setp.ne.u16 $r0, %w0, 0";
                else
                    stmt = ""; // Will use assemble callback set below
            }
        } else {
            if (vt != VarType::Bool) {
                if (unmasked)
                    stmt = "ld.global.nc.$t0 $r0, [$r1]";
                else
                    stmt = ""; // Will use assemble callback set below
            } else {
                if (unmasked)
                    stmt = "ld.global.nc.u8 %w0, [$r1]$n"
                           "setp.ne.u16 $r0, %w0, 0";
                else
                    stmt = ""; // Will use assemble callback set below
            }
        }

        if (unmasked) {
            dep[2] = 0;
            n_dep = 2;
        }

        if (debug_print)
            dep[n_dep++] = debug_print;
    } else {
        if (vt != VarType::Bool && vt != VarType::UInt8 && vt != VarType::Int8) {
            stmt = "$r0_0 = bitcast $<i8*$> $r1 to $<$t0*$>$n"
                   "$r0_1 = getelementptr $t0, $<$t0*$> $r0_0, <$w x $t2> $r2$n"
                   "$r0 = call <$w x $t0> @llvm.masked.gather.v$w$a0(<$w x $t0*> $r0_1, i32 $s0, <$w x $t3> $r3, <$w x $t0> zeroinitializer)"
                   "$[declare <$w x $t0> @llvm.masked.gather.v$w$a0(<$w x $t0*>, i32, <$w x $t3>, <$w x $t0>)$]";
        } else {
            stmt = "$r0_0 = getelementptr i8, $<i8*$> $r1, <$w x $t2> $r2$n"
                   "$r0_1 = bitcast <$w x i8*> $r0_0 to <$w x i32*>$n"
                   "$r0_2 = call <$w x i32> @llvm.masked.gather.v$wi32(<$w x i32*> $r0_1, i32 $s0, <$w x $t3> $r3, <$w x i32> zeroinitializer)$n"
                   "$r0 = trunc <$w x i32> $r0_2 to <$w x $t0>"
                   "$[declare <$w x i32> @llvm.masked.gather.v$wi32(<$w x i32*>, i32, <$w x $t3>, <$w x i32>)$]";
        }
    }

    uint32_t result = jitc_var_new_stmt(backend, vt, stmt, 1, n_dep, dep);
    jitc_log(Debug,
             "jit_var_new_gather(r%u <- r%u[r%u] if r%u, via ptr r%u)",
             result, source, (uint32_t) index, (uint32_t) mask, (uint32_t) ptr);

    if (backend == JitBackend::CUDA && !unmasked && strcmp(stmt, "") == 0) {
        jitc_var(result)->extra = 1;

        Extra &extra = state.extra[result];
        extra.assemble = [](const Variable *v, const Extra &/*extra*/) {
            const Variable *v_ptr   = jitc_var(v->dep[0]);
            const Variable *v_index = jitc_var(v->dep[1]);
            const Variable *v_mask  = jitc_var(v->dep[2]);

            uint32_t v_i     = v->reg_index;
            uint32_t ptr_i   = v_ptr->reg_index;
            uint32_t index_i = v_index->reg_index;
            uint32_t mask_i  = v_mask->reg_index;

            const char *v_tp     = type_prefix[v->type];
            const char *ptr_tp   = type_prefix[v_ptr->type];
            const char *index_tp = type_prefix[v_index->type];

            bool index_zero = v_index->is_literal() && v_index->literal == 0;
            bool is_bool = v->type == (uint32_t) VarType::Bool;

            if (!index_zero) {
                buffer.fmt("    mad.wide.%s %%rd3, %s%u, %s, %s%u;\n",
                           type_name_ptx[v_index->type], index_tp, index_i,
                           type_size_str[v->type], ptr_tp, ptr_i);
            }

            buffer.fmt("    @!%%p%u bra l_%u_masked;\n", mask_i, v_i);

            if (!index_zero) {
                if (!is_bool) {
                    buffer.fmt("    ld.global.nc.%s %s%u, [%%rd3];\n",
                               type_name_ptx[v->type], v_tp, v_i);
                } else {
                    buffer.put("    ld.global.nc.u8 %w0, [%rd3];\n");
                }
            } else {
                if (!is_bool) {
                    buffer.fmt("    ld.global.nc.%s %s%u, [%s%u];\n",
                               type_name_ptx[v->type], v_tp, v_i, ptr_tp, ptr_i);
                } else {
                    buffer.fmt("    ld.global.nc.u8 %%w0, [%s%u];\n",
                               ptr_tp, ptr_i);
                }
            }

            buffer.fmt("    bra.uni l_%u_done;\n"
                       "l_%u_masked:\n",
                       v_i, v_i);

            if (!is_bool) {
                buffer.fmt("    mov.%s %s%u, 0;\n"
                           "l_%u_done:\n",
                           type_name_ptx_bin[v->type], v_tp, v_i,
                           v_i);
            } else {
                buffer.fmt("    mov.u16 %%w0, 0;\n"
                           "l_%u_done:\n"
                           "    setp.ne.u16 %s%u, %%w0, 0;\n",
                           v_i,
                           v_tp, v_i);
            }
        };
    }

    return result;
}

static const char *reduce_op_name[(int) ReduceOp::Count] = {
    "none", "add", "mul", "min", "max", "and", "or"
};

uint32_t jitc_var_new_scatter(uint32_t target_, uint32_t value, uint32_t index_,
                              uint32_t mask_, ReduceOp reduce_op) {
    Ref target = borrow(target_), ptr;

    if (value == 0 && index_ == 0 && mask_ == 0)
        return 0;
    else if (unlikely(target_ == 0 || value == 0 || index_ == 0 || mask_ == 0))
        jitc_raise("jit_var_new_scatter(): uninitialized arguments!");

    auto print_log = [&](const char *reason, uint32_t index = 0) {
        if (index)
            jitc_log(Debug,
                     "jit_var_new_scatter(r%u[r%u] <- r%u if r%u, via "
                     "ptr r%u, reduce_op=%s): r%u (result=r%u, %s)",
                     (uint32_t) target_, (uint32_t) index_, value,
                     (uint32_t) mask_, (uint32_t) ptr,
                     reduce_op_name[(int) reduce_op], index_, (uint32_t) target,
                     reason);
        else
            jitc_log(Debug,
                     "jit_var_new_scatter(r%u[r%u] <- r%u if r%u, via "
                     "ptr r%u, reduce_op=%s): %s",
                     (uint32_t) target_, (uint32_t) index_, value,
                     (uint32_t) mask_, (uint32_t) ptr,
                     reduce_op_name[(int) reduce_op], reason);
    };

    uint32_t size = 0;
    bool dirty = false;
    JitBackend backend = (JitBackend) 0;
    VarType vt = (VarType) 0;
    bool placeholder = jitc_flags() & (uint32_t) JitFlag::Recording;

    // Get size, ensure no arrays are dirty
    for (uint32_t index : { index_, mask_, value }) {
        const Variable *v = jitc_var(index);
        size = std::max(v->size, size);
        dirty |= v->is_dirty();
        // Fetch type + backend from 'value'
        backend = (JitBackend) v->backend;
        vt = (VarType) v->type;
    }

    ThreadState *ts = thread_state(backend);
    if (dirty) {
        jitc_eval(ts);

        for (uint32_t index : { index_, mask_, value }) {
            if (jitc_var(index)->is_dirty())
                jitc_fail("jit_var_new_scatter(): variable remains dirty after "
                          "evaluation!");
        }
    }

    if (size == 0)
        return target.release();

    for (uint32_t index : { index_, mask_, value }) {
        const Variable *v = jitc_var(index);
        if (size != v->size && v->size != 1)
            jitc_raise("jitc_var_new_scatter(): arrays of incompatible size!");
    }

    if ((VarType) jitc_var(target)->type != vt)
        jitc_raise("jit_var_new_scatter(): target/value type mismatch!");

    // Don't do anything if the mask is always false
    {
        const Variable *v_mask = jitc_var(mask_);
        if (v_mask->is_literal() && v_mask->literal == 0) {
            print_log("skipped, always masked");
            return target.release();
        }
    }

    {
        const Variable *v_target = jitc_var(target),
                       *v_value  = jitc_var(value);

        if (v_target->placeholder)
            jitc_raise("jit_var_new_scatter(): cannot scatter to a placeholder variable!");

        if (v_target->is_literal() && v_value->is_literal() &&
            v_target->literal == v_value->literal && reduce_op == ReduceOp::None) {
            print_log("skipped, target/source are value variables with the same value");
            return target.release();
        }

        if (v_value->is_literal() && v_value->literal == 0 && reduce_op == ReduceOp::Add) {
            print_log("skipped, scatter_reduce(ScatterOp.Add) with zero-valued source variable");
            return target.release();
        }

        // Check if it is safe to write directly
        if (v_target->ref_count > 2) /// 1 from original array, 1 from borrow above
            target = steal(jitc_var_copy(target));
    }

    ptr = steal(jitc_var_new_pointer(backend, jitc_var_ptr(target), target, 1));

    Ref mask  = steal(jitc_var_mask_apply(mask_, size)),
        index = steal(jitc_scatter_gather_index(target, index_));

    // Special case for unmasked scatters
    bool unmasked = false;
    {
        const Variable *v_mask  = jitc_var(mask);
        unmasked   = v_mask->is_literal() && v_mask->literal  == 1;
        size = std::max(size, v_mask->size);
    }

    uint32_t dep[4] = { ptr, value, index, mask };
    uint32_t n_dep = 4;

    Buffer buf{50};

    bool is_float = jitc_is_float(vt);

    const char *red_op_name = nullptr;
    switch (reduce_op) {
        case ReduceOp::None:
            break;
        case ReduceOp::Add:
            red_op_name = (is_float && backend == JitBackend::LLVM) ? "fadd" : "add";
            break;
        case ReduceOp::Min: red_op_name = "min"; break;
        case ReduceOp::Max: red_op_name = "max"; break;
        case ReduceOp::And: red_op_name = "and"; break;
        case ReduceOp::Or: red_op_name = "or"; break;
        default:
            jitc_raise("jitc_var_new_scatter(): unsupported reduction!");
    }

    if (backend == JitBackend::LLVM) {
        if (is_float && reduce_op != ReduceOp::None &&
            reduce_op != ReduceOp::Add)
            jitc_raise("jitc_var_new_scatter(): LLVM %s reduction only "
                       "supports integer values!", red_op_name);

        if (red_op_name == nullptr) {
            buf.put("$r0_0 = bitcast $<i8*$> $r1 to $<$t2*$>$n"
                    "$r0_1 = getelementptr $t2, $<$t2*$> $r0_0, <$w x $t3> $r3$n"
                    "call void @llvm.masked.scatter.v$w$a2(<$w x $t2> $r2, <$w x $t2*> $r0_1, i32 $s2, <$w x $t4> $r4)"
                    "$[declare void @llvm.masked.scatter.v$w$a2(<$w x $t2>, <$w x $t2*>, i32, <$w x $t4>)$]");
        } else if (reduce_op == ReduceOp::Add && is_float) {
            /* Floating point scatter-add is such a crucial operation that we
               provide a special vectorized implementation that first tries to
               perform local reductions to decrease the number of atomic
               operations that must be performed */

            buf.put(
                // Code for this operation
                "$r0_0 = bitcast $<i8*$> $r1 to $<$t2*$>$n"
                "$r0_1 = getelementptr $t2, $<$t2*$> $r0_0, <$w x $t3> $r3$n"
                "call void @ek.scatter_add.v$w$a2(<$w x $t2*> $r0_1, <$w x $t2> $r2, <$w x $t4> $r4)"

                // Intrinsic/helper function
                "$[define internal void @ek.scatter_add.v$w$a2(<$w x $t2*> %ptrs, <$w x $t2> %value, <$w x i1> %active_in) #0 {\n"
                "L0:\n"
                "   br label %L1\n\n"
                "L1:\n"
                "   %index = phi i32 [ 0, %L0 ], [ %index_next, %L3 ]\n"
                "   %active = phi <$w x i1> [ %active_in, %L0 ], [ %active_next_2, %L3 ]\n"
                "   %active_i = extractelement <$w x i1> %active, i32 %index\n"
                "   br i1 %active_i, label %L2, label %L3\n\n"
                "L2:\n"
                "   %ptr_0 = extractelement <$w x $t2 *> %ptrs, i32 %index\n"
                "   %ptr_1 = insertelement <$w x $t2 *> undef, $t2* %ptr_0, i32 0\n"
                "   %ptr_2 = shufflevector <$w x $t2 *> %ptr_1, <$w x $t2 *> undef, <$w x i32> zeroinitializer\n"
                "   %ptr_eq = icmp eq <$w x $t2 *> %ptrs, %ptr_2\n"
                "   %active_cur = and <$w x i1> %ptr_eq, %active\n"
                "   %value_cur = select <$w x i1> %active_cur, <$w x $t2> %value, <$w x $t2> zeroinitializer\n"
                "   %sum = call reassoc $t2 @llvm.experimental.vector.reduce.v2.fadd.$a2.v$w$a2($t2 0.0, <$w x $t2> %value_cur)\n"
                "   atomicrmw fadd $t2* %ptr_0, $t2 %sum monotonic\n"
                "   %active_next = xor <$w x i1> %active, %active_cur\n"
                "   %active_red = call i1 @llvm.experimental.vector.reduce.or.v$wi1(<$w x i1> %active_next)\n"
                "   br i1 %active_red, label %L3, label %L4\n\n"
                "L3:\n"
                "   %active_next_2 = phi <$w x i1> [ %active, %L1 ], [ %active_next, %L2 ]\n"
                "   %index_next = add nuw nsw i32 %index, 1\n"
                "   %cond_2 = icmp eq i32 %index_next, $w\n"
                "   br i1 %cond_2, label %L4, label %L1\n\n"
                "L4:\n"
                "   ret void\n"
                "}$]"
                "$[declare $t2 @llvm.experimental.vector.reduce.v2.fadd.$a2.v$w$a2($t2, <$w x $t2>)$]"
                "$[declare i1 @llvm.experimental.vector.reduce.or.v$wi1(<$w x i1>)$]"
            );
        } else {
            buf.fmt(
                // Code for this operation
                "$r0_0 = bitcast $<i8*$> $r1 to $<$t2*$>$n"
                "$r0_1 = getelementptr $t2, $<$t2*$> $r0_0, <$w x $t3> $r3$n"
                "call void @ek.scatter_%s.v$w$a2(<$w x $t2*> $r0_1, <$w x $t2> $r2, <$w x $t4> $r4)"

                // Intrinsic/helper function
                "$[define internal void @ek.scatter_%s.v$w$a2(<$w x $t2*> %%ptrs, <$w x $t2> %%value, <$w x i1> %%active) #0 {\n"
                "L0:\n"
                "   br label %%L1\n\n"
                "L1:\n"
                "   %%index = phi i32 [ 0, %%L0 ], [ %%index_next, %%L3 ]\n"
                "   %%active_cur = extractelement <$w x i1> %%active, i32 %%index\n"
                "   br i1 %%active_cur, label %%L2, label %%L3\n\n"
                "L2:\n"
                "   %%ptr = extractelement <$w x $t2 *> %%ptrs, i32 %%index\n"
                "   %%value_cur = extractelement <$w x $t2> %%value, i32 %%index\n"
                "   atomicrmw %s $t2* %%ptr, $t2 %%value_cur monotonic\n"
                "   br label %%L3\n\n"
                "L3:\n"
                "   %%index_next = add nuw nsw i32 %%index, 1\n"
                "   %%cond_2 = icmp eq i32 %%index_next, $w\n"
                "   br i1 %%cond_2, label %%L4, label %%L1\n\n"
                "L4:\n"
                "   ret void\n"
                "}$]",

                red_op_name,
                red_op_name,
                red_op_name
            );
        }
    } else {
        if (unmasked) {
            dep[3] = 0;
            n_dep = 3;
        }
    }

    uint32_t result =
        jitc_var_new_stmt(backend, VarType::Void, buf.get(), 0, n_dep, dep);

    if (backend == JitBackend::CUDA) {
        jitc_var(result)->extra = 1;
        Extra &extra = state.extra[result];
        extra.callback_data = strdup(red_op_name ? red_op_name : "");

        extra.assemble = [](const Variable *v, const Extra &extra) {
            const Variable *v_dst  = jitc_var(v->dep[0]);
            const Variable *v_src  = jitc_var(v->dep[1]);
            const Variable *v_idx  = jitc_var(v->dep[2]);

            uint32_t v_i     = v->reg_index;
            uint32_t dst_i   = v_dst->reg_index;
            uint32_t src_i   = v_src->reg_index;
            uint32_t index_i = v_idx->reg_index;

            const char *dst_tp   = type_prefix[v_dst->type];
            const char *src_tp   = type_prefix[v_src->type];
            const char *index_tp = type_prefix[v_idx->type];

            bool index_zero = v_idx->is_literal() && v_idx->literal == 0;
            bool is_src_bool = v_src->type == (uint32_t) VarType::Bool;

            bool masked = false;
            const Variable *v_mask;
            if (v->dep[3]) {
                v_mask = jitc_var(v->dep[3]);
                masked = !(v_mask->is_literal() && v_mask->literal == 1);
            }

            ThreadState *ts = thread_state_cuda;
            const char *red_op_name = (const char *) extra.callback_data;
            bool red_op_none = (strcmp(red_op_name, "") == 0);
            Buffer dst_addr(8);
            Buffer src_reg(8);

            if (masked) {
                buffer.fmt("    @!%%p%u bra l_%u_done;\n",
                           v_mask->reg_index, v_i);
            }

            /* Performance optimization for an important special case (FP32
               atomic scatter-reduction). Begin with an intra-warp reduction to
               issue fewer atomic global memory transactions */

            if ((VarType) v_src->type == VarType::Float32 &&
                (VarType) v_idx->type == VarType::UInt32 &&
                red_op_name &&
                ts->ptx_version>= 62 &&
                ts->compute_capability >= 70 &&
                strcmp(red_op_name, "add") == 0) {
                jitc_register_global(
                    ".visible .func reduce_f32(.param .u64 ptr,\n"
                    "                          .param .u32 index,\n"
                    "                          .param .f32 value) {\n"
                    "    .reg .pred %p<15>;\n"
                    "    .reg .f32 %f<22>;\n"
                    "    .reg .b32 %r<41>;\n"
                    "    .reg .b64 %rd<4>;\n"
                    "\n"
                    "    ld.param.u64 %rd2, [ptr];\n"
                    "    ld.param.u32 %r11, [index];\n"
                    "    ld.param.f32 %f5, [value];\n"
                    "    mad.wide.u32 %rd1, %r11, 4, %rd2;\n"
                    "    activemask.b32 %r10;\n"
                    "    match.any.sync.b64 %r39, %rd1, %r10;\n"
                    "    setp.eq.s32 %p1, %r39, -1;\n"
                    "    @%p1 bra fast_path;\n"
                    "    brev.b32 %r12, %r39;\n"
                    "    bfind.shiftamt.u32 %r40, %r12;\n"
                    "    setp.ne.s32 %p2, %r39, 0;\n"
                    "    vote.sync.any.pred %p3, %p2, %r10;\n"
                    "    mov.f32 %f21, 0f00000000;\n"
                    "    not.pred %p4, %p3;\n"
                    "    @%p4 bra maybe_scatter;\n"
                    "    mov.b32 %r4, %f5;\n"
                    "    mov.u32 %r38, %r40;\n"
                    "\n"
                    "slow_path:\n"
                    "    mov.u32 %r14, 31;\n"
                    "    shfl.sync.idx.b32 %r15|%p5, %r4, %r38, %r14, %r10;\n"
                    "    mov.b32 %f8, %r15;\n"
                    "    add.f32 %f9, %f21, %f8;\n"
                    "    setp.eq.s32 %p6, %r39, 0;\n"
                    "    selp.f32 %f21, %f21, %f9, %p6;\n"
                    "    mov.u32 %r16, -2;\n"
                    "    shf.l.wrap.b32 %r17, %r16, %r16, %r38;\n"
                    "    and.b32 %r39, %r39, %r17;\n"
                    "    brev.b32 %r18, %r39;\n"
                    "    bfind.shiftamt.u32 %r38, %r18;\n"
                    "    setp.ne.s32 %p7, %r39, 0;\n"
                    "    vote.sync.any.pred %p8, %p7, %r10;\n"
                    "    @%p8 bra slow_path;\n"
                    "    bra.uni maybe_scatter;\n"
                    "\n"
                    "fast_path:\n"
                    "    mov.b32 %r21, %f5;\n"
                    "    mov.u32 %r22, 2;\n"
                    "    mov.u32 %r23, 31;\n"
                    "    mov.u32 %r24, 16;\n"
                    "    shfl.sync.down.b32 %r25|%p9, %r21, %r24, %r23, %r10;\n"
                    "    mov.b32 %f10, %r25;\n"
                    "    add.f32 %f11, %f10, %f5;\n"
                    "    mov.b32 %r26, %f11;\n"
                    "    mov.u32 %r27, 8;\n"
                    "    shfl.sync.down.b32 %r28|%p10, %r26, %r27, %r23, %r10;\n"
                    "    mov.b32 %f12, %r28;\n"
                    "    add.f32 %f13, %f11, %f12;\n"
                    "    mov.b32 %r29, %f13;\n"
                    "    mov.u32 %r30, 4;\n"
                    "    shfl.sync.down.b32 %r31|%p11, %r29, %r30, %r23, %r10;\n"
                    "    mov.b32 %f14, %r31;\n"
                    "    add.f32 %f15, %f13, %f14;\n"
                    "    mov.b32 %r32, %f15;\n"
                    "    shfl.sync.down.b32 %r33|%p12, %r32, %r22, %r23, %r10;\n"
                    "    mov.b32 %f16, %r33;\n"
                    "    add.f32 %f17, %f15, %f16;\n"
                    "    mov.b32 %r34, %f17;\n"
                    "    mov.u32 %r35, 1;\n"
                    "    shfl.sync.down.b32 %r36|%p13, %r34, %r35, %r23, %r10;\n"
                    "    mov.b32 %f18, %r36;\n"
                    "    add.f32 %f21, %f17, %f18;\n"
                    "    mov.u32 %r40, 0;\n"
                    "\n"
                    "maybe_scatter:\n"
                    "    mov.u32 %r37, %laneid;\n"
                    "    setp.ne.s32 %p14, %r37, %r40;\n"
                    "    @%p14 bra done;\n"
                    "    atom.add.f32 %f19, [%rd1], %f21;\n"
                    "\n"
                    "done:\n"
                    "    ret;\n"
                    "}\n\n");

                buffer.fmt("    call reduce_f32, (%%rd%u, %%r%u, %%f%u);\n",
                           dst_i, index_i, src_i);
                if (masked)
                    buffer.fmt("l_%u_done:\n", v_i);
                return;
            }

            if (!index_zero) {
                buffer.fmt("    mad.wide.%s %%rd3, %s%u, %s, %s%u;\n",
                           type_name_ptx[v_idx->type], index_tp, index_i,
                           type_size_str[v_src->type], dst_tp, dst_i);
                dst_addr.put("%rd3");
            } else {
                dst_addr.fmt("%s%u", dst_tp, dst_i);
            }

            if (is_src_bool) {
                buffer.fmt("    selp.u16 %%w0, 1, 0, %s%u;\n", src_tp, src_i);
                src_reg.put("%w0");
            } else {
                src_reg.fmt("%s%u", src_tp, src_i);
            }

            bool use_atom_op =
                red_op_none &&
                std::tie(jitc_cuda_version_major, jitc_cuda_version_minor) <
                    std::make_tuple(11, 5);

            const char *src_type = is_src_bool ? "u8" : type_name_ptx[v_src->type];

            if (red_op_none) {
                buffer.fmt("    st.global.%s [%s], %s;\n",
                           src_type, dst_addr.get(), src_reg.get());
            } else if (use_atom_op) {
                buffer.fmt("    atom.global.%s.%s %s%u, [%s], %s;\n",
                           red_op_name, src_type, src_tp, v_i, dst_addr.get(), src_reg.get());
            } else {
                buffer.fmt("    red.global.%s.%s [%s], %s;\n",
                           red_op_name, src_type, dst_addr.get(), src_reg.get());
            }

            if (masked)
                buffer.fmt("l_%u_done:\n", v_i);
        };
    }

    print_log(((uint32_t) target == target_) ? "direct" : "copy", result);

    Variable *v = jitc_var(result);
    {
        v->placeholder = placeholder;
        v->size = size;
    }

    jitc_var_mark_side_effect(result);

    return target.release();
}
