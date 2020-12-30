/*
    src/op.cpp -- Conversion of standard operations into PTX and LLVM IR

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/
#define EK_OPNAME 1

#include "internal.h"
#include "var.h"
#include "eval.h"
#include "log.h"
#include "op.h"

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

static bool jitc_is_not_void(VarType type) {
    return type != VarType::Void;
}

// ===========================================================================
// Evaluation helper routines for literal constant values
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

template <typename T, enable_if_t<std::is_signed<T>::value> = 0>
T eval_abs(T value) { return std::abs(value); }

template <typename T, enable_if_t<!std::is_signed<T>::value> = 0>
T eval_abs(T value) { return value; }

template <typename T, enable_if_t<!std::is_integral<T>::value> = 0>
T eval_mod(T, T) {
    jitc_raise("eval_mod(): unsupported operands!");
}

template <typename T, enable_if_t<std::is_integral<T>::value> = 0>
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
    return (T)(a * b + c);
}

template <typename T, enable_if_t<std::is_same<T, bool>::value> = 0>
T eval_fma(T, T, T) {
    jitc_raise("eval_fma(): unsupported operands!");
}

// ===========================================================================
// Helpe to apply a given expression (lambda) to literal constant variables
// ===========================================================================

template <typename Type> Type i2v(uint64_t value) {
    Type result;
    memcpy(&result, &value, sizeof(Type));
    return result;
}

template <typename Type> uint64_t v2i(Type value) {
    uint64_t result = 0;
    memcpy(&result, &value, sizeof(Type));
    if (std::is_same<Type, bool>::value)
        result &= 1;
    return result;
}

template <typename Arg, typename... Args> Arg first(Arg arg, Args...) { return arg; }

template <bool Cond = false, typename Func, typename... Args>
uint64_t jitc_eval_literal(Func func, const Args *... args) {
    switch ((VarType) first(args->type...)) {
        case VarType::Bool:    return v2i(func(i2v<   bool> (args->value)...));
        case VarType::Int8:    return v2i(func(i2v< int8_t> (args->value)...));
        case VarType::UInt8:   return v2i(func(i2v<uint8_t> (args->value)...));
        case VarType::Int16:   return v2i(func(i2v< int16_t>(args->value)...));
        case VarType::UInt16:  return v2i(func(i2v<uint16_t>(args->value)...));
        case VarType::Int32:   return v2i(func(i2v< int32_t>(args->value)...));
        case VarType::UInt32:  return v2i(func(i2v<uint32_t>(args->value)...));
        case VarType::Int64:   return v2i(func(i2v< int64_t>(args->value)...));
        case VarType::UInt64:  return v2i(func(i2v<uint64_t>(args->value)...));
        case VarType::Float32: return v2i(func(i2v<   float>(args->value)...));
        case VarType::Float64: return v2i(func(i2v<  double>(args->value)...));
        default: jitc_fail("jit_eval_literal(): unsupported variable type!");
    }
}

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
    jitc_var_dec_ref_ext(shift);
    return result;
}

// ===========================================================================
// jitc_var_new_op(): various standard operations for Enoki-JIT variables
// ===========================================================================

// Error handler
JIT_NOINLINE uint32_t jitc_var_new_op_fail(const char *error, JitOp op,
                                           uint32_t n_dep, const uint32_t *dep);

uint32_t jitc_var_new_op(JitOp op, uint32_t n_dep, const uint32_t *dep) {
    uint32_t size = 0;
    bool dirty = false, literal = true, uninitialized = false;
    uint32_t vti = 0;
    bool literal_zero[4] { }, literal_one[4] { };
    uint32_t backend_i = 0;
    Variable *v[4] { };

    if (unlikely(n_dep == 0 || n_dep > 4))
        jitc_fail("jit_var_new_op(): 1-4 dependent variables supported!");

    for (uint32_t i = 0; i < n_dep; ++i) {
        if (likely(dep[i])) {
            Variable *vi = jitc_var(dep[i]);
            vti = std::max(vti, vi->type);
            size = std::max(size, vi->size);
            dirty |= vi->dirty;
            backend_i |= (uint32_t) vi->backend;
            v[i] = vi;

            if (vi->literal) {
                uint64_t one;
                switch ((VarType) vi->type) {
                    case VarType::Float16: one = 0x3c00ull; break;
                    case VarType::Float32: one = 0x3f800000ull; break;
                    case VarType::Float64: one = 0x3ff0000000000000ull; break;
                    default: one = 1; break;
                }
                literal_zero[i] = vi->value == 0;
                literal_one[i] = vi->value == one;
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

    if (dirty) {
        jitc_eval(thread_state(backend));
        for (uint32_t i = 0; i < n_dep; ++i)
            v[i] = jitc_var(dep[i]);
    }

    bool is_float  = jitc_is_float(vt),
         is_uint = jitc_is_uint(vt),
         is_single = vt == VarType::Float32,
         is_valid = jitc_is_arithmetic(vt);

    const char *stmt = nullptr;

    // Used if the result produces a literal value
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
                lv = jitc_eval_literal([](auto value) { return -value; }, v[0]);
            } else if (backend == JitBackend::CUDA) {
                if (is_uint) {
                    stmt = "not.$b0 $r0, $r1$n"
                           "add.$t0 $r0, $r0, 1";
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
                    uint64_t mask_value = ((uint64_t) 1 << (var_type_size[vti] * 8 - 1)) - 1;
                    uint32_t mask = jitc_var_new_literal(backend, vt, &mask_value, 1, 0);
                    uint32_t deps[2] = { dep[0], mask };
                    li = jitc_var_new_op(JitOp::And, 2, deps);
                    li_created = true;
                    jitc_var_dec_ref_ext(mask);
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
                stmt = "$r0 = $call <$w x $t0> @llvm.sqrt.v$w$a1(<$w x $t1> $r1)";
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
                jitc_var_dec_ref_ext(one);
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
                jitc_var_dec_ref_ext(one);
                jitc_var_dec_ref_ext(result_1);
            }
            break;

        case JitOp::Ceil:
            is_valid = jitc_is_float(vt);
            if (literal) {
                lv = jitc_eval_literal([](auto value) { return eval_ceil(value); }, v[0]);
            } else if (backend == JitBackend::CUDA) {
                stmt = "cvt.rpi.$t0.$t0 $r0, $r1";
            } else {
                stmt = "$r0 = $call <$w x $t0> @llvm.ceil.v$w$a1(<$w x $t1> $r1)";
            }
            break;

        case JitOp::Floor:
            is_valid = jitc_is_float(vt);
            if (literal) {
                lv = jitc_eval_literal([](auto value) { return eval_floor(value); }, v[0]);
            } else if (backend == JitBackend::CUDA) {
                stmt = "cvt.rmi.$t0.$t0 $r0, $r1";
            } else {
                stmt = "$r0 = $call <$w x $t0> @llvm.floor.v$w$a1(<$w x $t1> $r1)";
            }
            break;

        case JitOp::Round:
            is_valid = jitc_is_float(vt);
            if (literal) {
                lv = jitc_eval_literal([](auto value) { return eval_round(value); }, v[0]);
            } else if (backend == JitBackend::CUDA) {
                stmt = "cvt.rni.$t0.$t0 $r0, $r1";
            } else {
                stmt = "$r0 = $call <$w x $t0> @llvm.nearbyint.v$w$a1(<$w x $t1> $r1)";
            }
            break;

        case JitOp::Trunc:
            is_valid = jitc_is_float(vt);
            if (literal) {
                lv = jitc_eval_literal([](auto value) { return eval_trunc(value); }, v[0]);
            } else if (backend == JitBackend::CUDA) {
                stmt = "cvt.rzi.$t0.$t0 $r0, $r1";
            } else {
                stmt = "$r0 = $call <$w x $t0> @llvm.trunc.v$w$a1(<$w x $t1> $r1)";
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
                stmt = "$r0 = $call <$w x $t0> @llvm.ctpop.v$w$a1(<$w x $t1> $r1)";
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
                stmt = "$r0 = $call <$w x $t0> @llvm.ctlz.v$w$a1(<$w x $t1> $r1, i1 0)";
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
                stmt = "$r0 = $call <$w x $t0> @llvm.cttz.v$w$a1(<$w x $t1> $r1, i1 0)";
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

        case JitOp::Mul:
            if (literal) {
                lv = jitc_eval_literal([](auto v0, auto v1) { return v0 * v1; },
                                       v[0], v[1]);
            } else if (literal_one[0]) {
                li = dep[1];
            } else if (literal_one[1]) {
                li = dep[0];
            } else if (is_uint && v[0]->literal && jitc_is_pow2(v[0]->value)) {
                li = jitc_var_shift(backend, vt, JitOp::Shl, dep[1], v[0]->value);
                li_created = true;
            } else if (is_uint && v[1]->literal && jitc_is_pow2(v[1]->value)) {
                li = jitc_var_shift(backend, vt, JitOp::Shl, dep[0], v[1]->value);
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
                lv = jitc_eval_literal([](auto v0, auto v1) { return v0 / v1; },
                                      v[0], v[1]);
            } else if (literal_one[1]) {
                li = dep[0];
            } else if (is_uint && v[1]->literal && jitc_is_pow2(v[1]->value)) {
                li = jitc_var_shift(backend, vt, JitOp::Shr, dep[0], v[1]->value);
                li_created = true;
            } else if (jitc_is_float(vt) && v[1]->literal) {
                uint32_t recip = jitc_var_new_op(JitOp::Rcp, 1, &dep[1]);
                uint32_t deps[2] = { dep[0], recip };
                li = jitc_var_new_op(JitOp::Mul, 2, deps);
                li_created = 1;
                jitc_var_dec_ref_ext(recip);
            } else if (backend == JitBackend::CUDA) {
                if (is_single)
                    stmt = "div.rn.ftz.$t0 $r0, $r1, $r2";
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
                    stmt = "$r0 = $call <$w x $t0> @llvm.minnum.v$w$a1(<$w x $t1> $r1, <$w x $t2> $r2)";
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
                    stmt = "$r0 = $call <$w x $t0> @llvm.maxnum.v$w$a1(<$w x $t1> "
                           "$r1, <$w x $t2> $r2)";
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
            } else if (((VarType) v[0]->type == VarType::Bool && literal_one[0]) || literal_zero[1]) {
                li = dep[1];
            } else if (((VarType) v[1]->type == VarType::Bool && literal_one[1]) || literal_zero[0]) {
                li = dep[0];
            } else if (backend == JitBackend::CUDA) {
                stmt = ((VarType) v[1]->type == VarType::Bool && vt != VarType::Bool)
                           ? "selp.$b0 $r0, $r1, 0, $r2"
                           : "and.$b0 $r0, $r1, $r2";
            } else {
                if ((VarType) v[1]->type == VarType::Bool && vt != VarType::Bool) {
                    stmt = "$r0_0 = sext <$w x $t2> $r2 to <$w x $b0>$n"
                           "$r0_1 = bitcast <$w x $t1> $r1 to <$w x $b0>$n"
                           "$r0_2 = and <$w x $b0> $r0_0, $r0_1$n"
                           "$r0 = bitcast <$w x $b0> $r0_2 to <$w x $t0>";
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
            } else if ((vt == VarType::Bool && literal_one[1]) || literal_zero[0]) {
                li = dep[1];
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
                    stmt = "$r0 = $call <$w x $t0> @llvm.fma.v$w$a1(<$w x $t1> "
                           "$r1, <$w x $t2> $r2, <$w x $t3> $r3)";
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
                stmt = (VarType) v[1]->type != VarType::Bool
                           ? "selp.$t0 $r0, $r2, $r3, $r1"
                           : "not.pred %p3, $r1$n"
                             "and.pred %p3, %p3, $r3$n"
                             "and.pred $r0, $r1, $r2$n"
                             "or.pred $r0, $r0, %p3";
            } else {
                stmt = "$r0 = select <$w x $t1> $r1, <$w x $t2> $r2, <$w x $t3> $r3";
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

    if (unlikely(std::max(state.log_level_stderr, state.log_level_callback) >=
                 LogLevel::Debug)) {
        var_buffer.clear();
        var_buffer.fmt("jit_var_new_op(%s", op_name[(int) op]);
        for (uint32_t i = 0; i < n_dep; ++i)
            var_buffer.fmt(", %u", dep[i]);
        var_buffer.put(")");
        if (li || literal)
            var_buffer.put(": constant.");
        jitc_log(Debug, "%s", var_buffer.get());
    }

    if (li) {
        uint32_t result = jitc_var_resize(li, size);
        if (li_created)
            jitc_var_dec_ref_ext(li);
        return result;
    } else {
        Variable v2;
        v2.size = size;
        v2.type = (uint32_t) vtr;
        v2.backend = (uint32_t) backend;

        if (literal) {
            v2.literal = 1;
            v2.value = lv;
        } else {
            v2.stmt = (char *) stmt;
            for (uint32_t i = 0; i < n_dep; ++i) {
                v2.dep[i] = dep[i];
                jitc_var_inc_ref_int(dep[i], v[i]);
            }
        }

        return jitc_var_new(v2);
    }
}

JIT_NOINLINE uint32_t jitc_var_new_op_fail(const char *error, JitOp op, uint32_t n_dep, const uint32_t *dep) {
    switch (n_dep) {
        case 1:
            jitc_raise("jit_var_new_op(%s, %u): %s", op_name[(int) op], dep[0],
                      error);
        case 2:
            jitc_raise("jit_var_new_op(%s, %u, %u): %s", op_name[(int) op],
                      dep[0], dep[1], error);
        case 3:
            jitc_raise("jit_var_new_op(%s, %u, %u, %u): %s", op_name[(int) op],
                      dep[0], dep[1], dep[2], error);
        case 4:
            jitc_raise("jit_var_new_op(%s, %u, %u, %u, %u): %s",
                      op_name[(int) op], dep[0], dep[1], dep[2], dep[3], error);
        default:
            jitc_fail("jit_var_new_op(): invalid number of arguments!");
    }
}

uint32_t jitc_var_new_cast(uint32_t index, VarType target_type,
                           int reinterpret) {
    Variable *v = jitc_var(index);
    const JitBackend backend = (JitBackend) v->backend;
    const VarType source_type = (VarType) v->type;

    if (source_type == target_type) {
        jitc_var_inc_ref_ext(index);
        return index;
    }

    bool source_bool = source_type == VarType::Bool,
         target_bool = target_type == VarType::Bool,
         source_float = jitc_is_float(source_type),
         target_float = jitc_is_float(target_type);

    uint32_t source_size =
                 source_bool ? 0 : var_type_size[(uint32_t) source_type],
             target_size =
                 target_bool ? 0 : var_type_size[(uint32_t) target_type];

    if (reinterpret && source_size != target_size) {
        jitc_raise("jit_var_new_cast(): reinterpret cast between types of "
                   "different size!");
    } else if (source_size == target_size && !source_float && !target_float) {
        reinterpret = 1;
    }

    if (v->dirty) {
        jitc_eval(thread_state(backend));
        v = jitc_var(index);
    }

    if (v->literal) {
        uint64_t value;
        if (reinterpret) {
            value = v->value;
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
        v2.size = v->size;
        v2.type = (uint32_t) target_type;
        v2.backend = (uint32_t) backend;
        v2.stmt = (char *) stmt;
        v2.dep[0] = index;
        jitc_var_inc_ref_int(index, v);
        return jitc_var_new(v2);
    }
}

/// Combine 'mask' with top element of the mask stack
static uint32_t jitc_scatter_gather_mask(uint32_t mask) {
    const Variable *v_mask = jitc_var(mask);
    if ((VarType) v_mask->type != VarType::Bool)
        jitc_raise("jit_scatter_gather_mask(): expected a boolean array as scatter/gather mask");

    Ref mask_top = jitc_var_mask_peek((JitBackend) v_mask->backend);
    uint32_t deps[2] = { mask, mask_top.get() };
    return jitc_var_new_op(JitOp::And, 2, deps);
}

static uint32_t jitc_scatter_gather_index(uint32_t src, uint32_t index) {
    const Variable *v_src = jitc_var(src),
                   *v_index = jitc_var(index);

    VarType source_type = (VarType) v_index->type;
    if (!jitc_is_uint(source_type) && !jitc_is_sint(source_type))
        jitc_raise("jit_scatter_gather_index(): expected an integer array as scatter/gather index");

    VarType target_type = VarType::UInt32;
    // Need 64 bit indices for upper 2G entries (gather indices are signed in LLVM)
    if (v_src->size > 0x7fffffff && (JitBackend) v_src->backend == JitBackend::LLVM)
        target_type = VarType::UInt64;

    return jitc_var_new_cast(index, target_type, 0);
}

uint32_t jitc_var_new_gather(uint32_t src, uint32_t index_, uint32_t mask_) {
    Ref mask (jitc_scatter_gather_mask(mask_)),
        index(jitc_scatter_gather_index(src, index_));

    const Variable *v_src = jitc_var(src),
                   *v_index = jitc_var(index.get()),
                   *v_mask = jitc_var(mask.get());

    uint32_t size = std::max(v_index->size, v_mask->size);

    // Completely avoid the gather operation for trivial arguments
    if (v_src->literal || v_src->size == 1) {
        uint32_t deps[2] = { src, mask.get() };
        Ref tmp = jitc_var_new_op(JitOp::And, 2, deps);
        return jitc_var_resize(tmp.get(), size);
    }

    JitBackend backend = (JitBackend) v_src->backend;

    // Ensure that the source array is fully evaluated
    if (!v_src->data || v_src->dirty || v_index->dirty || v_mask->dirty) {
        jitc_var_schedule(src);
        jitc_eval(thread_state(backend));
    }

    // Location of variable may have changed
    v_src = jitc_var(src);
    v_mask = jitc_var(mask.get());

    bool unmasked = v_mask->literal && v_mask->value == 1;
    VarType vt = (VarType) v_src->type;

    // Create a pointer + reference, invalidates the v_* variables
    Ref ptr = jitc_var_new_pointer(backend, v_src->data, src, 0);

    uint32_t dep[3] = { ptr.get(), index.get(), mask.get() };
    uint32_t dep_count = 3;

    const char *stmt;
    if (backend == JitBackend::CUDA) {
        if (vt != VarType::Bool) {
            if (unmasked)
                stmt = "mad.wide.$t2 %rd3, $r2, $s0, $r1$n"
                       "ld.global.nc.$t0 $r0, [%rd3]";
            else
                stmt = "mad.wide.$t2 %rd3, $r2, $s0, $r1$n"
                       "@$r3 ld.global.nc.$t0 $r0, [%rd3]$n"
                       "@!$r3 mov.$b0 $r0, 0";
        } else {
            if (unmasked)
                stmt = "mad.wide.$t2 %rd3, $r2, $s0, $r1$n"
                       "ld.global.nc.u8 %w0, [%rd3]$n"
                       "setp.ne.u16 $r0, %w0, 0";
            else
                stmt = "mad.wide.$t2 %rd3, $r2, $s0, $r1$n"
                       "@$r3 ld.global.nc.u8 %w0, [%rd3]$n"
                       "@!$r3 mov.u16 %w0, 0$n"
                       "setp.ne.u16 $r0, %w0, 0";
        }

        if (unmasked) {
            dep[2] = 0;
            dep_count = 2;
        }
    } else {
        if (vt != VarType::Bool && vt != VarType::UInt8 && vt != VarType::Int8) {
            stmt = "$r0_0 = bitcast $t1 $r1 to $t0*$n"
                   "$r0_1 = getelementptr $t0, $t0* $r0_0, <$w x $t2> $r2$n"
                   "$r0 = $call <$w x $t0> @llvm.masked.gather.v$w$a0(<$w x $t0*> $r0_1, i32 $s0, <$w x $t3> $r3, <$w x $t0> $z)";
        } else {
            stmt = "$r0_0 = bitcast $t1 $r1 to i8*$n"
                   "$r0_1 = getelementptr i8, i8* $r0_0, <$w x $t2> $r2$n"
                   "$r0_2 = bitcast <$w x i8*> $r0_1 to <$w x i32*>$n"
                   "$r0_3 = $call <$w x i32> @llvm.masked.gather.v$wi32(<$w x i32*> $r0_2, i32 $s0, <$w x $t3> $r3, <$w x i32> $z)$n"
                   "$r0 = trunc <$w x i32> $r0_3 to <$w x $t0>";
        }
    }

    return jitc_var_new_stmt(backend, vt, stmt, 1, dep_count, dep);
}

