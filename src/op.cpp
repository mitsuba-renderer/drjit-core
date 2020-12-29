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

template <typename T, enable_if_t<!std::is_floating_point<T>::value> = 0>
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

uint32_t jitc_var_shift(int cuda, VarType vt, OpType op, uint32_t index, uint64_t amount) {
    amount = 63 - jitc_clz(amount);
    uint32_t shift = jitc_var_new_literal(cuda, vt, &amount, 1, 0);
    uint32_t deps[2] = { index, shift };
    uint32_t result = jitc_var_new_op(op, 2, deps);
    jitc_var_dec_ref_ext(shift);
    return result;
}

// ===========================================================================
// jitc_var_new_op(): various standard operations for Enoki-JIT variables
// ===========================================================================

// Error handler
JITC_NOINLINE uint32_t jitc_var_new_op_fail(const char *error, OpType op,
                                           uint32_t n_dep, const uint32_t *dep);

uint32_t jitc_var_new_op(OpType op, uint32_t n_dep, const uint32_t *dep) {
    uint32_t size = 0;
    bool dirty = false, literal = true, uninitialized = false;
    uint32_t vti = 0;
    bool literal_zero[4] { }, literal_one[4] { };
    bool cuda = false;
    Variable *v[4] { };

    if (unlikely(n_dep == 0 || n_dep > 4))
        jitc_fail("jit_var_new_op(): 1-4 dependent variables supported!");

    for (uint32_t i = 0; i < n_dep; ++i) {
        if (likely(dep[i])) {
            Variable *vi = jitc_var(dep[i]);
            vti = std::max(vti, vi->type);
            size = std::max(size, vi->size);
            dirty |= vi->dirty;
            cuda |= vi->cuda;
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
        else if (unlikely(v[i]->size != size && v[i]->size != 1))
            error = "arithmetic involving arrays of incompatible size!";
        else if (unlikely(v[i]->type != vti))
            error = "arithmetic involving arrays of incompatible type!";
        else if (unlikely(v[i]->cuda != cuda))
            error = "mixed CUDA and LLVM arrays!";
    }

    if (unlikely(error))
        jitc_var_new_op_fail(error, op, n_dep, dep);

    if (dirty) {
        jitc_eval(thread_state(cuda));
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
       variable (in that case, it must set li_decref=true below) */
    uint32_t li = 0;
    bool li_decref = false;

    switch (op) {
        case OpType::Not:
            is_valid = jitc_is_not_void(vt) && !is_float;
            if (literal) {
                lv = jitc_eval_literal([](auto value) { return eval_not(value); }, v[0]);
            } else if (cuda) {
                stmt = "not.$b0 $r0, $r1";
            } else {
                stmt = !jitc_is_float(vt)
                           ? "$r0 = xor <$w x $t1> $r1, $o0"
                           : "$r0_0 = bitcast <$w x $t1> $r1 to <$w x $b0>$n"
                             "$r0_1 = xor <$w x $b0> $r0_0, $o0$n"
                             "$r0 = bitcast <$w x $b0> $r0_1 to <$w x $t0>";
            }
            break;

        case OpType::Neg:
            is_valid = jitc_is_arithmetic(vt);
            if (literal) {
                lv = jitc_eval_literal([](auto value) { return -value; }, v[0]);
            } else if (cuda) {
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

        case OpType::Abs:
            if (is_uint) {
                li = dep[0];
            } else if (literal) {
                lv = jitc_eval_literal([](auto value) { return eval_abs(value); }, v[0]);
            } else if (cuda) {
                stmt = "abs.$t0 $r0, $r1";
            } else {
                if (is_float) {
                    uint64_t mask_value = ((uint64_t) 1 << (var_type_size[vti] * 8 - 1)) - 1;
                    uint32_t mask = jitc_var_new_literal(cuda, vt, &mask_value, 1, 0);
                    uint32_t deps[2] = { dep[0], mask };
                    li = jitc_var_new_op(OpType::And, 2, deps);
                    li_decref = true;
                    jitc_var_dec_ref_ext(mask);
                } else {
                    stmt = "$r0_0 = icmp slt <$w x $t0> $r1, zeroinitializer$n"
                           "$r0_1 = sub <$w x $t0> zeroinitializer, $r1$n"
                           "$r0 = select <$w x i1> $r0_0, <$w x $t1> $r0_1, <$w x $t1> $r1";
                }
            }
            break;

        case OpType::Sqrt:
            is_valid = jitc_is_float(vt);
            if (literal) {
                lv = jitc_eval_literal([](auto value) { return eval_sqrt(value); }, v[0]);
            } else if (cuda) {
                stmt = is_single ? "sqrt.approx.ftz.$t0 $r0, $r1"
                                 : "sqrt.rn.$t0 $r0, $r1";
            } else {
                stmt = "$r0 = $call <$w x $t0> @llvm.sqrt.v$w$a1(<$w x $t1> $r1)";
            }
            break;

        case OpType::Rcp:
            is_valid = jitc_is_float(vt);
            if (literal) {
                lv = jitc_eval_literal([](auto value) { return eval_rcp(value); }, v[0]);
            } else if (cuda) {
                stmt = is_single ? "rcp.approx.ftz.$t0 $r0, $r1"
                                 : "rcp.rn.$t0 $r0, $r1";
            } else {
                // TODO can do better here..
                float f1 = 1.f; double d1 = 1.0;
                uint32_t one = jitc_var_new_literal(cuda, vt,
                                                   vt == VarType::Float32
                                                       ? (const void *) &f1
                                                       : (const void *) &d1,
                                                   1, 0);
                uint32_t deps[2] = { one, dep[0] };
                li = jitc_var_new_op(OpType::Div, 2, deps);
                jitc_var_dec_ref_ext(one);
                li_decref = true;
            }
            break;

        case OpType::Rsqrt:
            is_valid = jitc_is_float(vt);
            if (literal) {
                lv = jitc_eval_literal([](auto value) { return eval_rsqrt(value); }, v[0]);
            } else if (cuda) {
                stmt = is_single ? "rsqrt.approx.ftz.$t0 $r0, $r1"
                                 : "rcp.rn.$t0 $r0, $r1$n"
                                   "sqrt.rn.$t0 $r0, $r0";
            } else {
                // TODO can do better here..
                float f1 = 1.f; double d1 = 1.0;
                uint32_t one = jitc_var_new_literal(cuda, vt,
                                                   vt == VarType::Float32
                                                       ? (const void *) &f1
                                                       : (const void *) &d1,
                                                   1, 0);
                uint32_t deps[2] = { one, dep[0] };
                uint32_t result_1 = jitc_var_new_op(OpType::Div, 2, deps);
                li = jitc_var_new_op(OpType::Sqrt, 1, &result_1);
                li_decref = true;
                jitc_var_dec_ref_ext(one);
                jitc_var_dec_ref_ext(result_1);
            }
            break;

        case OpType::Ceil:
            is_valid = jitc_is_float(vt);
            if (literal) {
                lv = jitc_eval_literal([](auto value) { return eval_ceil(value); }, v[0]);
            } else if (cuda) {
                stmt = "cvt.rpi.$t0.$t0 $r0, $r1";
            } else {
                stmt = "$r0 = $call <$w x $t0> @llvm.ceil.v$w$a1(<$w x $t1> $r1)";
            }
            break;

        case OpType::Floor:
            is_valid = jitc_is_float(vt);
            if (literal) {
                lv = jitc_eval_literal([](auto value) { return eval_floor(value); }, v[0]);
            } else if (cuda) {
                stmt = "cvt.rmi.$t0.$t0 $r0, $r1";
            } else {
                stmt = "$r0 = $call <$w x $t0> @llvm.floor.v$w$a1(<$w x $t1> $r1)";
            }
            break;

        case OpType::Round:
            is_valid = jitc_is_float(vt);
            if (literal) {
                lv = jitc_eval_literal([](auto value) { return eval_round(value); }, v[0]);
            } else if (cuda) {
                stmt = "cvt.rni.$t0.$t0 $r0, $r1";
            } else {
                stmt = "$r0 = $call <$w x $t0> @llvm.nearbyint.v$w$a1(<$w x $t1> $r1)";
            }
            break;

        case OpType::Trunc:
            is_valid = jitc_is_float(vt);
            if (literal) {
                lv = jitc_eval_literal([](auto value) { return eval_trunc(value); }, v[0]);
            } else if (cuda) {
                stmt = "cvt.rzi.$t0.$t0 $r0, $r1";
            } else {
                stmt = "$r0 = $call <$w x $t0> @llvm.trunc.v$w$a1(<$w x $t1> $r1)";
            }
            break;

        case OpType::Popc:
            is_valid = jitc_is_arithmetic(vt) && !is_float;
            if (literal) {
                lv = jitc_eval_literal([](auto value) { return eval_popc(value); }, v[0]);
            } else if (cuda) {
                stmt = (vt == VarType::UInt32 || vt == VarType::Int32)
                           ? "popc.$b0 $r0, $r1"
                           : "popc.$b0 %r3, $r1$n"
                             "cvt.$t0.u32 $r0, %r3";
            } else {
                stmt = "$r0 = $call <$w x $t0> @llvm.ctpop.v$w$a1(<$w x $t1> $r1)";
            }
            break;

        case OpType::Clz:
            is_valid = jitc_is_arithmetic(vt) && !is_float;
            if (literal) {
                lv = jitc_eval_literal([](auto value) { return eval_clz(value); }, v[0]);
            } else if (cuda) {
                stmt = (vt == VarType::UInt32 || vt == VarType::Int32)
                           ? "clz.$b0 $r0, $r1"
                           : "clz.$b0 %r3, $r1$n"
                             "cvt.$t0.u32 $r0, %r3";
            } else {
                stmt = "$r0 = $call <$w x $t0> @llvm.ctlz.v$w$a1(<$w x $t1> $r1, i1 0)";
            }
            break;

        case OpType::Ctz:
            is_valid = jitc_is_arithmetic(vt) && !is_float;
            if (literal) {
                lv = jitc_eval_literal([](auto value) { return eval_ctz(value); }, v[0]);
            } else if (cuda) {
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

        case OpType::Add:
            if (literal) {
                lv = jitc_eval_literal([](auto v0, auto v1) { return v0 + v1; },
                                     v[0], v[1]);
            } else if (literal_zero[0]) {
                li = dep[1];
            } else if (literal_zero[1]) {
                li = dep[0];
            } else if (cuda) {
                stmt = is_single ? "add.ftz.$t0 $r0, $r1, $r2"
                                 : "add.$t0 $r0, $r1, $r2";
            } else {
                stmt = is_float ? "$r0 = fadd <$w x $t0> $r1, $r2"
                                : "$r0 = add <$w x $t0> $r1, $r2";
            }
            break;

        case OpType::Sub:
            if (literal) {
                lv = jitc_eval_literal([](auto v0, auto v1) { return v0 - v1; },
                                      v[0], v[1]);
            } else if (literal_zero[1]) {
                li = dep[0];
            } else if (cuda) {
                stmt = is_single ? "sub.ftz.$t0 $r0, $r1, $r2"
                                 : "sub.$t0 $r0, $r1, $r2";
            } else {
                stmt = is_float ? "$r0 = fsub <$w x $t0> $r1, $r2"
                                : "$r0 = sub <$w x $t0> $r1, $r2";
            }
            break;

        case OpType::Mul:
            if (literal) {
                lv = jitc_eval_literal([](auto v0, auto v1) { return v0 * v1; },
                                       v[0], v[1]);
            } else if (literal_one[0]) {
                li = dep[1];
            } else if (literal_one[1]) {
                li = dep[0];
            } else if (is_uint && v[0]->literal && jitc_is_pow2(v[0]->value)) {
                li = jitc_var_shift(cuda, vt, OpType::Shl, dep[1], v[0]->value);
                li_decref = true;
            } else if (is_uint && v[1]->literal && jitc_is_pow2(v[1]->value)) {
                li = jitc_var_shift(cuda, vt, OpType::Shl, dep[0], v[1]->value);
                li_decref = true;
            } else if (cuda) {
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

        case OpType::Div:
            if (literal) {
                lv = jitc_eval_literal([](auto v0, auto v1) { return v0 / v1; },
                                      v[0], v[1]);
            } else if (literal_one[1]) {
                li = dep[0];
            } else if (is_uint && v[1]->literal && jitc_is_pow2(v[1]->value)) {
                li = jitc_var_shift(cuda, vt, OpType::Shr, dep[0], v[1]->value);
                li_decref = true;
            } else if (jitc_is_float(vt) && v[1]->literal) {
                uint32_t recip = jitc_var_new_op(OpType::Rcp, 1, &dep[1]);
                uint32_t deps[2] = { dep[0], recip };
                li = jitc_var_new_op(OpType::Mul, 2, deps);
                li_decref = 1;
                jitc_var_dec_ref_ext(recip);
            } else if (cuda) {
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

        case OpType::Mod:
            is_valid = jitc_is_arithmetic(vt) && !jitc_is_float(vt);
            if (literal) {
                lv = jitc_eval_literal([](auto v0, auto v1) { return eval_mod(v0, v1); },
                                      v[0], v[1]);
            } else if (cuda) {
                stmt = "rem.$t0 $r0, $r1, $r2";
            } else {
                if (is_uint)
                    stmt = "$r0 = urem <$w x $t0> $r1, $r2";
                else
                    stmt = "$r0 = srem <$w x $t0> $r1, $r2";
            }
            break;

        case OpType::Min:
            if (literal) {
                lv = jitc_eval_literal([](auto v0, auto v1) { return std::min(v0, v1); },
                                      v[0], v[1]);
            } else if (cuda) {
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

        case OpType::Max:
            if (literal) {
                lv = jitc_eval_literal([](auto v0, auto v1) { return std::max(v0, v1); },
                                      v[0], v[1]);
            } else if (cuda) {
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

        case OpType::Shr:
            is_valid = jitc_is_arithmetic(vt) && !jitc_is_float(vt);
            if (literal) {
                lv = jitc_eval_literal([](auto v0, auto v1) { return eval_shr(v0, v1); },
                                      v[0], v[1]);
            } else if (literal_zero[0] || literal_zero[1]) {
                li = dep[0];
            } else if (cuda) {
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

        case OpType::Shl:
            is_valid = jitc_is_arithmetic(vt) && !jitc_is_float(vt);
            if (literal) {
                lv = jitc_eval_literal([](auto v0, auto v1) { return eval_shl(v0, v1); },
                                      v[0], v[1]);
            } else if (literal_zero[0] || literal_zero[1]) {
                li = dep[0];
            } else if (cuda) {
                stmt = (vt == VarType::UInt32 || vt == VarType::Int32)
                           ? "shl.$b0 $r0, $r1, $r2"
                           : "cvt.u32.$t2 %r3, $r2$n"
                             "shl.$b0 $r0, $r1, %r3";
            } else {
                stmt = "$r0 = shl <$w x $t0> $r1, $r2";
            }
            break;

        case OpType::And:
            is_valid = jitc_is_not_void(vt);
            if (literal) {
                lv = jitc_eval_literal([](auto v0, auto v1) { return eval_and(v0, v1); },
                                      v[0], v[1]);
            } else if ((vt == VarType::Bool && literal_one[0]) || literal_zero[1]) {
                li = dep[1];
            } else if ((vt == VarType::Bool && literal_one[1]) || literal_zero[0]) {
                li = dep[0];
            } else if (cuda) {
                stmt = "and.$b0 $r0, $r1, $r2";
            } else {
                stmt = !is_float
                           ? "$r0 = and <$w x $t1> $r1, $r2"
                           : "$r0_0 = bitcast <$w x $t1> $r1 to <$w x $b0>$n"
                             "$r0_1 = bitcast <$w x $t2> $r2 to <$w x $b0>$n"
                             "$r0_2 = and <$w x $b0> $r0_0, $r0_1$n"
                             "$r0 = bitcast <$w x $b0> $r0_2 to <$w x $t0>";
            }
            break;

        case OpType::Or:
            is_valid = jitc_is_not_void(vt);
            if (literal) {
                lv = jitc_eval_literal([](auto v0, auto v1) { return eval_or(v0, v1); },
                                      v[0], v[1]);
            } else if ((vt == VarType::Bool && literal_one[0]) || literal_zero[1]) {
                li = dep[0];
            } else if ((vt == VarType::Bool && literal_one[1]) || literal_zero[0]) {
                li = dep[1];
            } else if (cuda) {
                stmt = "or.$b0 $r0, $r1, $r2";
            } else {
                stmt = !is_float
                           ? "$r0 = or <$w x $t1> $r1, $r2"
                           : "$r0_0 = bitcast <$w x $t1> $r1 to <$w x $b0>$n"
                             "$r0_1 = bitcast <$w x $t2> $r2 to <$w x $b0>$n"
                             "$r0_2 = or <$w x $b0> $r0_0, $r0_1$n"
                             "$r0 = bitcast <$w x $b0> $r0_2 to <$w x $t0>";
            }
            break;

        case OpType::Xor:
            is_valid = jitc_is_not_void(vt);
            if (literal) {
                lv = jitc_eval_literal([](auto v0, auto v1) { return eval_xor(v0, v1); },
                                      v[0], v[1]);
            } else if (literal_zero[0]) {
                li = dep[1];
            } else if (literal_zero[1]) {
                li = dep[0];
            } else if (cuda) {
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

        case OpType::Eq:
            is_valid = jitc_is_not_void(vt);
            vtr = VarType::Bool;
            if (literal) {
                lv = jitc_eval_literal([](auto v0, auto v1) { return v0 == v1; },
                                      v[0], v[1]);
            } else if (cuda) {
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

        case OpType::Ne:
            is_valid = jitc_is_not_void(vt);
            vtr = VarType::Bool;
            if (literal) {
                lv = jitc_eval_literal([](auto v0, auto v1) { return v0 != v1; },
                                      v[0], v[1]);
            } else if (cuda) {
                if (vt == VarType::Bool)
                    stmt = "xor.$t1 $r0, $r1, $r2";
                else
                    stmt = "setp.ne.$t1 $r0, $r1, $r2";
            } else {
                stmt = is_float ? "$r0 = fcmp one <$w x $t1> $r1, $r2"
                                : "$r0 = icmp ne <$w x $t1> $r1, $r2";
            }
            break;

        case OpType::Lt:
            vtr = VarType::Bool;
            if (literal) {
                lv = jitc_eval_literal([](auto v0, auto v1) { return v0 < v1; },
                                      v[0], v[1]);
            } else if (cuda) {
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

        case OpType::Le:
            vtr = VarType::Bool;
            if (literal) {
                lv = jitc_eval_literal([](auto v0, auto v1) { return v0 <= v1; },
                                      v[0], v[1]);
            } else if (cuda) {
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

        case OpType::Gt:
            vtr = VarType::Bool;
            if (literal) {
                lv = jitc_eval_literal([](auto v0, auto v1) { return v0 > v1; },
                                      v[0], v[1]);
            } else if (cuda) {
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

        case OpType::Ge:
            vtr = VarType::Bool;
            if (literal) {
                lv = jitc_eval_literal([](auto v0, auto v1) { return v0 >= v1; },
                                      v[0], v[1]);
            } else if (cuda) {
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

        case OpType::Fma:
            is_valid = is_float;
            if (literal) {
                lv = jitc_eval_literal(
                    [](auto v0, auto v1, auto v2) {
                        return eval_fma(v0, v1, v2);
                    },
                    v[0], v[1], v[2]);
            } else if (literal_one[0]) {
                uint32_t deps[2] = { dep[1], dep[2] };
                li = jitc_var_new_op(OpType::Add, 2, deps);
                li_decref = true;
            } else if (literal_one[1]) {
                uint32_t deps[2] = { dep[0], dep[2] };
                li = jitc_var_new_op(OpType::Add, 2, deps);
                li_decref = true;
            } else if (literal_zero[2]) {
                uint32_t deps[2] = { dep[0], dep[1] };
                li = jitc_var_new_op(OpType::Mul, 2, deps);
                li_decref = true;
            } else if (literal_zero[0] && literal_zero[1]) {
                li = dep[2];
            } else if (cuda) {
                stmt = is_single ? "fma.rn.ftz.$t0 $r0, $r1, $r2, $r3"
                                 : "fma.rn.$t0 $r0, $r1, $r2, $r3";
            } else {
                stmt = "$r0 = $call <$w x $t0> @llvm.fma.v$w$a1(<$w x $t1> "
                       "$r1, <$w x $t2> $r2, <$w x $t3> $r3)";
            }
            break;

        case OpType::Select:
            is_valid = (VarType) v[0]->type == VarType::Bool &&
                       v[1]->type == v[2]->type;
            if (literal_one[0]) {
                li = dep[1];
            } else if (literal_zero[0]) {
                li = dep[0];
            } else if (literal) {
                jitc_fail("jit_var_new_op(): select: internal error!");
            } else if (dep[1] == dep[2]) {
                li = dep[1];
            } else if (cuda) {
                stmt = (VarType) v[1]->type != VarType::Bool
                           ? "selp.$t0 $r0, $r2, $r3, $r1"
                           : "not.pred %p3, $r1$n"
                             "and.pred %p3, %p3, $r3$n"
                             "and.pred $r0, $r1, $r2$n"
                             "or.pred $r0, $r0, %p3";
            } else {
                stmt = "$r0 = select <$w x $t3> $r3, <$w x $t1> $r1, <$w x $t2> $r2";
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
        li = jitc_var_resize(li, size);
        if (li_decref)
            jitc_var_dec_ref_ext(li);
        return li;
    } else {
        Variable v2;
        v2.size = size;
        v2.type = (uint32_t) vtr;
        v2.cuda = cuda;

        if (literal) {
            v2.literal = 1;
            v2.value = lv;
        } else {
            if (!stmt) // XXX
                jitc_fail("Internal error! %s", op_name[(int) op]);
            v2.stmt = (char *) stmt;
            for (uint32_t i = 0; i < n_dep; ++i) {
                v2.dep[i] = dep[i];
                jitc_var_inc_ref_int(dep[i], v[i]);
            }
        }

        return jitc_var_new(v2);
    }
}

JITC_NOINLINE uint32_t jitc_var_new_op_fail(const char *error, OpType op, uint32_t n_dep, const uint32_t *dep) {
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
