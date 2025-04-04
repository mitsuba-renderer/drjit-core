#if defined(_MSC_VER)
#  pragma warning (disable:4127) // conditional expression is constant
#endif

#include "test.h"
#include <initializer_list>
#include <cmath>
#include <cstring>
#include <typeinfo>

TEST_BOTH(01_creation_destruction_cse) {
    // Test CSE involving normal and evaluated constant literals
    for (int i = 0; i < 2; ++i) {
        uint32_t value = 1234;
        uint32_t v0 = jit_var_literal(Backend, VarType::UInt32, &value, i + 1);
        uint32_t v1 = jit_var_literal(Backend, VarType::UInt32, &value, i + 1);
        uint32_t v2 = jit_var_literal(Backend, VarType::UInt32, &value, i + 1, 1);
        uint32_t v3 = jit_var_literal(Backend, VarType::UInt32, &value, i + 1, 1);

        jit_assert(v0 == v1 && v0 != v3 && v2 != v3);
        for (uint32_t l : { v0, v1, v2, v3 })
            jit_assert(strcmp(jit_var_str(l),
                               i == 0 ? "[1234]" : "[1234, 1234]") == 0);

        jit_var_dec_ref(v0);
        jit_var_dec_ref(v1);
        jit_var_dec_ref(v2);
        jit_var_dec_ref(v3);
    }
}

TEST_BOTH(02_load_store) {
    /// Test CSE and simple variables loads/stores involving scalar and non-scalars
    for (int k = 0; k < 2; ++k) {
        for (int j = 0; j < 2; ++j) {
            uint32_t value = 1;
            uint32_t o = jit_var_literal(Backend, VarType::UInt32, &value, 1 + k);

            if (j == 1)
                jit_var_eval(o);

            for (int i = 0; i < 2; ++i) {
                uint32_t value2 = 1234;
                uint32_t v0 = jit_var_literal(Backend, VarType::UInt32, &value2, 1 + i);
                uint32_t v1 = jit_var_literal(Backend, VarType::UInt32, &value2, 1 + i);

                uint32_t v0p = jit_var_add(o, v0);
                jit_var_dec_ref(v0);
                uint32_t v1p = jit_var_add(o, v1);
                jit_var_dec_ref(v1);

                jit_assert(v0p == v1p);

                for (uint32_t l : { v0p, v1p })
                    jit_assert(strcmp(jit_var_str(l),
                                       (k == 0 && i == 0) ? "[1235]"
                                                          : "[1235, 1235]") == 0);
                jit_var_dec_ref(v0p);
                jit_var_dec_ref(v1p);
            }

            jit_var_dec_ref(o);
        }
    }
}

TEST_BOTH(03_load_store_mask) {
    /// Masks are a bit more tricky, check that those also work
    for (int i = 0; i < 2; ++i) {
        uint32_t ctr = jit_var_counter(Backend, i == 0 ? 1 : 10);
        uint32_t one_v = 1, one = jit_var_literal(Backend, VarType::UInt32, &one_v);
        uint32_t zero_v = 0, zero = jit_var_literal(Backend, VarType::UInt32, &zero_v);
        uint32_t odd = jit_var_and(ctr, one);
        uint32_t mask = jit_var_eq(odd, zero);

        jit_assert(strcmp(jit_var_str(mask),
                          i == 0 ? "[1]" : "[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]") == 0);

        uint32_t flip = jit_var_not(mask);
        jit_assert(strcmp(jit_var_str(flip),
                          i == 0 ? "[0]" : "[0, 1, 0, 1, 0, 1, 0, 1, 0, 1]") == 0);

        jit_var_dec_ref(flip);
        jit_var_dec_ref(ctr);
        jit_var_dec_ref(one);
        jit_var_dec_ref(zero);
        jit_var_dec_ref(odd);
        jit_var_dec_ref(mask);
    }
}

TEST_BOTH(04_load_store_float) {
    /// Check usage of halfs/floats/doubles (loading, storing, literal constants)
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
                uint32_t v0, v1;

                if (i == 0) {
                    float f1 = 1, f1234 = 1234;
                    v0 = jit_var_literal(Backend, VarType::Float32, &f1, 1 + j, k);
                    v1 = jit_var_literal(Backend, VarType::Float32, &f1234, 1 + j);
                } else if (i == 1) {
                    double d1 = 1, d1234 = 1234;
                    v0 = jit_var_literal(Backend, VarType::Float64, &d1, 1 + j, k);
                    v1 = jit_var_literal(Backend, VarType::Float64, &d1234, 1 + j);
                } else {
                    half h1 = 1, h1234 = 1234;
                    v0 = jit_var_literal(Backend, VarType::Float16, &h1.value, 1 + j, k);
                    v1 = jit_var_literal(Backend, VarType::Float16, &h1234.value, 1 + j);
                }

                uint32_t v2 = jit_var_add(v0, v1);

                jit_assert(strcmp(jit_var_str(v2),
                                   j == 0 ? "[1235]" : "[1235, 1235]") == 0);

                jit_var_dec_ref(v0);
                jit_var_dec_ref(v1);
                jit_var_dec_ref(v2);
            }
        }
    }
}

const char *op_name[(int) JitOp::Count] {
    // Common unary operations
    "neg", "not", "sqrt", "abs",

    // Common binary arithmetic operations
    "add", "sub", "mul", "div", "mod",

    // High multiplication
    "mulhi",

    // Fused multiply-add
    "fma",

    // Minimum, maximum
    "min", "max",

    // Rounding operations
    "ceil", "floor", "round", "trunc",

    // Comparisons
    "eq", "neq", "lt", "le", "gt", "ge",

    // Ternary operator
    "select",

    // Bit-level counting operations
    "popc", "clz", "ctz",

    /// Bit-wise operations
    "and", "or", "xor",

    // Shifts
    "shl", "shr",

    // Fast approximations
    "rcp", "rsqrt",

    // Multi-function generator (CUDA)
    "sin", "cos", "exp2", "log2",
};

template <typename T> bool test_const_prop() {
    using Value = typename T::Value;
    constexpr JitBackend Backend = T::Backend;
    constexpr bool IsFloat = drjit::detail::is_floating_point_v<Value>;
    constexpr bool IsSigned = drjit::detail::is_signed_v<Value>;
    constexpr bool IsMask = std::is_same<Value, bool>::value;
    constexpr bool IsInt = !IsFloat && !IsMask;
    constexpr size_t Size = IsMask ? 2 : 10, Size2 = 2 * Size;

    bool fast_math = jit_flags() & (uint32_t) JitFlag::FastMath;
    jit_set_flag(JitFlag::FastMath, false);

    bool fail = false;
    Value values[Size2];
    for (uint32_t i = 0; i < Size2; ++i) {
        int j = i % Size;
        values[i] = (Value) ((IsMask || !IsSigned) ? j : (j - 4));
        if constexpr (IsFloat) {
            if (values[i] < Value(-1) || values[i] > Value(1))
                values[i] = (Value) (1.1f * values[i]);
        }
    }

    uint32_t in[Size2] { }, out[Size2 * Size2 * Size2] { };

    // ===============================================================
    //  Test unary operations
    // ===============================================================

    for (JitOp op :
         { JitOp::Not, JitOp::Neg, JitOp::Abs, JitOp::Sqrt, JitOp::Rcp,
           JitOp::Rsqrt, JitOp::Ceil, JitOp::Floor, JitOp::Round,
           JitOp::Trunc, JitOp::Popc, JitOp::Clz, JitOp::Ctz }) {

        if ((op == JitOp::Popc || op == JitOp::Clz || op == JitOp::Ctz) && !IsInt)
            continue;
        else if (op == JitOp::Not && !IsMask && !IsInt)
            continue;
        else if ((op == JitOp::Sqrt || op == JitOp::Rcp ||
                  op == JitOp::Rsqrt || op == JitOp::Ceil ||
                  op == JitOp::Floor || op == JitOp::Round ||
                  op == JitOp::Trunc) && !IsFloat)
            continue;
        else if ((op == JitOp::Neg || op == JitOp::Abs) && IsMask)
            continue;

        for (uint32_t i = 0; i < Size2; ++i)
            in[i] = jit_var_literal(Backend, T::Type, values + i, 1, i >= Size);

        for (uint32_t i = 0; i < Size2; ++i) {
            uint32_t index = 0;
            if ((op == JitOp::Rcp) && values[i] == 0) {
                index = jit_var_inc_ref(in[i]);
            } else {
                index = jit_var_op(op, in + i);
            }

            if (i < Size)
                jit_assert(jit_var_state(index) == VarState::Literal);
            else
                jit_var_schedule(index);

            out[i] = index;
        }

        jit_eval();

        for (uint32_t i = 0; i < Size2; ++i) {
            int ir = i < Size ? i : i - Size;
            int value_id = out[i];
            int ref_id = out[ir];

            if (value_id == ref_id)
                continue;

            Value value, ref;
            jit_var_read(value_id, 0, &value);
            jit_var_read(ref_id, 0, &ref);

            if (memcmp(&value, &ref, sizeof(Value)) != 0) {
                if (std::isnan((double) value) &&
                    std::isnan((double) ref))
                    continue;
                if ((op == JitOp::Sqrt || op == JitOp::Rcp ||
                     op == JitOp::Rsqrt) &&
                    double(value - ref) < 1e-7)
                    continue;

                // FIXME: On R515.43 OptiX incorrectly handles `rsqrt(0)` (NaN instead of Inf)
                Value arg;
                jit_var_read(in[ir], 0, &arg);
                if (arg == 0 && op == JitOp::Rsqrt && jit_flag(JitFlag::ForceOptiX))
                    continue;

                char *v0 = strdup(jit_var_str(in[ir]));
                char *v1 = strdup(jit_var_str(value_id));
                char *v2 = strdup(jit_var_str(ref_id));
                fprintf(stderr, "Mismatch: op(%s, %s) == %s vs const %s, type %s\n",
                        op_name[(int) op], v0, v1, v2, typeid(T).name());
                free(v0); free(v1); free(v2);
                fail = true;
            }
        }

        for (uint32_t i = 0; i < Size2 * Size2; ++i)
            jit_var_dec_ref(out[i]);

        for (uint32_t i = 0; i < Size2; ++i)
            jit_var_dec_ref(in[i]);
    }

    // ===============================================================
    //  Test binary operations
    // ===============================================================

    for (JitOp op :
         { JitOp::Add, JitOp::Sub, JitOp::Mul, JitOp::Div,
           JitOp::Mod, JitOp::Min, JitOp::Max, JitOp::And, JitOp::Or,
           JitOp::Xor, JitOp::Shl, JitOp::Shr, JitOp::Eq, JitOp::Neq,
           JitOp::Lt, JitOp::Le, JitOp::Gt, JitOp::Ge, JitOp::Mulhi }) {
        if (op == JitOp::Mod && IsFloat)
            continue;
        if ((IsFloat || IsMask) && (op == JitOp::Shl || op == JitOp::Shr))
            continue;
        if (IsMask && !(op == JitOp::Or || op == JitOp::And || op == JitOp::Xor))
            continue;
        if (!IsInt && op == JitOp::Mulhi)
            continue;

        for (uint32_t i = 0; i < Size2; ++i)
            in[i] = jit_var_literal(Backend, T::Type, values + i, 1, i >= Size);

        for (uint32_t i = 0; i < Size2; ++i) {
            if constexpr (IsSigned) {
                if (op == JitOp::Shl && values[i] < 0)
                    continue;
            }
            for (uint32_t k = 0; k < Size2; ++k) {
                uint32_t deps[2] = { in[i], in[k] };
                uint32_t index = 0;

                if (((op == JitOp::Div || op == JitOp::Mod) && values[k] == 0) ||
                    ((op == JitOp::Shr || op == JitOp::Shl) && values[k] < Value(0))) {
                    index = jit_var_inc_ref(in[k]);
                } else {
                    index = jit_var_op(op, deps);
                }

                if (i < Size && k < Size) {
                    jit_assert(jit_var_state(index) == VarState::Literal);
                    jit_assert(jit_var_state(index) != VarState::Evaluated);
                } else {
                    jit_var_schedule(index);
                }

                out[i * Size2 + k] = index;
            }
        }

        jit_eval();

        for (uint32_t i = 0; i < Size2; ++i) {
            if constexpr (IsSigned) {
                if (op == JitOp::Shl && values[i] < 0)
                    continue;
            }
            for (uint32_t j = 0; j < Size2; ++j) {
                int ir = i < Size ? i : i - Size;
                int jr = j < Size ? j : j - Size;
                int value_id = out[i * Size2 + j];
                int ref_id = out[ir * Size2 + jr];

                if (value_id == ref_id)
                    continue;

                Value value = 0, ref = 0;
                jit_var_read(value_id, 0, &value);
                jit_var_read(ref_id, 0, &ref);

                if (memcmp(&value, &ref, sizeof(Value)) != 0) {
                    if (op == JitOp::Div && double(value - ref) < 1e-6)
                        continue;
                    char *v0 = strdup(jit_var_str(in[ir]));
                    char *v1 = strdup(jit_var_str(in[jr]));
                    char *v2 = strdup(jit_var_str(value_id));
                    char *v3 = strdup(jit_var_str(ref_id));
                    fprintf(stderr, "Mismatch: op(%s, %s, %s) == %s vs const %s\n",
                           op_name[(int) op], v0, v1, v2, v3);
                    free(v0); free(v1); free(v2); free(v3);
                    fail = true;
                }
            }
        }

        for (uint32_t i = 0; i < Size2; ++i) {
            if constexpr (IsSigned) {
                if (op == JitOp::Shl && values[i] < 0)
                    continue;
            }
            for (uint32_t j = 0; j < Size2; ++j)
                jit_var_dec_ref(out[i * Size2 + j]);
        }
        for (uint32_t i = 0; i < Size2; ++i)
            jit_var_dec_ref(in[i]);
    }

    // ===============================================================
    //  Test ternary operations
    // ===============================================================

    const uint32_t Small = Size < 4 ? 2 : (Size / 2);
    const uint32_t Small2 = Small * 2;

    uint32_t in_b[4] { };

    for (uint32_t i = 0; i < Small2; ++i) {
        int j = i % Small;
        values[i] = (Value) ((IsMask || !IsSigned) ? j : (j - 2));
        if constexpr (IsFloat) {
            if (values[i] < Value(-1) || values[i] > Value(1))
                values[i] = (Value) (1.1f * values[i]);
        }
    }

    for (JitOp op : { JitOp::Fma, JitOp::Select }) {
        if (op == JitOp::Fma && IsMask)
            continue;

        for (uint32_t i = 0; i < 4; ++i) {
            bool b = i & 1;
            in_b[i] = jit_var_literal(Backend, VarType::Bool, &b, 1, i >= Small);
        }

        for (uint32_t i = 0; i < Small2; ++i)
            in[i] = jit_var_literal(Backend, T::Type, values + i, 1, i >= Small);

        memset(out, 0, Small2 * Small2 * Small2 * sizeof(uint32_t));

        for (uint32_t i = 0; i < (op == JitOp::Select ? 4 : Small2); ++i) {
            for (uint32_t j = 0; j < Small2; ++j) {
                for (uint32_t k = 0; k < Small2; ++k) {
                    uint32_t deps[3] = { op == JitOp::Select ? in_b[i] : in[i], in[j], in[k] };
                    uint32_t index = jit_var_op(op, deps);

                    jit_var_schedule(index);
                    out[k + Small2 * (j + Small2 * i)] = index;
                }
            }
        }

        jit_eval();

        for (uint32_t i = 0; i < (op == JitOp::Select ? 4 : Small2); ++i) {
            for (uint32_t j = 0; j < Small2; ++j) {
                for (uint32_t k = 0; k < Small2; ++k) {
                    int ir;
                    if (op == JitOp::Select)
                         ir = i < 2 ? i : i - 2;
                    else
                         ir = i < Small ? i : i - Small;
                    int jr = j < Small ? j : j - Small;
                    int kr = k < Small ? k : k - Small;
                    uint32_t value_id = out[k + Small2 * (j + Small2 * i)];
                    uint32_t ref_id = out[kr + Small2 * (jr + Small2 * ir)];

                    if (value_id == ref_id)
                        continue;

                    Value value, ref;
                    jit_var_read(value_id, 0, &value);
                    jit_var_read(ref_id, 0, &ref);

                    if (memcmp(&value, &ref, sizeof(Value)) != 0) {
                        if (op == JitOp::Fma && value == ref)
                            continue;
                        char *v0 = strdup(jit_var_str(op == JitOp::Select ? in_b[ir] : in[ir]));
                        char *v1 = strdup(jit_var_str(in[jr]));
                        char *v2 = strdup(jit_var_str(in[kr]));
                        char *v3 = strdup(jit_var_str(value_id));
                        char *v4 = strdup(jit_var_str(ref_id));
                        fprintf(stderr, "Mismatch: op(%s, %s, %s, %s) == %s vs const %s\n",
                                op_name[(int) op], v0, v1, v2, v3, v4);
                        free(v0); free(v1); free(v2); free(v3); free(v4);
                        fail = true;
                    }
                }
            }
        }

        for (uint32_t i = 0; i < Small2 * Small2 * Small2; ++i)
            jit_var_dec_ref(out[i]);

        for (uint32_t i = 0; i < Small2; ++i)
            jit_var_dec_ref(in[i]);

        for (int i = 0; i < 4; ++i)
            jit_var_dec_ref(in_b[i]);
    }

    jit_set_flag(JitFlag::FastMath, fast_math);

    return fail;
}

TEST_BOTH_FLOAT_AGNOSTIC(05_const_prop) {
    /* This very large test runs every implemented operation with a variety of
       scalar and memory inputs and compares their output. This is to ensure
       that Dr.Jit's builtin constant propagation pass produces results
       that are equivalent to the native implementation. */
    bool fail = false;

    using Float32 = typename Float::template ReplaceValue<float>;
    using Float16 = typename Float::template ReplaceValue<drjit::half>;

    fail |= test_const_prop<Float32>();
    fail |= test_const_prop<Float16>();
    fail |= test_const_prop<Array<double>>();
    fail |= test_const_prop<UInt32>();
    fail |= test_const_prop<Int32>();
    fail |= test_const_prop<Array<int64_t>>();
    fail |= test_const_prop<Array<uint64_t>>();
    fail |= test_const_prop<Mask>();
    jit_assert(!fail);
}

TEST_BOTH_FLOAT_AGNOSTIC(06_cast) {
    /* This test tries every possible type conversion, verifying constant
       propagation to the native CUDA/LLVM implementation */

    VarType types[] {
        VarType::Float16,
        VarType::Float32,
        VarType::Float64,
        VarType::Int32,
        VarType::UInt32,
        VarType::Int64,
        VarType::UInt64,
        VarType::Bool
    };
    const char *type_names[(int) VarType::Count]{
        "Void",   "Bool",  "Int8",   "UInt8",   "Int16",   "UInt16",  "Int32",
        "UInt32", "Int64", "UInt64", "Pointer", "(BaseFloat)", "Float16", "Float32", "Float64"
    };

    size_t type_sizes[(int) VarType::Count]{
        0, 1, 1, 1, 2, 2, 4, 4, 8, 8, 8, 0, 2, 4, 8
    };

    uint32_t source_value[20];
    uint32_t target_value[20];
    bool fail = false;

    for (int reinterpret = 0; reinterpret < 2; ++reinterpret) {
        for (VarType source_type : types) {
            for (VarType target_type : types) {
                bool test_sign =
                    source_type != VarType::UInt32 &&
                    source_type != VarType::UInt64 &&
                    source_type != VarType::Bool &&
                    target_type != VarType::UInt32 &&
                    target_type != VarType::UInt64 &&
                    target_type != VarType::Bool;

                if (type_sizes[(int) source_type] == 0 ||
                    type_sizes[(int) target_type] == 0)
                    continue;

                if (reinterpret && type_sizes[(int) source_type] !=
                                   type_sizes[(int) target_type])
                    continue;

                int size = source_type == VarType::Bool ? 2 : 10;

                for (int i = 0; i < size * 2; ++i) {
                    int j = i % size;
                    int64_t value = j;
                    if (test_sign)
                        value -= 4;
                    if (source_type == VarType::Float32) {
                        float f = (float) value;
                        value = 0;
                        memcpy(&value, &f, sizeof(float));
                        if (std::abs(f) > 2.f)
                            f += f * .1f;
                    } else if (source_type == VarType::Float64) {
                        double d = (double) value;
                        memcpy(&value, &d, sizeof(double));
                        if (std::abs(d) > 2.0)
                            d += d * .1;
                    } else if (source_type == VarType::Float16) {
                        drjit::half h((float) value);
                        memcpy(&value, &h, sizeof(drjit::half));
                        if (std::abs(h) > 2.f)
                            h += h * drjit::half(.1f);
                    }

                    source_value[i] = jit_var_literal(Backend, source_type,
                                                          &value, 1, i < size);
                    target_value[i] = jit_var_cast(
                        source_value[i], target_type, reinterpret);
                    jit_var_schedule(target_value[i]);
                }
                jit_eval();

                for (int i = 0; i < size; ++i) {
                    int ref_id = target_value[i],
                      value_id = target_value[i + size];
                    uint64_t value = 0, ref = 0;
                    jit_var_read(value_id, 0, &value);
                    jit_var_read(ref_id, 0, &ref);

                    if (value != ref) {
                        char *v0 = strdup(jit_var_str(source_value[i]));
                        char *v1 = strdup(jit_var_str(value_id));
                        char *v2 = strdup(jit_var_str(ref_id));
                        fprintf(stderr,
                                "Mismatch: %scast(source_type=%s, "
                                "target_type=%s, in=%s): computed=%s vs const=%s\n",
                                reinterpret ? "reinterpret_" : "",
                                type_names[(uint32_t) source_type],
                                type_names[(uint32_t) target_type],
                                v0, v1, v2);
                        free(v0); free(v1); free(v2);
                        fail = true;
                    }
                }

                for (int i = 0; i < size * 2; ++i) {
                    jit_var_dec_ref(source_value[i]);
                    jit_var_dec_ref(target_value[i]);
                }
            }
        }
    }
    jit_assert(!fail);
}

TEST_BOTH_FLOAT_AGNOSTIC(07_and_or_mixed) {
    // Tests JitOp::And/Or applied to a non-mask type and a mask

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 2; ++j) {
            bool b = (i & 1);
            uint32_t v0 = jit_var_literal(Backend, VarType::Bool,
                                          &b, 1, i < 2);

            uint32_t u = 1234;
            uint32_t v1 = jit_var_literal(Backend, VarType::UInt32,
                                          &u, 1, j == 0);

            float f = 1234;
            uint32_t v2 = jit_var_literal(Backend, VarType::Float32,
                                          &f, 1, j == 0);

            uint32_t v3 = jit_var_and(v1, v0);
            uint32_t v4 = jit_var_and(v2, v0);
            uint32_t v5 = jit_var_or(v1, v0);
            uint32_t v6 = jit_var_or(v2, v0);

            jit_var_dec_ref(v0);
            jit_var_dec_ref(v1);
            jit_var_dec_ref(v2);

            jit_var_schedule(v3);
            jit_var_schedule(v4);
            jit_var_schedule(v5);
            jit_var_schedule(v6);

            uint32_t out_u = 0;
            float out_f = 0;
            jit_var_read(v3, 0, &out_u);
            jit_var_read(v4, 0, &out_f);
            jit_var_dec_ref(v3);
            jit_var_dec_ref(v4);

            jit_assert(out_u == (b ? 1234u : 0u));
            jit_assert(out_f == (b ? 1234u : 0u));

            jit_var_read(v5, 0, &out_u);
            jit_var_read(v6, 0, &out_f);

            jit_assert(out_u == (b ? 0xFFFFFFFF : 1234));
            jit_assert(b ? std::isnan(out_f) : (out_f == 1234));

            jit_var_dec_ref(v5);
            jit_var_dec_ref(v6);
        }
    }
}
