#define EK_OPNAME 1
#include "test.h"
#include <initializer_list>
#include <cmath>
#include <cstring>

TEST_BOTH(01_creation_destruction_cse) {
    // Test CSE involving normal and evaluated constant literals
    for (int i = 0; i < 2; ++i) {
        uint32_t value = 1234;
        uint32_t v0 = jit_var_new_literal(Backend, VarType::UInt32, &value, i + 1);
        uint32_t v1 = jit_var_new_literal(Backend, VarType::UInt32, &value, i + 1);
        uint32_t v2 = jit_var_new_literal(Backend, VarType::UInt32, &value, i + 1, 1);
        uint32_t v3 = jit_var_new_literal(Backend, VarType::UInt32, &value, i + 1, 1);

        jit_assert(v0 == v1 && v0 != v3 && v2 != v3);
        for (uint32_t l : { v0, v1, v2, v3 })
            jit_assert(strcmp(jit_var_str(l),
                               i == 0 ? "[1234]" : "[1234, 1234]") == 0);

        jit_var_dec_ref_ext(v0);
        jit_var_dec_ref_ext(v1);
        jit_var_dec_ref_ext(v2);
        jit_var_dec_ref_ext(v3);
    }
}

TEST_BOTH(02_load_store) {
    /// Test CSE and simple variables loads/stores involving scalar and non-scalars
    for (int k = 0; k < 2; ++k) {
        for (int j = 0; j < 2; ++j) {
            uint32_t value = 1;
            uint32_t o = jit_var_new_literal(Backend, VarType::UInt32, &value, 1 + k);

            if (j == 1)
                jit_var_eval(o);

            for (int i = 0; i < 2; ++i) {
                uint32_t value = 1234;
                uint32_t v0 = jit_var_new_literal(Backend, VarType::UInt32, &value, 1 + i);
                uint32_t v1 = jit_var_new_literal(Backend, VarType::UInt32, &value, 1 + i);

                uint32_t v0p = jit_var_new_op_2(OpType::Add, o, v0);
                jit_var_dec_ref_ext(v0);
                uint32_t v1p = jit_var_new_op_2(OpType::Add, o, v1);
                jit_var_dec_ref_ext(v1);

                jit_assert(v0p == v1p);

                for (uint32_t l : { v0p, v1p })
                    jit_assert(strcmp(jit_var_str(l),
                                       (k == 0 && i == 0) ? "[1235]"
                                                          : "[1235, 1235]") == 0);
                jit_var_dec_ref_ext(v0p);
                jit_var_dec_ref_ext(v1p);
            }

            jit_var_dec_ref_ext(o);
        }
    }
}

TEST_BOTH(03_load_store_mask) {
    /// Masks are a bit more tricky, check that those also work
    for (int i = 0; i < 2; ++i) {
        uint32_t ctr = jit_var_new_counter(Backend, i == 0 ? 1 : 10);
        uint32_t one_v = 1, one = jit_var_new_literal(Backend, VarType::UInt32, &one_v);
        uint32_t zero_v = 0, zero = jit_var_new_literal(Backend, VarType::UInt32, &zero_v);
        uint32_t odd = jit_var_new_op_2(OpType::And, ctr, one);
        uint32_t mask = jit_var_new_op_2(OpType::Eq, odd, zero);

        jit_assert(strcmp(jit_var_str(mask),
                           i == 0 ? "[1]" : "[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]") == 0);

        uint32_t flip = jit_var_new_op_1(OpType::Not, mask);
        jit_assert(strcmp(jit_var_str(flip),
                           i == 0 ? "[0]" : "[0, 1, 0, 1, 0, 1, 0, 1, 0, 1]") == 0);

        jit_var_dec_ref_ext(flip);
        jit_var_dec_ref_ext(ctr);
        jit_var_dec_ref_ext(one);
        jit_var_dec_ref_ext(zero);
        jit_var_dec_ref_ext(odd);
        jit_var_dec_ref_ext(mask);
    }
}

TEST_BOTH(04_load_store_float) {
    /// Check usage of floats/doubles (loading, storing, literals)
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
                uint32_t v0, v1;
                VarType vt;

                if (i == 0) {
                    float f1 = 1, f1234 = 1234;
                    v0 = jit_var_new_literal(Backend, VarType::Float32, &f1, 1 + j, k);
                    v1 = jit_var_new_literal(Backend, VarType::Float32, &f1234, 1 + j);
                } else {
                    double d1 = 1, d1234 = 1234;
                    v0 = jit_var_new_literal(Backend, VarType::Float64, &d1, 1 + j, k);
                    v1 = jit_var_new_literal(Backend, VarType::Float64, &d1234, 1 + j);
                }

                uint32_t v2 = jit_var_new_op_2(OpType::Add, v0, v1);

                jit_assert(strcmp(jit_var_str(v2),
                                   j == 0 ? "[1235]" : "[1235, 1235]") == 0);

                jit_var_dec_ref_ext(v0);
                jit_var_dec_ref_ext(v1);
                jit_var_dec_ref_ext(v2);
            }
        }
    }
}

template <typename T> void test_const_prop() {
    using Value = typename T::Value;
    constexpr JitBackend Backend = T::Backend;
    constexpr bool IsFloat = std::is_floating_point<Value>::value;
    constexpr bool IsSigned = std::is_signed<Value>::value;
    constexpr bool IsMask = std::is_same<Value, bool>::value;
    constexpr bool IsInt = !IsFloat && !IsMask;
    constexpr size_t Size = IsMask ? 2 : 10, Size2 = 2 * Size;

    bool fail = false;
    Value values[Size2];
    for (int i = 0; i < Size2; ++i) {
        int j = i % Size;
        values[i] = (Value) (IsMask || !IsSigned) ? j : (j - 4);
        if (IsFloat && (values[i] < -1 || values[i] > 1))
            values[i] = (Value) (1.1f * values[i]);
    }

    uint32_t in[Size2] { }, out[Size2 * Size2 * Size2] { };

    // ===============================================================
    //  Test scalar operations
    // ===============================================================

    for (OpType op :
         { OpType::Not, OpType::Neg, OpType::Abs, OpType::Sqrt, OpType::Rcp,
           OpType::Rsqrt, OpType::Ceil, OpType::Floor, OpType::Round,
           OpType::Trunc, OpType::Popc, OpType::Clz, OpType::Ctz }) {

        if ((op == OpType::Popc || op == OpType::Clz || op == OpType::Ctz) && !IsInt)
            continue;
        else if (op == OpType::Not && !IsMask && !IsInt)
            continue;
        else if ((op == OpType::Sqrt || op == OpType::Rcp ||
                  op == OpType::Rsqrt || op == OpType::Ceil ||
                  op == OpType::Floor || op == OpType::Round ||
                  op == OpType::Trunc) && !IsFloat)
            continue;
        else if ((op == OpType::Neg || op == OpType::Abs) && IsMask)
            continue;

        for (int i = 0; i < Size2; ++i)
            in[i] = jit_var_new_literal(Backend, T::Type, values + i, 1, i >= Size);

        for (uint32_t i = 0; i < Size2; ++i) {
            uint32_t index = 0;
            if ((op == OpType::Rcp) && values[i] == 0) {
                index = in[i];
                jit_var_inc_ref_ext(index);
            } else {
                index = jit_var_new_op(op, 1, in + i);
            }

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
                if ((op == OpType::Sqrt || op == OpType::Rcp ||
                     op == OpType::Rsqrt) &&
                    (value - ref) < 1e-7)
                    continue;
                char *v0 = strdup(jit_var_str(in[ir]));
                char *v1 = strdup(jit_var_str(value_id));
                char *v2 = strdup(jit_var_str(ref_id));
                fprintf(stderr, "Mismatch: op(%s, %s) == %s vs %s\n",
                        op_name[(int) op], v0, v1, v2);
                free(v0); free(v1); free(v2);
                fail = true;
            }
        }

        for (int i = 0; i < Size2 * Size2; ++i)
            jit_var_dec_ref_ext(out[i]);

        for (int i = 0; i < Size2; ++i)
            jit_var_dec_ref_ext(in[i]);
    }

    // ===============================================================
    //  Test binary operations
    // ===============================================================

    for (OpType op :
         { OpType::Add, OpType::Sub, OpType::Mul, OpType::Div,
           OpType::Mod, OpType::Min, OpType::Max, OpType::And, OpType::Or,
           OpType::Xor, OpType::Shl, OpType::Shr, OpType::Eq, OpType::Neq,
           OpType::Lt, OpType::Le, OpType::Gt, OpType::Ge }) {
        if (op == OpType::Mod && IsFloat)
            continue;
        if ((IsFloat || IsMask) && (op == OpType::Shl || op == OpType::Shr))
            continue;
        if (IsMask && !(op == OpType::Or || op == OpType::And || op == OpType::Xor))
            continue;

        for (int i = 0; i < Size2; ++i)
            in[i] = jit_var_new_literal(Backend, T::Type, values + i, 1, i >= Size);

        for (uint32_t i = 0; i < Size2; ++i) {
            for (uint32_t j = 0; j < Size2; ++j) {
                uint32_t deps[2] = { in[i], in[j] };
                uint32_t index = 0;

                if (((op == OpType::Div || op == OpType::Mod) && values[j] == 0) ||
                    ((op == OpType::Shr || op == OpType::Shl) && values[j] < 0)) {
                    index = in[j];
                    jit_var_inc_ref_ext(index);
                } else {
                    index = jit_var_new_op(op, 2, deps);
                }

                jit_var_schedule(index);
                out[i * Size2 + j] = index;
            }
        }

        jit_eval();

        for (uint32_t i = 0; i < Size2; ++i) {
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
                    if (op == OpType::Div && (value - ref) < 1e-6)
                        continue;
                    char *v0 = strdup(jit_var_str(in[ir]));
                    char *v1 = strdup(jit_var_str(in[jr]));
                    char *v2 = strdup(jit_var_str(value_id));
                    char *v3 = strdup(jit_var_str(ref_id));
                    fprintf(stderr, "Mismatch: op(%s, %s, %s) == %s vs %s\n",
                            op_name[(int) op], v0, v1, v2, v3);
                    free(v0); free(v1); free(v2); free(v3);
                    fail = true;
                }
            }
        }

        for (int i = 0; i < Size2 * Size2; ++i)
            jit_var_dec_ref_ext(out[i]);

        for (int i = 0; i < Size2; ++i)
            jit_var_dec_ref_ext(in[i]);
    }

    // ===============================================================
    //  Test ternary operations
    // ===============================================================

    const uint32_t Small = Size / 2;
    const uint32_t Small2 = Small * 2;

    uint32_t in_b[4] { };

    for (int i = 0; i < Small2; ++i) {
        int j = i % Small;
        values[i] = (Value) (IsMask || !IsSigned) ? j : (j - 2);
        if (IsFloat && (values[i] < -1 || values[i] > 1))
            values[i] = (Value) (1.1f * values[i]);
    }

    for (OpType op : { OpType::Fmadd, OpType::Select }) {
        if (op == OpType::Fmadd && IsMask)
            continue;

        for (int i = 0; i < 4; ++i) {
            bool b = i & 1;
            in_b[i] = jit_var_new_literal(Backend, VarType::Bool, &b, 1, i >= Small);
        }

        for (int i = 0; i < Small2; ++i)
            in[i] = jit_var_new_literal(Backend, T::Type, values + i, 1, i >= Small);

        memset(out, 0, Small2 * Small2 * Small2 * sizeof(uint32_t));
        for (uint32_t i = 0; i < (op == OpType::Select ? 4 : Small2); ++i) {
            for (uint32_t j = 0; j < Small2; ++j) {
                for (uint32_t k = 0; k < Small2; ++k) {
                    uint32_t deps[3] = { op == OpType::Select ? in_b[i] : in[i], in[j], in[k] };
                    uint32_t index = jit_var_new_op(op, 3, deps);

                    jit_var_schedule(index);
                    out[k + Small2 * (j + Small2 * i)] = index;
                }
            }
        }

        jit_eval();

        for (uint32_t i = 0; i < (op == OpType::Select ? 4 : Small2); ++i) {
            for (uint32_t j = 0; j < Small2; ++j) {
                for (uint32_t k = 0; k < Small2; ++k) {
                    int ir;
                    if (op == OpType::Select)
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
                        if (op == OpType::Fmadd && value == ref)
                            continue;
                        char *v0 = strdup(jit_var_str(op == OpType::Select ? in_b[ir] : in[ir]));
                        char *v1 = strdup(jit_var_str(in[jr]));
                        char *v2 = strdup(jit_var_str(in[kr]));
                        char *v3 = strdup(jit_var_str(value_id));
                        char *v4 = strdup(jit_var_str(ref_id));
                        fprintf(stderr, "Mismatch: op(%s, %s, %s, %s) == %s vs %s\n",
                                op_name[(int) op], v0, v1, v2, v3, v4);
                        free(v0); free(v1); free(v2); free(v3); free(v4);
                        fail = true;
                    }
                }
            }
        }

        for (int i = 0; i < Small2 * Small2 * Small2; ++i)
            jit_var_dec_ref_ext(out[i]);

        for (int i = 0; i < Small2; ++i)
            jit_var_dec_ref_ext(in[i]);

        for (int i = 0; i < 4; ++i)
            jit_var_dec_ref_ext(in_b[i]);
    }

    jit_assert(!fail);
}

TEST_BOTH(05_const_prop) {
    test_const_prop<Float>();
    test_const_prop<Array<double>>();
    test_const_prop<UInt32>();
    test_const_prop<Int32>();
    test_const_prop<Array<int64_t>>();
    test_const_prop<Array<uint64_t>>();
    test_const_prop<Mask>();
}

TEST_BOTH(06_select) {
}
