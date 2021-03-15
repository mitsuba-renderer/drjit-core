#include "test.h"
#include <enoki-jit/containers.h>
#include <utility>

namespace ek = enoki;

namespace enoki {
    template <typename Value, enable_if_t<Value::IsArray> = 0>
    void collect_indices(ek_index_vector &indices, const Value &value) {
        indices.push_back(value.index());
    }

    template <typename Value, enable_if_t<Value::IsArray> = 0>
    void write_indices(ek_index_vector &indices, Value &value, uint32_t &offset) {
        uint32_t &index = indices[offset++];
        value = Value::steal(index);
        index = 0;
    }

    template <typename... Ts, size_t... Is>
    void collect_indices_tuple(ek_index_vector &indices,
                               const ek_tuple<Ts...> &value,
                               std::index_sequence<Is...>) {
        (collect_indices(indices, value.template get<Is>()), ...);
    }

    template <typename... Ts>
    void collect_indices(ek_index_vector &indices, const ek_tuple<Ts...> &value) {
        collect_indices_tuple(indices, value, std::make_index_sequence<sizeof...(Ts)>());
    }

    template <typename... Ts, size_t... Is>
    void write_indices_tuple(ek_index_vector &indices, ek_tuple<Ts...> &value,
                             uint32_t &offset, std::index_sequence<Is...>) {
        (write_indices(indices, value.template get<Is>(), offset), ...);
    }

    template <typename... Ts>
    void write_indices(ek_index_vector &indices, ek_tuple<Ts...> &value,
                             uint32_t &offset) {
        write_indices_tuple(indices, value, offset, std::make_index_sequence<sizeof...(Ts)>());
    }
};

namespace detail {
    inline bool extract_mask() { return true; }
    template <typename T> decltype(auto) extract_mask(const T &v) {
        /// XXX
        // if constexpr (is_mask_v<T>) {
        //     return v;
        // } else {
        //     return true;
        // }
        return true;
    }

    template <typename T, typename... Ts, enable_if_t<sizeof...(Ts) != 0> = 0>
    decltype(auto) extract_mask(const T &v, const Ts &... vs) {
        return extract_mask(vs...);
    }

    template <size_t I, size_t N, typename T>
    decltype(auto) set_mask_true(const T &v) {
        return v;
    }
};

template <typename Result, typename Func, JitBackend Backend, typename Base,
          typename... Args, size_t... Is>
Result vcall_impl(const char *domain, uint32_t n_inst, const Func &func,
                  const JitArray<Backend, Base *> &self,
                  const JitArray<Backend, bool> &mask,
                  std::index_sequence<Is...>, const Args &... args) {
    using Mask = JitArray<Backend, bool>;
    constexpr size_t N = sizeof...(Args);
    Result result;

    ek_index_vector indices_in, indices_out_all;
    ek_vector<uint32_t> se_count(n_inst + 1, 0);
    ek_vector<uint32_t> inst_id(n_inst, 0);

    (collect_indices(indices_in, args), ...);
    se_count[0] = jit_side_effects_scheduled(Backend);

    for (uint32_t i = 1; i <= n_inst; ++i) {
        char label[128];
        snprintf(label, sizeof(label), "VCall: %s [instance %u]", domain, i);
        Base *base = (Base *) jit_registry_get_ptr(Backend, domain, i);

        jit_prefix_push(Backend, label);
        int flag_before = jit_flag(JitFlag::Recording);

        if (Backend == JitBackend::LLVM) {
            Mask vcall_mask = Mask::steal(jit_var_new_stmt(
                Backend, VarType::Bool,
                "$r0 = or <$w x i1> %mask, zeroinitializer", 1, 0,
                nullptr));
            jit_var_mask_push(Backend, vcall_mask.index(), 0);
        }

        try {
            jit_set_flag(JitFlag::Recording, 1);
            if constexpr (std::is_same_v<Result, std::nullptr_t>) {
                func(base, (detail::set_mask_true<Is, N>(args))...);
            } else {
                collect_indices(indices_out_all, func(base, args...));
            }
        } catch (...) {
            jit_prefix_pop(Backend);
            jit_side_effects_rollback(Backend, se_count[0]);
            jit_set_flag(JitFlag::Recording, flag_before);
            if (Backend == JitBackend::LLVM)
                jit_var_mask_pop(Backend);
            throw;
        }
        if (Backend == JitBackend::LLVM)
            jit_var_mask_pop(Backend);
        jit_set_flag(JitFlag::Recording, flag_before);
        jit_prefix_pop(Backend);
        se_count[i] = jit_side_effects_scheduled(Backend);
        inst_id[i - 1] = i;
    }

    ek_index_vector indices_out(indices_out_all.size() / n_inst);

    Mask mask_combined =
        mask & neq(self, nullptr) & Mask::steal(jit_var_mask_peek(Backend));

    jit_var_vcall(domain, self.index(), mask_combined.index(), n_inst,
                  inst_id.data(), indices_in.size(), indices_in.data(),
                  indices_out_all.size(), indices_out_all.data(),
                  se_count.data(), indices_out.data());

    if constexpr (!std::is_same_v<Result, std::nullptr_t>) {
        uint32_t offset = 0;
        write_indices(indices_out, result, offset);
        return result;
    } else {
        return nullptr;
    }
}

template <typename Func, JitBackend Backend, typename Base,
          typename... Args>
auto vcall(const char *domain, const Func &func,
           const JitArray<Backend, Base *> &self, const Args &... args) {
    using Result = decltype(func(std::declval<Base *>(), args...));
    constexpr bool IsVoid = std::is_void_v<Result>;
    using Result_2 = std::conditional_t<IsVoid, std::nullptr_t, Result>;
    using Bool = JitArray<Backend, bool>;

    uint32_t n_inst = jit_registry_get_max(Backend, domain);

#if 0
    if (n_inst == 0) {
        if constexpr (IsVoid)
            return std::nullptr_t;
        else
            return zero<Result>(ek::width(args...));
    } else if (n_inst == 1) {
        uint32_t i = 1;
        Base *inst = nullptr;
        do {
            inst = (Base *) jit_registry_get_ptr(Backend, domain, i++);
        } while (!inst);

        if constexpr (IsVoid) {
            func(inst, args...);
            return std::nullptr_t;
        } else {
            return func(inst, args...);
        }
    }
#endif

    constexpr size_t N = sizeof...(Args);
    return vcall_impl<Result_2>(
        domain, n_inst, func, self,
        Bool(detail::extract_mask(args...)),
        std::make_index_sequence<sizeof...(Args)>(),
        ek::placeholder(args, false, true)...);
}

TEST_BOTH(01_symbolic_vcall) {
    /// Test a simple virtual function call
    struct Base {
        virtual Float f(Float x) = 0;
    };

    struct A1 : Base {
        Float f(Float x) override { return (x + 10) * 2; }
    };

    struct A2 : Base {
        Float f(Float x) override { return (x + 100) * 2; }
    };

    A1 a1;
    A2 a2;

    // jit_llvm_set_target("skylake-avx512", "+avx512f,+avx512dq,+avx512vl,+avx512cd", 16);
    uint32_t i1 = jit_registry_put(Backend, "Base", &a1);
    uint32_t i2 = jit_registry_put(Backend, "Base", &a2);
    jit_assert(i1 == 1 && i2 == 2);

    using BasePtr = Array<Base *>;
    Float x = arange<Float>(10);
    BasePtr self = arange<UInt32>(10) % 3;

    for (uint32_t i = 0; i < 2; ++i) {
        for (uint32_t j = 0; j < 2; ++j) {
            jit_set_flag(JitFlag::VCallOptimize, i);
            jit_set_flag(JitFlag::VCallBranch, j);
            if (j == 1 && Backend == JitBackend::LLVM) continue;

            Float y =
                vcall("Base", [](Base *self2, Float x2) { return self2->f(x2); }, self, x);
            jit_assert(strcmp(y.str(), "[0, 22, 204, 0, 28, 210, 0, 34, 216, 0]") == 0);
        }
    }

    jit_registry_remove(Backend, &a1);
    jit_registry_remove(Backend, &a2);
}

TEST_BOTH(02_calling_conventions) {
    /* This tests 4 things at once: passing masks, reordering inputs/outputs to
       avoid alignment issues, immediate copying of an input to an output.
       Finally, it runs twice: the second time with optimizations, which
       optimizes away all of the inputs */
    using Double = Array<double>;

    struct Base {
        virtual ek_tuple<Mask, Float, Double, Float, Mask>
        f(Mask p0, Float p1, Double p2, Float p3, Mask p4) = 0;
    };

    struct B1 : Base {
        ek_tuple<Mask, Float, Double, Float, Mask>
        f(Mask p0, Float p1, Double p2, Float p3, Mask p4) override {
            return { p0, p1, p2, p3, p4 };
        }
    };

    struct B2 : Base {
        ek_tuple<Mask, Float, Double, Float, Mask>
        f(Mask p0, Float p1, Double p2, Float p3, Mask p4) override {
            return { !p0, p1 + 1, p2 + 2, p3 + 3, false };
        }
    };

    struct B3 : Base {
        ek_tuple<Mask, Float, Double, Float, Mask>
        f(Mask p0, Float p1, Double p2, Float p3, Mask p4) override {
            return { 0, 0, 0, 0, 0 };
        }
    };

    B1 b1; B2 b2; B3 b3;

    uint32_t i1 = jit_registry_put(Backend, "Base", &b1);
    uint32_t i2 = jit_registry_put(Backend, "Base", &b2);
    uint32_t i3 = jit_registry_put(Backend, "Base", &b3);

    for (uint32_t i = 0; i < 2; ++i) {
        for (uint32_t j = 0; j < 2; ++j) {
            jit_set_flag(JitFlag::VCallOptimize, i);
            jit_set_flag(JitFlag::VCallBranch, j);
            if (j == 1 && Backend == JitBackend::LLVM) continue;

            using BasePtr = Array<Base *>;
            BasePtr self = arange<UInt32>(10) % 3;

            Mask p0(false);
            Float p1(12);
            Double p2(34);
            Float p3(56);
            Mask p4(true);

            auto result = vcall("Base", [](Base *self2, Mask p0, Float p1, Double p2, Float p3,
                                   Mask p4) { return self2->f(p0, p1, p2, p3, p4); },
                                self, p0, p1, p2, p3, p4);

            jit_var_schedule(result.template get<0>().index());
            jit_var_schedule(result.template get<1>().index());
            jit_var_schedule(result.template get<2>().index());
            jit_var_schedule(result.template get<3>().index());
            jit_var_schedule(result.template get<4>().index());

            jit_assert(strcmp(result.template get<0>().str(), "[0, 0, 1, 0, 0, 1, 0, 0, 1, 0]") == 0);
            jit_assert(strcmp(result.template get<1>().str(), "[0, 12, 13, 0, 12, 13, 0, 12, 13, 0]") == 0);
            jit_assert(strcmp(result.template get<2>().str(), "[0, 34, 36, 0, 34, 36, 0, 34, 36, 0]") == 0);
            jit_assert(strcmp(result.template get<3>().str(), "[0, 56, 59, 0, 56, 59, 0, 56, 59, 0]") == 0);
            jit_assert(strcmp(result.template get<4>().str(), "[0, 1, 0, 0, 1, 0, 0, 1, 0, 0]") == 0);
        }
    }

    jit_registry_remove(Backend, &b1);
    jit_registry_remove(Backend, &b2);
    jit_registry_remove(Backend, &b3);
}

TEST_BOTH(03_optimize_away_outputs) {
    /* This test checks that unreferenced outputs are detected by the virtual
       function call interface, and that garbage collection propagates from
       outputs to inputs. It also checks that functions with identical code are
       collapsed, and that inputs which aren't referenced in the first place
       get optimized away. */
    struct Base {
        virtual ek_tuple<Float, Float> f(Float p1, Float p2, Float p3) = 0;
    };

    struct C12 : Base {
        ek_tuple<Float, Float> f(Float p1, Float p2, Float /* p3 */) override {
            return { p2 + 2.34567f, p1 + 1.f };
        }
    };

    struct C3 : Base {
        ek_tuple<Float, Float> f(Float p1, Float p2, Float /* p3 */) override {
            return { p2 + 1.f, p1 + 2.f };
        }
    };

    C12 c1; C12 c2; C3 c3;
    uint32_t i1 = jit_registry_put(Backend, "Base", &c1);
    uint32_t i2 = jit_registry_put(Backend, "Base", &c2);
    uint32_t i3 = jit_registry_put(Backend, "Base", &c3);
    jit_assert(i1 == 1 && i2 == 2 && i3 == 3);

    Float p1 = ek::opaque<Float>(12);
    Float p2 = ek::opaque<Float>(34);
    Float p3 = ek::opaque<Float>(56);

    using BasePtr = Array<Base *>;
    BasePtr self = arange<UInt32>(10) % 4;

    for (uint32_t i = 0; i < 2; ++i) {
        for (uint32_t j = 0; j < 2; ++j) {
            jit_set_flag(JitFlag::VCallOptimize, i);
            jit_set_flag(JitFlag::VCallBranch, j);
            if (j == 1 && Backend == JitBackend::LLVM) continue;

            jit_assert(jit_var_ref_ext(p3.index()) == 1 &&
                       jit_var_ref_int(p3.index()) == 0);

            auto result =
                vcall("Base", [](Base *self2, Float p1, Float p2, Float p3) { return self2->f(p1, p2, p3); },
                      self, p1, p2, p3);

            jit_assert(jit_var_ref_ext(p1.index()) == 1 &&
                       jit_var_ref_int(p1.index()) == 2);
            jit_assert(jit_var_ref_ext(p2.index()) == 1 &&
                       jit_var_ref_int(p2.index()) == 2);

            // Irrelevant input optimized away
            jit_assert(jit_var_ref_ext(p3.index()) == 1 &&
                       jit_var_ref_int(p3.index()) == 1 - i);

            result.template get<0>() = Float(0);

            jit_assert(jit_var_ref_ext(p1.index()) == 1 &&
                       jit_var_ref_int(p1.index()) == 2);
            jit_assert(jit_var_ref_ext(p2.index()) == 1 &&
                       jit_var_ref_int(p2.index()) == 2 - 2*i);
            jit_assert(jit_var_ref_ext(p3.index()) == 1 &&
                       jit_var_ref_int(p3.index()) == 1 - i);

            jit_assert(strcmp(jit_var_str(result.template get<1>().index()),
                              "[0, 13, 13, 14, 0, 13, 13, 14, 0, 13]") == 0);
        }
    }
    jit_registry_remove(Backend, &c1);
    jit_registry_remove(Backend, &c2);
    jit_registry_remove(Backend, &c3);
}

TEST_BOTH(04_devirtualize) {
    /* This test checks that outputs which produce identical values across
       all instances are moved out of the virtual call interface. */
    struct Base {
        virtual ek_tuple<Float, Float, Float> f(Float p1, Float p2) = 0;
    };

    struct D1 : Base {
        ek_tuple<Float, Float, Float> f(Float p1, Float p2) override {
            return { p2 + 2, p1 + 1, 0 };
        }
    };

    struct D2 : Base {
        ek_tuple<Float, Float, Float> f(Float p1, Float p2) override {
            return { p2 + 2, p1 + 2, 0 };
        }
    };

    D1 d1; D2 d2;
    uint32_t i1 = jit_registry_put(Backend, "Base", &d1);
    uint32_t i2 = jit_registry_put(Backend, "Base", &d2);
    jit_assert(i1 == 1 && i2 == 2);

    using BasePtr = Array<Base *>;
    BasePtr self = arange<UInt32>(10) % 3;

    for (uint32_t k = 0; k < 2; ++k) {
        Float p1, p2;
        if (k == 0) {
            p1 = 12;
            p2 = 34;
        } else {
            p1 = ek::opaque<Float>(12);
            p2 = ek::opaque<Float>(34);
        }

        for (uint32_t i = 0; i < 2; ++i) {
            for (uint32_t j = 0; j < 2; ++j) {
                jit_set_flag(JitFlag::VCallOptimize, i);
                jit_set_flag(JitFlag::VCallBranch, j);
                if (j == 1 && Backend == JitBackend::LLVM)
                    continue;

                auto result =
                    vcall("Base", [](Base *self2, Float p1, Float p2) { return self2->f(p1, p2); },
                          self, p1, p2);

                Mask mask_combined =
                    Mask(true) & neq(self, nullptr) & Mask::steal(jit_var_mask_peek(Backend));

                Float alt = (p2 + 2) & mask_combined;

                if (i == 1 && k == 0)
                    jit_assert(result.template get<0>().index() == alt.index());
                jit_assert(jit_var_is_literal(result.template get<2>().index()) == (i == 1));

                jit_var_schedule(result.template get<0>().index());
                jit_var_schedule(result.template get<1>().index());

                jit_assert(
                    strcmp(jit_var_str(result.template get<0>().index()),
                               "[0, 36, 36, 0, 36, 36, 0, 36, 36, 0]") == 0);
                jit_assert(strcmp(jit_var_str(result.template get<1>().index()),
                              "[0, 13, 14, 0, 13, 14, 0, 13, 14, 0]") == 0);
            }
        }
    }
    jit_registry_remove(Backend, &d1);
    jit_registry_remove(Backend, &d2);
}

TEST_BOTH(05_extra_data) {
    using Double = Array<double>;

    /// Ensure that evaluated scalar fields in instances can be accessed
    struct Base {
        virtual Float f(Float) = 0;
    };

    struct E1 : Base {
        Double local_1 = 4;
        Float local_2 = 5;
        Float f(Float x) override { return Float(Double(x) * local_1) + local_2; }
    };

    struct E2 : Base {
        Float local_1 = 3;
        Double local_2 = 5;
        Float f(Float x) override { return local_1 + Float(Double(x) * local_2); }
    };

    E1 e1; E2 e2;
    uint32_t i1 = jit_registry_put(Backend, "Base", &e1);
    uint32_t i2 = jit_registry_put(Backend, "Base", &e2);
    jit_assert(i1 == 1 && i2 == 2);

    using BasePtr = Array<Base *>;
    BasePtr self = arange<UInt32>(10) % 3;
    Float x = arange<Float>(10);

    for (uint32_t k = 0; k < 2; ++k) {
        if (k == 1) {
            e1.local_1.eval();
            e1.local_2.eval();
            e2.local_1.eval();
            e2.local_2.eval();
        }

        for (uint32_t i = 0; i < 2; ++i) {
            for (uint32_t j = 0; j < 2; ++j) {
                jit_set_flag(JitFlag::VCallOptimize, i);
                jit_set_flag(JitFlag::VCallBranch, j);
                if (j == 1 && Backend == JitBackend::LLVM) continue;
                Float result = vcall("Base", [](Base *self2, Float x) { return self2->f(x); }, self, x);
                jit_assert(strcmp(result.str(), "[0, 9, 13, 0, 21, 28, 0, 33, 43, 0]") == 0);
            }
        }
    }
    jit_registry_remove(Backend, &e1);
    jit_registry_remove(Backend, &e2);
}

TEST_BOTH(06_side_effects) {
    /*  This tests three things:
       - side effects in virtual functions
       - functions without inputs/outputs
       - functions with *only* side effects
    */

    struct Base {
        virtual void go() = 0;
    };

    struct F1 : Base {
        Float buffer = zero<Float>(5);
        void go() override {
            scatter_reduce(ReduceOp::Add, buffer, Float(1), UInt32(1));
            scatter_reduce(ReduceOp::Add, buffer, Float(2), UInt32(3));
        }
    };

    struct F2 : Base {
        Float buffer = arange<Float>(4);
        void go() override {
            scatter_reduce(ReduceOp::Add, buffer, Float(1), UInt32(2));
        }
    };

    using BasePtr = Array<Base *>;
    BasePtr self = arange<UInt32>(11) % 3;

    for (uint32_t i = 0; i < 2; ++i) {
        for (uint32_t j = 0; j < 2; ++j) {
            jit_set_flag(JitFlag::VCallOptimize, i);
            jit_set_flag(JitFlag::VCallBranch, j);
            if (j == 1 && Backend == JitBackend::LLVM) continue;

            F1 f1; F2 f2;
            uint32_t i1 = jit_registry_put(Backend, "Base", &f1);
            uint32_t i2 = jit_registry_put(Backend, "Base", &f2);
            jit_assert(i1 == 1 && i2 == 2);

            vcall("Base", [](Base *self2) { self2->go(); }, self);
            jit_assert(strcmp(f1.buffer.str(), "[0, 4, 0, 8, 0]") == 0);
            jit_assert(strcmp(f2.buffer.str(), "[0, 1, 5, 3]") == 0);

            jit_registry_remove(Backend, &f1);
            jit_registry_remove(Backend, &f2);
            jit_registry_trim();
        }
    }
}

TEST_BOTH(07_side_effects_only_once) {
    /* This tests ensures that side effects baked into a function only happen
       once, even when that function is evaluated multiple times. */

    struct Base {
        virtual ek_tuple<Float, Float> f() = 0;
    };

    struct G1 : Base {
        Float buffer = zero<Float>(5);
        ek_tuple<Float, Float> f() override {
            scatter_reduce(ReduceOp::Add, buffer, Float(1), UInt32(1));
            return { 1, 2 };
        }
    };

    struct G2 : Base {
        Float buffer = zero<Float>(5);
        ek_tuple<Float, Float> f() override {
            scatter_reduce(ReduceOp::Add, buffer, Float(1), UInt32(2));
            return { 2, 1 };
        }
    };

    using BasePtr = Array<Base *>;
    BasePtr self = arange<UInt32>(11) % 3;

    for (uint32_t i = 0; i < 2; ++i) {
        for (uint32_t j = 0; j < 2; ++j) {
            jit_set_flag(JitFlag::VCallOptimize, i);
            jit_set_flag(JitFlag::VCallBranch, j);
            if (j == 1 && Backend == JitBackend::LLVM) continue;

            G1 g1; G2 g2;
            uint32_t i1 = jit_registry_put(Backend, "Base", &g1);
            uint32_t i2 = jit_registry_put(Backend, "Base", &g2);
            jit_assert(i1 == 1 && i2 == 2);

            auto result = vcall("Base", [](Base *self2) { return self2->f(); }, self);
            Float f1 = result.template get<0>();
            Float f2 = result.template get<1>();
            jit_assert(strcmp(f1.str(), "[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1]") == 0);
            jit_assert(strcmp(g1.buffer.str(), "[0, 4, 0, 0, 0]") == 0);
            jit_assert(strcmp(g2.buffer.str(), "[0, 0, 3, 0, 0]") == 0);
            jit_assert(strcmp(f2.str(), "[0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2]") == 0);
            jit_assert(strcmp(g1.buffer.str(), "[0, 4, 0, 0, 0]") == 0);
            jit_assert(strcmp(g2.buffer.str(), "[0, 0, 3, 0, 0]") == 0);

            jit_registry_remove(Backend, &g1);
            jit_registry_remove(Backend, &g2);
            jit_registry_trim();
        }
    }
}

TEST_BOTH(08_multiple_calls) {
    /* This tests ensures that a function can be called several times,
       reusing the generated code (at least in the function-based variant).
       This reuse cannot be verified automatically via assertions, you must
       look at the generated code or ensure consistency via generated .ref
       files!*/

    struct Base {
        virtual Float f(Float) = 0;
    };

    struct H1 : Base {
        Float f(Float x) override {
            return x + Float(1);
        }
    };

    struct H2 : Base {
        Float f(Float x) override {
            return x + Float(2);
        }
    };

    using BasePtr = Array<Base *>;
    BasePtr self = arange<UInt32>(10) % 3;
    Float x = opaque<Float>(10, 1);

    H1 h1; H2 h2;
    uint32_t i1 = jit_registry_put(Backend, "Base", &h1);
    uint32_t i2 = jit_registry_put(Backend, "Base", &h2);
    jit_assert(i1 == 1 && i2 == 2);


    for (uint32_t i = 0; i < 2; ++i) {
        for (uint32_t j = 0; j < 2; ++j) {
            jit_set_flag(JitFlag::VCallOptimize, i);
            jit_set_flag(JitFlag::VCallBranch, j);
            if (j == 1 && Backend == JitBackend::LLVM) continue;

            Float y = vcall("Base", [](Base *self2, Float x2) { return self2->f(x2); }, self, x);
            Float z = vcall("Base", [](Base *self2, Float x2) { return self2->f(x2); }, self, y);

            jit_assert(strcmp(z.str(), "[0, 12, 14, 0, 12, 14, 0, 12, 14, 0]") == 0);
        }
    }

    jit_registry_remove(Backend, &h1);
    jit_registry_remove(Backend, &h2);
}

TEST_BOTH(09_big) {
    /* This performs two vcalls with different numbers of instances, and
       relatively many of them. This tests the various tables, offset
       calculations, binary search trees, etc. */

    struct Base1 { virtual Float f() = 0; };
    struct Base2 { virtual Float f() = 0; };

    struct I1 : Base1 {
        Float v;
        Float f() override { return v; }
    };

    struct I2 : Base2 {
        Float v;
        Float f() override { return v; }
    };

    const int n1 = 71, n2 = 123, n = 125;
    I1 v1[n1];
    I2 v2[n2];
    uint32_t i1[n1];
    uint32_t i2[n2];

    for (int i = 0; i < n1; ++i) {
        v1[i].v = i;
        i1[i] = jit_registry_put(Backend, "Base1", &v1[i]);
    }

    for (int i = 0; i < n2; ++i) {
        v2[i].v = 100 + i;
        i2[i] = jit_registry_put(Backend, "Base2", &v2[i]);
    }

    using Base1Ptr = Array<Base1 *>;
    using Base2Ptr = Array<Base2 *>;
    UInt32 self1 = arange<UInt32>(n + 1);
    UInt32 self2 = arange<UInt32>(n + 1);

    self1 = select(self1 <= n1, self1, 0);
    self2 = select(self2 <= n2, self2, 0);

    for (uint32_t i = 0; i < 2; ++i) {
        for (uint32_t j = 0; j < 2; ++j) {
            jit_set_flag(JitFlag::VCallOptimize, i);
            jit_set_flag(JitFlag::VCallBranch, j);
            if (j == 1 && Backend == JitBackend::LLVM) continue;

            Float x = vcall("Base1", [](Base1 *self_) { return self_->f(); }, Base1Ptr(self1));
            Float y = vcall("Base2", [](Base2 *self_) { return self_->f(); }, Base2Ptr(self2));

            jit_var_schedule(x.index());
            jit_var_schedule(y.index());

            jit_assert(x.read(0) == 0);
            jit_assert(y.read(0) == 0);

            for (uint32_t i = 1; i <= n1; ++i)
                jit_assert(x.read(i) == i - 1);
            for (uint32_t i = 1; i <= n2; ++i)
                jit_assert(y.read(i) == 100 + i - 1);

            for (uint32_t i = n1 + 1; i < n; ++i)
                jit_assert(x.read(i + 1) == 0);
            for (uint32_t i = n2 + 1; i < n; ++i)
                jit_assert(y.read(i + 1) == 0);
        }
    }

    for (int i = 0; i < n1; ++i)
        jit_registry_remove(Backend, &v1[i]);
    for (int i = 0; i < n2; ++i)
        jit_registry_remove(Backend, &v2[i]);
}
