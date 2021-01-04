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

template <typename Result, typename Func, JitBackend Backend, typename Base,
          typename... Args>
Result vcall_impl(const Func &func, const JitArray<Backend, Base *> &self,
                  const Args &... args) {
    const char *domain = "Base";
    uint32_t n_inst = jit_registry_get_max(domain);

    Result result;

    // if (n_inst == 0) { /// XXX
    //     return zero<Result>(ek::width(args));
    // } else { ... }

    ek_index_vector indices_in, indices_out_all;
    ek_vector<uint32_t> se_count(n_inst, 0);

    (collect_indices(indices_in, args), ...);

    for (uint32_t i = 1; i <= n_inst; ++i) {
        char label[128];
        snprintf(label, sizeof(label), "VCall: %s [instance %u]", domain, i);
        Base *base = (Base *) jit_registry_get_ptr(domain, i);
        se_count[i - 1] = jit_side_effects_scheduled(Backend);

        jit_prefix_push(Backend, label);
        try {
            if constexpr (std::is_same_v<Result, std::nullptr_t>) {
                func(base, args...);
            } else {
                collect_indices(indices_out_all, func(base, args...));

            }
        } catch (...) {
            /// XXX reset side effects to beginning?
            jit_prefix_pop(Backend);
            throw;
        }
        jit_prefix_pop(Backend);

    }

    ek_index_vector indices_out(indices_out_all.size() / n_inst);
    jit_var_vcall(domain, self.index(), n_inst, indices_in.size(),
                  indices_in.data(), indices_out_all.size(),
                  indices_out_all.data(), se_count.data(), indices_out.data());

    if constexpr (!std::is_same_v<Result, std::nullptr_t>) {
        uint32_t offset = 0;
        write_indices(indices_out, result, offset);
    }

    return result;
}

template <typename Func, JitBackend Backend, typename Base, typename... Args>
auto vcall(const Func &func, const JitArray<Backend, Base *> &self,
           const Args &... args) {
    using Result = decltype(func(std::declval<Base *>(), args...));
    using Result_2 = std::conditional_t<std::is_void_v<Result>, std::nullptr_t, Result>;
    return vcall_impl<Result_2>(func, self, ek::placeholder(args)...);
}

TEST_CUDA(01_symbolic_vcall) {
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

    uint32_t i1 = jit_registry_put("Base", &a1);
    uint32_t i2 = jit_registry_put("Base", &a2);
    jit_assert(i1 == 1 && i2 == 2);

    using BasePtr = Array<Base *>;
    Float x = arange<Float>(10);
    BasePtr self = arange<UInt32>(10) % 3;

    for (uint32_t i = 0; i < 2; ++i) {
        for (uint32_t j = 0; j < 2; ++j) {
            jit_set_flag(JitFlag::VCallOptimize, i);
            jit_set_flag(JitFlag::VCallBranch, j);
            Float y =
                vcall([](Base *self2, Float x2) { return self2->f(x2); }, self, x);
            jit_assert(strcmp(y.str(), "[0, 22, 204, 0, 28, 210, 0, 34, 216, 0]") == 0);
        }
    }

    jit_registry_remove(&a1);
    jit_registry_remove(&a2);
}

TEST_CUDA(02_calling_conventions) {
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

    B1 b1; B2 b2;
    uint32_t i1 = jit_registry_put("Base", &b1);
    uint32_t i2 = jit_registry_put("Base", &b2);

    for (uint32_t i = 0; i < 2; ++i) {
        for (uint32_t j = 0; j < 2; ++j) {
            jit_set_flag(JitFlag::VCallOptimize, i);
            jit_set_flag(JitFlag::VCallBranch, j);

            using BasePtr = Array<Base *>;
            BasePtr self = arange<UInt32>(10) % 3;

            Mask p0(false);
            Float p1(12);
            Double p2(34);
            Float p3(56);
            Mask p4(true);

            auto result = vcall([](Base *self2, Mask p0, Float p1, Double p2, Float p3,
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

    jit_registry_remove(&b1);
    jit_registry_remove(&b2);
}

TEST_CUDA(03_optimize_away_outputs) {
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
    uint32_t i1 = jit_registry_put("Base", &c1);
    uint32_t i2 = jit_registry_put("Base", &c2);
    uint32_t i3 = jit_registry_put("Base", &c3);
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

            jit_assert(jit_var_ref_ext(p3.index()) == 1 &&
                       jit_var_ref_int(p3.index()) == 0);

            auto result =
                vcall([](Base *self2, Float p1, Float p2, Float p3) { return self2->f(p1, p2, p3); },
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
    jit_registry_remove(&c1);
    jit_registry_remove(&c2);
    jit_registry_remove(&c3);
}

TEST_CUDA(04_devirtualize) {
    /* This test checks that outputs which produce identical values across
       all instances are moved out of the virtual call interface. */
    struct Base {
        virtual ek_tuple<Float, Float> f(Float p1, Float p2) = 0;
    };

    struct D1 : Base {
        ek_tuple<Float, Float> f(Float p1, Float p2) override {
            return { p2 + 2, p1 + 1 };
        }
    };

    struct D2 : Base {
        ek_tuple<Float, Float> f(Float p1, Float p2) override {
            return { p2 + 2, p1 + 2 };
        }
    };

    D1 d1; D2 d2;
    uint32_t i1 = jit_registry_put("Base", &d1);
    uint32_t i2 = jit_registry_put("Base", &d2);
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

                auto result =
                    vcall([](Base *self2, Float p1, Float p2) { return self2->f(p1, p2); },
                          self, p1, p2);

                jit_assert(jit_var_is_literal(result.template get<0>().index()) == ((i == 1 && k == 0) ? 1 : 0) &&
                           jit_var_is_literal(result.template get<1>().index()) == 0);

                jit_var_schedule(result.template get<0>().index());
                jit_var_schedule(result.template get<1>().index());

                jit_assert(
                    strcmp(jit_var_str(result.template get<0>().index()),
                           i == 0
                               ? "[0, 36, 36, 0, 36, 36, 0, 36, 36, 0]"
                               : "[36, 36, 36, 36, 36, 36, 36, 36, 36, 36]") == 0);
                jit_assert(strcmp(jit_var_str(result.template get<1>().index()),
                              "[0, 13, 14, 0, 13, 14, 0, 13, 14, 0]") == 0);
            }
        }
    }
    jit_registry_remove(&d1);
    jit_registry_remove(&d2);
}

TEST_CUDA(05_extra_data) {
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
    uint32_t i1 = jit_registry_put("Base", &e1);
    uint32_t i2 = jit_registry_put("Base", &e2);
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
                Float result = vcall([](Base *self2, Float x) { return self2->f(x); }, self, x);
                jit_assert(strcmp(result.str(), "[0, 9, 13, 0, 21, 28, 0, 33, 43, 0]") == 0);
            }
        }
    }
    jit_registry_remove(&e1);
    jit_registry_remove(&e2);
}


#if 0
/// 1 instance!
/// function that don't receive/return *any* arrays
/// function with only side effects
/// evaluating multiple times, and ensuring that side effects only happen once
/// sequence of multiple vcalls..
#endif
