#include "test.h"
#include "traits.h"
#include <drjit-core/containers.h>
#include <utility>

namespace dr = drjit;

namespace drjit {
namespace detail {
    struct dr_index_vector : dr_vector<uint32_t> {
        using Base = dr_vector<uint32_t>;
        using Base::Base;
        using Base::operator=;

        dr_index_vector(size_t size) : Base(size, 0) { }
        ~dr_index_vector() { clear(); }

        void push_back(uint32_t value) {
            jit_var_inc_ref_impl(value);
            Base::push_back(value);
        }

        void clear() {
            for (size_t i = 0; i < size(); ++i)
                jit_var_dec_ref_impl(operator[](i));
            Base::clear();
        }
    };

    template <typename Value, enable_if_t<Value::IsArray> = 0>
    void collect_indices(dr_index_vector &indices, const Value &value) {
        indices.push_back(value.index());
    }

    template <typename Value, enable_if_t<Value::IsArray> = 0>
    void update_indices(dr_index_vector &indices, Value &value, uint32_t &offset) {
        uint32_t &index = indices[offset++];
        value = Value::steal(index);
        index = 0;
    }

    template <typename... Ts, size_t... Is>
    void collect_indices_tuple(dr_index_vector &indices,
                               const dr_tuple<Ts...> &value,
                               std::index_sequence<Is...>) {
        (collect_indices(indices, value.template get<Is>()), ...);
    }

    template <typename... Ts>
    void collect_indices(dr_index_vector &indices, const dr_tuple<Ts...> &value) {
        collect_indices_tuple(indices, value, std::make_index_sequence<sizeof...(Ts)>());
    }

    template <typename... Ts, size_t... Is>
    void update_indices_tuple(dr_index_vector &indices, dr_tuple<Ts...> &value,
                             uint32_t &offset, std::index_sequence<Is...>) {
        (update_indices(indices, value.template get<Is>(), offset), ...);
    }

    template <typename... Ts>
    void update_indices(dr_index_vector &indices, dr_tuple<Ts...> &value,
                             uint32_t &offset) {
        update_indices_tuple(indices, value, offset, std::make_index_sequence<sizeof...(Ts)>());
    }

    inline bool extract_mask() { return true; }
    template <typename T> decltype(auto) extract_mask(const T &) {
        return true;
    }

    template <typename T, typename... Ts, enable_if_t<sizeof...(Ts) != 0> = 0>
    decltype(auto) extract_mask(const T &, const Ts &... vs) {
        return extract_mask(vs...);
    }

    template <size_t I, size_t N, typename T>
    decltype(auto) set_mask_true(const T &v) {
        return v;
    }

    template <typename T> T wrap_vcall(const T &value) {
        if constexpr (array_depth_v<T> > 1) {
            T result;
            for (size_t i = 0; i < value.derived().size(); ++i)
                result.derived().entry(i) = wrap_vcall(value.derived().entry(i));
            return result;
        } else if constexpr (is_diff_array_v<T>) {
            return wrap_vcall(value.detach_());
        } else if constexpr (is_jit_array_v<T>) {
            return T::steal(jit_var_wrap_vcall(value.index()));
        } else if constexpr (is_drjit_struct_v<T>) {
            T result;
            struct_support_t<T>::apply_2(
                result, value,
                [](auto &x, const auto &y) {
                    x = wrap_vcall(y);
                });
            return result;
        } else {
            return (const T &) value;
        }
    }

    struct scoped_set_mask {
        scoped_set_mask(JitBackend backend, uint32_t index) : backend(backend) {
            jit_var_mask_push(backend, index);
            jit_var_dec_ref(index);
        }

        ~scoped_set_mask() {
            jit_var_mask_pop(backend);
        }

        JitBackend backend;
    };

    struct scoped_set_self {
        scoped_set_self(JitBackend backend, uint32_t value, uint32_t self_index = 0)
            : m_backend(backend) {
            jit_vcall_self(backend, &m_self_value, &m_self_index);
            jit_var_inc_ref(m_self_index);
            jit_vcall_set_self(m_backend, value, self_index);
        }

        ~scoped_set_self() {
            jit_vcall_set_self(m_backend, m_self_value, m_self_index);
            jit_var_dec_ref(m_self_index);
        }

    private:
        JitBackend m_backend;
        uint32_t m_self_value;
        uint32_t m_self_index;
    };

    struct scoped_record {
        scoped_record(JitBackend backend, const char *name) : backend(backend) {
            checkpoint = jit_record_begin(backend, name);
            scope = jit_new_scope(backend);
        }

        uint32_t checkpoint_and_rewind() {
            jit_set_scope(backend, scope);
            return jit_record_checkpoint(backend);
        }

        ~scoped_record() {
            jit_record_end(backend, checkpoint);
        }

        JitBackend backend;
        uint32_t checkpoint, scope;
    };
} // namespace detail
} // namespace drjit

template <typename Result, typename Func, JitBackend Backend, typename Base,
          typename... Args, size_t... Is>
Result vcall_impl(const char *domain, uint32_t n_inst, const Func &func,
                  const JitArray<Backend, Base *> &self,
                  const JitArray<Backend, bool> &mask,
                  std::index_sequence<Is...>, const Args &... args) {
    using Mask = JitArray<Backend, bool>;
    constexpr size_t N = sizeof...(Args);
    (void) N;
    Result result;

    detail::dr_index_vector indices_in, indices_out_all;
    dr_vector<uint32_t> state(n_inst + 1, 0);
    dr_vector<uint32_t> inst_id(n_inst, 0);

    (detail::collect_indices(indices_in, args), ...);

    {
        detail::scoped_record rec(Backend, "Test");

        state[0] = jit_record_checkpoint(Backend);

        for (uint32_t i = 1; i <= n_inst; ++i) {
            char label[128];
            snprintf(label, sizeof(label), "VCall: %s [instance %u]", domain, i);
            Base *base = (Base *) jit_registry_ptr(Backend, domain, i);
            detail::scoped_set_mask mask_guard(Backend, jit_var_vcall_mask(Backend));
            detail::scoped_set_self self_guard(Backend, i);

            if constexpr (std::is_same_v<Result, std::nullptr_t>)
                func(base, (detail::set_mask_true<Is, N>(args))...);
            else
                detail::collect_indices(indices_out_all, func(base, args...));

            state[i] = jit_record_checkpoint(Backend);

            inst_id[i - 1] = i;
        }
    }

    detail::dr_index_vector indices_out(indices_out_all.size() / n_inst);

    uint32_t se = jit_var_vcall(
        domain, self.index(), mask.index(), n_inst, inst_id.data(),
        (uint32_t) indices_in.size(), indices_in.data(),
        (uint32_t) indices_out_all.size(), indices_out_all.data(), state.data(),
        indices_out.data());

    jit_var_mark_side_effect(se);
    jit_new_scope(Backend);

    if constexpr (!std::is_same_v<Result, std::nullptr_t>) {
        uint32_t offset = 0;
        detail::update_indices(indices_out, result, offset);
        return result;
    } else {
        (void) result;
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

    uint32_t n_inst = jit_registry_id_bound(Backend, domain);

    return vcall_impl<Result_2>(
        domain, n_inst, func, self,
        Bool(detail::extract_mask(args...)),
        std::make_index_sequence<sizeof...(Args)>(),
        detail::wrap_vcall(args)...);
}


TEST_BOTH(01_recorded_vcall) {
    /// Test a simple virtual function call
    struct Base {
        virtual Float f(Float x) = 0;
    };

    struct A1 : Base {
        Float f(Float x) override {
            return (x + 10) * 2;
        }
    };

    struct A2 : Base {
        Float f(Float x) override {
            return (x + 100) * 2;
        }
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
        jit_set_flag(JitFlag::VCallOptimize, i);
        Float y = vcall(
            "Base", [](Base *self2, Float x2) { return self2->f(x2); }, self, x);
        jit_assert(strcmp(y.str(), "[0, 22, 204, 0, 28, 210, 0, 34, 216, 0]") == 0);
    }

    jit_registry_remove(&a1);
    jit_registry_remove(&a2);
}

TEST_BOTH(02_calling_conventions) {
    /* This tests 4 things at once: passing masks, reordering inputs/outputs to
       avoid alignment issues, immediate copying of an input to an output.
       Finally, it runs twice: the second time with optimizations, which
       optimizes away all of the inputs */
    using Double = Array<double>;

    struct Base {
        virtual dr_tuple<Mask, Float, Double, Float, Mask>
        f(Mask p0, Float p1, Double p2, Float p3, Mask p4) = 0;
    };

    struct B1 : Base {
        dr_tuple<Mask, Float, Double, Float, Mask>
        f(Mask p0, Float p1, Double p2, Float p3, Mask p4) override {
            return { p0, p1, p2, p3, p4 };
        }
    };

    struct B2 : Base {
        dr_tuple<Mask, Float, Double, Float, Mask>
        f(Mask p0, Float p1, Double p2, Float p3, Mask) override {
            return { !p0, p1 + Float(1), p2 + Double(2), p3 + Float(3), false };
        }
    };

    struct B3 : Base {
        dr_tuple<Mask, Float, Double, Float, Mask>
        f(Mask, Float, Double, Float, Mask) override {
            return { 0, Float(0), Double(0), Float(0), 0 };
        }
    };

    B1 b1; B2 b2; B3 b3;

    uint32_t i1 = jit_registry_put(Backend, "Base", &b1);
    uint32_t i2 = jit_registry_put(Backend, "Base", &b2);
    uint32_t i3 = jit_registry_put(Backend, "Base", &b3);
    (void) i1; (void) i2; (void) i3;

    for (uint32_t i = 0; i < 2; ++i) {
        jit_set_flag(JitFlag::VCallOptimize, i);

        using BasePtr = Array<Base *>;
        BasePtr self = arange<UInt32>(10) % 3;

        Mask p0(false);
        Float p1(12);
        Double p2(34);
        Float p3(56);
        Mask p4(true);

        auto result = vcall(
            "Base",
            [](Base *self2, Mask p0, Float p1, Double p2, Float p3, Mask p4) {
                return self2->f(p0, p1, p2, p3, p4);
            },
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

    jit_registry_remove(&b1);
    jit_registry_remove(&b2);
    jit_registry_remove(&b3);
}

TEST_BOTH(03_optimize_away_outputs) {
    /* This test checks that unreferenced outputs are detected by the virtual
       function call interface, and that garbage collection propagates from
       outputs to inputs. It also checks that functions with identical code are
       collapsed, and that inputs which aren't referenced in the first place
       get optimized away. */
    struct Base {
        virtual dr_tuple<Float, Float> f(Float p1, Float p2, Float& p3) = 0;
    };

    struct C12 : Base {
        dr_tuple<Float, Float> f(Float p1, Float p2, Float& /* p3 */) override {
            return { p2 + Float(2.34567f), p1 + Float(1.f) };
        }
    };

    struct C3 : Base {
        dr_tuple<Float, Float> f(Float p1, Float p2, Float& /* p3 */) override {
            return { p2 + Float(1.f), p1 + Float(2.f) };
        }
    };

    C12 c1; C12 c2; C3 c3;
    uint32_t i1 = jit_registry_put(Backend, "Base", &c1);
    uint32_t i2 = jit_registry_put(Backend, "Base", &c2);
    uint32_t i3 = jit_registry_put(Backend, "Base", &c3);
    jit_assert(i1 == 1 && i2 == 2 && i3 == 3);

    Float p1 = dr::opaque<Float>(12);
    Float p2 = dr::opaque<Float>(34);
    Float p3 = dr::opaque<Float>(56);

    using BasePtr = Array<Base *>;
    BasePtr self = arange<UInt32>(10) % 4;

    for (uint32_t i = 0; i < 2; ++i) {
        jit_set_flag(JitFlag::VCallOptimize, i);

        jit_assert(jit_var_ref(p3.index()) == 1);
        jit_set_log_level_stderr((LogLevel) 10);

        auto result = vcall(
            "Base",
            [](Base *self2, Float p1, Float p2, Float p3) {
                return self2->f(p1, p2, p3);
            },
            self, p1, p2, p3);

        jit_assert(jit_var_ref(p1.index()) == 3);
        jit_assert(jit_var_ref(p2.index()) == 3);

        // Irrelevant input optimized away
        jit_assert(jit_var_ref(p3.index()) == 2 - i);

        result.template get<0>() = Float(0);

        jit_assert(jit_var_ref(p1.index()) == 3);
        jit_assert(jit_var_ref(p2.index()) == 3 - 2*i);
        jit_assert(jit_var_ref(p3.index()) == 2 - i);

        jit_assert(strcmp(jit_var_str(result.template get<1>().index()),
                            "[0, 13, 13, 14, 0, 13, 13, 14, 0, 13]") == 0);
    }

    jit_registry_remove(&c1);
    jit_registry_remove(&c2);
    jit_registry_remove(&c3);
}

TEST_BOTH(04_devirtualize) {
    /* This test checks that outputs which produce identical values across
       all instances are moved out of the virtual call interface. */
    struct Base {
        virtual dr_tuple<Float, Float, Float> f(Float p1, Float p2) = 0;
    };

    struct D1 : Base {
        dr_tuple<Float, Float, Float> f(Float p1, Float p2) override {
            return { p2 + Float(2), p1 + Float(1), Float(0) };
        }
    };

    struct D2 : Base {
        dr_tuple<Float, Float, Float> f(Float p1, Float p2) override {
            return { p2 + Float(2), p1 + Float(2), Float(0) };
        }
    };

    D1 d1; D2 d2;
    uint32_t i1 = jit_registry_put(Backend, "Base", &d1);
    uint32_t i2 = jit_registry_put(Backend, "Base", &d2);
    jit_assert(i1 == 1 && i2 == 2);

    using BasePtr = Array<Base *>;
    BasePtr self = arange<UInt32>(10) % 3;

    for (uint32_t k = 0; k < 2; ++k) {
        for (uint32_t i = 0; i < 2; ++i) {
            Float p1, p2;
            if (k == 0) {
                p1 = Float(12);
                p2 = Float(34);
            } else {
                p1 = dr::opaque<Float>(12);
                p2 = dr::opaque<Float>(34);
            }

            jit_set_flag(JitFlag::VCallOptimize, i);
            uint32_t scope = jit_scope(Backend);

            scoped_set_log_level x((LogLevel) 10);
            auto result = vcall(
                "Base",
                [](Base *self2, Float p1, Float p2) {
                    return self2->f(p1, p2);
                },
                self, p1, p2);

            jit_set_scope(Backend, scope + 1);

            Float p1_wrap = Float::steal(jit_var_wrap_vcall(p1.index()));
            Float p2_wrap = Float::steal(jit_var_wrap_vcall(p2.index()));

            Mask mask = neq(self, nullptr),
                 mask_combined = Mask::steal(jit_var_mask_apply(mask.index(), 10));

            Float alt0 = (p2_wrap + Float(2)) & mask_combined;
            Float alt1 = (p1_wrap + Float(2)) & mask_combined;
            Float alt2 = Float(0) & mask_combined;

            jit_set_scope(Backend, scope + 2);

            jit_assert((result.template get<0>().index() == alt0.index()) == ((i == 1) && (k == 0)));
            jit_assert((result.template get<1>().index() != alt1.index()));
            jit_assert((result.template get<2>().index() == alt2.index()) == (i == 1));

            jit_assert((jit_var_state(result.template get<2>().index()) == VarState::Literal) == (i == 1));

            jit_var_schedule(result.template get<0>().index());
            jit_var_schedule(result.template get<1>().index());

            jit_assert(
                strcmp(jit_var_str(result.template get<0>().index()),
                            "[0, 36, 36, 0, 36, 36, 0, 36, 36, 0]") == 0);
            jit_assert(strcmp(jit_var_str(result.template get<1>().index()),
                            "[0, 13, 14, 0, 13, 14, 0, 13, 14, 0]") == 0);
        }
    }
    jit_registry_remove(&d1);
    jit_registry_remove(&d2);
}

TEST_BOTH(05_extra_data) {
    using Double = Array<double>;

    /// Ensure that evaluated scalar fields in instances can be accessed
    struct Base {
        virtual Float f(Float) = 0;
    };

    struct E1 : Base {
        Double local_1 = Double(4);
        Float local_2 = Float(5);
        Float f(Float x) override { return Float(Double(x) * local_1) + local_2; }
    };

    struct E2 : Base {
        Float local_1 = Float(3);
        Double local_2 = Float(5);
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
            jit_set_flag(JitFlag::VCallOptimize, i);
            Float result = vcall(
                "Base", [](Base *self2, Float x) { return self2->f(x); }, self,
                x);
            jit_assert(strcmp(result.str(), "[0, 9, 13, 0, 21, 28, 0, 33, 43, 0]") == 0);
        }
    }
    jit_registry_remove(&e1);
    jit_registry_remove(&e2);
}

TEST_BOTH_FP32(06_side_effects) {
    /*  This tests three things:
       - side effects in virtual functions
       - functions without inputs/outputs
       - functions with *only* side effects
    */

    struct Base {
        virtual void go() = 0;
    };

    struct F1 : Base {
        Float buffer = zeros<Float>(5);
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
        jit_set_flag(JitFlag::VCallOptimize, i);

        F1 f1; F2 f2;
        uint32_t i1 = jit_registry_put(Backend, "Base", &f1);
        uint32_t i2 = jit_registry_put(Backend, "Base", &f2);
        jit_assert(i1 == 1 && i2 == 2);

        vcall("Base", [](Base *self2) { self2->go(); }, self);
        jit_assert(strcmp(f1.buffer.str(), "[0, 4, 0, 8, 0]") == 0);
        jit_assert(strcmp(f2.buffer.str(), "[0, 1, 5, 3]") == 0);

        jit_registry_remove(&f1);
        jit_registry_remove(&f2);
    }
}

TEST_BOTH_FP32(07_side_effects_only_once) {
    /* This tests ensures that side effects baked into a function only happen
       once, even when that function is evaluated multiple times. */

    struct Base {
        virtual dr_tuple<Float, Float> f() = 0;
    };

    struct G1 : Base {
        Float buffer = zeros<Float>(5);
        dr_tuple<Float, Float> f() override {
            scatter_reduce(ReduceOp::Add, buffer, Float(1), UInt32(1));
            return { Float(1), Float(2) };
        }
    };

    struct G2 : Base {
        Float buffer = zeros<Float>(5);
        dr_tuple<Float, Float> f() override {
            scatter_reduce(ReduceOp::Add, buffer, Float(1), UInt32(2));
            return { Float(2), Float(1) };
        }
    };

    using BasePtr = Array<Base *>;
    BasePtr self = arange<UInt32>(11) % 3;

    for (uint32_t i = 0; i < 2; ++i) {
        jit_set_flag(JitFlag::VCallOptimize, i);

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

        jit_registry_remove(&g1);
        jit_registry_remove(&g2);
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
        jit_set_flag(JitFlag::VCallOptimize, i);

        Float y = vcall("Base", [](Base *self2, Float x2) { return self2->f(x2); }, self, x);
        Float z = vcall("Base", [](Base *self2, Float x2) { return self2->f(x2); }, self, y);

        jit_assert(strcmp(z.str(), "[0, 12, 14, 0, 12, 14, 0, 12, 14, 0]") == 0);
    }

    jit_registry_remove(&h1);
    jit_registry_remove(&h2);
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

    (void) i1;
    (void) i2;

    for (int i = 0; i < n1; ++i) {
        v1[i].v = (Float) (float) i;
        i1[i] = jit_registry_put(Backend, "Base1", &v1[i]);
    }

    for (int i = 0; i < n2; ++i) {
        v2[i].v = (Float) (100.f + (float) i);
        i2[i] = jit_registry_put(Backend, "Base2", &v2[i]);
    }

    using Base1Ptr = Array<Base1 *>;
    using Base2Ptr = Array<Base2 *>;
    UInt32 self1 = arange<UInt32>(n + 1);
    UInt32 self2 = arange<UInt32>(n + 1);

    self1 = select(self1 <= n1, self1, 0);
    self2 = select(self2 <= n2, self2, 0);

    for (uint32_t i = 0; i < 2; ++i) {
        jit_set_flag(JitFlag::VCallOptimize, i);

        Float x = vcall("Base1", [](Base1 *self_) { return self_->f(); }, Base1Ptr(self1));
        Float y = vcall("Base2", [](Base2 *self_) { return self_->f(); }, Base2Ptr(self2));

        jit_var_schedule(x.index());
        jit_var_schedule(y.index());

        jit_assert((float)x.read(0) == 0);
        jit_assert((float)y.read(0) == 0);

        for (uint32_t j = 1; j <= n1; ++j)
            jit_assert((float)x.read(j) == j - 1);
        for (uint32_t j = 1; j <= n2; ++j)
            jit_assert((float)y.read(j) == 100 + j - 1);

        for (uint32_t j = n1 + 1; j < n; ++j)
            jit_assert((float)x.read(j + 1) == 0);
        for (uint32_t j = n2 + 1; j < n; ++j)
            jit_assert((float)y.read(j + 1) == 0);
    }

    for (int i = 0; i < n1; ++i)
        jit_registry_remove(&v1[i]);
    for (int i = 0; i < n2; ++i)
        jit_registry_remove(&v2[i]);
}

TEST_BOTH(09_self) {
    struct Base;
    using BasePtr = Array<Base *>;

    struct Base { virtual Array<Base *> f() = 0; };
    struct I : Base { BasePtr f() {
        BasePtr result = this;
        // jit_assert(strstr(jit_var_get_stmt(result.index()), "self")); /// XXX
        return result;
    } };

    I i1, i2;
    uint32_t i1_id = jit_registry_put(Backend, "Base", &i1);
    uint32_t i2_id = jit_registry_put(Backend, "Base", &i2);

    UInt32 self(i1_id, i2_id);
    UInt32 y = vcall(
        "Base",
        [](Base *self_) { return self_->f(); },
        BasePtr(self));

    jit_assert(strcmp(y.str(), "[1, 2]") == 0);

    jit_registry_remove(&i1);
    jit_registry_remove(&i2);
}

TEST_BOTH(10_recursion) {
    struct Base1 { virtual Float f(const Float &x) = 0; };
    using Base1Ptr = Array<Base1 *>;

    struct Base2 { virtual Float g(const Base1Ptr &ptr, const Float &x) = 0; };
    using Base2Ptr = Array<Base2 *>;

    struct I1 : Base1 {
        Float c;
        Float f(const Float &x) override { return x * c; }
    };

    struct I2 : Base2 {
        Float g(const Base1Ptr &ptr, const Float &x) override {
            return vcall("Base1", [&](Base1 *self_, Float x_) { return self_->f(x_); }, ptr, x) + Float(1);
        }
    };

    I1 i11, i12;
    i11.c = Float(2);
    i12.c = Float(3);
    I2 i21, i22;
    uint32_t i11_id = jit_registry_put(Backend, "Base1", &i11);
    uint32_t i12_id = jit_registry_put(Backend, "Base1", &i12);
    uint32_t i21_id = jit_registry_put(Backend, "Base2", &i21);
    uint32_t i22_id = jit_registry_put(Backend, "Base2", &i22);

    UInt32 self1(i11_id, i12_id);
    UInt32 self2(i21_id, i22_id);
    Float x(3.f, 5.f);

    Float y = vcall(
        "Base2",
        [](Base2 *self_, const Base1Ptr &ptr_, const Float &x_) {
            return self_->g(ptr_, x_);
        },
        Base2Ptr(self2), Base1Ptr(self1), x);

    jit_assert(strcmp(y.str(), "[7, 16]") == 0);

    jit_registry_remove(&i11);
    jit_registry_remove(&i12);
    jit_registry_remove(&i21);
    jit_registry_remove(&i22);
}

TEST_BOTH(11_recursion_with_local) {
    struct Base1 { virtual Float f(const Float &x) = 0; };
    using Base1Ptr = Array<Base1 *>;

    struct Base2 { virtual Float g(const Base1Ptr &ptr, const Float &x) = 0; };
    using Base2Ptr = Array<Base2 *>;

    struct I1 : Base1 {
        Float c;
        Float f(const Float &x) override { return x * c; }
    };

    struct I2 : Base2 {
        Float g(const Base1Ptr &ptr, const Float &x) override {
            return vcall("Base1", [&](Base1 *self_, Float x_) { return self_->f(x_); }, ptr, x) + Float(1);
        }
    };

    I1 i11, i12;
    i11.c = dr::opaque<Float>(2);
    i12.c = dr::opaque<Float>(3);
    I2 i21, i22;
    uint32_t i11_id = jit_registry_put(Backend, "Base1", &i11);
    uint32_t i12_id = jit_registry_put(Backend, "Base1", &i12);
    uint32_t i21_id = jit_registry_put(Backend, "Base2", &i21);
    uint32_t i22_id = jit_registry_put(Backend, "Base2", &i22);

    UInt32 self1(i11_id, i12_id);
    UInt32 self2(i21_id, i22_id);
    Float x(3.f, 5.f);

    Float y = vcall(
        "Base2",
        [](Base2 *self_, const Base1Ptr &ptr_, const Float &x_) {
            return self_->g(ptr_, x_);
        },
        Base2Ptr(self2), Base1Ptr(self1), x);

    jit_assert(strcmp(y.str(), "[7, 16]") == 0);

    jit_registry_remove(&i11);
    jit_registry_remove(&i12);
    jit_registry_remove(&i21);
    jit_registry_remove(&i22);
}

TEST_BOTH_FP32(12_nested_with_side_effects) {
    struct Base {
        virtual void f() = 0;
        virtual void g() = 0;
    };
    using BasePtr = Array<Base *>;

    struct F1 : Base {
        Float buffer = zeros<Float>(5);
        void f() override {
            BasePtr self = full<UInt32>(1, 11);
            vcall("Base", [](Base *self2) { self2->g(); }, self);
        }

        void g() override {
            scatter_reduce(ReduceOp::Add, buffer, Float(1), UInt32(1));
            scatter_reduce(ReduceOp::Add, buffer, Float(2), UInt32(3));
        }
    };

    struct F2 : Base {
        Float buffer = arange<Float>(4);
        void f() override {
            scatter_reduce(ReduceOp::Add, buffer, Float(1), UInt32(2));
        }
        void g() override {
        }
    };

    BasePtr self = arange<UInt32>(11) % 3;

    for (uint32_t i = 0; i < 2; ++i) {
        jit_set_flag(JitFlag::VCallOptimize, i);

        F1 f1; F2 f2;
        uint32_t i1 = jit_registry_put(Backend, "Base", &f1);
        uint32_t i2 = jit_registry_put(Backend, "Base", &f2);
        jit_assert(i1 == 1 && i2 == 2);

        vcall("Base", [](Base *self2) { self2->f(); }, self);
        jit_assert(strcmp(f1.buffer.str(), "[0, 4, 0, 8, 0]") == 0);
        jit_assert(strcmp(f2.buffer.str(), "[0, 1, 5, 3]") == 0);

        jit_registry_remove(&f1);
        jit_registry_remove(&f2);
    }
}

TEST_BOTH(13_load_bool_data) {
    struct Base {
        virtual Float f() = 0;
    };
    using BasePtr = Array<Base *>;

    struct F1 : Base {
        Mask cond = dr::opaque<Mask>(true);
        Float val1 = dr::opaque<Float>(1);
        Float val2 = dr::opaque<Float>(2);
        Float f() override {
            return select(cond, val1, val2);
        }
    };

    struct F2 : Base {
        Mask cond = dr::opaque<Mask>(false);
        Float val1 = dr::opaque<Float>(3);
        Float val2 = dr::opaque<Float>(4);
        Float f() override {
            return select(cond, val1, val2);
        }
    };

    BasePtr self = arange<UInt32>(5) % 3;
    for (uint32_t i = 0; i < 2; ++i) {
        jit_set_flag(JitFlag::VCallOptimize, i);

        F1 f1; F2 f2;
        uint32_t i1 = jit_registry_put(Backend, "Base", &f1);
        uint32_t i2 = jit_registry_put(Backend, "Base", &f2);
        jit_assert(i1 == 1 && i2 == 2);

        Float result = vcall(
            "Base", [](Base *self2) { return self2->f(); }, self);

        jit_var_schedule(result.index());
        jit_assert(strcmp(result.str(), "[0, 1, 4, 0, 1]") == 0);

        jit_registry_remove(&f1);
        jit_registry_remove(&f2);
    }
}
