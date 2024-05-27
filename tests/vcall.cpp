#include "drjit-core/jit.h"
#include "test.h"
#include "traits.h"

namespace dr = drjit;

namespace drjit {

    /// RAII helper to temporarily record symbolic computation
    struct scoped_record {
        scoped_record(JitBackend backend, const char *name = nullptr,
                      bool new_scope = false)
            : backend(backend) {
            checkpoint = jit_record_begin(backend, name);
            if (new_scope)
                scope = jit_new_scope(backend);
        }

        ~scoped_record() {
            jit_record_end(backend, checkpoint, cleanup);
        }

        uint32_t checkpoint_and_rewind() {
            jit_set_scope(backend, scope);
            return jit_record_checkpoint(backend);
        }

        void disarm() { cleanup = false; }

        JitBackend backend;
        uint32_t checkpoint, scope;
        bool cleanup = true;
    };

    /// RAII helper to temporarily push a mask onto the Dr.Jit mask stack
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

    /// RAII helper to temporarily set the 'self' instance
    struct scoped_set_self {
        scoped_set_self(JitBackend backend, uint32_t value, uint32_t self_index = 0)
            : m_backend(backend) {
            jit_var_self(backend, &m_self_value, &m_self_index);
            jit_var_inc_ref(m_self_index);
            jit_var_set_self(m_backend, value, self_index);
        }

        ~scoped_set_self() {
            jit_var_set_self(m_backend, m_self_value, m_self_index);
            jit_var_dec_ref(m_self_index);
        }

    private:
        JitBackend m_backend;
        uint32_t m_self_value;
        uint32_t m_self_index;
    };

} // namespace drjit

// Basic wrapper for recording a symbolic call used by tests exclusively
template <size_t n_callables, size_t n_inputs, size_t n_outputs>
void symbolic_call(
        JitBackend backend,
        const char* domain,
        bool symbolic,
        uint32_t self,
        uint32_t mask,
        void (*call) (void*, uint32_t*, uint32_t*),
        uint32_t* inputs,
        uint32_t* outputs) {

    uint32_t checkpoints[n_callables + 1] = { 0 };
    uint32_t inst_id[n_callables] = { 0 };

    uint32_t call_inputs[n_inputs == 0 ? 1 : n_inputs] = { 0 };
    uint32_t rv_values[n_outputs == 0 ? 1 : n_callables * n_outputs] = { 0 };

    jit_new_scope(backend);
    {
        dr::scoped_record rec(backend, domain, true);

        for (size_t i = 0; i < n_inputs; ++i) {
            call_inputs[i] = jit_var_call_input(inputs[i]);
        }

        {
            dr::scoped_set_mask mask_guard(backend, jit_var_call_mask(backend));
            for (size_t i = 0; i < n_callables; ++i) {
                checkpoints[i] = rec.checkpoint_and_rewind();

                uint32_t call_index =  (uint32_t) i + 1;

                void *ptr = jit_registry_ptr(backend, domain,
                    call_index);

                scoped_set_self set_self(backend, 
                    call_index);

                call(ptr, call_inputs, &rv_values[i * n_outputs]);
                inst_id[i] = call_index;
            }

            checkpoints[n_callables] = rec.checkpoint_and_rewind();
        }

        jit_new_scope(backend);

        jit_var_call(
            domain, symbolic,
            self, mask,
            (uint32_t) n_callables,
            inst_id,
            (uint32_t) n_inputs,
            call_inputs,
            (uint32_t) n_callables * n_outputs,
            rv_values,
            checkpoints,
            outputs
        );

        for (size_t i = 0; i < n_inputs; ++i) {
            jit_var_dec_ref(call_inputs[i]);
        }

        for (size_t i = 0; i < n_callables * n_outputs; ++i) {
            jit_var_dec_ref(rv_values[i]);
        }

        rec.disarm();
    }
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

    const char* domain = "Base";
    const size_t n_callables    = 2;
    const size_t n_inputs       = 1;
    const size_t n_outputs      = 1;

    uint32_t i1 = jit_registry_put(Backend, domain, &a1);
    uint32_t i2 = jit_registry_put(Backend, domain, &a2);
    jit_assert(i1 == 1 && i2 == 2);

    using BasePtr = Array<Base *>;
    Float x = arange<Float>(10);
    BasePtr self = arange<UInt32>(10) % 3;

    auto f_call = [](void *self2, uint32_t* inputs, uint32_t* outputs) {
        Base* base = (Base*)self2;
        Float x2 = Float::borrow(inputs[0]);
        Float rv = base->f(x2);
        jit_var_inc_ref(rv.index());

        outputs[0] = rv.index();
    };

    for (size_t i = 0; i < 2; ++i) {
        jit_set_flag(JitFlag::OptimizeCalls, i);

        uint32_t inputs[n_inputs] = { x.index() };
        uint32_t outputs[n_outputs] = { 0 };

        Mask mask = Mask::steal(jit_var_bool(Backend, true));

        symbolic_call<n_callables, n_inputs, n_outputs>(
            Backend, domain,
            false, /* symbolic */
            self.index(), mask.index(),
            f_call,
            inputs,
            outputs);

        Float y = Float::steal(outputs[0]);

        jit_assert(strcmp(y.str(), 
            "[0, 22, 204, 0, 28, 210, 0, 34, 216, 0]") == 0);
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
        virtual tuple<Mask, Float, Double, Float, Mask>
        f(Mask p0, Float p1, Double p2, Float p3, Mask p4) = 0;
    };

    struct B1 : Base {
        tuple<Mask, Float, Double, Float, Mask>
        f(Mask p0, Float p1, Double p2, Float p3, Mask p4) override {
            return { p0, p1, p2, p3, p4 };
        }
    };

    struct B2 : Base {
        tuple<Mask, Float, Double, Float, Mask>
        f(Mask p0, Float p1, Double p2, Float p3, Mask) override {
            return { !p0, p1 + 1, p2 + 2, p3 + 3, false };
        }
    };

    struct B3 : Base {
        tuple<Mask, Float, Double, Float, Mask>
        f(Mask, Float, Double, Float, Mask) override {
            return { 0, 0, 0, 0, 0 };
        }
    };

    B1 b1; B2 b2; B3 b3;

    const char* domain = "Base";
    const size_t n_callables    = 3;
    const size_t n_inputs       = 5;
    const size_t n_outputs      = 5;

    uint32_t i1 = jit_registry_put(Backend, "Base", &b1);
    uint32_t i2 = jit_registry_put(Backend, "Base", &b2);
    uint32_t i3 = jit_registry_put(Backend, "Base", &b3);
    (void) i1; (void) i2; (void) i3;

    auto f_call = [](void *self2, uint32_t* inputs, uint32_t* outputs) {
        Base* base  = (Base*)self2;
        Mask p0     = Mask::borrow(inputs[0]);
        Float p1    = Float::borrow(inputs[1]);
        Double p2   = Double::borrow(inputs[2]);
        Float p3    = Float::borrow(inputs[3]);
        Mask p4     = Mask::borrow(inputs[4]);

        auto [rv0, rv1, rv2, rv3, rv4] = base->f(p0, p1, p2, p3, p4);

        outputs[0] = rv0.index();
        outputs[1] = rv1.index();
        outputs[2] = rv2.index();
        outputs[3] = rv3.index();
        outputs[4] = rv4.index();

        for (size_t i = 0; i < 5; ++i) {
            jit_var_inc_ref(outputs[i]);
        }
    };

    for (uint32_t i = 0; i < 2; ++i) {
        jit_set_flag(JitFlag::OptimizeCalls, i);

        using BasePtr = Array<Base *>;
        BasePtr self = arange<UInt32>(10) % 3;

        Mask p0(false);
        Float p1(12);
        Double p2(34);
        Float p3(56);
        Mask p4(true);

        Mask mask = Mask::steal(jit_var_bool(Backend, true));

        uint32_t inputs[n_inputs] = { 
            p0.index(), p1.index(), p2.index(), p3.index(), p4.index() };
        uint32_t outputs[n_outputs] = { 0 };

        symbolic_call<n_callables, n_inputs, n_outputs>(
            Backend, domain,
            false, /* symbolic */
            self.index(), mask.index(),
            f_call,
            inputs,
            outputs);

        jit_var_schedule(outputs[0]);
        jit_var_schedule(outputs[1]);
        jit_var_schedule(outputs[2]);
        jit_var_schedule(outputs[3]);
        jit_var_schedule(outputs[4]);

        Mask rv0    = Mask::steal(outputs[0]);
        Float rv1   = Float::steal(outputs[1]);
        Double rv2  = Double::steal(outputs[2]);
        Float rv3   = Float::steal(outputs[3]);
        Mask rv4    = Mask::steal(outputs[4]);

        jit_assert(strcmp(rv0.str(), "[0, 0, 1, 0, 0, 1, 0, 0, 1, 0]") == 0);
        jit_assert(strcmp(rv1.str(), "[0, 12, 13, 0, 12, 13, 0, 12, 13, 0]") == 0);
        jit_assert(strcmp(rv2.str(), "[0, 34, 36, 0, 34, 36, 0, 34, 36, 0]") == 0);
        jit_assert(strcmp(rv3.str(), "[0, 56, 59, 0, 56, 59, 0, 56, 59, 0]") == 0);
        jit_assert(strcmp(rv4.str(), "[0, 1, 0, 0, 1, 0, 0, 1, 0, 0]") == 0);
    }

    jit_registry_remove(&b1);
    jit_registry_remove(&b2);
    jit_registry_remove(&b3);
}


TEST_BOTH(03_devirtualize) {
    /* This test checks that outputs which produce identical values across
       all instances are moved out of the virtual call interface. */
    struct Base {
        virtual tuple<Float, Float, Float> f(Float p1, Float p2) = 0;
    };

    struct D1 : Base {
        tuple<Float, Float, Float> f(Float p1, Float p2) override {
            return { p2 + 2, p1 + 1, 0 };
        }
    };

    struct D2 : Base {
        tuple<Float, Float, Float> f(Float p1, Float p2) override {
            return { p2 + 2, p1 + 2, 0 };
        }
    };

    const char* domain = "Base";
    const size_t n_callables    = 2;
    const size_t n_inputs       = 2;
    const size_t n_outputs      = 3;

    D1 d1; D2 d2;
    uint32_t i1 = jit_registry_put(Backend, domain, &d1);
    uint32_t i2 = jit_registry_put(Backend, domain, &d2);
    jit_assert(i1 == 1 && i2 == 2);

    using BasePtr = Array<Base *>;
    BasePtr self = arange<UInt32>(10) % 3;

    auto f_call = [](void *self2, uint32_t* inputs, uint32_t* outputs) {
        Base* base  = (Base*)self2;
        Float p1    = Float::borrow(inputs[0]);
        Float p2    = Float::borrow(inputs[1]);

        auto [rv0, rv1, rv2] = base->f(p1, p2);

        outputs[0] = rv0.index();
        outputs[1] = rv1.index();
        outputs[2] = rv2.index();

        for (size_t i = 0; i < 3; ++i) {
            jit_var_inc_ref(outputs[i]);
        }
    };

    for (uint32_t k = 0; k < 2; ++k) {
        for (uint32_t i = 0; i < 2; ++i) {
            Float p1, p2;
            if (k == 0) {
                p1 = 12;
                p2 = 34;
            } else {
                p1 = dr::opaque<Float>(12);
                p2 = dr::opaque<Float>(34);
            }

            jit_set_flag(JitFlag::OptimizeCalls, i);
            uint32_t scope = jit_scope(Backend);

            scoped_set_log_level x((LogLevel) 10);

            Mask call_mask = Mask::steal(jit_var_bool(Backend, true));

            uint32_t inputs[n_inputs] = { p1.index(), p2.index() };
            uint32_t outputs[n_outputs] = { 0 };

            symbolic_call<n_callables, n_inputs, n_outputs>(
                Backend, domain,
                false, /* symbolic */
                self.index(), call_mask.index(),
                f_call,
                inputs,
                outputs);

            jit_set_scope(Backend, scope + 1);

            Float p1_wrap = Float::steal(jit_var_call_input(p1.index()));
            Float p2_wrap = Float::steal(jit_var_call_input(p2.index()));

            Mask mask = neq(self, nullptr),
                 mask_combined = Mask::steal(jit_var_mask_apply(mask.index(), 10));

            Float alt0 = (p2_wrap + 2) & mask_combined;
            Float alt1 = (p1_wrap + 2) & mask_combined;
            Float alt2 = Float(0) & mask_combined;

            jit_set_scope(Backend, scope + 2);

            Float rv0 = Float::steal(outputs[0]);
            Float rv1 = Float::steal(outputs[1]);
            Float rv2 = Float::steal(outputs[2]);

            auto is_call_output = [](uint32_t index) {
                return strcmp(jit_var_kind_name(index), "call_output") == 0;
            };

            bool literal_input   = k == 0;
            bool optimize_output = i == 1;

            jit_assert(is_call_output(rv0.index()) != (optimize_output && literal_input));
            jit_assert(is_call_output(rv1.index()) == true);
            jit_assert(is_call_output(rv2.index()) != optimize_output);

            jit_var_schedule(rv0.index());
            jit_var_schedule(rv1.index());

            jit_assert(
                strcmp(rv0.str(),
                            "[0, 36, 36, 0, 36, 36, 0, 36, 36, 0]") == 0);
            jit_assert(
                strcmp(rv1.str(),
                            "[0, 13, 14, 0, 13, 14, 0, 13, 14, 0]") == 0);
        }
    }
    jit_registry_remove(&d1);
    jit_registry_remove(&d2);
}


TEST_BOTH(04_extra_data) {
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

    const char* domain = "Base";
    const size_t n_callables    = 2;
    const size_t n_inputs       = 1;
    const size_t n_outputs      = 1;

    E1 e1; E2 e2;
    uint32_t i1 = jit_registry_put(Backend, domain, &e1);
    uint32_t i2 = jit_registry_put(Backend, domain, &e2);
    jit_assert(i1 == 1 && i2 == 2);

    auto f_call = [](void *self2, uint32_t* inputs, uint32_t* outputs) {
        Base* base  = (Base*)self2;
        Float x    = Float::borrow(inputs[0]);

        auto rv0 = base->f(x);

        outputs[0] = rv0.index();
        jit_var_inc_ref(outputs[0]);
    };

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
            jit_set_flag(JitFlag::OptimizeCalls, i);

            Mask mask = Mask::steal(jit_var_bool(Backend, true));

            uint32_t inputs[n_inputs] = { x.index() };
            uint32_t outputs[n_outputs] = { 0 };

            symbolic_call<n_callables, n_inputs, n_outputs>(
                Backend, domain,
                false, /* symbolic */
                self.index(), mask.index(),
                f_call,
                inputs,
                outputs);

            Float result = Float::steal(outputs[0]);
            jit_assert(strcmp(result.str(), 
                "[0, 9, 13, 0, 21, 28, 0, 33, 43, 0]") == 0);
        }
    }
    jit_registry_remove(&e1);
    jit_registry_remove(&e2);
}


TEST_BOTH_FP32(05_side_effects) {
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

    const char* domain = "Base";
    const size_t n_callables    = 2;
    const size_t n_inputs       = 0;
    const size_t n_outputs      = 0;

    auto f_call = [](void *self, uint32_t* /*inputs*/, uint32_t* /*outputs*/) {
        ((Base*)self)->go();
    };

    for (uint32_t i = 0; i < 2; ++i) {
        jit_set_flag(JitFlag::OptimizeCalls, i);

        F1 f1; F2 f2;
        uint32_t i1 = jit_registry_put(Backend, domain, &f1);
        uint32_t i2 = jit_registry_put(Backend, domain, &f2);
        jit_assert(i1 == 1 && i2 == 2);

        Mask mask = Mask::steal(jit_var_bool(Backend, true));

        symbolic_call<n_callables, n_inputs, n_outputs>(
            Backend, domain,
            false, /* symbolic */
            self.index(), mask.index(),
            f_call,
            nullptr,
            nullptr);

        jit_assert(strcmp(f1.buffer.str(), "[0, 4, 0, 8, 0]") == 0);
        jit_assert(strcmp(f2.buffer.str(), "[0, 1, 5, 3]") == 0);

        jit_registry_remove(&f1);
        jit_registry_remove(&f2);
    }
}


TEST_BOTH_FP32(06_side_effects_only_once) {
    /* This tests ensures that side effects baked into a function only happen
       once, even when that function is evaluated multiple times. */

    struct Base {
        virtual tuple<Float, Float> f() = 0;
    };

    struct G1 : Base {
        Float buffer = zeros<Float>(5);
        tuple<Float, Float> f() override {
            scatter_reduce(ReduceOp::Add, buffer, Float(1), UInt32(1));
            return { 1, 2 };
        }
    };

    struct G2 : Base {
        Float buffer = zeros<Float>(5);
        tuple<Float, Float> f() override {
            scatter_reduce(ReduceOp::Add, buffer, Float(1), UInt32(2));
            return { 2, 1 };
        }
    };

    using BasePtr = Array<Base *>;
    BasePtr self = arange<UInt32>(11) % 3;

    const char* domain = "Base";
    const size_t n_callables    = 2;
    const size_t n_inputs       = 0;
    const size_t n_outputs      = 2;

    auto f_call = [](void *self2, uint32_t* /*inputs*/, uint32_t* outputs) {
        Base* base  = (Base*)self2;

        auto [rv0, rv1] = base->f();

        outputs[0] = rv0.index();
        outputs[1] = rv1.index();
        for (size_t i = 0; i < 2; ++i) {
            jit_var_inc_ref(outputs[i]);
        }
    };

    for (uint32_t i = 0; i < 2; ++i) {
        jit_set_flag(JitFlag::OptimizeCalls, i);

        G1 g1; G2 g2;
        uint32_t i1 = jit_registry_put(Backend, domain, &g1);
        uint32_t i2 = jit_registry_put(Backend, domain, &g2);
        jit_assert(i1 == 1 && i2 == 2);

        uint32_t outputs[n_outputs] = { 0 };

        Mask mask = Mask::steal(jit_var_bool(Backend, true));

        symbolic_call<n_callables, n_inputs, n_outputs>(
            Backend, domain,
            false, /* symbolic */
            self.index(), mask.index(),
            f_call,
            nullptr,
            outputs);

        Float f1 = Float::steal(outputs[0]);
        Float f2 = Float::steal(outputs[1]);

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


TEST_BOTH(07_multiple_calls) {
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

    const char* domain = "Base";
    const size_t n_callables    = 2;
    const size_t n_inputs       = 1;
    const size_t n_outputs      = 1;

    using BasePtr = Array<Base *>;
    BasePtr self = arange<UInt32>(10) % 3;
    Float x = opaque<Float>(10, 1);

    H1 h1; H2 h2;
    uint32_t i1 = jit_registry_put(Backend, domain, &h1);
    uint32_t i2 = jit_registry_put(Backend, domain, &h2);
    jit_assert(i1 == 1 && i2 == 2);

    auto f_call = [](void *self2, uint32_t* inputs, uint32_t* outputs) {
        Base* base  = (Base*)self2;
        Float x     = Float::borrow(inputs[0]);

        Float rv0 = base->f(x);

        outputs[0] = rv0.index();
        jit_var_inc_ref(outputs[0]);
    };

    for (uint32_t i = 0; i < 2; ++i) {
        jit_set_flag(JitFlag::OptimizeCalls, i);

        Mask mask = Mask::steal(jit_var_bool(Backend, true));

        uint32_t inputs[n_inputs] = { x.index() };
        uint32_t outputs[n_outputs] = { 0 };

        symbolic_call<n_callables, n_inputs, n_outputs>(
            Backend, domain,
            false, /* symbolic */
            self.index(), mask.index(),
            f_call,
            inputs,
            outputs);

        Float y = Float::steal(outputs[0]);
        inputs[0] = y.index();

        symbolic_call<n_callables, n_inputs, n_outputs>(
            Backend, domain,
            false, /* symbolic */
            self.index(), mask.index(),
            f_call,
            inputs,
            outputs);

        Float z = Float::steal(outputs[0]);

        jit_assert(strcmp(z.str(), "[0, 12, 14, 0, 12, 14, 0, 12, 14, 0]") == 0);
    }

    jit_registry_remove(&h1);
    jit_registry_remove(&h2);
}


TEST_BOTH(08_big) {
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

    const char* domain1 = "Base1";
    const char* domain2 = "Base2";
    const size_t n_inputs        = 0;
    const size_t n_outputs       = 1;

    for (int i = 0; i < n1; ++i) {
        v1[i].v = (Float) (float) i;
        i1[i] = jit_registry_put(Backend, domain1, &v1[i]);
    }

    for (int i = 0; i < n2; ++i) {
        v2[i].v = (Float) (100.f + (float) i);
        i2[i] = jit_registry_put(Backend, domain2, &v2[i]);
    }

    using Base1Ptr = Array<Base1 *>;
    using Base2Ptr = Array<Base2 *>;
    UInt32 self1 = arange<UInt32>(n + 1);
    UInt32 self2 = arange<UInt32>(n + 1);

    self1 = select(self1 <= n1, self1, 0);
    self2 = select(self2 <= n2, self2, 0);

    auto f_call1 = [](void *self2, uint32_t* /*inputs*/, uint32_t* outputs) {
        Base1* base  = (Base1*)self2;

        Float rv0 = base->f();

        outputs[0] = rv0.index();
        jit_var_inc_ref(outputs[0]);
    };

    auto f_call2 = [](void *self2, uint32_t* /*inputs*/, uint32_t* outputs) {
        Base2* base  = (Base2*)self2;

        Float rv0 = base->f();

        outputs[0] = rv0.index();
        jit_var_inc_ref(outputs[0]);
    };

    for (uint32_t i = 0; i < 2; ++i) {
        jit_set_flag(JitFlag::OptimizeCalls, i);

        Mask mask1 = Mask::steal(jit_var_bool(Backend, true));
        Mask mask2 = Mask::steal(jit_var_bool(Backend, true));

        uint32_t outputs1[n_outputs] = { 0 };
        uint32_t outputs2[n_outputs] = { 0 };

        symbolic_call<n1, n_inputs, n_outputs>(
            Backend, domain1,
            false, /* symbolic */
            self1.index(), mask1.index(),
            f_call1,
            nullptr,
            outputs1);

        symbolic_call<n2, n_inputs, n_outputs>(
            Backend, domain2,
            false, /* symbolic */
            self2.index(), mask2.index(),
            f_call2,
            nullptr,
            outputs2);

        Float x = Float::steal(outputs1[0]);
        Float y = Float::steal(outputs2[0]);

        jit_var_schedule(x.index());
        jit_var_schedule(y.index());

        jit_assert((uint32_t)x.read(0) == 0);
        jit_assert((uint32_t)y.read(0) == 0);

        for (uint32_t j = 1; j <= n1; ++j)
            jit_assert((uint32_t)x.read(j) == j - 1);
        for (uint32_t j = 1; j <= n2; ++j)
            jit_assert((uint32_t)y.read(j) == 100 + j - 1);

        for (uint32_t j = n1 + 1; j < n; ++j)
            jit_assert((uint32_t)x.read(j + 1) == 0);
        for (uint32_t j = n2 + 1; j < n; ++j)
            jit_assert((uint32_t)y.read(j + 1) == 0);
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
        return result;
    } };

    const char* domain = "Base";
    const size_t n_callables    = 2;
    const size_t n_inputs       = 0;
    const size_t n_outputs      = 1;

    I i1, i2;
    uint32_t i1_id = jit_registry_put(Backend, domain, &i1);
    uint32_t i2_id = jit_registry_put(Backend, domain, &i2);

    UInt32 self(i1_id, i2_id);

    auto f_call = [](void *self2, uint32_t* /*inputs*/, uint32_t* outputs) {
        Base* base  = (Base*)self2;

        UInt32 rv0 = base->f();
        outputs[0] = rv0.index();
        jit_var_inc_ref(outputs[0]);
    };

    uint32_t outputs[n_outputs] = { 0 };

    Mask mask = Mask::steal(jit_var_mask_default(Backend, self.size()));

    symbolic_call<n_callables, n_inputs, n_outputs>(
        Backend, domain,
        false, /* symbolic */
        self.index(), mask.index(),
        f_call,
        nullptr,
        outputs);

    UInt32 y = UInt32::steal(outputs[0]);

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

            const size_t n_callables    = 2;
            const size_t n_inputs       = 1;
            const size_t n_outputs      = 1;

            auto f_call = [](void *self1, uint32_t* inputs, uint32_t* outputs) {
                Base1* base = (Base1*)self1;
                Float x     = Float::borrow(inputs[0]);

                Float rv0 = base->f(x);
                outputs[0] = rv0.index();
                jit_var_inc_ref(outputs[0]);
            };

            uint32_t inputs[n_inputs] = { x.index() };
            uint32_t outputs[n_outputs] = { 0 };

            Mask mask = Mask::steal(jit_var_bool(Backend, true));

            symbolic_call<n_callables, n_inputs, n_outputs>(
                Backend, "Base1",
                true, /* symbolic */
                ptr.index(), mask.index(),
                f_call,
                inputs,
                outputs);

            return Float::steal(outputs[0]) + 1;
        }
    };

    I1 i11, i12;
    i11.c = 2;
    i12.c = 3;
    I2 i21, i22;
    uint32_t i11_id = jit_registry_put(Backend, "Base1", &i11);
    uint32_t i12_id = jit_registry_put(Backend, "Base1", &i12);
    uint32_t i21_id = jit_registry_put(Backend, "Base2", &i21);
    uint32_t i22_id = jit_registry_put(Backend, "Base2", &i22);

    const size_t n_callables    = 2;
    const size_t n_inputs       = 2;
    const size_t n_outputs      = 1;

    auto f_call = [](void *self2, uint32_t* inputs, uint32_t* outputs) {
        Base2* base     = (Base2*)self2;
        UInt32 self1    = UInt32::borrow(inputs[0]);
        Float x         = Float::borrow(inputs[1]);

        Float rv0 = base->g(self1, x);
        outputs[0] = rv0.index();
        jit_var_inc_ref(outputs[0]);
    };

    UInt32 self1(i11_id, i12_id);
    UInt32 self2(i21_id, i22_id);
    Float x(3.f, 5.f);

    uint32_t inputs[n_inputs] = { self1.index(), x.index() };
    uint32_t outputs[n_outputs] = { 0 };

    Mask mask = Mask::steal(jit_var_bool(Backend, true));

    symbolic_call<n_callables, n_inputs, n_outputs>(
        Backend, "Base2",
        false, /* symbolic */
        self2.index(), mask.index(),
        f_call,
        inputs,
        outputs);

    Float y = Float::steal(outputs[0]);

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

            const size_t n_callables    = 2;
            const size_t n_inputs       = 1;
            const size_t n_outputs      = 1;

            auto f_call = [](void *self1, uint32_t* inputs, uint32_t* outputs) {
                Base1* base = (Base1*)self1;
                Float x     = Float::borrow(inputs[0]);

                Float rv0 = base->f(x);
                outputs[0] = rv0.index();
                jit_var_inc_ref(outputs[0]);
            };

            uint32_t inputs[n_inputs] = { x.index() };
            uint32_t outputs[n_outputs] = { 0 };

            Mask mask = Mask::steal(jit_var_bool(Backend, true));

            symbolic_call<n_callables, n_inputs, n_outputs>(
                Backend, "Base1",
                true, /* symbolic */
                ptr.index(), mask.index(),
                f_call,
                inputs,
                outputs);

            return Float::steal(outputs[0]) + 1;
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

    const size_t n_callables    = 2;
    const size_t n_inputs       = 2;
    const size_t n_outputs      = 1;

    auto f_call = [](void *self2, uint32_t* inputs, uint32_t* outputs) {
        Base2* base     = (Base2*)self2;
        UInt32 self1    = UInt32::borrow(inputs[0]);
        Float x         = Float::borrow(inputs[1]);

        Float rv0 = base->g(self1, x);
        outputs[0] = rv0.index();
        jit_var_inc_ref(outputs[0]);
    };

    UInt32 self1(i11_id, i12_id);
    UInt32 self2(i21_id, i22_id);
    Float x(3.f, 5.f);

    uint32_t inputs[n_inputs] = { self1.index(), x.index() };
    uint32_t outputs[n_outputs] = { 0 };

    Mask mask = Mask::steal(jit_var_bool(Backend, true));

    symbolic_call<n_callables, n_inputs, n_outputs>(
        Backend, "Base2",
        false, /* symbolic */
        self2.index(), mask.index(),
        f_call,
        inputs,
        outputs);

    Float y = Float::steal(outputs[0]);

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
        BasePtr self = full<UInt32>(1, 11);

        void f() override {

            const size_t n_callables    = 2;
            const size_t n_inputs       = 0;
            const size_t n_outputs      = 0;

            Mask mask = Mask::steal(jit_var_bool(Backend, true));

            auto f_call = [](void *self, uint32_t* /*inputs*/, uint32_t* /*outputs*/) {
                Base* base = (Base*)self;
                base->g();
            };

            symbolic_call<n_callables, n_inputs, n_outputs>(
                Backend, "Base",
                true, /* symbolic */
                self.index(), mask.index(),
                f_call,
                nullptr,
                nullptr);
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

    auto f_call = [](void *self, uint32_t* /*inputs*/, uint32_t* /*outputs*/) {
        Base* base = (Base*)self;
        base->f();
    };

    const size_t n_callables    = 2;
    const size_t n_inputs       = 0;
    const size_t n_outputs      = 0;

    for (uint32_t i = 0; i < 2; ++i) {
        jit_set_flag(JitFlag::OptimizeCalls, i);

        F1 f1; F2 f2;
        uint32_t i1 = jit_registry_put(Backend, "Base", &f1);
        uint32_t i2 = jit_registry_put(Backend, "Base", &f2);
        jit_assert(i1 == 1 && i2 == 2);

        Mask mask = Mask::steal(jit_var_bool(Backend, true));

        symbolic_call<n_callables, n_inputs, n_outputs>(
            Backend, "Base",
            false, /* symbolic */
            self.index(), mask.index(),
            f_call,
            nullptr,
            nullptr);

        jit_var_schedule(f1.buffer.index());
        jit_var_schedule(f2.buffer.index());

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

    auto f_call = [](void *self2, uint32_t* /*inputs*/, uint32_t* outputs) {
        Base* base  = (Base*)self2;

        Float rv0 = base->f();
        outputs[0] = rv0.index();
        jit_var_inc_ref(outputs[0]);
    };

    const size_t n_callables    = 2;
    const size_t n_inputs       = 0;
    const size_t n_outputs      = 1;

    BasePtr self = arange<UInt32>(5) % 3;
    for (uint32_t i = 0; i < 2; ++i) {
        jit_set_flag(JitFlag::OptimizeCalls, i);

        F1 f1; F2 f2;
        uint32_t i1 = jit_registry_put(Backend, "Base", &f1);
        uint32_t i2 = jit_registry_put(Backend, "Base", &f2);
        jit_assert(i1 == 1 && i2 == 2);

        uint32_t outputs[n_outputs] = { 0 };

        Mask mask = Mask::steal(jit_var_bool(Backend, true));

        symbolic_call<n_callables, n_inputs, n_outputs>(
            Backend, "Base",
            false, /* symbolic */
            self.index(), mask.index(),
            f_call,
            nullptr,
            outputs);

        Float result = Float::steal(outputs[0]);

        jit_var_schedule(result.index());
        jit_assert(strcmp(result.str(), "[0, 1, 4, 0, 1]") == 0);

        jit_registry_remove(&f1);
        jit_registry_remove(&f2);
    }
}

TEST_BOTH(14_kernel_record) {
    jit_set_flag(JitFlag::VCallOptimize, true);
    jit_set_flag(JitFlag::SymbolicCalls, true);
    
    Recording *recording;
    
    struct Base {
        virtual UInt32 f(UInt32 x) = 0;
    };

    struct A1 : Base {
        UInt32 f(UInt32 x) override {
            return x+1;
        }
    };

    struct A2 : Base {
        UInt32 f(UInt32 x) override {
            return x+2;
        }
    };

    A1 a1;
    A2 a2;
    
    const char* domain = "Base";
    const size_t n_callables    = 2;
    const size_t n_inputs       = 1;
    const size_t n_outputs      = 1;
    
    uint32_t i1 = jit_registry_put(Backend, domain, &a1);
    uint32_t i2 = jit_registry_put(Backend, domain, &a2);
    jit_assert(i1 == 1 && i2 == 2);
    
    using BasePtr = Array<Base *>;

    auto f_call = [](void *self, uint32_t *inputs, uint32_t *outputs){
        Base *base = (Base*)self;
        UInt32 i0 = UInt32::borrow(inputs[0]);
        UInt32 o0 = base->f(i0);
        jit_var_inc_ref(o0.index());

        outputs[0] = o0.index();
    };

    {
        BasePtr self = arange<UInt32>(10) % 3;
        self.eval();
        UInt32 i0 = arange<UInt32>(10);
        i0.eval();
        UInt32 r0(0, 2, 4, 0, 5, 7, 0, 8, 10, 0);
        
        jit_log(LogLevel::Info, "Recording:");

        uint32_t inputs[] = {
            self.index(),
            i0.index(),
        };

        jit_record_start(Backend, inputs, 2);


        uint32_t outputs[1];
        UInt32 o0;
        
        {
            uint32_t vcall_inputs[n_inputs] = { i0.index() };
            uint32_t vcall_outputs[n_outputs] = { 0 };
            
            Mask mask = Mask::steal(jit_var_bool(Backend, true));

            jit_log(LogLevel::Info, "self: %u", self.index());
            jit_log(LogLevel::Info, "mask: %u", mask.index());
            jit_log(LogLevel::Info, "i0: %u", i0.index());
            symbolic_call<n_callables, n_inputs, n_outputs>(
                Backend, domain,
                false, 
                self.index(), mask.index(), 
                f_call,
                vcall_inputs, vcall_outputs);

            o0 = UInt32::borrow(vcall_outputs[0]);
            o0.eval();
            
            jit_log(LogLevel::Info, "o0: %u", o0.index());
            
            outputs[0] = o0.index();
        }

        recording = jit_record_stop(Backend, outputs, 1);

        jit_log(LogLevel::Info, "o0: %s", jit_var_str(outputs[0]));
        jit_log(LogLevel::Info, "r0: %s", jit_var_str(r0.index()));
        
        jit_assert(jit_var_all(jit_var_eq(r0.index(), outputs[0])));
    }

    jit_log(LogLevel::Info, "Replay:");
    {
        BasePtr self = (arange<UInt32>(10) + 1) % 3;
        self.eval();
        UInt32 i0 = arange<UInt32>(10);
        i0.eval();
        UInt32 r0(1, 3, 0, 4, 6, 0, 7, 9, 0, 10);

        uint32_t inputs[] = {
            self.index(),
            i0.index(),
        };
        uint32_t outputs[1];

        jit_record_replay(recording, inputs, outputs);

        jit_log(LogLevel::Info, "o0: %s", jit_var_str(outputs[0]));
        jit_assert(jit_var_all(jit_var_eq(r0.index(), outputs[0])));
    }
    
    jit_record_destroy(recording);
    
    jit_registry_remove(&a1);
    jit_registry_remove(&a2);
}
