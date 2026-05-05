// metal_vcall_test.cpp — Tests for virtual function calls on Metal.

#include <drjit-core/jit.h>
#include <drjit-core/array.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>

using namespace drjit;

using Float   = MetalArray<float>;
using UInt32  = MetalArray<uint32_t>;
using Mask    = MetalArray<bool>;

static constexpr JitBackend Backend = JitBackend::Metal;

namespace dr = drjit;

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

struct scoped_set_mask {
    scoped_set_mask(JitBackend backend, uint32_t index) : backend(backend) {
        jit_var_mask_push(backend, index);
        jit_var_dec_ref(index);
    }
    ~scoped_set_mask() { jit_var_mask_pop(backend); }
    JitBackend backend;
};

struct scoped_set_self {
    scoped_set_self(JitBackend backend, uint32_t value, uint32_t self_index = 0)
        : m_backend(backend) {
        uint32_t tmp;
        jit_var_self(backend, &m_self_value, &tmp);
        m_self_index = jit_var_inc_ref(tmp);
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

template <size_t n_callables, size_t n_inputs, size_t n_outputs>
void symbolic_call(
        JitBackend backend,
        const char* variant,
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
        scoped_record rec(backend, domain, true);

        for (size_t i = 0; i < n_inputs; ++i)
            call_inputs[i] = jit_var_call_input(inputs[i]);

        {
            scoped_set_mask mask_guard(backend, jit_var_call_mask(backend));
            for (size_t i = 0; i < n_callables; ++i) {
                checkpoints[i] = rec.checkpoint_and_rewind();
                uint32_t call_index = (uint32_t) i + 1;
                void *ptr = jit_registry_ptr(variant, domain, call_index);
                scoped_set_self set_self(backend, call_index);
                call(ptr, call_inputs, &rv_values[i * n_outputs]);
                inst_id[i] = call_index;
            }
            checkpoints[n_callables] = rec.checkpoint_and_rewind();
        }

        jit_new_scope(backend);
        jit_var_call(
            domain, symbolic,
            self, mask,
            (uint32_t) n_callables, (uint32_t) n_callables,
            inst_id,
            (uint32_t) n_inputs, call_inputs,
            (uint32_t) n_callables * n_outputs, rv_values,
            checkpoints, outputs
        );

        for (size_t i = 0; i < n_inputs; ++i)
            jit_var_dec_ref(call_inputs[i]);
        for (size_t i = 0; i < n_callables * n_outputs; ++i)
            jit_var_dec_ref(rv_values[i]);

        rec.disarm();
    }
}

int main() {
    jit_set_log_level_stderr(LogLevel::Info);
    jit_init((uint32_t) JitBackend::Metal);

    if (!jit_has_backend(JitBackend::Metal)) {
        fprintf(stderr, "Metal backend not available\n");
        return 1;
    }

    int failures = 0;

    // ===================================================================
    // Test 1: Basic vcall — two callables with different computations
    // ===================================================================
    {
        printf("Test 1: Basic vcall\n");

        struct Base {
            virtual ~Base() = default;
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
        const char *domain = "Base1";

        uint32_t i1 = jit_registry_put("Metal", domain, &a1);
        uint32_t i2 = jit_registry_put("Metal", domain, &a2);

        Float x = arange<Float>(10);

        // self = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
        UInt32 self = arange<UInt32>(10) % 3;

        auto f_call = [](void *self2, uint32_t *inputs, uint32_t *outputs) {
            Base *base = (Base *) self2;
            Float x2 = Float::borrow(inputs[0]);
            Float rv = base->f(x2);
            outputs[0] = rv.release();
        };

        for (int opt = 0; opt < 2; opt++) {
            jit_set_flag(JitFlag::OptimizeCalls, opt);

            uint32_t inputs[1] = { x.index() };
            uint32_t outputs[1] = { 0 };
            Mask mask = Mask::steal(jit_var_bool(Backend, true));

            symbolic_call<2, 1, 1>(
                Backend, "Metal", domain, false,
                self.index(), mask.index(),
                f_call, inputs, outputs);

            Float y = Float::steal(outputs[0]);
            const char *result = y.str();
            // self=0 → zero, self=1 → A1: (x+10)*2, self=2 → A2: (x+100)*2
            // x=[0..9], self=[0,1,2,0,1,2,0,1,2,0]
            // Expected: [0, 22, 204, 0, 28, 210, 0, 34, 216, 0]
            const char *expected = "[0, 22, 204, 0, 28, 210, 0, 34, 216, 0]";
            if (strcmp(result, expected) != 0) {
                printf("  FAIL (opt=%d): got %s, expected %s\n", opt, result, expected);
                failures++;
            } else {
                printf("  PASS (opt=%d)\n", opt);
            }
        }

        jit_registry_remove(&a1);
        jit_registry_remove(&a2);
    }

    // ===================================================================
    // Test 2: Single callable (no switch needed)
    // ===================================================================
    {
        printf("Test 2: Single callable\n");

        struct Base {
            virtual ~Base() = default;
            virtual Float f(Float x) = 0;
        };

        struct OnlyOne : Base {
            Float f(Float x) override { return x * 3; }
        };

        OnlyOne a1;
        const char *domain = "Base2";
        jit_registry_put("Metal", domain, &a1);

        Float x = arange<Float>(5);

        // All elements use instance 1
        UInt32 self = full<UInt32>(1, 5);

        auto f_call = [](void *self2, uint32_t *inputs, uint32_t *outputs) {
            Base *base = (Base *) self2;
            Float x2 = Float::borrow(inputs[0]);
            Float rv = base->f(x2);
            outputs[0] = rv.release();
        };

        uint32_t inputs[1] = { x.index() };
        uint32_t outputs[1] = { 0 };
        Mask mask = Mask::steal(jit_var_bool(Backend, true));

        symbolic_call<1, 1, 1>(
            Backend, "Metal", domain, false,
            self.index(), mask.index(),
            f_call, inputs, outputs);

        Float y = Float::steal(outputs[0]);
        const char *result = y.str();
        const char *expected = "[0, 3, 6, 9, 12]";
        if (strcmp(result, expected) != 0) {
            printf("  FAIL: got %s, expected %s\n", result, expected);
            failures++;
        } else {
            printf("  PASS\n");
        }

        jit_registry_remove(&a1);
    }

    jit_shutdown(0);

    if (failures == 0)
        printf("\nAll Metal vcall tests passed!\n");
    else
        printf("\n%d Metal vcall test(s) FAILED!\n", failures);

    return failures;
}
