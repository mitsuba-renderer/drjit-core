#include "reorder.h"
#include "eval.h"
#include "internal.h"
#include "log.h"
#include "var.h"

void jitc_reorder(uint32_t key, uint32_t num_bits, uint32_t n_values,
                        uint32_t *values, uint32_t *out) {
    Variable *v_key  = jitc_var(key);

    if ((JitBackend) v_key->backend != JitBackend::CUDA) {
        for (uint32_t i = 0; i < n_values; ++i) {
            jitc_var_inc_ref(values[i]);
            out[i] = values[i];
        }
        return;
    }

    if ((VarType) v_key->type != VarType::UInt32)
        jitc_raise("jit_reorder(): 'key' must be an unsigned 32-bit array.");

    uint32_t size = 0;
    bool symbolic = v_key->symbolic;
    bool dirty = v_key->is_dirty();
    for (uint32_t i = 0; i <= n_values; ++i) {
        uint32_t index = (i < n_values) ? values[i] : key;
        Variable *v = jitc_var(index);
        size = std::max(size, v->size);
        symbolic |= (bool) v->symbolic;
        dirty |= v->is_dirty();
    }

    for (uint32_t i = 0; i <= n_values; ++i) {
        uint32_t index = (i < n_values) ? values[i] : key;
        const Variable *v = jitc_var(index);
        if (v->size != 1 && v->size != size)
            jitc_raise("jit_reorder(): incompatible array sizes!");
    }

    if (dirty) {
        jitc_eval(thread_state(JitBackend::CUDA));

        for (uint32_t i = 0; i < n_values; ++i) {
            if (jitc_var(values[i])->is_dirty())
                jitc_raise_dirty_error(values[i]);
        }
    }

    if (num_bits < 1)
        jitc_fail("jit_reorder(): the key must be at least one bit!");
    if (num_bits > 16)
        jitc_fail("jit_reorder(): a maximum of 16 bits can be used for "
                  "the key!");

    // Guarantee that the reordering is assembled after anything that precedes this operation
    jitc_new_scope(JitBackend::CUDA);

    uint32_t reorder = jitc_var_new_node_1(
        JitBackend::CUDA, VarKind::ReorderThread, VarType::Void, size, symbolic,
        key, v_key, num_bits);
    Variable *v_reorder = jitc_var(reorder);
    v_reorder->optix = 1;

    // Guarantee that the reordering is assembled before anything that follows
    jitc_new_scope(JitBackend::CUDA);

    for (uint32_t i = 0; i < n_values; ++i) {
        // Values are just a `Bitcast` of their original value (no-op). We
        // additionally add the reodering node as the second dependency. This
        // guarantees that it will be detected during `jitc_var_traverse` and
        // that it will only be assembled once, despite it not being directly
        // used when assembling a `Bitcast` node.
        Variable *v_value = jitc_var(values[i]);
        out[i] = jitc_var_new_node_2(JitBackend::CUDA, VarKind::Bitcast,
                                     (VarType) v_value->type, v_value->size,
                                     v_value->symbolic, values[i], v_value,
                                     reorder, v_reorder);
    }

    jitc_var_dec_ref(reorder, v_reorder);
}
