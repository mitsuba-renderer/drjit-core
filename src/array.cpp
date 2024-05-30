/*
    src/array.cpp -- Functionality to create, read, and write variable arrays

    Copyright (c) 2024 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "internal.h"
#include "var.h"
#include "log.h"
#include "eval.h"
#include "array.h"
#include "var.h"

uint32_t jitc_array_create(JitBackend backend, VarType vt, size_t size,
                           size_t length, uint32_t pred) {
    jitc_check_size("jit_array_create", size);
    if (length > 0xFFFF)
        jitc_raise("jit_array_create(): variable arrays are limited to a "
                   "maximum of 65536 elements");

    if (size == 0 || length == 0)
        return 0;

    Variable v;
    v.size = (uint32_t) size;
    v.kind = (uint32_t) VarKind::Array;
    v.backend = (uint32_t) backend;
    v.type = (uint32_t) vt;
    v.array_state = (uint32_t) ArrayState::Clean;
    v.array_length = (uint16_t) length;

    if (pred) {
        v.dep[0] = pred;
        jitc_var_inc_ref(pred);
    }

    uint32_t index = jitc_var_new(v, true);
    jitc_var(index)->scope = 0;
    return index;
}

uint32_t jitc_array_init(uint32_t target, uint32_t value) {
    if (target == 0 && value == 0)
        return 0;

    #define fail_if(cond, fmt, ...)                               \
        if (unlikely(cond))                                       \
            jitc_raise(                                           \
                "jit_array_init(target=r%u, value=r%u): " fmt,    \
                target, target, ##__VA_ARGS__);

    fail_if(target == 0, "target array is uninitialized.");
    fail_if(value == 0, "value is uninitialized.");

    Variable *vt = jitc_var(target),
             *vv = jitc_var(value);

    fail_if(!vt->is_array() || vv->is_array(),
            "the 'target' argument must be an array, the others not.");
    fail_if(vt->backend != vv->backend, "cannot mix backends!");
    fail_if(vt->type != vv->type, "incompatible types!");
    fail_if(vt->size != vv->size && vt->size != 1,
            "incompatible sizes (%u and %u)!", vt->size, vv->size);

    #undef fail_if

    Variable v2;
    v2.size = (uint32_t) std::max(vv->size, vt->size);
    v2.kind = (uint32_t) VarKind::ArrayInit;
    v2.backend = (uint32_t) vv->backend;
    v2.type = (uint32_t) vv->type;
    v2.array_state = (uint32_t) ArrayState::Clean;
    v2.array_length = (uint16_t) vt->array_length;
    v2.dep[0] = target;
    v2.dep[1] = value;
    jitc_var_inc_ref(target, vt);
    jitc_var_inc_ref(value, vv);

    return jitc_var_new(v2, true);
}

size_t jitc_array_length(uint32_t index) {
    if (index == 0)
        return 0;
    const Variable *v = jitc_var(index);
    if (!v->is_array())
        jitc_raise("jit_array_length(r%u): target is not an array!", index);
    return v->array_length;
}

uint32_t jitc_array_read(uint32_t source, uint32_t offset, uint32_t mask_) {
    if (source == 0 && offset == 0 && mask_ == 0)
        return 0;

    #define fail_if(cond, fmt, ...)                                          \
        if (unlikely(cond))                                                  \
            jitc_raise(                                                      \
                "jit_array_read(source=r%u, offset=r%u, mask=r%u): " fmt,    \
                source, offset, mask_, ##__VA_ARGS__);

    fail_if(source == 0, "source array is uninitialized.");
    fail_if(offset == 0, "offset is uninitialized.");
    fail_if(mask_ == 0, "mask is uninitialized.");

    Ref mask = steal(jitc_var_mask_apply(mask_, 1));
    Variable *vo = jitc_var(offset),
             *vs = jitc_var(source),
             *vm = jitc_var(mask);
    JitBackend backend = (JitBackend) vm->backend;
    uint32_t array_length = vs->array_length;

    if (vm->is_literal() && vm->literal == 0) {
        // Elide the read (it is fully masked)
        uint64_t zero = 0;
        return jitc_var_literal(backend, (VarType) jitc_var(source)->type,
                                &zero, 1, 0);
    } else if ((vo->is_literal() || vo->size == 1) &&
               (vm->kind == (uint32_t) VarKind::DefaultMask ||
                vm->kind == (uint32_t) VarKind::CallMask)) {
        // When we know that the read is always from a uniform location, we
        // can remove any default masking, which slightly improves code generation
        uint64_t one = 1;
        mask = steal(jitc_var_literal(backend, VarType::Bool, &one, 1, 0));
    }
    uint32_t flags = jit_flags();

    if (flags & (uint32_t) JitFlag::Debug)
        mask = steal(jitc_var_check_bounds(
            BoundsCheckType::ArrayRead, offset, mask, array_length));

    vs = jitc_var(source);
    vo = jitc_var(offset);
    vm = jitc_var(mask);

    size_t size = std::max(std::max(vs->size, vo->size), vm->size);

    // A few sanity checks
    fail_if(!vs->is_array() || vo->is_array() || vm->is_array(),
            "the 'source' argument must be an array, the others not.");
    fail_if(vo->backend != vs->backend, "can't mix backends.");
    fail_if((VarType) vo->type != VarType::UInt32,
            "offset must be an unsigned 32-bit integer.");
    fail_if((VarType) vm->type != VarType::Bool,
            "mask must be a boolean array.");
    fail_if((vo->size != size && vo->size != 1) ||
            (vs->size != size && vs->size != 1) ||
            (vm->size != size && vm->size != 1),
            "incompatible sizes (%u, %u, and %u)", vs->size, vo->size, vm->size);
    fail_if(vo->is_literal() && vo->literal >= (uint64_t) array_length,
            "out of bounds read (source size=%u, offset=%zu)",
            array_length, vo->literal);

    #undef fail_if

    if ((flags & (uint32_t) JitFlag::SymbolicScope) == 0) {
        if (vs->kind == (uint32_t) VarKind::ArrayInit) {
            jitc_var_inc_ref(vs->dep[1]);
            return vs->dep[1];
        } else if (vs->kind == (uint32_t) VarKind::Array) {
            // uninitialized read
            return jitc_var_undefined(backend, (VarType) vs->type, size);
        }
    }

    if (vo->is_dirty()) {
        jitc_eval(thread_state((JitBackend) vs->backend));

        vs = jitc_var(source);
        vo = jitc_var(offset);

        if (vo->is_dirty())
            jitc_raise_dirty_error(offset);
    }

    Variable v;
    v.kind = (uint32_t) VarKind::ArrayRead;
    v.backend = vs->backend;
    v.type = vs->type;
    v.size = std::max(vs->size, vo->size);
    v.symbolic = vs->symbolic || vo->symbolic;

    v.dep[0] = source;
    jitc_var_inc_ref(source, vs);
    v.dep[1] = mask;
    jitc_var_inc_ref(mask, vm);

    if (vo->is_literal()) {
        v.literal = vo->literal;
    } else {
        v.dep[2] = offset;
        jitc_var_inc_ref(offset, vo);
    }

    return jitc_var_new(v);
}

uint32_t jitc_array_buffer(uint32_t index) {
    const Variable *v = jitc_var(index);
    if (v->kind == (uint32_t) VarKind::Array ||
        v->kind == (uint32_t) VarKind::ArrayInit)
        return index;
    return jitc_array_buffer(v);
}

uint32_t jitc_array_write(uint32_t target, uint32_t offset, uint32_t value,
                          uint32_t mask_) {
    if (target == 0 && offset == 0 && value == 0 && mask_ == 0)
        return 0;

    #define fail_if(cond, fmt, ...)                                           \
        if (unlikely(cond))                                                   \
            jitc_raise("jitc_array_write(target=r%u, offset=r%u, value=r%u, " \
                       "mask=r%u): " fmt,                                     \
                   target, offset, value, mask_, ##__VA_ARGS__);

    fail_if(target == 0, "target array is uninitialized.");
    fail_if(offset == 0, "offset is uninitialized.");
    fail_if(value == 0, "value is uninitialized.");
    fail_if(mask_ == 0, "mask is uninitialized.");

    Ref mask = steal(jitc_var_mask_apply(mask_, 1));
    Variable *vm = jitc_var(mask),
             *vo = jitc_var(offset);
    JitBackend backend = (JitBackend) vm->backend;

    if (vm->is_literal() && vm->literal == 0) {
        // Elide the write (it is fully masked)
        jitc_var_inc_ref(target);
        return target;
    } else if ((vo->is_literal() || vo->size == 1) &&
               (vm->kind == (uint32_t) VarKind::DefaultMask ||
                vm->kind == (uint32_t) VarKind::CallMask)) {
        // When we know that the write is always to a uniform location, we
        // can remove any default masking, which slightly improves code generation
        uint64_t one = 1;
        mask = steal(jitc_var_literal(backend, VarType::Bool, &one, 1, 0));
    }

    Variable *vt = jitc_var(target),
             *vv = jitc_var(value);

    vm = jitc_var(mask);
    vo = jitc_var(offset);

    VarType type = (VarType) vt->type;
    uint16_t array_length = vt->array_length;
    uint32_t size =
        std::max(std::max(vt->size, vo->size), std::max(vv->size, vm->size));

    // A few sanity checks
    fail_if(!vt->is_array() || vv->is_array() || vm->is_array(),
            "the 'target' argument must be an array, the others not.");
    fail_if(vo->is_array() || vv->is_array(), "origin/value may not be arrays.");
    fail_if(vo->backend != vt->backend || vo->backend != vv->backend ||
            vo->backend != vm->backend, "can't mix backends.");
    fail_if(vv->type != vt->type, "target/value type mismatch.");
    fail_if((VarType) vo->type != VarType::UInt32,
            "offset must be an unsigned 32-bit integer array.");
    fail_if((VarType) vm->type != VarType::Bool,
            "mask must be a boolean array.");
    fail_if((vo->size != size && vo->size != 1) ||
            (vt->size != size && vt->size != 1) ||
            (vv->size != size && vv->size != 1) ||
            (vm->size != size && vm->size != 1),
            "incompatible sizes (%u, %u, %u, and %u)",
            vt->size, vo->size, vv->size, vm->size);
    fail_if(vo->is_literal() && vo->literal >= array_length,
            "out of bounds write (target size=%u, offset=%zu)",
            (uint32_t) array_length, vo->literal);

    #undef fail_if

    if (vo->is_dirty() || vv->is_dirty() || vm->is_dirty()) {
        jitc_eval(thread_state(backend));

        vt = jitc_var(target);
        vo = jitc_var(offset);
        vv = jitc_var(value);
        vm = jitc_var(mask);

        if (vo->is_dirty())
            jitc_raise_dirty_error(offset);
        if (vv->is_dirty())
            jitc_raise_dirty_error(value);
        if (vm->is_dirty())
            jitc_raise_dirty_error(mask);
    }

    if (jit_flag(JitFlag::Debug)) {
        mask = steal(jitc_var_check_bounds(
            BoundsCheckType::ArrayWrite, offset, mask, array_length));
        vt = jitc_var(target);
        vo = jitc_var(offset);
        vv = jitc_var(value);
        vm = jitc_var(mask);
    }

    bool literal_offset = vo->is_literal();

    Variable v;
    v.kind = (uint32_t) VarKind::ArrayWrite;
    v.type = (uint32_t) type;
    v.backend = (uint32_t) backend;
    v.size = size;
    v.array_state = (uint32_t) ArrayState::Clean;
    v.array_length = array_length;
    v.symbolic = vt->symbolic || vv->symbolic || vm->symbolic;
    v.dep[0] = target;
    jitc_var_inc_ref(target, vt);
    v.dep[1] = value;
    jitc_var_inc_ref(value, vv);
    v.dep[2] = mask;
    jitc_var_inc_ref(mask, vm);

    if (literal_offset) {
        v.literal = vo->literal;
    } else {
        v.dep[3] = offset;
        jitc_var_inc_ref(offset, vo);
    }

    // Array writes increment the scope ID to prevent reordering of subsequent
    // reads. For example, consider the following program:
    //
    // arr1 = array_write(arr0, x0)
    // x1 = array_read(arr1)
    // arr2 = array_write(arr1, x2)
    //
    // If compilation were to move the read to a location following the second
    // write, a redundant array copy would need to be made.

    jitc_new_scope(backend);
    uint32_t op = jitc_var_new(v, false);
    jitc_new_scope(backend);

    // Create a new storage region to store the modified array. It is
    // often not needed and optimized away during JIT compilation.
    Ref storage = steal(jitc_array_create(backend, type, size, array_length,
                                          jitc_array_buffer(target)));

    Variable *v_op = jitc_var(op);

    if (literal_offset) {
        v_op->dep[3] = storage.release();
    } else {
        // Out of indices, store in 'extra' data structure
        VariableExtra *ext = jitc_var_extra(v_op);
        ext->callback_data = (void *) (uintptr_t) storage.release();
        ext->callback_internal = true;
        ext->callback = [](uint32_t, int free, void *p) {
            if (free)
                jitc_var_dec_ref((uint32_t) (uintptr_t) p);
        };
    }

    return op;
}

uint32_t jitc_array_buffer(const Variable *v) {
    if (v->extra)
        return (uint32_t) (uintptr_t) state.extra[v->extra].callback_data;
    else
        return v->dep[3];
}

void jitc_process_array_op(VarKind kind, Variable *v) {
    // This function evolves a simple state machine that elides unnecessary
    // array copies. For context, a 'write' operation performed via the
    // jit_array_* API does not modify the input. Instead, it produces a copy of
    // the input array representing the array state following the write. In most
    // cases, this copy is not actually needed, and we wish to avoid the
    // expense. The following detects cases where an array is modified and
    // subsequently accessed via further reads/writes. In this case, the
    // original array is marked as 'conflicted', which means that any subsequent
    // writes to it must make a copy.
    //
    if (v->is_array()) {
        v->array_state = (uint32_t) ArrayState::Clean;

        if (kind == VarKind::Array)
            return;
    }

    uint32_t parent_id = v->dep[0];
    Variable *parent = jitc_var(parent_id);

    ArrayState array_state = (ArrayState) parent->array_state;

    if (kind == VarKind::ArrayWrite || kind == VarKind::ArrayInit) {
        switch (array_state) {
            case ArrayState::Clean:      array_state = ArrayState::Modified; break;
            case ArrayState::Modified:
            case ArrayState::Conflicted: array_state = ArrayState::Conflicted; break;
            default: break;
        }
        if (v->output_flag)
            array_state = ArrayState::Conflicted;
    } else if (kind == VarKind::ArrayRead || kind == VarKind::ArrayPhi) {
        switch (array_state) {
            case ArrayState::Clean:      array_state = ArrayState::Clean; break;
            case ArrayState::Modified:
            case ArrayState::Conflicted: array_state = ArrayState::Conflicted; break;
            default: break;
        }
    } else {
        jitc_raise("jit_process_array_op(): internal error! (1)");
    }

    if (array_state == ArrayState::Invalid)
        jitc_raise("jit_process_array_op(): internal error! (2)");

    parent->array_state = (uint32_t) array_state;
    if (parent->kind == (uint32_t) VarKind::ArrayWrite) {
        Variable *buf = jitc_var(jitc_array_buffer(parent_id));
        buf->array_state = (uint32_t) array_state;
    }
}
