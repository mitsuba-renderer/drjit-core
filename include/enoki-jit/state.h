/*
    enoki-jit/state.h -- JitState RAII helper class (implementation detail)

    This header file defines the 'JitState' class, which is an RAII helper
    class used by Enoki's loop/vcall recording code. It enables setting various
    attributes of the JIT compiler and recovering if an exception is thrown.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <enoki-jit/jit.h>
#include <cassert>

NAMESPACE_BEGIN(enoki)
NAMESPACE_BEGIN(detail)

/**
 * \brief JitState RAII wrapper
 *
 * This class encapsulates several configuration attributes of Enoki-JIT (the
 * mask stack, variable name prefixes, the CSE scope, whether program recording
 * is enabled.)
 *
 * The <tt>set_*</tt>, <tt>clear_*</tt>, <tt>begin_*</tt>, and <tt>end_*</tt>
 * methods can be used to change and clear these attributes. The deconstructor
 * will clear any remaining state.
 */
template <JitBackend Backend> struct JitState {
    JitState()
        : m_mask_set(false), m_prefix_set(false),
          m_cse_scope_set(false), m_recording(false) { }

    ~JitState() {
        if (m_mask_set)
            clear_mask();
        if (m_prefix_set)
            clear_prefix();
        if (m_recording)
            end_recording();
        if (m_cse_scope_set)
            jit_set_cse_scope(Backend, m_cse_scope);
    }

    void begin_recording() {
        assert(!m_recording);
        m_checkpoint = jit_record_begin(Backend);
        m_recording = true;
    }

    uint32_t checkpoint() const { return m_checkpoint; }

    void end_recording() {
        assert(m_recording);
        jit_record_end(Backend, m_checkpoint);
        m_recording = false;
    }

    void set_mask(uint32_t index, bool combine = true) {
        assert(!m_mask_set);
        jit_var_mask_push(Backend, index, combine);
        m_mask_set = true;
    }

    void clear_mask() {
        assert(m_mask_set);
        jit_var_mask_pop(Backend);
        m_mask_set = false;
    }

    void set_prefix(const char *label) {
        assert(!m_prefix_set);
        jit_prefix_push(Backend, label);
        m_prefix_set = true;
    }

    void clear_prefix() {
        assert(m_prefix_set);
        jit_prefix_pop(Backend);
        m_prefix_set = false;
    }

    void new_scope() {
        if (!m_cse_scope_set) {
            m_cse_scope = jit_cse_scope(Backend);
            m_cse_scope_set = true;
        }
        jit_new_cse_scope(Backend);
    }

    void clear_scope() {
        assert(m_cse_scope_set);
        jit_set_cse_scope(Backend, m_cse_scope);
        m_cse_scope_set = false;
    }

private:
    bool m_mask_set;
    bool m_prefix_set;
    bool m_cse_scope_set;
    bool m_recording;
    uint32_t m_cse_scope;
    uint32_t m_checkpoint;
};


NAMESPACE_END(detail)
NAMESPACE_END(enoki)
