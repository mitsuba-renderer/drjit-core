/*
    enoki/loop.h -- Infrastructure to record CUDA and LLVM loops

    Enoki is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <enoki-jit/jit.h>
#include <enoki-jit/containers.h>
#include <enoki-jit/state.h>

NAMESPACE_BEGIN(enoki)
NAMESPACE_BEGIN(detail)
template <typename Value> void ad_inc_ref(int32_t) noexcept;
template <typename Value> void ad_dec_ref(int32_t) noexcept;
template <typename Value> void ad_traverse_postponed();
template <typename Value, typename Mask>
int32_t ad_new_select(const char *, size_t, const Mask &, int32_t, int32_t);

NAMESPACE_END(detail)

template <typename Mask, typename SFINAE = int> struct Loop;

/// Scalar fallback, expands into normal C++ loop
template <typename Value>
struct Loop<Value, enable_if_t<std::is_scalar_v<Value>>> {
    Loop(const Loop &) = delete;
    Loop(Loop &&) = delete;
    Loop& operator=(const Loop &) = delete;
    Loop& operator=(Loop &&) = delete;

    void init() { }
    template <typename... Ts> void put(Ts&...) { }
    bool operator()(bool mask) { return mask; }
    template <typename... Args> Loop(const char*, Args&...) { }
};

/// Array case, expands into a symbolic or wavefront-style loop
template <typename Value>
struct Loop<Value, enable_if_jit_array_t<Value>> {
    static constexpr JitBackend Backend = backend_v<Value>;
    static constexpr bool IsDiff = is_diff_array_v<Value> &&
        std::is_floating_point_v<scalar_t<Value>>;

    using Mask = mask_t<Value>;

    Loop(const Loop &) = delete;
    Loop(Loop &&) = delete;
    Loop& operator=(const Loop &) = delete;
    Loop& operator=(Loop &&) = delete;

    template <typename... Args>
    Loop(const char *name, Args &... args)
        : m_state(0), m_size(0), m_record(jit_flag(JitFlag::LoopRecord)) {

        size_t size = strlen(name) + 1;
        m_name = ek_unique_ptr<char[]>(new char[size]);
        memcpy(m_name.get(), name, size);

        /// Immediately initialize if loop state is specified
        if constexpr (sizeof...(Args) > 0) {
            put(args...);
            init();
        }
    }

    ~Loop() {
        if (m_state != 0 && m_state != 3 && m_state != 4)
            jit_log(
                LogLevel::Warn,
                "Loop(\"%s\"): destructed in an inconsistent state. An "
                "exception or disallowed scalar control flow (break, continue) "
                "likely caused the loop to exit prematurely. Cleaning up..",
                m_name.get());

        if (m_record) {
            for (size_t i = 0; i < m_index_body.size(); ++i)
                jit_var_dec_ref_ext(m_index_body[i]);

            jit_var_dec_ref_ext(m_loop_cond);
            jit_var_dec_ref_ext(m_loop_body);
        } else {
            for (size_t i = 0; i < m_index_out.size(); ++i)
                jit_var_dec_ref_ext(m_index_out[i]);

            if constexpr (IsDiff) {
                using Type = typename Value::Type;

                for (size_t i = 0; i < m_index_out_ad.size(); ++i) {
                    int32_t index = m_index_out_ad[i];
                    detail::ad_dec_ref<Type>(index);
                }
            }
        }
    }

    /// Register JIT variable indices of loop variables
    template <typename T, typename... Ts>
    void put(T &value, Ts &... args) {
        if constexpr (is_array_v<T>) {
            if constexpr (array_depth_v<T> == 1) {
                if constexpr (IsDiff && is_diff_array_v<T> &&
                              std::is_floating_point_v<scalar_t<T>>) {
                    if (m_record && grad_enabled(value))
                        jit_raise(
                            "Loop::put(): one of the supplied loop "
                            "variables is attached to the AD graph (i.e. "
                            "grad_enabled(..) is true). However, recorded "
                            "loops cannot be differentiated in their entirety. "
                            "You have two options: either disable loop "
                            "recording via set_flag(JitFlag::LoopRecord, "
                            "false). Alternatively, you could implement the "
                            "adjoint of the loop using ek::CustomOp.");
                    put(value.detach_());
                    m_index_p_ad[m_index_p_ad.size() - 1] = value.index_ad_ptr();
                } else if constexpr (is_jit_array_v<T>) {
                    if (m_state)
                        jit_raise("Loop::put(): must be called "
                                  "*before* initialization!");
                    if (value.index() == 0)
                        jit_raise("Loop::put(): a loop variable (or "
                                  "an element of a data structure provided "
                                  "as a loop variable) is unintialized!");
                    m_index_p.push_back(value.index_ptr());
                    m_index_p_ad.push_back(nullptr);
                    m_index_in.push_back(value.index());
                    m_invariant.push_back(0);
                }
            } else {
                for (size_t i = 0; i < value.size(); ++i)
                    put(value.entry(i));
            }
        } else if constexpr (is_enoki_struct_v<T>) {
            struct_support_t<T>::apply_1(value, [&](auto &x) { put(x); });
        }
        put(args...);
    }

    void put() { }

    /// Configure the loop variables for recording
    void init() {
        if (!m_record)
            return;

        if (m_state)
            jit_raise("Loop(\"%s\"): was already initialized!", m_name.get());

        // Capture JIT state and begin recording session
        m_jit_state.begin_recording();
        m_jit_state.new_scope();

        m_loop_cond = jit_var_loop_init(m_index_p.data(), m_index_p.size());

        m_state = 1;
        jit_log(::LogLevel::InfoSym,
                "Loop(\"%s\"): --------- begin recording loop ---------", m_name.get());
    }

    bool operator()(const Mask &cond) {
        if (m_record)
            return cond_record(cond);
        else
            return cond_wavefront(cond);
    }

protected:
    bool cond_record(const Mask &cond) {
        uint32_t n = (uint32_t) m_index_p.size();
        bool has_invariant;
        uint32_t se = 0;

        switch (m_state) {
            case 0:
                jit_raise("Loop(\"%s\"): must be initialized first!", m_name.get());
                break;

            case 1:
                m_cond = detach(cond);

                /// New CSE scope for the loop body
                m_jit_state.new_scope();
                m_loop_body = jit_var_new_stmt(Backend, VarType::Void, "", 1, 1,
                                               m_cond.index_ptr());

                // Phi nodes represent state at the beginning of the loop body
                for (size_t i = 0; i < m_index_p.size(); ++i) {
                    uint32_t &index = *m_index_p[i];
                    uint32_t next = jit_var_new_stmt_2(
                        Backend, jit_var_type(index),
                        Backend == JitBackend::LLVM
                            ? "$r0 = phi <$w x $t0> [ $r1, %l_$i2_cond ]"
                            : "mov.$t0 $r0, $r1",
                        index, m_loop_cond);
                    jit_var_dec_ref_ext(index);
                    jit_var_inc_ref_ext(next);
                    m_index_body.push_back(next);
                    index = next;
                }

                m_state++;
                if constexpr (Backend == JitBackend::LLVM)
                    m_jit_state.set_mask(m_cond.index());
                return true;

            case 2:
            case 3:
                if constexpr (Backend == JitBackend::LLVM)
                    m_jit_state.clear_mask();

                for (uint32_t i = 0; i < n; ++i)
                    m_index_out.push_back(*m_index_p[i]);

                se = jit_var_loop(
                    m_name.get(), m_loop_cond, m_loop_body, (uint32_t) n,
                    m_index_body.data(), m_index_out.data(),
                    m_jit_state.checkpoint(), m_index_out.data(), m_state == 2,
                    m_invariant.data());
                m_jit_state.end_recording();

                has_invariant = false;
                for (uint32_t i = 0; i < n; ++i)
                    has_invariant |= (bool) m_invariant[i];

                if (has_invariant && m_state == 2) {
                    /* Some loop variables don't change while running the loop.
                       This can be exploited by recording the loop a second time
                       while taking this information into account. */
                    jit_var_dec_ref_ext(se);
                    m_jit_state.begin_recording();

                    m_index_out.clear();

                    for (uint32_t i = 0; i < n; ++i) {
                        // Free outputs produced by current iteration
                        uint32_t &index = *m_index_p[i];
                        jit_var_dec_ref_ext(index);

                        if (m_invariant[i]) {
                            uint32_t input = m_index_in[i],
                                    &cur = m_index_body[i];
                            jit_var_inc_ref_ext(input);
                            jit_var_dec_ref_ext(cur);
                            m_index_body[i] = input;
                        }

                        index = m_index_body[i];
                        jit_var_inc_ref_ext(index);
                    }

                    m_state++;
                    if constexpr (Backend == JitBackend::LLVM)
                        m_jit_state.set_mask(m_cond.index());
                    jit_log(::LogLevel::InfoSym,
                            "Loop(\"%s\"): ----- recording loop body *again* ------", m_name.get());
                    return true;
                } else {
                    jit_var_mark_side_effect(se);
                    jit_log(::LogLevel::InfoSym,
                            "Loop(\"%s\"): --------- done recording loop ----------", m_name.get());
                    // No optimization opportunities, stop now.
                    for (uint32_t i = 0; i < n; ++i)
                        jit_var_dec_ref_ext(m_index_body[i]);
                    m_index_body.clear();

                    for (uint32_t i = 0; i < n; ++i) {
                        uint32_t &index = *m_index_p[i];
                        jit_var_dec_ref_ext(index);
                        index = m_index_out[i]; // steal ref
                    }

                    m_index_out.clear();
                    m_jit_state.clear_scope();
                    m_cond = detached_t<Mask>();
                    m_state++;

                    if constexpr (IsDiff) {
                        using Type = typename Value::Type;
                        if (m_jit_state.m_record_state == 0)
                            detail::ad_traverse_postponed<Type>();
                    }

                    return false;
                }

            default:
                jit_raise("Loop(): invalid state!");
        }

        return false;
    }

    bool cond_wavefront(const Mask &cond_) {
        Mask cond = cond_;

        if (m_cond.index()) {
            cond &= m_cond;

            // Need to mask loop variables for disabled lanes
            m_jit_state.clear_mask();
            for (uint32_t i = 0; i < m_index_p.size(); ++i) {
                uint32_t i1 = *m_index_p[i], i2 = m_index_out[i];
                *m_index_p[i] = jit_var_new_op_3(JitOp::Select, m_cond.index(), i1, i2);
                jit_var_dec_ref_ext(i1);
                jit_var_dec_ref_ext(i2);
            }
            m_index_out.clear();

            if constexpr (is_diff_array_v<Value>) {
                using Type = typename Value::Type;
                for (uint32_t i = 0; i < m_index_p_ad.size(); ++i) {
                    if (!m_index_p_ad[i])
                        continue;

                    int32_t i1 = *m_index_p_ad[i], i2 = m_index_out_ad[i],
                            index_new = 0;
                    if (i1 > 0 || i2 > 0)
                        index_new = detail::ad_new_select<Type>(
                            "select", jit_var_size(*m_index_p[i]),
                            m_cond, i1, i2);
                    *m_index_p_ad[i] = index_new;
                    detail::ad_dec_ref<Type>(i1);
                    detail::ad_dec_ref<Type>(i2);
                }
                m_index_out_ad.clear();
            }
        }

        // Ensure all loop state is evaluated
        jit_var_schedule(cond.index());
        for (uint32_t i = 0; i < m_index_p.size(); ++i)
            jit_var_schedule(*m_index_p[i]);
        jit_eval();

        // Do we run another iteration?
        if (jit_var_any(cond.index())) {
            for (uint32_t i = 0; i < m_index_p.size(); ++i) {
                uint32_t index = *m_index_p[i];
                jit_var_inc_ref_ext(index);
                m_index_out.push_back(index);
            }

            if constexpr (IsDiff) {
                using Type = typename Value::Type;

                for (uint32_t i = 0; i < m_index_p_ad.size(); ++i) {
                    int32_t index = 0;
                    if (m_index_p_ad[i]) {
                        index = *m_index_p_ad[i];
                        detail::ad_inc_ref<Type>(index);
                    }
                    m_index_out_ad.push_back(index);
                }
            }

            // Mask scatters/gathers/vcalls in the next iteration
            m_cond = detach(cond);
            m_jit_state.set_mask(m_cond.index());
            return true;
        } else {
            return false;
        }
    }

protected:
    /// A descriptive name
    ek_unique_ptr<char[]> m_name;

    /// Bariable representing the start of a symbolic loop
    uint32_t m_loop_cond = 0;

    /// Bariable representing the body of a symbolic loop
    uint32_t m_loop_body = 0;

    /// Pointers to loop variable indices
    ek_vector<uint32_t *> m_index_p;
    ek_vector<int32_t *> m_index_p_ad;

    /// Loop variable indices before entering the loop
    ek_vector<uint32_t> m_index_in;

    /// Loop variable indices at the top of the loop body
    ek_vector<uint32_t> m_index_body;

    /// Loop variable indices after the end of the loop
    ek_vector<uint32_t> m_index_out;
    ek_vector<int32_t> m_index_out_ad;

    /// Detects loop-invariant variables to trigger optimizations
    ek_vector<uint8_t> m_invariant;

    /// Stashed mask variable from the previous iteration
    detached_t<Mask> m_cond;

    /// RAII wrapper for the mask stack
    detail::JitState<Backend> m_jit_state;

    /// Index of the symbolic loop state machine
    uint32_t m_state;

    /// Keeps track of the size of loop variables to catch issues
    size_t m_size;

    /// Is the loop being recorded symbolically?
    bool m_record;
};

NAMESPACE_END(enoki)
