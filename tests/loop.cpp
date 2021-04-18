#include "test.h"
#include <enoki-jit/containers.h>
#include <utility>

template <typename Mask> struct Loop {
    static constexpr JitBackend Backend = Mask::Backend;

    template <typename... Args>
    Loop(const char *name, Args &... args)
        : m_name(name), m_state(0), m_se_offset((uint32_t) -1),
          m_size(0), m_record(jit_flag(JitFlag::LoopRecord)),
          m_cse_scope(jit_cse_scope(Backend)) {
        if constexpr (sizeof...(Args) > 0) {
            (put(args), ...);
            init();
        }
    }

    ~Loop() {
        // Recover if an error occurred while recording a loop symbolically
        if (m_record && m_se_offset != (uint32_t) -1) {
            jit_side_effects_rollback(Backend, m_se_offset);
            jit_set_flag(JitFlag::Recording, m_se_flag);

            for (size_t i = 0; i < m_index_body.size(); ++i)
                jit_var_dec_ref_ext(m_index_body[i]);
        }

        // Recover if an error occurred while running a wavefront-style loop
        if (!m_record && m_index_out.size() > 0) {
            for (size_t i = 0; i < m_index_out.size(); ++i)
                jit_var_dec_ref_ext(m_index_out[i]);
        }

        if (m_state != 0 && m_state != 3 && m_state != 4)
            jit_log(Warn, "enoki::Loop(): destructed in an inconsistent state.");

        jit_var_dec_ref_ext(m_loop_start);
        jit_var_dec_ref_ext(m_loop_cond);
        jit_set_cse_scope(Backend, m_cse_scope);
    }

    /// Register a loop variable // TODO: nested arrays, structs, etc.
    template <typename Value> void put(Value &value) {
        /// XXX complain when variables are attached
        m_index_p.push_back(value.index_ptr());
        m_index_in.push_back(value.index());
        m_invariant.push_back(0);
        size_t size = value.size();
        if (m_size != 0 && size != 1 && size != m_size)
            jit_raise("enoki::Loop.put(): loop variables have inconsistent sizes!");
        if (size > m_size)
            m_size = size;
    }

    /// Configure the loop variables for recording
    void init() {
        if (m_state)
            jit_raise("Loop(): was already initialized!");

        if (m_record) {
            /* Wrap loop variables using placeholders that represent
               their state just before the loop condition is evaluated */
            m_se_flag = jit_flag(JitFlag::Recording);
            jit_set_flag(JitFlag::Recording, 1);
            m_se_offset = jit_side_effects_scheduled(Backend);

            jit_new_cse_scope(Backend);
            m_loop_start = jit_var_new_stmt(Backend, VarType::Void, "", 1, 0, nullptr);
            step();
            m_state = 1;
        }
    }

    bool operator()(const Mask &cond) {
        if (m_record)
            return cond_record(cond);
        else
            return cond_wavefront(cond);
    }

protected:
    struct MaskStackHelper {
    public:
        void push(uint32_t index) {
            if (m_armed)
                jit_fail("MaskStackHelper::internal error! (1)");
            jit_var_mask_push(Mask::Backend, index);
            m_armed = true;
        }
        void pop() {
            if (!m_armed)
                jit_fail("MaskStackHelper::internal error! (2)");
            jit_var_mask_pop(Mask::Backend);
            m_armed = false;
        }
        ~MaskStackHelper() {
            if (m_armed)
                pop();
        }
    private:
        bool m_armed = false;
    };

    bool cond_record(const Mask &cond) {
        uint32_t n = (uint32_t) m_index_p.size();
        bool has_invariant;

        switch (m_state) {
            case 0:
                jit_raise("Loop(): must be initialized first!");

            case 1:
                /* The loop condition has been evaluated now.  Wrap loop
                   variables using placeholders once more. They will represent
                   their state at the start of the loop body. */
                m_cond = cond; // detach
                m_loop_cond = jit_var_new_stmt(Backend, VarType::Void, "", 1, 1,
                                               m_cond.index_ptr());
                jit_new_cse_scope(Backend);
                step();
                for (uint32_t i = 0; i < n; ++i) {
                    uint32_t index = *m_index_p[i];
                    m_index_body.push_back(index);
                    jit_var_inc_ref_ext(index);
                }
                m_state++;
                if constexpr (Backend == JitBackend::LLVM)
                    m_mask_stack.push(cond.index());
                return true;

            case 2:
            case 3:
                if constexpr (Backend == JitBackend::LLVM)
                    m_mask_stack.pop();
                for (uint32_t i = 0; i < n; ++i)
                    m_index_out.push_back(*m_index_p[i]);

                jit_var_loop(m_name, m_loop_start, m_loop_cond,
                             (uint32_t) n, m_index_body.data(),
                             m_index_out.data(), m_se_offset,
                             m_index_out.data(), m_state == 2,
                             m_invariant.data());

                has_invariant = false;
                for (uint32_t i = 0; i < n; ++i)
                    has_invariant |= (bool) m_invariant[i];

                if (has_invariant && m_state == 2) {
                    /* Some loop variables don't change while running the loop.
                       This can be exploited by recording the loop a second time
                       while taking this information into account. */
                    jit_side_effects_rollback(Backend, m_se_offset);
                    m_index_out.clear();

                    for (uint32_t i = 0; i < n; ++i) {
                        // Free outputs produced by current iteration
                        uint32_t &index = *m_index_p[i];
                        jit_var_dec_ref_ext(index);

                        if (m_invariant[i]) {
                            uint32_t input = m_index_in[i],
                                    &cur = m_index_body[i];
                            if (jit_var_is_placeholder(input))
                                abort();
                            jit_var_inc_ref_ext(input);
                            jit_var_dec_ref_ext(cur);
                            m_index_body[i] = input;
                        }

                        index = m_index_body[i];
                        jit_var_inc_ref_ext(index);
                    }

                    m_state++;
                    if constexpr (Backend == JitBackend::LLVM)
                        m_mask_stack.push(cond.index());
                    return true;
                } else {
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
                    jit_set_flag(JitFlag::Recording, m_se_flag);
                    m_se_offset = (uint32_t) -1;
                    m_cond = Mask();
                    m_state++;
                    return false;
                }

            default:
                jit_raise("Loop(): invalid state!");
        }

        return false;
    }

    // Insert an indirection via placeholder variables
    void step() {
        for (size_t i = 0; i < m_index_p.size(); ++i) {
            uint32_t &index = *m_index_p[i],
                     next = jit_var_new_placeholder(index, 1, 0);
            jit_var_dec_ref_ext(index);
            index = next;
        }
    }

    bool cond_wavefront(const Mask &cond) {
        // Need to mask loop variables for disabled lanes
        if (m_cond.index()) {
            m_mask_stack.pop();
            for (uint32_t i = 0; i < m_index_p.size(); ++i) {
                uint32_t i1 = *m_index_p[i], i2 = m_index_out[i];
                *m_index_p[i] = jit_var_new_op_3(JitOp::Select, m_cond.index(), i1, i2);
                jit_var_dec_ref_ext(i1);
                jit_var_dec_ref_ext(i2);
            }
            m_index_out.clear();
            m_cond = m_cond && cond;
        } else {
            m_cond = cond;
        }

        // Ensure all loop state is evaluated
        jit_var_schedule(m_cond.index());
        for (uint32_t i = 0; i < m_index_p.size(); ++i)
            jit_var_schedule(*m_index_p[i]);
        jit_eval();

        // Do we run another iteration?
        if (jit_var_any(m_cond.index())) {
            for (uint32_t i = 0; i < m_index_p.size(); ++i) {
                uint32_t index = *m_index_p[i];
                jit_var_inc_ref_ext(index);
                m_index_out.push_back(index);
            }

            m_mask_stack.push(m_cond.index());
            return true;
        } else {
            m_cond = Mask();
            return false;
        }
    }

private:
    /// A descriptive name
    const char *m_name;

    /// Pointers to loop variable indices
    ek_vector<uint32_t *> m_index_p;

    /// Loop variable indices before entering the loop
    ek_vector<uint32_t> m_index_in;

    /// Loop variable indices at the top of the loop body
    ek_vector<uint32_t> m_index_body;

    /// Loop variable indices after the end of the loop
    ek_vector<uint32_t> m_index_out;

    /// Detects loop-invariant variables to trigger optimizations
    ek_vector<uint8_t> m_invariant;

    /// Stashed mask variable from the previous iteration
    Mask m_cond; // XXX detached_t<Mask>

    /// RAII wrapper for the mask stack
    MaskStackHelper m_mask_stack;

    /// Keeps track of the size of loop variables to catch issues
    size_t m_size;

    /// Offset in the side effects queue before the beginning of the loop
    uint32_t m_se_offset;

    /// State of the PostPoneSideEffects flag
    int m_se_flag;

    /// Index of the symbolic loop state machine
    uint32_t m_state;

    /// Is the loop being recorded symbolically
    bool m_record;

    /// Loop code generation hooks
    uint32_t m_loop_start = 0;
    uint32_t m_loop_cond = 0;
    uint32_t m_cse_scope;
};

TEST_BOTH(01_record_loop) {
    // Tests a simple loop evaluated at once, or in parts
    for (uint32_t i = 0; i < 3; ++i) {
        jit_set_flag(JitFlag::LoopRecord, i != 0);
        jit_set_flag(JitFlag::LoopOptimize, i == 2);

        for (uint32_t j = 0; j < 2; ++j) {
            UInt32 x = arange<UInt32>(10);
            Float y = zero<Float>(1);
            Float z = 1;

            Loop<Mask> loop("MyLoop", x, y, z);
            while (loop(x < 5)) {
                y += Float(x);
                x += 1;
                z += 1;
            }

            if (j == 0) {
                jit_var_schedule(z.index());
                jit_var_schedule(y.index());
                jit_var_schedule(x.index());
            }

            jit_assert(strcmp(z.str(), "[6, 5, 4, 3, 2, 1, 1, 1, 1, 1]") == 0);
            jit_assert(strcmp(y.str(), "[10, 10, 9, 7, 4, 0, 0, 0, 0, 0]") == 0);
            jit_assert(strcmp(x.str(), "[5, 5, 5, 5, 5, 5, 6, 7, 8, 9]") == 0);
        }
    }
}

TEST_BOTH(02_side_effect) {
    // Tests that side effects only happen once
    for (uint32_t i = 0; i < 3; ++i) {
        jit_set_flag(JitFlag::LoopRecord, i != 0);
        jit_set_flag(JitFlag::LoopOptimize, i == 2);

        for (uint32_t j = 0; j < 2; ++j) {
            UInt32 x = arange<UInt32>(10);
            Float y = zero<Float>(1);
            UInt32 target = zero<UInt32>(11);

            Loop<Mask> loop("MyLoop", x, y);
            while (loop(x < 5)) {
                scatter_reduce(ReduceOp::Add, target, UInt32(1), x);
                y += Float(x);
                x += 1;
            }

            if (j == 0) {
                jit_var_schedule(x.index());
                jit_var_schedule(y.index());
            }

            jit_assert(strcmp(y.str(), "[10, 10, 9, 7, 4, 0, 0, 0, 0, 0]") == 0);
            jit_assert(strcmp(x.str(), "[5, 5, 5, 5, 5, 5, 6, 7, 8, 9]") == 0);
            jit_assert(strcmp(target.str(), "[1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0]") == 0);
        }
    }
}

TEST_BOTH(03_side_effect_2) {
    // Tests that side effects work that don't reference loop variables
    for (uint32_t i = 0; i < 3; ++i) {
        jit_set_flag(JitFlag::LoopRecord, i != 0);
        jit_set_flag(JitFlag::LoopOptimize, i == 2);

        for (uint32_t j = 0; j < 3; ++j) {
            UInt32 x = arange<UInt32>(10);
            Float y = zero<Float>(1);
            UInt32 target = zero<UInt32>(11);

            Loop<Mask> loop("MyLoop", x, y);
            while (loop(x < 5)) {
                scatter_reduce(ReduceOp::Add, target, UInt32(2), UInt32(2));
                y += Float(x);
                x += 1;
            }

            if (j == 0) {
                jit_var_schedule(x.index());
                jit_var_schedule(y.index());
            }

            jit_assert(strcmp(y.str(), "[10, 10, 9, 7, 4, 0, 0, 0, 0, 0]") == 0);
            jit_assert(strcmp(x.str(), "[5, 5, 5, 5, 5, 5, 6, 7, 8, 9]") == 0);
            jit_assert(strcmp(target.str(), "[0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0]") == 0);
        }
    }
}

TEST_BOTH(04_side_effect_masking) {
    // Tests that side effects work that don't reference loop variables
    for (uint32_t i = 0; i < 3; ++i) {
        jit_set_flag(JitFlag::LoopRecord, i != 0);
        jit_set_flag(JitFlag::LoopOptimize, i == 2);

        for (uint32_t j = 0; j < 3; ++j) {
            UInt32 x = arange<UInt32>(1000000);
            UInt32 target = zero<UInt32>(10);

            Loop<Mask> loop("MyLoop", x);
            while (loop(x < 10)) {
                // This is sure to segfault if not masked correctly
                scatter_reduce(ReduceOp::Add, target, UInt32(1), x);
                x += 1;
            }

            jit_assert(strcmp(target.str(), "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]") == 0);
        }
    }
}

TEST_BOTH(05_optimize_invariant) {
    /* Test to check that variables which stay unchanged or constant and
       equal-valued are optimized out of the loop */
    for (uint32_t i = 0; i < 3; ++i) {
        jit_set_flag(JitFlag::LoopRecord, i != 0);
        jit_set_flag(JitFlag::LoopOptimize, i == 2);

        UInt32 j = 0,
               v1 = 123,
               v1_orig = v1,
               v2 = opaque<UInt32>(123),
               v2_orig = v2,
               v3 = 124,
               v3_orig = v3,
               v4 = 125,
               v4_orig = v4,
               v5 = 1,
               v6 = 0;

        Loop<Mask> loop("MyLoop", j, v1, v2, v3, v4, v5, v6);
        int count = 0;
        while (loop(j < 10)) {
            j += 1;

            (void) v1; // v1 stays unchanged
            (void) v2; // v2 stays unchanged
            v3 = 124;  // v3 is overwritten with same value
            v4 = 100;  // v4 is overwritten with different value
            (void) v2; // v5 stays unchanged
            v6 += v5;  // v6 is modified by a loop-invariant variable
            ++count;
        }

        if (i == 0)
            jit_assert(count == 10);
        else if (i == 1)
            jit_assert(count == 1);
        else if (i == 2)
            jit_assert(count == 2);

        if (i == 2) {
            jit_assert( jit_var_is_literal(v1.index()) && v1.index() == v1_orig.index());
            jit_assert(!jit_var_is_literal(v2.index()) && v2.index() == v2_orig.index());
            jit_assert( jit_var_is_literal(v3.index()) && v3.index() == v3_orig.index());
            jit_assert(!jit_var_is_literal(v4.index()) && v4.index() != v4_orig.index());
            jit_assert( jit_var_is_literal(v5.index()));
            jit_assert(!jit_var_is_literal(v6.index()));
        }

        jit_var_schedule(v1.index());
        jit_var_schedule(v2.index());
        jit_var_schedule(v3.index());
        jit_var_schedule(v4.index());
        jit_var_schedule(v5.index());
        jit_var_schedule(v6.index());

        jit_assert(v1 == 123 && v2 == 123 && v3 == 124 && v4 == 100 && v5 == 1 && v6 == 10);
    }
}

TEST_BOTH(06_garbage_collection) {
    // Checks that unused loop variables are optimized away

    jit_set_flag(JitFlag::LoopRecord, 1);
    for (uint32_t i = 0; i < 2; ++i) {
        jit_set_flag(JitFlag::LoopOptimize, i == 1);

        UInt32 j = 0;
        UInt32 v1 = opaque<UInt32>(1);
        UInt32 v2 = opaque<UInt32>(2);
        UInt32 v3 = opaque<UInt32>(3);
        UInt32 v4 = opaque<UInt32>(4);
        uint32_t v1i = v1.index(), v2i = v2.index(),
                 v3i = v3.index(), v4i = v4.index();
        jit_var_set_label(v1i, "v1");
        jit_var_set_label(v2i, "v2");
        jit_var_set_label(v3i, "v3");
        jit_var_set_label(v4i, "v4");

        Loop<Mask> loop("MyLoop", j, v1, v2, v3, v4);
        while (loop(j < 4)) {
            UInt32 tmp = v4;
            v4 = v1;
            v1 = v2;
            v2 = v3;
            v3 = tmp;
            j += 1;
        }

        v1 = UInt32();
        v2 = UInt32();
        v3 = UInt32();
        jit_assert(jit_var_exists(v1i) && jit_var_exists(v2i) && jit_var_exists(v3i) && jit_var_exists(v4i));

        v4 = UInt32();
        if (i == 0)
            jit_assert(jit_var_exists(v1i) && jit_var_exists(v2i) && jit_var_exists(v3i) && jit_var_exists(v4i));
        else
            jit_assert(!jit_var_exists(v1i) && !jit_var_exists(v2i) && !jit_var_exists(v3i) && !jit_var_exists(v4i));

        j = UInt32();
        jit_assert(!jit_var_exists(v1i) && !jit_var_exists(v2i) && !jit_var_exists(v3i) && !jit_var_exists(v4i));
    }
}

TEST_BOTH(07_collatz) {
    // A more interesting nested loop
    auto collatz = [](const char *name, UInt32 value) -> UInt32 {
        UInt32 counter = 0;
        jit_var_set_label(value.index(), "value");
        jit_var_set_label(counter.index(), "counter");

        Loop<Mask> loop(name, value, counter);
        while (loop(neq(value, 1))) {
            Mask is_even = eq(value & UInt32(1), 0);
            value = select(is_even, value / 2, value*3 + 1);
            counter += 1;
        }

        return counter;
    };

    for (uint32_t i = 0; i < 3; ++i) {
        jit_set_flag(JitFlag::LoopRecord, i != 0);
        jit_set_flag(JitFlag::LoopOptimize, i == 2);
        for (uint32_t j = 0; j < 2; ++j) {
            UInt32 buf = full<UInt32>(1000, 11);
            if (j == 1) {
                UInt32 k = 1;
                Loop<Mask> loop_1("Outer", k);
                while (loop_1(k <= 10)) {
                    scatter(buf, collatz("Inner", k), k - 1);
                    k += 1;
                }
            } else {
                for (uint32_t k = 1; k <= 10; ++k) {
                    char tmpname[20];
                    snprintf(tmpname, sizeof(tmpname), "Inner [%u]", k);
                    scatter(buf, collatz(tmpname, k), UInt32(k - 1));
                }
            }
            jit_assert(strcmp(buf.str(), "[0, 1, 7, 2, 5, 8, 16, 3, 19, 6, 1000]") == 0);
        }
    }
}

TEST_BOTH(08_nested_write) {
    // Nested loop where both loops write to the same loop variable
    for (uint32_t i = 0; i < 3; ++i) {
        jit_set_flag(JitFlag::LoopRecord, i != 0);
        jit_set_flag(JitFlag::LoopOptimize, i == 2);

        UInt32 k = arange<UInt32>(10)*12;
        Loop<Mask> loop_1("Outer", k);

        while (loop_1(neq(k % 7, 0))) {
            Loop<Mask> loop_2("Inner", k);
            while (loop_2(neq(k % 3, 0))) {
                k += 1;
            }
            k += 1;
        }
        jit_assert(strcmp(k.str(), "[0, 28, 28, 49, 49, 70, 91, 84, 112, 112]") == 0);
    }
}
