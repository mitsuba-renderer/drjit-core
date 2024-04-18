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

} // namespace drjit

struct Ref {
    friend Ref steal(uint32_t);
    friend Ref borrow(uint32_t);

    Ref() : index(0) { }
    ~Ref() { jit_var_dec_ref(index); }

    Ref(Ref &&r) : index(r.index) { r.index = 0; }

    Ref &operator=(Ref &&r) {
        jit_var_dec_ref(index);
        index = r.index;
        r.index = 0;
        return *this;
    }

    Ref(const Ref &) = delete;
    Ref &operator=(const Ref &) = delete;

    operator uint32_t() const { return index; }
    uint32_t release() {
        uint32_t value = index;
        index = 0;
        return value;
    }
    void reset() {
        jit_var_dec_ref(index);
        index = 0;
    }

private:
    uint32_t index = 0;
};

inline Ref steal(uint32_t index) {
    Ref r;
    r.index = index;
    return r;
}

inline Ref borrow(uint32_t index) {
    Ref r;
    r.index = index;
    jit_var_inc_ref(index);
    return r;
}

TEST_BOTH(01_symbolic_loop) {
    // Tests a simple loop
    for (uint32_t i = 0; i < 2; ++i) {
        jit_set_flag(JitFlag::OptimizeLoops, i == 1);

        for (uint32_t j = 0; j < 2; ++j) {
            UInt32 x    = arange<UInt32>(10);
            Float y     = zeros<Float>(1);
            Float z     = 1;
            {
                dr::scoped_record record_guard(Backend);
                uint32_t var_indices[3] = { x.index(), y.index(), z.index() };

                Ref loop = steal(jit_var_loop_start("MyLoop",
                    false, /* symbolic */
                    3, var_indices));

                UInt32 xl   = UInt32::steal(var_indices[0]);
                Float yl    = Float::steal(var_indices[1]);
                Float zl    = Float::steal(var_indices[2]);

                do {
                    Mask active_i = xl < 5;

                    Mask active = Mask::steal(jit_var_mask_apply(
                        active_i.index(), (uint32_t) jit_var_size(active_i.index())));

                    Ref loop_cond = steal(jit_var_loop_cond(loop, active.index()));

                    yl += Float(xl);
                    xl += 1;
                    zl += 1;

                    var_indices[0] = xl.index();
                    var_indices[1] = yl.index();
                    var_indices[2] = zl.index();

                    int rv = jit_var_loop_end(loop, loop_cond, var_indices,
                        record_guard.checkpoint);

                    // All done
                    if (rv) {
                        x = UInt32::steal(var_indices[0]);
                        y = Float::steal(var_indices[1]);
                        z = Float::steal(var_indices[2]);
                        record_guard.disarm();
                        break;
                    }

                } while (true);
            }

            if (j == 0) {
                jit_var_schedule(z.index());
                jit_var_schedule(y.index());
                jit_var_schedule(x.index());
            }

            jit_assert(strcmp(x.str(), "[5, 5, 5, 5, 5, 5, 6, 7, 8, 9]") == 0);
            jit_assert(strcmp(y.str(), "[10, 10, 9, 7, 4, 0, 0, 0, 0, 0]") == 0);
            jit_assert(strcmp(z.str(), "[6, 5, 4, 3, 2, 1, 1, 1, 1, 1]") == 0);
        }
    }
}


TEST_BOTH_FP32(02_side_effect) {
    // Tests that side effects happen (and only once, even if the loop is re-evaluated)
    for (uint32_t i = 0; i < 2; ++i) {
        jit_set_flag(JitFlag::OptimizeLoops, i == 1);

        for (uint32_t j = 0; j < 2; ++j) {
            UInt32 x = arange<UInt32>(10);
            Float y = zeros<Float>(1);
            UInt32 target = zeros<UInt32>(11);
            {
                uint32_t var_indices[2] = { x.index(), y.index() };
                dr::scoped_record record_guard(Backend);

                Ref loop = steal(jit_var_loop_start("MyLoop", 
                    false, /* symbolic */
                    2, var_indices));

                UInt32 xl = UInt32::steal(var_indices[0]);
                Float yl = Float::steal(var_indices[1]);

                do {
                    Mask active_i = xl < 5;

                    Mask active = Mask::steal(jit_var_mask_apply(
                        active_i.index(), (uint32_t) jit_var_size(active_i.index())));

                    Ref loop_cond = steal(jit_var_loop_cond(loop, active.index()));

                    scatter_reduce(ReduceOp::Add, target, UInt32(1), xl, active);
                    yl += Float(xl);
                    xl += 1;

                    var_indices[0] = xl.index();
                    var_indices[1] = yl.index();

                    int rv = jit_var_loop_end(loop, loop_cond, var_indices,
                        record_guard.checkpoint);

                    // All done
                    if (rv) {
                        x = UInt32::steal(var_indices[0]);
                        y = Float::steal(var_indices[1]);
                        record_guard.disarm();
                        break;
                    }

                } while (true);
            }

            if (j == 0) {
                jit_var_schedule(x.index());
                jit_var_schedule(y.index());
            }

            jit_assert(strcmp(x.str(), "[5, 5, 5, 5, 5, 5, 6, 7, 8, 9]") == 0);
            jit_assert(strcmp(y.str(), "[10, 10, 9, 7, 4, 0, 0, 0, 0, 0]") == 0);
            jit_assert(strcmp(target.str(), "[1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0]") == 0);
        }
    }
}


TEST_BOTH_FP32(03_side_effect_2) {
    // Tests that side effects work even if they don't reference any loop variables
    for (uint32_t i = 0; i < 2; ++i) {
        jit_set_flag(JitFlag::OptimizeLoops, i == 1);

        for (uint32_t j = 0; j < 3; ++j) {
            UInt32 x = arange<UInt32>(10);
            Float y = zeros<Float>(1);
            UInt32 target = zeros<UInt32>(11);
            {
                uint32_t var_indices[2] = { x.index(), y.index() };
                dr::scoped_record record_guard(Backend);

                Ref loop = steal(jit_var_loop_start("MyLoop", 
                    false, /* symbolic */
                    2, var_indices));

                UInt32 xl = UInt32::steal(var_indices[0]);
                Float yl = Float::steal(var_indices[1]);

                do {
                    Mask active_i = xl < 5;

                    Mask active = Mask::steal(jit_var_mask_apply(
                        active_i.index(), (uint32_t) jit_var_size(active_i.index())));

                    Ref loop_cond = steal(jit_var_loop_cond(loop, active.index()));

                    scatter_reduce(ReduceOp::Add, target, UInt32(2), UInt32(2), active);
                    yl += Float(xl);
                    xl += 1;

                    var_indices[0] = xl.index();
                    var_indices[1] = yl.index();

                    int rv = jit_var_loop_end(loop, loop_cond, var_indices,
                        record_guard.checkpoint);

                    // All done
                    if (rv) {
                        x = UInt32::steal(var_indices[0]);
                        y = Float::steal(var_indices[1]);
                        record_guard.disarm();
                        break;
                    }

                } while (true);
            }

            if (j == 0) {
                jit_var_schedule(x.index());
                jit_var_schedule(y.index());
            }

            jit_assert(strcmp(x.str(), "[5, 5, 5, 5, 5, 5, 6, 7, 8, 9]") == 0);
            jit_assert(strcmp(y.str(), "[10, 10, 9, 7, 4, 0, 0, 0, 0, 0]") == 0);
            jit_assert(strcmp(target.str(), "[0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0]") == 0);
        }
    }
}


TEST_BOTH(04_optimize_invariant) {
    /* Test to check that variables which stay unchanged or constant and
       equal-valued are optimized out of the loop */
    for (uint32_t i = 0; i < 2; ++i) {
        jit_set_flag(JitFlag::OptimizeLoops, i == 1);

        UInt32 j = 0,
               v1 = 123,
               v1_orig = v1,
               v2 = opaque<UInt32>(123),
               v2_orig = v2,
               v3 = 124, v3_orig = v3,
               v4 = 125, v4_orig = v4,
               v5 = 1,   v5_orig = v5,
               v6 = 0,   v6_orig = v6;

        int count = 0;
        {
            uint32_t var_indices[7] = { 
                j.index(), v1.index(), v2.index(), v3.index(),
                v4.index(), v5.index(), v6.index()
            };
            dr::scoped_record record_guard(Backend);

            Ref loop = steal(jit_var_loop_start("MyLoop", 
                false, /* symbolic */
                7, var_indices));

            UInt32  jl = UInt32::steal(var_indices[0]),
                    v1l = UInt32::steal(var_indices[1]),
                    v2l = UInt32::steal(var_indices[2]),
                    v3l = UInt32::steal(var_indices[3]),
                    v4l = UInt32::steal(var_indices[4]),
                    v5l = UInt32::steal(var_indices[5]),
                    v6l = UInt32::steal(var_indices[6]);
            do {
                Mask active_i = jl < 10;

                Mask active = Mask::steal(jit_var_mask_apply(
                    active_i.index(), (uint32_t) jit_var_size(active_i.index())));

                Ref loop_cond = steal(jit_var_loop_cond(loop, active.index()));

                jl += 1;
                (void) v1l; // v1 stays unchanged
                (void) v2l; // v2 stays unchanged
                v3l = 124;  // v3 is overwritten with same value
                v4l = 100;  // v4 is overwritten with different value
                (void) v5l; // v5 stays unchanged
                v6l += v5l; // v6 is modified by a loop-invariant variable
                ++count;

                var_indices[0] = jl.index();
                var_indices[1] = v1l.index();
                var_indices[2] = v2l.index();
                var_indices[3] = v3l.index();
                var_indices[4] = v4l.index();
                var_indices[5] = v5l.index();
                var_indices[6] = v6l.index();
                int rv = jit_var_loop_end(loop, loop_cond, var_indices, 
                    record_guard.checkpoint);

                // Loop-invariants may have been eliminated
                if (!rv) {
                    jl  = UInt32::steal(var_indices[0]);
                    v1l = UInt32::steal(var_indices[1]);
                    v2l = UInt32::steal(var_indices[2]);
                    v3l = UInt32::steal(var_indices[3]);
                    v4l = UInt32::steal(var_indices[4]);
                    v5l = UInt32::steal(var_indices[5]);
                    v6l = UInt32::steal(var_indices[6]);
                }
                // All done
                else {
                    j  = UInt32::steal(var_indices[0]);
                    v1 = UInt32::steal(var_indices[1]);
                    v2 = UInt32::steal(var_indices[2]);
                    v3 = UInt32::steal(var_indices[3]);
                    v4 = UInt32::steal(var_indices[4]);
                    v5 = UInt32::steal(var_indices[5]);
                    v6 = UInt32::steal(var_indices[6]);
                    record_guard.disarm();
                    break;
                }

            } while(true);
        }

        if (i == 0)
            jit_assert(count == 1);
        else if (i == 1)
            jit_assert(count == 2);

        #define IS_LITERAL(x) (jit_var_state(x.index()) == VarState::Literal)
        if (i == 1) {
            jit_assert( IS_LITERAL(v1) && v1.index() == v1_orig.index());
            jit_assert(!IS_LITERAL(v2) && v2.index() == v2_orig.index());
            jit_assert( IS_LITERAL(v3) && v3.index() == v3_orig.index());
            jit_assert(!IS_LITERAL(v4) && v4.index() != v4_orig.index());
            jit_assert( IS_LITERAL(v5) && v5.index() == v5_orig.index());
            jit_assert(!IS_LITERAL(v6) && v6.index() != v6_orig.index());
        }
        #undef IS_LITERAL

        jit_var_schedule(v1.index());
        jit_var_schedule(v2.index());
        jit_var_schedule(v3.index());
        jit_var_schedule(v4.index());
        jit_var_schedule(v5.index());
        jit_var_schedule(v6.index());
    }
}


TEST_BOTH(05_collatz) {
    // A more interesting nested loop
    auto collatz = [](const char *name, UInt32 value, bool symbolic) -> UInt32 {
        UInt32 counter = 0;
        {
            uint32_t var_indices[2] = { 
                value.index(), counter.index()
            };

            dr::scoped_record record_guard(Backend);

            Ref loop = steal(jit_var_loop_start(name, symbolic, 2, var_indices));

            UInt32 valuel   = UInt32::steal(var_indices[0]),
                   counterl = UInt32::steal(var_indices[1]);
            do {
                Mask active_i = neq(valuel, 1);

                Mask active = Mask::steal(jit_var_mask_apply(
                    active_i.index(), (uint32_t) jit_var_size(active_i.index())));

                Ref loop_cond = steal(jit_var_loop_cond(loop, active.index()));

                Mask is_even = eq(valuel & UInt32(1), 0);
                valuel = select(is_even, valuel / 2, valuel*3 + 1);
                counterl += 1;

                var_indices[0] = valuel.index();
                var_indices[1] = counterl.index();

                int rv = jit_var_loop_end(loop, loop_cond, var_indices, 
                    record_guard.checkpoint);

                // All done
                if (rv) {
                    value = UInt32::steal(var_indices[0]);
                    counter = UInt32::steal(var_indices[1]);
                    record_guard.disarm();
                    break;
                }
            } while(true);
        }

        return counter;
    };

    for (uint32_t i = 0; i < 2; ++i) {
        jit_set_flag(JitFlag::LoopOptimize, i == 1);
        for (uint32_t j = 0; j < 2; ++j) {
            UInt32 buf = full<UInt32>(1000, 11);
            if (j == 0) {
                UInt32 k = 1;
                {
                    uint32_t var_indices[1] = { k.index() };

                    dr::scoped_record record_guard(Backend);

                    Ref loop = steal(jit_var_loop_start("Outer", 
                        false, /* symbolic */
                        1, var_indices));

                    UInt32 kl = UInt32::steal(var_indices[0]);

                    do {
                        Mask active_i = kl <= 10;

                        Mask active = Mask::steal(jit_var_mask_apply(
                            active_i.index(), (uint32_t) jit_var_size(active_i.index())));

                        Ref loop_cond = steal(jit_var_loop_cond(loop, active.index()));

                        scatter(buf, collatz("Inner", kl, true), kl - 1, active);
                        kl += 1;
                        var_indices[0] = kl.index();

                        int rv = jit_var_loop_end(loop, loop_cond, var_indices, 
                            record_guard.checkpoint);

                        // All done
                        if (rv) {
                            k = UInt32::steal(var_indices[0]);
                            record_guard.disarm();
                            break;
                        }
                    } while(true);
                }

            } else {
                for (uint32_t k = 1; k <= 10; ++k) {
                    char tmpname[20];
                    snprintf(tmpname, sizeof(tmpname), "Inner [%u]", k);
                    scatter(buf, collatz(tmpname, k, false), UInt32(k - 1));
                }
            }
            jit_assert(strcmp(buf.str(), "[0, 1, 7, 2, 5, 8, 16, 3, 19, 6, 1000]") == 0);
        }
    }
}


TEST_BOTH(06_nested_write) {
    // Nested loop where both loops write to the same loop variable
    auto inner_loop = [](UInt32 k) -> UInt32 {
        uint32_t var_indices[1] = { k.index() };

        dr::scoped_record record_guard(Backend);
        Ref loop = steal(jit_var_loop_start("Inner", 
            true, /* symbolic */
            1, var_indices));

        UInt32 kl = UInt32::steal(var_indices[0]);

        do {
            Mask active_i = neq(kl % 3, 0);

            Mask active = Mask::steal(jit_var_mask_apply(
                active_i.index(), (uint32_t) jit_var_size(active_i.index())));

            Ref loop_cond = steal(jit_var_loop_cond(loop, active.index()));

            kl += 1;

            var_indices[0] = kl.index();
            int rv = jit_var_loop_end(loop, loop_cond, var_indices,
                record_guard.checkpoint);

            // All done
            if (rv) {
                k = UInt32::steal(var_indices[0]);
                record_guard.disarm();
                break;
            }
        } while (true);

        return k;
    };


    for (uint32_t i = 0; i < 2; ++i) {
        jit_set_flag(JitFlag::LoopOptimize, i == 1);

        UInt32 k = arange<UInt32>(10)*12;
        {
            uint32_t var_indices[1] = { k.index() };

            dr::scoped_record record_guard(Backend);

            Ref loop = steal(jit_var_loop_start("Outer", 
                false, /* symbolic */
                1, var_indices));

            UInt32 kl = UInt32::steal(var_indices[0]);

            do {
                Mask active_i = neq(kl % 7, 0);

                Mask active = Mask::steal(jit_var_mask_apply(
                    active_i.index(), (uint32_t) jit_var_size(active_i.index())));

                Ref loop_cond = steal(jit_var_loop_cond(loop, active.index()));

                kl = inner_loop(kl);
                kl += 1;

                var_indices[0] = kl.index();

                int rv = jit_var_loop_end(loop, loop_cond, var_indices,
                    record_guard.checkpoint);

                // All done
                if (rv) {
                    k = UInt32::steal(var_indices[0]);
                    record_guard.disarm();
                    break;
                }

            } while (true);
        }

        jit_assert(strcmp(k.str(), "[0, 28, 28, 49, 49, 70, 91, 84, 112, 112]") == 0);
    }
}



TEST_BOTH(07_optim_cond) {
    // Loop condition depends on variables that are optimized away (loop-invariants)

    for (uint32_t i = 0; i < 2; ++i) {
        jit_set_flag(JitFlag::LoopOptimize, i == 1);

        UInt32 k = arange<UInt32>(3), l = 10;
        {
            uint32_t var_indices[2] = { k.index(), l.index() };

            dr::scoped_record record_guard(Backend);

            bool nested_symbolic = false;
            Ref loop = steal(jit_var_loop_start("Outer", nested_symbolic, 2, var_indices));

            UInt32 kl = UInt32::steal(var_indices[0]),
                   ll = UInt32::steal(var_indices[1]);

            do {
                Mask active_i = kl + ll < 30;

                Mask active = Mask::steal(jit_var_mask_apply(
                    active_i.index(), (uint32_t) jit_var_size(active_i.index())));

                Ref loop_cond = steal(jit_var_loop_cond(loop, active.index()));

                kl += 1;
                var_indices[0] = kl.index();
                var_indices[1] = ll.index();

                int rv = jit_var_loop_end(loop, loop_cond, var_indices,
                    record_guard.checkpoint);

                // Loop-invariants may have been eliminated
                if (!rv) {
                    kl = UInt32::steal(var_indices[0]);
                    ll = UInt32::steal(var_indices[1]);
                }
                // All done
                else {
                    k = UInt32::steal(var_indices[0]);
                    l = UInt32::steal(var_indices[1]);
                    record_guard.disarm();
                    break;
                }

            } while (true);
        }

        const char  *k_ref = "[20, 20, 20]";

        auto compare_var = [](const char* a, const char* ref) {
            bool match = strcmp(a, ref) == 0;
            if (!match)
                fprintf(stderr, "Mismatch: %s, %s\n", a, ref);
            return match;
        };

        jit_assert(compare_var(k.str(), k_ref));
        if (i == 1)
            jit_assert((VarState)jit_var_state(l.index()) == VarState::Literal);
    }
}


TEST_BOTH(08_eval_side_effect_in_loop) {
    // Loop depends needs to access a dirty variable
    for (uint32_t i = 0; i < 2; ++i) {
        jit_set_flag(JitFlag::LoopOptimize, i == 0);

        UInt32 j = arange<UInt32>(3); // [0, 1, 2]

        UInt32 k = full<UInt32>(3, 3);
        scatter(k, full<UInt32>(4, 2), arange<UInt32>(2)); // [4, 4, 3]

        // k is a dirty variable
        bool first_iteration = true;
        {
            dr::scoped_record record_guard(Backend);
            uint32_t var_indices[1] = { j.index() };
            Ref loop = steal(jit_var_loop_start("Outer", false, 1, var_indices));

            UInt32 jl = UInt32::steal(var_indices[0]);

            do {
                Mask active_i = jl < 10;

                Mask active = Mask::steal(jit_var_mask_apply(
                    active_i.index(), (uint32_t) jit_var_size(active_i.index())));

                Ref loop_cond = steal(jit_var_loop_cond(loop, active.index()));

                if (first_iteration) {
                    jit_assert(jit_var_state(k.index()) != VarState::Evaluated);
                    first_iteration = false;
                }

                jl += k; // Needs to evaluate k
                jit_assert(jit_var_state(k.index()) == VarState::Evaluated);

                var_indices[0] = jl.index();

                int rv = jit_var_loop_end(loop, loop_cond, var_indices,
                    record_guard.checkpoint);

                // All done
                if (rv) {
                    j = UInt32::steal(var_indices[0]);
                    record_guard.disarm();
                    break;
                }

            } while (true);
        }

        jit_assert(strcmp(j.str(), "[12, 13, 11]") == 0);
    }
}