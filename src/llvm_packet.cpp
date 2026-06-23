/*
    src/llvm_packet.cpp -- Specialized memory operations that read or write
    multiple adjacent values at once.

    Copyright (c) 2024 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "eval.h"
#include "llvm.h"
#include "var.h"
#include "llvm_eval.h"
#include "llvm_packet.h"
#include "call.h"
#include "log.h"
#include "op.h"
#include <algorithm>

void perm4(const Variable *v, const char *out, const char *arg0, const char *arg1, uint32_t pattern) {
    uint32_t width = jitc_llvm_vector_width;
    fmt("    $s = shufflevector $M $s, $M $s, <$w x i32> <", out, v, arg0, v, arg1);
    for (uint32_t i = 0; i < width; ++i) {
        uint32_t pat_i = pattern >> ((i & 0b11u) * 3u);
        uint32_t pos = ((i & ~0b11u) | (pat_i & 0b11u)) + ((pat_i >> 2) & 1) * width;
        fmt("i32 $u$s", pos, i + 1 < width ? ", " : ">\n");
    }
}

void perm8(const Variable *v, const char *out, const char *arg0, const char *arg1, uint32_t pattern) {
    uint32_t width = jitc_llvm_vector_width;
    fmt("    $s = shufflevector $M $s, $M $s, <$w x i32> <", out, v, arg0, v, arg1);
    for (uint32_t i = 0; i < width; ++i) {
        uint32_t pat_i = pattern >> ((i & 0b111u) * 4u);
        uint32_t pos = ((i & ~0b111u) | (pat_i & 0b111u)) + ((pat_i >> 3) & 1) * width;
        fmt("i32 $u$s", pos, i + 1 < width ? ", " : ">\n");
    }
}

static void jitc_llvm_permute_n(const Variable *v, uint32_t n) {
    const uint32_t unpacklo = 0b101'001'100'000u,
                   unpackhi = 0b111'011'110'010u,
                   movelh   = 0b101'100'001'000u,
                   movehl   = 0b011'010'111'110u;

    if (n == 2) {
        const uint32_t c0 = 0b110'010'100'000u,
                       c1 = 0b111'011'101'001u;
        perm4(v, "%r0", "%v0", "%v1", c0);
        perm4(v, "%r1", "%v0", "%v1", c1);
    } else if (n == 4) {
        perm4(v, "%t0", "%v0", "%v1", unpacklo);
        perm4(v, "%t1", "%v2", "%v3", unpacklo);
        perm4(v, "%t2", "%v0", "%v1", unpackhi);
        perm4(v, "%t3", "%v2", "%v3", unpackhi);
        perm4(v, "%r0", "%t0", "%t1", movelh);
        perm4(v, "%r1", "%t1", "%t0", movehl);
        perm4(v, "%r2", "%t2", "%t3", movelh);
        perm4(v, "%r3", "%t3", "%t2", movehl);
    } else if (n == 8) {
        const uint32_t c0 = 0b1011'1010'1001'1000'0011'0010'0001'0000u,
                       c1 = 0b1111'1110'1101'1100'0111'0110'0101'0100u;

        perm4(v, "%t0", "%v0", "%v1", unpacklo);
        perm4(v, "%t1", "%v0", "%v1", unpackhi);
        perm4(v, "%t2", "%v2", "%v3", unpacklo);
        perm4(v, "%t3", "%v2", "%v3", unpackhi);
        perm4(v, "%t4", "%v4", "%v5", unpacklo);
        perm4(v, "%t5", "%v4", "%v5", unpackhi);
        perm4(v, "%t6", "%v6", "%v7", unpacklo);
        perm4(v, "%t7", "%v6", "%v7", unpackhi);

        perm4(v, "%t8", "%t0", "%t2", movelh);
        perm4(v, "%t9", "%t2", "%t0", movehl);
        perm4(v, "%ta", "%t1", "%t3", movelh);
        perm4(v, "%tb", "%t3", "%t1", movehl);
        perm4(v, "%tc", "%t4", "%t6", movelh);
        perm4(v, "%td", "%t6", "%t4", movehl);
        perm4(v, "%te", "%t5", "%t7", movelh);
        perm4(v, "%tf", "%t7", "%t5", movehl);

        perm8(v, "%r0", "%t8", "%tc", c0);
        perm8(v, "%r1", "%t9", "%td", c0);
        perm8(v, "%r2", "%ta", "%te", c0);
        perm8(v, "%r3", "%tb", "%tf", c0);
        perm8(v, "%r4", "%t8", "%tc", c1);
        perm8(v, "%r5", "%t9", "%td", c1);
        perm8(v, "%r6", "%ta", "%te", c1);
        perm8(v, "%r7", "%tb", "%tf", c1);
    } else {
        jitc_raise("jitc_llvm_permute_n(): permutation is too big!");
    }
}

void gather_packet_recursive(uint32_t l, uint32_t i, uint32_t n, const Variable *v) {
    uint32_t s = jitc_llvm_vector_width >> l;

    if (s == n) {
        for (uint32_t j = 0; j < n; ++j) {
            uint32_t k = i*n+j,
                     align = type_size[v->type]*n;

            fmt( "    %p$u = extractelement <$w x ptr> %p, i32 $u\n"
                 "    %v$u_$u_$u = load <$u x $m>, ptr %p$u, align $u, !alias.scope !2, !noalias !2\n",
                k, k,
                l, i, j, n, v, k, align);
        }
    } else {
        uint32_t hs = s / 2;

        gather_packet_recursive(l + 1, 2*i, n, v);
        gather_packet_recursive(l + 1, 2*i+1, n, v);

        for (uint32_t j = 0; j < n; ++j) {
            fmt("    %v$u_$u_$u = shufflevector <$u x $m> %v$u_$u_$u, <$u x $m> %v$u_$u_$u, <$u x i32> <",
                l, i, j, hs, v, l+1, 2*i, j, hs, v, l+1, 2*i+1, j, s);
            for (uint32_t k = 0; k < s; ++k)
                fmt("i32 $u$s", k, k + 1 < s ? ", " : ">\n");
        }
    }
}

/// Emit (and register) the internal helper ``@gather_<n>x<H>`` that reads ``n``
/// adjacent elements of type ``v`` from each lane's base address and transposes
/// the result into ``n`` SoA vectors. The per-lane loads are ``type_size * n``
/// aligned, which both callers guarantee.
void jitc_llvm_gather_packet_define(const Variable *v, uint32_t n) {
    size_t offset = buffer.size();

    fmt_intrinsic("@zero_$m_$u = private constant [$u x $m] $z, align $u", v, n, n, v,
                  type_size[v->type] * jitc_llvm_vector_width);

    fmt("define internal fastcc [$u x <$w x $m>] @gather_$ux$H(<$w x ptr> %p) local_unnamed_addr #0 {\n",
        n, v, n, v);

    gather_packet_recursive(0, 0, n, v);

    for (uint32_t i = 0; i < n; ++i)
        fmt("    %v$u = bitcast $M %v0_0_$u to $M\n", i, v, i, v);

    jitc_llvm_permute_n(v, n);

    fmt("    %q0 = insertvalue [$u x <$w x $m>] undef, <$w x $m> %r0, 0\n", n, v, v);
    for (uint32_t i = 1; i < n; ++i)
        fmt("    %q$u = insertvalue [$u x <$w x $m>] %q$u, <$w x $m> %r$u, $u\n",
            i, n, v, i - 1, v, i, i);

    fmt("    ret [$u x <$w x $m>] %q$u\n"
        "}",
        n, v, n - 1);

    jitc_register_global(buffer.get() + offset);
    buffer.rewind_to(offset);
}

void jitc_llvm_render_gather_packet(const Variable *v, const Variable *ptr,
                                    const Variable *index, const Variable *mask) {
    uint32_t n = (uint32_t) v->literal;

    jitc_llvm_gather_packet_define(v, n);

    const char *ext = "";
    if (v->type == (uint32_t) VarType::Bool)
        ext = "_e";

    fmt("    $v_2 = getelementptr [$u x $m], $<ptr$> $v, $T $v\n"
        "    $v_3 = getelementptr [$u x $m], ptr @zero_$m_$u, $T $z\n"
        "    $v_4 = select $V, <$w x ptr> $v_2, <$w x ptr> $v_3\n"
        "    $v_5 = call fastcc [$u x <$w x $m>] @gather_$ux$H(<$w x ptr> $v_4)\n",
        v, n, v, ptr, index, index,
        v, n, v, v, n, index,
        v, mask, v, v,
        v, n, v, n, v, v
    );

    for (uint32_t i = 0; i < n; ++i)
        fmt("    $v_out_$u$s = extractvalue [$u x <$w x $m>] $v_5, $u\n",
            v, i, ext, n, v, v, i);

    if (v->type == (uint32_t) VarType::Bool) {
        for (uint32_t i = 0; i < n; ++i)
            fmt("    $v_out_$u = trunc <$w x $m> $v_out_$u_e to <$w x i1>\n",
                v, i, v, v, i);
    }
}

void jitc_llvm_render_call_data(const CallData *call, uint32_t inst,
                                uint32_t packetized_until[4]) {
    const CallData::InstanceLayout &layout = call->instance_layout[inst];
    uint32_t width = jitc_llvm_vector_width,
             uid   = 0;

    for (uint32_t c = 0; c < 4; ++c)
        packetized_until[c] = layout.bucket[c].slot_start;

    // A field is loadable iff scheduled (reg_index != 0): only scheduled vars
    // have an SSA name to bind. This function only iterates coalesceable buckets,
    // whose membership was classified when slots were captured.
    //
    // A coalesceable slot can be unscheduled when it is captured solely by a call
    // output (or side effect) that the optimizer later eliminated -- the
    // trace-time layout still reserved its offset, but no codegen references it.
    // Such "holes" are loaded across (their bytes are contiguous and uploaded)
    // but bound nowhere; they are dead.
    auto loadable = [](const Variable *v) {
        return v && v->reg_index != 0;
    };

    // The layout pass already grouped coalesceable slots into four homogeneous
    // size-class buckets. LLVM packet gathers require such homogeneous runs, so
    // each bucket can be streamed directly rather than rediscovered here.
    //
    // A bucket is coalesced on "coalesceable", not "loadable": it may contain
    // holes -- slots that the optimizer left unscheduled (reg_index 0) and that
    // are therefore not bound here. Their bytes are contiguous and uploaded like
    // any other slot, so the packet load reads across them harmlessly and we skip
    // binding those lanes. Trailing holes are trimmed so the scalar path can pick
    // up an under-utilized live tail.
    for (uint32_t c = 0; c < 4; ++c) {
        const CallData::InstanceLayout::Bucket &bucket = layout.bucket[c];
        uint32_t k = bucket.slot_start,
                 kend = bucket.slot_end();

        if (k == kend)
            continue;

        uint32_t s = CallData::SizeBucket::size_from_id(c),
                 rel0 = call->slots[k].offset - layout.data_offset,
                 last_load = (uint32_t) -1;

        for (uint32_t slot = k; slot < kend; ++slot) {
            const Variable *vk = jitc_var(call->slots[slot].ref);
            if (loadable(vk))
                last_load = slot;
        }

        // Skip a bucket with no loadable fields; otherwise trim trailing holes.
        if (last_load == (uint32_t) -1)
            continue;
        uint32_t run_len = last_load - k + 1;

        // Canonical integer type matching the run's element width.
        VarType cvt = s == 8 ? VarType::UInt64
                    : s == 4 ? VarType::UInt32
                    : s == 2 ? VarType::UInt16
                             : VarType::UInt8;
        Variable rep{};
        rep.type = (uint32_t) cvt;

        uint32_t pos = 0;
        while (pos < run_len) {
            uint32_t rem = run_len - pos;

            // Largest packet count (>= 2, <= 8, <= SIMD width) that stays
            // sufficiently utilized. A return value < 2 falls back to the
            // per-field gather path below.
            uint32_t p = jitc_call_pick_llvm_packet_count(
                rem, width < 8u ? width : 8u);
            if (p < 2)
                break;

            uint32_t valid   = std::min(rem, p),
                     run_off = rel0 + pos * s,
                     id      = uid++;

            // Each run starts at its size class's packet-aligned boundary (we
            // coalesce on "coalesceable", the property the layout aligns to) and
            // advances by whole packets, so ``run_off`` is a multiple of ``p*s`` --
            // the alignment jitc_llvm_gather_packet_define() bakes into the load.
            jitc_assert(run_off % (p * s) == 0,
                        "jitc_llvm_render_call_data(): misaligned packet load "
                        "(run_off=%u, p=%u, s=%u)", run_off, p, s);

            jitc_llvm_gather_packet_define(&rep, p);

            fmt("    %cd$u_p1 = getelementptr inbounds i8, ptr %data, i32 $u\n"
                "    %cd$u_p2 = getelementptr inbounds i8, ptr %cd$u_p1, <$w x i32> %offsets\n"
                "    %cd$u_p3 = getelementptr [$u x $m], ptr @zero_$m_$u, <$w x i32> zeroinitializer\n"
                "    %cd$u_p4 = select <$w x i1> %mask, <$w x ptr> %cd$u_p2, <$w x ptr> %cd$u_p3\n"
                "    %cd$u_r = call fastcc [$u x <$w x $m>] @gather_$ux$H(<$w x ptr> %cd$u_p4)\n",
                id, run_off,
                id, id,
                id, p, &rep, &rep, p,
                id, id, id,
                id, p, &rep, p, &rep, id);

            for (uint32_t j = 0; j < valid; ++j) {
                uint32_t slot = k + pos + j;
                const Variable *vf = jitc_var(call->slots[slot].ref);

                // Skip holes: their lane was loaded but has no SSA value to bind.
                if (!loadable(vf))
                    continue;

                VarType ft = (VarType) vf->type;

                // Integers of the run's width are the same LLVM type as the
                // gathered words and bind directly; floats/bool/pointers convert
                // from the canonical integer.
                bool is_int = ft != VarType::Bool && ft != VarType::Pointer &&
                              ft != VarType::Float16 && ft != VarType::Float32 &&
                              ft != VarType::Float64;

                if (is_int) {
                    fmt("    $v = extractvalue [$u x <$w x $m>] %cd$u_r, $u\n",
                        vf, p, &rep, id, j);
                } else {
                    fmt("    %cd$u_o$u = extractvalue [$u x <$w x $m>] %cd$u_r, $u\n",
                        id, j, p, &rep, id, j);
                    if (ft == VarType::Bool)
                        fmt("    $v = trunc <$w x i8> %cd$u_o$u to <$w x i1>\n",
                            vf, id, j);
                    else if (ft == VarType::Pointer)
                        // A pointer's '$T' is '<w x i64>'; the SSA value is the
                        // '<w x ptr>' produced here (cf. the scalar path).
                        fmt("    $v = inttoptr <$w x $m> %cd$u_o$u to <$w x ptr>\n",
                            vf, &rep, id, j);
                    else // float / half / double
                        fmt("    $v = bitcast <$w x $m> %cd$u_o$u to $T\n",
                            vf, &rep, id, j, vf);
                }
            }

            pos += valid;
            packetized_until[c] = k + pos;
        }
    }
}

void scatter_packet_recursive(ReduceOp op, uint32_t l, uint32_t i, uint32_t n, const Variable *v) {
    uint32_t s = jitc_llvm_vector_width >> l;

    if (s == n) {
        for (uint32_t j = 0; j < n; ++j) {
            uint32_t k = i*n+j, align = type_size[v->type]*n;

            fmt( "    %p$u = extractelement <$w x ptr> %p, i32 $u\n",
                k, k);

            if (false) {
                /* Compile via Masked loads/store */
                fmt("    %m$u = shufflevector <$w x i1> %m, <$w x i1> undef, <$u x i32> <", k, n);
                for (uint32_t m = 0; m < n; ++m)
                    fmt("i32 $u$s", k, m + 1 < n ? ", " : ">\n");
                if (op == ReduceOp::Identity) {
                    fmt("    call void @llvm.masked.store.v$u$H.p0(<$u x $m> %v$u_$u_$u, ptr %p$u, i32 $u, <$u x i1> %m$u)\n",
                        n, v, n, v, l, i, j, k, align, n, k);
                } else if (op == ReduceOp::Add) {
                    fmt("    %v$u_0 = call <$u x $m> @llvm.masked.load.v$u$H.p0(ptr %p$u, i32 $u, <$u x i1> %m$u, <$u x $m> undef)\n"
                        "    %v$u_1 = $s <$u x $m> %v$u_0, %v$u_$u_$u\n"
                        "    call void @llvm.masked.store.v$u$H.p0(<$u x $m> %v$u_1, ptr %p$u, i32 $u, <$u x i1> %m$u)\n",
                        k, n, v, n, v, k, align, n, k, n, v,
                        k, jitc_is_float(v) ? "fadd" : "add", n, v, k, l, i, j,
                        n, v, n, v, k, k, align, n, k);
                }
            } else {
                fmt("    br label %l$u_pre\n\n"
                    "l$u_pre:\n"
                    "    %m$u = extractelement <$w x i1> %m, i32 $u\n"
                    "    br i1 %m$u, label %l$u_store, label %l$u_post\n\n"
                    "l$u_store:\n",
                    k, k, k, k, k, k, k, k);

                if (op == ReduceOp::Identity) {
                    fmt("    store <$u x $m> %v$u_$u_$u, ptr "
                        "%p$u, align $u, !alias.scope !2, !noalias !2\n",
                        n, v, l, i, j, k, align);
                } else if (op == ReduceOp::Add) {
                    fmt("    %v$u_0 = load <$u x $m>, ptr %p$u, align $u\n"
                        "    %v$u_1 = $s <$u x $m> %v$u_0, %v$u_$u_$u\n"
                        "    store <$u x $m> %v$u_1, ptr %p$u, align $u, !noalias !2\n",
                        k, n, v, k, align,
                        k, jitc_is_float(v) ? "fadd" : "add", n, v, k, l, i, j,
                        n, v, k, k, align);
                }

                fmt("    br label %l$u_post\n\n"
                    "l$u_post:\n",
                    k, k);
            }
        }
    } else {
        uint32_t hs = s / 2;

        for (uint32_t m = 0; m < 2; ++m) {
            for (uint32_t j = 0; j < n; ++j) {
                fmt("    %v$u_$u_$u = shufflevector <$u x $m> %v$u_$u_$u, <$u x $m> undef, <$u x i32> <",
                    l+1, i*2+m, j, s, v, l, i, j, s, v, hs);
                for (uint32_t k = 0; k < hs; ++k)
                    fmt("i32 $u$s", k+m*hs, k + 1 < hs ? ", " : ">\n");
            }
        }

        scatter_packet_recursive(op, l + 1, 2*i, n, v);
        scatter_packet_recursive(op, l + 1, 2*i+1, n, v);
    }
}

void jitc_llvm_scatter_packet_render_function(const Variable *v0, uint32_t n,
                                              ReduceOp op) {
    // Render scatter function
    size_t offset = buffer.size();
    const char *op_name = "";
    if (op == ReduceOp::Add) {
        op_name = "add_";
        fmt_intrinsic("declare <$u x $m> @llvm.masked.load.v$u$H.p0(ptr, i32, <$u x i1>, <$u x $m>)",
                      n, v0, n, v0, n, n, v0);
    }
    fmt_intrinsic("declare void @llvm.masked.store.v$u$H.p0(<$u x $m>, ptr, i32, <$u x i1>)",
                  n, v0, n, v0, n);

    fmt("define internal fastcc void @scatter_$s$ux$H(<$w x ptr> %p, <$w x i1> %m, [$u x <$w x $m>] %v) local_unnamed_addr #0 {\n",
        op_name, n, v0, n, v0);

    for (uint32_t i = 0; i < n; ++i)
        fmt("    %v$u = extractvalue [$u x <$w x $m>] %v, $u\n",
            i, n, v0, i);

    jitc_llvm_permute_n(v0, n);

    for (uint32_t i = 0; i < n; ++i)
        fmt("    %v0_0_$u = bitcast $M %r$u to $M\n", i, v0, i, v0);

    scatter_packet_recursive(op, 0, 0, n, v0);

    fmt("    ret void\n"
        "}");

    jitc_register_global(buffer.get() + offset);
    buffer.rewind_to(offset);
}

void jitc_llvm_render_scatter_packet(const Variable *v, const Variable *ptr,
                                     const Variable *index, const Variable *mask) {
    PacketScatterData *psd = (PacketScatterData *) v->data;
    uint32_t n             = (uint32_t) psd->values.size();
    const Variable *v0     = jitc_var(psd->values[0]);
    ReduceOp op            = psd->op;

    const char *op_name    = "";
    if (op == ReduceOp::Add)
        op_name = "add_";

    // Split large requests into the largest possible packet sizes. For
    // example, a packet of 6 variables will be split into 3 scatters with 2
    // variables each.
    uint32_t packet_size = std::min(8u, jitc_llvm_vector_width);
    while ((n & (packet_size - 1)) != 0)
        packet_size /= 2;

    jitc_llvm_scatter_packet_render_function(v0, packet_size, op);

    for (uint32_t offset = 0; offset < n; offset += packet_size) {
        if (v0->type != (uint32_t) VarType::Bool) {
            fmt("    $v_$u = insertvalue [$u x <$w x $t>] undef, $V, 0\n",
                v, offset, packet_size, v0, jitc_var(psd->values[offset]));

            for (uint32_t i = 1; i < packet_size; ++i)
                fmt("    $v_$u = insertvalue [$u x <$w x $t>] $v_$u, $V, $u\n",
                    v, i + offset, packet_size, v0, v, i-1 + offset, jitc_var(psd->values[i + offset]), i);
        } else {
            fmt("    $v_$u_e = zext $V to $M\n"
                "    $v_$u = insertvalue [$u x <$w x $m>] undef, $M $v_$u_e, 0\n",
                v, offset, jitc_var(psd->values[offset]), v0,
                v, offset, packet_size, v0, v0, v, offset);

            for (uint32_t i = 1; i < packet_size; ++i){
                fmt("    $v_$u_e = zext $V to $M\n"
                    "    $v_$u = insertvalue [$u x <$w x $m>] $v_$u, $M $v_$u_e, $u\n",
                    v, i + offset, jitc_var(psd->values[i + offset]), v0,
                    v, i + offset, packet_size, v0, v, i-1 + offset, v0, v, i + offset, i);
            }
        }

        // First offset the base pointer by the packet offset.
        fmt("    $v_$u_p1 = getelementptr $m, $<ptr$> $v, i32 $u\n"
            "    $v_$u_p2 = getelementptr $m, $<ptr$> $v_$u_p1, $V\n",
            v, offset, v0, ptr, offset,
            v, offset, v0, v, offset, index);

        fmt("    call fastcc void @scatter_$s$ux$H(<$w x ptr> $v_$u_p2, $V, [$u x <$w x $m>] $v_$u)\n",
            op_name, packet_size, v0, v, offset, mask, packet_size, v0, v, offset + packet_size-1);
    }
}
