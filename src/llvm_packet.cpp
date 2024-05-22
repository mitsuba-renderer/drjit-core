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
#include "log.h"
#include "op.h"

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

            fmt( "    %p$u = extractelement <$w x {$m*}> %p, i32 $u\n"
                "{    %p$u_0 = bitcast $m* %p$u to <$u x $m>*\n|}"
                 "    %v$u_$u_$u = load <$u x $m>, {<$u x $m>*} %p$u{_0|}, align $u, !alias.scope !2\n",
                k, v, k,
                k, v, k, n, v,
                l, i, j, n, v, n, v, k, align);
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


void jitc_llvm_render_gather_packet(const Variable *v, const Variable *ptr,
                                    const Variable *index, const Variable *mask) {
    size_t offset = buffer.size();
    uint32_t n = (uint32_t) v->literal;

    fmt_intrinsic("@zero_$m_$u = private constant [$u x $m] $z, align $A", v, n, n, v, v);

    fmt("define internal fastcc [$u x <$w x $m>] @gather_$ux$H(<$w x {$m*}> %p) local_unnamed_addr #0 ${\n",
        n, v, n, v, v);

    gather_packet_recursive(0, 0, n, v);

    for (uint32_t i = 0; i < n; ++i)
        fmt("    %v$u = bitcast $M %v0_0_$u to $M\n", i, v, i, v);

    jitc_llvm_permute_n(v, n);

    fmt("    %q0 = insertvalue [$u x <$w x $m>] undef, <$w x $m> %r0, 0\n", n, v, v);
    for (uint32_t i = 1; i < n; ++i)
        fmt("    %q$u = insertvalue [$u x <$w x $m>] %q$u, <$w x $m> %r$u, $u\n",
            i, n, v, i - 1, v, i, i);

    fmt("    ret [$u x <$w x $m>] %q$u\n"
        "$}",
        n, v, n - 1);

    jitc_register_global(buffer.get() + offset);
    buffer.rewind_to(offset);

    const char *ext = "";
    if (v->type == (uint32_t) VarType::Bool)
        ext = "_e";

    fmt("{    $v_0 = bitcast $<i8*$> $v to $<[$u x $m]*$>\n"
         "    $v_2 = getelementptr [$u x $m], $<{[$u x $m]*}$> {$v_0|$v}, $T $v\n"
         "    $v_3 = getelementptr [$u x $m], [$u x $m]* @zero_$m_$u, $T $z\n"
         "    $v_4 = select $V, <$w x {[$u x $m]*}> $v_2, <$w x {[$u x $m]*}> $v_3\n"
        "{    $v_5 = bitcast <$w x [$u x $m]*> $v_4 to <$w x $m*>\n|}"
         "    $v_6 = call fastcc [$u x <$w x $m>] @gather_$ux$H(<$w x {$m*}> $v_{5|4})\n",
        v, ptr, n, v,
        v, n, v, n, v, v, ptr, index, index,
        v, n, v, n, v, v, n, index,
        v, mask, n, v, v, n, v, v,
        v, n, v, v,  v,
        v, n, v, n, v, v, v
    );

    for (uint32_t i = 0; i < n; ++i)
        fmt("    $v_out_$u$s = extractvalue [$u x <$w x $m>] $v_6, $u\n",
            v, i, ext, n, v, v, i);

    if (v->type == (uint32_t) VarType::Bool) {
        for (uint32_t i = 0; i < n; ++i)
            fmt("    $v_out_$u = trunc <$w x $m> $v_out_$u_e to <$w x i1>\n",
                v, i, v, v, i);
    }
}

void scatter_packet_recursive(ReduceOp op, uint32_t l, uint32_t i, uint32_t n, const Variable *v) {
    uint32_t s = jitc_llvm_vector_width >> l;

    if (s == n) {
        for (uint32_t j = 0; j < n; ++j) {
            uint32_t k = i*n+j, align = type_size[v->type]*n;

            fmt( "    %p$u = extractelement <$w x {$m*}> %p, i32 $u\n"
                "{    %p$u_0 = bitcast $m* %p$u to <$u x $m>*\n|}",
                k, v, k,
                k, v, k, n, v);

            if (false) {
                /* Compile via Masked loads/store */
                fmt("    %m$u = shufflevector <$w x i1> %m, <$w x i1> undef, <$u x i32> <", k, n);
                for (uint32_t i = 0; i < n; ++i)
                    fmt("i32 $u$s", k, i + 1 < n ? ", " : ">\n");
                if (op == ReduceOp::Identity) {
                    fmt("    call void @llvm.masked.store.v$u$H.p0{v$u$H|}(<$u x $m> %v$u_$u_$u, {<$u x $m> *} %p$u{_0|}, i32 $u, <$u x i1> %m$u)\n",
                        n, v, n, v, n, v, l, i, j, n, v, k, align, n, k);
                } else if (op == ReduceOp::Add) {
                    fmt("    %v$u_0 = call <$u x $m> @llvm.masked.load.v$u$H.p0{v$u$H|}({<$u x $m> *} %p$u{_0|}, i32 $u, <$u x i1> %m$u, <$u x $m> undef)\n"
                        "    %v$u_1 = $s <$u x $m> %v$u_0, %v$u_$u_$u\n"
                        "    call void @llvm.masked.store.v$u$H.p0{v$u$H|}(<$u x $m> %v$u_1, {<$u x $m> *} %p$u{_0|}, i32 $u, <$u x i1> %m$u)\n",
                        k, n, v, n, v, n, v, n, v, k, align, n, k, n, v,
                        k, jitc_is_float(v) ? "fadd" : "add", n, v, k, l, i, j,
                        n, v, n, v, n, v, k, n, v, k, align, n, k);
                }
            } else {
                fmt("    br label %l$u_pre\n\n"
                    "l$u_pre:\n"
                    "    %m$u = extractelement <$w x i1> %m, i32 $u\n"
                    "    br i1 %m$u, label %l$u_store, label %l$u_post\n\n"
                    "l$u_store:\n",
                    k, k, k, k, k, k, k, k);

                if (op == ReduceOp::Identity) {
                    fmt("    store <$u x $m> %v$u_$u_$u, {<$u x $m>*} "
                        "%p$u{_0|}, align $u, !noalias !2\n",
                        n, v, l, i, j, n, v, k, align);
                } else if (op == ReduceOp::Add) {
                    fmt("    %v$u_0 = load <$u x $m>, {<$u x $m>*} %p$u{_0|}, align $u\n"
                        "    %v$u_1 = $s <$u x $m> %v$u_0, %v$u_$u_$u\n"
                        "    store <$u x $m> %v$u_1, {<$u x $m>*} %p$u{_0|}, align $u, !noalias !2\n",
                        k, n, v, n, v, k, align,
                        k, jitc_is_float(v) ? "fadd" : "add", n, v, k, l, i, j,
                        n, v, k, n, v, k, align);
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


void jitc_llvm_render_scatter_packet(const Variable *v, const Variable *ptr,
                                     const Variable *index, const Variable *mask) {
    size_t offset = buffer.size();
    PacketScatterData *psd = (PacketScatterData *) v->data;
    uint32_t n = (uint32_t) psd->values.size();
    const Variable *v0 = jitc_var(psd->values[0]);

    const char *op_name = "";
    if (psd->op == ReduceOp::Add) {
        op_name = "add_";
        fmt_intrinsic("declare <$u x $m> @llvm.masked.load.v$u$H.p0{v$u$H|}({<$u x $m> *}, i32, <$u x i1>, <$u x $m>)",
                      n, v0, n, v0, n, v0, n, v0, n, n, v0);
    }
    fmt_intrinsic("declare void @llvm.masked.store.v$u$H.p0{v$u$H|}(<$u x $m>, {<$u x $m> *}, i32, <$u x i1>)",
                  n, v0, n, v0, n, v0, n, v0, n);

    fmt("define internal fastcc void @scatter_$s$ux$H(<$w x {$m*}> %p, <$w x i1> %m, [$u x <$w x $m>] %v) local_unnamed_addr #0 ${\n",
        op_name, n, v0, v0, n, v0);

    for (uint32_t i = 0; i < n; ++i)
        fmt("    %v$u = extractvalue [$u x <$w x $m>] %v, $u\n",
            i, n, v0, i);

    jitc_llvm_permute_n(v0, n);

    for (uint32_t i = 0; i < n; ++i)
        fmt("    %v0_0_$u = bitcast $M %r$u to $M\n", i, v0, i, v0);

    scatter_packet_recursive(psd->op, 0, 0, n, v0);

    fmt("    ret void\n"
        "$}");

    jitc_register_global(buffer.get() + offset);
    buffer.rewind_to(offset);

    if (v0->type != (uint32_t) VarType::Bool) {
        fmt("    $v_0 = insertvalue [$u x <$w x $t>] undef, $V, 0\n", v, n, v0, v0);

        for (uint32_t i = 1; i < n; ++i)
            fmt("    $v_$u = insertvalue [$u x <$w x $t>] $v_$u, $V, $u\n",
                v, i, n, v0, v, i-1, jitc_var(psd->values[i]), i);
    } else {
        fmt("    $v_0_e = zext $V to $M\n"
            "    $v_0 = insertvalue [$u x <$w x $m>] undef, $M $v_0_e, 0\n",
            v, v0, v0,
            v, n, v0, v0, v);

        for (uint32_t i = 1; i < n; ++i)
            fmt("    $v_$u_e = zext $V to $M\n"
                "    $v_$u = insertvalue [$u x <$w x $m>] $v_$u, $M $v_$u_e, $u\n",
                v, i, jitc_var(psd->values[i]), v0,
                v, i, n, v0, v, i-1, v0, v, i, i);
    }

    fmt("{    $v_p0 = bitcast $<i8*$> $v to $<$m*$>\n|}"
         "    $v_p1 = getelementptr $m, $<$m*$> {$v_p0|$v}, $V\n"
         "    call fastcc void @scatter_$s$ux$H(<$w x {$m*}> $v_p1, $V, [$u x <$w x $m>] $v_$u)\n",
        v, ptr, v0,
        v, v0, v0, v, ptr, index,
        op_name, n, v0, v0, v, mask, n, v0, v, n-1);
}
