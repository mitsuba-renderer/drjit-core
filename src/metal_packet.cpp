/*
    src/metal_packet.cpp -- Packet scatter/gather for MSL

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "metal_packet.h"
#include "metal_scatter.h"
#include "internal.h"
#include "var.h"
#include "op.h"
#include "log.h"
#include "strbuf.h"
#include "metal_eval.h"

/// MSL binary (storage) type for the desired byte size
static const char *metal_packet_word_type(uint32_t bytes) {
    switch (bytes) {
        case 16: return "uint4";
        case 8:  return "uint2";
        case 4:  return "uint";
        case 2:  return "ushort";
        case 1:  return "uchar";
        default: jitc_fail("metal_packet_word_type(): unsupported "
                           "chunk width %u!", bytes);
    }
}

/// Widest naturally aligned chunk size (in bytes) for transferring a packet of
/// ``total_bytes`` bytes with element size ``tsize``
static uint32_t metal_packet_chunk_bytes(uint32_t tsize, uint32_t total_bytes) {
    uint32_t bytes = 4 * tsize > 16 ? 16 : 4 * tsize;
    while (total_bytes % bytes)
        bytes /= 2;
    return bytes;
}

void jitc_metal_render_gather_packet(const Variable *v, const Variable *ptr,
                                    const Variable *index, const Variable *mask) {
    bool is_masked = !mask->is_literal() || mask->literal != 1;
    VarType vt     = (VarType) v->type;

    uint32_t count       = (uint32_t) v->literal,
             tsize       = type_size[v->type],
             total_bytes = count * tsize;

    uint32_t load_bytes = metal_packet_chunk_bytes(tsize, total_bytes);
    uint32_t lanes      = load_bytes / tsize,  // elements decoded per load (1, 2 or 4)
             n_loads    = total_bytes / load_bytes;

    const char *load_ty = metal_packet_word_type(load_bytes);

    // Element type used when reinterpreting a loaded word. Booleans are stored
    // as bytes, decoded through ``uchar`` and converted implicitly on assignment.
    const char *dec_base = vt == VarType::Bool ? "uchar" : type_name_metal[(int) vt];

    // Base pointer of the packet for this lane
    fmt("device const $s *$v_base = (device const $s*) ($v + (ulong) $v * $u);\n",
        load_ty, v, load_ty, ptr, index, total_bytes);

    // Issue the coalesced loads
    for (uint32_t c = 0; c < n_loads; ++c) {
        if (is_masked)
            fmt("$s $v_pkt_$u = $v ? $v_base[$u] : $s(0);\n",
                load_ty, v, c, mask, v, c, load_ty);
        else
            fmt("$s $v_pkt_$u = $v_base[$u];\n",
                load_ty, v, c, v, c);

        // Reinterpret the raw bits as a vector of `lanes` elements.
        if (lanes == 1)
            fmt("$s $v_dec_$u = as_type<$s>($v_pkt_$u);\n",
                dec_base, v, c, dec_base, v, c);
        else
            fmt("$s$u $v_dec_$u = as_type<$s$u>($v_pkt_$u);\n",
                dec_base, lanes, v, c, dec_base, lanes, v, c);
    }

    // Extract the individual elements into `$v_out_<i>`
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t c = i / lanes,
                 l = i % lanes;
        if (lanes == 1)
            fmt("$t $v_out_$u = $v_dec_$u;\n", v, v, i, v, c);
        else
            fmt("$t $v_out_$u = $v_dec_$u[$u];\n", v, v, i, v, c, l);
    }
}

/// Render the MSL to scatter-reduce a variable packet
static void jitc_metal_render_scatter_reduce_packet(const Variable *ptr,
                                                    const Variable *index,
                                                    const Variable *mask,
                                                    PacketScatterData *psd) {
    const std::vector<uint32_t> &values = psd->values;
    bool is_masked = !mask->is_literal() || mask->literal != 1;

    if (is_masked)
        fmt("if ($v) {\n", mask);

    jitc_metal_emit_reduce_block((uint32_t) values.size(), values.data(), ptr,
                                 index, psd->op, psd->mode == ReduceMode::Local);

    if (is_masked)
        put("}\n");
}

void jitc_metal_render_scatter_packet(const Variable *v, const Variable *ptr,
                                     const Variable *index, const Variable *mask) {
    PacketScatterData *psd              = (PacketScatterData *) v->data;
    const std::vector<uint32_t> &values = psd->values;
    const Variable *v0                  = jitc_var(values[0]);

    if (psd->op != ReduceOp::Identity) {
        jitc_metal_render_scatter_reduce_packet(ptr, index, mask, psd);
        return;
    }

    bool is_masked = !mask->is_literal() || mask->literal != 1;
    VarType vt     = (VarType) v0->type;

    uint32_t count       = (uint32_t) values.size(),
             tsize       = type_size[v0->type],
             total_bytes = count * tsize;

    uint32_t store_bytes = metal_packet_chunk_bytes(tsize, total_bytes);
    uint32_t lanes       = store_bytes / tsize,  // elements packed per store (1, 2 or 4)
             n_stores    = total_bytes / store_bytes;

    const char *store_ty = metal_packet_word_type(store_bytes);

    // Element type used when assembling a word. Booleans are stored as bytes.
    const char *dec_base = vt == VarType::Bool ? "uchar" : type_name_metal[(int) vt];

    // The scatter index is already scaled by `count` (see op.cpp)
    fmt("device $s *$v_base = (device $s*) ($v + (ulong) $v * $u);\n",
        store_ty, v, store_ty, ptr, index, tsize);

    for (uint32_t c = 0; c < n_stores; ++c) {
        if (lanes == 1) {
            fmt("$s $v_pkt_$u = as_type<$s>($v);\n",
                store_ty, v, c, store_ty, jitc_var(values[c]));
        } else {
            fmt("$s$u $v_dec_$u = $s$u(", dec_base, lanes, v, c, dec_base, lanes);
            for (uint32_t l = 0; l < lanes; ++l) {
                if (l)
                    put(", ");
                fmt("$v", jitc_var(values[c * lanes + l]));
            }
            put(");\n");
            fmt("$s $v_pkt_$u = as_type<$s>($v_dec_$u);\n",
                store_ty, v, c, store_ty, v, c);
        }

        if (is_masked)
            fmt("if ($v) $v_base[$u] = $v_pkt_$u;\n", mask, v, c, v, c);
        else
            fmt("$v_base[$u] = $v_pkt_$u;\n", v, c, v, c);
    }
}
