#include "eval.h"
#include "cuda_queue.h"
#include "cuda_eval.h"
#include "internal.h"
#include "queue.h"
#include "var.h"

/// Stride between head/tail/virtual head pointers to avoid false sharing
constexpr uint32_t CounterStride = 64 * (uint32_t) sizeof(uint32_t);

/**
 * \brief Generate efficient PTX to load/store a potentially heterogeneous
 * sequence of values from/to contiguous memory.
 *
 * The function groups consecutive loads/stores values into a minimal number of
 * maximally large vector memory operations. It handles 16-, 32-, and 64-bit
 * integer and floating point data types. It uses weak stores and relaxed/strong
 * loads to enable communication with another thread in conjunction with memory
 * barriers that must be inserted by the caller.
 *
 * The function assumes that:
 *  - all values are properly aligned without need for padding
 *  - 16-bit values come in properly paired sequences
 *
 * Example output for loads:
 * \\code
 *     .reg.b32 $v_0;
 *     .reg.b32 $v_1;
 *     {
 *         .reg.b32 %tmp_<4>;
 *         ld.relaxed.global.v2.b32 {%tmp_0, %tmp_1}, [%ptr + 0];
 *         mov.b32 $v_0, %tmp_0;
 *         mov.b32 $v_1, %tmp_1;
 *     }
 * \\endcode
 */
static void packet_memop(const Variable *v, const Variable *ptr, uint32_t n,
                         const uint32_t *values, const VarType *vt, bool load) {
    // Generate local variable declarations for loads
    if (load) {
        bool uniform_tp = true;
        for (uint32_t i = 1; i < n; i++)
            uniform_tp &= vt[i] == vt[0];
        if (uniform_tp) {
            fmt("        .reg.b$u $v_<$u>;\n", type_size[(int) vt[0]] * 8, v, n);
        } else {
            for (uint32_t i = 0; i < n; i++)
                fmt("        .reg.b$u $v_$u;\n", type_size[(int) vt[i]] * 8, v, i);
        }
    }

    // Open local scope and declare temp registers
    put("    {\n"
        "        .reg.b32 %tmp_<4>;\n");

    const uint32_t max_packet =
        thread_state_cuda->compute_capability > 100 ? 32 : 16;

    uint32_t index = 0, offset = 0, count, nbytes;

    while (index < n) {
        // Choose the largest packet size in [32 for sm_100, 16, 8, 4 bytes]
        for (uint32_t packet_size = max_packet; packet_size >= 4; packet_size /= 2) {
            count = nbytes = 0;

            for (uint32_t i = index; i < n; i++) {
                uint32_t vsize =
                    type_size[load ? (int) vt[i] : jitc_var(values[i])->type];

                if (nbytes + vsize > packet_size)
                    break;

                count++;
                nbytes += vsize;
            }

            if (count > 0) {
                if (nbytes % 4 != 0)
                    jitc_fail("packet_memop(): number of bytes (%u) must be a "
                              "multiple of 4", nbytes);

                if (nbytes == packet_size)
                    break;
            }
        }

        // Generate load instructions
        if (load) {
            if (nbytes == 4) {
                fmt("        ld.relaxed.global.b32 %tmp_0, [$v_data+$u];\n", ptr, offset);
            } else {
                fmt("        ld.relaxed.global.v$u.b32 {", nbytes / 4);
                for (uint32_t i = 0; i < nbytes / 4; i++)
                    fmt("$s%tmp_$u", i> 0 ? ", " : "", i);
                fmt("}, [$v_data+$u];\n", ptr, offset);
            }
        }

        // Pack or unpack values
        for (uint32_t i = 0, pos = 0; i < count; i++) {
            uint32_t j          = index + i,
                     vsize      = type_size[load ? (int) vt[j] : jitc_var(values[j])->type],
                     word_index = pos / 4;

            if (vsize == 2) {
                if (load)
                    fmt("        mov.b32 {$v_$u, $v_$u}, %tmp_$u;\n", v, j, v, j + 1, word_index);
                else
                    fmt("        mov.b32 %tmp_$u, {$v, $v};\n", word_index, values[j], values[j + 1]);
                vsize = 4; i += 1; // Process two half precision floats at once
            } else if (vsize == 4) {
                if (load)
                    fmt("        mov.b32 $v_$u, %tmp_$u;\n", v, j, word_index);
                else
                    fmt("        mov.b32 %tmp_$u, $v;\n", word_index, values[j]);
            } else if (vsize == 8) {
                if (load)
                    fmt("        mov.b64 $v_$u, {%tmp_$u, %tmp_$u};\n",
                           v, j, word_index, word_index + 1);
                else
                    fmt("        mov.b64 {%tmp_$u, %tmp_$u}, $v;\n",
                           word_index, word_index + 1, values[j]);
            }
            pos += vsize;
        }

        // Generate store instructions
        if (!load) {
            if (nbytes == 4) {
                fmt("        st.weak.global.b32 [$v_data+$u], %tmp_0;\n", ptr, offset);
            } else {
                fmt("        st.weak.global.v$u.b32 [$v_data+$u], {", nbytes / 4, ptr, offset);
                for (uint32_t i = 0; i < nbytes / 4; i++)
                    fmt("$s%tmp_$u", i > 0 ? ", " : "", i);
                put("};\n");
            }
        }

        index += count;
        offset += nbytes;
    }

    // Close local scope
    put("    }\n");
}

void jitc_cuda_render_queue_send(Variable *v,
                                 Variable *queue_buffer,
                                 Variable *msg_type) {
    const QueueSendData *qsd = (const QueueSendData *) jitc_var_extra(v)->callback_data;

    // We might have to further group operations into peers if they
    // go to different virtual queues
    bool virtual_queue = qsd->msg_types != 1;
    const char *peers = "active";

    const uint32_t block_size_bits = log2i_ceil(qsd->block_size),
                   blocks_bits     = log2i_ceil(qsd->blocks);

    fmt("    // Queue encode logic for msg_types=$u, block_size=$u ($u bits), blocks=$u ($u bits)\n",
        qsd->msg_types, qsd->block_size, block_size_bits, qsd->blocks, blocks_bits);

    fmt("    // 1. Determine mask for opportunistic warp-level cooporation\n"
        "    .reg.b32 $v_active, $v_lane_id, $v_lane_lt, $v_queue_id, $v_tmp, $v_tmp2;\n"
        "    .reg.pred $v_ready;\n"
        "    activemask.b32 $v_active;\n"
        "    mov.b32 $v_lane_id, %laneid;\n"
        "    mov.b32 $v_lane_lt, %lanemask_lt;\n",
        v, v, v, v, v, v,
        v,
        v,
        v,
        v);

    if (virtual_queue)
        fmt("    add.u32 $v_queue_id, $v, 1;\n", v, msg_type);
    else
        fmt("    mov.u32 $v_queue_id, 0;\n", v);

    if (virtual_queue && !msg_type->is_literal()) {
        fmt("    .reg.b32 $v_peers;\n"
            "    match.any.sync.b32 $v_peers, $v, $v_active;\n",
            v,
            v, msg_type, v);
        peers = "peers";
    }

    fmt("\n"
        "    // 2. Determine a leader and count the messages in each peer group\n"
        "    .reg.b32 $v_leader_id, $v_count;\n"
        "    .reg.pred $v_leader;\n"
        "    bfind.u32 $v_leader_id, $v_$s;\n"
        "    popc.b32 $v_count, $v_$s;\n"
        "    setp.eq.u32 $v_leader, $v_leader_id, $v_lane_id;\n",
        v, v,
        v,
        v, v, peers,
        v, v, peers,
        v, v, v);

    fmt("\n"
        "    // 3. The leader reserves space by bumping the logical queue head pointer\n"
        "    .reg.b64 $v_head;\n"
        "    mad.wide.b32 $v_head, $v_queue_id, $u, $v;"
        "    @$v_leader atom.relaxed.gpu.global.add.u32 $v_offset, [$v_head], $v_count;\n",
        v,
        v, v, CounterStride, queue_buffer,
        v, v, v, v);

    fmt("\n"
        "    // 4. Generate indices for all peers based on the leader's response\n"
        "    .reg.b32 $v_peers_lt, $v_offset_local, $v_offset;\n"
        "    and.b32 $v_peers_lt, $v_lane_lt, $v_$s;\n"
        "    popc.b32 $v_offset_local, $v_peers_lt;\n"
        "    shfl.sync.idx.b32 $v_offset, $v_offset, $v_leader_id, 31, $v_active;\n"
        "    add.u32 $v_offset, $v_offset, $v_offset_local;\n",
        v, v, v,
        v, v, v, peers,
        v, v,
        v, v, v, v,
        v, v, v);

    uint32_t state_offset = (qsd->msg_types + uint32_t(virtual_queue)) * CounterStride,
             state_size   = qsd->blocks * (uint32_t) sizeof(uint32_t),
             data_offset  = state_offset + state_size * (qsd->msg_types + uint32_t(virtual_queue));

    /*
        The queue_buffer array consists of three parts:

        1. Queue heads -- a 32-bit unsigned integer each.

           - Physical queue head. SKIPPED when ``virtual_queue == false``.
           - A set of ``msg_types`` logical queue heads

           They are padded and each take up ``CounterStride`` bytes

           (The server queue tail is stored elsewhere.)

        2. Virtual/logical queue state

           - Physical queue state: ``blocks`` uint32_t values. SKIPPED when ``virtual_queue == false``
           - Logical queue state: ``msg_types * blocks`` uint32_t values.

        3. The physical message payload

        -------------------------------------------------------------------------

        The logical queue head entries are counters, whose elements are interpreted as follows:

        Bit 31         .....               0
        | log_ver | log_block | log_offset |


        where

          log_offset: position within the block. Uses ``block_size_bits`` bits
          log_block:  block ID within the physical or logical queue. log2i(blocks) bits
          log_ver:    version number. Increases as blocks are repeatedly used.

        -------------------------------------------------------------------------

        The logical queue state entries have the following interpretation:

        Bit 31         .....               0
        | log_ver | unused | phy_block |

          phy_block:  physical queue block associated with this logical queue block.
          log_ver:    version number matching the bit position & depth in the queue head

        -------------------------------------------------------------------------

        The physical queue state entries have the following interpretation

        Bit 31             .....                     0
        | (unused) | msg_type | num_resp | num_query |

        where

          msg_type:  Message type encoded in this block. log2i(msg_types) bits.
          num_resp:  Number of consumed responses. Uses ``block_size_bits+1`` bits.
          num_query: Number of submitted queries.  Uses ``block_size_bits+1`` bits.

        -------------------------------------------------------------------------
    */

    const uint32_t version_mask = 0xFFFFFFFFu /
                                  (qsd->blocks * qsd->block_size) *
                                  (qsd->blocks * qsd->block_size);

    fmt("\n"
        "    // 5. Determine the associated logical block, the offset within, and the version\n"
        "    .reg.b32 $v_log_offset, $v_log_block, $v_phy_block, $v_log_ver, $v_target;\n"
        "    .reg.b64 $v_phy_p;\n"
        "    and.b32 $v_log_offset, $v_offset, $u;\n"
        "    shr.u32 $v_log_block, $v_offset, $u;\n"
        "    and.b32 $v_log_block, $v_log_block, $u;\n"
        "    and.u32 $v_log_ver, $v_offset, $u;\n",
        v, v, v, v, v,
        v,
        v, v, qsd->block_size - 1,
        v, v, block_size_bits,
        v, v, qsd->blocks - 1,
        v, v, version_mask);

    if (virtual_queue) {
        fmt("\n"
            "    // 6.1. If log_offset == 0, allocate a logical block mapping:\n"
            "    .reg.pred $v_new_block;\n"
            "    .reg.b64 $v_log_p;\n"
            "    setp.eq.b32 $v_new_block, $v_log_offset, 0;\n"
            "    mad.wide.u32 $v_log_p, $v_queue_id, $u, $v;",
            "    @!$v_new_block l$u_sync;\n",
            v,
            v,
            v, v,
            v, v, state_size, queue_buffer,
            v, v->reg_index);

        fmt("\n"
            "    // 6.2. Request a block ID from the physical queue\n"
            "    atom.relaxed.gpu.global.add.u32 $v_phy_block, [$v], 1;\n"
            "    and.b32 $v_phy_block, $v_phy_block, $u;\n",
            v, queue_buffer,
            v, v, qsd->blocks - 1);

        fmt("\n"
            "    // 6.3. Lock the physical block\n"
            "    mad.wide.u32 $v_phy_p, $v_phy_block, 4, $v;\n"
            "    shl.u32 $v_target, $v_queue_id, $u;\n"
            "l$u_wait_phy:\n"
            "    atom.relaxed.gpu.global.cas.b32 $v_tmp, [$v_phy_p+$u], 0, $v_target;\n"
            "    setp.eq.b32 $v_ready, $v_tmp, 0;\n"
            "    @!$v_ready bra l$u_wait_phy;\n",
            v, v, queue_buffer,
            v, v, (block_size_bits+1)*2,
            v->reg_index,
            v, v, state_offset, v,
            v, v,
            v, v->reg_index);

        fmt("\n"
            "    // 6.4. Record the physical block in the logical queue map\n"
            "    and.b32 $v_tmp, $v_log_ver, $v_phy_block;\n"
            "    st.relaxed.gpu.global.b32 [$v_log_p+$u], $v_tmp;\n",
            v, v, v,
            v, state_offset, v);

        fmt("\n"
            "    // 6.5. The other threads must now determine the physical block ID\n"
            "l$u_sync:\n",
            v->reg_index);

        #if 0
        fmt("    // 6.4. Try to get the information from a neighboring thread\n"
            "    ballot.sync.b32 $v_tmp, $v_new_block;\n"
            "    and.b32 $v_tmp, $v_tmp, $v_peers;\n"
            "    setp.eq.b32 $v_pred, $v_tmp, 0;\n"
            "    @$v_pred $bra l$u_sync_fallback;\n"
            "    bfind.u32 $v_leader_id, $v_tmp;\n"
            "    shfl.sync.idx.b32 $v_phy_block, $v_phy_block, $v_leader_id, 31, $v_active;\n"
            "    bra l$u_mapped;\n",
            v, v,
            v, v, v,
            v, v,
            v, v->reg_index,
            v, v,
            v, v, v, v,
            v->reg_index
        );
        #endif

        fmt("\n"
            "    // 6.6. Query the logical queue map in global memory.\n"
            "l$u_wait_log:\n"
            "    ld.relaxed.gpu.global.b32 $v_tmp, [$v_log_p+$u];"
            "    and.b32 $v_tmp2, $v_tmp, $u;\n"
            "    setp.eq.b32 $v_ready, $v_tmp_2, $v_log_ver;\n"
            "    @!$v_ready bra l$u_wait_log;\n"
            "    and.b32 $v_phy_block, $v_tmp, $u;\n",
            v->reg_index,
            v, v, state_offset,
            v, v, version_mask,
            v, v, v,
            v, v->reg_index,
            v, v, qsd->blocks - 1);

        fmt("\nl$u_mapped:\n", v->reg_index);
    } else {
        fmt("\n"
            "    // 6. Simplified case -- logical and physical queues coincide\n"
            "    mov $v_phy_block, $v_log_block;\n",
            v, v);
    }

    fmt("\n"
        "    // 7. Designate a new leader based on the physical block ID\n"
        "    match.any.sync.b32 $v_peers, $v_phy_block, $v_active;\n"
        "    bfind.u32 $v_leader_id, $v_peers;\n"
        "    popc.b32 $v_count, $v_peers;\n"
        "    setp.eq.u32 $v_leader, $v_leader_id, $v_lane_id;\n",
        v, v, v,
        v, v,
        v, v,
        v, v, v
    );

    if (!virtual_queue) {
        fmt("\n"
            "    // 8. Lock the physical block\n"
            "    @!$v_leader br l$u_sync;\n"
            "    shr.u32 $v_target, $v_log_ver, $u;\n"
            "    add.u32 $v_target, $v_target, 1;\n"
            "    shl.u32 $v_target, $v_target, $u;\n"
            "    mad.wide.u32 $v_phy_p, $v_phy_block, 4, $v;\n"
            "    mov.u32 $v_target, $u;\n"
            "\n"
            "l$u_wait_phy:\n"
            "    atom.relaxed.gpu.global.cas.b32 $v_tmp, [$v_phy_p+$u], 0, $v_target;\n"
            "    setp.eq.b32 $v_ready, $v_tmp, 0;\n"
            "    @!$v_ready bra l$u_wait_phy;\n"
            "\n"
            "l$u_sync:\n"
            "    bar.warp.sync $v_active;\n",
            v, v->reg_index,
            v, v, block_size_bits+blocks_bits,
            v, v,
            v, v, (block_size_bits+1)*2,
            v, v, queue_buffer,
            v, 1u << ((block_size_bits+1)*2),
            v->reg_index,
            v, v, state_offset, v,
            v, v,
            v, v->reg_index);
    }

    // Compute the precise position for reading/writing our message
    fmt("    mad.lo.u32 $v_offset, $v_phy_block, $u, $u;\n"
        "    mad.lo.u32 $v_offset, $v_log_offset, $u, $v_offset;\n"
        "    cvt.u64.u32 $v_data, $v_offset;\n"
        "    add.u64 $v_data, $v_data, $v;\n",
        v->reg_index,
        v,
        v, v, qsd->msg_max_size*qsd->block_size, data_offset,
        v, v, qsd->msg_max_size, v,
        v, v,
        v, v, queue_buffer);

    // Insert an efficient write sequence for the message body
    packet_memop(v, v, (uint32_t) qsd->indices.size(),
                 qsd->indices.data(), nullptr, false);

    // Update peer/count values based on physical blocks, and signal successful write of messages
    fmt("    // 10. Inform the server that the messages have been written\n"
        "    @$v_leader red.release.gpu.global.add.u32 [$v_phy_p+$u], $v_count;\n",
        v, v, state_offset, v);

    queue_callbacks.push_back(qsd->callback);
}

void jitc_cuda_render_queue_recv(Variable *vr,
                                 Variable *v) {
    const QueueSendData *qsd = (const QueueSendData *) jitc_var_extra(v)->callback_data;
    const QueueRecvData *qrd = (const QueueRecvData *) jitc_var_extra(vr)->callback_data;

    bool virtual_queue = qsd->msg_types != 1;
    uint32_t state_offset = (qsd->msg_types + uint32_t(virtual_queue)) * CounterStride;

    fmt("\n"
        "l$u_wait_response:\n"
        "    ld.relaxed.gpu.global.u32 $v_tmp, [$v_phy_p+$u];\n"
        "    and.b32 $v_tmp, $v_tmp, $u;\n"
        "    setp.ge.u32 %v_ready, $v_tmp, $u;\n"
        "    @$v_ready bra l$u_wait_response;\n\n",
        v->reg_index,
        v, v, state_offset,
        v, v, qsd->block_size*2-1,
        v, v->reg_index
    );

    // Insert an efficient load sequence for the message response
    packet_memop(v, v, (uint32_t) qrd->vt.size(), nullptr,
                 qrd->vt.data(), true);

    fmt("    shr.u32 $v_count, $v_count, $u;\n"
        "    @$v_leader red.release.gpu.global.sub.u32 [$v_phy_p+$u], $v_count;\n",
        v, v, log2i_ceil(qsd->block_size)+1,
        v, v, state_offset, v);
}
