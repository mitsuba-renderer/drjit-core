/*
   Implementation of the client part of the dr.Queue communication primitive,
   which is responsible for JIT-compiling two key operations into user code:
   sending a message, and waiting for the response.

   The implementation uses different code paths depending on whether the queue
   services a single or multiple message types. In the latter case, the
   implementation uses a *virtual* queue per message type backed by a single
   physical queue. This is important because the queue has substantial
   memory requirements to unlock sufficient parallelism. Allocating a separate
   queue per message type (of which there can be many) would be too costly. The
   ``logical_queue`` variable indicates whether the indirection through a
   virtual queue is used.

   Messages are grouped into *batches*. The server processes batches atomically.
   Each batch must have a uniform message type. Enqueuing a message entails
   figuring out the right target batch in the physical queue. In the simple case
   (``logical_queue==false`), the queue data structure is organized as follows:

   The client uses warp communication primitives to locally aggregate
   information and minimize the number of atomic memory transactions needed to
   update the queue data structure.

   1. Head counter: 32-bit queue head counter, padded to ``CounterStride``
      bytes. Each submitted message increases this counter.

   2. Queue state: ``batches`` 32-bit values identifying the state of each
      message batch.

      - bit 0..``log2i(batch_size)``: number of messages to be processed.
      - remainder: generation counter. Increases each each time a message
        or its reply is ready to be picked up on the other end.

   3. Message body (``batches`` * ``batch_size`` * ``max_message_size`` bytes).
      This memory region is used for both the message and its reply.

*/

#include "eval.h"
#include "cuda_queue.h"
#include "cuda_eval.h"
#include "internal.h"
#include "queue.h"
#include "var.h"

/// Byte stride between head/tail/virtual head pointers to avoid false sharing
constexpr uint32_t CounterStride = 64;

/// Helper function to dump internal state for debugging
void jitc_device_printf(const char *str, const Variable *v,
                        uint32_t count, ...) {
    // Build format string: "%03i: <str>\n\0", zero-padded to 4-byte boundary
    size_t len = strlen("%03i: ") + strlen(str) + 1;
    size_t len_padded = ceil_div(len, 4)*4;
    std::unique_ptr<char[]> full_str(new char[len_padded]{});

    size_t pos = 0;
    for (const char *p = "%03i: "; *p; p++)
        full_str[pos++] = *p;
    for (const char *p = str; *p; p++)
        full_str[pos++] = *p;

    uint32_t arg_size = (count + 1) * 4;

    put("    {\n"
        "        .extern .func (.param .b32 rv) vprintf "
            "(.param .b64 fmt, .param .b64 args);\n");
    fmt("        .local .align 4 .b8 __fmt[$u];\n"
        "        .local .align 4 .b8 __args[$u];\n",
        (uint32_t) len_padded, arg_size);
    put("        .reg.b32 %_pf_tmp;\n"
        "        .reg.b64 %_pf_fmt, %_pf_args;\n");

    // Store format string to local memory, reading 4 bytes at a time (LE)
    for (uint32_t i = 0; i < (uint32_t) len_padded; i += 4) {
        uint32_t word;
        memcpy(&word, full_str.get() + i, 4);
        fmt("        mov.b32 %_pf_tmp, 0x$x;\n"
            "        st.local.b32 [__fmt+$u], %_pf_tmp;\n", word, i);
    }

    // Store thread index (%r0) as first argument
    put("        st.local.b32 [__args], %r0;\n");

    va_list args;
    va_start(args, count);
    for (uint32_t i = 0; i < count; i++) {
        const char *suffix = va_arg(args, const char *);
        fmt("        st.local.b32 [__args+$u], $v_$s;\n",
            (i + 1) * 4, v, suffix);
    }
    va_end(args);

    // Convert to generic addresses and call vprintf
    put("        cvta.local.u64 %_pf_fmt, __fmt;\n"
        "        cvta.local.u64 %_pf_args, __args;\n"
        "        {\n"
        "            .param .b64 _fmt;\n"
        "            .param .b64 _args;\n"
        "            .reg.b32 %_pf_rv;\n"
        "            st.param.b64 [_fmt], %_pf_fmt;\n"
        "            st.param.b64 [_args], %_pf_args;\n"
        "            call.uni (%_pf_rv), vprintf, (_fmt, _args);\n"
        "        }\n"
        "    }\n");
}

void elect(const Variable *v, const char *dst_id, const char *dst_mask, const char *peers) {
#if 1
    fmt("    elect.sync $v_$s|$v_$s, $v_$s;\n", v, dst_id, v, dst_mask, v, peers);
#else
    //"    mov.b32 $v_lane_id, %laneid;\n"
    fmt(
        "    bfind.u32 $v_$s, $v_$s;\n"
        "    setp.eq.u32 $v_$s, $v_$s, $v_lane_id;\n",
        v, dst_id, v, peers,
        v, mask, v, peers
    );
#endif
}

void jitc_cuda_render_queue_send(Variable *v,
                                 Variable *queue_buffer,
                                 Variable *msg_type) {
    const QueueSendData *qsd = (const QueueSendData *) jitc_var_extra(v)->callback_data;
    bool logical_queue = false;

    const uint32_t batch_size      = qsd->batch_size,
                   batches         = qsd->batches,
                   batch_mask      = batches - 1,
                   batch_size_mask = batch_size - 1,
                   batch_size_bits = log2i_ceil(batch_size),
                   batches_bits    = log2i_ceil(batches),
                   version_mask    = ((uint32_t) -1) * batches*batch_size;

    // Offsets within the queue buffer
    uint32_t state_offset = (qsd->msg_types + uint32_t(logical_queue)) * CounterStride,
             state_size   = qsd->batches * (uint32_t) sizeof(uint32_t),
             data_offset  = state_offset + state_size * (qsd->msg_types + uint32_t(logical_queue));

    // ================================================================
    fmt("\n"
        "    // Enqueue message: msg_types=$u, batches=$u [$u bits], batch_size=$u [$u bits]\n",
        qsd->msg_types, qsd->batches, batches_bits, qsd->batch_size, batch_size_bits);

    fmt( "    // 1. Mask for opportunistic warp-level cooporation\n"
        "    .reg.b32 $v_active, $v_peers;\n"
        "    activemask.b32 $v_active;\n",
        v,
        v,
        v
    );

    // ================================================================

    fmt(
        "\n"
        "    // 2. Identify target queue and peer threads\n"
        "    .reg.b32 $v_queue_id;\n",
        v
    );
    if (logical_queue) {
        fmt("    .reg.b32 $v_peers;\n"
            "    add.u32 $v_queue_id, $v, 1;\n"
            "    match.any.sync.b32 $v_peers, $v_queue_id, $v_active;\n",
            v,
            v, msg_type,
            v, v, v);
    } else {
        fmt("    mov.u32 $v_queue_id, 0;\n"
            "    mov.u32 $v_peers, $v_active;\n",
            v,
            v, v);
    }

    // ================================================================

    fmt("\n"
        "    // 3. Elect a leader and count the messages in each peer group\n"
        "    .reg.b32 $v_leader_id, $v_count;\n"
        "    .reg.pred $v_leader_mask;\n",
        v, v, v
    );

    elect(v, "leader_id", "leader_mask", "peers");
    fmt("    popc.b32 $v_count, $v_$s;\n",
        v, v, "peers");

    // ================================================================

    fmt("\n"
        "    // 4. The leader reserves space by bumping the logical queue head pointer\n"
        "    .reg.b64 $v_head;\n"
        "    .reg.b32 $v_offset;\n"
        "    mad.wide.u32 $v_head, $v_queue_id, $u, $v;\n"
        "    @$v_leader_mask atom.relaxed.gpu.global.add.u32 $v_offset, [$v_head], $v_count;\n",
        v,
        v,
        v, v, CounterStride, queue_buffer,
        v, v, v, v);

    fmt("\n"
        "    // 5. Generate indices for all peers based on the leader's response\n"
        "    .reg.b32 $v_lane_lt, $v_peers_lt, $v_offset_local;\n"
        "    mov.b32 $v_lane_lt, %lanemask_lt;\n"
        "    and.b32 $v_peers_lt, $v_lane_lt, $v_peers;\n"
        "    popc.b32 $v_offset_local, $v_peers_lt;\n"
        "    shfl.sync.idx.b32 $v_offset, $v_offset, $v_leader_id, 31, $v_active;\n"
        "    add.u32 $v_offset, $v_offset, $v_offset_local;\n",
        v, v, v,
        v,
        v, v, v,
        v, v,
        v, v, v, v,
        v, v, v);

    fmt("\n"
        "    // 6. Determine the logical batch, generation, and offset within the batch\n"
        "    .reg.b32 $v_log_batch, $v_log_gen;\n"
        "    shr.u32 $v_log_batch, $v_offset, $u;\n" // logical batch index
        "    and.b32 $v_log_batch, $v_log_batch, 0x$x;\n"
        "    and.b32 $v_log_gen, $v_offset, 0x$x;\n"
        "    and.b32 $v_offset, $v_offset, 0x$x;\n",   // offset within batch (logical + physical)
        v, v,
        v, v, batch_size_bits,
        v, v, batch_mask,
        v, v, version_mask,
        v, v, batch_size_mask);

    // ================================================================

    fmt("\n"
        "    // 7. Simple case: there is no difference between logical/physical queues\n"
        "    .reg.b32 $v_phy_batch;\n"
        "    .reg.b64 $v_phy_p;\n"
        "    mad.wide.u32 $v_phy_p, $v_phy_batch, 4, $v;\n"
        "    mov.b32 $v_phy_batch, $v_log_batch;\n",
        v,
        v,
        v, v, queue_buffer,
        v, v);

    fmt("\n"
        "    // 8. Designate a new leader based on the physical batch ID\n"
        "    match.any.sync.b32 $v_peers, $v_phy_batch, $v_active;\n"
        "    popc.b32 $v_count, $v_peers;\n",
        v, v, v,
        v, v
    );

    elect(v, "leader_id", "leader_mask", "peers");


    // ================================================================

    fmt("\n"
        "    // 9. Ensure that the previous message batch was fully processed.\n"
        "    .reg.b32 $v_state, $v_exp_state;\n"
        "    .reg.pred $v_ready;\n"
        "    shr.b32 $v_exp_state, $v_log_gen, $u;\n" // generation ID at bit position batch_bits+1
        "    @!$v_leader_mask bra l$u_sync;\n"
        "\n"
        "l$u_wait_phy:\n"
        "    ld.relaxed.gpu.global.b32 $v_state, [$v_phy_p+$u];\n"
        "    and.b32 $v_state, $v_state, 0x$x;\n"
        "    setp.eq.u32 $v_ready, $v_state, $v_log_gen;\n"
        "    @!$v_ready bra l$u_wait_phy;\n"
        "\n"
        "l$u_sync:\n"
        "    bar.warp.sync $v_active;\n", // wait for other threads
        v, v,
        v,
        v, v, batches_bits-1,
        v, v->reg_index,
        v->reg_index,
        v, v, state_offset,
        v, v, ~batch_size_mask,
        v, v, v,
        v, v->reg_index,
        v->reg_index, v);

    fmt("    // 10. Inform the server that the message has been written\n"
        "    @$v_leader_mask red.release.gpu.global.add.u32 [$v_phy_p+$u], $v_count;\n",
        v, v, state_offset, v);

    queue_callbacks.push_back(qsd->callback);
}

void jitc_cuda_render_queue_recv(Variable *vr,
                                 Variable *v) {
    const QueueSendData *qsd = (const QueueSendData *) jitc_var_extra(v)->callback_data;
    const QueueRecvData *qrd = (const QueueRecvData *) jitc_var_extra(vr)->callback_data;

    const std::vector<VarType> &vt = qrd->vt;
    uint32_t n = (uint32_t) vt.size();

    // Declare output variables of recv operation
    bool uniform_tp = true;
    for (auto value: vt)
        uniform_tp &= value == vt[0];

    if (uniform_tp) {
        fmt("    .reg.b$u $v_out_<$u>;\n", type_size[(int) vt[0]] * 8, vr, n);
    } else {
        for (uint32_t i = 0; i < n; i++)
            fmt("    .reg.b$u $v_out_$u;\n", type_size[(int) vt[i]] * 8, vr, i);
    }
}
