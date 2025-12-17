#include "internal.h"
#include "queue.h"
#include "var.h"
#include "log.h"
#include "util.h"

uint32_t jitc_queue_send(uint32_t buffer, uint32_t msg_types,
                         uint32_t msg_max_size, uint32_t block_size,
                         uint32_t blocks, int debug, uint32_t msg_id,
                         uint32_t n_indices, const uint32_t *indices,
                         QueueCallback *callback) {
    void *queue_buffer_p = nullptr;
    Ref queue_buffer = steal(jitc_var_data(buffer, false, &queue_buffer_p));

    Variable *queue_buffer_v = jitc_var(buffer);

    // We're already checking for these things in the Python bindings with
    // more specific error messages, so this is a safety net for other usage.
    if (queue_buffer_v->backend != (uint32_t) JitBackend::CUDA &&
        queue_buffer_v->type != (uint32_t) VarType::UInt32 &&
        queue_buffer_v->is_evaluated() && !queue_buffer_v->is_dirty()) {
        jitc_raise("jit_queue_send(): queue buffer must be an evaluated 32-bit unsigned integer CUDA array.");
    }

    Variable *msg_id_v = jitc_var(msg_id);
    if (msg_id_v->backend != (uint32_t) JitBackend::CUDA &&
        msg_id_v->type != (uint32_t) VarType::UInt32 && !msg_id_v->is_dirty()) {
        jitc_raise("jit_queue_send(): message ID must be a 32-bit unsigned integer CUDA array.");
    }

    if (msg_id_v->is_literal() && msg_id_v->literal >= (uint64_t) msg_types)
        jitc_raise("jit_queue_send(): the message ID %llu was not in the range "
                   "[0, %u)", (unsigned long long) msg_id_v->literal, msg_types);

    if (blocks == 0 || (blocks & (blocks-1)) != 0)
        jitc_raise("jit_queue_send(): block count (%u) must be a power of two.", blocks);

    if (block_size < 32 || (block_size & (block_size-1)) != 0)
        jitc_raise("jit_queue_send(): block size (%u) must be a power of two >= 32.", block_size);

    uint32_t queue_count = msg_types == 1 ? 1 : (msg_types + 1u);
    constexpr uint32_t CounterStride = 64;
    uint32_t ctrl_size = queue_count * (CounterStride + blocks);

    uint32_t size_expected =
        (msg_max_size * block_size * blocks) / sizeof(uint32_t) + ctrl_size;
    if (size_expected != queue_buffer_v->size)
        jitc_raise("jit_queue_send(): buffer has an unexpected size (expected "
                   "%u entries, got %u)", size_expected, queue_buffer_v->size);

    bool symbolic = (jitc_flags() & (uint32_t) JitFlag::SymbolicScope) != 0;
    uint32_t size = msg_id_v->size, nbytes = 0;

    drjit::unique_ptr<QueueSendData> qsd(new QueueSendData(n_indices));
    for (uint32_t i = 0; i < n_indices; ++i) {
        uint32_t index = indices[i];
        Variable *v = jitc_var(index);
        if (v->backend != (uint32_t) JitBackend::CUDA)
            break;

        if (v->size != size && size != 1 && v->size != 1)
            jitc_raise("jit_queue_send(): argument %u has an incompatible size (%u vs %u)", i, v->size, size);
        uint32_t bytes_i = type_size[v->type];

        if (nbytes % bytes_i != 0)
            jitc_raise("jit_queue_send(): argument %u is unaligned (byte offset %u, size %u)", i, nbytes, bytes_i);

        size = std::max(size, v->size);
        qsd->indices[i] = index;
        jitc_var_inc_ref(index, v);
        nbytes += bytes_i;
    }

    if (nbytes > msg_max_size)
        jitc_raise("jitc_queue_send(): message size (%u) exeeds the maximum "
                   "permitted by the queue (%u)", nbytes, msg_max_size);

    qsd->msg_types = msg_types;
    qsd->msg_max_size = msg_max_size;
    qsd->block_size = block_size;
    qsd->blocks = blocks;
    qsd->debug = debug;
    qsd->callback = callback;
    callback->inc_ref(callback);

    Ref queue_buffer_2 = steal(jitc_var_pointer(
        JitBackend::CUDA, queue_buffer_p, queue_buffer, false));

    jitc_new_scope(JitBackend::CUDA);

    Ref index = steal(jitc_var_new_node_2(
        JitBackend::CUDA, VarKind::QueueSend, VarType::UInt32, size, symbolic,
        queue_buffer_2, jitc_var(queue_buffer_2),
        msg_id, jitc_var(msg_id), (uintptr_t) qsd.get()));

    // Free resources when this variable is destroyed
    auto free_callback = [](uint32_t /*index*/, int free, void *ptr) {
        if (free)
            delete (QueueSendData *) ptr;
    };

    jitc_var_set_callback(index, free_callback, qsd.release(), true);

    jitc_log(Debug,
             "jit_queue_send(buffer=r%u, msg_types=%u, msg_max_size=%u, "
             "block_size=%u, blocks=%u, debug=%i): r%u",
             buffer, msg_types, msg_max_size, block_size, blocks, debug, (uint32_t) index);

    return index.release();
}

void jitc_queue_recv(uint32_t ticket, uint32_t n_indices, const VarType *recv_vt, uint32_t *recv_idx) {
    Variable *ticket_v = jitc_var(ticket);

    if (ticket_v->kind != (uint32_t) VarKind::QueueSend ||
        ticket_v->backend != (uint32_t) JitBackend::CUDA ||
        ticket_v->type != (uint32_t) VarType::UInt32)
        jitc_raise("jit_queue_recv(): invalid input!");

    drjit::unique_ptr<QueueRecvData> qrd(new QueueRecvData(n_indices));
    for (uint32_t i = 0; i < n_indices; ++i)
        qrd->vt[i] = recv_vt[i];

    bool symbolic = ticket_v->symbolic;
    uint32_t size = ticket_v->size;

    jitc_new_scope(JitBackend::CUDA);

    Ref index = steal(
        jitc_var_new_node_1(JitBackend::CUDA, VarKind::QueueRecv, VarType::Void,
                            size, symbolic, ticket, jitc_var(ticket)));

    // Free resources when this variable is destroyed
    auto free_callback = [](uint32_t /*index*/, int free, void *ptr) {
        if (free)
            delete (QueueRecvData *) ptr;
    };

    jitc_var_set_callback(index, free_callback, qrd.release(), true);

    for (size_t i = 0; i < n_indices; ++i)
        recv_idx[i] =
            jitc_var_new_node_1(JitBackend::CUDA, VarKind::Extract, recv_vt[i],
                                size, symbolic, index, jitc_var(index), i);
}
