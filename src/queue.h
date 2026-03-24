#pragma once

struct QueueSendData {
    std::vector<uint32_t> indices;
    uint32_t msg_types{0};
    uint32_t msg_max_size{0};
    uint32_t batches{0};
    uint32_t batch_size{0};
    uint32_t debug{0};
    QueueCallback *callback{nullptr};

    QueueSendData(size_t n_indices) : indices(n_indices, 0) { }
    ~QueueSendData() {
        for (uint32_t index: indices)
            jitc_var_dec_ref(index);
        if (callback)
            callback->dec_ref(callback);
    }
};

struct QueueRecvData {
    std::vector<VarType> vt;
    QueueRecvData(size_t n_indices) : vt(n_indices, VarType::Void) { }
};

extern uint32_t jitc_queue_send(uint32_t buffer, uint32_t msg_types,
                                uint32_t msg_max_size, uint32_t block_size,
                                uint32_t blocks, int debug, uint32_t msg_id,
                                uint32_t n_indices, const uint32_t *indices,
                                QueueCallback *callback);

extern void jitc_queue_recv(uint32_t ticket, uint32_t n_indices,
                            const VarType *recv_vt, uint32_t *recv_idx);
