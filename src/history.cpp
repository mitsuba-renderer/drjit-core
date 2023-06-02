
/// ==========================================================================

KernelHistory::KernelHistory() : m_data(nullptr), m_size(0), m_capacity(0) { }

KernelHistory::~KernelHistory() { free(m_data); }

void KernelHistory::append(const KernelHistoryEntry &value) {
    /* Expand kernel history buffer if necessary. There should always be
       enough memory for an additional end-of-list marker at the end */

    if (m_size + 2 > m_capacity) {
        m_capacity = (m_size + 2) * 2;
        void *tmp = malloc_check(m_capacity * sizeof(KernelHistoryEntry));
        memcpy(tmp, m_data, m_size * sizeof(KernelHistoryEntry));
        free(m_data);
        m_data = (KernelHistoryEntry *) tmp;
    }

    m_data[m_size++] = value;
    memset(m_data + m_size, 0, sizeof(KernelHistoryEntry));
}

KernelHistoryEntry *KernelHistory::get() {
    KernelHistoryEntry *data = m_data;

    for (size_t i = 0; i < m_size; i++) {
        KernelHistoryEntry &k = data[i];
        if (k.backend == JitBackend::CUDA) {
            cuEventElapsedTime(&k.execution_time,
                               (CUevent) k.event_start,
                               (CUevent) k.event_end);
            cuEventDestroy((CUevent) k.event_start);
            cuEventDestroy((CUevent) k.event_end);
            k.event_start = k.event_end = 0;
        } else {
            k.execution_time = task_time((Task *) k.task);
            task_release((Task *) k.task);
            k.task = nullptr;
        }
    }

    m_data = nullptr;
    m_size = m_capacity = 0;

    return data;
}

void KernelHistory::clear() {
    if (m_size == 0)
        return;

    for (size_t i = 0; i < m_size; i++) {
        KernelHistoryEntry &k = m_data[i];
        if (k.backend == JitBackend::CUDA) {
            cuEventDestroy((CUevent) k.event_start);
            cuEventDestroy((CUevent) k.event_end);
        } else {
            task_release((Task *) k.task);
        }
    }

    free(m_data);
    m_data = nullptr;
    m_size = m_capacity = 0;
}

