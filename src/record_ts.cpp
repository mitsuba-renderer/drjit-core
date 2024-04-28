#include "record_ts.h"
#include "common.h"
#include "drjit-core/jit.h"
#include "internal.h"
#include "log.h"

void jitc_record_start(JitBackend backend, const uint32_t *inputs,
                       uint32_t n_inputs) {

    ThreadState *ts = thread_state(backend);
    RecordThreadState *record_ts = new RecordThreadState(ts);

    if (backend == JitBackend::CUDA) {
        thread_state_cuda = record_ts;
    } else {
        thread_state_llvm = record_ts;
    }

    for (uint32_t i = 0; i < n_inputs; ++i) {
        record_ts->set_input(inputs[i]);
    }
}
Recording *jitc_record_stop(JitBackend backend, const uint32_t *outputs,
                                    uint32_t n_outputs) {
    ThreadState *ts = thread_state(backend);
    RecordThreadState *record_ts = dynamic_cast<RecordThreadState *>(ts);

    for (uint32_t i = 0; i < n_outputs; ++i) {
        record_ts->set_output(outputs[i]);
    }

    if (backend == JitBackend::CUDA) {
        thread_state_cuda = record_ts->internal;
    } else {
        thread_state_llvm = record_ts->internal;
    }
    return new Recording(record_ts->recording);
}
