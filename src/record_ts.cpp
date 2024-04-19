#include "record_ts.h"
#include "common.h"
#include "drjit-core/jit.h"
#include "internal.h"
#include "log.h"

void jitc_record_start(JitBackend backend, uint32_t *inputs, size_t n_inputs){
    
    ThreadState *ts = thread_state(backend);
    RecordThreadState* record_ts = new RecordThreadState(ts);
    
    if (backend == JitBackend::CUDA) {
        thread_state_cuda = record_ts;
    } else {
        thread_state_llvm = record_ts;
    }
    
    for (uint32_t i = 0; i < n_inputs; ++i){
        Variable* variable = jitc_var(inputs[i]);

        // TODO: asserts
        
        record_ts->set_input(variable->data);
    }
}
RecordThreadState *jitc_record_stop(JitBackend backend, uint32_t *outputs, size_t n_outputs){
    ThreadState *ts = thread_state(backend);
    RecordThreadState *record_ts = dynamic_cast<RecordThreadState*>(ts);

    for (uint32_t i = 0; i < n_outputs; ++i){
        Variable *variable = jitc_var(outputs[i]);
        record_ts->set_output(variable->data);
    }
    
    if (backend == JitBackend::CUDA) {
        thread_state_cuda = record_ts->internal;
    } else {
        thread_state_llvm = record_ts;
    }
    return record_ts;
}

void jitc_test_record(JitBackend backend){
    
    uint32_t v0 = jit_var_counter(backend, 100);
    
    jit_var_schedule(v0);
    jit_eval();

    uint32_t inputs[] = {
        v0
    };
    
    jitc_record_start(backend, inputs, 1);

    uint32_t v1 = jit_var_u32(backend, 1);

    uint32_t v2 = jit_var_add(v0, v1);

    jit_var_schedule(v2);
    jit_eval();
    
    uint32_t outputs[] = {
        v2
    };
    
    RecordThreadState* record_ts = jitc_record_stop(backend, outputs, 1);

    jitc_log(LogLevel::Info,"Recording: v2 = %s", jit_var_str(v2));

    {
        uint32_t inputs[] = {
            v0
        };
        std::vector<uint32_t> outputs = record_ts->replay(inputs, 1);
        
        uint32_t v2 = outputs[0];
        
        // Why is the lock never released by jitc_malloc?
        unlock_guard guard(state.lock);
        jitc_log(LogLevel::Info, "Replay finished");
        jitc_log(LogLevel::Info,"Replay: v2 = %s", jit_var_str(v2));
    }
}
