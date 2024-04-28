#include "drjit-core/jit.h"
#include "test.h"

TEST_BOTH(01_basic_replay) {
    UInt32 r(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);

    uint32_t inputs[] = {
        r.index()
    };

    jit_record_start(Backend, inputs, 1);

    UInt32 result = r + 1;
    result.eval();

    uint32_t outputs[] = {
        result.index()
    };

    Recording *record = jit_record_stop(Backend, outputs, 1);


    {
        UInt32 r2(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        jit_log(LogLevel::Info, "r2: %s", jit_var_str(r2.index()));
        
        uint32_t inputs[] = {
            r2.index()
        };
        uint32_t outputs[1];
        
        jit_record_replay(record, inputs, outputs);

        jit_log(LogLevel::Info, "result: %s", jit_var_str(outputs[0]));
    }

    jit_record_destroy(record);
}
