#include "drjit-core/jit.h"
#include "test.h"

TEST_BOTH(01_basic_replay) {
    Recording *recording;

    jit_log(LogLevel::Info, "Recording:");
    {
        UInt32 r(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        UInt32 ref(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

        uint32_t inputs[] = {
            r.index()
        };

        jit_record_start(Backend, inputs, 1);

        UInt32 result = r + 1;
        result.eval();

        uint32_t outputs[] = {
            result.index()
        };

        recording = jit_record_stop(Backend, outputs, 1);
        
        jit_log(LogLevel::Info, "result: %s", jit_var_str(outputs[0]));
        jit_assert(jit_var_all(jit_var_eq(ref.index(), outputs[0])));
    }


    jit_log(LogLevel::Info, "Replay:");
    {
        UInt32 r(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        UInt32 ref(2, 3, 4, 5, 6, 7, 8, 9, 10, 11);

        uint32_t inputs[] = {
            r.index()
        };
        uint32_t outputs[1];

        jit_record_replay(recording, inputs, outputs);

        jit_log(LogLevel::Info, "result: %s", jit_var_str(outputs[0]));
        jit_assert(jit_var_all(jit_var_eq(ref.index(), outputs[0])));
    }

    jit_record_destroy(recording);
}
