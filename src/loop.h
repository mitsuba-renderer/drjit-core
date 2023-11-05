#include <stdint.h>

struct LoopData {
    std::string name;
    size_t size;
    uint32_t loop_start;
    std::vector<uint32_t> outer_inputs;
    std::vector<uint32_t> inner_inputs;
    std::vector<uint32_t> inner_outputs;
    std::vector<WeakRef> outer_outputs;
    bool symbolic;
    bool retry;

    LoopData(const char *name, uint32_t loop_start, size_t size, bool symbolic)
        : name(name), size(size), loop_start(loop_start), symbolic(symbolic),
          retry(false) {
        outer_inputs.reserve(size);
        inner_inputs.reserve(size);
        inner_outputs.reserve(size);
        outer_outputs.reserve(size);
    }

    ~LoopData() {
        jitc_var_dec_ref(loop_start);
        for (uint32_t index: outer_inputs)
            jitc_var_dec_ref(index);
        for (uint32_t index: inner_inputs)
            jitc_var_dec_ref(index);
        for (uint32_t index: inner_outputs)
            jitc_var_dec_ref(index);
    }
};

extern uint32_t jitc_var_loop_start(const char *name, size_t n_indices,
                                    uint32_t *indices);
extern uint32_t jitc_var_loop_cond(uint32_t loop, uint32_t active);
extern bool jitc_var_loop_end(uint32_t loop, uint32_t cond, uint32_t *indices);
