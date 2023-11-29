#include <stdint.h>

struct LoopData {
    std::string name;
    size_t size;
    uint32_t loop_start;
    std::vector<uint32_t> outer_in;
    std::vector<uint32_t> inner_in;
    std::vector<uint32_t> inner_out;
    std::vector<WeakRef> outer_out;
    bool symbolic;
    bool retry;

    LoopData(const char *name, uint32_t loop_start, size_t size, bool symbolic)
        : name(name), size(size), loop_start(loop_start), symbolic(symbolic),
          retry(false) {
        outer_in.reserve(size);
        inner_in.reserve(size);
        inner_out.reserve(size);
        outer_out.reserve(size);
    }

    ~LoopData() {
        jitc_var_dec_ref(loop_start);
        for (uint32_t index: outer_in)
            jitc_var_dec_ref(index);
        for (uint32_t index: inner_in)
            jitc_var_dec_ref(index);
        for (uint32_t index: inner_out)
            jitc_var_dec_ref(index);
    }
};

extern uint32_t jitc_var_loop_start(const char *name, bool symbolic,
                                    size_t n_indices, uint32_t *indices);
extern uint32_t jitc_var_loop_cond(uint32_t loop, uint32_t active);
extern bool jitc_var_loop_end(uint32_t loop, uint32_t cond, uint32_t *indices,
                              uint32_t checkpoint);
