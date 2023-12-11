#include <stdint.h>

struct CondData {
    std::string name;
    std::vector<uint32_t> indices_t, se_t;
    std::vector<uint32_t> indices_f, se_f;
    std::vector<WeakRef> indices_out;
    uint32_t labels[2] { };
    uint32_t se_offset = 0;

    ~CondData() {
        for (uint32_t index: indices_t)
            jitc_var_dec_ref(index);
        for (uint32_t index: indices_f)
            jitc_var_dec_ref(index);
        for (uint32_t index: se_t)
            jitc_var_dec_ref(index);
        for (uint32_t index: se_f)
            jitc_var_dec_ref(index);
    }
};

extern uint32_t jitc_var_cond_start(const char *name, bool symbolic,
                                    uint32_t cond_t, uint32_t cond_f);
extern uint32_t jitc_var_cond_append(uint32_t index, const uint32_t *rv,
                                     size_t count);
extern void jitc_var_cond_end(uint32_t index, uint32_t *rv_out);
