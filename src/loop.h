#include <stdint.h>

extern uint32_t jitc_var_loop(const char *name, uint32_t loop_var_start,
                              uint32_t loop_var_cond, uint32_t n,
                              const uint32_t *in, const uint32_t *out_body,
                              uint32_t se_offset, uint32_t *out,
                              int check_invariant, uint8_t *invariant);
