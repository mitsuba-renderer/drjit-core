#include <stdint.h>

extern void jitc_var_loop(const char *name, uint32_t cond, uint32_t n,
                          const uint32_t *in, const uint32_t *out_body,
                          uint32_t se_offset, uint32_t *out,
                          int check_invariant, uint8_t *invariant);
