#include <stdint.h>

extern void jitc_var_vcall(const char *domain, uint32_t self, uint32_t n_inst,
                           uint32_t n_in, const uint32_t *in,
                           uint32_t n_out_nested, const uint32_t *out_nested,
                           const uint32_t *n_se, uint32_t *out);
