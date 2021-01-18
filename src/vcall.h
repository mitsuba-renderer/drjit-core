#include <stdint.h>

extern void jitc_var_vcall(const char *domain, uint32_t self, uint32_t mask,
                           uint32_t n_inst, const uint32_t *inst_id,
                           uint32_t n_in, const uint32_t *in,
                           uint32_t n_out_nested, const uint32_t *out_nested,
                           const uint32_t *n_se, uint32_t *out);

extern VCallBucket *jitc_var_vcall_reduce(JitBackend backend,
                                          const char *domain, uint32_t index,
                                          uint32_t *bucket_count_out);

