#include <stdint.h>

extern void jitc_embree_trace(uint32_t func, uint32_t context, uint32_t scene,
                              int occlude, const uint32_t *in, uint32_t *out);
