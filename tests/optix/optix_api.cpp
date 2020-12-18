#include <enoki-jit/optix.h>
#define OPTIX_API_IMPL
#include "optix_api.h"

void init_optix_api() {
    jitc_optix_context(); // Ensure OptiX is initialized

    #define L(name) name = (decltype(name)) jitc_optix_lookup(#name);

    L(optixAccelComputeMemoryUsage);
    L(optixAccelBuild);
    L(optixAccelCompact);
    L(optixModuleCreateFromPTX);
    L(optixModuleDestroy)
    L(optixProgramGroupCreate);
    L(optixProgramGroupDestroy)
    L(optixSbtRecordPackHeader);

    #undef L
}
