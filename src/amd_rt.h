#pragma once

#include "common.h"

#if defined(DRJIT_ENABLE_AMD)

#include <string>
#include <vector>

/// Per-scene HIPRT state. One instance is allocated by
/// ``jit_amd_configure_scene`` and wrapped in a Void JIT variable so that
/// pending/frozen trace nodes keep the application-owned HIPRT scene alive.
struct AMDScene {
    /// hiprtScene handle. Not retained by Dr.Jit; the caller owns it.
    void *scene = nullptr;

    /// Bit 0 = triangles, bit 3 = triangle backface culling. Other bits are
    /// reserved to match the Metal scene mask and rejected by the first pass.
    uint32_t geometry_types_mask = 0;

    /// Cached resource-handle variable for ``scene``.
    uint32_t scene_handle = 0;

    /// Optional hiprtFuncTable handle plus cached resource variable.
    void *func_table = nullptr;
    uint32_t func_table_handle = 0;

    /// Custom function-table layout used by hiprtBuildTraceKernels().
    uint32_t num_geom_types = 0;
    uint32_t num_ray_types = 0;
    std::vector<std::string> intersect_fns;
    std::vector<std::string> filter_fns;
    std::string device_source;

    /// Invoked once after this wrapper is destroyed, letting the application
    /// release the HIPRT scene and associated geometry buffers.
    void (*cleanup)(void *) = nullptr;
    void *cleanup_payload = nullptr;
};

extern uint32_t jitc_amd_configure_scene(void *scene,
                                         uint32_t geometry_types_mask);
extern uint32_t jitc_amd_configure_scene_ex(
    void *scene, void *func_table, uint32_t num_geom_types,
    uint32_t num_ray_types, const char **intersect_function_names,
    const char **filter_function_names, const char *device_source,
    uint32_t geometry_types_mask);
extern AMDScene *jitc_amd_get_scene(uint32_t scene_index);
extern uint32_t jitc_amd_scene_resource_handle(AMDScene *scene);
extern uint32_t jitc_amd_scene_func_table_handle(uint32_t scene_index);
extern uint32_t jitc_amd_scene_owner_handle(uint32_t scene_index);
extern void jitc_amd_ray_trace(uint32_t n_args, uint32_t *args,
                               uint32_t mask, uint32_t *out,
                               uint32_t n_out, uint32_t scene, int shadow);

#endif // defined(DRJIT_ENABLE_AMD)
