/*
    src/amd_rt.cpp -- HIPRT scene and TraceRay plumbing for the AMD backend
*/

#include "amd_rt.h"

#if defined(DRJIT_ENABLE_AMD)

#include "internal.h"
#include "log.h"
#include "trace.h"
#include "var.h"

#include <algorithm>
#include <array>

uint32_t jitc_amd_configure_scene_ex(
        void *scene_handle, void *func_table, uint32_t num_geom_types,
        uint32_t num_ray_types, const char **intersect_function_names,
        const char **filter_function_names, const char *device_source,
        uint32_t geometry_types_mask) {
    jitc_log(InfoSym,
             "jit_amd_configure_scene(scene=" DRJIT_PTR ", func_table="
             DRJIT_PTR ", geom_types=%u, ray_types=%u, geom_mask=%u)",
             (uintptr_t) scene_handle, (uintptr_t) func_table,
             num_geom_types, num_ray_types, geometry_types_mask);

    AMDScene *scene = new AMDScene();
    scene->scene = scene_handle;
    scene->func_table = func_table;
    scene->geometry_types_mask = geometry_types_mask;
    scene->num_geom_types = num_geom_types;
    scene->num_ray_types = num_ray_types;

    uint32_t fn_count = num_geom_types * num_ray_types;
    scene->intersect_fns.reserve(fn_count);
    scene->filter_fns.reserve(fn_count);
    for (uint32_t i = 0; i < fn_count; ++i) {
        scene->intersect_fns.emplace_back(
            intersect_function_names && intersect_function_names[i]
                ? intersect_function_names[i] : "");
        scene->filter_fns.emplace_back(
            filter_function_names && filter_function_names[i]
                ? filter_function_names[i] : "");
    }
    if (device_source)
        scene->device_source = device_source;

    uint32_t index =
        jitc_var_new_node_0(JitBackend::AMD, VarKind::Nop, VarType::Void,
                            1, 0, (uintptr_t) scene);

    auto callback = [](uint32_t, int free, void *ptr) {
        if (!free)
            return;

        AMDScene *scene = (AMDScene *) ptr;
        jitc_log(InfoSym,
                 "jit_amd_configure_scene(): freeing AMDScene(scene="
                 DRJIT_PTR ")",
                 (uintptr_t) scene->scene);

        if (scene->scene_handle)
            jitc_var_dec_ref(scene->scene_handle);
        if (scene->func_table_handle)
            jitc_var_dec_ref(scene->func_table_handle);

        void (*cleanup)(void *) = scene->cleanup;
        void *cleanup_payload = scene->cleanup_payload;
        delete scene;

        if (cleanup)
            cleanup(cleanup_payload);
    };

    jitc_var_set_callback(index, callback, scene, true);
    return index;
}

uint32_t jitc_amd_configure_scene(void *scene_handle,
                                  uint32_t geometry_types_mask) {
    return jitc_amd_configure_scene_ex(scene_handle, nullptr, 0, 0,
                                       nullptr, nullptr, nullptr,
                                       geometry_types_mask);
}

AMDScene *jitc_amd_get_scene(uint32_t scene_index) {
    Variable *v = scene_index ? jitc_var(scene_index) : nullptr;
    if (!v || (VarKind) v->kind != VarKind::Nop ||
        (VarType) v->type != VarType::Void)
        jitc_fail("jitc_amd_get_scene(): r%u does not wrap a HIPRT scene.",
                  scene_index);
    return (AMDScene *) v->literal;
}

uint32_t jitc_amd_scene_resource_handle(AMDScene *scene) {
    if (!scene || !scene->scene)
        return 0;

    if (!scene->scene_handle) {
        uint32_t backing =
            jitc_var_mem_map(JitBackend::AMD, VarType::UInt64,
                             (void *) scene, 1, /* free = */ 0);
        scene->scene_handle =
            jitc_var_resource_pointer(backing, ResourceKind::Accel);
        jitc_var_dec_ref(backing);
    }

    jitc_var_inc_ref(scene->scene_handle);
    return scene->scene_handle;
}

static uint32_t jitc_amd_scene_func_table_resource_handle(AMDScene *scene) {
    if (!scene || !scene->func_table)
        return 0;

    if (!scene->func_table_handle) {
        uint32_t backing =
            jitc_var_mem_map(JitBackend::AMD, VarType::UInt64,
                             (void *) scene, 1, /* free = */ 0);
        scene->func_table_handle =
            jitc_var_resource_pointer(backing, ResourceKind::IFT);
        jitc_var_dec_ref(backing);
    }

    jitc_var_inc_ref(scene->func_table_handle);
    return scene->func_table_handle;
}

uint32_t jitc_amd_scene_owner_handle(uint32_t scene_index) {
    AMDScene *scene = jitc_amd_get_scene(scene_index);
    return jitc_var_mem_map(JitBackend::AMD, VarType::UInt64,
                            (void *) scene, 1, /* free = */ 0);
}

uint32_t jitc_amd_scene_func_table_handle(uint32_t scene_index) {
    AMDScene *scene = jitc_amd_get_scene(scene_index);
    if (!scene->func_table)
        return 0;
    return jitc_var_mem_map(JitBackend::AMD, VarType::UInt64,
                            scene->func_table, 1, /* free = */ 0);
}

void jitc_amd_ray_trace(uint32_t n_args, uint32_t *args,
                        uint32_t mask, uint32_t *out,
                        uint32_t n_out, uint32_t scene, int shadow) {
    if (n_args != 8)
        jitc_raise("jit_amd_ray_trace(): expected 8 ray arguments, got %u.",
                   n_args);
    if (n_out != 8)
        jitc_raise("jit_amd_ray_trace(): expected 8 outputs, got %u.", n_out);
    if (!scene)
        jitc_raise("jit_amd_ray_trace(): a valid scene_index "
                   "(returned by jit_amd_configure_scene) is required.");

    if (jitc_var_type(scene) != VarType::Void)
        jitc_raise("jit_amd_ray_trace(): type mismatch for scene argument!");

    std::array<Ref, 8> promoted_literals;
    uint32_t trace_args[8];

    uint32_t size = 0;
    bool symbolic = false;
    for (uint32_t i = 0; i < n_args; ++i) {
        const Variable *vi = jitc_var(args[i]);
        if ((VarType) vi->type != VarType::Float32)
            jitc_raise("jit_amd_ray_trace(): type mismatch for arg. %u "
                       "(got %s, expected Float32).", i,
                       type_name[vi->type]);

        trace_args[i] = args[i];
        if (vi->is_literal() && vi->size == 1) {
            promoted_literals[i] =
                steal(jitc_var_literal(JitBackend::AMD, VarType::Float32,
                                        &vi->literal, 1, /* eval = */ 1));
            trace_args[i] = promoted_literals[i];
            vi = jitc_var(trace_args[i]);
        }

        size = std::max(size, vi->size);
        symbolic |= (bool) vi->symbolic;
    }

    const Variable *vm = jitc_var(mask);
    size = std::max(size, vm->size);
    symbolic |= (bool) vm->symbolic;

    if (size == 0)
        size = 1;

    Ref valid = steal(jitc_var_mask_apply(mask, size));

    TraceData *td = new TraceData();
    td->shadow = shadow != 0;
    td->indices.reserve(n_args);
    for (uint32_t i = 0; i < n_args; ++i) {
        td->indices.push_back(trace_args[i]);
        jitc_var_inc_ref(trace_args[i]);
    }

    AMDScene *scene_obj = jitc_amd_get_scene(scene);
    Ref scene_h = steal(jitc_amd_scene_resource_handle(scene_obj));
    if (!scene_h)
        jitc_raise("jit_amd_ray_trace(): scene has no HIPRT handle.");
    Ref func_h = steal(jitc_amd_scene_func_table_resource_handle(scene_obj));

    Ref trace;
    if (func_h) {
        trace = steal(jitc_var_new_node_4(
            JitBackend::AMD, VarKind::TraceRay, VarType::Void, size, symbolic,
            valid, jitc_var(valid), scene, jitc_var(scene),
            scene_h, jitc_var(scene_h), func_h, jitc_var(func_h),
            (uintptr_t) td));
    } else {
        trace = steal(jitc_var_new_node_3(
            JitBackend::AMD, VarKind::TraceRay, VarType::Void, size, symbolic,
            valid, jitc_var(valid), scene, jitc_var(scene),
            scene_h, jitc_var(scene_h), (uintptr_t) td));
    }

    jitc_var_set_callback(
        trace,
        [](uint32_t, int free, void *ptr) {
            if (free)
                delete (TraceData *) ptr;
        },
        td, true);

    VarType out_types[8] = {
        VarType::Bool,    // valid
        VarType::Float32, // distance
        VarType::Float32, // bary_u
        VarType::Float32, // bary_v
        VarType::UInt32,  // instance_id
        VarType::UInt32,  // primitive_id
        VarType::UInt32,  // geometry_id
        VarType::UInt32   // user_instance_id
    };

    for (uint32_t i = 0; i < (td->shadow ? 1u : 8u); ++i)
        out[i] = jitc_var_new_node_1(
            JitBackend::AMD, VarKind::Extract, out_types[i],
            size, symbolic, trace, jitc_var(trace), (uint64_t) i);
}

#endif // defined(DRJIT_ENABLE_AMD)
