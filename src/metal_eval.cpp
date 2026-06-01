/**
 * src/metal_eval.cpp -- MSL kernel generation
 *
 * Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>
 *
 * All rights reserved. Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 *
 * --------------------------------------------------------------------------
 *
 * This file assembles a Metal Shading Language (MSL) kernel from a recorded
 * Dr.Jit computation graph. It is the Metal counterpart of
 * ``src/cuda_eval.cpp`` and follows the same overall structure.
 *
 * The format-string syntax accepted by ``buffer.fmt_metal`` mirrors
 * ``fmt_cuda`` but emits MSL-friendly identifiers (e.g. ``v5`` instead of
 * ``%f5``) and types (``float`` instead of ``f32``).
 *
 *  Format  Input          Example result      Description
 * --------------------------------------------------------------------------
 *  $u      uint32_t       1234                Decimal number (32 bit)
 *  $U      uint64_t       1234                Decimal number (64 bit)
 *  $x      uint32_t       4d2                 Hex number (32 bit)
 *  $X      uint64_t       4d2                 Hex number (64 bit)
 *  $Q      uint64_t       00000000000004d2    Zero-padded hex (64 bit)
 *  $s      const char *   foo                 Zero-terminated string
 *  $c      char           f                   Single character
 *  $t      Variable       float               Variable type (MSL)
 *  $b      Variable       uint                Binary view of the variable type
 *  $v/$V   Variable       v1234               Variable name
 *  $a      Variable       4                   Variable size in bytes
 *  $l      Variable       0x3f800000ull       Literal value (hex)
 *  $o      Variable       2                   Index into ``params.args[]``
 */

#if defined(DRJIT_ENABLE_METAL)

#include "eval.h"
#include "internal.h"
#include "var.h"
#include "log.h"
#include "strbuf.h"
#include "metal.h"
#include "metal_eval.h"

#include "metal_scatter.h"
#include "metal_array.h"
#include "metal_coop_vec.h"
#include "array.h"
#include "trace.h"
#include "call.h"
#include "metal_ts.h"
#include "loop.h"
#include "cond.h"
#include "record_ts.h"

#include <unordered_set>

#define fmt(fmt, ...) buffer.fmt_metal(count_args(__VA_ARGS__), fmt, ##__VA_ARGS__)
#define put(...)      buffer.put(__VA_ARGS__)

// Forward declaration
static void jitc_metal_render(Variable *v);

// ----------------------------------------------------------------------------
//  Internal API to track of scenes that are used by the kernel being compiled
// ----------------------------------------------------------------------------
std::vector<MetalScene *> metal_kernel_scenes;
std::vector<int32_t> metal_kernel_ift_slot;

void jitc_metal_assemble_reset() {
    metal_kernel_scenes.clear();
    metal_kernel_ift_slot.clear();
}

uint32_t metal_register_kernel_scene(MetalScene *scene) {
    for (size_t i = 0; i < metal_kernel_scenes.size(); ++i)
        if (metal_kernel_scenes[i] == scene)
            return (uint32_t) i;
    uint32_t slot = (uint32_t) metal_kernel_scenes.size();
    metal_kernel_scenes.push_back(scene);
    return slot;
}

void jitc_metal_finalize_scene_layout() {
    uint32_t n = (uint32_t) metal_kernel_scenes.size();
    metal_kernel_ift_slot.assign(n, -1);
    if (n == 0)
        return;
    // Layout: [[buffer(0)]] = params, [[buffer(1)]]..[[buffer(N)]] = accels,
    // [[buffer(N+1)]]..[[buffer(N+M)]] = IFTs (only for scenes that have one).
    uint32_t next = 1u + n;
    for (uint32_t i = 0; i < n; ++i) {
        if (metal_kernel_scenes[i] &&
            metal_kernel_scenes[i]->intersection_fn_library)
            metal_kernel_ift_slot[i] = (int32_t) next++;
    }
}

void jitc_metal_persist_kernel_scenes(Kernel &kernel) {
    delete[] kernel.metal.scenes;
    uint32_t n = (uint32_t) metal_kernel_scenes.size();
    if (n > 0) {
        kernel.metal.scenes = new void *[n];
        for (uint32_t i = 0; i < n; ++i)
            kernel.metal.scenes[i] = metal_kernel_scenes[i];
    } else {
        kernel.metal.scenes = nullptr;
    }
    kernel.metal.scene_count = n;
}

// ----------------------------------------------------------------------------
//  Recursive scene-discovery for the assemble pre-walk.
//
//  The Metal kernel signature must declare one ``accel_<i>`` argument per
//  distinct ``MetalScene*`` referenced by any TraceRay node *anywhere* in
//  the kernel — top-level schedule, callable bodies, symbolic loops, and
//  symbolic conditionals. A naive top-level-only scan misses scenes used
//  by virtual function calls (e.g. an area emitter's
//  ``eval_parameterization`` ray-traces against a shape's
//  separate parameterization scene from inside its sample_direction
//  vcall body).
//
//  This walk traverses the variable DAG starting from each top-level
//  schedule entry and visits, in addition to ordinary ``v->dep[]`` edges:
//    - ``CallData::inner_out`` and ``::side_effects`` for VarKind::Call
//    - ``LoopData::inner_out`` and ``::outer_in``   for VarKind::LoopStart
//    - ``CondData::indices_t/_f`` and ``::se_t/_f`` for VarKind::CondStart
//  A visited set keeps DAG/cycle traversal in linear time.
// ----------------------------------------------------------------------------
static void jitc_metal_collect_scenes(uint32_t index,
                                      std::unordered_set<uint32_t> &visited) {
    if (!index || !visited.insert(index).second)
        return;

    Variable *v = jitc_var(index);
    if (!v)
        return;

    VarKind kind = (VarKind) v->kind;

    // Register this scene if the node is a TraceRay. dep[1] holds the
    // scene_index variable returned by ``jit_metal_configure_scene``.
    if (kind == VarKind::TraceRay && v->dep[1]) {
        MetalScene *scene = jitc_metal_get_scene(v->dep[1]);
        if (scene)
            metal_register_kernel_scene(scene);
    }

    // Ordinary data-flow dependencies.
    for (uint32_t d : v->dep)
        if (d) jitc_metal_collect_scenes(d, visited);

    // Structured payloads — descend into the variable indices stored
    // inside the corresponding sidecar struct.
    if (kind == VarKind::Call) {
        if (auto *call = (CallData *) v->data) {
            for (uint32_t out : call->inner_out)
                if (out) jitc_metal_collect_scenes(out, visited);
            for (uint32_t se : call->side_effects)
                if (se) jitc_metal_collect_scenes(se, visited);
        }
    } else if (kind == VarKind::LoopStart) {
        if (auto *ld = (LoopData *) v->data) {
            for (uint32_t out : ld->inner_out)
                if (out) jitc_metal_collect_scenes(out, visited);
            for (uint32_t in : ld->outer_in)
                if (in) jitc_metal_collect_scenes(in, visited);
        }
    } else if (kind == VarKind::CondStart) {
        if (auto *cd = (CondData *) v->data) {
            for (uint32_t i : cd->indices_t)
                if (i) jitc_metal_collect_scenes(i, visited);
            for (uint32_t i : cd->indices_f)
                if (i) jitc_metal_collect_scenes(i, visited);
            for (uint32_t i : cd->se_t)
                if (i) jitc_metal_collect_scenes(i, visited);
            for (uint32_t i : cd->se_f)
                if (i) jitc_metal_collect_scenes(i, visited);
        }
    }
}

void jitc_metal_assemble(ThreadState *ts, ScheduledGroup group,
                         uint32_t /*n_regs*/, uint32_t n_params) {

    // -------------------------------------------------------------------
    //   0. Discover all distinct scenes referenced by TraceRay nodes
    //      anywhere in the kernel (top-level + callable bodies +
    //      symbolic loops/conds), and finalize the per-scene buffer
    //      slot layout.
    //
    //   Each scene becomes a kernel argument: ``accel_<i>`` at slot
    //   ``1 + i`` and (optionally) ``ift_<i>`` at the next available
    //   slot in ``[1+N, 1+N+M)``. Per-TraceRay codegen routes each
    //   intersect call to its scene's slot by linear-scanning
    //   ``metal_kernel_scenes``; the launch path mirrors the same
    //   layout when binding TLASes / IFTs.
    // -------------------------------------------------------------------
    {
        std::unordered_set<uint32_t> visited;
        for (uint32_t gi = group.start; gi != group.end; ++gi)
            jitc_metal_collect_scenes(schedule[gi].index, visited);
    }
    jitc_metal_finalize_scene_layout();

    // Cache the first scene on the ThreadState. It serves two purposes
    // for the launch path: (1) it remains the single fallback for
    // ``accel_0`` if the captured pointer goes stale (frozen replay
    // after scene rebuild); (2) per-TraceRay codegen uses it as a
    // backstop when an unregistered ``dep[1]`` slips through (defensive
    // — pre-walk should normally cover everything).
    MetalScene *scene =
        metal_kernel_scenes.empty() ? nullptr : metal_kernel_scenes[0];
    ts->metal_active_scene = scene;
    if (auto *rts = dynamic_cast<RecordThreadState *>(ts))
        rts->m_internal->metal_active_scene = scene;

    // -------------------------------------------------------------------
    //   1. MSL header
    // -------------------------------------------------------------------
    put("#include <metal_stdlib>\n"
        "#include <metal_atomic>\n");

    // Conditionally include the ray tracing header.
    if (uses_metal_rt /* repurposed flag: TraceRay used */)
        put("#include <metal_raytracing>\n");

    put("using namespace metal;\n");
    if (uses_metal_rt)
        put("using namespace raytracing;\n");
    put("\n");

    fmt("struct Params {\n"
        "    uint size;\n"
        "    device void *args[$u];\n"
        "};\n\n",
        n_params > 1 ? n_params - 1 : 1);

    // Emit one comment per registered scene capturing the PSO link
    // identity: the intersection-function names that must be linked into
    // the pipeline plus the scene's geometry-type mask. ``kernel_hash``
    // picks these up, so two kernels with identical MSL but different
    // scenes hash to distinct cache keys (no separate flag-mixing
    // required, c.f. the OptiX path which mixes pipeline-compile options
    // into the cache flags). This iterates the same ``metal_kernel_scenes``
    // list that drives the signature and the linker, so the cache key
    // matches exactly what gets linked.
    for (size_t i = 0; i < metal_kernel_scenes.size(); ++i) {
        MetalScene *si = metal_kernel_scenes[i];
        fmt("// Scene properties: accel_$u mask=0x$x fns=[",
            (uint32_t) i, si ? si->geometry_types_mask : 0u);
        if (si) {
            for (size_t j = 0; j < si->intersection_fn_names.size(); ++j) {
                if (j) put(", ");
                fmt("$s", si->intersection_fn_names[j].c_str());
            }
        }
        put("]\n");
    }

    fmt("kernel void drjit_^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^(\n"
        "    constant Params& params [[buffer(0)]],\n");

    // Multi-scene kernel signature: one ``accel_<i>`` per registered
    // scene at slots [1, N+1), then one ``ift_<i>`` for every scene
    // that has an ``intersection_fn_library`` (slots assigned by
    // ``jitc_metal_finalize_scene_layout``). Each IFT's template
    // depends on the scene's geometry-type mix (triangle-only vs
    // with-curves).

    if (uses_metal_rt) {
        for (size_t i = 0; i < metal_kernel_scenes.size(); ++i)
            fmt("    instance_acceleration_structure accel_$u [[buffer($u)]],\n",
                (uint32_t) i, (uint32_t) (1 + i));
        for (size_t i = 0; i < metal_kernel_scenes.size(); ++i) {
            int32_t ift_slot = metal_kernel_ift_slot[i];
            if (ift_slot < 0)
                continue;
            MetalScene *si = metal_kernel_scenes[i];
            bool has_curves_i =
                si && (si->geometry_types_mask & 0x4u) != 0;
            const char *ift_template =
                has_curves_i
                    ? "intersection_function_table<triangle_data, curve_data, instancing>"
                    : "intersection_function_table<triangle_data, instancing>";
            fmt("    $s ift_$u [[buffer($u)]],\n",
                ift_template, (uint32_t) i, (uint32_t) ift_slot);
        }
    }
    put("    uint r0 [[thread_position_in_grid]]) {\n");

    // -------------------------------------------------------------------
    //   4. Render every variable in the schedule
    // -------------------------------------------------------------------
    for (uint32_t gi = group.start; gi != group.end; ++gi) {
        uint32_t index = schedule[gi].index;
        Variable *v = jitc_var(index);
        VarKind kind = (VarKind) v->kind;
        ParamType ptype = (ParamType) v->param_type;

        // Apple GPUs have no hardware FP64, so Float64 is promoted to
        // Float32 at variable creation (see jitc_var_new) and never reaches
        // codegen. Seeing one here means a variable slipped past promotion.
        if ((VarType) v->type == VarType::Float64)
            jitc_fail("jitc_metal_assemble(): the program should not contain "
                      "Float64 variables.\n");

        // Declare output variables for TraceRay nodes before they're used
        if (kind == VarKind::TraceRay) {
            fmt("    bool  $v_out_0 = false;\n"
                "    float $v_out_1 = 0.0f;\n"
                "    float $v_out_2 = 0.0f;\n"
                "    float $v_out_3 = 0.0f;\n"
                "    uint  $v_out_4 = 0u;\n"
                "    uint  $v_out_5 = 0u;\n"
                "    uint  $v_out_6 = 0u;\n",
                v, v, v, v, v, v, v);
        }

        if (likely(ptype == ParamType::Input)) {
            // -------------------------------------------------
            // Load an input from the params buffer
            // -------------------------------------------------
            if (v->is_literal()) {
                VarType vt = (VarType) v->type;
                if (vt == VarType::Pointer) {
                    // Pointer literals must be loaded from params (not inlined)
                    // so frozen function replay can update the address
                    fmt("    $t $v = ($t) params.args[$o];\n",
                              v, v, v, v);
                } else if (vt == VarType::Float32) {
                    fmt("    $t $v = as_type<float>($lu);\n", v, v, v);
                } else if (vt == VarType::Float16) {
                    fmt("    $t $v = as_type<half>((ushort) $lu);\n", v, v, v);
                } else {
                    fmt("    $t $v = ($t) $lu;\n", v, v, v, v);
                }
                continue;
            }

            if (!v->is_array()) {
                if (v->size > 1)
                    fmt("    $t $v = ((device const $t*) params.args[$o])[r0];\n",
                              v, v, v, v);
                else
                    fmt("    $t $v = *(device const $t*) params.args[$o];\n",
                              v, v, v, v);
            } else {
                // Array memcpy helpers reference the named ``p<r>`` pointer.
                fmt("    device const $t* p$v = (device const $t*) params.args[$o];\n",
                    v, v, v, v);
                jitc_metal_render_array_memcpy_in(v);
            }
            continue;
        }

        // -------------------------------------------------
        // Output / regular operations
        // -------------------------------------------------
        jitc_metal_render(v);

        if (ptype == ParamType::Output) {
            if (!v->is_array()) {
                fmt("    ((device $t*) params.args[$o])[r0] = $v;\n",
                    v, v, v);
            } else {
                // Array memcpy helpers reference the named ``p<r>`` pointer.
                fmt("    device $t* p$v = (device $t*) params.args[$o];\n",
                    v, v, v, v);
                jitc_metal_render_array_memcpy_out(v);
            }
        }
    }

    put("}\n");

    // -------------------------------------------------------------------
    //   5. Emit callable functions and globals
    // -------------------------------------------------------------------

    // Assign callable_index for jitc_call_upload()
    {
        uint32_t ctr = 0;
        for (auto &it : globals_map) {
            if (it.first.type == GlobalType::IndirectCallable)
                it.second.callable_index = ctr++;
        }
    }

    // Emit Callable / IndirectCallable bodies after the kernel — MSL
    // accepts forward-declared callables defined later in the TU.
    // GlobalType::Global entries (DD helpers and other fmt_intrinsic
    // output) are emitted SEPARATELY into the chunk that gets moved
    // before the kernel: each one is a full inline definition, not a
    // forward-declarable callable, so it must precede every use.
    for (auto &it : globals_map) {
        if (it.first.type == GlobalType::Global)
            continue;
        put('\n');
        put(globals.get() + it.second.start, it.second.length);
        put('\n');
    }

    // Move callable forward declarations + Global bodies to before the
    // kernel. MSL needs functions defined or declared before use.
    if (!globals_map.empty()) {
        size_t suffix_start = buffer.size();
        // Find the insertion point: after "using namespace metal;\n"
        // (or "using namespace raytracing;\n" if present)
        const char *marker = uses_metal_rt
            ? strstr(buffer.get(), "using namespace raytracing;\n")
            : strstr(buffer.get(), "using namespace metal;\n");
        if (marker) {
            const char *eol = strchr(marker, '\n');
            size_t suffix_target = (eol - buffer.get()) + 1;

            // Emit Global bodies in full (DD helpers, fmt_intrinsic output).
            for (auto &it : globals_map) {
                if (it.first.type != GlobalType::Global)
                    continue;
                put('\n');
                put(globals.get() + it.second.start, it.second.length);
                put('\n');
            }

            // Emit forward declarations for all callable functions.
            for (auto &it : globals_map) {
                if (it.first.type != GlobalType::IndirectCallable &&
                    it.first.type != GlobalType::Callable)
                    continue;
                const char *sig = globals.get() + it.second.start;
                const char *brace = (const char *) memchr(sig, '{', it.second.length);
                if (brace) {
                    // Emit everything before the opening brace as a forward decl
                    size_t sig_len = brace - sig;
                    // Trim trailing whitespace
                    while (sig_len > 0 && (sig[sig_len-1] == ' ' || sig[sig_len-1] == '\n'))
                        sig_len--;
                    put(sig, sig_len);
                    put(";\n");
                }
            }

            if (suffix_start != buffer.size())
                buffer.move_suffix(suffix_start, suffix_target);
        }
    }

    // Upload offset tables to device memory
    jitc_call_upload(ts);
}

static void jitc_metal_render_unary(Variable *v, const char *op) {
    Variable *a = jitc_var(v->dep[0]);
    fmt("    $t $v = $s($v);\n", v, v, op, a);
}

static void jitc_metal_render_binary(Variable *v, const char *op) {
    Variable *a = jitc_var(v->dep[0]);
    Variable *b = jitc_var(v->dep[1]);
    fmt("    $t $v = $v $s $v;\n", v, v, a, op, b);
}

static void jitc_metal_render_call(Variable *v, const char *fn,
                                   uint32_t n_args) {
    Variable *a0 = jitc_var(v->dep[0]);
    fmt("    $t $v = $s($v", v, v, fn, a0);
    if (n_args >= 2) {
        Variable *a1 = jitc_var(v->dep[1]);
        fmt(", $v", a1);
    }
    if (n_args >= 3) {
        Variable *a2 = jitc_var(v->dep[2]);
        fmt(", $v", a2);
    }
    put(");\n");
}

static void jitc_metal_render(Variable *v) {
    if (v->coop_vec) {
        Variable *a0 = v->dep[0] ? jitc_var(v->dep[0]) : nullptr,
                 *a1 = v->dep[1] ? jitc_var(v->dep[1]) : nullptr,
                 *a2 = v->dep[2] ? jitc_var(v->dep[2]) : nullptr,
                 *a3 = v->dep[3] ? jitc_var(v->dep[3]) : nullptr;
        return jitc_metal_render_coop_vec(v, a0, a1, a2, a3);
    }

    switch ((VarKind) v->kind) {
        case VarKind::Nop:
            break;

        case VarKind::Literal: {
            VarType vt = (VarType) v->type;
            if (vt == VarType::Float32)
                fmt("    $t $v = as_type<float>($lu);\n", v, v, v);
            else if (vt == VarType::Float16)
                fmt("    $t $v = as_type<half>((ushort) $lu);\n", v, v, v);
            else if (vt == VarType::Bool)
                fmt("    $t $v = ($t) ($lu);\n", v, v, v, v);
            else
                fmt("    $t $v = ($t) $lu;\n", v, v, v, v);
            break;
        }

        case VarKind::Counter:
            fmt("    $t $v = ($t) r0;\n", v, v, v);
            break;

        // -- Arithmetic --
        case VarKind::Add: jitc_metal_render_binary(v, "+"); break;
        case VarKind::Sub: jitc_metal_render_binary(v, "-"); break;
        case VarKind::Mul: jitc_metal_render_binary(v, "*"); break;
        case VarKind::Div:
            jitc_metal_render_binary(v, "/");
            break;
        case VarKind::DivApprox: {
            Variable *a0 = jitc_var(v->dep[0]),
                     *a1 = jitc_var(v->dep[1]);
            fmt("    $t $v = fast::divide($v, $v);\n", v, v, a0, a1);
            break;
        }
        case VarKind::Mod: jitc_metal_render_binary(v, "%"); break;

        case VarKind::MulHi: {
            Variable *a = jitc_var(v->dep[0]);
            Variable *b = jitc_var(v->dep[1]);
            fmt("    $t $v = mulhi($v, $v);\n", v, v, a, b);
            break;
        }

        case VarKind::MulWide: {
            // 32-bit × 32-bit → 64-bit
            Variable *a = jitc_var(v->dep[0]);
            Variable *b = jitc_var(v->dep[1]);
            fmt("    $t $v = ($t)$v * ($t)$v;\n", v, v, v, a, v, b);
            break;
        }

        case VarKind::Fma: {
            Variable *a0 = jitc_var(v->dep[0]),
                     *a1 = jitc_var(v->dep[1]),
                     *a2 = jitc_var(v->dep[2]);
            if (jitc_is_float(v))
                fmt("    $t $v = fma($v, $v, $v);\n", v, v, a0, a1, a2);
            else
                fmt("    $t $v = $v * $v + $v;\n", v, v, a0, a1, a2);
            break;
        }
        case VarKind::Min:  jitc_metal_render_call(v, "min",  2); break;
        case VarKind::Max:  jitc_metal_render_call(v, "max",  2); break;

        // -- Rounding --
        case VarKind::Ceil:  jitc_metal_render_call(v, "ceil",  1); break;
        case VarKind::Floor: jitc_metal_render_call(v, "floor", 1); break;
        case VarKind::Round: jitc_metal_render_call(v, "rint",  1); break;
        case VarKind::Trunc: jitc_metal_render_call(v, "trunc", 1); break;

        // -- Comparisons --
        case VarKind::Eq:  jitc_metal_render_binary(v, "=="); break;
        case VarKind::Neq: jitc_metal_render_binary(v, "!="); break;
        case VarKind::Lt:  jitc_metal_render_binary(v, "<");  break;
        case VarKind::Le:  jitc_metal_render_binary(v, "<="); break;
        case VarKind::Gt:  jitc_metal_render_binary(v, ">");  break;
        case VarKind::Ge:  jitc_metal_render_binary(v, ">="); break;

        // -- Ternary --
        case VarKind::Select: {
            Variable *cond = jitc_var(v->dep[0]);
            Variable *a    = jitc_var(v->dep[1]);
            Variable *b    = jitc_var(v->dep[2]);
            fmt("    $t $v = $v ? $v : $v;\n", v, v, cond, a, b);
            break;
        }

        // -- Bit operations --
        case VarKind::Popc:  jitc_metal_render_call(v, "popcount", 1); break;
        case VarKind::Clz:   jitc_metal_render_call(v, "clz", 1); break;
        case VarKind::Ctz:   jitc_metal_render_call(v, "ctz", 1); break;
        case VarKind::Brev: {
            Variable *a = jitc_var(v->dep[0]);
            fmt("    $t $v = reverse_bits($v);\n", v, v, a);
            break;
        }

        case VarKind::And: {
            Variable *a0 = jitc_var(v->dep[0]),
                     *a1 = jitc_var(v->dep[1]);
            if (a0->type != a1->type)
                fmt("    $t $v = $v ? $v : ($t) 0;\n", v, v, a1, a0, v);
            else if (jitc_is_float(v))
                fmt("    $t $v = as_type<$t>(($b)(as_type<$b>($v) & as_type<$b>($v)));\n",
                    v, v, v, v, v, a0, v, a1);
            else
                fmt("    $t $v = $v & $v;\n", v, v, a0, a1);
            break;
        }

        case VarKind::Or: {
            Variable *a0 = jitc_var(v->dep[0]),
                     *a1 = jitc_var(v->dep[1]);
            if (a0->type != a1->type)
                fmt("    $t $v = as_type<$t>(($b)($v ? ($b) ~0u : as_type<$b>($v)));\n",
                    v, v, v, v, a1, v, v, a0);
            else if (jitc_is_float(v))
                fmt("    $t $v = as_type<$t>(($b)(as_type<$b>($v) | as_type<$b>($v)));\n",
                    v, v, v, v, v, a0, v, a1);
            else
                fmt("    $t $v = $v | $v;\n", v, v, a0, a1);
            break;
        }

        case VarKind::Xor: {
            Variable *a0 = jitc_var(v->dep[0]),
                     *a1 = jitc_var(v->dep[1]);
            if (jitc_is_float(v))
                fmt("    $t $v = as_type<$t>(($b)(as_type<$b>($v) ^ as_type<$b>($v)));\n",
                    v, v, v, v, v, a0, v, a1);
            else
                fmt("    $t $v = $v ^ $v;\n", v, v, a0, a1);
            break;
        }
        case VarKind::Shl: jitc_metal_render_binary(v, "<<"); break;
        case VarKind::Shr: jitc_metal_render_binary(v, ">>"); break;

        // -- Unary --
        case VarKind::Neg: jitc_metal_render_unary(v, "-"); break;
        case VarKind::Not: {
            Variable *a = jitc_var(v->dep[0]);
            if ((VarType) v->type == VarType::Bool)
                fmt("    $t $v = !$v;\n", v, v, a);
            else
                fmt("    $t $v = ~$v;\n", v, v, a);
            break;
        }
        case VarKind::Abs: jitc_metal_render_unary(v, "abs"); break;

        case VarKind::Sqrt:       jitc_metal_render_call(v, "sqrt", 1); break;
        case VarKind::SqrtApprox: jitc_metal_render_call(v, "fast::sqrt", 1); break;

        // -- Fast builtin transcendentals --
        case VarKind::Sin:  jitc_metal_render_call(v, "fast::sin",  1); break;
        case VarKind::Cos:  jitc_metal_render_call(v, "fast::cos",  1); break;
        case VarKind::Exp2: jitc_metal_render_call(v, "fast::exp2", 1); break;
        case VarKind::Log2: jitc_metal_render_call(v, "fast::log2", 1); break;
        case VarKind::Tanh: jitc_metal_render_call(v, "fast::tanh", 1); break;

        // -- Reciprocal --
        case VarKind::Rcp:
            fmt("    $t $v = ($t) 1.0 / $v;\n",
                v, v, v, jitc_var(v->dep[0]));
            break;

        case VarKind::RcpApprox:
            fmt("    $t $v = fast::divide(($t) 1.0, $v);\n",
                v, v, v, jitc_var(v->dep[0]));
            break;

        case VarKind::RSqrtApprox:
            jitc_metal_render_call(v, "fast::rsqrt", 1);
            break;

        // -- Casts --
        case VarKind::Cast: {
            Variable *a = jitc_var(v->dep[0]);
            fmt("    $t $v = ($t) $v;\n", v, v, v, a);
            break;
        }

        case VarKind::Bitcast: {
            Variable *a = jitc_var(v->dep[0]);
            if (v->type == a->type)
                fmt("    $t $v = $v;\n", v, v, a);
            else if (type_size[v->type] == type_size[a->type])
                fmt("    $t $v = as_type<$t>($v);\n", v, v, v, a);
            else
                fmt("    $t $v = as_type<$t>(($b) $v);\n", v, v, v, v, a);
            break;
        }

        // -- Bounds check --
        case VarKind::BoundsCheck: {
            Variable *index  = jitc_var(v->dep[0]);
            Variable *mask   = jitc_var(v->dep[1]);
            Variable *buf    = jitc_var(v->dep[2]);
            uint32_t size    = (uint32_t) v->literal;
            fmt("    bool $v = $v && ($v < (uint) $uu);\n"
                "    if ($v && !$v)\n"
                "        atomic_store_explicit((device atomic_uint*) $v, $v, memory_order_relaxed);\n",
                v, mask, index, size,
                mask, v,
                buf, index);
            break;
        }

        // -- Memory operations --
        case VarKind::Gather: {
            Variable *src   = jitc_var(v->dep[0]);
            Variable *index = jitc_var(v->dep[1]);
            Variable *mask  = jitc_var(v->dep[2]);
            bool is_unmasked = mask->is_literal() && mask->literal == 1;
            if (is_unmasked)
                fmt("    $t $v = ((device const $t*) $v)[$v];\n",
                    v, v, v, src, index);
            else
                fmt("    $t $v = ($v) ? "
                          "((device const $t*) $v)[$v] : ($t) 0;\n",
                    v, v, mask, v, src, index, v);
            break;
        }

        case VarKind::Scatter:
            jitc_metal_render_scatter(v);
            break;

        case VarKind::ScatterInc:
            jitc_metal_render_scatter_inc(v);
            break;

        case VarKind::ScatterKahan:
            jitc_metal_render_scatter_kahan(v);
            break;

        case VarKind::ScatterCAS:
            jitc_metal_render_scatter_cas(v);
            break;

        case VarKind::ScatterExch:
            jitc_metal_render_scatter_exch(v);
            break;

        // -- Extract (multi-output ops) --
        case VarKind::Extract: {
            Variable *src = jitc_var(v->dep[0]);
            uint32_t sub_index = (uint32_t) v->literal;

            if ((VarKind) src->kind == VarKind::TraceRay ||
                (VarKind) src->kind == VarKind::ScatterCAS) {
                // Extract from a multi-output op — reference the pre-declared outputs
                fmt("    $t $v = $v_out_$u;\n",
                    v, v, src, sub_index);
            } else {
                fmt("    $t $v = $v; // extract[$u]\n",
                    v, v, src, sub_index);
            }
            break;
        }

        // -- Loops --
        case VarKind::LoopStart: {
            const LoopData *ld = (LoopData *) v->data;
            for (size_t i = 0; i < ld->size; ++i) {
                Variable *inner_in = jitc_var(ld->inner_in[i]),
                         *outer_in = jitc_var(ld->outer_in[i]);
                if (inner_in == outer_in || !inner_in->reg_index ||
                    inner_in->is_array())
                    continue;
                if (outer_in->reg_index)
                    fmt("    $t $v = $v;\n", inner_in, inner_in, outer_in);
                else
                    fmt("    $t $v = ($t) 0;\n", inner_in, inner_in, inner_in);
            }
            put("    while (true) {\n");
            break;
        }

        case VarKind::LoopCond: {
            Variable *cond = jitc_var(v->dep[1]);
            fmt("        if (!$v) break;\n", cond);
            break;
        }

        case VarKind::LoopEnd: {
            Variable *a0 = jitc_var(v->dep[0]);
            const LoopData *ld = (LoopData *) a0->data;
            uint32_t size = (uint32_t) ld->size;

            // Handle back-edge copies with temporaries for aliasing
            for (uint32_t i = 0; i < size; ++i) {
                Variable *in  = jitc_var(ld->inner_in[i]),
                         *out = jitc_var(ld->inner_out[i]);
                in->scratch = out->scratch = 0;
            }

            for (uint32_t i = 0; i < size; ++i) {
                Variable *in  = jitc_var(ld->inner_in[i]),
                         *out = jitc_var(ld->inner_out[i]);
                if (in == out || !in->reg_index || !out->reg_index ||
                    in->is_array())
                    continue;
                in->scratch = 1;
            }

            for (uint32_t i = 0; i < size; ++i) {
                Variable *in  = jitc_var(ld->inner_in[i]),
                         *out = jitc_var(ld->inner_out[i]);
                if (in == out || !in->reg_index || !out->reg_index ||
                    in->is_array() || out->scratch != 1)
                    continue;
                fmt("    $t $v_tmp = $v;\n", in, in, out);
                out->scratch = 2;
            }

            for (uint32_t i = 0; i < size; ++i) {
                Variable *in  = jitc_var(ld->inner_in[i]),
                         *out = jitc_var(ld->inner_out[i]);
                if (in == out || !in->reg_index || !out->reg_index ||
                    in->is_array())
                    continue;
                if (out->scratch == 2)
                    fmt("    $v = $v_tmp;\n", in, in);
                else
                    fmt("    $v = $v;\n", in, out);
            }

            put("    }\n");
            break;
        }

        case VarKind::LoopPhi: {
            if (v->is_array()) {
                Variable *a3 = jitc_var(v->dep[3]);
                v->reg_index = a3->reg_index;
            }
            // In MSL, phi nodes that alias their input still need a
            // declaration so they can be referenced in the loop body
            break;
        }

        case VarKind::LoopOutput: {
            // Find which inner_in variable this output corresponds to
            Variable *loop_start = jitc_var(v->dep[0]);
            const LoopData *ld = (LoopData *) loop_start->data;
            for (size_t i = 0; i < ld->size; ++i) {
                Variable *outer_out = jitc_var(ld->outer_out[i]);
                if (outer_out == v) {
                    Variable *inner_in = jitc_var(ld->inner_in[i]);
                    if (v->reg_index && inner_in->reg_index)
                        fmt("    $t $v = $v;\n", v, v, inner_in);
                    break;
                }
            }
            break;
        }

        // -- Conditionals --
        case VarKind::CondStart: {
            const CondData *cd = (CondData *) v->data;
            Variable *cond = jitc_var(v->dep[0]);
            // Declare output variables before the if-statement
            for (size_t i = 0; i < cd->indices_out.size(); ++i) {
                Variable *vo = jitc_var(cd->indices_out[i]);
                if (vo && vo->reg_index)
                    fmt("    $t $v;\n", vo, vo);
            }
            fmt("    if ($v) {\n", cond);
            break;
        }

        case VarKind::CondMid: {
            Variable *a0 = jitc_var(v->dep[0]);
            const CondData *cd = (CondData *) a0->data;
            for (size_t i = 0; i < cd->indices_out.size(); ++i) {
                Variable *vt = jitc_var(cd->indices_t[i]),
                         *vo = jitc_var(cd->indices_out[i]);
                if (vo && vo->reg_index && vt->reg_index)
                    fmt("    $v = $v;\n", vo, vt);
            }
            put("    } else {\n");
            break;
        }

        case VarKind::CondEnd: {
            Variable *a0 = jitc_var(v->dep[0]);
            const CondData *cd = (CondData *) a0->data;
            for (size_t i = 0; i < cd->indices_out.size(); ++i) {
                Variable *vf = jitc_var(cd->indices_f[i]),
                         *vo = jitc_var(cd->indices_out[i]);
                if (vo && vo->reg_index && vf->reg_index)
                    fmt("    $v = $v;\n", vo, vf);
            }
            put("    }\n");
            break;
        }

        case VarKind::CondOutput:
            break;

        // -- Ray tracing (Metal inline intersection) --
        case VarKind::TraceRay: {
            TraceData *td = (TraceData *) v->data;
            Variable *valid = jitc_var(v->dep[0]);
            bool is_unmasked = valid->is_literal() && valid->literal == 1;

            // Resolve which kernel-bound ``accel_<i>`` / ``ift_<i>`` to
            // use for THIS trace by linear-scanning ``metal_kernel_scenes``.
            // If the scene wasn't registered (shouldn't happen, the
            // recursive pre-walk visits every TraceRay reachable from the
            // kernel), or no scene at all, fall back to the active scene
            // at slot 0.
            MetalScene *scene_local = nullptr;
            if (v->dep[1])
                scene_local = jitc_metal_get_scene(v->dep[1]);
            if (!scene_local)
                scene_local = jitc_metal_active_scene();

            uint32_t accel_idx = 0;
            if (scene_local) {
                for (size_t i = 0; i < metal_kernel_scenes.size(); ++i) {
                    if (metal_kernel_scenes[i] == scene_local) {
                        accel_idx = (uint32_t) i;
                        break;
                    }
                }
            }

            // Guard with active mask
            if (!is_unmasked)
                fmt("    if ($v) {\n", valid);

            // Read ray parameters from TraceData indices
            Variable *ox   = jitc_var(td->indices[0]);
            Variable *oy   = jitc_var(td->indices[1]);
            Variable *oz   = jitc_var(td->indices[2]);
            Variable *dx   = jitc_var(td->indices[3]);
            Variable *dy   = jitc_var(td->indices[4]);
            Variable *dz   = jitc_var(td->indices[5]);
            Variable *tmin = jitc_var(td->indices[6]);
            Variable *tmax = jitc_var(td->indices[7]);

            bool has_ift_local = scene_local && scene_local->intersection_fn_library;
            bool has_curves_local =
                scene_local && (scene_local->geometry_types_mask & 0x4u) != 0;
            bool extended = has_ift_local || has_curves_local;

            // Emit intersector + ray setup + intersect call.
            // - Triangle-only fast path: `intersector<triangle_data, instancing>`
            //   with `assume_geometry_type(triangle)`.
            // - Mixed (any non-triangle): drop the assume hint so curves and
            //   bounding-box geometry are also tested. The IFT is passed only
            //   when custom-intersection shapes are present.
            put("    {\n");
            if (extended) {
                if (has_curves_local) {
                    put("        metal::raytracing::intersector<metal::raytracing::triangle_data, "
                        "metal::raytracing::curve_data, metal::raytracing::instancing> _inter;\n");
                } else {
                    put("        metal::raytracing::intersector<metal::raytracing::triangle_data, "
                        "metal::raytracing::instancing> _inter;\n");
                }
                put("        _inter.force_opacity(metal::raytracing::forced_opacity::opaque);\n"
                    "        _inter.accept_any_intersection(false);\n");
            } else {
                put("        metal::raytracing::intersector<metal::raytracing::triangle_data, "
                    "metal::raytracing::instancing> _inter;\n"
                    "        _inter.assume_geometry_type(metal::raytracing::geometry_type::triangle);\n"
                    "        _inter.force_opacity(metal::raytracing::forced_opacity::opaque);\n"
                    "        _inter.accept_any_intersection(false);\n");
            }
            fmt("        metal::raytracing::ray _r;\n"
                "        _r.origin = float3($v, $v, $v);\n"
                "        _r.direction = float3($v, $v, $v);\n"
                "        _r.min_distance = $v;\n"
                "        _r.max_distance = $v;\n",
                ox, oy, oz, dx, dy, dz, tmin, tmax);
            // Route the intersect call to this scene's kernel-bound
            // accel + IFT (per-scene slots assigned at signature time).
            if (has_ift_local)
                fmt("        auto _hit = _inter.intersect(_r, accel_$u, ift_$u);\n",
                    accel_idx, accel_idx);
            else
                fmt("        auto _hit = _inter.intersect(_r, accel_$u);\n",
                    accel_idx);

            // Hit-result extraction.
            // - Triangle hit: prim_uv = triangle_barycentric_coord
            // - Curve hit:    prim_uv = (curve_parameter, 0)
            // - Bbox hit:     prim_uv = (0, 0) — compute_surface_interaction()
            //                 will recompute it from the hit point.
            if (extended && has_curves_local) {
                // Apple's HW curve intersector reports a degenerate ``curve``
                // hit at ``distance = 0`` when the ray's origin lies inside
                // the swept-tube volume of a `CurveTypeRound` geometry — even
                // though there's no actual surface at the origin. Embree and
                // OptiX never produce such hits. Treat zero-distance curve
                // hits as misses so the public ray_intersect contract agrees
                // across backends (otherwise backface tests on rays starting
                // inside a curve report a spurious hit at t=0 with
                // garbage curve_parameter).
                fmt("        auto _ht = _hit.type;\n"
                    "        bool _zero_curve_hit = (_ht == metal::raytracing::intersection_type::curve) && (_hit.distance <= 0.0f);\n"
                    "        $v_out_0 = (_ht != metal::raytracing::intersection_type::none) && !_zero_curve_hit;\n"
                    "        $v_out_1 = _hit.distance;\n"
                    "        $v_out_2 = (_ht == metal::raytracing::intersection_type::triangle)\n"
                    "                   ? _hit.triangle_barycentric_coord.x\n"
                    "                   : ((_ht == metal::raytracing::intersection_type::curve)\n"
                    "                      ? _hit.curve_parameter : 0.0f);\n"
                    "        $v_out_3 = (_ht == metal::raytracing::intersection_type::triangle)\n"
                    "                   ? _hit.triangle_barycentric_coord.y : 0.0f;\n"
                    "        $v_out_4 = _hit.instance_id;\n"
                    "        $v_out_5 = _hit.primitive_id;\n"
                    "        $v_out_6 = _hit.geometry_id;\n"
                    "    }\n",
                    v, v, v, v, v, v, v);
            } else {
                // No curves — both the triangle-only fast path and the
                // triangle+bbox extended path can use the same extraction:
                // triangle_barycentric_coord is correct for triangle hits and
                // unused (overwritten) for bbox hits.
                fmt("        $v_out_0 = (_hit.type != metal::raytracing::intersection_type::none);\n"
                          "        $v_out_1 = _hit.distance;\n"
                          "        $v_out_2 = _hit.triangle_barycentric_coord.x;\n"
                          "        $v_out_3 = _hit.triangle_barycentric_coord.y;\n"
                          "        $v_out_4 = _hit.instance_id;\n"
                          "        $v_out_5 = _hit.primitive_id;\n"
                          "        $v_out_6 = _hit.geometry_id;\n"
                          "    }\n",
                          v, v, v, v, v, v, v);
            }

            if (!is_unmasked)
                put("    }\n");
            break;
        }

        case VarKind::TexLookup:
            jitc_fail("jitc_metal_render(): texture lookup (VarKind::TexLookup) "
                      "is not yet implemented on the Metal backend. Metal "
                      "requires MTLTexture + sampler infrastructure (vs CUDA's "
                      "CUtexObject).");

        case VarKind::TexFetchBilerp:
            jitc_fail("jitc_metal_render(): bilinear texel fetch "
                      "(VarKind::TexFetchBilerp) is not yet implemented on "
                      "the Metal backend.");

        case VarKind::Call: {
            Variable *a0 = jitc_var(v->dep[0]),
                     *a1 = jitc_var(v->dep[1]),
                     *a2 = jitc_var(v->dep[2]),
                     *a3 = v->dep[3] ? jitc_var(v->dep[3]) : nullptr;
            jitc_var_call_assemble((CallData *) v->data, v->reg_index,
                                   a0->reg_index, a1->reg_index, a2->reg_index,
                                   a3 ? a3->reg_index : 0);
            break;
        }

        case VarKind::CallOutput:
            break;

        case VarKind::CallSelf:
            fmt("    uint $v = self;\n", v);
            break;

        case VarKind::CallInput:
            break;

        case VarKind::CoopVecUnpack: {
            Variable *a0 = jitc_var(v->dep[0]);
            jitc_metal_render_coop_vec_unpack(v, a0);
            break;
        }

        case VarKind::CoopVecAccum: {
            Variable *a0 = jitc_var(v->dep[0]),
                     *a1 = jitc_var(v->dep[1]),
                     *a2 = jitc_var(v->dep[2]);
            jitc_metal_render_coop_vec_accum(v, a0, a1, a2);
            break;
        }

        case VarKind::CoopVecOuterProductAccum: {
            Variable *a0 = jitc_var(v->dep[0]),
                     *a1 = jitc_var(v->dep[1]),
                     *a2 = jitc_var(v->dep[2]),
                     *a3 = v->dep[3] ? jitc_var(v->dep[3]) : nullptr;
            jitc_metal_render_coop_vec_outer_product_accum(v, a0, a1, a2, a3);
            break;
        }

        // -- Local arrays --
        case VarKind::Array:
            jitc_metal_render_array(v, v->dep[0] ? jitc_var(v->dep[0]) : nullptr);
            break;

        case VarKind::ArrayInit:
            jitc_metal_render_array_init(v, jitc_var(v->dep[0]), jitc_var(v->dep[1]));
            break;

        case VarKind::ArrayWrite:
            jitc_metal_render_array_write(v, jitc_var(v->dep[0]), jitc_var(v->dep[1]),
                                          jitc_var(v->dep[2]),
                                          v->dep[3] ? jitc_var(v->dep[3]) : nullptr);
            break;

        case VarKind::ArrayRead:
            jitc_metal_render_array_read(v, jitc_var(v->dep[0]), jitc_var(v->dep[1]),
                                         v->dep[2] ? jitc_var(v->dep[2]) : nullptr);
            break;

        case VarKind::ArrayPhi:
            v->reg_index = jitc_var(v->dep[0])->reg_index;
            break;

        case VarKind::ArraySelect:
            jitc_metal_render_array_select(v, jitc_var(v->dep[0]),
                                           jitc_var(v->dep[1]), jitc_var(v->dep[2]));
            break;

        default:
            jitc_fail("jitc_metal_render(): unhandled VarKind=%u (%s) for "
                      "variable r%u (type=%s).",
                      (uint32_t) v->kind, var_kind_name[v->kind],
                      v->reg_index, type_name[v->type]);
    }
}

// ============================================================================
//  Virtual function call support (switch-case dispatch)
// ============================================================================

void jitc_metal_assemble_func(const CallData *call, uint32_t inst,
                              uint32_t in_size, uint32_t /*in_align*/,
                              uint32_t out_size, uint32_t /*out_align*/,
                              uint32_t /*n_regs*/) {
    // Emit the function signature with '^' placeholders for the hash
    put("void func_^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^(");

    if (call->use_index)
        put("uint index, ");
    if (call->use_self)
        put("uint self, ");
    if (!call->data_map.empty())
        put("device uint8_t* data, ");
    if (in_size)
        put("thread const uint8_t* params, ");
    if (out_size)
        put("thread uint8_t* result, ");
    // Callables receive every kernel-bound scene resource so that any
    // TraceRay inside the callable body can reach any registered
    // scene. Per-callable analysis (passing only the subset actually
    // used) is a future optimization — Metal is generally good at
    // eliminating unused parameters during pipeline compilation.
    if (uses_metal_rt) {
        for (size_t i = 0; i < metal_kernel_scenes.size(); ++i)
            fmt("instance_acceleration_structure accel_$u, ",
                (uint32_t) i);
        for (size_t i = 0; i < metal_kernel_scenes.size(); ++i) {
            int32_t ift_slot = metal_kernel_ift_slot[i];
            if (ift_slot < 0)
                continue;
            MetalScene *si = metal_kernel_scenes[i];
            bool has_curves_i =
                si && (si->geometry_types_mask & 0x4u) != 0;
            const char *ift_template =
                has_curves_i
                    ? "intersection_function_table<triangle_data, curve_data, instancing>"
                    : "intersection_function_table<triangle_data, instancing>";
            fmt("$s ift_$u, ", ift_template, (uint32_t) i);
        }
    }

    // Remove trailing ", "
    buffer.delete_trailing_commas();

    fmt(") {\n"
        "    // Call: $s\n",
        call->name.c_str());

    for (size_t i = 0; i < schedule.size(); ++i) {
        ScheduledVariable &sv = schedule[i];
        Variable *v = jitc_var(sv.index);
        VarType vt = (VarType) v->type;
        VarKind kind = (VarKind) v->kind;

        // Pre-declare TraceRay output variables in callable functions
        if (kind == VarKind::TraceRay) {
            fmt("    bool  $v_out_0 = false;\n"
                "    float $v_out_1 = 0.0f;\n"
                "    float $v_out_2 = 0.0f;\n"
                "    float $v_out_3 = 0.0f;\n"
                "    uint  $v_out_4 = 0u;\n"
                "    uint  $v_out_5 = 0u;\n"
                "    uint  $v_out_6 = 0u;\n",
                v, v, v, v, v, v, v);
        }

        if (kind == VarKind::Counter) {
            fmt("    $t $v = ($t) index;\n", v, v, v);
        } else if (kind == VarKind::CallInput) {
            Variable *a = jitc_var(v->dep[0]);
            if (vt != VarType::Bool)
                fmt("    $t $v = *(thread const $t*)(params + $u);\n",
                          v, v, v, a->param_offset);
            else
                fmt("    bool $v = *(thread const uint8_t*)(params + $u) != 0;\n",
                          v, a->param_offset);
        } else if (kind == VarKind::CallSelf) {
            fmt("    uint $v = self;\n", v);
        } else if (v->is_evaluated() || (vt == VarType::Pointer && kind == VarKind::Literal)) {
            uint64_t key = (uint64_t) sv.index + (((uint64_t) inst) << 32);
            auto it = call->data_map.find(key);

            if (unlikely(it == call->data_map.end())) {
                jitc_fail("jitc_metal_assemble_func(): could not find entry for "
                          "variable r%u in 'data_map' for function %s",
                          sv.index, call->name.c_str());
                continue;
            }

            if (it->second == (uint32_t) -1)
                jitc_fail(
                    "jitc_metal_assemble_func(): variable r%u is referenced by "
                    "a recorded function call. However, it was evaluated "
                    "between the recording step and code generation (which is "
                    "happening now). This is not allowed.", sv.index);

            // Track device buffers referenced from callable data for
            // useResource(). During freeze recording the active TS is a
            // ``RecordThreadState`` (which does NOT inherit from
            // ``MetalThreadState``), so we must unwrap to the underlying
            // Metal TS before touching MetalThreadState-specific fields —
            // otherwise the cast would alias garbage memory.
            auto *ts_curr = thread_state(JitBackend::Metal);
            if (auto *rts = dynamic_cast<RecordThreadState *>(ts_curr))
                ts_curr = rts->m_internal;
            auto *mts = static_cast<MetalThreadState *>(ts_curr);
            if (vt == VarType::Pointer) {
                mts->metal_call_resources.push_back((void *) v->literal);
            } else if (v->is_evaluated() && v->data) {
                mts->metal_call_resources.push_back(v->data);
            }

            uint32_t offset = it->second - call->data_offset[inst];
            if (vt == VarType::Bool)
                fmt("    bool $v = *(device uint8_t*)(data + $u) != 0;\n",
                    v, offset);
            else if (vt == VarType::Pointer)
                fmt("    device uint8_t* $v = "
                    "*(device uint8_t* device const*)(data + $u);\n",
                    v, offset);
            else
                fmt("    $t $v = *(device $t*)(data + $u);\n",
                    v, v, v, offset);
        } else if (v->is_literal()) {
            jitc_metal_render(v);
        } else {
            jitc_metal_render(v);
        }
    }

    // Write outputs to result buffer
    for (uint32_t i = 0; i < call->n_out; ++i) {
        const Variable *v = jitc_var(call->inner_out[inst * call->n_out + i]);
        uint32_t offset = call->out_offset[i];

        if (offset == (uint32_t) -1)
            continue;

        if ((VarType) v->type != VarType::Bool)
            fmt("    *(thread $t*)(result + $u) = $v;\n",
                v, offset, v);
        else
            fmt("    *(thread uint8_t*)(result + $u) = $v ? 1 : 0;\n",
                offset, v);
    }

    put("}\n");
}

void jitc_var_call_assemble_metal(CallData *call, uint32_t call_reg,
                                  uint32_t self_reg, uint32_t mask_reg,
                                  uint32_t offset_reg, uint32_t data_reg,
                                  uint32_t in_size, uint32_t /*in_align*/,
                                  uint32_t out_size, uint32_t /*out_align*/) {
    // =====================================================
    // 1. Conditional branch (masked lanes)
    // =====================================================

    Variable *mask = jitc_var(jitc_var(call->id)->dep[1]);
    bool is_masked = !mask->is_literal() || mask->literal != 1;

    fmt("\n    // VCall: $s\n", call->name.c_str());

    // =====================================================
    // 2. Declare input local buffers and pack inputs
    // =====================================================

    if (in_size) {
        fmt("    uint8_t _in_$u[$u];\n", call_reg, in_size);

        for (uint32_t i = 0; i < call->n_in; ++i) {
            const Variable *v = jitc_var(call->outer_in[i]);
            const Variable *v_in = jitc_var(call->inner_in[i]);
            bool unused =
                !v->reg_index ||
                (call->optimize &&
                    (v->is_literal() || v_in->ref_count == 1));
            if (unused)
                continue;

            if ((VarType) v->type != VarType::Bool)
                fmt("    *(thread $t*)(_in_$u + $u) = $v;\n",
                          v, call_reg, v->param_offset, v);
            else
                fmt("    *(thread uint8_t*)(_in_$u + $u) = $v ? 1 : 0;\n",
                          call_reg, v->param_offset, v);
        }
    }

    // =====================================================
    // 3. Declare output buffer (zero-initialized)
    // =====================================================

    if (out_size)
        fmt("    uint8_t _out_$u[$u] = {};\n", call_reg, out_size);

    // =====================================================
    // 4. Masked guard + offset table lookup
    // =====================================================

    const char *indent = is_masked ? "            " : "        ";

    if (is_masked)
        fmt("    if (v$u) {\n", mask_reg);

    // Load the uint64_t entry: (data_offset << 32) | callable_index
    fmt("$sulong _oe_$u = ((device const ulong*) v$u)[v$u];\n"
        "$suint _ci_$u = (uint)(_oe_$u & 0xFFFFFFFFu);\n",
        indent, call_reg, offset_reg, self_reg,
        indent, call_reg, call_reg);

    if (data_reg)
        fmt("$sdevice uint8_t* _cd_$u = "
            "(device uint8_t*) v$u + (uint)(_oe_$u >> 32);\n",
            indent, call_reg, data_reg, call_reg);

    // =====================================================
    // 5. Switch-case dispatch
    // =====================================================

    // Inside a callable function the kernel-level `r0` (thread_position_in_grid)
    // is not in scope; the enclosing callable receives the thread index as its
    // `index` parameter. Use that name for nested vcalls.
    const char *index_name = (callable_depth > 0) ? "index" : "r0";

    if (call->n_inst == 1) {
        // Single instance: direct call, no switch needed
        XXH128_hash_t hash = call->inst_hash[0];
        fmt("$sfunc_", indent);
        buffer.put_q64_unchecked(hash.high64);
        buffer.put_q64_unchecked(hash.low64);
        put("(");
        if (call->use_index)
            fmt("$s, ", index_name);
        if (call->use_self)
            fmt("v$u, ", self_reg);
        if (data_reg)
            fmt("_cd_$u, ", call_reg);
        if (in_size)
            fmt("_in_$u, ", call_reg);
        if (out_size)
            fmt("_out_$u, ", call_reg);
        // Forward every accel + IFT from the kernel-level bindings
        // into the callable, matching the callable's signature.
        if (uses_metal_rt) {
            for (size_t i = 0; i < metal_kernel_scenes.size(); ++i)
                fmt("accel_$u, ", (uint32_t) i);
            for (size_t i = 0; i < metal_kernel_scenes.size(); ++i) {
                if (metal_kernel_ift_slot[i] >= 0)
                    fmt("ift_$u, ", (uint32_t) i);
            }
        }
        buffer.delete_trailing_commas();
        put(");\n");
    } else {
        fmt("$sswitch (_ci_$u) {\n", indent, call_reg);

        for (uint32_t i = 0; i < call->n_inst; ++i) {
            XXH128_hash_t hash = call->inst_hash[i];

            fmt("$s    case $u: func_$Q$Q(",
                      indent, i,
                      hash.high64, hash.low64);

            if (call->use_index)
                fmt("$s, ", index_name);
            if (call->use_self)
                fmt("v$u, ", self_reg);
            if (data_reg)
                fmt("_cd_$u, ", call_reg);
            if (in_size)
                fmt("_in_$u, ", call_reg);
            if (out_size)
                fmt("_out_$u, ", call_reg);
            // Forward all kernel-bound scene resources into each
            // switch-case branch (mirror of the single-instance path
            // above).
            if (uses_metal_rt) {
                for (size_t i = 0; i < metal_kernel_scenes.size(); ++i)
                    fmt("accel_$u, ", (uint32_t) i);
                for (size_t i = 0; i < metal_kernel_scenes.size(); ++i) {
                    if (metal_kernel_ift_slot[i] >= 0)
                        fmt("ift_$u, ", (uint32_t) i);
                }
            }
            buffer.delete_trailing_commas();
            put("); break;\n");
        }

        fmt("$s    default: break;\n"
                  "$s}\n", indent, indent);
    }

    if (is_masked)
        put("    }\n");

    // =====================================================
    // 6. Read outputs into registers
    // =====================================================

    for (uint32_t i = 0; i < call->n_out; ++i) {
        const Variable *v = jitc_var(call->outer_out[i]);
        if (!v || !v->reg_index || v->param_type == ParamType::Input)
            continue;

        uint32_t offset = call->out_offset[i];
        if (offset == (uint32_t) -1)
            continue;

        if ((VarType) v->type != VarType::Bool)
            fmt("    $t $v = *(thread const $t*)(_out_$u + $u);\n",
                      v, v, v, call_reg, offset);
        else
            fmt("    bool $v = *(thread const uint8_t*)(_out_$u + $u) != 0;\n",
                      v, call_reg, offset);
    }

    put("\n");
}

#endif // defined(DRJIT_ENABLE_METAL)
