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
 *  $o      Variable       16                  Offset in the params buffer
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
#include "metal_dd_preamble.h"
#include "array.h"
#include "trace.h"
#include "call.h"
#include "metal_ts.h"
#include "loop.h"
#include "cond.h"
#include "record_ts.h"

#include <unordered_set>

// Convenience macros local to this translation unit (mirroring cuda_eval.cpp).
// The CUDA macro takes the format string as a separate parameter so that
// ``count_args`` only counts the variadic arguments.
#define fmt_metal(fmt, ...) buffer.fmt_metal(count_args(__VA_ARGS__), fmt, ##__VA_ARGS__)
#define put(...)            buffer.put(__VA_ARGS__)

// Forward declaration
static void jitc_metal_render(Variable *v);

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
                         uint32_t /*n_regs*/, uint32_t /*n_params*/) {

    // -------------------------------------------------------------------
    //   0. Discover all distinct scenes referenced by TraceRay nodes
    //      anywhere in the kernel (top-level + callable bodies +
    //      symbolic loops/conds), and finalize the per-scene buffer
    //      slot layout.
    //
    //   Each scene becomes a kernel argument: ``accel_<i>`` at slot
    //   ``1 + i`` and (optionally) ``ift_<i>`` at the next available
    //   slot in ``[1+N, 1+N+M)``. Per-TraceRay codegen routes each
    //   intersect call to its scene's slot via
    //   ``metal_kernel_scene_slot``; the launch path mirrors the same
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
        "#include <metal_atomic>\n"
        "#include <metal_simdgroup>\n");

    // Conditionally include the ray tracing header.
    if (uses_metal_rt /* repurposed flag: TraceRay used */)
        put("#include <metal_raytracing>\n");

    put("using namespace metal;\n");
    if (uses_metal_rt)
        put("using namespace raytracing;\n");
    put("\n");

    // Float64 (double-double) helpers are emitted on demand: each codegen
    // site in jitc_metal_render_dd registers exactly the dd_* helpers it
    // references via fmt_intrinsic, and jitc_register_global dedups them.
    // Kernels that don't touch Float64 emit zero dd_* text.
    bool dd_enabled = (jitc_flags() &
                       (uint32_t) JitFlag::MetalEmulateFloat64) != 0;

    // -------------------------------------------------------------------
    //   2. Kernel entry point
    //
    //   The kernel receives:
    //     buffer(0)  -- flat parameter buffer (pointers + leading uint32 size)
    //     buffer(1)  -- (optional) instance acceleration structure
    //     [[thread_position_in_grid]] -- linear thread index
    //
    //   The leading 32-bit word of the params buffer is the launch ``size``.
    //   This matches the CUDA kernel calling convention.
    // -------------------------------------------------------------------
    // simdgroup_matrix MatVec fast path constrains every threadgroup to a
    // single SIMD-group (32 threads) so the per-kernel threadgroup memory
    // is per-SG by definition — no partitioning needed.
    if (uses_simdgroup_matrix)
        put("[[max_total_threads_per_threadgroup(32)]]\n");

    fmt_metal("kernel void drjit_^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^(\n"
              "    device const uint8_t* params [[buffer(0)]],\n");
    // Multi-scene kernel signature: one ``accel_<i>`` per registered
    // scene at slots [1, N+1), then one ``ift_<i>`` for every scene
    // that has an ``intersection_fn_library`` (slots assigned by
    // ``jitc_metal_finalize_scene_layout``). Each IFT's template
    // depends on the scene's geometry-type mix (triangle-only vs
    // with-curves).
    if (uses_metal_rt) {
        for (size_t i = 0; i < metal_kernel_scenes.size(); ++i)
            fmt_metal("    instance_acceleration_structure accel_$u [[buffer($u)]],\n",
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
            fmt_metal("    $s ift_$u [[buffer($u)]],\n",
                      ift_template, (uint32_t) i, (uint32_t) ift_slot);
        }
    }
    put("    uint r0 [[thread_position_in_grid]]");
    if (uses_simdgroup_matrix)
        put(",\n    uint sg_lane [[thread_index_in_simdgroup]]");
    put(") {\n");

    // -------------------------------------------------------------------
    //   3. Bounds check
    // -------------------------------------------------------------------
    put("    uint r2 = *(device const uint*) params;\n"
        "    if (r0 >= r2) return;\n\n");

    // -------------------------------------------------------------------
    //   3b. Threadgroup memory for simdgroup_matrix matvec staging
    // -------------------------------------------------------------------
    if (uses_simdgroup_matrix)
        fmt_metal("    threadgroup float _sg_tgm[$u];\n\n",
                  simdgroup_tgm_floats * 32);

    // -------------------------------------------------------------------
    //   4. Render every variable in the schedule
    // -------------------------------------------------------------------
    for (uint32_t gi = group.start; gi != group.end; ++gi) {
        uint32_t index = schedule[gi].index;
        Variable *v = jitc_var(index);
        VarKind kind = (VarKind) v->kind;
        ParamType ptype = (ParamType) v->param_type;

        // Float64 only reaches codegen when DD emulation is enabled (the
        // demotion path in jitc_var_new strips Float64 otherwise). If we
        // ever see a Float64 variable with the flag OFF, it's a bug --
        // either an evaluated buffer slipped past the demotion, or the
        // flag was toggled mid-computation. Fail loudly with guidance.
        if ((VarType) v->type == VarType::Float64 && !dd_enabled)
            jitc_fail("jitc_metal_assemble(): a Float64 variable (r%u) "
                      "reached MSL codegen with JitFlag::MetalEmulateFloat64 "
                      "OFF. Either enable the flag (slow but precise) or "
                      "cast the variable to Float32 before evaluation.",
                      index);

        // Declare output variables for TraceRay nodes before they're used
        if (kind == VarKind::TraceRay) {
            fmt_metal("    bool  $v_out_0 = false;\n"
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
                    fmt_metal("    $t $v = *(device const $t*)(params + $o);\n",
                              v, v, v, v);
                } else if (vt == VarType::Float32) {
                    fmt_metal("    $t $v = as_type<float>($lu);\n", v, v, v);
                } else if (vt == VarType::Float16) {
                    fmt_metal("    $t $v = as_type<half>((ushort) $lu);\n", v, v, v);
                } else if (vt == VarType::Float64) {
                    // DD literal: split the IEEE 754 double bit-pattern into a
                    // (hi, lo) float pair, then materialize via dd_from_bits.
                    double d;
                    memcpy(&d, &v->literal, sizeof(double));
                    float hi = (float) d;
                    float lo = (float) (d - (double) hi);
                    uint32_t hi_bits, lo_bits;
                    memcpy(&hi_bits, &hi, sizeof(float));
                    memcpy(&lo_bits, &lo, sizeof(float));
                    drjit::register_dd_from_bits();
                    fmt_metal("    dd_t $v = dd_from_bits(0x$xu, 0x$xu);\n",
                              v, hi_bits, lo_bits);
                } else {
                    fmt_metal("    $t $v = ($t) $lu;\n", v, v, v, v);
                }
                continue;
            }

            // ``device const T* p<r> = *(device const T* device const*)
            //  (params + offset);``
            // ``T v<r> = (size > 1) ? p<r>[r0] : *p<r>;``
            // For Float64 the `$t` formatter emits `dd_t` (= float2), so we
            // register the typedef even when no dd_* helper is touched.
            if ((VarType) v->type == VarType::Float64)
                drjit::register_dd_t();
            fmt_metal("    device const $t* p$v = "
                      "*(device const $t* device const*)(params + $o);\n",
                      v, v, v, v);

            if (!v->is_array()) {
                if (v->size > 1)
                    fmt_metal("    $t $v = p$v[r0];\n", v, v, v);
                else
                    fmt_metal("    $t $v = *p$v;\n", v, v, v);
            } else {
                jitc_metal_render_array_memcpy_in(v);
            }
            continue;
        }

        // -------------------------------------------------
        // Output / regular operations
        // -------------------------------------------------
        jitc_metal_render(v);

        if (ptype == ParamType::Output) {
            if ((VarType) v->type == VarType::Float64)
                drjit::register_dd_t();
            fmt_metal("    device $t* p$v = "
                      "*(device $t* device const*)(params + $o);\n",
                      v, v, v, v);

            if (!v->is_array()) {
                fmt_metal("    p$v[r0] = $v;\n", v, v);
            } else {
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

// ============================================================================
//
//  Per-VarKind code emission. This implementation covers a representative
//  subset of the full ~80-case switch statement in cuda_eval.cpp; the
//  remaining cases follow the same template and are scheduled for
//  completion as part of Phase 2 testing on a real Metal device.
//
// ============================================================================

static void jitc_metal_render_unary(Variable *v, const char *op) {
    Variable *a = jitc_var(v->dep[0]);
    fmt_metal("    $t $v = $s($v);\n", v, v, op, a);
}

static void jitc_metal_render_binary(Variable *v, const char *op) {
    Variable *a = jitc_var(v->dep[0]);
    Variable *b = jitc_var(v->dep[1]);
    fmt_metal("    $t $v = $v $s $v;\n", v, v, a, op, b);
}

static void jitc_metal_render_call(Variable *v, const char *fn,
                                   uint32_t n_args) {
    Variable *a0 = jitc_var(v->dep[0]);
    fmt_metal("    $t $v = $s($v", v, v, fn, a0);
    if (n_args >= 2) {
        Variable *a1 = jitc_var(v->dep[1]);
        fmt_metal(", $v", a1);
    }
    if (n_args >= 3) {
        Variable *a2 = jitc_var(v->dep[2]);
        fmt_metal(", $v", a2);
    }
    put(");\n");
}

// ----------------------------------------------------------------------------
//  Float64 (DD) codegen
//
//  When JitFlag::MetalEmulateFloat64 is enabled, Float64 variables are
//  represented as `dd_t` (= float2) and lowered via helpers from
//  metal_dd_preamble.h. Each branch below calls the matching
//  drjit::register_dd_*() before emitting its dd_*(...) use, which appends
//  the helper (and any transitive dependencies) to the kernel's global
//  preamble via fmt_intrinsic. jitc_register_global dedups by content
//  hash, so only the helpers actually referenced land in the kernel.
//
//  This routine handles the VarKinds whose default codegen (componentwise
//  float2 operators, plain casts, etc.) would produce wrong results.
//  Other VarKinds either fall through and work correctly (the dd_t bits
//  are opaque under Gather/Scatter/Select/Move/Copy), or are rejected
//  with a clear error.
//
//  Returns true if the VarKind was handled here.
// ----------------------------------------------------------------------------
static bool jitc_metal_render_dd(Variable *v) {
    auto bin = [&](void (*reg)(), const char *fn) {
        reg();
        Variable *a = jitc_var(v->dep[0]);
        Variable *b = jitc_var(v->dep[1]);
        fmt_metal("    dd_t $v = $s($v, $v);\n", v, fn, a, b);
    };
    auto unary = [&](void (*reg)(), const char *fn) {
        reg();
        Variable *a = jitc_var(v->dep[0]);
        fmt_metal("    dd_t $v = $s($v);\n", v, fn, a);
    };
    auto cmp = [&](void (*reg)(), const char *fn) {
        reg();
        Variable *a = jitc_var(v->dep[0]);
        Variable *b = jitc_var(v->dep[1]);
        fmt_metal("    bool $v = $s($v, $v);\n", v, fn, a, b);
    };

    switch ((VarKind) v->kind) {
        case VarKind::Literal: {
            // Out-of-input-load literal: re-emit dd_from_bits using the
            // stored 64-bit double pattern.
            double d;
            memcpy(&d, &v->literal, sizeof(double));
            float hi = (float) d;
            float lo = (float) (d - (double) hi);
            uint32_t hi_bits, lo_bits;
            memcpy(&hi_bits, &hi, sizeof(float));
            memcpy(&lo_bits, &lo, sizeof(float));
            drjit::register_dd_from_bits();
            fmt_metal("    dd_t $v = dd_from_bits(0x$xu, 0x$xu);\n",
                      v, hi_bits, lo_bits);
            return true;
        }

        case VarKind::Add: bin(drjit::register_dd_add, "dd_add"); return true;
        case VarKind::Sub: bin(drjit::register_dd_sub, "dd_sub"); return true;
        case VarKind::Mul: bin(drjit::register_dd_mul, "dd_mul"); return true;
        case VarKind::Min: bin(drjit::register_dd_min, "dd_min"); return true;
        case VarKind::Max: bin(drjit::register_dd_max, "dd_max"); return true;
        case VarKind::Neg: unary(drjit::register_dd_neg, "dd_neg"); return true;
        case VarKind::Abs: unary(drjit::register_dd_abs, "dd_abs"); return true;

        case VarKind::Fma: {
            drjit::register_dd_fma();
            Variable *a0 = jitc_var(v->dep[0]),
                     *a1 = jitc_var(v->dep[1]),
                     *a2 = jitc_var(v->dep[2]);
            fmt_metal("    dd_t $v = dd_fma($v, $v, $v);\n",
                      v, a0, a1, a2);
            return true;
        }

        case VarKind::Eq:  cmp(drjit::register_dd_eq, "dd_eq"); return true;
        case VarKind::Neq: cmp(drjit::register_dd_ne, "dd_ne"); return true;
        case VarKind::Lt:  cmp(drjit::register_dd_lt, "dd_lt"); return true;
        case VarKind::Le:  cmp(drjit::register_dd_le, "dd_le"); return true;
        case VarKind::Gt:  cmp(drjit::register_dd_gt, "dd_gt"); return true;
        case VarKind::Ge:  cmp(drjit::register_dd_ge, "dd_ge"); return true;

        case VarKind::Cast: {
            // Cast TO Float64 (v->type == Float64): wrap source via dd_from_*
            // Cast FROM Float64 (a->type == Float64): unwrap via dd_to_*
            Variable *a = jitc_var(v->dep[0]);
            VarType src_t = (VarType) a->type;
            VarType dst_t = (VarType) v->type;

            if (dst_t == VarType::Float64) {
                const char *from = nullptr;
                void (*reg)() = nullptr;
                switch (src_t) {
                    case VarType::Float32:
                        from = "dd_from_f32"; reg = drjit::register_dd_from_f32; break;
                    case VarType::Float16:
                        from = "dd_from_half"; reg = drjit::register_dd_from_half; break;
                    case VarType::Int8:
                    case VarType::Int16:
                    case VarType::Int32:
                        from = "dd_from_i32"; reg = drjit::register_dd_from_i32; break;
                    case VarType::UInt8:
                    case VarType::UInt16:
                    case VarType::UInt32:
                        from = "dd_from_u32"; reg = drjit::register_dd_from_u32; break;
                    case VarType::Int64:
                        from = "dd_from_i64"; reg = drjit::register_dd_from_i64; break;
                    case VarType::UInt64:
                        from = "dd_from_u64"; reg = drjit::register_dd_from_u64; break;
                    case VarType::Bool:
                        from = "dd_from_bool"; reg = drjit::register_dd_from_bool; break;
                    case VarType::Float64:
                        // Identity (rare, but valid).
                        fmt_metal("    dd_t $v = $v;\n", v, a);
                        return true;
                    default:
                        jitc_fail("jitc_metal_render_dd(): unsupported "
                                  "Cast source type for Float64 dest.");
                }
                reg();
                fmt_metal("    dd_t $v = $s($v);\n", v, from, a);
                return true;
            }

            if (src_t == VarType::Float64) {
                const char *to = nullptr;
                void (*reg)() = nullptr;
                switch (dst_t) {
                    case VarType::Float32:
                        to = "dd_to_f32"; reg = drjit::register_dd_to_f32; break;
                    case VarType::Float16:
                        to = "dd_to_half"; reg = drjit::register_dd_to_half; break;
                    case VarType::Int8:
                    case VarType::Int16:
                    case VarType::Int32:
                        to = "dd_to_i32"; reg = drjit::register_dd_to_i32; break;
                    case VarType::UInt8:
                    case VarType::UInt16:
                    case VarType::UInt32:
                        to = "dd_to_u32"; reg = drjit::register_dd_to_u32; break;
                    case VarType::Int64:
                        to = "dd_to_i64"; reg = drjit::register_dd_to_i64; break;
                    case VarType::UInt64:
                        to = "dd_to_u64"; reg = drjit::register_dd_to_u64; break;
                    case VarType::Bool:
                        to = "dd_to_bool"; reg = drjit::register_dd_to_bool; break;
                    default:
                        jitc_fail("jitc_metal_render_dd(): unsupported "
                                  "Cast dest type from Float64.");
                }
                reg();
                fmt_metal("    $t $v = $s($v);\n", v, v, to, a);
                return true;
            }
            return false;
        }

        case VarKind::Div:
        case VarKind::DivApprox: bin(drjit::register_dd_div, "dd_div"); return true;
        case VarKind::Rcp:
        case VarKind::RcpApprox: unary(drjit::register_dd_rcp, "dd_rcp"); return true;
        case VarKind::Sqrt:
        case VarKind::SqrtApprox: unary(drjit::register_dd_sqrt, "dd_sqrt"); return true;
        case VarKind::RSqrtApprox: unary(drjit::register_dd_rsqrt, "dd_rsqrt"); return true;
        case VarKind::Sin:  unary(drjit::register_dd_sin,  "dd_sin");  return true;
        case VarKind::Cos:  unary(drjit::register_dd_cos,  "dd_cos");  return true;
        case VarKind::Exp2: unary(drjit::register_dd_exp2, "dd_exp2"); return true;
        case VarKind::Log2: unary(drjit::register_dd_log2, "dd_log2"); return true;
        case VarKind::Tanh: unary(drjit::register_dd_tanh, "dd_tanh"); return true;
        case VarKind::Ceil:  unary(drjit::register_dd_ceil,  "dd_ceil");  return true;
        case VarKind::Floor: unary(drjit::register_dd_floor, "dd_floor"); return true;
        case VarKind::Round: unary(drjit::register_dd_round, "dd_round"); return true;
        case VarKind::Trunc: unary(drjit::register_dd_trunc, "dd_trunc"); return true;

        case VarKind::Mod: bin(drjit::register_dd_mod, "dd_mod"); return true;

        default:
            return false;  // fall through to default switch
    }
}


static void jitc_metal_render(Variable *v) {
    // Cooperative vectors have a dedicated lowering path mirroring
    // jitc_optix_render_coop_vec / jitc_llvm_render_coop_vec. The
    // coop_vec flag is set on every variable whose value IS a coopvec,
    // so this catches CoopVecLiteral/Pack/Load/Cast/UnaryOp/BinaryOp/
    // TernaryOp/MatVec. Operations whose output is a regular scalar
    // (CoopVecUnpack/Accum/OuterProductAccum) fall through to the
    // switch below and are dispatched by VarKind.
    if (v->coop_vec) {
        Variable *a0 = v->dep[0] ? jitc_var(v->dep[0]) : nullptr,
                 *a1 = v->dep[1] ? jitc_var(v->dep[1]) : nullptr,
                 *a2 = v->dep[2] ? jitc_var(v->dep[2]) : nullptr,
                 *a3 = v->dep[3] ? jitc_var(v->dep[3]) : nullptr;
        return jitc_metal_render_coop_vec(v, a0, a1, a2, a3);
    }

    // Float64 (DD) lowering: intercept VarKinds whose default codegen would
    // produce wrong results on a `dd_t = float2`. Compares are dispatched
    // here when the *operands* are Float64 (the result is Bool); arithmetic
    // and casts are dispatched when the result type is Float64. The helper
    // returns false for VarKinds it doesn't claim, in which case we fall
    // through to the default switch (Gather, Scatter, Select work as-is).
    bool is_f64_result = (VarType) v->type == VarType::Float64;
    bool dep0_is_f64 =
        v->dep[0] && (VarType) jitc_var(v->dep[0])->type == VarType::Float64;
    // Any time we touch a Float64 var we'll emit `dd_t` somewhere via the
    // `$t` formatter — register the typedef once per kernel so the DD
    // helpers compile even when no dd_* call is involved.
    if (is_f64_result || dep0_is_f64)
        drjit::register_dd_t();
    if (is_f64_result || (dep0_is_f64 &&
                          ((VarKind) v->kind == VarKind::Eq ||
                           (VarKind) v->kind == VarKind::Neq ||
                           (VarKind) v->kind == VarKind::Lt ||
                           (VarKind) v->kind == VarKind::Le ||
                           (VarKind) v->kind == VarKind::Gt ||
                           (VarKind) v->kind == VarKind::Ge ||
                           (VarKind) v->kind == VarKind::Cast))) {
        if (jitc_metal_render_dd(v))
            return;
    }

    switch ((VarKind) v->kind) {
        case VarKind::Nop:
            break;

        case VarKind::Literal: {
            VarType vt = (VarType) v->type;
            if (vt == VarType::Float32)
                fmt_metal("    $t $v = as_type<float>($lu);\n", v, v, v);
            else if (vt == VarType::Float16)
                fmt_metal("    $t $v = as_type<half>((ushort) $lu);\n", v, v, v);
            else if (vt == VarType::Bool)
                fmt_metal("    $t $v = ($t) ($lu);\n", v, v, v, v);
            else
                fmt_metal("    $t $v = ($t) $lu;\n", v, v, v, v);
            break;
        }

        case VarKind::Counter:
            fmt_metal("    $t $v = ($t) r0;\n", v, v, v);
            break;

        // -- Arithmetic --
        case VarKind::Add: jitc_metal_render_binary(v, "+"); break;
        case VarKind::Sub: jitc_metal_render_binary(v, "-"); break;
        case VarKind::Mul: jitc_metal_render_binary(v, "*"); break;
        case VarKind::Div:
        case VarKind::DivApprox:
            jitc_metal_render_binary(v, "/");
            break;
        case VarKind::Mod: jitc_metal_render_binary(v, "%"); break;

        case VarKind::MulHi: {
            Variable *a = jitc_var(v->dep[0]);
            Variable *b = jitc_var(v->dep[1]);
            fmt_metal("    $t $v = mulhi($v, $v);\n", v, v, a, b);
            break;
        }

        case VarKind::MulWide: {
            // 32-bit × 32-bit → 64-bit
            Variable *a = jitc_var(v->dep[0]);
            Variable *b = jitc_var(v->dep[1]);
            fmt_metal("    $t $v = ($t)$v * ($t)$v;\n", v, v, v, a, v, b);
            break;
        }

        case VarKind::Fma: {
            Variable *a0 = jitc_var(v->dep[0]),
                     *a1 = jitc_var(v->dep[1]),
                     *a2 = jitc_var(v->dep[2]);
            if (jitc_is_float(v))
                fmt_metal("    $t $v = fma($v, $v, $v);\n", v, v, a0, a1, a2);
            else
                fmt_metal("    $t $v = $v * $v + $v;\n", v, v, a0, a1, a2);
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
            fmt_metal("    $t $v = $v ? $v : $v;\n", v, v, cond, a, b);
            break;
        }

        // -- Bit operations --
        case VarKind::Popc:  jitc_metal_render_call(v, "popcount", 1); break;
        case VarKind::Clz:   jitc_metal_render_call(v, "clz", 1); break;
        case VarKind::Ctz:   jitc_metal_render_call(v, "ctz", 1); break;
        case VarKind::Brev: {
            Variable *a = jitc_var(v->dep[0]);
            fmt_metal("    $t $v = reverse_bits($v);\n", v, v, a);
            break;
        }

        case VarKind::And: {
            Variable *a0 = jitc_var(v->dep[0]),
                     *a1 = jitc_var(v->dep[1]);
            if (a0->type != a1->type)
                fmt_metal("    $t $v = $v ? $v : ($t) 0;\n", v, v, a1, a0, v);
            else if (jitc_is_float(v))
                fmt_metal("    $t $v = as_type<$t>(($b)(as_type<$b>($v) & as_type<$b>($v)));\n",
                          v, v, v, v, v, a0, v, a1);
            else
                fmt_metal("    $t $v = $v & $v;\n", v, v, a0, a1);
            break;
        }

        case VarKind::Or: {
            Variable *a0 = jitc_var(v->dep[0]),
                     *a1 = jitc_var(v->dep[1]);
            if (a0->type != a1->type)
                fmt_metal("    $t $v = as_type<$t>(($b)($v ? ($b) ~0u : as_type<$b>($v)));\n",
                          v, v, v, v, a1, v, v, a0);
            else if (jitc_is_float(v))
                fmt_metal("    $t $v = as_type<$t>(($b)(as_type<$b>($v) | as_type<$b>($v)));\n",
                          v, v, v, v, v, a0, v, a1);
            else
                fmt_metal("    $t $v = $v | $v;\n", v, v, a0, a1);
            break;
        }

        case VarKind::Xor: {
            Variable *a0 = jitc_var(v->dep[0]),
                     *a1 = jitc_var(v->dep[1]);
            if (jitc_is_float(v))
                fmt_metal("    $t $v = as_type<$t>(($b)(as_type<$b>($v) ^ as_type<$b>($v)));\n",
                          v, v, v, v, v, a0, v, a1);
            else
                fmt_metal("    $t $v = $v ^ $v;\n", v, v, a0, a1);
            break;
        }
        case VarKind::Shl: jitc_metal_render_binary(v, "<<"); break;
        case VarKind::Shr: jitc_metal_render_binary(v, ">>"); break;

        // -- Unary --
        case VarKind::Neg: jitc_metal_render_unary(v, "-"); break;
        case VarKind::Not: {
            Variable *a = jitc_var(v->dep[0]);
            if ((VarType) v->type == VarType::Bool)
                fmt_metal("    $t $v = !$v;\n", v, v, a);
            else
                fmt_metal("    $t $v = ~$v;\n", v, v, a);
            break;
        }
        case VarKind::Abs: jitc_metal_render_unary(v, "abs"); break;

        case VarKind::Sqrt:
        case VarKind::SqrtApprox:
            jitc_metal_render_call(v, "sqrt", 1);
            break;

        // -- Math functions --
        case VarKind::Sin:  jitc_metal_render_call(v, "sin",  1); break;
        case VarKind::Cos:  jitc_metal_render_call(v, "cos",  1); break;
        case VarKind::Exp2: jitc_metal_render_call(v, "exp2", 1); break;
        case VarKind::Log2: jitc_metal_render_call(v, "log2", 1); break;
        case VarKind::Tanh: jitc_metal_render_call(v, "tanh", 1); break;

        // -- Reciprocal --
        case VarKind::Rcp:
        case VarKind::RcpApprox:
            fmt_metal("    $t $v = ($t) 1.0 / $v;\n",
                      v, v, v, jitc_var(v->dep[0]));
            break;

        case VarKind::RSqrtApprox:
            jitc_metal_render_call(v, "rsqrt", 1);
            break;

        // -- Casts --
        case VarKind::Cast: {
            Variable *a = jitc_var(v->dep[0]);
            fmt_metal("    $t $v = ($t) $v;\n", v, v, v, a);
            break;
        }

        case VarKind::Bitcast: {
            Variable *a = jitc_var(v->dep[0]);
            if (v->type == a->type) {
                fmt_metal("    $t $v = $v;\n", v, v, a);
            } else if (type_size[v->type] == type_size[a->type]) {
                fmt_metal("    $t $v = as_type<$t>($v);\n", v, v, v, a);
            } else {
                fmt_metal("    $t $v = as_type<$t>(($b) $v);\n", v, v, v, v, a);
            }
            break;
        }

        // -- Bounds check --
        case VarKind::BoundsCheck: {
            Variable *index  = jitc_var(v->dep[0]);
            Variable *mask   = jitc_var(v->dep[1]);
            Variable *buf    = jitc_var(v->dep[2]);
            uint32_t size    = (uint32_t) v->literal;
            fmt_metal("    bool $v = $v && ($v < (uint) $uu);\n"
                      "    if ($v && !$v)\n"
                      "        atomic_store_explicit("
                      "(device atomic_uint*) $v, $v, "
                      "memory_order_relaxed);\n",
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
                fmt_metal("    $t $v = ((device const $t*) $v)[$v];\n",
                          v, v, v, src, index);
            else
                fmt_metal("    $t $v = ($v) ? "
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
                fmt_metal("    $t $v = $v_out_$u;\n",
                          v, v, src, sub_index);
            } else {
                fmt_metal("    $t $v = $v; // extract[$u]\n",
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
                    fmt_metal("    $t $v = $v;\n", inner_in, inner_in, outer_in);
                else
                    fmt_metal("    $t $v = ($t) 0;\n", inner_in, inner_in, inner_in);
            }
            put("    while (true) {\n");
            break;
        }

        case VarKind::LoopCond: {
            Variable *cond = jitc_var(v->dep[1]);
            fmt_metal("        if (!$v) break;\n", cond);
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
                fmt_metal("    $t $v_tmp = $v;\n", in, in, out);
                out->scratch = 2;
            }

            for (uint32_t i = 0; i < size; ++i) {
                Variable *in  = jitc_var(ld->inner_in[i]),
                         *out = jitc_var(ld->inner_out[i]);
                if (in == out || !in->reg_index || !out->reg_index ||
                    in->is_array())
                    continue;
                if (out->scratch == 2)
                    fmt_metal("    $v = $v_tmp;\n", in, in);
                else
                    fmt_metal("    $v = $v;\n", in, out);
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
                        fmt_metal("    $t $v = $v;\n", v, v, inner_in);
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
                    fmt_metal("    $t $v;\n", vo, vo);
            }
            fmt_metal("    if ($v) {\n", cond);
            break;
        }

        case VarKind::CondMid: {
            Variable *a0 = jitc_var(v->dep[0]);
            const CondData *cd = (CondData *) a0->data;
            for (size_t i = 0; i < cd->indices_out.size(); ++i) {
                Variable *vt = jitc_var(cd->indices_t[i]),
                         *vo = jitc_var(cd->indices_out[i]);
                if (vo && vo->reg_index && vt->reg_index)
                    fmt_metal("    $v = $v;\n", vo, vt);
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
                    fmt_metal("    $v = $v;\n", vo, vf);
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
            // use for THIS trace by looking up the trace's scene in
            // ``metal_kernel_scene_slot``. If the scene wasn't
            // registered (shouldn't happen — the recursive pre-walk
            // visits every TraceRay reachable from the kernel), or no
            // scene at all, fall back to the active scene at slot 0.
            MetalScene *scene_local = nullptr;
            if (v->dep[1])
                scene_local = jitc_metal_get_scene(v->dep[1]);
            if (!scene_local)
                scene_local = jitc_metal_active_scene();

            uint32_t accel_idx = 0;
            if (scene_local) {
                auto it = metal_kernel_scene_slot.find(scene_local);
                if (it != metal_kernel_scene_slot.end())
                    accel_idx = it->second;
            }

            // Guard with active mask
            if (!is_unmasked)
                fmt_metal("    if ($v) {\n", valid);

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
            fmt_metal("        metal::raytracing::ray _r;\n"
                      "        _r.origin = float3($v, $v, $v);\n"
                      "        _r.direction = float3($v, $v, $v);\n"
                      "        _r.min_distance = $v;\n"
                      "        _r.max_distance = $v;\n",
                      ox, oy, oz, dx, dy, dz, tmin, tmax);
            // Route the intersect call to this scene's kernel-bound
            // accel + IFT (per-scene slots assigned at signature time).
            if (has_ift_local)
                fmt_metal("        auto _hit = _inter.intersect(_r, accel_$u, ift_$u);\n",
                          accel_idx, accel_idx);
            else
                fmt_metal("        auto _hit = _inter.intersect(_r, accel_$u);\n",
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
                fmt_metal("        auto _ht = _hit.type;\n"
                          "        bool _zero_curve_hit = "
                          "(_ht == metal::raytracing::intersection_type::curve) "
                          "&& (_hit.distance <= 0.0f);\n"
                          "        $v_out_0 = (_ht != metal::raytracing::intersection_type::none) "
                          "&& !_zero_curve_hit;\n"
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
                fmt_metal("        $v_out_0 = (_hit.type != metal::raytracing::intersection_type::none);\n"
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

        // -- Features requiring Metal-specific implementations --
        // These CUDA/OptiX-specific features need dedicated Metal ports:
        //
        // Textures: Metal uses MTLTexture + sampler (not CUtexObject).
        //   Requires a new metal_tex.cpp with MTLTextureDescriptor,
        //   MTLSamplerDescriptor, and MSL texture2d<T>::sample() codegen.
        //
        // Virtual calls: Metal has no PTX-style indirect function calls.
        //   Options: (1) switch-case dispatch, (2) visible_function_table,
        //   (3) function pointers via argument buffers.
        //   See LuisaCompute's metal backend for a reference.
        //
        // Cooperative vectors: lowered via metal_coop_vec.cpp. Variables whose
        //   *output* is a coopvec are caught by the early dispatch at the top
        //   of this function (`if (v->coop_vec)`); the three ops below
        //   produce scalar outputs and so reach this switch.

        case VarKind::TexLookup:
            jitc_fail("jitc_metal_render(): texture lookup (VarKind::TexLookup) "
                      "is not yet implemented on the Metal backend. Metal "
                      "requires MTLTexture + sampler infrastructure (vs CUDA's "
                      "CUtexObject). This is needed for Mitsuba's texture "
                      "evaluation.");

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
            fmt_metal("    uint $v = self;\n", v);
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

        // -- Genuinely unsupported / irrelevant on Metal --
        case VarKind::DefaultMask:   // LLVM-only (SIMD lane masking)
        case VarKind::ThreadIndex:   // LLVM-only
        case VarKind::CallMask:      // LLVM-only
        case VarKind::PacketGather:  // LLVM packet mode
        case VarKind::PacketScatter: // LLVM packet mode
        case VarKind::VectorLoad:    // OptiX SBT data load
        case VarKind::ReorderThread: // OptiX SER
            jitc_fail("jitc_metal_render(): VarKind::%s is not applicable "
                      "to the Metal backend.",
                      var_kind_name[v->kind]);

        default:
            jitc_fail("jitc_metal_render(): unhandled VarKind=%u (%s) for "
                      "variable r%u (type=%s).",
                      (uint32_t) v->kind, var_kind_name[v->kind],
                      v->reg_index, type_name[v->type]);
    }
}

// ============================================================================
//
//  Virtual function call support (switch-case dispatch)
//
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
            fmt_metal("instance_acceleration_structure accel_$u, ",
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
            fmt_metal("$s ift_$u, ", ift_template, (uint32_t) i);
        }
    }

    // Remove trailing ", "
    buffer.delete_trailing_commas();

    fmt_metal(") {\n"
              "    // Call: $s\n",
              call->name.c_str());

    for (size_t i = 0; i < schedule.size(); ++i) {
        ScheduledVariable &sv = schedule[i];
        Variable *v = jitc_var(sv.index);
        VarType vt = (VarType) v->type;
        VarKind kind = (VarKind) v->kind;

        // Pre-declare TraceRay output variables in callable functions
        if (kind == VarKind::TraceRay) {
            fmt_metal("    bool  $v_out_0 = false;\n"
                      "    float $v_out_1 = 0.0f;\n"
                      "    float $v_out_2 = 0.0f;\n"
                      "    float $v_out_3 = 0.0f;\n"
                      "    uint  $v_out_4 = 0u;\n"
                      "    uint  $v_out_5 = 0u;\n"
                      "    uint  $v_out_6 = 0u;\n",
                      v, v, v, v, v, v, v);
        }

        if (vt == VarType::Float64)
            drjit::register_dd_t();

        if (kind == VarKind::Counter) {
            fmt_metal("    $t $v = ($t) index;\n", v, v, v);
        } else if (kind == VarKind::CallInput) {
            Variable *a = jitc_var(v->dep[0]);
            if (vt != VarType::Bool)
                fmt_metal("    $t $v = *(thread const $t*)(params + $u);\n",
                          v, v, v, a->param_offset);
            else
                fmt_metal("    bool $v = *(thread const uint8_t*)(params + $u) != 0;\n",
                          v, a->param_offset);
        } else if (kind == VarKind::CallSelf) {
            fmt_metal("    uint $v = self;\n", v);
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
                fmt_metal("    bool $v = *(device uint8_t*)(data + $u) != 0;\n",
                          v, offset);
            else if (vt == VarType::Pointer)
                fmt_metal("    device uint8_t* $v = "
                          "*(device uint8_t* device const*)(data + $u);\n",
                          v, offset);
            else
                fmt_metal("    $t $v = *(device $t*)(data + $u);\n",
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

        if ((VarType) v->type == VarType::Float64)
            drjit::register_dd_t();

        if ((VarType) v->type != VarType::Bool)
            fmt_metal("    *(thread $t*)(result + $u) = $v;\n",
                      v, offset, v);
        else
            fmt_metal("    *(thread uint8_t*)(result + $u) = $v ? 1 : 0;\n",
                      offset, v);
    }

    put("}\n");
}

void jitc_var_call_assemble_metal(CallData *call, uint32_t call_reg,
                                   uint32_t self_reg, uint32_t mask_reg,
                                   uint32_t offset_reg, uint32_t data_reg,
                                   uint32_t in_size, uint32_t in_align,
                                   uint32_t out_size, uint32_t out_align) {
    // =====================================================
    // 1. Conditional branch (masked lanes)
    // =====================================================

    Variable *mask = jitc_var(jitc_var(call->id)->dep[1]);
    bool is_masked = !mask->is_literal() || mask->literal != 1;

    fmt_metal("\n    // VCall: $s\n", call->name.c_str());

    // =====================================================
    // 2. Declare input local buffers and pack inputs
    // =====================================================

    if (in_size) {
        fmt_metal("    uint8_t _in_$u[$u];\n", call_reg, in_size);

        for (uint32_t i = 0; i < call->n_in; ++i) {
            const Variable *v = jitc_var(call->outer_in[i]);
            const Variable *v_in = jitc_var(call->inner_in[i]);
            bool unused =
                !v->reg_index ||
                (call->optimize &&
                    (v->is_literal() || v_in->ref_count == 1));
            if (unused)
                continue;

            if ((VarType) v->type == VarType::Float64)
                drjit::register_dd_t();

            if ((VarType) v->type != VarType::Bool)
                fmt_metal("    *(thread $t*)(_in_$u + $u) = $v;\n",
                          v, call_reg, v->param_offset, v);
            else
                fmt_metal("    *(thread uint8_t*)(_in_$u + $u) = $v ? 1 : 0;\n",
                          call_reg, v->param_offset, v);
        }
    }

    // =====================================================
    // 3. Declare output buffer (zero-initialized)
    // =====================================================

    if (out_size)
        fmt_metal("    uint8_t _out_$u[$u] = {};\n", call_reg, out_size);

    // =====================================================
    // 4. Masked guard + offset table lookup
    // =====================================================

    const char *indent = is_masked ? "            " : "        ";

    if (is_masked)
        fmt_metal("    if (v$u) {\n", mask_reg);

    // Load the uint64_t entry: (data_offset << 32) | callable_index
    fmt_metal("$sulong _oe_$u = ((device const ulong*) v$u)[v$u];\n"
              "$suint _ci_$u = (uint)(_oe_$u & 0xFFFFFFFFu);\n",
              indent, call_reg, offset_reg, self_reg,
              indent, call_reg, call_reg);

    if (data_reg)
        fmt_metal("$sdevice uint8_t* _cd_$u = "
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
        fmt_metal("$sfunc_", indent);
        buffer.put_q64_unchecked(hash.high64);
        buffer.put_q64_unchecked(hash.low64);
        put("(");
        if (call->use_index)
            fmt_metal("$s, ", index_name);
        if (call->use_self)
            fmt_metal("v$u, ", self_reg);
        if (data_reg)
            fmt_metal("_cd_$u, ", call_reg);
        if (in_size)
            fmt_metal("_in_$u, ", call_reg);
        if (out_size)
            fmt_metal("_out_$u, ", call_reg);
        // Forward every accel + IFT from the kernel-level bindings
        // into the callable, matching the callable's signature.
        if (uses_metal_rt) {
            for (size_t i = 0; i < metal_kernel_scenes.size(); ++i)
                fmt_metal("accel_$u, ", (uint32_t) i);
            for (size_t i = 0; i < metal_kernel_scenes.size(); ++i) {
                if (metal_kernel_ift_slot[i] >= 0)
                    fmt_metal("ift_$u, ", (uint32_t) i);
            }
        }
        buffer.delete_trailing_commas();
        put(");\n");
    } else {
        fmt_metal("$sswitch (_ci_$u) {\n", indent, call_reg);

        for (uint32_t i = 0; i < call->n_inst; ++i) {
            XXH128_hash_t hash = call->inst_hash[i];

            fmt_metal("$s    case $u: func_$Q$Q(",
                      indent, i,
                      hash.high64, hash.low64);

            if (call->use_index)
                fmt_metal("$s, ", index_name);
            if (call->use_self)
                fmt_metal("v$u, ", self_reg);
            if (data_reg)
                fmt_metal("_cd_$u, ", call_reg);
            if (in_size)
                fmt_metal("_in_$u, ", call_reg);
            if (out_size)
                fmt_metal("_out_$u, ", call_reg);
            // Forward all kernel-bound scene resources into each
            // switch-case branch (mirror of the single-instance path
            // above).
            if (uses_metal_rt) {
                for (size_t i = 0; i < metal_kernel_scenes.size(); ++i)
                    fmt_metal("accel_$u, ", (uint32_t) i);
                for (size_t i = 0; i < metal_kernel_scenes.size(); ++i) {
                    if (metal_kernel_ift_slot[i] >= 0)
                        fmt_metal("ift_$u, ", (uint32_t) i);
                }
            }
            buffer.delete_trailing_commas();
            put("); break;\n");
        }

        fmt_metal("$s    default: break;\n"
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

        if ((VarType) v->type == VarType::Float64)
            drjit::register_dd_t();

        if ((VarType) v->type != VarType::Bool)
            fmt_metal("    $t $v = *(thread const $t*)(_out_$u + $u);\n",
                      v, v, v, call_reg, offset);
        else
            fmt_metal("    bool $v = *(thread const uint8_t*)(_out_$u + $u) != 0;\n",
                      v, call_reg, offset);
    }

    put("\n");
}

#endif // defined(DRJIT_ENABLE_METAL)
