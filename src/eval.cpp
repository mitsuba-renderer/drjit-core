/*
    src/eval.cpp -- Main computation graph evaluation routine

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "internal.h"
#include "llvm.h"
#include "log.h"
#include "var.h"
#include "eval.h"
#include "profile.h"
#include "util.h"
#include "optix.h"
#include "loop.h"
#include "call.h"
#include "coop_vec.h"
#include "trace.h"
#include "tex.h"
#include "op.h"
#include "array.h"

#if defined(DRJIT_ENABLE_METAL)
#  include "metal.h"
#endif

// ====================================================================
//  The following data structures are temporarily used during program
//  generation. They are declared as global variables to enable memory
//  reuse across jitc_eval() calls. Access to them is protected by the
//  central Dr.Jit mutex.
// ====================================================================

/// Ordered list of variables that should be computed
std::vector<ScheduledVariable> schedule;

/// Groups of variables with the same size
static std::vector<ScheduledGroup> schedule_groups;

/// Scratch buffer reused by the schedule sort
static std::vector<ScheduledVariable> schedule_scratch;

struct VisitedKey {
    uint32_t size;
    uint32_t index;
    uint32_t depth;
    VisitedKey(uint32_t size, uint32_t index, uint32_t depth)
        : size(size), index(index), depth(depth) { }
};

/// A traversal "task": a root or side effect to be visited at a given size
struct EvalTask {
    uint32_t size;
    uint32_t index;
    bool side_effect;
};

/// Roots & side effects collected at the start of jit_eval(), grouped by size
static std::vector<EvalTask> eval_tasks;

/// Indices of the explicitly scheduled variables (to be marked as outputs)
static std::vector<uint32_t> eval_roots;

/* Generation counter used by jitc_var_traverse() to detect already-visited
   variables. The traversal stamps the current generation into 'Variable::scratch'
   and treats a variable as visited iff its stamp belongs to 'visit_gen'. Starting
   a new traversal then takes only a single increment of 'visit_gen', which makes
   every existing stamp stale and so marks all variables unvisited.

   jitc_eval_impl() starts a fresh generation for each group of equally-sized
   variables, so the traversal visits (and schedules) a shared sub-expression
   exactly once for each distinct kernel size it feeds into. The generation only
   needs to encode size, as the traversal size stays constant within a group.

   The stamp's least significant bit records whether the visit happened at
   depth 0 — the only depth at which variables are scheduled. Most variables
   are only ever reachable at a single depth (call/loop boundaries route outer
   references through CallInput nodes), but dependency-free variables such as
   literals and evaluated arrays are deduplicated across scopes and may be
   referenced both from within a call/loop body (depth > 0) and from the
   surrounding kernel (depth == 0). A visit at depth > 0 must therefore not
   elide a later depth-0 visit, which would leave the variable unscheduled.

   Generation 0 serves as the "never visited" sentinel. A freshly constructed
   variable has scratch == 0, stamps are always >= 2, so the traversal
   correctly treats such a variable as unvisited. On the rare 31-bit wraparound,
   reset all stamps to 0 and restart. */
uint32_t visit_gen = 0;

void jitc_visit_new_gen() {
    if (unlikely(visit_gen == 0x7FFFFFFFu)) {
        for (Variable &v : state.variables)
            v.scratch = 0;
        visit_gen = 0;
    }
    ++visit_gen;
}

/// Kernel parameter buffer and variable ids
static std::vector<void *> kernel_params;
static std::vector<uint32_t> kernel_param_ids;

/// Call-data base pointer variables created during this jit_eval()
/// (one per kernel that performs calls). Released after the launch barrier.
static std::vector<uint32_t> call_base_vars;

/// Metadata for kernel parameters
std::vector<KernelParamInfo> kernel_param_info;

/// Ensure uniqueness of globals/callables arrays
GlobalsMap globals_map;

/// StringBuffer for global definitions (intrinsics, callables, etc.)
StringBuffer globals { 1000 };

/// Hash code of the last generated kernel
XXH128_hash_t kernel_hash { 0, 0 };

/// Name of the last generated kernel
char kernel_name[52 /* strlen("__direct_callable__") + 32 + 1 */] { };

// Total number of operations used across the entire kernel (including functions)
static uint32_t n_ops_total = 0;

/// Are we recording an OptiX kernel?
bool uses_optix = false;

bool uses_metal4 = false;

/// Size and alignment of auxiliary buffer needed by virtual function calls
int32_t alloca_size = -1;
int32_t alloca_align = -1;

/// Number of tentative indirect callables that were assembled in the kernel being compiled
uint32_t indirect_callable_count = 0;

/// Number of unique indirect callables in the kernel being compiled
uint32_t indirect_callable_count_unique = 0;

/// Specifies the nesting level of virtual calls being compiled
uint32_t callable_depth = 0;

/// Information about the kernel launch to go in the kernel launch history
KernelHistoryEntry kernel_history_entry;

/// List of enqueued callbacks (bound checks, async dr.print statements, etc.)
static std::vector<uint32_t> eval_callbacks;

/// Temporary todo list needed to correctly process loops in jitc_var_traverse()
static std::vector<VisitedKey> visit_later;

// ====================================================================

// Check whether we can elide the given scatter operation, for example if its
// output buffer is found to be unreferenced.
bool jitc_elide_scatter(uint32_t index, const Variable *v) {
    if ((VarKind) v->kind != VarKind::Scatter)
        return false;
    Variable *target = jitc_var(v->dep[0]);
    Variable *target_ptr = jitc_var(target->dep[3]);
    jitc_log(Debug, "jit_eval(): eliding scatter r%u, whose output is unreferenced.", index);
    return target_ptr->ref_count == 0;
}

/// Recursively traverse the computation graph to find variables needed by a computation
static void jitc_var_traverse(uint32_t size, uint32_t index, uint32_t depth = 0) {
    Variable *v = jitc_var(index);
    uint32_t stamp = (visit_gen << 1) | (depth == 0 ? 1u : 0u);
    if (v->scratch == stamp || (depth != 0 && v->scratch == (stamp | 1u)))
        return;
    v->scratch = stamp;
    switch ((VarKind) v->kind) {
        case VarKind::Scatter:
            if (jitc_elide_scatter(index, v))
                return;
            break;

        case VarKind::ArrayWrite:
            jitc_var_traverse(size, jitc_array_buffer(index), depth);
            break;

        case VarKind::LoopPhi: {
                LoopData *loop = (LoopData *) jitc_var(v->dep[0])->data;
                if (!loop)
                    jitc_raise("jit_var_traverse(): internal error: "
                               "computation references variables from a loop "
                               "that was optimized away!");
                jitc_var_traverse(size, loop->outer_in[v->literal], depth);
                visit_later.emplace_back(
                    size, loop->inner_out[v->literal], depth);
            }
            break;

        case VarKind::LoopOutput: {
                LoopData *loop = (LoopData *) jitc_var(v->dep[0])->data;
                if (!loop)
                    jitc_raise("jit_var_traverse(): internal error: "
                               "computation references variables from a loop "
                               "that was optimized away!");
                jitc_var_traverse(size, loop->inner_out[v->literal], depth);
                jitc_var_traverse(size, loop->outer_in[v->literal], depth);
                jitc_var_traverse(size, loop->inner_in[v->literal], depth);
            }
            break;

        case VarKind::CallInput: {
                if (depth == 0) {
                    schedule.emplace_back(size, v->scope, index);
                    jitc_var_inc_ref(index, v);
                    return;
                } else {
                    jitc_var_traverse(size, v->dep[0], depth - 1);
                    return;
                }
            };

        case VarKind::Call: {
                CallData *call = (CallData *) v->data;
                if (unlikely(!call->optimize)) {
                    for (uint32_t i = 0; i < call->n_in; ++i)
                        jitc_var_traverse(size, call->outer_in[i], depth);
                    for (uint32_t i = 0; i < call->n_inst; ++i)
                        for (uint32_t j = 0; j < call->n_out; ++j)
                            jitc_var_traverse(size, call->inner_out[j + i*call->n_out], depth + 1);
                }
                for (uint32_t i : call->side_effects)
                    jitc_var_traverse(size, i, depth + 1);
            };
            break;

        case VarKind::CallOutput: {
                CallData *call = (CallData *) jitc_var(v->dep[0])->data;
                for (uint32_t i = 0; i < call->n_inst; ++i)
                    jitc_var_traverse(size, call->inner_out[v->literal + i * call->n_out], depth + 1);

            }
            break;

        case VarKind::CoopVecPack: {
                CoopVecPackData *cvid = (CoopVecPackData *) v->data;
                for (uint32_t index2 : cvid->indices) {
                    if (index2 == 0)
                        continue;
                    jitc_var_traverse(size, index2, depth);
                }
            }
            break;

#if defined(DRJIT_ENABLE_OPTIX)
        case VarKind::CoopVecMatVec:
            jitc_optix_max_coopvec_size = std::max(
                std::max(
                    jitc_optix_max_coopvec_size,
                    (uint32_t) jitc_var(v->dep[1])->array_length
                ),
                (uint32_t) v->array_length
            );
            break;
#endif

        case VarKind::TraceRay: {
                TraceData *call = (TraceData *) v->data;
                for (uint32_t i: call->indices)
                    jitc_var_traverse(size, i, depth);
#if defined(DRJIT_ENABLE_OPTIX)
                if (call->reorder && call->reorder_hint)
                    jitc_var_traverse(size, call->reorder_hint, depth);
#endif
            }
            break;

        case VarKind::PacketScatter: {
                PacketScatterData *psd = (PacketScatterData *) v->data;
                for (uint32_t i : psd->values)
                    jitc_var_traverse(size, i, depth);
            }
            break;

        case VarKind::TexLookup:
        case VarKind::TexFetchBilerp:
        case VarKind::TexWrite: {
                // Texture writes always reference their coordinates and values.
                // On Metal, reads do too, since the coordinates are passed
                // out-of-band.
                bool refs_coords =
                    (VarKind) v->kind == VarKind::TexWrite ||
                    (JitBackend) v->backend == JitBackend::Metal;

                if (v->data && refs_coords) {
                    TexData *td = (TexData *) v->data;
                    for (uint32_t i = 0; i < td->ndim; ++i)
                        jitc_var_traverse(size, td->indices[i], depth);
                    for (uint32_t i = 0; i < td->n_values; ++i)
                        jitc_var_traverse(size, td->values[i], depth);
                }
            }
            break;

        case VarKind::ScatterCAS: {
                ScatterCASDData *cas_data = (ScatterCASDData *) v->data;
                jitc_var_traverse(size, cas_data->mask, depth);
            }
            break;

        case VarKind::CallGetter:
            // Only the [index, mask] deps (walked below) are scheduled; the
            // value table is captured into the buffer at upload, like 'Call'.
            break;

        default:
            break;
    }

    for (int i = 0; i < 4; ++i) {
        uint32_t index2 = v->dep[i];
        if (index2 == 0)
            break;
        jitc_var_traverse(size, index2, depth);
    }

    if (unlikely(v->consumed))
        jitc_raise(
            "Trying to launch a kernel that depends on a variable that was "
            "already consumed!"
            "\nThe variable r%u was already consumed by a prior kernel, it "
            "cannot be re-computed withing this kernel. You must explicitly "
            "evalute the consumed variables if they are required by operations "
            "in an other kernel."
            "\nThis can happen when trying to re-use the outputs of the "
            "dr.scatter_inc(), dr.scatter_exch() or dr.sactter_cas() "
            "operations.", index);

    if (depth == 0) {
        // A scheduled variable is an output only if it was explicitly requested;
        // clear the flag here and let jitc_eval_impl() set it on the roots once
        // all traversal is done.
        v->output_flag = false;
        schedule.emplace_back(size, v->scope, index);
        jitc_var_inc_ref(index, v);
    }
}

void jitc_assemble(ThreadState *ts, ScheduledGroup group) {
    JitBackend backend = ts->backend;

    kernel_params.clear();
    kernel_param_ids.clear();
    kernel_param_info.clear();
    globals.clear();
    globals_map.clear();
    alloca_size = alloca_align = -1;
    indirect_callable_count = 0;
    indirect_callable_count_unique = 0;
    call_buffer.reset();
    uses_metal4 = false;
#if defined(DRJIT_ENABLE_METAL)
    jitc_metal_assemble_reset();
#endif
    kernel_history_entry = { };

#if defined(DRJIT_ENABLE_OPTIX)
    uses_optix = jitc_is_cuda(ts->backend) &&
                 jit_flag(JitFlag::ForceOptiX);
#endif

    uint32_t n_params_in    = 0,
             n_params_out   = 0,
             n_side_effects = 0,
             n_regs         = 0,
             width = jitc_llvm_vector_width;

    // Append a kernel parameter and its metadata. The two vectors grow in
    // lockstep so ``kernel_param_info`` always stays parallel to ``kernel_params``
    // (read-only plain buffer by default).
    auto add_param = [](void *value, uint8_t write = 0, uint8_t kind = 0) {
        kernel_params.push_back(value);
        kernel_param_info.push_back({ write, kind });
    };

    if (jitc_is_gpu(backend)) {
        add_param((void *) (uintptr_t) group.size);

        // CUDA reserves r0..r3 for the thread-index computation (ctaid/ntid/tid)
        // and compound-op temporaries. Metal receives the thread index directly
        // via [[thread_position_in_grid]] (r0), so locals can start at r1.
        n_regs = jitc_is_metal(backend) ? 1 : 4;
    } else {
        // First 3 parameters reserved for: kernel ptr, size, ITT identifier
        for (int i = 0; i < 3; ++i)
            add_param(nullptr);
        n_regs = 1;
    }

    (void) timer();

    // Set while scanning the schedule below if this kernel performs any
    // indirect call (used further down to reserve the call base pointer).
    // 'needs_call_buffer' additionally covers getters, which read from the same
    // shared buffer but emit no visible function table.
    bool has_call = false, needs_call_buffer = false;

    for (uint32_t group_index = group.start; group_index != group.end; ++group_index) {
        ScheduledVariable &sv = schedule[group_index];
        uint32_t index = sv.index;
        Variable *v = jitc_var(index);

        // Some sanity checks
        if (unlikely((JitBackend) v->backend != backend))
            jitc_raise("jit_assemble(): variable r%u scheduled in wrong ThreadState", index);
        if (unlikely(v->ref_count == 0))
            jitc_fail("jit_assemble(): schedule contains unreferenced variable r%u!", index);
        if (unlikely(v->size != 1 && v->size != group.size))
            jitc_fail("jit_assemble(): schedule contains variable r%u of kind \"%s\" with incompatible size "
                     "(var=%u and kernel=%u)!", index, var_kind_name[v->kind], v->size, group.size);
        if (unlikely(v->is_dirty()))
            jitc_fail("jit_assemble(): dirty variable r%u encountered!", index);

        uint32_t scope = v->scope;
        for (int i = 0; i < 4; ++i) {
            uint32_t index2 = v->dep[i];
            if (!index2)
                break;
            Variable *v2 = jitc_var(index2);
            uint32_t scope2 = v2->scope;
            if (unlikely(scope2 > scope)) {
                jitc_raise(
                    "jitc_assemble(): variable r%u (scope %u) depends on r%u "
                    "(scope %u). However, the scope ID of predecessors must be "
                    "lower! Very likely, a computation is split across a "
                    "parent/child thread, which requires incrementing the "
                    "scope ID at relevant handoff points (via "
                    "dr.detail.new_scope()). You must do so before the child "
                    "thread accesses a variable from a parent, and before the "
                    "parent accesses a variable from the child.\n",
                    index, scope, index2, scope2);
            }
        }

        v->param_offset = (uint32_t) kernel_params.size() * sizeof(void *);
        v->reg_index = n_regs++;

        VarKind kind = (VarKind) v->kind;
        has_call |= (kind == VarKind::Call);
        needs_call_buffer |= (kind == VarKind::Call || kind == VarKind::CallGetter);

        if (unlikely(kind == VarKind::Array ||
                     kind == VarKind::ArrayInit ||
                     kind == VarKind::ArrayPhi ||
                     kind == VarKind::ArrayWrite ||
                     kind == VarKind::ArrayRead)) {
            jitc_process_array_op(kind, v);
        }

        if (v->is_evaluated()) {
            n_params_in++;
            v->param_type = ParamType::Input;
            add_param(v->data);
            kernel_param_ids.push_back(index);
        } else if (v->output_flag && v->size == group.size) {
            n_params_out++;
            v->param_type = ParamType::Output;

            size_t isize = (size_t) type_size[v->type],
                   dsize = (size_t) group.size;

            if (jitc_is_llvm(backend))
                dsize = (dsize + width - 1) / width * width;
            if (v->is_array())
                dsize *= v->array_length;
            dsize *= isize;

            // Padding to support out-of-bounds accesses in LLVM gather operations
            if (jitc_is_llvm(backend) && isize == 1)
                dsize += 4 - isize;

            sv.data = jitc_malloc(backend, dsize); // Note: unsafe to access 'v' after jitc_malloc().

            add_param(sv.data, /*write=*/1);
            kernel_param_ids.push_back(index);
        } else if (v->is_literal() && (VarType) v->type == VarType::Pointer) {
            n_params_in++;
            v->param_type = ParamType::Input;
            // A scatter target is written; resource_kind marks a GPU resource
            // handle (texture / acceleration structure, currently Metal-only).
            add_param((void *) v->literal, (uint8_t) v->written,
                      (uint8_t) v->resource_kind());
            kernel_param_ids.push_back(index);
        } else {
            n_side_effects += (uint32_t) v->side_effect;
            v->param_type = ParamType::Register;
            v->param_offset = 0xFFFF;

            #if defined(DRJIT_ENABLE_OPTIX)
                uses_optix |= v->optix;
            #endif
        }
    }

    // If this kernel performs indirect calls or invokes getters, pass a data
    // buffer containing the call data required by these operations. The code
    // below reserves a register and parameter slot.
    if (needs_call_buffer) {
        uint32_t base_src = jitc_var_mem_map(backend, VarType::UInt8,
                                             nullptr, 1, /*free=*/1);

        uint32_t base_v = jitc_var_pointer(backend, nullptr, base_src,
                                           /*write=*/0, /*written=*/false,
                                           /*disable_lvn=*/true);
        jitc_var_dec_ref(base_src); // 'base_v' now owns the backing reference

        uint32_t param_index = (uint32_t) kernel_params.size();
        Variable *bv = jitc_var(base_v);
        bv->param_type = ParamType::Input;
        bv->reg_index = n_regs++;
        bv->param_offset = param_index * (uint32_t) sizeof(void *);

        call_buffer.base_v = base_v;
        call_buffer.base_src = base_src;
        call_buffer.base_reg = bv->reg_index;
        call_buffer.base_param_index = param_index;

        add_param((void *) (uintptr_t) bv->literal);
        kernel_param_ids.push_back(base_v);

        // Released after the launch barrier (see jitc_eval_impl).
        call_base_vars.push_back(base_v);
    }

#if defined(DRJIT_ENABLE_METAL)
    // If the kernel performs any call, reserve a trailing ``params.args[]`` slot
    // for its visible function table. The slot is not an IR variable, so it stays
    // out of ``kernel_param_ids``.
    metal_vft_arg_index = -1;
    if (jitc_is_metal(backend) && has_call)
        metal_vft_arg_index = (int) kernel_params.size() - 1;
#endif

    if (unlikely(n_regs > 0xFFFFF))
        jitc_log(Warn,
                 "jit_run(): The generated kernel uses a more than 1 million "
                 "variables (%u) and will likely not run efficiently. Consider "
                 "periodically running jit_eval() to break the computation "
                 "into smaller chunks.", n_regs);

    if (unlikely(kernel_params.size() > 0xFFFF))
        jitc_log(Warn,
                 "jit_run(): The generated kernel accesses more than 64K "
                 "arrays (%zu) and will likely not run efficiently. Consider "
                 "periodically running jit_eval() to break the computation "
                 "into smaller chunks.", kernel_params.size());

    n_ops_total = n_regs;

    bool trace = jitc_log_active(LogLevel::Trace);

    if (unlikely(trace)) {
        buffer.clear();
        for (uint32_t group_index = group.start; group_index != group.end; ++group_index) {
            uint32_t index = schedule[group_index].index;
            Variable *v = jitc_var(index);

            buffer.fmt("   - %s%u -> r%u: ", type_prefix[v->type],
                       v->reg_index, index);

            const char *label = jitc_var_label(index);
            if (label)
                buffer.fmt("label=\"%s\", ", label);
            if (v->param_type == ParamType::Input)
                buffer.fmt("in, offset=%u, ", v->param_offset);
            if (v->param_type == ParamType::Output)
                buffer.fmt("out, offset=%u, ", v->param_offset);
            if (v->is_literal())
                buffer.put("literal, ");
            if (v->size == 1 && v->param_type != ParamType::Output)
                buffer.put("scalar, ");
            if (v->side_effect)
                buffer.put("side effects, ");
            buffer.rewind_to(buffer.size() - 2);
            buffer.put('\n');
        }
        jitc_trace("jit_assemble(size=%u): register map:\n%s",
                  group.size, buffer.get());
    }

    buffer.clear();
#if defined(DRJIT_ENABLE_CUDA)
    if (jitc_is_cuda(backend))
        jitc_cuda_assemble(ts, group, n_regs, (uint32_t) kernel_params.size());
    else
#endif
#if defined(DRJIT_ENABLE_METAL)
    if (jitc_is_metal(backend))
        jitc_metal_assemble(ts, group, n_regs, (uint32_t) kernel_params.size());
    else
#endif
        jitc_llvm_assemble(ts, group);

    // Bind the call-data buffer (built by jitc_call_upload() above and stored in
    // the base pointer variable) into its reserved parameter slot.
    if (call_buffer.base_v)
        kernel_params[call_buffer.base_param_index] =
            (void *) (uintptr_t) jitc_var(call_buffer.base_v)->literal;

    // Replace '^'s in '__raygen__^^^..' or 'drjit_^^^..' with hash.
    // Search for the kernel-name marker rather than the first '^', because
    // injected preamble code may contain '^' in comments that must not be
    // overwritten.
    kernel_hash = hash_kernel(buffer.get(), buffer.size(), backend);

    const char *needle =
        (uses_optix && jitc_is_cuda(backend)) ? "__raygen__^"
                                              : "drjit_^";
    const char *marker = strstr(buffer.get(), needle);
    if (!marker)
        jitc_fail("jitc_eval(): could not locate kernel-name placeholder "
                  "(needle=\"%s\") in generated source.", needle);

    size_t hash_offset = (marker - buffer.get()) + (strlen(needle) - 1),
           end_offset = buffer.size(),
           prefix_len = (uses_optix && jitc_is_cuda(backend)) ? 10 : 6;

    buffer.rewind_to(hash_offset);
    buffer.put_q64_unchecked(kernel_hash.high64);
    buffer.put_q64_unchecked(kernel_hash.low64);
    buffer.rewind_to(end_offset);
    memset(kernel_name, 0, sizeof(kernel_name));
    memcpy(kernel_name, buffer.get() + hash_offset - prefix_len,
           prefix_len + 32);

#if defined(DRJIT_ENABLE_OPTIX)
    if (uses_optix && indirect_callable_count > 0) {
        // Work around a bug in OptiX with driver version 570. When a two
        // pipelines share the same set of direct callables, the driver shares the
        // compiled result. This caching contaminates the second pipeline with
        // globals from the first, which causes a linker issue when the 'params'
        // buffer has a different size. We work around the issue by inserting a
        // unique callable that prevents this optimization.

        const char *id_fmt =
            "\n.visible .func (.param .align 8 .b8 result[16]) __direct_callable__id() {\n"
            "    st.param.u64 [result+0], 0x$Q;\n"
            "    st.param.u64 [result+8], 0x$Q;\n"
            "}";
        buffer.fmt_cuda(2, fmt_strlen(id_fmt), id_fmt,
            kernel_hash.low64,
            kernel_hash.high64
        );
    }
#endif

    // PrintIR / high log levels dump the generated IR to the console.
    // In the case of Metal, properly indent the generated MSL so that it is
    // easier to read.
    if (unlikely(trace || (jitc_flags() & (uint32_t) JitFlag::PrintIR))) {
        const char *ir = buffer.get();
#if defined(DRJIT_ENABLE_METAL)
        if (jitc_is_metal(backend))
            ir = jitc_metal_format();
#endif
        if (state.log_callback)
            state.log_callback(LogLevel::Info, ir);
        else {
            fputs(ir, stderr);
            fputc('\n', stderr);
        }
    }

    float codegen_time = timer();

    if (n_side_effects)
        jitc_log(
            Info, "  -> launching %016llx (%sn=%u, in=%u, out=%u, se=%u, ops=%u, jit=%s):",
            (unsigned long long) kernel_hash.high64,
            uses_optix ? "via OptiX, " : "", group.size, n_params_in,
            n_params_out, n_side_effects, n_ops_total, jitc_time_string(codegen_time));
    else
        jitc_log(
            Info, "  -> launching %016llx (%sn=%u, in=%u, out=%u, ops=%u, jit=%s):",
            (unsigned long long) kernel_hash.high64,
            uses_optix ? "via OptiX, " : "", group.size, n_params_in,
            n_params_out, n_ops_total, jitc_time_string(codegen_time));

    if (unlikely(jit_flag(JitFlag::KernelHistory))) {
        kernel_history_entry.backend = backend;
        kernel_history_entry.type = KernelType::JIT;
        kernel_history_entry.recording_mode = ts->recording_mode;
        kernel_history_entry.hash[0] = kernel_hash.low64;
        kernel_history_entry.hash[1] = kernel_hash.high64;
        // Store syntax-formatted MSL for Metal (see jitc_metal_format()).
        const char *ir = buffer.get();
        size_t ir_size = buffer.size();
#if defined(DRJIT_ENABLE_METAL)
        if (jitc_is_metal(backend))
            ir = jitc_metal_format(&ir_size);
#endif
        kernel_history_entry.ir = (char *) malloc_check(ir_size + 1);
        memcpy(kernel_history_entry.ir, ir, ir_size + 1);
        kernel_history_entry.uses_optix = uses_optix;
        kernel_history_entry.size = group.size;
        kernel_history_entry.input_count = n_params_in;
        kernel_history_entry.output_count = n_params_out + n_side_effects;
        kernel_history_entry.operation_count = n_ops_total;
        kernel_history_entry.codegen_time = codegen_time * 1e-3f;
    }
}

static ProfilerRegion profiler_region_backend_compile("jit_eval: compiling");

Task *jitc_run(ThreadState *ts, ScheduledGroup group) {
    uint64_t flags = 0;

#if defined(DRJIT_ENABLE_OPTIX)
    if (uses_optix) {
        const OptixPipelineCompileOptions &pco = ts->optix_pipeline->compile_options;
        flags =
            ((uint64_t) pco.numAttributeValues << 0)      + // 4 bit
            ((uint64_t) pco.numPayloadValues   << 4)      + // 4 bit
            ((uint64_t) pco.usesMotionBlur     << 8)      + // 1 bit
            ((uint64_t) pco.traversableGraphFlags  << 9)  + // 16 bit
            ((uint64_t) pco.usesPrimitiveTypeFlags << 25);  // 32 bit
    }
#endif

    KernelKey kernel_key(kernel_hash, ts->device, flags);
    auto it = state.kernel_cache.find(kernel_key);
    Kernel kernel;
    memset(&kernel, 0, sizeof(Kernel)); // quench uninitialized variable warning on MSVC
    kernel.operation_count = n_ops_total;

    if (it == state.kernel_cache.end()) {
        bool cache_hit = false;

#if defined(DRJIT_ENABLE_CUDA)
        if (jitc_is_cuda(ts->backend)) {
            ProfilerPhase profiler(profiler_region_backend_compile);
            if (!uses_optix) {
                kernel.size = 1; // dummy size value to distinguish between OptiX and CUDA kernels
                kernel.data = nullptr;
                std::tie(kernel.cuda.mod, cache_hit) = jitc_cuda_compile(buffer.get());
            } else {
                #if defined(DRJIT_ENABLE_OPTIX)
                    cache_hit = jitc_optix_compile(
                        ts, buffer.get(), buffer.size(), kernel_name, kernel);
                #endif
            }
        } else
#endif
#if defined(DRJIT_ENABLE_METAL)
        if (jitc_is_metal(ts->backend)) {
            ProfilerPhase profiler(profiler_region_backend_compile);
            cache_hit = jitc_metal_kernel_compile(buffer.get(), buffer.size(),
                                                  kernel_name, kernel);
        } else
#endif
        {
            cache_hit = jitc_kernel_load(buffer.get(), (uint32_t) buffer.size(),
                                         ts->backend, kernel_hash, kernel);

            if (!cache_hit) {
                ProfilerPhase profiler(profiler_region_backend_compile);
                jitc_llvm_compile(kernel);
                jitc_kernel_write(buffer.get(), (uint32_t) buffer.size(),
                                  ts->backend, kernel_hash, kernel);
                jitc_llvm_disasm(kernel);
            }
        }

        // Persist per-slot parameter metadata (writability + resource kind)
        // onto the kernel; freed in jitc_kernel_free().
        size_t np = kernel_param_info.size();
        kernel.param_info = new KernelParamInfo[np];
        std::memcpy(kernel.param_info, kernel_param_info.data(),
                    np * sizeof(KernelParamInfo));

#if defined(DRJIT_ENABLE_CUDA)
        if (jitc_is_cuda(ts->backend) && !uses_optix) {
            // Locate the kernel entry point
            size_t offset = buffer.size();
            const char *name_fmt = "drjit_$Q$Q";
            buffer.fmt_cuda(2, fmt_strlen(name_fmt), name_fmt,
                            kernel_hash.high64, kernel_hash.low64);
            cuda_check(cuModuleGetFunction(&kernel.cuda.func, kernel.cuda.mod,
                                           buffer.get() + offset));
            buffer.rewind_to(offset);

            // Determine a suitable thread count to maximize occupancy
            int unused, block_size;
            cuda_check(cuOccupancyMaxPotentialBlockSize(
                &unused, &block_size,
                kernel.cuda.func, nullptr, 0, 0));
            kernel.cuda.block_size = (uint32_t) block_size;

            // DrJit doesn't use shared memory at all, prefer to have more L1 cache.
            cuda_check(cuFuncSetAttribute(
                kernel.cuda.func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 0));
            cuda_check(cuFuncSetAttribute(
                kernel.cuda.func, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                CU_SHAREDMEM_CARVEOUT_MAX_L1));
        }
#endif

        float link_time = timer();
        jitc_log(Info, "     cache %s, %s: %s, %s.",
                cache_hit ? "hit" : "miss",
                cache_hit ? "load" : "build",
                std::string(jitc_time_string(link_time)).c_str(),
                std::string(jitc_mem_string(kernel.size)).c_str());

        kernel.src_size = buffer.size();
        kernel.src = (char *) malloc_check(kernel.src_size + 1);
        memcpy(kernel.src, buffer.get(), kernel.src_size + 1);
        state.kernel_cache.emplace(kernel_key, kernel);

        if (cache_hit)
            state.kernel_soft_misses++;
        else
            state.kernel_hard_misses++;

        if (unlikely(jit_flag(JitFlag::KernelHistory))) {
            kernel_history_entry.cache_disk = cache_hit;
            kernel_history_entry.cache_hit = cache_hit;
            if (!cache_hit)
                kernel_history_entry.backend_time = link_time * 1e-3f;
        }
    } else {
        kernel_history_entry.cache_hit = true;
        kernel = it.value();
        state.kernel_hits++;
    }
    state.kernel_launches++;

    KernelHistoryEntry *e = nullptr;
    if(unlikely(jit_flag(JitFlag::KernelHistory)))
        e = &kernel_history_entry;

    Task *task = ts->launch(kernel, kernel_key, kernel_hash, group.size,
                            kernel_params, kernel_param_ids, e);

    // The kernel history now owns the entry's 'ir' string (if enabled);
    // jitc_eval_rollback() frees this pointer when it is still non-null
    kernel_history_entry.ir = nullptr;

    return task;
}

static ProfilerRegion profiler_region_eval("jit_eval");

// Forward declaration
void jitc_eval_impl(ThreadState *ts);

/// Implementation detail of jitc_schedule_sort()
struct Bucket { uint32_t size, scope, count, offset; };
static std::vector<Bucket> buckets;

/// Stable count sort of ``schedule`` by size (descending) and scope (ascending).
static void jitc_schedule_sort() {
    auto less = [](const auto &a, const auto &b) {
        return a.size != b.size ? a.size > b.size : a.scope < b.scope;
    };

    // A distinct (size, scope) key and the entries that map to it
    buckets.clear();

    // Locate the bucket for a key. Entries sharing a key are contiguous in the
    // schedule, so the most recent match is the likeliest hit.
    uint32_t hint = 0;
    auto find = [&](const ScheduledVariable &sv) -> Bucket * {
        if (hint < buckets.size() && buckets[hint].size == sv.size &&
            buckets[hint].scope == sv.scope)
            return &buckets[hint];
        for (hint = 0; hint < buckets.size(); ++hint)
            if (buckets[hint].size == sv.size && buckets[hint].scope == sv.scope)
                return &buckets[hint];
        return nullptr;
    };

    // Pass 1: tally the distinct keys
    const uint32_t max_buckets = 32;
    bool fallback = false;
    for (const ScheduledVariable &sv : schedule) {
        if (Bucket *b = find(sv)) {
            b->count++;
        } else if (likely(buckets.size() < max_buckets)) {
            hint = (uint32_t) buckets.size();
            buckets.push_back({ sv.size, sv.scope, 1, 0 });
        } else {
            fallback = true; // Pathological key count; use a comparison sort
            break;
        }
    }

    if (fallback) {
        std::stable_sort(schedule.begin(), schedule.end(), less);
        return;
    }

    // A single key means the schedule is already in the desired order
    if (buckets.size() <= 1)
        return;

    // Order the (few) keys and turn their counts into output offsets
    std::sort(buckets.begin(), buckets.end(), less);
    uint32_t offset = 0;
    for (Bucket &b : buckets) {
        b.offset = offset;
        offset += b.count;
    }

    // Pass 2: stably scatter each entry into its group, then swap into place
    schedule_scratch.resize(schedule.size());
    hint = 0;
    for (const ScheduledVariable &sv : schedule)
        schedule_scratch[find(sv)->offset++] = sv;
    schedule.swap(schedule_scratch);
}

/* Called when an exception interrupts jitc_eval_impl() partway through.
   (E.g. when running out of memory or when a code generation routine throws).
   Release references and memory. */
static void jitc_eval_rollback(size_t callbacks_baseline) {
    for (ScheduledVariable &sv : schedule) {
        Variable *v = jitc_var(sv.index);
        v->reg_index = 0;
        v->output_flag = false;
        if (sv.data)
            jitc_free(sv.data);
        jitc_var_dec_ref(sv.index, v);
    }
    schedule.clear();

    // Temporary references taken during root collection
    for (uint32_t index : eval_roots)
        jitc_var_dec_ref(index);
    eval_roots.clear();

    // References owned by the dequeued side effects.
    for (const EvalTask &t : eval_tasks) {
        if (t.side_effect)
            jitc_var_dec_ref(t.index);
    }
    eval_tasks.clear();
    visit_later.clear();

    // Callbacks enqueued on behalf of kernels that never ran
    while (eval_callbacks.size() > callbacks_baseline) {
        jitc_var_dec_ref(eval_callbacks.back());
        eval_callbacks.pop_back();
    }

    // Call-data base pointers created before the failure
    for (uint32_t index : call_base_vars)
        jitc_var_dec_ref(index);
    call_base_vars.clear();

    // Captured-source references whose aggregate never ran
    call_buffer.data_entries.clear();

    // Release references taken by jitc_var_call_assemble()
    for (CallData *call : calls_assembled)
        jitc_var_dec_ref(call->id);
    calls_assembled.clear();

    // Release references taken by jitc_var_call_getter_assemble()
    for (GetterData *gd : call_buffer.getters)
        jitc_var_dec_ref(gd->id);
    call_buffer.getters.clear();

    // IR string of a kernel history entry that no launch consumed
    free(kernel_history_entry.ir);
    kernel_history_entry.ir = nullptr;
}

/// Evaluate all computation that is queued on the given ThreadState
void jitc_eval(ThreadState *ts) {
    if (!ts || (ts->scheduled.empty() && ts->side_effects.empty()))
        return;

    ProfilerPhase profiler(profiler_region_eval);

    /* The function 'jitc_eval()' modifies several global data structures
       and should never be executed concurrently. However, there are a few
       places where it needs to temporarily release the main lock as part of
       its work, which is dangerous because another thread could use that
       opportunity to enter 'jitc_eval()' and cause corruption. The following
       therefore temporarily unlocks 'state.lock' and then locks a separate
       lock 'state.eval_lock' specifically guarding these data structures */

    {
        lock_release(state.lock);
        lock_guard guard(state.eval_lock);
        lock_acquire(state.lock);

        size_t callbacks_baseline = eval_callbacks.size();
        try {
            jitc_eval_impl(ts);
        } catch (...) {
            jitc_eval_rollback(callbacks_baseline);
            throw;
        }
    }

    if (unlikely(!eval_callbacks.empty())) {
        std::vector<uint32_t> cb;
        eval_callbacks.swap(cb);

        jitc_log(Trace, "jit_eval(): running %zu callbacks ..", cb.size());
        for (uint32_t index: cb) {
            VariableExtra &extra = state.extra[jitc_var(index)->extra];
            if (extra.callback_internal) {
                extra.callback(index, 0, extra.callback_data);
            } else {
                unlock_guard guard_2(state.lock);
                extra.callback(index, 0, extra.callback_data);
            }
            jitc_var_dec_ref(index);
        }
        cb.clear();
        if (eval_callbacks.empty())
            cb.swap(eval_callbacks);
    }

    jitc_log(Info, "jit_eval(): done.");
}

void jitc_eval_impl(ThreadState *ts) {
    visit_later.clear();
    schedule.clear();
    eval_tasks.clear();
    eval_roots.clear();

#if defined(DRJIT_ENABLE_OPTIX)
    jitc_optix_max_coopvec_size = 0;
#endif

    // Collect the roots (explicitly scheduled variables). 'ts->scheduled' only
    // holds weak references, so take a temporary strong reference on each root.
    for (WeakRef wr: ts->scheduled) {
        // Skip variables that expired, or which we already evaluated
        Variable *v = jitc_var(wr);
        if (!v || v->is_evaluated())
            continue;
        jitc_var_inc_ref(wr.index, v);
        eval_tasks.push_back({ v->size, wr.index, false });
        eval_roots.push_back(wr.index);
    }

    ts->scheduled.clear();

    // ... and the side effects. A scatter whose target buffer has become
    // unreferenced writes a result that nobody reads and can be elided.
    for (uint32_t index: ts->side_effects) {
        Variable *v = jitc_var(index);

        if (jitc_elide_scatter(index, v))
            jitc_var_dec_ref(index);
        else
            eval_tasks.push_back({ v->size, index, true });
    }

    ts->side_effects.clear();

    if (eval_tasks.empty())
        return;

    /* Traverse same-size tasks consecutively, each size under its own
       generation. */
    std::stable_sort(eval_tasks.begin(), eval_tasks.end(),
                     [](const EvalTask &a, const EvalTask &b) {
                         if (a.size != b.size)
                             return a.size > b.size;
                         return a.side_effect < b.side_effect;
                     });

    for (size_t i = 0, n = eval_tasks.size(); i < n; ) {
        uint32_t group_size = eval_tasks[i].size;

        // Advance the traversal generation counter per distinct size
        jitc_visit_new_gen();

        size_t j = i;
        for (; j < n && eval_tasks[j].size == group_size; ++j)
            jitc_var_traverse(group_size, eval_tasks[j].index);

        // Process deferred nodes. Do not use a range-based for loop, as
        // traversal adds items.
        for (size_t k = 0; k < visit_later.size(); ++k) {
            VisitedKey vk = visit_later[k];
            jitc_var_traverse(vk.size, vk.index, vk.depth);
        }
        visit_later.clear();

        i = j;
    }

    // Mark roots as outputs
    for (uint32_t index: eval_roots)
        jitc_var(index)->output_flag = true;

    // Release the temporary references taken during root collection.
    for (uint32_t index: eval_roots)
        jitc_var_dec_ref(index);
    eval_roots.clear();

    if (schedule.empty())
        return;

    // Order variables into kernel groups (by size, then scope)
    jitc_schedule_sort();

    // Partition into groups of matching size
    schedule_groups.clear();
    if (schedule[0].size == schedule[schedule.size() - 1].size) {
        schedule_groups.emplace_back(schedule[0].size, 0,
                                     (uint32_t) schedule.size());
    } else {
        uint32_t cur = 0;
        for (uint32_t i = 1; i < (uint32_t) schedule.size(); ++i) {
            if (schedule[i - 1].size != schedule[i].size) {
                schedule_groups.emplace_back(schedule[cur].size, cur, i);
                cur = i;
            }
        }

        schedule_groups.emplace_back(schedule[cur].size,
                                     cur, (uint32_t) schedule.size());
    }

    jitc_log(Info, "jit_eval(): launching %zu kernel%s.",
            schedule_groups.size(),
            schedule_groups.size() == 1 ? "" : "s");

#if defined(DRJIT_ENABLE_CUDA)
    scoped_set_context_maybe guard2(ts->context);
#endif

    for (ScheduledGroup &group : schedule_groups) {
        jitc_assemble(ts, group);
        jitc_run(ts, group);
    }

    // Subsequent kernel launches must be serialized wrt. the launches above.
    // barrier() also drains shared allocations parked for deferred release.
    ts->barrier();

    // Release strong references to opaque objects accessed by vcalls
    call_buffer.data_entries.clear();

    // Release the call-data base pointers now that every launch that
    // referenced them has completed; this frees the fused buffers they own.
    for (uint32_t index : call_base_vars)
        jitc_var_dec_ref(index);
    call_base_vars.clear();

    /* Variables and their dependencies are now computed, hence internal edges
       between them can be removed. This will cause many variables to expire. */
    jitc_log(Debug, "jit_eval(): cleaning up..");

    for (ScheduledVariable sv : schedule) {
        uint32_t index = sv.index;
        Variable *v = jitc_var(index);
        v->reg_index = 0;

        if (unlikely(v->extra)) {
            VariableExtra &extra = state.extra[v->extra];
            if (extra.callback) {
                eval_callbacks.push_back(index);
                jitc_var_inc_ref(index, v);
            }
        }

        if (!(v->output_flag || v->side_effect)) {
            jitc_var_dec_ref(index, v);
            continue;
        }

        jitc_assert(!v->is_literal(),
                   "jit_eval(): internal error: did not expect a literal "
                   "constant variable here!");

        jitc_lvn_drop(index, v);

        if (v->output_flag && v->size == sv.size) {
            v->kind = (uint32_t) VarKind::Evaluated;
            v->data = sv.data;
            v->output_flag = false;
            v->consumed = false;

            // Now a data buffer; reset scope to match the rule in jitc_var_new.
            v->scope = v->is_array() ? SCOPE_ARRAY : SCOPE_BUFFER;

#ifndef NDEBUG
            state.ptr_to_variable.insert({ v->data, index });
#endif
        }

        uint32_t dep[4], side_effect = v->side_effect;
        memcpy(dep, v->dep, sizeof(uint32_t) * 4);
        memset(v->dep, 0, sizeof(uint32_t) * 4);
        v->side_effect = false;

        jitc_var_dec_ref(index, v);

        if (side_effect)
            jitc_var_dec_ref(index, v);

        for (int j = 0; j < 4; ++j)
            jitc_var_dec_ref(dep[j]);
    }
}

static ProfilerRegion profiler_region_assemble_func("jit_assemble_func");

XXH128_hash_t jitc_assemble_func(const CallData *call, uint32_t inst,
                                 uint32_t in_size, uint32_t in_align,
                                 uint32_t out_size, uint32_t out_align) {
    ProfilerPhase profiler(profiler_region_assemble_func);

    jitc_visit_new_gen();
    visit_later.clear();
    schedule.clear();

    // Track callable depth using an RAII guard (assembly may raise)
    struct ScopedCallableDepth {
        ScopedCallableDepth() { callable_depth++; }
        ~ScopedCallableDepth() { callable_depth--; }
    } scoped_callable_depth;

    const std::vector<uint32_t> &check = call->checkpoints;

    const size_t n_out = call->n_out,
                 n_se = check.empty() ? 0 : (check[inst + 1] - check[inst]);

    const uint32_t *out = call->inner_out.data() + n_out * inst,
                   *se = call->side_effects.empty()
                             ? nullptr
                             : (call->side_effects.data() + check[inst]);

    for (size_t i = 0; i < n_out; ++i) {
        if (call->out_offset[i] == (uint32_t) -1)
            continue;
        jitc_var_traverse(1, out[i]);
    }

    for (uint32_t i = 0; i < n_se; ++i)
        jitc_var_traverse(1, se[i]);

    // Should not be replaced by a range-based for loop,
    // as the traversal may append further items
    for (size_t i = 0; i < visit_later.size(); ++i) {
        VisitedKey vk = visit_later[i];
        jitc_var_traverse(vk.size, vk.index, vk.depth);
    }

    std::stable_sort(
        schedule.begin(), schedule.end(),
        [](const ScheduledVariable &a, const ScheduledVariable &b) {
            return a.scope < b.scope;
        });

    // Reset the register index of this instance's captured call-data slots
    // before assigning fresh registers below. This is needed to detect
    // slots that are not used in the callable's body.
    const CallData::InstanceLayout &slot_layout = call->instance_layout[inst];
    for (uint32_t k = slot_layout.slot_start; k < slot_layout.slot_end(); ++k) {
        if (Variable *v = jitc_var(call->slots[k].ref))
            v->reg_index = 0;
    }

    uint32_t n_regs = jitc_is_cuda(call->backend) ? 4 : 1;
    for (ScheduledVariable &sv : schedule) {
        Variable *v = jitc_var(sv.index);
        v->reg_index = n_regs++;
    }

    size_t kernel_offset = buffer.size();

    switch (call->backend) {
        case JitBackend::LLVM:
            jitc_llvm_assemble_func(call, inst);
            break;

#if defined(DRJIT_ENABLE_CUDA)
        case JitBackend::CUDA:
            jitc_cuda_assemble_func(call, inst, in_size, in_align, out_size,
                                    out_align, n_regs);
            break;
#endif

#if defined(DRJIT_ENABLE_METAL)
        case JitBackend::Metal:
            jitc_metal_assemble_func(call, inst, in_size, in_align, out_size,
                                     out_align, n_regs);
            break;
#endif

        default:
            jitc_fail("jitc_assemble_func(): unsupported backend!");
    }

    size_t kernel_length = buffer.size() - kernel_offset;

    if (jit_flag(JitFlag::MergeFunctions)) {
        kernel_hash = XXH128(buffer.get() + kernel_offset, kernel_length, 0);
    } else {
        kernel_hash.low64 = indirect_callable_count;
        kernel_hash.high64 = 0;
    }

    if (globals_map.emplace(GlobalKey(kernel_hash, call->n_inst != 1 ? GlobalType::IndirectCallable : GlobalType::Callable),
                            GlobalValue(globals.size(), kernel_length)).second) {
        // Replace '^'s in 'func_^^^..' or '__direct_callable__^^^..' with hash
        size_t hash_offset = strchr(buffer.get() + kernel_offset, '^') - buffer.get(),
               end_offset = buffer.size();

        buffer.rewind_to(hash_offset);
        buffer.put_q64_unchecked(kernel_hash.high64);
        buffer.put_q64_unchecked(kernel_hash.low64);
        buffer.rewind_to(end_offset);

        n_ops_total += n_regs;
        globals.put(buffer.get() + kernel_offset, kernel_length);

        if (call->n_inst != 1)
            indirect_callable_count_unique++;
    }

    if (call->n_inst != 1)
        indirect_callable_count++;

    buffer.rewind_to(kernel_offset);

    for (ScheduledVariable &sv: schedule) {
        Variable *v = jitc_var(sv.index);
        if (unlikely(v->extra)) {
            VariableExtra &extra = state.extra[v->extra];
            if (extra.callback) {
                eval_callbacks.push_back(sv.index);
                jitc_var_inc_ref(sv.index, v);
            }
        }
        v->reg_index = 0;
        v->output_flag = false;
        jitc_var_dec_ref(sv.index, v);
    }

    // The references owned by the schedule entries are now consumed (the
    // unwind path of jitc_var_call_assemble() releases any still present)
    schedule.clear();

    return kernel_hash;
}

/// Register a global declaration that will be included in the final program
void jitc_register_global(const char *str, GlobalType type) {
    size_t length = strlen(str);
    if (globals_map.emplace(GlobalKey(XXH128(str, length, 0), type),
                            GlobalValue(globals.size(), length)).second)
        globals.put(str, length);
}
