/**
 * This file contains the logic that assembles a Metal MSL representation from a
 * recorded Dr.Jit computation graph. It implements a small template engine
 * involving plentiful use of the 'fmt' formatting routine.
 *
 * Its format interface supports the following format string characters.
 *
 *  Format  Input          Example result    Description
 * --------------------------------------------------------------------------
 *  $u      uint32_t      `1234`             Decimal number (32 bit)
 * --------------------------------------------------------------------------
 *  $s      const char *  `foo`              Zero-terminated string
 * --------------------------------------------------------------------------
 *  $t      Variable      `float`            Variable type
 * --------------------------------------------------------------------------
 *  $b      Variable      `uint`             Variable type, binary format
 * --------------------------------------------------------------------------
 *  $v      Variable      `r1234`            Variable name
 * --------------------------------------------------------------------------
 *  $l      Variable      `0x1`              Literal value of variable (hex)
 *  $o      Variable      `2`                Index into ``params.args[]``
 * --------------------------------------------------------------------------
 */

#include "eval.h"
#include "internal.h"
#include "var.h"
#include "log.h"
#include "strbuf.h"
#include "metal.h"
#include "metal_eval.h"
#include "metal_tex.h"

#include "metal_scatter.h"
#include "metal_packet.h"
#include "metal_array.h"
#include "metal_coop_vec.h"
#include "array.h"
#include "trace.h"
#include "tex.h"
#include "call.h"
#include "metal_ts.h"
#include "loop.h"
#include "cond.h"
#include "record_ts.h"

// Forward declaration
static void jitc_metal_render(Variable *v);

// Keep track of scenes discovered during code generation
std::vector<MetalScene *> metal_kernel_scenes;

// Names of the kernel's indirect-callable functions, ordered by callable index.
std::vector<XXH128_hash_t> metal_kernel_callables;

// Index into ``params.args[]`` when the kernel receives a
// visible-function-table handle, or -1 otherwise.
int metal_vft_arg_index = -1;

void jitc_metal_assemble_reset() {
    metal_kernel_scenes.clear();
    metal_kernel_callables.clear();
    metal_vft_arg_index = -1;
}

void metal_register_kernel_scene(MetalScene *scene) {
    if (!scene)
        return;
    for (MetalScene *s : metal_kernel_scenes)
        if (s == scene)
            return;
    metal_kernel_scenes.push_back(scene);
}

/// Intersection functions and the geometry type influence the PSO linking step
/// but are otherwise invisible in the generated MSL. This function folds them
/// in via a comment to ensure that incompatibilities produce unique kernels.
static void jitc_metal_render_scene_configuration() {
    for (size_t i = 0; i < metal_kernel_scenes.size(); ++i) {
        MetalScene *si = metal_kernel_scenes[i];
        fmt("// Scene properties: scene_$u mask=$u fns=[",
            (uint32_t) i, si ? si->geometry_types_mask : 0u);
        if (si) {
            for (size_t j = 0; j < si->intersection_fns.size(); ++j) {
                if (j) put(", ");
                fmt("$s", si->intersection_fns[j].c_str());
            }
        }
        put("]\n");
    }
}

// ----------------------------------------------------------------------------
//  Resource-handle helpers
// ----------------------------------------------------------------------------

/// If ``v`` is an opaque resource handle (texture, sampler, acceleration
/// structure, IFT), emit the MSL that reconstructs a typed reference from its
/// gpuResourceID and return true. Returns false for ordinary buffer pointers.
static bool jitc_metal_render_resource_handle(const Variable *v,
                                              bool in_callable,
                                              uint32_t data_offset) {
    ResourceKind kind = v->resource_kind();
    if (kind == ResourceKind::Buffer)
        return false;

    const char *tname;
    switch (kind) {
        case ResourceKind::Texture: {
            MetalTexResource *rec = (MetalTexResource *) (uintptr_t) v->literal;
            bool w = v->write_ptr;
            if (rec->parent->ndim == 3)
                tname = w ? "texture3d<float, access::write>"
                          : "texture3d<float>";
            else
                tname = w ? "texture2d<float, access::write>"
                          : "texture2d<float>";
            break;
        }

        case ResourceKind::Sampler:
            tname = "sampler";
            break;

        case ResourceKind::Accel:
        case ResourceKind::IFT: {
            MetalScene *scene = (MetalScene *) (uintptr_t) v->literal;
            metal_register_kernel_scene(scene);
            if (kind == ResourceKind::Accel) {
                tname = "raytracing::instance_acceleration_structure";
            } else {
                bool curves = scene && (scene->geometry_types_mask & 0x4u) != 0;
                tname = curves
                    ? "raytracing::intersection_function_table<raytracing::triangle_data, raytracing::instancing, raytracing::world_space_data, raytracing::curve_data>"
                    : "raytracing::intersection_function_table<raytracing::triangle_data, raytracing::instancing, raytracing::world_space_data>";
            }
            break;
        }

        default:
            jitc_fail("jitc_metal_render_resource_handle(): unexpected resource "
                      "kind %u.", (uint32_t) kind);
    }

    if (in_callable)
        fmt("device $s& $v = *(device $s*)(data + $u);\n",
            tname, v, tname, data_offset);
    else
        fmt("constant $s& $v = *(constant $s*) &params.args[$o];\n",
            tname, v, tname, v);
    return true;
}

/// Scratch buffer holding the formatted copy produced by jitc_metal_format()
static StringBuffer metal_reindent_scratch;

/// The Metal backend generates unformatted MSL since indentation tracking is
/// costly during code generation. This function returns an indented copy if
/// requested by JitFlags.PrintIR or the log level is high enough. In this case,
/// the MSL is printed to the console where improved readability is desirable.
///
/// The function relies on three properties of the emitted MSL:
///   1. Every ``{``/``}`` is related to control flow (i.e. braces don't occur
///      in strings, initializer lists, etc.)
///   2. The only comments are ``//`` line comments
///   3. No statement is split across lines by a blank or comment line, which
///      would reset the continuation state mid-statement.
///
/// These are currently satisfied. Violating them degrades indentation quality
/// but never changes semantics.
const char *jitc_metal_format(size_t *size_out) {
    metal_reindent_scratch.swap(buffer);
    buffer.clear();

    const char *src = metal_reindent_scratch.get();
    size_t n = metal_reindent_scratch.size();

    int depth = 0;
    bool stmt_open = false; // previous code line left a statement unterminated

    for (size_t i = 0; i < n; ) {
        // Carve out one line and trim its horizontal whitespace.
        size_t b = i;
        while (i < n && src[i] != '\n')
            i++;
        size_t e = i;
        if (i < n)
            i++; // consume '\n'
        while (b < e && (src[b] == ' ' || src[b] == '\t'))
            b++;
        while (e > b && (src[e - 1] == ' ' || src[e - 1] == '\t'))
            e--;

        if (b == e) { // blank line
            put('\n');
            stmt_open = false;
            continue;
        }

        bool is_comment = src[b] == '/' && b + 1 < e && src[b + 1] == '/';
        bool is_preproc = src[b] == '#';

        // Scan for brace deltas and the last significant character, stopping at
        // a trailing ``//`` comment.
        int opens = 0, closes = 0, leading_close = 0;
        bool seen = false;
        char last = 0;
        for (size_t k = b; k < e; k++) {
            char c = src[k];
            if (c == '/' && k + 1 < e && src[k + 1] == '/')
                break;
            if (c == '{') {
                opens++;
                seen = true;
            } else if (c == '}') {
                closes++;
                if (!seen)
                    leading_close++;
            } else if (c != ' ' && c != '\t') {
                seen = true;
            }
            if (c != ' ' && c != '\t')
                last = c;
        }

        int indent = depth - leading_close;
        if (indent < 0)
            indent = 0;
        if (stmt_open && !is_preproc && !is_comment)
            indent++; // hanging indent for continuation lines

        if (indent > 0)
            put(' ', (size_t) indent * 4);
        put(src + b, e - b);
        put('\n');

        depth += opens - closes;
        if (depth < 0)
            depth = 0;

        stmt_open = !(is_comment || is_preproc || last == ';' || last == '{' ||
                      last == '}');
    }

    jitc_assert(depth == 0, "jitc_metal_format(): mismatched braces!");

    // Restore the unformatted source to ``buffer`` and keep the formatted copy in
    // the scratch buffer, whose contents we return.
    metal_reindent_scratch.swap(buffer);
    if (size_out)
        *size_out = metal_reindent_scratch.size();
    return metal_reindent_scratch.get();
}

void jitc_metal_assemble(ThreadState *ts, ScheduledGroup group,
                         uint32_t /*n_regs*/, uint32_t n_params) {

    put("#pragma clang diagnostic ignored \"-Wunused-variable\"\n"
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "\n");

    // ``args`` covers every real kernel parameter, plus a trailing slot for the
    // visible function table (appended at launch) when the kernel has calls.
    fmt("struct Params {\n"
        "    uint size;\n"
        "    device void *args[$u];\n"
        "};\n\n",
        metal_vft_arg_index >= 0 ? n_params
                                 : (n_params > 1 ? n_params - 1 : 1));

    fmt("kernel void drjit_^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^(\n"
        "    constant Params& params [[buffer(0)]],\n"
        "    uint r0 [[thread_position_in_grid]]) {\n");

    // Load the call-data base pointer once, shared by every call's dispatch in
    // this kernel. The args index is the parameter index minus one (the leading
    // 'size' field, see the '$o' convention).
    if (call_buffer.base_v)
        fmt("    device uint8_t* r$u = (device uint8_t*) params.args[$u];\n",
            call_buffer.base_reg, call_buffer.base_param_index - 1);

    // -------------------------------------------------------------------
    //   4. Render every variable in the schedule
    // -------------------------------------------------------------------
    for (uint32_t gi = group.start; gi != group.end; ++gi) {
        uint32_t index = schedule[gi].index;
        Variable *v = jitc_var(index);
        ParamType ptype = (ParamType) v->param_type;

        // Apple GPUs have no hardware FP64, so Float64 is promoted to
        // Float32 at variable creation (see jitc_var_new) and never reaches
        // codegen. Seeing one here means a variable slipped past promotion.
        if ((VarType) v->type == VarType::Float64)
            jitc_fail("jitc_metal_assemble(): the program should not contain "
                      "Float64 variables.\n");

        if (likely(ptype == ParamType::Input)) {
            // -------------------------------------------------
            // Load an input from the params buffer
            // -------------------------------------------------
            if (v->is_literal()) {
                VarType vt = (VarType) v->type;
                if (vt == VarType::Pointer) {
                    // Opaque resource handles (textures, samplers, ray-tracing
                    // structures) reconstruct a typed reference in place
                    if (jitc_metal_render_resource_handle(
                            v, /*in_callable=*/false, /*data_offset=*/0))
                        continue;
                    // Pointer literals must be loaded from params (not inlined)
                    // so frozen function replay can update the address
                    fmt("$t $v = ($t) params.args[$o];\n",
                              v, v, v, v);
                } else if (vt == VarType::Float32) {
                    fmt("$t $v = as_type<float>($lu);\n", v, v, v);
                } else if (vt == VarType::Float16) {
                    fmt("$t $v = as_type<half>((ushort) $lu);\n", v, v, v);
                } else {
                    fmt("$t $v = ($t) $lu;\n", v, v, v, v);
                }
                continue;
            }

            if (!v->is_array()) {
                if (v->size > 1)
                    fmt("$t $v = ((device const $t*) params.args[$o])[r0];\n",
                              v, v, v, v);
                else
                    fmt("$t $v = *(device const $t*) params.args[$o];\n",
                              v, v, v, v);
            } else {
                // Array memcpy helpers reference the named ``p<r>`` pointer.
                fmt("device const $t* p$v = (device const $t*) params.args[$o];\n",
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
                fmt("((device $t*) params.args[$o])[r0] = $v;\n",
                    v, v, v);
            } else {
                // Array memcpy helpers reference the named ``p<r>`` pointer.
                fmt("device $t* p$v = (device $t*) params.args[$o];\n",
                    v, v, v, v);
                jitc_metal_render_array_memcpy_out(v);
            }
        }
    }

    put("}\n");

    jitc_metal_render_scene_configuration();

    // -------------------------------------------------------------------
    //   5. Emit callable functions and globals
    // -------------------------------------------------------------------

    // Assign callable_index for jitc_call_upload() and record the matching
    // function hash so jitc_metal_kernel_compile() can populate the visible
    // function table in the same order.
    {
        metal_kernel_callables.reserve(indirect_callable_count_unique);
        uint32_t ctr = 0;
        for (auto &it : globals_map) {
            if (it.first.type != GlobalType::IndirectCallable)
                continue;
            it.second.callable_index = ctr++;
            metal_kernel_callables.push_back(it.first.hash);
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
        // Find the insertion point: just after "using namespace metal;\n".
        const char *marker = strstr(buffer.get(), "using namespace metal;\n");
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

            // Emit forward declarations only for single-target callables: they
            // are invoked by name (in the kernel's direct-call fast path or a
            // nested direct call) and may appear before their definition.
            // Multi-target callables are reached only through the visible
            // function table, never named in MSL, so they need no declaration.
            for (auto &it : globals_map) {
                if (it.first.type != GlobalType::Callable)
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
    fmt("$t $v = $s($v);\n", v, v, op, a);
}

static void jitc_metal_render_binary(Variable *v, const char *op) {
    Variable *a = jitc_var(v->dep[0]);
    Variable *b = jitc_var(v->dep[1]);
    fmt("$t $v = $v $s $v;\n", v, v, a, op, b);
}

static void jitc_metal_render_call(Variable *v, const char *fn,
                                   uint32_t n_args) {
    Variable *a0 = jitc_var(v->dep[0]);
    fmt("$t $v = $s($v", v, v, fn, a0);
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

// Emit a rounding op that is optionally fused with a float-to-int conversion
static void jitc_metal_render_round(Variable *v, const char *fn) {
    Variable *a0 = jitc_var(v->dep[0]);
    if (jitc_is_float(v))
        fmt("$t $v = $s($v);\n", v, v, fn, a0);
    else if (v->kind == (uint32_t) VarKind::Trunc)
        fmt("$t $v = ($t) $v;\n", v, v, v, a0);
    else
        fmt("$t $v = ($t) $s($v);\n", v, v, v, fn, a0);
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

        case VarKind::Undefined:
            fmt("$t $v = ($t) 0;\n", v, v, v);
            break;

        case VarKind::Literal: {
            VarType vt = (VarType) v->type;
            if (vt == VarType::Float32)
                fmt("$t $v = as_type<float>($lu);\n", v, v, v);
            else if (vt == VarType::Float16)
                fmt("$t $v = as_type<half>((ushort) $lu);\n", v, v, v);
            else if (vt == VarType::Bool)
                fmt("$t $v = ($t) ($lu);\n", v, v, v, v);
            else
                fmt("$t $v = ($t) $lu;\n", v, v, v, v);
            break;
        }

        case VarKind::Counter:
            fmt("$t $v = ($t) r0;\n", v, v, v);
            break;

        // -- Arithmetic --
        case VarKind::Add: jitc_metal_render_binary(v, "+"); break;
        case VarKind::Sub: jitc_metal_render_binary(v, "-"); break;
        case VarKind::Mul: jitc_metal_render_binary(v, "*"); break;
        case VarKind::Div: jitc_metal_render_binary(v, "/"); break;
        case VarKind::DivApprox: {
            Variable *a0 = jitc_var(v->dep[0]),
                     *a1 = jitc_var(v->dep[1]);
            fmt("$t $v = fast::divide($v, $v);\n", v, v, a0, a1);
            break;
        }
        case VarKind::Mod: jitc_metal_render_binary(v, "%"); break;

        case VarKind::MulHi: {
            Variable *a = jitc_var(v->dep[0]);
            Variable *b = jitc_var(v->dep[1]);
            fmt("$t $v = mulhi($v, $v);\n", v, v, a, b);
            break;
        }

        case VarKind::MulWide: {
            // 32-bit × 32-bit → 64-bit
            Variable *a = jitc_var(v->dep[0]);
            Variable *b = jitc_var(v->dep[1]);
            fmt("$t $v = ($t)$v * ($t)$v;\n", v, v, v, a, v, b);
            break;
        }

        case VarKind::Fma: {
            Variable *a0 = jitc_var(v->dep[0]),
                     *a1 = jitc_var(v->dep[1]),
                     *a2 = jitc_var(v->dep[2]);
            if (jitc_is_float(v))
                fmt("$t $v = fma($v, $v, $v);\n", v, v, a0, a1, a2);
            else
                fmt("$t $v = $v * $v + $v;\n", v, v, a0, a1, a2);
            break;
        }
        case VarKind::Min:  jitc_metal_render_call(v, "min",  2); break;
        case VarKind::Max:  jitc_metal_render_call(v, "max",  2); break;

        // -- Rounding --
        case VarKind::Ceil:  jitc_metal_render_round(v, "ceil");  break;
        case VarKind::Floor: jitc_metal_render_round(v, "floor"); break;
        case VarKind::Round: jitc_metal_render_round(v, "rint");  break;
        case VarKind::Trunc: jitc_metal_render_round(v, "trunc"); break;

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
            fmt("$t $v = $v ? $v : $v;\n", v, v, cond, a, b);
            break;
        }

        // -- Bit operations --
        case VarKind::Popc:  jitc_metal_render_call(v, "popcount", 1); break;
        case VarKind::Clz:   jitc_metal_render_call(v, "clz", 1); break;
        case VarKind::Ctz:   jitc_metal_render_call(v, "ctz", 1); break;
        case VarKind::Brev: {
            Variable *a = jitc_var(v->dep[0]);
            fmt("$t $v = reverse_bits($v);\n", v, v, a);
            break;
        }

        case VarKind::And: {
            Variable *a0 = jitc_var(v->dep[0]),
                     *a1 = jitc_var(v->dep[1]);
            if (a0->type != a1->type)
                fmt("$t $v = $v ? $v : ($t) 0;\n", v, v, a1, a0, v);
            else if (jitc_is_float(v))
                fmt("$t $v = as_type<$t>(($b)(as_type<$b>($v) & as_type<$b>($v)));\n",
                    v, v, v, v, v, a0, v, a1);
            else
                fmt("$t $v = $v & $v;\n", v, v, a0, a1);
            break;
        }

        case VarKind::Or: {
            Variable *a0 = jitc_var(v->dep[0]),
                     *a1 = jitc_var(v->dep[1]);
            if (a0->type != a1->type)
                fmt("$t $v = as_type<$t>(($b)($v ? ~($b) 0 : as_type<$b>($v)));\n",
                    v, v, v, v, a1, v, v, a0);
            else if (jitc_is_float(v))
                fmt("$t $v = as_type<$t>(($b)(as_type<$b>($v) | as_type<$b>($v)));\n",
                    v, v, v, v, v, a0, v, a1);
            else
                fmt("$t $v = $v | $v;\n", v, v, a0, a1);
            break;
        }

        case VarKind::Xor: {
            Variable *a0 = jitc_var(v->dep[0]),
                     *a1 = jitc_var(v->dep[1]);
            if (jitc_is_float(v))
                fmt("$t $v = as_type<$t>(($b)(as_type<$b>($v) ^ as_type<$b>($v)));\n",
                    v, v, v, v, v, a0, v, a1);
            else
                fmt("$t $v = $v ^ $v;\n", v, v, a0, a1);
            break;
        }
        case VarKind::Shl: jitc_metal_render_binary(v, "<<"); break;
        case VarKind::Shr: jitc_metal_render_binary(v, ">>"); break;

        // -- Unary --
        case VarKind::Neg: jitc_metal_render_unary(v, "-"); break;
        case VarKind::Not: {
            Variable *a = jitc_var(v->dep[0]);
            if ((VarType) v->type == VarType::Bool)
                fmt("$t $v = !$v;\n", v, v, a);
            else if (jitc_is_float(v))
                fmt("$t $v = as_type<$t>(($b)(~as_type<$b>($v)));\n",
                    v, v, v, v, v, a);
            else
                fmt("$t $v = ~$v;\n", v, v, a);
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
            fmt("$t $v = ($t) 1.0 / $v;\n",
                v, v, v, jitc_var(v->dep[0]));
            break;

        case VarKind::RcpApprox:
            fmt("$t $v = fast::divide(($t) 1.0, $v);\n",
                v, v, v, jitc_var(v->dep[0]));
            break;

        case VarKind::RSqrtApprox:
            jitc_metal_render_call(v, "fast::rsqrt", 1);
            break;

        // -- Casts --
        case VarKind::Cast: {
            Variable *a = jitc_var(v->dep[0]);
            fmt("$t $v = ($t) $v;\n", v, v, v, a);
            break;
        }

        case VarKind::Bitcast: {
            Variable *a = jitc_var(v->dep[0]);
            if (v->type == a->type)
                fmt("$t $v = $v;\n", v, v, a);
            else if (type_size[v->type] == type_size[a->type])
                fmt("$t $v = as_type<$t>($v);\n", v, v, v, a);
            else
                fmt("$t $v = as_type<$t>(($b) $v);\n", v, v, v, v, a);
            break;
        }

        // -- Bounds check --
        case VarKind::BoundsCheck: {
            Variable *index  = jitc_var(v->dep[0]);
            Variable *mask   = jitc_var(v->dep[1]);
            Variable *buf    = jitc_var(v->dep[2]);
            uint32_t size    = (uint32_t) v->literal;
            fmt_intrinsic("#include <metal_atomic>");
            fmt("bool $v = $v && ($v < (uint) $uu);\n"
                "if ($v && !$v)\n"
                "    atomic_store_explicit((device atomic_uint*) $v, $v, memory_order_relaxed);\n",
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
                fmt("$t $v = ((device const $t*) $v)[$v];\n",
                    v, v, v, src, index);
            else
                fmt("$t $v = ($v) ? "
                          "((device const $t*) $v)[$v] : ($t) 0;\n",
                    v, v, mask, v, src, index, v);
            break;
        }

        case VarKind::PacketGather: {
            Variable *ptr   = jitc_var(v->dep[0]);
            Variable *index = jitc_var(v->dep[1]);
            Variable *mask  = jitc_var(v->dep[2]);
            jitc_metal_render_gather_packet(v, ptr, index, mask);
            break;
        }

        case VarKind::PacketScatter: {
            Variable *ptr   = jitc_var(v->dep[0]);
            Variable *index = jitc_var(v->dep[1]);
            Variable *mask  = jitc_var(v->dep[2]);
            jitc_metal_render_scatter_packet(v, ptr, index, mask);
            break;
        }

        case VarKind::Scatter:
            jitc_metal_render_scatter(v);
            break;

        case VarKind::ScatterInc:
            jitc_metal_render_scatter_inc(v);
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
                (VarKind) src->kind == VarKind::ScatterCAS ||
                (VarKind) src->kind == VarKind::PacketGather ||
                (VarKind) src->kind == VarKind::TexLookup ||
                (VarKind) src->kind == VarKind::TexFetchBilerp) {
                // Extract from a multi-output op — reference the pre-declared outputs
                fmt("$t $v = $v_out_$u;\n",
                    v, v, src, sub_index);
            } else {
                fmt("$t $v = $v; // extract[$u]\n",
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
                    fmt("$t $v = $v;\n", inner_in, inner_in, outer_in);
                else
                    fmt("$t $v = ($t) 0;\n", inner_in, inner_in, inner_in);
            }
            put("while (true) {\n");
            break;
        }

        case VarKind::LoopCond: {
            Variable *cond = jitc_var(v->dep[1]);
            fmt("if (!$v) break;\n", cond);
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
                fmt("$t $v_tmp = $v;\n", out, out, out);
                out->scratch = 2;
            }

            for (uint32_t i = 0; i < size; ++i) {
                Variable *in  = jitc_var(ld->inner_in[i]),
                         *out = jitc_var(ld->inner_out[i]);
                if (in == out || !in->reg_index || !out->reg_index ||
                    in->is_array())
                    continue;
                if (out->scratch == 2)
                    fmt("$v = $v_tmp;\n", in, out);
                else
                    fmt("$v = $v;\n", in, out);
            }

            // Reset 'scratch' to not interfere with visited tracking in jit_eval()
            for (uint32_t i = 0; i < size; ++i) {
                jitc_var(ld->inner_in[i])->scratch = 0;
                jitc_var(ld->inner_out[i])->scratch = 0;
            }

            put("}\n");
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
                        fmt("$t $v = $v;\n", v, v, inner_in);
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
                    fmt("$t $v;\n", vo, vo);
            }
            fmt("if ($v) {\n", cond);
            break;
        }

        case VarKind::CondMid: {
            Variable *a0 = jitc_var(v->dep[0]);
            const CondData *cd = (CondData *) a0->data;
            for (size_t i = 0; i < cd->indices_out.size(); ++i) {
                Variable *vt = jitc_var(cd->indices_t[i]),
                         *vo = jitc_var(cd->indices_out[i]);
                if (vo && vo->reg_index && vt->reg_index)
                    fmt("$v = $v;\n", vo, vt);
            }
            put("} else {\n");
            break;
        }

        case VarKind::CondEnd: {
            Variable *a0 = jitc_var(v->dep[0]);
            const CondData *cd = (CondData *) a0->data;
            for (size_t i = 0; i < cd->indices_out.size(); ++i) {
                Variable *vf = jitc_var(cd->indices_f[i]),
                         *vo = jitc_var(cd->indices_out[i]);
                if (vo && vo->reg_index && vf->reg_index)
                    fmt("$v = $v;\n", vo, vf);
            }
            put("}\n");
            break;
        }

        case VarKind::CondOutput:
            break;

        // -- Ray tracing (Metal inline intersection) --
        case VarKind::TraceRay: {
            TraceData *td = (TraceData *) v->data;
            Variable *valid = jitc_var(v->dep[0]);
            bool is_unmasked = valid->is_literal() && valid->literal == 1;

            // Initialize all outputs to their canonical miss values (distance
            // +infinity, validity false, the rest zero). Masked lanes skip the
            // block below and missed lanes skip the hit-field write, keeping these.
            if (td->shadow) {
                fmt("bool $v_out_0 = false;\n", v);
            } else {
                fmt("bool $v_out_0 = false;\n"
                    "float $v_out_1 = as_type<float>(0x7f800000u);\n"
                    "float $v_out_2 = 0.0f;\n"
                    "float $v_out_3 = 0.0f;\n"
                    "uint $v_out_4 = 0u;\n"
                    "uint $v_out_5 = 0u;\n"
                    "uint $v_out_6 = 0u;\n"
                    "uint $v_out_7 = 0u;\n",
                    v, v, v, v, v, v, v, v);
            }

            Variable *accel_h = jitc_var(v->dep[2]);
            Variable *ift_h = v->dep[3] ? jitc_var(v->dep[3]) : nullptr;
            MetalScene *scene_local = (MetalScene *) (uintptr_t) accel_h->literal;

            if (!is_unmasked)
                fmt("if ($v) {\n", valid);
            else
                put("{\n");

            // Read ray parameters from TraceData indices
            Variable *ox   = jitc_var(td->indices[0]);
            Variable *oy   = jitc_var(td->indices[1]);
            Variable *oz   = jitc_var(td->indices[2]);
            Variable *dx   = jitc_var(td->indices[3]);
            Variable *dy   = jitc_var(td->indices[4]);
            Variable *dz   = jitc_var(td->indices[5]);
            Variable *tmin = jitc_var(td->indices[6]);
            Variable *tmax = jitc_var(td->indices[7]);

            bool has_ift_local = ift_h != nullptr;
            bool has_curves_local =
                scene_local && (scene_local->geometry_types_mask & 0x4u) != 0;
            bool has_backface_culled_triangles_local =
                scene_local && (scene_local->geometry_types_mask & 0x8u) != 0;
            bool extended = has_ift_local || has_curves_local;

            fmt("raytracing::intersector<raytracing::triangle_data, raytracing::instancing$s$s> _inter;\n",
                has_ift_local ? ", raytracing::world_space_data" : "",
                has_curves_local ? ", raytracing::curve_data" : "");

            if (!extended)
                put("_inter.assume_geometry_type(raytracing::geometry_type::triangle);\n");

            fmt("_inter.force_opacity(raytracing::forced_opacity::opaque);\n"
                "_inter.accept_any_intersection($s);\n",
                td->shadow ? "true" : "false");
            if (has_backface_culled_triangles_local)
                put("_inter.set_triangle_cull_mode(raytracing::triangle_cull_mode::back);\n");

            fmt("raytracing::ray _r;\n"
                "_r.origin = float3($v, $v, $v);\n"
                "_r.direction = float3($v, $v, $v);\n"
                "_r.min_distance = $v;\n"
                "_r.max_distance = $v;\n",
                ox, oy, oz, dx, dy, dz, tmin, tmax);

            // Route the intersect call to this trace's reconstructed accel
            // (+ IFT) reference variables.
            if (has_ift_local)
                fmt("auto _hit = _inter.intersect(_r, $v, $v);\n",
                    accel_h, ift_h);
            else
                fmt("auto _hit = _inter.intersect(_r, $v);\n",
                    accel_h);

            // Hit-result extraction.
            // - Triangle hit: prim_uv = triangle_barycentric_coord
            // - Curve hit:    prim_uv = (curve_parameter, 0)
            // - Bbox hit:     prim_uv = (0, 0) — compute_surface_interaction()
            //                 will recompute it from the hit point.
            // On a hit, overwrite the miss defaults set above.
            put("    auto _ht = _hit.type;\n"
                "    if (_ht != raytracing::intersection_type::none) {\n");

            if (td->shadow) {
                fmt("        $v_out_0 = true;\n",
                    v);
            } else {
                const char *prim_u, *prim_v;
                if (has_curves_local) {
                    prim_u = "(_ht == raytracing::intersection_type::curve)"
                             " ? _hit.curve_parameter : _hit.triangle_barycentric_coord.x";
                    prim_v = "(_ht == raytracing::intersection_type::curve)"
                             " ? 0.0f : _hit.triangle_barycentric_coord.y";
                } else {
                    prim_u = "_hit.triangle_barycentric_coord.x";
                    prim_v = "_hit.triangle_barycentric_coord.y";
                }

                fmt("        $v_out_0 = true;\n"
                    "        $v_out_1 = _hit.distance;\n"
                    "        $v_out_2 = $s;\n"
                    "        $v_out_3 = $s;\n"
                    "        $v_out_4 = _hit.instance_id;\n"
                    "        $v_out_5 = _hit.primitive_id;\n"
                    "        $v_out_6 = _hit.geometry_id;\n"
                    "        $v_out_7 = _hit.user_instance_id;\n",
                    v, v, v, prim_u, v, prim_v, v, v, v, v);
            }

            put("    }\n" // close: if (_ht != none)
                "}\n");   // close: if (valid) / unconditional block
            break;
        }

        case VarKind::TexLookup: {
            // dep[0] = texture handle, dep[1] = sampler handle (both
            // reconstructed as reference variables in the input loop); the
            // coordinates are carried in a side TexData payload. dep[2] is an
            // optional active mask.
            Variable *tex_h = jitc_var(v->dep[0]);
            Variable *smp_h = jitc_var(v->dep[1]);
            Variable *mask  = v->dep[2] ? jitc_var(v->dep[2]) : nullptr;
            TexData *td = (TexData *) v->data;
            size_t ndim = td->ndim;
            Variable *p0 = jitc_var(td->indices[0]);

            // For a masked lookup, default the result to zero and guard the
            // sample with the mask, so the compiler can predicate off the fetch
            // for inactive lanes.
            if (mask) {
                fmt("float4 $v_s = float4(0.f);\n", v);
                fmt("if ($v) $v_s = ", mask, v);
            } else {
                fmt("float4 $v_s = ", v);
            }

            if (ndim == 1)
                // 1D is backed by a height-1 2D texture; sample the single row
                // at its texel center (y = 0.5).
                fmt("$v.sample($v, float2($v, 0.5f));\n", tex_h, smp_h, p0);
            else if (ndim == 2)
                fmt("$v.sample($v, float2($v, $v));\n",
                    tex_h, smp_h, p0, jitc_var(td->indices[1]));
            else
                fmt("$v.sample($v, float3($v, $v, $v));\n",
                    tex_h, smp_h, p0, jitc_var(td->indices[1]),
                    jitc_var(td->indices[2]));

            fmt("float $v_out_0 = $v_s.x;\n"
                "float $v_out_1 = $v_s.y;\n"
                "float $v_out_2 = $v_s.z;\n"
                "float $v_out_3 = $v_s.w;\n",
                v, v, v, v, v, v, v, v);
            break;
        }

        case VarKind::TexFetchBilerp: {
            // 2D-only; ``component`` selects the texture channel to gather. The
            // four returned texels are ordered counter-clockwise from the lower
            // left, matching CUDA's ``tld4``. dep[2] is an optional active mask.
            Variable *tex_h = jitc_var(v->dep[0]);
            Variable *smp_h = jitc_var(v->dep[1]);
            Variable *mask  = v->dep[2] ? jitc_var(v->dep[2]) : nullptr;
            TexData *td = (TexData *) v->data;
            char comp[2] = { "xyzw"[td->component & 0x3u], '\0' };

            // For a masked fetch, default to zero and guard the gather with the
            // mask, so the compiler can predicate off the fetch for inactive lanes.
            if (mask) {
                fmt("float4 $v_s = float4(0.f);\n", v);
                fmt("if ($v) $v_s = ", mask, v);
            } else {
                fmt("float4 $v_s = ", v);
            }
            fmt("$v.gather($v, float2($v, $v), int2(0), component::$s);\n",
                tex_h, smp_h, jitc_var(td->indices[0]),
                jitc_var(td->indices[1]), comp);

            fmt("float $v_out_0 = $v_s.x;\n"
                "float $v_out_1 = $v_s.y;\n"
                "float $v_out_2 = $v_s.z;\n"
                "float $v_out_3 = $v_s.w;\n",
                v, v, v, v, v, v, v, v);
            break;
        }

        case VarKind::TexWrite: {
            Variable *tex_h = jitc_var(v->dep[0]);
            Variable *mask  = v->dep[1] ? jitc_var(v->dep[1]) : nullptr;
            TexData *td = (TexData *) v->data;
            size_t ndim = td->ndim;

            fmt("float4 $v_c = float4(0.f);\n", v);
            for (uint32_t c = 0; c < td->n_values; ++c)
                fmt("$v_c[$u] = $v;\n", v, c, jitc_var(td->values[c]));

            if (mask)
                fmt("if ($v) ", mask);

            if (ndim == 1)
                // 1D is backed by a height-1 2D texture; write row y = 0.
                fmt("$v.write($v_c, uint2($v, 0));\n",
                    tex_h, v, jitc_var(td->indices[0]));
            else if (ndim == 2)
                fmt("$v.write($v_c, uint2($v, $v));\n",
                    tex_h, v, jitc_var(td->indices[0]),
                    jitc_var(td->indices[1]));
            else
                fmt("$v.write($v_c, uint3($v, $v, $v));\n",
                    tex_h, v, jitc_var(td->indices[0]),
                    jitc_var(td->indices[1]), jitc_var(td->indices[2]));
            break;
        }

        case VarKind::Call: {
            Variable *a0 = jitc_var(v->dep[0]),
                     *a1 = jitc_var(v->dep[1]);
            jitc_var_call_assemble((CallData *) v->data, v->reg_index,
                                   a0->reg_index, a1->reg_index);
            break;
        }

        case VarKind::CallGetter: {
            Variable *index = jitc_var(v->dep[0]),
                     *mask  = jitc_var(v->dep[1]);
            jitc_var_call_getter_assemble(v, index, mask);
            break;
        }

        case VarKind::CallOutput:
            break;

        case VarKind::CallSelf:
            fmt("uint $v = self;\n", v);
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
//  Virtual function call support (visible-function-table dispatch)
// ============================================================================

/// Does this call have at least one live output? (If not, its callables return
/// ``void`` instead of a struct.)
static bool jitc_metal_call_has_out(const CallData *call) {
    for (uint32_t i = 0; i < call->n_out; ++i)
        if (call->out_offset[i] != (uint32_t) -1)
            return true;
    return false;
}

/// Emit the callables' return type into the code buffer: ``void`` if the call
/// has no live outputs, otherwise the mangled struct name ``Ret_<codes>`` (one
/// ``type_mangle`` code per active output, in order).
static void jitc_metal_put_ret_type(const CallData *call) {
    if (!jitc_metal_call_has_out(call)) {
        put("void");
        return;
    }
    put("Ret_");
    for (uint32_t i = 0; i < call->n_out; ++i)
        if (call->out_offset[i] != (uint32_t) -1)
            put(type_mangle[jitc_var(call->inner_out[i])->type]);
}

/// Register the return struct's definition (one field ``r<k>`` per active
/// output). Registered as a global, so it precedes the callables and kernel;
/// keyed by content, so repeated registration dedups.
static void jitc_metal_emit_ret_struct(const CallData *call) {
    if (!jitc_metal_call_has_out(call))
        return;
    size_t off = buffer.size();
    put("struct ");
    jitc_metal_put_ret_type(call);
    put(" {\n");
    uint32_t k = 0;
    for (uint32_t i = 0; i < call->n_out; ++i) {
        if (call->out_offset[i] == (uint32_t) -1)
            continue;
        fmt("$t r$u;\n", jitc_var(call->inner_out[i]), k);
        k++;
    }
    put("};");
    jitc_register_global(buffer.get() + off);
    buffer.rewind_to(off);
}

/// Emit the typed parameter list shared by a call's callable definition and the
/// ``visible_function_table`` type used at its dispatch site: a fixed prefix
/// (index, self, data, call_table) followed by one by-value parameter per live
/// input (call->in_active), in argument order. Outputs are returned
/// by value (see jitc_metal_put_ret_type), not passed here. ``with_names``
/// selects a definition signature ("type name") vs. a bare type list (the table
/// element type).
static void jitc_metal_callable_signature(const CallData *call, bool with_names) {
    if (with_names)
        put("uint index, uint self, device uint8_t* data, "
            "constant void* call_table");
    else
        put("uint, uint, device uint8_t*, constant void*");

    // Callables containing a nested call also receive the base pointer.
    if (call->use_nested) {
        if (with_names)
            put(", device uint8_t* base");
        else
            put(", device uint8_t*");
    }

    for (uint32_t i = 0; i < call->n_in; ++i) {
        if (!call->in_active[i])
            continue;
        Variable *vo = jitc_var(call->outer_in[i]);
        if (with_names)
            fmt(", $t a$u", vo, i);
        else
            fmt(", $t", vo);
    }
}

void jitc_metal_assemble_func(const CallData *call, uint32_t inst,
                              uint32_t /*in_size*/, uint32_t /*in_align*/,
                              uint32_t /*out_size*/, uint32_t /*out_align*/,
                              uint32_t /*n_regs*/) {
    jitc_metal_emit_ret_struct(call);

    if (call->n_inst != 1)
        put("[[visible]] ");
    jitc_metal_put_ret_type(call);
    put(" func_^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^(");
    jitc_metal_callable_signature(call, /*with_names=*/true);
    put(") {\n");
    fmt("// Call: $s\n", call->name.c_str());

    // Bind this instance's data slots so jitc_call_slot_rel_offset() resolves them O(1)
    jitc_call_bind_slots(call, inst);

    // Issue one or more vector loads to fetch the coalesced part of the call
    // data (see call.h).
    const CallData::InstanceLayout &call_layout = call->instance_layout[inst];
    uint32_t numeric_end = call_layout.coalesce_end;

    // Only worth coalescing with >= 2 coalesceable fields.
    bool cd_coalesce = call_layout.coalesce_count >= 2;

    if (cd_coalesce) {
        uint32_t end = (numeric_end + 3u) & ~3u; // round up to a full word

        uint32_t p = 0, chunk = 0;
        while (p < end) {
            uint32_t w_bytes = jitc_call_pick_word_chunk_size(p, end - p, 16u);

            const char *lt = w_bytes == 16 ? "uint4"
                           : w_bytes == 8  ? "uint2" : "uint";
            fmt("$s cd_c$u = *(device const $s*)(data + $u);\n",
                lt, chunk, lt, p);

            ++chunk;
            p += w_bytes;
        }
    }

    // Emit the generated MSL expression holding 32-bit word ``word`` of the
    // coalesced prefix.
    auto put_cd_word_expr = [&](uint32_t word) {
        uint32_t target = word * 4,
                 end    = (numeric_end + 3u) & ~3u,
                 p      = 0,
                 chunk  = 0;

        while (p < end) {
            uint32_t w_bytes = jitc_call_pick_word_chunk_size(p, end - p, 16u);
            if (target >= p && target < p + w_bytes) {
                uint32_t lane = (target - p) / 4;
                fmt("cd_c$u", chunk);
                if (w_bytes != 4) {
                    put('.');
                    put("xyzw"[lane]);
                }
                return;
            }
            ++chunk;
            p += w_bytes;
        }

        jitc_fail("jitc_metal_assemble_func(): internal call-data word lookup "
                  "failure.");
    };

    // Emit a declaration extracting a coalesced field of type ``svt`` at relative
    // byte offset ``off``. Fields outside the packet-loadable SizeBucket prefix
    // use the per-field typed load instead.
    auto put_cd_extract = [&](const Variable *v, uint32_t off, VarType svt) {
        uint32_t s = type_size[(int) svt], j = off / 4;
        const char *base = type_name_metal[(int) svt];

        if (svt == VarType::Pointer) {
            // Buffer pointer: reassemble the 64-bit address and C-cast it back to
            // a device pointer (MSL forbids 'as_type' to a pointer type, but the
            // integer-to-device-pointer C cast is well-formed).
            fmt("device uint8_t* $v = "
                "(device uint8_t*)(as_type<ulong>(uint2(", v);
            put_cd_word_expr(j);
            put(", ");
            put_cd_word_expr(j + 1);
            put(")));\n");
        } else if (s == 8) { // Int64 / UInt64 (Float64 cannot occur on Metal)
            fmt("$t $v = as_type<$s>(uint2(", v, v, base);
            put_cd_word_expr(j);
            put(", ");
            put_cd_word_expr(j + 1);
            put("));\n");
        } else if (s == 4) {
            fmt("$t $v = as_type<$s>(", v, v, base);
            put_cd_word_expr(j);
            put(");\n");
        } else { // s == 2
            fmt("$t $v = as_type<$s2>(", v, v, base);
            put_cd_word_expr(j);
            fmt(").$s;\n", (off % 4) ? "y" : "x");
        }
    };

    for (size_t i = 0; i < schedule.size(); ++i) {
        ScheduledVariable &sv = schedule[i];
        Variable *v = jitc_var(sv.index);
        VarType vt = (VarType) v->type;
        VarKind kind = (VarKind) v->kind;

        if (kind == VarKind::Counter) {
            fmt("$t $v = ($t) index;\n", v, v, v);
        } else if (kind == VarKind::CallInput) {
            // Read the typed by-value parameter named after the input index
            // (this node is call->inner_in[i]).
            uint32_t in_i = 0;
            for (; in_i < call->n_in; ++in_i)
                if (call->inner_in[in_i] == sv.index)
                    break;
            fmt("$t $v = a$u;\n", v, v, in_i);
        } else if (kind == VarKind::CallSelf) {
            fmt("uint $v = self;\n", v);
        } else if (v->is_evaluated() || (vt == VarType::Pointer && kind == VarKind::Literal)) {
            uint32_t offset = jitc_call_slot_rel_offset(call, inst, v, sv.index);

            // Opaque resource handles (textures, samplers, ray-tracing
            // structures) reconstruct a typed reference directly from the
            // ``device`` call-data section
            if (vt == VarType::Pointer &&
                jitc_metal_render_resource_handle(v, /*in_callable=*/true,
                                                  offset))
                continue;

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
                mts->metal_call_resources.push_back(
                    { (void *) v->literal, ResourceKind::Buffer,
                      (bool) v->written });
            } else if (v->is_evaluated() && v->data) {
                mts->metal_call_resources.push_back(
                    { v->data, ResourceKind::Buffer, false });
            }

            const CallData::CaptureSlot &capture = call->slots[v->param_offset];
            if (cd_coalesce && capture.bucket.coalesceable()) {
                // Extract this field from the coalesced vector loads; the rest
                // (1-byte fields; opaque handles took the typed path above) fall
                // through to the plain typed loads below.
                put_cd_extract(v, offset, vt);
            } else if (vt == VarType::Bool)
                fmt("bool $v = *(device uint8_t*)(data + $u) != 0;\n",
                    v, offset);
            else if (vt == VarType::Pointer)
                fmt("device uint8_t* $v = "
                    "*(device uint8_t* device const*)(data + $u);\n",
                    v, offset);
            else
                fmt("$t $v = *(device $t*)(data + $u);\n",
                    v, v, v, offset);
        } else if (v->is_literal()) {
            jitc_metal_render(v);
        } else {
            jitc_metal_render(v);
        }
    }

    // Collect outputs into the return struct and return it by value.
    if (jitc_metal_call_has_out(call)) {
        put("    ");
        jitc_metal_put_ret_type(call);
        put(" ret;\n");
        uint32_t k = 0;
        for (uint32_t i = 0; i < call->n_out; ++i) {
            if (call->out_offset[i] == (uint32_t) -1)
                continue;
            const Variable *v = jitc_var(call->inner_out[inst * call->n_out + i]);
            fmt("ret.r$u = $v;\n", k, v);
            k++;
        }
        put("return ret;\n");
    }

    put("}\n");
}

/// Getter masked load (Metal/MSL). Mirrors the 'Gather' case, sourcing from
/// 'base + header_offset'.
void jitc_var_call_getter_assemble_metal(Variable *v, const Variable *index,
                                         const Variable *mask) {
    GetterData *gd = (GetterData *) v->data;
    uint32_t header_offset = gd->header_offset;

    // Name of the base pointer holding the kernel's combined call data
    // (a 'device uint8_t*' at every nesting level).
    char base[32];
    if (callable_depth == 0)
        snprintf(base, sizeof(base), "r%u", call_buffer.base_reg);
    else
        snprintf(base, sizeof(base), "base");

    bool is_unmasked = mask->is_literal() && mask->literal == 1;
    if (is_unmasked)
        fmt("$t $v = ((device const $t*) ($s + $u))[$v];\n",
            v, v, v, base, header_offset, index);
    else
        fmt("$t $v = ($v) ? ((device const $t*) ($s + $u))[$v] : ($t) 0;\n",
            v, v, mask, v, base, header_offset, index, v);
}

void jitc_var_call_assemble_metal(CallData *call, uint32_t call_reg,
                                  uint32_t self_reg, uint32_t mask_reg,
                                  uint32_t /*in_size*/, uint32_t /*in_align*/,
                                  uint32_t /*out_size*/, uint32_t /*out_align*/) {
    // =====================================================
    // 1. Conditional branch (masked lanes)
    // =====================================================

    Variable *mask = jitc_var(jitc_var(call->id)->dep[1]);
    bool is_masked = !mask->is_literal() || mask->literal != 1;

    fmt("\n// VCall: $s\n", call->name.c_str());

    // The return struct is registered while assembling the callable bodies,
    // which precede this dispatch, so it is not (re)emitted here.

    // The outputs the kernel actually consumes, paired with their return-struct
    // field index (``r<k>``, k running over *all* active outputs to match
    // jitc_metal_emit_ret_struct). Drives the declaration, unpacking, and
    // masked-off zeroing below.
    std::vector<std::pair<Variable *, uint32_t>> out_regs;
    out_regs.reserve(call->n_out);
    for (uint32_t i = 0, k = 0; i < call->n_out; ++i) {
        if (call->out_offset[i] == (uint32_t) -1)
            continue;
        Variable *v = jitc_var(call->outer_out[i]);
        if (v && v->reg_index && v->param_type != ParamType::Input)
            out_regs.emplace_back(v, k);
        k++;
    }

    // =====================================================
    // 2. Declare output registers
    // =====================================================

    // Declared up front (uninitialized): the call unpacks its by-value return
    // struct into them; when masked, the ``else`` branch below zeros them on
    // lanes that did not call.
    for (auto [v, field] : out_regs)
        fmt("$t $v;\n", v, v);

    // =====================================================
    // 3. Masked guard + offset table lookup
    // =====================================================

    const char *indent = is_masked ? "            " : "        ";

    if (is_masked)
        fmt("if (r$u) {\n", mask_reg);

    // Name of the base pointer holding the kernel's combined call data
    char base[32];
    if (callable_depth == 0)
        snprintf(base, sizeof(base), "r%u", call_buffer.base_reg);
    else
        snprintf(base, sizeof(base), "base");

    bool has_slots  = !call->slots.empty();
    bool need_entry = call->n_inst != 1 || has_slots;

    // Load the uint64_t offset-table entry: (abs_data_offset << 32) |
    // callable_index. This call's table begins at element 'offset_base/8' of B.
    if (need_entry) {
        fmt("$sulong _oe_$u = ((device const ulong*) $s)[$u + r$u];\n",
            indent, call_reg, base,
            call->offset_base / (uint32_t) sizeof(uint64_t), self_reg);
        if (call->n_inst != 1)
            fmt("$suint _ci_$u = (uint)(_oe_$u & 0xFFFFFFFFu);\n",
                indent, call_reg, call_reg);
    }

    if (has_slots)
        fmt("$sdevice uint8_t* _cd_$u = "
            "(device uint8_t*) $s + (uint)(_oe_$u >> 32);\n",
            indent, call_reg, base, call_reg);

    // =====================================================
    // 4. Dispatch
    // =====================================================

    // Inside a callable the kernel-level `r0` (thread_position_in_grid) and the
    // top-level `params` buffer are not in scope; the enclosing callable
    // receives the thread index as `index` and the table handle as
    // `call_table`. Use those for nested vcalls.
    const char *index_name = (callable_depth > 0) ? "index" : "r0";

    // Emit the typed argument list, matching jitc_metal_callable_signature():
    // index, self, data, call_table, live inputs (by value).
    auto put_args = [&]() {
        fmt("$s, r$u, ", index_name, self_reg);
        if (has_slots)
            fmt("_cd_$u", call_reg);
        else
            put("(device uint8_t*) nullptr");
        if (callable_depth > 0)
            put(", call_table");
        else
            fmt(", (constant void*) &params.args[$u]",
                (uint32_t) metal_vft_arg_index);
        // Callables containing a nested call receive the base pointer.
        if (call->use_nested)
            fmt(", $s", base);
        // Live inputs always have a register: in_active mirrors the packing
        // predicate, which excludes !reg_index inputs.
        for (uint32_t i = 0; i < call->n_in; ++i)
            if (call->in_active[i])
                fmt(", r$u", jitc_var(call->outer_in[i])->reg_index);
    };

    // Capture the by-value return struct (if the call has live outputs).
    fmt("$s", indent);
    if (jitc_metal_call_has_out(call)) {
        jitc_metal_put_ret_type(call);
        fmt(" ret_$u = ", call_reg);
    }

    if (call->n_inst == 1) {
        // Single target: call the (inline-able) callable directly by name.
        XXH128_hash_t hash = call->inst_hash[0];
        put("func_");
        buffer.put_q64_unchecked(hash.high64);
        buffer.put_q64_unchecked(hash.low64);
        put("(");
        put_args();
        put(");\n");
    } else {
        // Multiple targets: indirect dispatch through the kernel's single
        // visible function table, reinterpreted with this site's signature.
        // The callable index is the low 32 bits of the offset entry.
        put("(*(constant visible_function_table<");
        jitc_metal_put_ret_type(call);
        put(" (");
        jitc_metal_callable_signature(call, /*with_names=*/false);
        if (callable_depth > 0)
            fmt(")>*) call_table)[_ci_$u](", call_reg);
        else
            fmt(")>*) &params.args[$u])[_ci_$u](",
                (uint32_t) metal_vft_arg_index, call_reg);
        put_args();
        put(");\n");
    }

    // Unpack the return struct into the output registers.
    for (auto [v, field] : out_regs)
        fmt("$s$v = ret_$u.r$u;\n", indent, v, call_reg, field);

    if (is_masked) {
        // Lanes that did not call still need defined outputs: zero them in the
        // ``else`` branch.
        if (!out_regs.empty()) {
            put("} else {\n");
            for (auto [v, field] : out_regs)
                fmt("$s$v = ($t) 0;\n", indent, v, v);
        }
        put("}\n");
    }

    put("\n");
}
