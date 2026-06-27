/*
    src/amd_eval.cpp -- AMD/HIP source code generation via HIPRTC
*/

#include "eval.h"

#if defined(DRJIT_ENABLE_AMD)

#include "amd_api.h"
#include "amd_rt.h"
#include "array.h"
#include "call.h"
#include "cond.h"
#include "internal.h"
#include "log.h"
#include "loop.h"
#include "op.h"
#include "trace.h"
#include "util.h"
#include "var.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#  include <stdlib.h>
#endif

#if defined(DRJIT_ENABLE_HIPRT)
#  include <hiprt/hiprt.h>
#endif

#if defined(DRJIT_ENABLE_HIPRT)
static std::vector<hiprtFuncNameSet> amd_kernel_func_name_sets;
static uint32_t amd_kernel_num_geom_types = 1;
static uint32_t amd_kernel_num_ray_types = 1;
static std::string amd_kernel_device_source;
#endif

static const char *jitc_amd_type(VarType vt) {
    switch (vt) {
        case VarType::Void:    return "void";
        case VarType::Bool:    return "bool";
        case VarType::Int8:    return "int8_t";
        case VarType::UInt8:   return "uint8_t";
        case VarType::Int16:   return "int16_t";
        case VarType::UInt16:  return "uint16_t";
        case VarType::Int32:   return "int32_t";
        case VarType::UInt32:  return "uint32_t";
        case VarType::Int64:   return "int64_t";
        case VarType::UInt64:  return "uint64_t";
        case VarType::Pointer: return "uintptr_t";
        case VarType::Float16: return "__half";
        case VarType::Float32: return "float";
        case VarType::Float64: return "double";
        default:
            jitc_fail("jitc_amd_type(): unsupported variable type %u.",
                      (uint32_t) vt);
    }
}

static const char *jitc_amd_type(const Variable *v) {
    return jitc_amd_type((VarType) v->type);
}

static const char *jitc_amd_binary_type(uint32_t size) {
    switch (size) {
        case 1: return "uint8_t";
        case 2: return "uint16_t";
        case 4: return "uint32_t";
        case 8: return "uint64_t";
        default:
            jitc_fail("jitc_amd_binary_type(): unsupported type size %u.",
                      size);
    }
}

static const char *jitc_amd_zero(VarType vt) {
    switch (vt) {
        case VarType::Bool:
            return "false";
        case VarType::Float16:
            return "__ushort_as_half((unsigned short) 0)";
        case VarType::Float32:
            return "0.0f";
        case VarType::Float64:
            return "0.0";
        default:
            return "0";
    }
}

static uint32_t jitc_amd_param_slot(const Variable *v) {
    return v->param_offset / (uint32_t) sizeof(void *);
}

static void jitc_amd_var(const Variable *v) {
    buffer.fmt("r%u", v->reg_index);
}

static void jitc_amd_literal(const Variable *v) {
    VarType vt = (VarType) v->type;
    switch (vt) {
        case VarType::Bool:
            buffer.fmt("%s", v->literal ? "true" : "false");
            break;

        case VarType::Float16:
            buffer.fmt("__ushort_as_half((unsigned short) 0x%xu)",
                       (uint32_t) v->literal & 0xffffu);
            break;

        case VarType::Float32:
            buffer.fmt("drjit_bits_to_float(0x%xu)",
                       (uint32_t) v->literal);
            break;

        case VarType::Float64:
            buffer.fmt("drjit_bits_to_double(0x%llxull)",
                       (unsigned long long) v->literal);
            break;

        case VarType::Pointer:
            buffer.fmt("(uintptr_t) 0x%llxull",
                       (unsigned long long) v->literal);
            break;

        case VarType::Int8:
        case VarType::Int16:
        case VarType::Int32:
        case VarType::Int64:
            buffer.fmt("(%s) %lld", jitc_amd_type(v),
                       (long long) v->literal);
            break;

        case VarType::UInt8:
        case VarType::UInt16:
        case VarType::UInt32:
        case VarType::UInt64:
            buffer.fmt("(%s) %lluull", jitc_amd_type(v),
                       (unsigned long long) v->literal);
            break;

        default:
            jitc_fail("jitc_amd_literal(): unsupported literal type %u.",
                      (uint32_t) vt);
    }
}

static void jitc_amd_emit_decl(const Variable *v) {
    buffer.fmt("%s ", jitc_amd_type(v));
    jitc_amd_var(v);
    buffer.put(" = ");
}

static void jitc_amd_render_binary(Variable *v, const char *op) {
    Variable *a0 = jitc_var(v->dep[0]),
             *a1 = jitc_var(v->dep[1]);
    jitc_amd_emit_decl(v);
    jitc_amd_var(a0);
    buffer.fmt(" %s ", op);
    jitc_amd_var(a1);
    buffer.put(";\n");
}

static void jitc_amd_render_unary(Variable *v, const char *op) {
    Variable *a0 = jitc_var(v->dep[0]);
    jitc_amd_emit_decl(v);
    buffer.fmt("%s", op);
    jitc_amd_var(a0);
    buffer.put(";\n");
}

static void jitc_amd_render_call(Variable *v, const char *fn, uint32_t n_args) {
    jitc_amd_emit_decl(v);
    buffer.fmt("%s(", fn);
    for (uint32_t i = 0; i < n_args; ++i) {
        if (i)
            buffer.put(", ");
        jitc_amd_var(jitc_var(v->dep[i]));
    }
    buffer.put(");\n");
}

static void jitc_amd_render_float_call(Variable *v, const char *fn_half,
                                       const char *fn_float,
                                       const char *fn_double) {
    if (jitc_is_half(v))
        jitc_amd_render_call(v, fn_half, 1);
    else if (jitc_is_single(v))
        jitc_amd_render_call(v, fn_float, 1);
    else
        jitc_amd_render_call(v, fn_double, 1);
}

static void jitc_amd_render_round(Variable *v, const char *fn) {
    Variable *a0 = jitc_var(v->dep[0]);
    jitc_amd_emit_decl(v);
    if (jitc_is_half(v)) {
        const char *fn_half =
            strcmp(fn, "ceil") == 0  ? "drjit_ceil"  :
            strcmp(fn, "floor") == 0 ? "drjit_floor" :
            strcmp(fn, "rint") == 0  ? "drjit_rint"  : "drjit_trunc";
        buffer.fmt("%s(", fn_half);
        jitc_amd_var(a0);
        buffer.put(")");
    } else if (jitc_is_float(v)) {
        buffer.fmt("%s(", jitc_is_single(v) ? (strcmp(fn, "ceil") == 0  ? "ceilf"  :
                                               strcmp(fn, "floor") == 0 ? "floorf" :
                                               strcmp(fn, "rint") == 0  ? "rintf"  : "truncf")
                                           : fn);
        jitc_amd_var(a0);
        buffer.put(")");
    } else if ((VarKind) v->kind == VarKind::Trunc) {
        buffer.fmt("(%s) ", jitc_amd_type(v));
        jitc_amd_var(a0);
    } else {
        buffer.fmt("(%s) %s(", jitc_amd_type(v), fn);
        jitc_amd_var(a0);
        buffer.put(")");
    }
    buffer.put(";\n");
}

static void jitc_amd_render_array(Variable *v, Variable *pred) {
    if (pred && pred->array_state != (uint32_t) ArrayState::Conflicted) {
        v->reg_index = pred->reg_index;
        return;
    }

    buffer.fmt("%s arr_%u[%u];\n", jitc_amd_type(v), v->reg_index,
               (uint32_t) v->array_length);
}

static void jitc_amd_render_array_init(Variable *v, Variable *pred,
                                       Variable *value) {
    v->reg_index = pred->reg_index;

    buffer.fmt("for (uint32_t _i = 0; _i < %uu; ++_i)\n"
               "    arr_%u[_i] = r%u;\n",
               (uint32_t) v->array_length, v->reg_index, value->reg_index);
}

static void jitc_amd_render_array_read(Variable *v, Variable *source,
                                       Variable *mask, Variable *offset) {
    if (!mask->is_literal())
        buffer.fmt("%s r%u = (%s) (%s);\n", jitc_amd_type(v), v->reg_index,
                   jitc_amd_type(v), jitc_amd_zero((VarType) v->type));

    if (offset) {
        if (!mask->is_literal())
            buffer.fmt("if (r%u) r%u = arr_%u[r%u];\n",
                       mask->reg_index, v->reg_index, source->reg_index,
                       offset->reg_index);
        else
            buffer.fmt("%s r%u = arr_%u[r%u];\n", jitc_amd_type(v),
                       v->reg_index, source->reg_index, offset->reg_index);
    } else {
        if (!mask->is_literal())
            buffer.fmt("if (r%u) r%u = arr_%u[%u];\n",
                       mask->reg_index, v->reg_index, source->reg_index,
                       (uint32_t) v->literal);
        else
            buffer.fmt("%s r%u = arr_%u[%u];\n", jitc_amd_type(v),
                       v->reg_index, source->reg_index,
                       (uint32_t) v->literal);
    }
}

static void jitc_amd_render_array_write(Variable *v, Variable *target,
                                        Variable *value, Variable *mask,
                                        Variable *offset) {
    if (offset && offset->is_array())
        offset = nullptr;

    bool copy = target->array_state == (uint32_t) ArrayState::Conflicted;
    uint32_t target_buffer = target->reg_index;

    if (copy) {
        target_buffer = jitc_array_buffer(v)->reg_index;
        buffer.fmt("for (uint32_t _i = 0; _i < %uu; ++_i)\n"
                   "    arr_%u[_i] = arr_%u[_i];\n",
                   (uint32_t) v->array_length, target_buffer,
                   target->reg_index);
    }

    if (offset) {
        if (!mask->is_literal())
            buffer.fmt("if (r%u) arr_%u[r%u] = r%u;\n",
                       mask->reg_index, target_buffer, offset->reg_index,
                       value->reg_index);
        else
            buffer.fmt("arr_%u[r%u] = r%u;\n", target_buffer,
                       offset->reg_index, value->reg_index);
    } else {
        if (!mask->is_literal())
            buffer.fmt("if (r%u) arr_%u[%u] = r%u;\n",
                       mask->reg_index, target_buffer, (uint32_t) v->literal,
                       value->reg_index);
        else
            buffer.fmt("arr_%u[%u] = r%u;\n", target_buffer,
                       (uint32_t) v->literal, value->reg_index);
    }

    v->reg_index = target_buffer;
}

static void jitc_amd_render_array_select(Variable *v, Variable *mask,
                                         Variable *t, Variable *f) {
    uint32_t reg_index = jitc_array_buffer(v)->reg_index;
    buffer.fmt("if (r%u) {\n"
               "    for (uint32_t _i = 0; _i < %uu; ++_i)\n"
               "        arr_%u[_i] = arr_%u[_i];\n"
               "} else {\n"
               "    for (uint32_t _i = 0; _i < %uu; ++_i)\n"
               "        arr_%u[_i] = arr_%u[_i];\n"
               "}\n",
               mask->reg_index,
               (uint32_t) f->array_length, reg_index, t->reg_index,
               (uint32_t) f->array_length, reg_index, f->reg_index);

    v->reg_index = reg_index;
}

static void jitc_amd_render_array_memcpy_in(const Variable *v,
                                            uint32_t slot) {
    buffer.fmt("%s arr_%u[%u];\n", jitc_amd_type(v), v->reg_index,
               (uint32_t) v->array_length);
    buffer.fmt("for (uint32_t _i = 0; _i < %uu; ++_i)\n"
               "    arr_%u[_i] = ((const %s *) params[%u])[_i * size + r0];\n",
               (uint32_t) v->array_length, v->reg_index, jitc_amd_type(v),
               slot);
}

static void jitc_amd_render_array_memcpy_out(const Variable *v,
                                             uint32_t slot) {
    buffer.fmt("for (uint32_t _i = 0; _i < %uu; ++_i)\n"
               "    ((%s *) params[%u])[_i * size + r0] = arr_%u[_i];\n",
               (uint32_t) v->array_length, jitc_amd_type(v), slot,
               v->reg_index);
}

static void jitc_amd_render_bitwise_float_binary(Variable *v,
                                                 const char *op) {
    Variable *a0 = jitc_var(v->dep[0]),
             *a1 = jitc_var(v->dep[1]);
    const char *bt = type_size[v->type] == 8 ? "uint64_t" :
                     type_size[v->type] == 4 ? "uint32_t" : "uint16_t";
    jitc_amd_emit_decl(v);
    buffer.fmt("drjit_bitcast<%s>((%s) (drjit_bitcast<%s>(", jitc_amd_type(v),
               bt, bt);
    jitc_amd_var(a0);
    buffer.fmt(") %s drjit_bitcast<%s>(", op, bt);
    jitc_amd_var(a1);
    buffer.put(")));\n");
}

static void jitc_amd_render_bitwise_float_unary(Variable *v,
                                                const char *op) {
    Variable *a = jitc_var(v->dep[0]);
    const char *bt = type_size[v->type] == 8 ? "uint64_t" :
                     type_size[v->type] == 4 ? "uint32_t" : "uint16_t";
    jitc_amd_emit_decl(v);
    buffer.fmt("drjit_bitcast<%s>((%s) (%sdrjit_bitcast<%s>(",
               jitc_amd_type(v), bt, op, bt);
    jitc_amd_var(a);
    buffer.put(")));\n");
}

static void jitc_amd_render_gather(Variable *v) {
    Variable *src   = jitc_var(v->dep[0]);
    Variable *index = jitc_var(v->dep[1]);
    Variable *mask  = jitc_var(v->dep[2]);
    bool is_unmasked = mask->is_literal() && mask->literal == 1;

    jitc_amd_emit_decl(v);
    if (!is_unmasked) {
        jitc_amd_var(mask);
        buffer.put(" ? ");
    }
    buffer.fmt("((const %s *) (uintptr_t) ", jitc_amd_type(v));
    jitc_amd_var(src);
    buffer.put(")[");
    jitc_amd_var(index);
    buffer.put("]");
    if (!is_unmasked) {
        buffer.fmt(" : (%s) (%s)", jitc_amd_type(v),
                   jitc_amd_zero((VarType) v->type));
    }
    buffer.put(";\n");
}

static void jitc_amd_render_gather_packet(Variable *v) {
    Variable *ptr   = jitc_var(v->dep[0]);
    Variable *index = jitc_var(v->dep[1]);
    Variable *mask  = jitc_var(v->dep[2]);
    bool is_unmasked = mask->is_literal() && mask->literal == 1;

    VarType vt = (VarType) v->type;
    uint32_t count = (uint32_t) v->literal,
             tsize = type_size[v->type],
             total_bytes = count * tsize,
             id = v->reg_index;

    for (uint32_t i = 0; i < count; ++i) {
        buffer.fmt("%s r%u_out_%u = (%s) (%s);\n",
                   jitc_amd_type(v), id, i, jitc_amd_type(v),
                   jitc_amd_zero(vt));
    }

    if (!is_unmasked) {
        buffer.put("if (");
        jitc_amd_var(mask);
        buffer.put(") {\n");
    } else {
        buffer.put("{\n");
    }

    buffer.fmt("const uint8_t *_packet_%u = "
               "(const uint8_t *) (uintptr_t) ", id);
    jitc_amd_var(ptr);
    buffer.put(" + (uintptr_t) ");
    jitc_amd_var(index);
    buffer.fmt(" * %uu;\n", total_bytes);

    for (uint32_t i = 0; i < count; ++i) {
        if (vt == VarType::Bool) {
            buffer.fmt("r%u_out_%u = *((const uint8_t *) "
                       "(_packet_%u + %uu)) != 0;\n",
                       id, i, id, i * tsize);
        } else {
            buffer.fmt("r%u_out_%u = *((const %s *) "
                       "(_packet_%u + %uu));\n",
                       id, i, jitc_amd_type(v), id, i * tsize);
        }
    }

    buffer.put("}\n");
}

static void jitc_amd_render_scatter_packet(Variable *v) {
    Variable *ptr   = jitc_var(v->dep[0]);
    Variable *index = jitc_var(v->dep[1]);
    Variable *mask  = jitc_var(v->dep[2]);
    PacketScatterData *psd = (PacketScatterData *) v->data;
    const std::vector<uint32_t> &values = psd->values;
    Variable *v0 = jitc_var(values[0]);

    if (psd->op != ReduceOp::Identity)
        jitc_raise("jitc_amd_render_scatter_packet(): packet scatter-reduce "
                   "nodes should have been lowered to scalar AMD scatters.");

    bool is_unmasked = mask->is_literal() && mask->literal == 1;
    VarType vt = (VarType) v0->type;
    uint32_t count = (uint32_t) values.size(),
             id = v->reg_index;
    const char *packet_type =
        vt == VarType::Bool ? "uint8_t" : jitc_amd_type(v0);

    if (!is_unmasked) {
        buffer.put("if (");
        jitc_amd_var(mask);
        buffer.put(") {\n");
    } else {
        buffer.put("{\n");
    }

    buffer.fmt("%s *_packet_%u = (%s *) (uintptr_t) ",
               packet_type, id, packet_type);
    jitc_amd_var(ptr);
    buffer.put(" + (uintptr_t) ");
    jitc_amd_var(index);
    buffer.put(";\n");

    for (uint32_t i = 0; i < count; ++i) {
        Variable *value = jitc_var(values[i]);
        if (vt == VarType::Bool) {
            buffer.fmt("_packet_%u[%u] = (uint8_t) (", id, i);
            jitc_amd_var(value);
            buffer.put(" ? 1u : 0u);\n");
        } else {
            buffer.fmt("_packet_%u[%u] = ", id, i);
            jitc_amd_var(value);
            buffer.put(";\n");
        }
    }

    buffer.put("}\n");
}

static void jitc_amd_render_scatter(Variable *v) {
    Variable *ptr   = jitc_var(v->dep[0]);
    Variable *value = jitc_var(v->dep[1]);
    Variable *index = jitc_var(v->dep[2]);
    Variable *mask  = jitc_var(v->dep[3]);

    ReduceOp op = (ReduceOp) (uint32_t) v->literal;
    VarType vt = (VarType) value->type;
    bool is_unmasked = mask->is_literal() && mask->literal == 1;

    if (!is_unmasked) {
        buffer.put("if (");
        jitc_amd_var(mask);
        buffer.put(") ");
    }

    if (op == ReduceOp::Identity) {
        buffer.fmt("((%s *) (uintptr_t) ", jitc_amd_type(value));
        jitc_amd_var(ptr);
        buffer.put(")[");
        jitc_amd_var(index);
        buffer.put("] = ");
        jitc_amd_var(value);
        buffer.put(";\n");
        return;
    }

    if (vt == VarType::Float16) {
        uint32_t id = v->reg_index;
        if (op == ReduceOp::Add) {
            buffer.put("atomicAdd((__half *) (uintptr_t) ");
            jitc_amd_var(ptr);
            buffer.put(" + ");
            jitc_amd_var(index);
            buffer.put(", ");
            jitc_amd_var(value);
            buffer.put(");\n");
            return;
        } else if (op == ReduceOp::Min || op == ReduceOp::Max) {
            const char *fn = op == ReduceOp::Min ? "fminf" : "fmaxf";
            buffer.fmt("{\n"
                       "unsigned short *_addr_%u = "
                       "(unsigned short *) ((__half *) (uintptr_t) ",
                       id);
            jitc_amd_var(ptr);
            buffer.put(" + ");
            jitc_amd_var(index);
            buffer.fmt(");\n"
                       "unsigned short _old_%u = *_addr_%u;\n"
                       "while (true) {\n"
                       "    __half _old_h_%u = __ushort_as_half(_old_%u);\n"
                       "    __half _new_h_%u = (__half) %s((float) _old_h_%u, (float) ",
                       id, id, id, id, id, fn, id);
            jitc_amd_var(value);
            buffer.fmt(");\n"
                       "    unsigned short _new_%u = __half_as_ushort(_new_h_%u);\n"
                       "    unsigned short _prev_%u = atomicCAS(_addr_%u, _old_%u, _new_%u);\n"
                       "    if (_prev_%u == _old_%u)\n"
                       "        break;\n"
                       "    _old_%u = _prev_%u;\n"
                       "}\n"
                       "}\n",
                       id, id, id, id, id, id, id, id, id, id);
            return;
        }
    }

    const char *atomic_fn = nullptr,
               *atomic_type = jitc_amd_type(value);
    switch (op) {
        case ReduceOp::Add:
            atomic_fn = "atomicAdd";
            if (vt == VarType::Int64)
                atomic_type = "uint64_t";
            if (!(vt == VarType::Int32 || vt == VarType::UInt32 ||
                  vt == VarType::Int64 || vt == VarType::UInt64 ||
                  vt == VarType::Float32 || vt == VarType::Float64))
                jitc_raise("jitc_amd_render_scatter(): atomic add for %s is "
                           "not implemented yet.", type_name[(uint32_t) vt]);
            break;

        case ReduceOp::Min:
            atomic_fn = "atomicMin";
            if (!(vt == VarType::Int32 || vt == VarType::UInt32 ||
                  vt == VarType::Int64 || vt == VarType::UInt64 ||
                  vt == VarType::Float32 || vt == VarType::Float64))
                jitc_raise("jitc_amd_render_scatter(): atomic min for %s is "
                           "not implemented yet.", type_name[(uint32_t) vt]);
            break;

        case ReduceOp::Max:
            atomic_fn = "atomicMax";
            if (!(vt == VarType::Int32 || vt == VarType::UInt32 ||
                  vt == VarType::Int64 || vt == VarType::UInt64 ||
                  vt == VarType::Float32 || vt == VarType::Float64))
                jitc_raise("jitc_amd_render_scatter(): atomic max for %s is "
                           "not implemented yet.", type_name[(uint32_t) vt]);
            break;

        case ReduceOp::And:
            atomic_fn = "atomicAnd";
            if (vt == VarType::Int32)
                atomic_type = "uint32_t";
            else if (vt == VarType::Int64 || vt == VarType::UInt64)
                atomic_type = "uint64_t";
            if (!(vt == VarType::Int32 || vt == VarType::UInt32 ||
                  vt == VarType::Int64 || vt == VarType::UInt64))
                jitc_raise("jitc_amd_render_scatter(): atomic and for %s is "
                           "not implemented yet.", type_name[(uint32_t) vt]);
            break;

        case ReduceOp::Or:
            atomic_fn = "atomicOr";
            if (vt == VarType::Int32)
                atomic_type = "uint32_t";
            else if (vt == VarType::Int64 || vt == VarType::UInt64)
                atomic_type = "uint64_t";
            if (!(vt == VarType::Int32 || vt == VarType::UInt32 ||
                  vt == VarType::Int64 || vt == VarType::UInt64))
                jitc_raise("jitc_amd_render_scatter(): atomic or for %s is "
                           "not implemented yet.", type_name[(uint32_t) vt]);
            break;

        default:
            jitc_raise("jitc_amd_render_scatter(): scatter-reduce operation "
                       "\"%s\" is not implemented yet.",
                       red_name[(uint32_t) op]);
    }

    buffer.fmt("%s((%s *) (uintptr_t) ", atomic_fn, atomic_type);
    jitc_amd_var(ptr);
    buffer.put(" + ");
    jitc_amd_var(index);
    buffer.fmt(", (%s) ", atomic_type);
    jitc_amd_var(value);
    buffer.put(");\n");
}

static void jitc_amd_render_scatter_inc(Variable *v) {
    Variable *ptr   = jitc_var(v->dep[0]);
    Variable *index = jitc_var(v->dep[1]);
    Variable *mask  = jitc_var(v->dep[2]);

    bool is_unmasked = mask->is_literal() && mask->literal == 1;

    jitc_amd_emit_decl(v);
    buffer.put("0u;\n");

    if (!is_unmasked) {
        buffer.put("if (");
        jitc_amd_var(mask);
        buffer.put(") ");
    }

    jitc_amd_var(v);
    buffer.put(" = atomicAdd((uint32_t *) (uintptr_t) ");
    jitc_amd_var(ptr);
    buffer.put(" + ");
    jitc_amd_var(index);
    buffer.put(", 1u);\n");

    v->consumed = 1;
}

static const char *jitc_amd_atomic_uint_type(VarType vt) {
    switch (type_size[(int) vt]) {
        case 2: return "unsigned short";
        case 4: return "uint32_t";
        case 8: return "unsigned long long";
        default:
            jitc_raise("jitc_amd_atomic_uint_type(): unsupported type %s.",
                       type_name[(int) vt]);
    }
}

static const char *jitc_amd_atomic_value_expr(Variable *v) {
    VarType vt = (VarType) v->type;
    if (vt == VarType::Float16)
        return "__half_as_ushort";
    else if (vt == VarType::Float32)
        return "drjit_bitcast<uint32_t>";
    else if (vt == VarType::Float64)
        return "drjit_bitcast<unsigned long long>";
    else
        return nullptr;
}

static void jitc_amd_put_atomic_value(Variable *v) {
    const char *expr = jitc_amd_atomic_value_expr(v);
    if (expr) {
        buffer.fmt("%s(", expr);
        jitc_amd_var(v);
        buffer.put(")");
    } else {
        buffer.fmt("(%s) ", jitc_amd_atomic_uint_type((VarType) v->type));
        jitc_amd_var(v);
    }
}

static void jitc_amd_put_atomic_result_cast(VarType vt, const char *var) {
    switch (vt) {
        case VarType::Float16:
            buffer.fmt("__ushort_as_half((unsigned short) %s)", var);
            break;
        case VarType::Float32:
            buffer.fmt("drjit_bitcast<float>((uint32_t) %s)", var);
            break;
        case VarType::Float64:
            buffer.fmt("drjit_bitcast<double>((uint64_t) %s)", var);
            break;
        default:
            buffer.fmt("(%s) %s", jitc_amd_type(vt), var);
            break;
    }
}

static void jitc_amd_render_scatter_exch(Variable *v) {
    Variable *ptr   = jitc_var(v->dep[0]);
    Variable *value = jitc_var(v->dep[1]);
    Variable *index = jitc_var(v->dep[2]);
    Variable *mask  = jitc_var(v->dep[3]);
    VarType vt = (VarType) value->type;
    bool is_unmasked = mask->is_literal() && mask->literal == 1;
    uint32_t id = v->reg_index;

    jitc_amd_emit_decl(v);
    buffer.fmt("%s", jitc_amd_zero(vt));
    buffer.put(";\n");

    if (!is_unmasked) {
        buffer.put("if (");
        jitc_amd_var(mask);
        buffer.put(") {\n");
    } else {
        buffer.put("{\n");
    }

    const char *atomic_type = jitc_amd_atomic_uint_type(vt);
    buffer.fmt("%s *_addr_%u = (%s *) ((%s *) (uintptr_t) ",
               atomic_type, id, atomic_type, jitc_amd_type(value));
    jitc_amd_var(ptr);
    buffer.put(" + ");
    jitc_amd_var(index);
    buffer.fmt(");\n%s _old_%u = atomicExch(_addr_%u, ",
               atomic_type, id, id);
    jitc_amd_put_atomic_value(value);
    buffer.put(");\n");
    jitc_amd_var(v);
    buffer.put(" = ");
    char tmp[64];
    snprintf(tmp, sizeof(tmp), "_old_%u", id);
    jitc_amd_put_atomic_result_cast(vt, tmp);
    buffer.put(";\n}\n");

    v->consumed = 1;
}

static void jitc_amd_render_scatter_cas(Variable *v) {
    Variable *ptr     = jitc_var(v->dep[0]);
    Variable *compare = jitc_var(v->dep[1]);
    Variable *value   = jitc_var(v->dep[2]);
    Variable *index   = jitc_var(v->dep[3]);

    ScatterCASDData *cas_data = (ScatterCASDData *) v->data;
    Variable *mask = jitc_var(cas_data->mask);
    VarType vt = (VarType) value->type;
    bool is_unmasked = mask->is_literal() && mask->literal == 1;
    uint32_t id = v->reg_index;

    buffer.fmt("%s r%u_out_0 = %s;\n"
               "bool r%u_out_1 = false;\n",
               jitc_amd_type(value), id, jitc_amd_zero(vt), id);

    if (!is_unmasked) {
        buffer.put("if (");
        jitc_amd_var(mask);
        buffer.put(") {\n");
    } else {
        buffer.put("{\n");
    }

    const char *atomic_type = jitc_amd_atomic_uint_type(vt);
    buffer.fmt("%s *_addr_%u = (%s *) ((%s *) (uintptr_t) ",
               atomic_type, id, atomic_type, jitc_amd_type(value));
    jitc_amd_var(ptr);
    buffer.put(" + ");
    jitc_amd_var(index);
    buffer.fmt(");\n%s _expected_%u = ", atomic_type, id);
    jitc_amd_put_atomic_value(compare);
    buffer.fmt(";\n%s _old_%u = atomicCAS(_addr_%u, _expected_%u, ",
               atomic_type, id, id, id);
    jitc_amd_put_atomic_value(value);
    buffer.put(");\n");
    buffer.fmt("r%u_out_0 = ", id);
    char tmp[64];
    snprintf(tmp, sizeof(tmp), "_old_%u", id);
    jitc_amd_put_atomic_result_cast(vt, tmp);
    buffer.fmt(";\nr%u_out_1 = _old_%u == _expected_%u;\n}\n",
               id, id, id);

    v->consumed = 1;
}

static void jitc_amd_render_trace(Variable *v) {
    TraceData *td = (TraceData *) v->data;
    Variable *valid = jitc_var(v->dep[0]);
    Variable *scene_h = jitc_var(v->dep[2]);
    Variable *func_h = v->dep[3] ? jitc_var(v->dep[3]) : nullptr;
    bool is_unmasked = valid->is_literal() && valid->literal == 1;
    uint32_t id = v->reg_index;

    if (td->shadow) {
        buffer.fmt("bool r%u_out_0 = false;\n", id);
    } else {
        buffer.fmt("bool r%u_out_0 = false;\n"
                   "float r%u_out_1 = drjit_bits_to_float(0x7f800000u);\n"
                   "float r%u_out_2 = 0.0f;\n"
                   "float r%u_out_3 = 0.0f;\n"
                   "uint32_t r%u_out_4 = 0u;\n"
                   "uint32_t r%u_out_5 = 0u;\n"
                   "uint32_t r%u_out_6 = 0u;\n"
                   "uint32_t r%u_out_7 = 0u;\n",
                   id, id, id, id, id, id, id, id);
    }

    if (!is_unmasked) {
        buffer.put("if (");
        jitc_amd_var(valid);
        buffer.put(") {\n");
    } else {
        buffer.put("{\n");
    }

    Variable *ox   = jitc_var(td->indices[0]);
    Variable *oy   = jitc_var(td->indices[1]);
    Variable *oz   = jitc_var(td->indices[2]);
    Variable *dx   = jitc_var(td->indices[3]);
    Variable *dy   = jitc_var(td->indices[4]);
    Variable *dz   = jitc_var(td->indices[5]);
    Variable *tmin = jitc_var(td->indices[6]);
    Variable *tmax = jitc_var(td->indices[7]);

    buffer.fmt("hiprtRay _ray_%u;\n"
               "_ray_%u.origin = hiprtFloat3{", id, id);
    jitc_amd_var(ox);
    buffer.put(", ");
    jitc_amd_var(oy);
    buffer.put(", ");
    jitc_amd_var(oz);
    buffer.put("};\n");
    buffer.fmt("_ray_%u.direction = hiprtFloat3{", id);
    jitc_amd_var(dx);
    buffer.put(", ");
    jitc_amd_var(dy);
    buffer.put(", ");
    jitc_amd_var(dz);
    buffer.put("};\n");
    buffer.fmt("_ray_%u.minT = ", id);
    jitc_amd_var(tmin);
    buffer.put(";\n");
    buffer.fmt("_ray_%u.maxT = ", id);
    jitc_amd_var(tmax);
    buffer.put(";\n");

    buffer.fmt("hiprtScene _scene_%u = (hiprtScene) (uintptr_t) ", id);
    jitc_amd_var(scene_h);
    buffer.put(";\n");
    if (func_h) {
        buffer.fmt("hiprtFuncTable _func_table_%u = (hiprtFuncTable) "
                   "(uintptr_t) ", id);
        jitc_amd_var(func_h);
        buffer.put(";\n");
    }
    std::string func_table_arg =
        func_h ? ("_func_table_" + std::to_string(id)) : "nullptr";

    if (td->shadow) {
        buffer.fmt("hiprtSceneTraversalAnyHit _tr_%u(_scene_%u, _ray_%u, "
                   "hiprtFullRayMask, hiprtTraversalHintShadowRays, nullptr, "
                   "%s);\n"
                   "hiprtHit _hit_%u = _tr_%u.getNextHit();\n"
                   "if (_hit_%u.hasHit())\n"
                   "    r%u_out_0 = true;\n",
                   id, id, id,
                   func_table_arg.c_str(),
                   id, id, id, id);
    } else {
        buffer.fmt("hiprtSceneTraversalClosest _tr_%u(_scene_%u, _ray_%u, "
                   "hiprtFullRayMask, hiprtTraversalHintDefault, nullptr, "
                   "%s);\n"
                   "hiprtHit _hit_%u = _tr_%u.getNextHit();\n"
                   "if (_hit_%u.hasHit()) {\n"
                   "    r%u_out_0 = true;\n"
                   "    r%u_out_1 = _hit_%u.t;\n"
                   "    r%u_out_2 = _hit_%u.uv.x;\n"
                   "    r%u_out_3 = _hit_%u.uv.y;\n"
                   "    r%u_out_4 = _hit_%u.instanceID;\n"
                   "    r%u_out_5 = _hit_%u.primID;\n"
                   "    r%u_out_6 = 0u;\n"
                   "    r%u_out_7 = _hit_%u.instanceID;\n"
                   "}\n",
                   id, id, id,
                   func_table_arg.c_str(),
                   id, id, id,
                   id, id, id, id, id, id, id, id,
                   id, id, id, id, id, id, id, id, id);
    }

    buffer.put("}\n");
}

static void jitc_amd_render(Variable *v) {
    if (v->coop_vec)
        jitc_raise("jitc_amd_render(): cooperative vectors are not supported "
                   "by the AMD/HIP backend without an OptiX-like HIPRT "
                   "cooperative-vector facility.");

    switch ((VarKind) v->kind) {
        case VarKind::Nop:
            break;

        case VarKind::Undefined:
            jitc_amd_emit_decl(v);
            buffer.fmt("(%s) (%s);\n", jitc_amd_type(v),
                       jitc_amd_zero((VarType) v->type));
            break;

        case VarKind::Literal:
            jitc_amd_emit_decl(v);
            jitc_amd_literal(v);
            buffer.put(";\n");
            break;

        case VarKind::Counter:
            jitc_amd_emit_decl(v);
            buffer.fmt("(%s) r0;\n", jitc_amd_type(v));
            break;

        case VarKind::Add: jitc_amd_render_binary(v, "+"); break;
        case VarKind::Sub: jitc_amd_render_binary(v, "-"); break;
        case VarKind::Mul: jitc_amd_render_binary(v, "*"); break;
        case VarKind::Div:
        case VarKind::DivApprox:
            jitc_amd_render_binary(v, "/"); break;
        case VarKind::Mod: jitc_amd_render_binary(v, "%"); break;
        case VarKind::Eq:  jitc_amd_render_binary(v, "=="); break;
        case VarKind::Neq: jitc_amd_render_binary(v, "!="); break;
        case VarKind::Lt:  jitc_amd_render_binary(v, "<"); break;
        case VarKind::Le:  jitc_amd_render_binary(v, "<="); break;
        case VarKind::Gt:  jitc_amd_render_binary(v, ">"); break;
        case VarKind::Ge:  jitc_amd_render_binary(v, ">="); break;
        case VarKind::Shl: jitc_amd_render_binary(v, "<<"); break;
        case VarKind::Shr: jitc_amd_render_binary(v, ">>"); break;
        case VarKind::MulHi: jitc_amd_render_call(v, "drjit_mul_hi", 2); break;

        case VarKind::MulWide: {
            Variable *a0 = jitc_var(v->dep[0]),
                     *a1 = jitc_var(v->dep[1]);
            jitc_amd_emit_decl(v);
            buffer.fmt("(%s) ", jitc_amd_type(v));
            jitc_amd_var(a0);
            buffer.fmt(" * (%s) ", jitc_amd_type(v));
            jitc_amd_var(a1);
            buffer.put(";\n");
            break;
        }

        case VarKind::Fma:
            if (jitc_is_half(v)) {
                jitc_amd_render_call(v, "drjit_fma", 3);
            } else if (jitc_is_single(v)) {
                jitc_amd_render_call(v, "fmaf", 3);
            } else if (jitc_is_double(v)) {
                jitc_amd_render_call(v, "fma", 3);
            } else {
                Variable *a0 = jitc_var(v->dep[0]),
                         *a1 = jitc_var(v->dep[1]),
                         *a2 = jitc_var(v->dep[2]);
                jitc_amd_emit_decl(v);
                jitc_amd_var(a0);
                buffer.put(" * ");
                jitc_amd_var(a1);
                buffer.put(" + ");
                jitc_amd_var(a2);
                buffer.put(";\n");
            }
            break;

        case VarKind::Min: jitc_amd_render_call(v, "drjit_min", 2); break;
        case VarKind::Max: jitc_amd_render_call(v, "drjit_max", 2); break;

        case VarKind::Ceil:  jitc_amd_render_round(v, "ceil"); break;
        case VarKind::Floor: jitc_amd_render_round(v, "floor"); break;
        case VarKind::Round: jitc_amd_render_round(v, "rint"); break;
        case VarKind::Trunc: jitc_amd_render_round(v, "trunc"); break;

        case VarKind::Select: {
            Variable *cond = jitc_var(v->dep[0]);
            Variable *a    = jitc_var(v->dep[1]);
            Variable *b    = jitc_var(v->dep[2]);
            jitc_amd_emit_decl(v);
            jitc_amd_var(cond);
            buffer.put(" ? ");
            jitc_amd_var(a);
            buffer.put(" : ");
            jitc_amd_var(b);
            buffer.put(";\n");
            break;
        }

        case VarKind::And: {
            Variable *a0 = jitc_var(v->dep[0]),
                     *a1 = jitc_var(v->dep[1]);
            if (a0->type != a1->type) {
                jitc_amd_emit_decl(v);
                jitc_amd_var(a1);
                buffer.put(" ? ");
                jitc_amd_var(a0);
                buffer.fmt(" : (%s) (%s);\n", jitc_amd_type(v),
                           jitc_amd_zero((VarType) v->type));
            } else if (jitc_is_float(v)) {
                jitc_amd_render_bitwise_float_binary(v, "&");
            } else {
                jitc_amd_render_binary(v, "&");
            }
            break;
        }

        case VarKind::Or: {
            Variable *a0 = jitc_var(v->dep[0]),
                     *a1 = jitc_var(v->dep[1]);
            if (a0->type != a1->type) {
                jitc_amd_emit_decl(v);
                buffer.put("drjit_bitcast<");
                buffer.put(jitc_amd_type(v), strlen(jitc_amd_type(v)));
                buffer.put(">(");
                jitc_amd_var(a1);
                buffer.put(" ? ~drjit_bitcast<");
                buffer.fmt("%s", jitc_amd_binary_type(type_size[v->type]));
                buffer.put(">(");
                jitc_amd_var(a0);
                buffer.put(") : drjit_bitcast<");
                buffer.fmt("%s", jitc_amd_binary_type(type_size[v->type]));
                buffer.put(">(");
                jitc_amd_var(a0);
                buffer.put("));\n");
            } else if (jitc_is_float(v)) {
                jitc_amd_render_bitwise_float_binary(v, "|");
            } else {
                jitc_amd_render_binary(v, "|");
            }
            break;
        }

        case VarKind::Xor:
            if (jitc_is_float(v))
                jitc_amd_render_bitwise_float_binary(v, "^");
            else
                jitc_amd_render_binary(v, "^");
            break;

        case VarKind::Neg: jitc_amd_render_unary(v, "-"); break;
        case VarKind::Not: {
            Variable *a = jitc_var(v->dep[0]);
            if ((VarType) v->type == VarType::Bool) {
                jitc_amd_emit_decl(v);
                buffer.put("!");
                jitc_amd_var(a);
                buffer.put(";\n");
            } else if (jitc_is_float(v)) {
                jitc_amd_render_bitwise_float_unary(v, "~");
            } else {
                jitc_amd_emit_decl(v);
                buffer.put("~");
                jitc_amd_var(a);
                buffer.put(";\n");
            }
            break;
        }

        case VarKind::Abs: {
            Variable *a = jitc_var(v->dep[0]);
            jitc_amd_emit_decl(v);
            if (jitc_is_uint(v) || (VarType) v->type == VarType::Bool) {
                jitc_amd_var(a);
            } else {
                buffer.put("drjit_abs(");
                jitc_amd_var(a);
                buffer.put(")");
            }
            buffer.put(";\n");
            break;
        }

        case VarKind::Sqrt:
        case VarKind::SqrtApprox:
            jitc_amd_render_float_call(v, "drjit_sqrt", "sqrtf", "sqrt"); break;
        case VarKind::Sin:
            jitc_amd_render_float_call(v, "drjit_sin", "sinf", "sin"); break;
        case VarKind::Cos:
            jitc_amd_render_float_call(v, "drjit_cos", "cosf", "cos"); break;
        case VarKind::Exp2:
            jitc_amd_render_float_call(v, "drjit_exp2", "exp2f", "exp2"); break;
        case VarKind::Log2:
            jitc_amd_render_float_call(v, "drjit_log2", "log2f", "log2"); break;
        case VarKind::Tanh:
            jitc_amd_render_float_call(v, "drjit_tanh", "tanhf", "tanh"); break;

        case VarKind::Rcp:
        case VarKind::RcpApprox:
            jitc_amd_emit_decl(v);
            buffer.fmt("(%s) 1 / ", jitc_amd_type(v));
            jitc_amd_var(jitc_var(v->dep[0]));
            buffer.put(";\n");
            break;

        case VarKind::RSqrtApprox:
            jitc_amd_render_float_call(v, "drjit_rsqrt", "rsqrtf", "rsqrt");
            break;

        case VarKind::Popc:
        case VarKind::Clz:
        case VarKind::Ctz:
        case VarKind::Brev: {
            Variable *a0 = jitc_var(v->dep[0]);
            bool is_64 = type_size[a0->type] == 8;
            const char *fn =
                v->kind == (uint32_t) VarKind::Popc ? (is_64 ? "__popcll" : "__popc") :
                v->kind == (uint32_t) VarKind::Clz  ? (is_64 ? "__clzll"  : "__clz")  :
                v->kind == (uint32_t) VarKind::Ctz  ? (is_64 ? "__ffsll"  : "__ffs")  :
                                                       (is_64 ? "__brevll" : "__brev");
            const char *cast = is_64 ? "unsigned long long" : "unsigned int";
            jitc_amd_emit_decl(v);
            buffer.fmt("(%s) %s((%s) ", jitc_amd_type(v), fn, cast);
            jitc_amd_var(a0);
            buffer.put(");\n");

            if (v->kind == (uint32_t) VarKind::Ctz)
                buffer.fmt("r%u = r%u ? r%u - 1 : %u;\n",
                           v->reg_index, v->reg_index, v->reg_index,
                           is_64 ? 64u : 32u);
            break;
        }

        case VarKind::Cast: {
            Variable *a = jitc_var(v->dep[0]);
            jitc_amd_emit_decl(v);
            buffer.fmt("(%s) ", jitc_amd_type(v));
            jitc_amd_var(a);
            buffer.put(";\n");
            break;
        }

        case VarKind::Bitcast: {
            Variable *a = jitc_var(v->dep[0]);
            jitc_amd_emit_decl(v);
            if (v->type == a->type) {
                jitc_amd_var(a);
            } else if (type_size[v->type] == type_size[a->type]) {
                buffer.fmt("drjit_bitcast<%s>(", jitc_amd_type(v));
                jitc_amd_var(a);
                buffer.put(")");
            } else {
                buffer.fmt("drjit_bitcast<%s>((%s) ",
                           jitc_amd_type(v),
                           type_size[v->type] == 8 ? "uint64_t" :
                           type_size[v->type] == 4 ? "uint32_t" : "uint16_t");
                jitc_amd_var(a);
                buffer.put(")");
            }
            buffer.put(";\n");
            break;
        }

        case VarKind::BoundsCheck: {
            Variable *index = jitc_var(v->dep[0]);
            Variable *mask  = jitc_var(v->dep[1]);
            Variable *buf   = jitc_var(v->dep[2]);
            jitc_amd_emit_decl(v);
            jitc_amd_var(mask);
            buffer.put(" && (");
            jitc_amd_var(index);
            buffer.fmt(" < (uint32_t) %u);\n", (uint32_t) v->literal);
            buffer.put("if (");
            jitc_amd_var(mask);
            buffer.put(" && !");
            jitc_amd_var(v);
            buffer.put(") *((uint32_t *) (uintptr_t) ");
            jitc_amd_var(buf);
            buffer.put(") = ");
            jitc_amd_var(index);
            buffer.put(";\n");
            break;
        }

        case VarKind::Gather:
            jitc_amd_render_gather(v);
            break;

        case VarKind::PacketGather:
            jitc_amd_render_gather_packet(v);
            break;

        case VarKind::PacketScatter:
            jitc_amd_render_scatter_packet(v);
            break;

        case VarKind::Scatter:
            jitc_amd_render_scatter(v);
            break;

        case VarKind::ScatterInc:
            jitc_amd_render_scatter_inc(v);
            break;

        case VarKind::ScatterCAS:
            jitc_amd_render_scatter_cas(v);
            break;

        case VarKind::ScatterExch:
            jitc_amd_render_scatter_exch(v);
            break;

        case VarKind::TraceRay:
            jitc_amd_render_trace(v);
            break;

        case VarKind::CoopVecUnpack:
            jitc_raise("jitc_amd_render(): cooperative vectors are not "
                       "supported by the AMD/HIP backend.");

        case VarKind::CoopVecAccum:
            jitc_raise("jitc_amd_render(): cooperative vectors are not "
                       "supported by the AMD/HIP backend.");

        case VarKind::CoopVecOuterProductAccum:
            jitc_raise("jitc_amd_render(): cooperative vectors are not "
                       "supported by the AMD/HIP backend.");

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
            buffer.fmt("uint32_t r%u = self;\n", v->reg_index);
            break;

        case VarKind::CallInput:
            break;

        case VarKind::Extract: {
            Variable *src = jitc_var(v->dep[0]);
            jitc_amd_emit_decl(v);
            buffer.fmt("r%u_out_%u;\n", src->reg_index,
                       (uint32_t) v->literal);
            break;
        }

        case VarKind::Array:
            jitc_amd_render_array(v, v->dep[0] ? jitc_var(v->dep[0]) : nullptr);
            break;

        case VarKind::ArrayInit:
            jitc_amd_render_array_init(v, jitc_var(v->dep[0]),
                                       jitc_var(v->dep[1]));
            break;

        case VarKind::ArrayWrite:
            jitc_amd_render_array_write(v, jitc_var(v->dep[0]),
                                        jitc_var(v->dep[1]),
                                        jitc_var(v->dep[2]),
                                        v->dep[3] ? jitc_var(v->dep[3]) : nullptr);
            break;

        case VarKind::ArrayRead:
            jitc_amd_render_array_read(v, jitc_var(v->dep[0]),
                                       jitc_var(v->dep[1]),
                                       v->dep[2] ? jitc_var(v->dep[2]) : nullptr);
            break;

        case VarKind::ArrayPhi:
            v->reg_index = jitc_var(v->dep[0])->reg_index;
            break;

        case VarKind::ArraySelect:
            jitc_amd_render_array_select(v, jitc_var(v->dep[0]),
                                         jitc_var(v->dep[1]),
                                         jitc_var(v->dep[2]));
            break;

        case VarKind::LoopStart: {
            const LoopData *ld = (LoopData *) v->data;
            for (size_t i = 0; i < ld->size; ++i) {
                Variable *inner_in = jitc_var(ld->inner_in[i]),
                         *outer_in = jitc_var(ld->outer_in[i]);

                if (inner_in == outer_in || !inner_in->reg_index ||
                    inner_in->is_array())
                    continue;

                buffer.fmt("%s r%u = ", jitc_amd_type(inner_in),
                           inner_in->reg_index);
                if (outer_in->reg_index)
                    buffer.fmt("r%u", outer_in->reg_index);
                else
                    buffer.fmt("(%s) (%s)", jitc_amd_type(inner_in),
                               jitc_amd_zero((VarType) inner_in->type));
                buffer.put(";\n");
            }
            buffer.put("while (true) {\n");
            break;
        }

        case VarKind::LoopCond: {
            Variable *cond = jitc_var(v->dep[1]);
            buffer.put("if (!");
            jitc_amd_var(cond);
            buffer.put(") break;\n");
            break;
        }

        case VarKind::LoopEnd: {
            Variable *a0 = jitc_var(v->dep[0]);
            const LoopData *ld = (LoopData *) a0->data;
            uint32_t size = (uint32_t) ld->size;

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
                buffer.fmt("%s r%u_tmp = r%u;\n", jitc_amd_type(out),
                           out->reg_index, out->reg_index);
                out->scratch = 2;
            }

            for (uint32_t i = 0; i < size; ++i) {
                Variable *in  = jitc_var(ld->inner_in[i]),
                         *out = jitc_var(ld->inner_out[i]);
                if (in == out || !in->reg_index || !out->reg_index ||
                    in->is_array())
                    continue;
                if (out->scratch == 2)
                    buffer.fmt("r%u = r%u_tmp;\n", in->reg_index,
                               out->reg_index);
                else
                    buffer.fmt("r%u = r%u;\n", in->reg_index,
                               out->reg_index);
            }

            for (uint32_t i = 0; i < size; ++i) {
                jitc_var(ld->inner_in[i])->scratch = 0;
                jitc_var(ld->inner_out[i])->scratch = 0;
            }

            buffer.put("}\n");
            break;
        }

        case VarKind::LoopPhi:
            if (v->is_array()) {
                Variable *a3 = jitc_var(v->dep[3]);
                v->reg_index = a3->reg_index;
            }
            break;

        case VarKind::LoopOutput: {
            Variable *loop_start = jitc_var(v->dep[0]);
            const LoopData *ld = (LoopData *) loop_start->data;
            for (size_t i = 0; i < ld->size; ++i) {
                Variable *outer_out = jitc_var(ld->outer_out[i]);
                if (outer_out == v) {
                    Variable *inner_in = jitc_var(ld->inner_in[i]);
                    if (v->reg_index && inner_in->reg_index)
                        buffer.fmt("%s r%u = r%u;\n", jitc_amd_type(v),
                                   v->reg_index, inner_in->reg_index);
                    break;
                }
            }
            break;
        }

        case VarKind::CondStart: {
            const CondData *cd = (CondData *) v->data;
            Variable *cond = jitc_var(v->dep[0]);
            for (size_t i = 0; i < cd->indices_out.size(); ++i) {
                Variable *vo = jitc_var(cd->indices_out[i]);
                if (vo && vo->reg_index)
                    buffer.fmt("%s r%u;\n", jitc_amd_type(vo),
                               vo->reg_index);
            }
            buffer.fmt("if (r%u) {\n", cond->reg_index);
            break;
        }

        case VarKind::CondMid: {
            Variable *a0 = jitc_var(v->dep[0]);
            const CondData *cd = (CondData *) a0->data;
            for (size_t i = 0; i < cd->indices_out.size(); ++i) {
                Variable *vt = jitc_var(cd->indices_t[i]),
                         *vo = jitc_var(cd->indices_out[i]);
                if (vo && vo->reg_index && vt->reg_index)
                    buffer.fmt("r%u = r%u;\n", vo->reg_index,
                               vt->reg_index);
            }
            buffer.put("} else {\n");
            break;
        }

        case VarKind::CondEnd: {
            Variable *a0 = jitc_var(v->dep[0]);
            const CondData *cd = (CondData *) a0->data;
            for (size_t i = 0; i < cd->indices_out.size(); ++i) {
                Variable *vf = jitc_var(cd->indices_f[i]),
                         *vo = jitc_var(cd->indices_out[i]);
                if (vo && vo->reg_index && vf->reg_index)
                    buffer.fmt("r%u = r%u;\n", vo->reg_index,
                               vf->reg_index);
            }
            buffer.put("}\n");
            break;
        }

        case VarKind::CondOutput:
            break;

        default:
            jitc_raise("jitc_amd_render(): node \"%s\" is not implemented yet.",
                       var_kind_name[v->kind]);
    }
}

static bool jitc_amd_call_has_out(const CallData *call) {
    for (uint32_t i = 0; i < call->n_out; ++i)
        if (call->out_offset[i] != (uint32_t) -1)
            return true;
    return false;
}

static void jitc_amd_put_ret_type(const CallData *call) {
    if (!jitc_amd_call_has_out(call)) {
        buffer.put("void");
        return;
    }

    buffer.put("Ret_");
    for (uint32_t i = 0; i < call->n_out; ++i)
        if (call->out_offset[i] != (uint32_t) -1)
            buffer.put(type_mangle[jitc_var(call->inner_out[i])->type]);
}

static void jitc_amd_emit_ret_struct(const CallData *call) {
    if (!jitc_amd_call_has_out(call))
        return;

    size_t off = buffer.size();
    buffer.put("struct ");
    jitc_amd_put_ret_type(call);
    buffer.put(" {\n");

    uint32_t k = 0;
    for (uint32_t i = 0; i < call->n_out; ++i) {
        if (call->out_offset[i] == (uint32_t) -1)
            continue;
        Variable *v = jitc_var(call->inner_out[i]);
        buffer.fmt("%s r%u;\n", jitc_amd_type(v), k++);
    }

    buffer.put("};\n");
    jitc_register_global(buffer.get() + off);
    buffer.rewind_to(off);
}

static void jitc_amd_callable_signature(const CallData *call,
                                        bool with_names) {
    if (with_names)
        buffer.put("uint32_t index, uint32_t self, const uint8_t *data");
    else
        buffer.put("uint32_t, uint32_t, const uint8_t *");

    if (call->use_nested) {
        if (with_names)
            buffer.put(", uintptr_t base");
        else
            buffer.put(", uintptr_t");
    }

    for (uint32_t i = 0; i < call->n_in; ++i) {
        if (!call->in_active[i])
            continue;

        Variable *vo = jitc_var(call->outer_in[i]);
        if (with_names)
            buffer.fmt(", %s a%u", jitc_amd_type(vo), i);
        else
            buffer.fmt(", %s", jitc_amd_type(vo));
    }
}

void jitc_amd_assemble_func(const CallData *call, uint32_t inst,
                            uint32_t /*in_size*/, uint32_t /*in_align*/,
                            uint32_t /*out_size*/, uint32_t /*out_align*/,
                            uint32_t n_regs) {
    (void) n_regs;

    jitc_amd_emit_ret_struct(call);

    buffer.put("__device__ inline ");
    jitc_amd_put_ret_type(call);
    if (call->n_inst == 1)
        buffer.put(" func_unique_^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^(");
    else
        buffer.put(" func_^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^(");
    jitc_amd_callable_signature(call, /*with_names=*/true);
    buffer.put(") {\n");
    buffer.fmt("// Call: %s\n", call->name.c_str());

    jitc_call_bind_slots(call, inst);

    for (size_t i = 0; i < schedule.size(); ++i) {
        ScheduledVariable &sv = schedule[i];
        Variable *v = jitc_var(sv.index);
        VarType vt = (VarType) v->type;
        VarKind kind = (VarKind) v->kind;

        if (kind == VarKind::Counter) {
            buffer.fmt("%s r%u = (%s) index;\n",
                       jitc_amd_type(v), v->reg_index, jitc_amd_type(v));
        } else if (kind == VarKind::CallInput) {
            uint32_t in_i = 0;
            for (; in_i < call->n_in; ++in_i)
                if (call->inner_in[in_i] == sv.index)
                    break;
            buffer.fmt("%s r%u = a%u;\n",
                       jitc_amd_type(v), v->reg_index, in_i);
        } else if (kind == VarKind::CallSelf) {
            buffer.fmt("uint32_t r%u = self;\n", v->reg_index);
        } else if (v->is_evaluated() ||
                   (vt == VarType::Pointer && kind == VarKind::Literal)) {
            uint32_t offset = jitc_call_slot_rel_offset(call, inst, v, sv.index);

            if (vt == VarType::Bool) {
                buffer.fmt("bool r%u = *((const uint8_t *) (data + %u)) != 0;\n",
                           v->reg_index, offset);
            } else {
                buffer.fmt("%s r%u = *((const %s *) (data + %u));\n",
                           jitc_amd_type(v), v->reg_index,
                           jitc_amd_type(v), offset);
            }
        } else if (v->is_literal()) {
            jitc_amd_render(v);
        } else {
            jitc_amd_render(v);
        }
    }

    if (jitc_amd_call_has_out(call)) {
        buffer.put("    ");
        jitc_amd_put_ret_type(call);
        buffer.put(" ret;\n");

        uint32_t k = 0;
        for (uint32_t i = 0; i < call->n_out; ++i) {
            if (call->out_offset[i] == (uint32_t) -1)
                continue;
            Variable *v = jitc_var(call->inner_out[inst * call->n_out + i]);
            buffer.fmt("ret.r%u = r%u;\n", k++, v->reg_index);
        }

        buffer.put("return ret;\n");
    }

    buffer.put("}\n");
}

void jitc_var_call_getter_assemble_amd(Variable *v, const Variable *index,
                                       const Variable *mask) {
    GetterData *gd = (GetterData *) v->data;
    uint32_t header_offset = gd->header_offset;

    char base[32];
    if (callable_depth == 0)
        snprintf(base, sizeof(base), "r%u", call_buffer.base_reg);
    else
        snprintf(base, sizeof(base), "base");

    bool is_unmasked = mask->is_literal() && mask->literal == 1;
    jitc_amd_emit_decl(v);

    if (!is_unmasked) {
        jitc_amd_var(mask);
        buffer.put(" ? ");
    }

    if ((VarType) v->type == VarType::Bool) {
        buffer.fmt("(((const uint8_t *) ((uintptr_t) %s + %u))[",
                   base, header_offset);
        jitc_amd_var(index);
        buffer.put("] != 0)");
    } else {
        buffer.fmt("((const %s *) ((uintptr_t) %s + %u))[",
                   jitc_amd_type(v), base, header_offset);
        jitc_amd_var(index);
        buffer.put("]");
    }

    if (!is_unmasked) {
        buffer.fmt(" : (%s) (%s)", jitc_amd_type(v),
                   jitc_amd_zero((VarType) v->type));
    }

    buffer.put(";\n");
}

void jitc_var_call_assemble_amd(CallData *call, uint32_t call_reg,
                                uint32_t self_reg, uint32_t mask_reg,
                                uint32_t /*in_size*/, uint32_t /*in_align*/,
                                uint32_t /*out_size*/, uint32_t /*out_align*/) {
    Variable *mask = jitc_var(jitc_var(call->id)->dep[1]);
    bool is_masked = !mask->is_literal() || mask->literal != 1;

    buffer.fmt("\n// VCall: %s\n", call->name.c_str());

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

    for (auto [v, field] : out_regs) {
        (void) field;
        buffer.fmt("%s r%u;\n", jitc_amd_type(v), v->reg_index);
    }

    if (is_masked)
        buffer.fmt("if (r%u) {\n", mask_reg);

    char base[32];
    if (callable_depth == 0)
        snprintf(base, sizeof(base), "r%u", call_buffer.base_reg);
    else
        snprintf(base, sizeof(base), "base");

    bool has_slots = !call->slots.empty();

    if (has_slots) {
        buffer.fmt("uint64_t _oe_%u = ((const uint64_t *) (uintptr_t) %s)[%u + r%u];\n"
                   "const uint8_t *_cd_%u = (const uint8_t *) ((uintptr_t) %s + "
                   "(uint32_t) (_oe_%u >> 32));\n",
                   call_reg, base,
                   call->offset_base / (uint32_t) sizeof(uint64_t), self_reg,
                   call_reg, base, call_reg);
    }

    bool has_out = jitc_amd_call_has_out(call);

    if (has_out) {
        jitc_amd_put_ret_type(call);
        if (call->n_inst == 1)
            buffer.fmt(" ret_%u = ", call_reg);
        else
            buffer.fmt(" ret_%u = {};\n", call_reg);
    }

    auto put_call = [&]() {
        buffer.put("(");

        if (callable_depth > 0)
            buffer.put("index");
        else
            buffer.put("r0");

        buffer.fmt(", r%u, ", self_reg);
        if (has_slots)
            buffer.fmt("_cd_%u", call_reg);
        else
            buffer.put("(const uint8_t *) nullptr");

        if (call->use_nested)
            buffer.fmt(", %s", base);

        for (uint32_t i = 0; i < call->n_in; ++i)
            if (call->in_active[i])
                buffer.fmt(", r%u", jitc_var(call->outer_in[i])->reg_index);

        buffer.put(")");
    };

    auto put_func = [&](XXH128_hash_t hash, bool unique) {
        if (unique)
            buffer.put("func_unique_");
        else
            buffer.put("func_");
        buffer.put_q64_unchecked(hash.high64);
        buffer.put_q64_unchecked(hash.low64);
        put_call();
    };

    if (call->n_inst == 1) {
        put_func(call->inst_hash[0], true);
        buffer.put(";\n");
    } else {
        for (uint32_t i = 0; i < call->n_inst; ++i) {
            if (i == 0)
                buffer.fmt("if (r%u == %u) {\n", self_reg, call->inst_id[i]);
            else
                buffer.fmt("else if (r%u == %u) {\n", self_reg, call->inst_id[i]);

            if (has_out)
                buffer.fmt("ret_%u = ", call_reg);

            put_func(call->inst_hash[i], false);
            buffer.put(";\n}\n");
        }
    }

    for (auto [v, field] : out_regs)
        buffer.fmt("r%u = ret_%u.r%u;\n", v->reg_index, call_reg, field);

    if (is_masked) {
        if (!out_regs.empty()) {
            buffer.put("} else {\n");
            for (auto [v, field] : out_regs) {
                (void) field;
                buffer.fmt("r%u = (%s) (%s);\n",
                           v->reg_index, jitc_amd_type(v),
                           jitc_amd_zero((VarType) v->type));
            }
        }
        buffer.put("}\n");
    }

    buffer.put("\n");
}

void jitc_amd_assemble(ThreadState *ts, ScheduledGroup group,
                       uint32_t, uint32_t) {
    bool uses_hiprt = false;
    AMDScene *hiprt_scene_config = nullptr;
    for (uint32_t gi = group.start; gi != group.end; ++gi) {
        Variable *v = jitc_var(schedule[gi].index);
        if ((VarKind) v->kind == VarKind::TraceRay) {
            uses_hiprt = true;
            AMDScene *scene = (AMDScene *) (uintptr_t) jitc_var(v->dep[1])->literal;
            if (scene && scene->func_table) {
                if (!hiprt_scene_config) {
                    hiprt_scene_config = scene;
                } else if (hiprt_scene_config->num_geom_types != scene->num_geom_types ||
                           hiprt_scene_config->num_ray_types != scene->num_ray_types ||
                           hiprt_scene_config->intersect_fns != scene->intersect_fns ||
                           hiprt_scene_config->filter_fns != scene->filter_fns ||
                           hiprt_scene_config->device_source != scene->device_source) {
                    jitc_raise("jitc_amd_assemble(): a single HIPRT kernel "
                               "cannot mix scenes with different custom "
                               "function tables.");
                }
            }
        }
    }

#if defined(DRJIT_ENABLE_HIPRT)
    amd_kernel_func_name_sets.clear();
    amd_kernel_device_source.clear();
    amd_kernel_num_geom_types = 1;
    amd_kernel_num_ray_types = 1;
    if (hiprt_scene_config && hiprt_scene_config->num_geom_types &&
        hiprt_scene_config->num_ray_types) {
        amd_kernel_num_geom_types = hiprt_scene_config->num_geom_types;
        amd_kernel_num_ray_types = hiprt_scene_config->num_ray_types;
        uint32_t count = amd_kernel_num_geom_types * amd_kernel_num_ray_types;
        amd_kernel_func_name_sets.resize(count);
        for (uint32_t i = 0; i < count; ++i) {
            const std::string &isect = hiprt_scene_config->intersect_fns[i];
            const std::string &filter = hiprt_scene_config->filter_fns[i];
            amd_kernel_func_name_sets[i].intersectFuncName =
                isect.empty() ? nullptr : isect.c_str();
            amd_kernel_func_name_sets[i].filterFuncName =
                filter.empty() ? nullptr : filter.c_str();
        }
        amd_kernel_device_source = hiprt_scene_config->device_source;
    }
#endif

    buffer.put(
        "#include <stdint.h>\n"
        "#include <hip/hip_runtime.h>\n"
        "#include <hip/hip_fp16.h>\n"
    );

    if (uses_hiprt) {
        buffer.put(
            "#include <hiprt/hiprt_device.h>\n"
            "#include <hiprt/hiprt_vec.h>\n"
            "\n");
#if defined(DRJIT_ENABLE_HIPRT)
        if (hiprt_scene_config) {
            buffer.fmt("// HIPRT scene properties: geom_types=%u ray_types=%u "
                       "geom_mask=%u\n",
                       hiprt_scene_config->num_geom_types,
                       hiprt_scene_config->num_ray_types,
                       hiprt_scene_config->geometry_types_mask);
            for (size_t i = 0; i < hiprt_scene_config->intersect_fns.size(); ++i) {
                buffer.fmt("// HIPRT fn[%u]: intersect=%s filter=%s\n",
                           (uint32_t) i,
                           hiprt_scene_config->intersect_fns[i].c_str(),
                           hiprt_scene_config->filter_fns[i].c_str());
            }
            if (!amd_kernel_device_source.empty()) {
                buffer.put(amd_kernel_device_source.c_str(),
                           amd_kernel_device_source.size());
                buffer.put("\n");
            }
        }
#endif
    } else {
        buffer.put("\n");
    }

    buffer.put(
        "template <typename To, typename From>\n"
        "__device__ inline To drjit_bitcast(From value) {\n"
        "    union { From from; To to; } u;\n"
        "    u.from = value;\n"
        "    return u.to;\n"
        "}\n"
        "\n"
        "__device__ inline float drjit_bits_to_float(uint32_t value) {\n"
        "    return drjit_bitcast<float>(value);\n"
        "}\n"
        "\n"
        "__device__ inline double drjit_bits_to_double(uint64_t value) {\n"
        "    return drjit_bitcast<double>(value);\n"
        "}\n"
        "\n"
        "template <typename T> __device__ inline T drjit_min(T a, T b) { return a < b ? a : b; }\n"
        "template <typename T> __device__ inline T drjit_max(T a, T b) { return a > b ? a : b; }\n"
        "template <typename T> __device__ inline T drjit_abs(T a) { return a < (T) 0 ? -a : a; }\n"
        "__device__ inline __half drjit_fma(__half a, __half b, __half c) { return (__half) fmaf((float) a, (float) b, (float) c); }\n"
        "__device__ inline __half drjit_sqrt(__half a) { return (__half) sqrtf((float) a); }\n"
        "__device__ inline __half drjit_sin(__half a) { return (__half) sinf((float) a); }\n"
        "__device__ inline __half drjit_cos(__half a) { return (__half) cosf((float) a); }\n"
        "__device__ inline __half drjit_exp2(__half a) { return (__half) exp2f((float) a); }\n"
        "__device__ inline __half drjit_log2(__half a) { return (__half) log2f((float) a); }\n"
        "__device__ inline __half drjit_tanh(__half a) { return (__half) tanhf((float) a); }\n"
        "__device__ inline __half drjit_rsqrt(__half a) { return (__half) (1.0f / sqrtf((float) a)); }\n"
        "__device__ inline __half drjit_ceil(__half a) { return (__half) ceilf((float) a); }\n"
        "__device__ inline __half drjit_floor(__half a) { return (__half) floorf((float) a); }\n"
        "__device__ inline __half drjit_rint(__half a) { return (__half) rintf((float) a); }\n"
        "__device__ inline __half drjit_trunc(__half a) { return (__half) truncf((float) a); }\n"
        "\n"
        "__device__ inline uint8_t drjit_mul_hi(uint8_t a, uint8_t b) { return (uint8_t) (((uint16_t) a * (uint16_t) b) >> 8); }\n"
        "__device__ inline int8_t drjit_mul_hi(int8_t a, int8_t b) { return (int8_t) (((int16_t) a * (int16_t) b) >> 8); }\n"
        "__device__ inline uint16_t drjit_mul_hi(uint16_t a, uint16_t b) { return (uint16_t) (((uint32_t) a * (uint32_t) b) >> 16); }\n"
        "__device__ inline int16_t drjit_mul_hi(int16_t a, int16_t b) { return (int16_t) (((int32_t) a * (int32_t) b) >> 16); }\n"
        "__device__ inline uint32_t drjit_mul_hi(uint32_t a, uint32_t b) { return (uint32_t) (((uint64_t) a * (uint64_t) b) >> 32); }\n"
        "__device__ inline int32_t drjit_mul_hi(int32_t a, int32_t b) { return (int32_t) (((int64_t) a * (int64_t) b) >> 32); }\n"
        "__device__ inline uint64_t drjit_mul_hi(uint64_t a, uint64_t b) { return (uint64_t) (((unsigned __int128) a * (unsigned __int128) b) >> 64); }\n"
        "__device__ inline int64_t drjit_mul_hi(int64_t a, int64_t b) { return (int64_t) (((__int128) a * (__int128) b) >> 64); }\n"
        "\n");

    size_t kernel_start = buffer.size();

    buffer.put(
        "extern \"C\" __global__ void drjit_^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^(void **params) {\n"
        "uint32_t r0 = (uint32_t) (blockIdx.x * blockDim.x + threadIdx.x);\n"
        "uint32_t size = (uint32_t) (uintptr_t) params[0];\n"
        "if (r0 >= size) return;\n");
    if (call_buffer.base_v)
        buffer.fmt("uintptr_t r%u = (uintptr_t) params[%u];\n",
                   call_buffer.base_reg, call_buffer.base_param_index);
    buffer.put("\nbody:\n");

    for (uint32_t gi = group.start; gi != group.end; ++gi) {
        uint32_t index = schedule[gi].index;
        Variable *v = jitc_var(index);
        ParamType ptype = (ParamType) v->param_type;

        if (ptype == ParamType::Input) {
            uint32_t slot = jitc_amd_param_slot(v);

            if (v->is_literal()) {
                jitc_amd_emit_decl(v);
                if ((VarType) v->type == VarType::Pointer)
                    buffer.fmt("(uintptr_t) params[%u]", slot);
                else
                    jitc_amd_literal(v);
                buffer.put(";\n");
                continue;
            }

            if (v->is_array()) {
                jitc_amd_render_array_memcpy_in(v, slot);
                continue;
            }

            jitc_amd_emit_decl(v);
            if (v->size > 1) {
                buffer.fmt("((const %s *) params[%u])[r0]",
                           jitc_amd_type(v), slot);
            } else {
                buffer.fmt("*((const %s *) params[%u])",
                           jitc_amd_type(v), slot);
            }
            buffer.put(";\n");
            continue;
        }

        jitc_amd_render(v);

        if (ptype == ParamType::Output) {
            if (v->is_array()) {
                jitc_amd_render_array_memcpy_out(v, jitc_amd_param_slot(v));
                continue;
            }

            buffer.fmt("((%s *) params[%u])[r0] = ",
                       jitc_amd_type(v), jitc_amd_param_slot(v));
            jitc_amd_var(v);
            buffer.put(";\n");
        }
    }

    buffer.put("}\n");

    if (!globals_map.empty()) {
        size_t suffix_start = buffer.size();

        auto emit_globals = [](GlobalType type) {
            for (auto &it : globals_map) {
                if (it.first.type != type)
                    continue;
                buffer.put('\n');
                buffer.put(globals.get() + it.second.start, it.second.length);
                buffer.put('\n');
            }
        };

        auto emit_callable_decls = [](GlobalType type) {
            for (auto &it : globals_map) {
                if (it.first.type != type)
                    continue;

                const char *sig = globals.get() + it.second.start;
                const char *brace =
                    (const char *) memchr(sig, '{', it.second.length);
                if (!brace)
                    continue;

                size_t sig_len = (size_t) (brace - sig);
                while (sig_len > 0 &&
                       (sig[sig_len - 1] == ' ' ||
                        sig[sig_len - 1] == '\n' ||
                        sig[sig_len - 1] == '\r' ||
                        sig[sig_len - 1] == '\t'))
                    --sig_len;

                buffer.put(sig, sig_len);
                buffer.put(";\n");
            }
        };

        emit_globals(GlobalType::Type);
        emit_globals(GlobalType::Global);
        emit_callable_decls(GlobalType::IndirectCallable);
        emit_callable_decls(GlobalType::Callable);
        emit_globals(GlobalType::IndirectCallable);
        emit_globals(GlobalType::Callable);

        if (suffix_start != buffer.size())
            buffer.move_suffix(suffix_start, kernel_start);
    }

    jitc_call_upload(ts);
}

bool jitc_amd_compile(ThreadState *ts, const char *source,
                      size_t source_size, const char *kernel_name,
                      Kernel &kernel) {
    if (!ts->amd_arch || !*ts->amd_arch)
        jitc_raise("jitc_amd_compile(): the active HIP device does not report "
                   "an AMDGPU architecture.");

    auto normalize_path = [](std::string value) {
        for (char &c : value)
            if (c == '\\')
                c = '/';
        return value;
    };

    std::vector<std::string> option_storage;
    option_storage.push_back(std::string("--gpu-architecture=") +
                             ts->amd_arch);

    auto add_include = [&](const char *path) {
        if (path && *path)
            option_storage.push_back(std::string("-I") + normalize_path(path));
    };

    if (const char *rocm_home = getenv("ROCM_HOME"))
        add_include((std::string(rocm_home) + "/include").c_str());
    else if (const char *hip_path = getenv("HIP_PATH"))
        add_include((std::string(hip_path) + "/include").c_str());

#if defined(DRJIT_HIPRT_ROOT)
    add_include(DRJIT_HIPRT_ROOT);
#endif
#if defined(DRJIT_CORE_RESOURCE_DIR)
    add_include(DRJIT_CORE_RESOURCE_DIR);
#endif
    add_include(getenv("HIPRT_ROOT"));

    std::vector<const char *> options {
        "-std=c++17",
        "-O3"
    };
    for (const std::string &option : option_storage)
        options.push_back(option.c_str());

    hip_check(hipCtxSetCurrent(ts->amd_context));

    bool uses_hiprt = strstr(source, "hiprtSceneTraversal") != nullptr;
    size_t code_size = source_size;

    if (uses_hiprt) {
#if defined(DRJIT_ENABLE_HIPRT)
#if defined(DRJIT_HIPRT_ROOT)
        if (!getenv("HIPRT_PATH")) {
#  if defined(_WIN32)
            _putenv_s("HIPRT_PATH", DRJIT_HIPRT_ROOT);
#  else
            setenv("HIPRT_PATH", DRJIT_HIPRT_ROOT, 0);
#  endif
        }
#endif

        AMDDevice &device = state.amd_devices[(size_t) ts->device];

        if (!device.hiprt_context) {
            hiprtContextCreationInput input {};
            input.ctxt = (hiprtApiCtx) ts->amd_context;
            input.device = (hiprtApiDevice) ts->amd_raw_device;
            input.deviceType = hiprtDeviceAMD;

            hiprtContext context = nullptr;
            hiprtError err =
                hiprtCreateContext(HIPRT_API_VERSION, input, context);
            if (err != hiprtSuccess)
                jitc_raise("jitc_amd_compile(): hiprtCreateContext() failed "
                           "with error %u.", (uint32_t) err);
            hiprtSetLogLevel(context, hiprtLogLevelError);
            device.hiprt_context = context;
        }

        const char *func_name = kernel_name;
        std::string hiprt_module_name = std::string(kernel_name) + ".hip";
        hiprtApiFunction function = nullptr;
        hiprtApiModule module = nullptr;
        hiprtError err = hiprtBuildTraceKernels(
            (hiprtContext) device.hiprt_context,
            1, &func_name,
            source, hiprt_module_name.c_str(),
            0, nullptr, nullptr,
            (uint32_t) options.size(), options.data(),
            amd_kernel_num_geom_types, amd_kernel_num_ray_types,
            amd_kernel_func_name_sets.empty()
                ? nullptr
                : amd_kernel_func_name_sets.data(),
            &function, &module,
            true);

        if (err != hiprtSuccess)
            jitc_raise("jitc_amd_compile(): hiprtBuildTraceKernels() failed "
                       "with error %u.", (uint32_t) err);

        kernel.amd.mod = (hipModule_t) module;
        kernel.amd.func = (hipFunction_t) function;
        kernel.amd.hiprt_owned = true;
#else
        jitc_raise("jitc_amd_compile(): this Dr.Jit build was configured "
                   "without HIPRT support.");
#endif
    } else {
        hiprtcProgram program = nullptr;
        hiprtc_check(hiprtcCreateProgram(&program, source, "drjit_amd.hip", 0,
                                         nullptr, nullptr));

        hiprtcResult result =
            hiprtcCompileProgram(program, (int) options.size(), options.data());

        size_t log_size = 0;
        hiprtc_check(hiprtcGetProgramLogSize(program, &log_size));
        std::vector<char> compile_log;
        if (log_size > 1) {
            compile_log.resize(log_size);
            hiprtc_check(hiprtcGetProgramLog(program, compile_log.data()));
        }

        if (result != HIPRTC_SUCCESS) {
            std::string log = compile_log.empty() ? "" : compile_log.data();
            hiprtcDestroyProgram(&program);
            jitc_raise("jitc_amd_compile(): HIPRTC compilation failed: %s\n%s",
                       hiprtcGetErrorString(result), log.c_str());
        } else if (!compile_log.empty()) {
            jitc_log(Debug, "jitc_amd_compile(): HIPRTC log:\n%s",
                     compile_log.data());
        }

        hiprtc_check(hiprtcGetCodeSize(program, &code_size));
        std::vector<char> code(code_size);
        hiprtc_check(hiprtcGetCode(program, code.data()));
        hiprtc_check(hiprtcDestroyProgram(&program));

        hip_check(hipModuleLoadData(&kernel.amd.mod, code.data()));
        hip_check(hipModuleGetFunction(&kernel.amd.func, kernel.amd.mod,
                                       kernel_name));
        kernel.amd.hiprt_owned = false;
    }

    int unused = 0, block_size = 0;
    hipError_t occ = hipModuleOccupancyMaxPotentialBlockSize(
        &unused, &block_size, kernel.amd.func, 0, 0);
    if (occ == hipSuccess && block_size > 0)
        kernel.amd.block_size = (uint32_t) block_size;
    else
        kernel.amd.block_size = ts->amd_max_threads ? ts->amd_max_threads : 256;

    kernel.data = nullptr;
    kernel.size = (uint32_t) code_size;
    (void) source_size;
    return false;
}

#endif
