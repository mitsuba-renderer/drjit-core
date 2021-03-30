#include "eval.h"
#include "var.h"
#include "log.h"
#include "op.h"
#include "llvm_api.h"

static void jitc_embree_trace_assemble(const Variable *v, const Extra &extra);

void jitc_embree_trace(uint32_t func, uint32_t context, uint32_t scene,
                       int occluded, const uint32_t *in, uint32_t *out) {
    bool double_precision = ((VarType) jitc_var(in[1])->type) == VarType::Float64;
    VarType float_type = double_precision ? VarType::Float64 : VarType::Float32;
    VarType types[]{ VarType::Int32, float_type,      float_type,
                     float_type,     float_type,      float_type,
                     float_type,     float_type,      float_type,
                     float_type,     VarType::UInt32, VarType::UInt32,
                     VarType::UInt32 };

    bool placeholder = false, dirty = false;
    uint32_t size = 0;
    for (uint32_t i = 0; i < 13; ++i) {
        const Variable *v = jitc_var(in[i]);
        if ((VarType) v->type != types[i])
            jitc_raise("jit_embree_trace(): type mismatch for arg. %u (got %s, "
                       "expected %s)",
                       i, type_name[v->type], type_name[(int) types[i]]);
        size = std::max(size, v->size);
        placeholder |= v->placeholder;
        dirty |= v->dirty;
    }

    for (uint32_t i = 0; i < 13; ++i) {
        const Variable *v = jitc_var(in[i]);
        if (v->size != 1 && v->size != size)
            jitc_raise("jit_embree_trace(): arithmetic involving arrays of "
                       "incompatible size!");
    }

    if ((VarType) jitc_var(func)->type != VarType::Pointer ||
        (VarType) jitc_var(context)->type != VarType::Pointer ||
        (VarType) jitc_var(scene)->type != VarType::Pointer)
        jitc_raise("jit_embree_trace(): 'func', 'context', and 'scene' must be pointer variables!");

    if (dirty)
        jitc_eval(thread_state(JitBackend::LLVM));

    jitc_log(InfoSym, "jit_embree_trace(): tracing %u %sray%s%s%s", size,
             occluded ? "shadow " : "", size != 1 ? "s" : "",
             placeholder ? " (part of a recorded computation)" : "",
             double_precision ? " (double precision)" : "");

    Ref op = steal(jitc_var_new_stmt_n(JitBackend::LLVM, VarType::Void,
                               occluded ? "// Embree rtcOccluded()"
                                        : "// Embree rtcIntersect()",
                               1, func, context, scene));
    Variable *v_op = jitc_var(op);
    v_op->size = size;
    v_op->extra = 1;

    Extra &e = state.extra[op];
    e.dep = (uint32_t *) malloc_check(sizeof(uint32_t) * 13);
    for (int i = 0; i < 13; ++i) {
        jitc_var_inc_ref_int(in[i]);
        e.dep[i] = in[i];
    }
    e.n_dep = 13;
    e.assemble = jitc_embree_trace_assemble;

    char tmp[128];
    for (int i = 0; i < (occluded ? 1 : 6); ++i) {
        snprintf(tmp, sizeof(tmp),
                 "$r0 = bitcast <$w x $t0> $r1_out_%u to <$w x $t0>", i);
        VarType vt = (i < 3) ? float_type : VarType::UInt32;
        out[i] = jitc_var_new_stmt_n(JitBackend::LLVM, vt, tmp, 0, op);
    }
}

static void jitc_embree_trace_assemble(const Variable *v, const Extra &extra) {
    const uint32_t width = jitc_llvm_vector_width;
    const uint32_t id = v->reg_index;
    bool occluded = strstr(v->stmt, "rtcOccluded") != nullptr;
    bool double_precision = ((VarType) jitc_var(extra.dep[1])->type) == VarType::Float64;
    VarType float_type = double_precision ? VarType::Float64 : VarType::Float32;
    uint32_t float_size = double_precision ? 8 : 4;

    if (occluded)
        alloca_size =
            std::max(alloca_size, (int32_t)((9 * float_size + 4 * 4) * width));
    else
        alloca_size = std::max(alloca_size, (int32_t)((14 * float_size + 7 * 4) * width));
    alloca_align = std::max(alloca_align, (int32_t) (float_size * width));

    /* Offsets:
        0  uint32_t valid
        1  float org_x
        2  float org_y
        3  float org_z
        4  float tnear
        5  float dir_x
        6  float dir_y
        7  float dir_z
        8  float time
        9  float tfar
        10 uint32_t mask
        11 uint32_t id
        12 uint32_t flags
        13 float Ng_x
        14 float Ng_y
        15 float Ng_z
        16 float u
        17 float v
        18 uint32_t primID
        19 uint32_t geomID
        20 uint32_t instID[] */
    buffer.fmt("\n    ; -------- %s -------\n",
               occluded ? "rtcOccluded" : "rtcIntersect");

    uint32_t offset = 0;
    for (int i = 0; i < 13; ++i) {
        Variable *v2 = jitc_var(extra.dep[i]);
        const char *tname = type_name_llvm[v2->type];
        uint32_t tsize = type_size[v2->type];
        buffer.fmt(
            "    %%u%u_in_%u_0 = getelementptr inbounds i8, i8* %%buffer, i32 %u\n"
            "    %%u%u_in_%u_1 = bitcast i8* %%u%u_in_%u_0 to <%u x %s> *\n"
            "    store <%u x %s> %s%u, <%u x %s>* %%u%u_in_%u_1, align %u\n",
            id, i, offset,
            id, i, id, i, width, tname,
            width, tname, type_prefix[v2->type], v2->reg_index, width, tname, id, i,
            float_size * width);
        offset += tsize * width;
    }

    if (!occluded) {
        buffer.fmt(
            "    %%u%u_in_geomid_0 = getelementptr inbounds i8, i8* %%buffer, i32 %u\n"
            "    %%u%u_in_geomid_1 = bitcast i8* %%u%u_in_geomid_0 to <%u x i32> *\n"
            "    store <%u x i32> %s, <%u x i32>* %%u%u_in_geomid_1, align %u\n",
            id, (14 * float_size + 5 * 4) * width, id, id, width, width,
            jitc_llvm_ones_str[(int) VarType::Int32], width, id, float_size * width);
    }

    const Variable *func    = jitc_var(v->dep[0]),
                   *context = jitc_var(v->dep[1]),
                   *scene   = jitc_var(v->dep[2]);

    // jitc_register_global("declare void @llvm.debugtrap()\n\n");
    // buffer.put("    call void @llvm.debugtrap()\n");
    //
    buffer.fmt(
        "    %%u%u_func = bitcast i8* %%rd%u to void (i8*, i8*, i8*, i8*)*\n"
        "    call void %%u%u_func(i8* %%u%u_in_0_0, i8* %%rd%u, i8* %%rd%u, i8* %%u%u_in_1_0)\n",
        id, func->reg_index,
        id, id, scene->reg_index, context->reg_index, id
    );

    offset = (8 * float_size + 4) * width;
    for (int i = 0; i < (occluded ? 1 : 6); ++i) {
        VarType vt = (i < 3) ? float_type : VarType::UInt32;
        const char *tname = type_name_llvm[(int) vt];
        uint32_t tsize = type_size[(int) vt];
        buffer.fmt(
            "    %%u%u_out_%u_0 = getelementptr inbounds i8, i8* %%buffer, i32 %u\n"
            "    %%u%u_out_%u_1 = bitcast i8* %%u%u_out_%u_0 to <%u x %s> *\n"
            "    %%u%u_out_%u = load <%u x %s>, <%u x %s>* %%u%u_out_%u_1, align %u\n",
            id, i, offset,
            id, i, id, i, width, tname,
            id, i, width, tname, width, tname, id, i, float_size * width);
        if (i == 0)
            offset += (4 * float_size + 3 * 4) * width;
        else
            offset += tsize * width;
    }
    buffer.fmt("    ; -------------------\n\n");
}
