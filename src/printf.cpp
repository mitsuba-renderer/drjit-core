#include "internal.h"
#include "var.h"
#include "log.h"
#include "eval.h"

/// Forward declaration
static void jitc_var_printf_assemble(const Variable *v, const Extra &extra);

void jitc_var_printf(JitBackend backend, uint32_t mask, const char *fmt,
                     uint32_t narg, const uint32_t *arg) {
    if (backend != JitBackend::CUDA)
        jitc_raise("jit_var_printf(): only supported for the CUDA backend at the moment.");

    if (unlikely(mask && (VarType) jitc_var(mask)->type != VarType::Bool))
        jitc_raise("jit_var_printf(): mask argument must be a boolean variable!");

    uint32_t size = 1;
    for (uint32_t i = 0; i < narg; ++i) {
        const Variable *v = jitc_var(arg[i]);
        if (unlikely(size != v->size && v->size != 1 && size != 1))
            jitc_raise("jit_var_printf(): arrays have incompatible size!");
        size = std::max(size, v->size);
    }

    Ref printf_var =
        steal(jitc_var_new_stmt(backend, VarType::Void, "", 1, 0, nullptr));

    Variable *v = jitc_var(printf_var);
    v->extra = 1;
    v->side_effect = 1;
    v->size = size;
    v->dep[0] = mask;
    jitc_var_inc_ref_int(mask);
    size_t dep_size = narg * sizeof(uint32_t);
    Extra &e = state.extra[printf_var];
    e.n_dep = narg;
    e.dep = (uint32_t *) malloc(dep_size);
    memcpy(e.dep, arg, dep_size);
    for (uint32_t i = 0; i < narg; ++i)
        jitc_var_inc_ref_int(arg[i]);
    e.assemble = jitc_var_printf_assemble;
    e.callback_data = strdup(fmt);
    e.callback = [](uint32_t, int free_var, void *ptr) {
        if (free_var && ptr)
            free(ptr);
    };
    e.callback_internal = true;
    thread_state(backend)->side_effects.push_back(printf_var.release());
}

static void jitc_var_printf_assemble(const Variable *v, const Extra &extra) {
    buffer.put("    {\n"
               "        .global .align 1 .b8 fmt[] = { ");
    const char *fmt = (const char *) extra.callback_data;
    for (uint32_t i = 0; ; ++i) {
        buffer.put_uint32((uint32_t) fmt[i]);
        if (fmt[i] == '\0')
            break;
        buffer.put(", ");
    }
    buffer.put(" };\n");

    uint32_t offset = 0, align = 0;
    for (uint32_t i = 0; i < extra.n_dep; ++i) {
        Variable *v = jitc_var(extra.dep[i]);
        uint32_t vti = v->type;
        uint32_t tsize = type_size[vti];
        if ((VarType) vti == VarType::Float32)
            tsize = 8;

        offset = (offset + tsize - 1) / tsize * tsize;
        offset += tsize;
        align = std::max(align, tsize);
    }
    if (align == 0)
        align = 1;
    if (offset == 0)
        offset = 1;

    buffer.fmt("        .local .align %u .b8 buf[%u];\n", align, offset);

    offset = 0;
    for (uint32_t i = 0; i < extra.n_dep; ++i) {
        Variable *v = jitc_var(extra.dep[i]);
        uint32_t vti = v->type;
        uint32_t tsize = type_size[vti];
        if ((VarType) vti == VarType::Float32)
            tsize = 8;

        offset = (offset + tsize - 1) / tsize * tsize;

        if ((VarType) vti == VarType::Float32) {
            buffer.fmt("        cvt.f64.f32 %%d3, %%f%u;\n"
                       "        st.local.f64 [buf+%u], %%d3;\n",
                       v->reg_index, offset);
        } else {
            buffer.fmt("        st.local.%s [buf+%u], %s%u;\n",
                       type_name_ptx[vti], offset, type_prefix[vti],
                       v->reg_index);
        }

        offset += tsize;
    }
    buffer.put("\n"
               "        .reg.b64 %fmt_generic, %buf_generic;\n"
               "        cvta.global.u64 %fmt_generic, fmt;\n"
               "        cvta.local.u64 %buf_generic, buf;\n"
               "        {\n"
               "            .param .b64 fmt_p;\n"
               "            .param .b64 buf_p;\n"
               "            .param .b32 rv_p;\n"
               "            st.param.b64 [fmt_p], %fmt_generic;\n"
               "            st.param.b64 [buf_p], %buf_generic;\n"
               "            ");
    if (v->dep[0]) {
        Variable *v2 = jitc_var(v->dep[0]);
        if (!v2->literal || v2->value != 1)
            buffer.fmt("@%%p%u ", v2->reg_index);
    }
    buffer.put("call (rv_p), vprintf, (fmt_p, buf_p);\n"
               "        }\n"
               "    }\n");

    jitc_register_global(".extern .func (.param .b32 rv) vprintf "
                         "(.param .b64 fmt, .param .b64 buf);\n");
}
