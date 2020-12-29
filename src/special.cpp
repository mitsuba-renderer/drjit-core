

void jitc_var_printf(int cuda, const char *fmt, uint32_t narg,
                    const uint32_t *arg) {
    if (!cuda)
        jitc_raise("jit_var_printf(): only supported in CUDA mode at the moment.");
    buffer.clear();
    buffer.put(
        "{\n"
        "        .global .align 1 .b8 fmt[] = { ");

    for (uint32_t i = 0; ; ++i) {
        buffer.put_uint32((uint32_t) fmt[i]);
        if (fmt[i] == '\0')
            break;
        buffer.put(", ");
    }
    buffer.put(" };\n");
    buffer.fmt("        .local .align 8 .b8 buf[%u];\n", 8 * narg);

    for (uint32_t i = 0, offset = 0; i < narg; ++i) {
        VarType vt = jitc_var_type(arg[i]);
        uint32_t size = var_type_size[(int) vt];
        if (vt == VarType::Float32)
            size = 8;

        offset = (offset + size - 1) / size * size;

        if (vt == VarType::Float32) {
            buffer.fmt("        cvt.f64.f32 %%d0, $r%u;\n"
                       "        st.local.f64 [buf+%u], %%d0;\n",
                       i + 1, offset);
        } else {
            buffer.fmt("        st.local.$t%u [buf+%u], $r%u;\n",
                       i + 1, offset, i + 1);
        }
        offset += size;
    }

    buffer.put("\n        .reg.b64 %fmt_r, %buf_r;\n"
               "        cvta.global.u64 %fmt_r, fmt;\n"
               "        cvta.local.u64 %buf_r, buf;\n"
               "        {\n"
               "            .param .b64 fmt_p;\n"
               "            .param .b64 buf_p;\n"
               "            .param .b32 rv_p;\n"
               "            st.param.b64 [fmt_p], %fmt_r;\n"
               "            st.param.b64 [buf_p], %buf_r;\n"
               "            call (rv_p), vprintf, (fmt_p, buf_p);\n"
               "        }\n"
               "    }\n");

    uint32_t decl = jitc_var_new_0(cuda, VarType::Global,
                                  ".extern .func (.param .b32 rv) vprintf "
                                  "(.param .b64 fmt, .param .b64 buf);\n",
                                  1, 1);

    uint32_t idx = 0;
    switch (narg) {
        case 0:
            idx = jitc_var_new_1(cuda, VarType::Void, buffer.get(), 0, decl);
            break;
        case 1:
            idx = jitc_var_new_2(cuda, VarType::Void, buffer.get(), 0, arg[0], decl);
            break;
        case 2:
            idx = jitc_var_new_3(cuda, VarType::Void, buffer.get(), 0, arg[0], arg[1], decl);
            break;
        case 3:
            idx = jitc_var_new_4(cuda, VarType::Void, buffer.get(), 0, arg[0], arg[1], arg[2], decl);
            break;
        default:
            jitc_raise("jit_var_printf(): max 3 arguments supported!");
    }

    jitc_var_dec_ref_ext(decl);
    jitc_var_mark_side_effect(idx, 0);
}

void jitc_var_vcall(int cuda,
                   const char *domain,
                   const char *name,
                   uint32_t self,
                   uint32_t n_inst,
                   const uint32_t *inst_ids,
                   const uint64_t *inst_hash,
                   uint32_t n_in, const uint32_t *in,
                   uint32_t n_out, uint32_t *out,
                   const uint32_t *need_in,
                   const uint32_t *need_out,
                   uint32_t n_extra,
                   const uint32_t *extra,
                   const uint32_t *extra_offset,
                   int side_effects) {
    state.mutex.unlock();
    lock_guard guard(state.eval_mutex);
    state.mutex.lock();

    std::vector<uint64_t> sorted(inst_hash, inst_hash + n_inst);
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());

    uint32_t elided_in = 0, elided_out = 0;
    for (uint32_t i = 0; i < n_in; ++i)
        elided_in += need_in && need_in[i] == 0 ? 1 : 0;
    for (uint32_t i = 0; i < n_out; ++i)
        elided_out += need_out && need_out[i] == 0 ? 1 : 0;

    jitc_log(
        Info,
        "jit_var_vcall(): %s::%s(), %u instances (%u elided), "
        "%u inputs (%u elided), %u outputs (%u elided), needs %u pointers, %s.",
        domain, name, (uint32_t) sorted.size(),
        n_inst - (uint32_t) sorted.size(), n_in - elided_in, elided_in,
        n_out - elided_out, elided_out, n_extra,
        side_effects ? "side effects" : "no side effects");

    uint32_t index = jitc_var_new_0(cuda, VarType::Void, "", 1, 1);

    buffer.clear();

    uint32_t width = jitc_llvm_vector_width;
    if (cuda) {
        buffer.put(".global .u64 $r0[] = { ");
        for (uint32_t i = 0; i < n_inst; ++i) {
            buffer.fmt("func_%016llx%s", (unsigned long long) inst_hash[i],
                       i + 1 < n_inst ? ", " : "");
            uint32_t prev = index;
            index = jitc_var_new_2(cuda, VarType::Void, "", 1, inst_ids[i], index);
            jitc_var_dec_ref_ext(prev);
            jitc_var_dec_ref_ext(inst_ids[i]);
        }
        buffer.put(" };\n");
    } else {
        buffer.fmt("@$r0 = private unnamed_addr constant [%u x void (i8*, i8*, i8**, <%u x i1>)*] [", n_inst, width);
        for (uint32_t i = 0; i < n_inst; ++i) {
            buffer.fmt("void (i8*, i8*, i8**, <%u x i1>)* @func_%016llx%s", width,
                       (unsigned long long) inst_hash[i],
                       i + 1 < n_inst ? ", " : "");
            uint32_t prev = index;
            index = jitc_var_new_2(cuda, VarType::Void, "", 1, inst_ids[i], index);
            jitc_var_dec_ref_ext(prev);
            jitc_var_dec_ref_ext(inst_ids[i]);
        }
        buffer.put(" ], align 8\n");
    }

    uint32_t call_table =
        jitc_var_new_1(cuda, VarType::Global, buffer.get(), 0, index);
    uint32_t call_target = 0;

    buffer.clear();
    buffer.fmt("// %s::%s\n    ", domain, name);
    buffer.put("// indirect call via table $r2: ");
    for (size_t i = 0; i < sorted.size(); ++i)
        buffer.fmt("%016llx%s", (unsigned long long) sorted[i],
                   i + 1 < sorted.size() ? ", " : "");

    if (cuda) {
        // Don't delete comment, patch code in optix_api.cpp looks for it
        buffer.put(
            "\n    "
            "// OptiX variant:\n    "
            "// add.u32 %r3, $r1, sbt_id_offset;\n    "
            "// call ($r0), _optix_call_direct_callable, (%r3);\n    "
            "// CUDA variant:\n    "
            "   mov.$t0 $r0, $r2;\n    "
            "   mad.wide.u32 $r0, $r1, 8, $r0;\n    "
            "   ld.global.$t0 $r0, [$r0]"
        );
        call_target = jitc_var_new_2(1, VarType::UInt64, buffer.get(), 0, self,
                                    call_table);
    }

    uint32_t extra_id;
    if (n_extra > 0) {
        std::unique_ptr<void *[]> tmp(new void *[n_extra]);
        for (uint32_t i = 0; i < n_extra; ++i) {
            uint32_t id = extra[i];
            tmp[i] = jitc_var(id)->data;
            uint32_t prev = index;
            index = jitc_var_new_1(cuda, VarType::Void, "", 1, index);
            jitc_var(index)->dep[3] = id;
            jitc_var_dec_ref_ext(prev);
        }

        uint32_t extra_offset_buf = jitc_var_mem_copy(
            cuda, AllocType::Host, VarType::UInt32, extra_offset, n_inst);
        uint32_t extra_buf = jitc_var_mem_copy(
            cuda, AllocType::Host, VarType::UInt64, tmp.get(), n_extra);

        uint32_t extra_offset_ptr =
            jitc_var_copy_ptr(cuda, jitc_var_ptr(extra_offset_buf), extra_offset_buf);
        uint32_t extra_ptr =
            jitc_var_copy_ptr(cuda, jitc_var_ptr(extra_buf), extra_buf);

        jitc_var_dec_ref_ext(extra_offset_buf);
        jitc_var_dec_ref_ext(extra_buf);

        extra_id = jitc_var_new_3(cuda, VarType::UInt64,
                "mad.wide.u32 $r0, $r1, 4, $r2$n"
                "ld.global.nc.u32 %rd3, [$r0]$n"
                "add.u64 $r0, $r3, %rd3",
                1, self, extra_offset_ptr, extra_ptr);

        jitc_var_dec_ref_ext(extra_offset_ptr);
        jitc_var_dec_ref_ext(extra_ptr);
    } else {
        extra_id = jitc_var_new_0(cuda, VarType::UInt64, "mov.$t0 $r0, 0", 1, 1);
    }

    uint32_t prev = index;
    index = jitc_var_new_3(cuda, VarType::Void, "", 1, call_target, extra_id,
                          index);
    jitc_var_dec_ref_ext(call_target);
    jitc_var_dec_ref_ext(extra_id);
    jitc_var_dec_ref_ext(prev);

    std::unique_ptr<uint32_t[]> in_new(new uint32_t[n_in]);
    uint32_t offset_in = 0, align_in = 1;
    for (uint32_t i = 0; i < n_in; ++i) {
        if (need_in && need_in[i] == 0)
            continue;
        VarType vt = jitc_var_type(in[i]);
        uint32_t size = var_type_size[(uint32_t) vt], prev2 = index;

        if (vt == VarType::Bool) {
            in_new[i] = jitc_var_new_1(cuda, VarType::UInt16,
                                      "selp.$t0 $r0, 1, 0, $r1", 1, in[i]);
        } else {
            in_new[i] = in[i];
            jitc_var_inc_ref_ext(in[i]);
        }

        index = jitc_var_new_2(cuda, VarType::Void, "", 1, in_new[i], index);
        jitc_var_dec_ref_ext(in_new[i]);
        jitc_var_dec_ref_ext(prev2);
        offset_in = (offset_in + size - 1) / size * size;
        offset_in += size;
        align_in = std::max(align_in, size);
    }

    uint32_t offset_out = 0, align_out = 1;
    for (uint32_t i = 0; i < n_out; ++i) {
        if (need_out && need_out[i] == 0)
            continue;
        uint32_t size = var_type_size[(uint32_t) jitc_var_type(out[i])];
        offset_out = (offset_out + size - 1) / size * size;
        offset_out += size;
        align_out = std::max(align_out, size);
    }

    if (offset_in == 0)
        offset_in = 1;
    if (offset_out == 0)
        offset_out = 1;

    buffer.clear();
    buffer.fmt("\n    {\n"
	        "        .param .align %u .b8 param_out[%u];\n"
	        "        .param .align %u .b8 param_in[%u];\n",
	        align_out, offset_out, align_in, offset_in
    );

    buffer.fmt("        Fproto: .callprototype (.param .align %u .b8 _[%u]) _ "
               "(.param .align %u .b8 _[%u], .reg .u64 _);\n",
               align_out, offset_out, align_in, offset_in);

    prev = index;
    index = jitc_var_new_1(cuda, VarType::Void, buffer.get(), 0, index);
    jitc_var_dec_ref_ext(prev);

    offset_in = 0;
    for (uint32_t i = 0; i < n_in; ++i) {
        if (need_in && need_in[i] == 0)
            continue;
        VarType vt = jitc_var_type(in[i]);
        uint32_t size = var_type_size[(uint32_t) vt], prev2 = index;
        offset_in = (offset_in + size - 1) / size * size;
        buffer.clear();
        buffer.fmt("    st.param.%s [param_in+%u], $r1",
                   vt == VarType::Bool ? "u8" : "$t1", offset_in);
        index = jitc_var_new_2(cuda, VarType::Void, buffer.get(), 0,
                              in_new[i], index);
        jitc_var_dec_ref_ext(prev2);
        offset_in += size;
    }

    prev = index;
#if 0
    index = jitc_var_new_4(cuda, VarType::Void,
                          "    call (param_out), $r1, (param_in, $r2), $r3", 1,
                          call_target, extra_id, call_table, index);
#else
    index = jitc_var_new_3(cuda, VarType::Void,
                          "    call (param_out), $r1, (param_in, $r2), Fproto", 1,
                          call_target, extra_id, index);
#endif

    jitc_var_dec_ref_ext(call_table);
    jitc_var_dec_ref_ext(prev);

    offset_out = 0;
    for (uint32_t i = 0; i < n_out; ++i) {
        if (need_out && need_out[i] == 0)
            continue;
        VarType type = jitc_var_type(out[i]);
        uint32_t size = var_type_size[(uint32_t) type];
        offset_out = (offset_out + size - 1) / size * size;
        uint32_t prev2 = index;
        buffer.clear();
        buffer.fmt("    ld.param.$t0 $r0, [param_out+%u]", offset_out);
        index = jitc_var_new_1(cuda, type, buffer.get(), 0, index);
        out[i] = index;
        jitc_var_dec_ref_ext(prev2);
        offset_out += size;
    }

    prev = index;
    index = jitc_var_new_1(cuda, VarType::Void, "}\n", 1, index);
    jitc_var_dec_ref_ext(prev);

    if (side_effects) {
        jitc_var_inc_ref_ext(index);
        jitc_var_mark_side_effect(index, 0);
    }

    for (uint32_t i = 0; i < n_out; ++i) {
        if (need_out && need_out[i] == 0) {
            out[i] = jitc_var_new_1(cuda, jitc_var_type(out[i]),
                                   "mov.$b0 $r0, 0",
                                   1, out[i]);
        } else {
            out[i] = jitc_var_new_2(cuda, jitc_var_type(out[i]),
                                   "mov.$t0 $r0, $r1",
                                   1, out[i], index);
        }
    }

    jitc_var_dec_ref_ext(index);
}

