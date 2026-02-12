#define put(...)                                                               \
    buffer.put(__VA_ARGS__)

#define fmt(fmt, ...)                                                          \
    buffer.fmt_llvm(count_args(__VA_ARGS__), fmt, ##__VA_ARGS__)

#define fmt_intrinsic(fmt, ...)                                                \
    do {                                                                       \
        size_t tmpoff = buffer.size();                                         \
        buffer.fmt_llvm(count_args(__VA_ARGS__), fmt, ##__VA_ARGS__);          \
        jitc_register_global(buffer.get() + tmpoff);                           \
        buffer.rewind_to(tmpoff);                                              \
    } while (0)

#if !defined(__aarch64__)

#define def_fma_vec_f16_intrinsic()                                            \
    fmt_intrinsic(                                                             \
        "define internal <$w x half> @fma.v$wf16(<$w x half> %a, <$w x half> %b, <$w x half> %c) #0 ${\n" \
        "    %a_f32  = fpext <$w x half> %a to <$w x float>\n"                 \
        "    %b_f32  = fpext <$w x half> %b to <$w x float>\n"                 \
        "    %c_f32  = fpext <$w x half> %c to <$w x float>\n"                 \
        "    %out_f32 = call fast <$w x float> @llvm.fma.v$wf32(<$w x float> %a_f32, <$w x float> %b_f32, <$w x float> %c_f32)\n" \
        "    %out = fptrunc <$w x float> %out_f32 to <$w x half>\n"            \
        "    ret <$w x half> %out\n"                                           \
        "$}"                                                                   \
    )

#define def_minnum_vec_f16_intrinsic()                                         \
    fmt_intrinsic(                                                             \
        "define internal <$w x half> @minnum.v$wf16(<$w x half> %a, <$w x half> %b) local_unnamed_addr #0 ${\n" \
        "    %a_f32  = fpext <$w x half> %a to <$w x float>\n"                 \
        "    %b_f32  = fpext <$w x half> %b to <$w x float>\n"                 \
        "    %out_f32 = call fast <$w x float> @llvm.minnum.v$wf32(<$w x float> %a_f32, <$w x float> %b_f32)\n" \
        "    %out = fptrunc <$w x float> %out_f32 to <$w x half>\n"            \
        "    ret <$w x half> %out\n"                                           \
        "$}"                                                                   \
    )

#define def_maxnum_vec_f16_intrinsic()                                         \
    fmt_intrinsic(                                                             \
        "define internal <$w x half> @maxnum.v$wf16(<$w x half> %a, <$w x half> %b) local_unnamed_addr #0 ${\n" \
        "    %a_f32  = fpext <$w x half> %a to <$w x float>\n"                 \
        "    %b_f32  = fpext <$w x half> %b to <$w x float>\n"                 \
        "    %out_f32 = call fast <$w x float> @llvm.maxnum.v$wf32(<$w x float> %a_f32, <$w x float> %b_f32)\n" \
        "    %out = fptrunc <$w x float> %out_f32 to <$w x half>\n"            \
        "    ret <$w x half> %out\n"                                           \
        "$}"                                                                   \
    )

#endif
