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
#define def_f16_wrapper_binary_intrinsic(op_str)                               \
    fmt_intrinsic(                                                             \
        "define internal <$w x half> @$s.v$wf16(<$w x half> %a, <$w x half> %b) #0 ${\n" \
        "    %a_f32  = fpext <$w x half> %a to <$w x float>\n"                 \
        "    %b_f32  = fpext <$w x half> %b to <$w x float>\n"                 \
        "    %out_f32 = call fast <$w x float> @llvm.$s.v$wf32(<$w x float> %a_f32, <$w x float> %b_f32)\n" \
        "    %out = fptrunc <$w x float> %out_f32 to <$w x half>\n"            \
        "    ret <$w x half> %out\n"                                           \
        "$}", op_str, op_str)

#define def_f16_wrapper_ternary_intrinsic(op_str)                              \
    fmt_intrinsic(                                                             \
        "define internal <$w x half> @$s.v$wf16(<$w x half> %a, <$w x half> %b, <$w x half> %c) #0 ${\n" \
        "    %a_f32  = fpext <$w x half> %a to <$w x float>\n"                 \
        "    %b_f32  = fpext <$w x half> %b to <$w x float>\n"                 \
        "    %c_f32  = fpext <$w x half> %c to <$w x float>\n"                 \
        "    %out_f32 = call fast <$w x float> @llvm.$s.v$wf32(<$w x float> %a_f32, <$w x float> %b_f32, <$w x float> %c_f32)\n" \
        "    %out = fptrunc <$w x float> %out_f32 to <$w x half>\n"            \
        "    ret <$w x half> %out\n"                                           \
        "$}", op_str, op_str)
#endif
