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
