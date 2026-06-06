#include "strbuf.h"
#include "internal.h"
#include "var.h"
#include "eval.h"
#include <cstdarg>

#if defined(_MSC_VER)
#  include <intrin.h>
#endif

/// Global string buffer used to generate PTX/LLVM IR
StringBuffer buffer { 1024 };

static const char num[] = "0123456789abcdef";

// Two-digit lookup table ("00010203...99") for fast unsigned->decimal conversion
static const char digit_pairs[201] =
    "00010203040506070809"
    "10111213141516171819"
    "20212223242526272829"
    "30313233343536373839"
    "40414243444546474849"
    "50515253545556575859"
    "60616263646566676869"
    "70717273747576777879"
    "80818283848586878889"
    "90919293949596979899";

static const uint32_t pow10_u32[10] = {
    0, 10, 100, 1000, 10000,
    100000, 1000000, 10000000, 100000000, 1000000000
};

static const uint64_t pow10_u64[20] = {
    0ull, 10ull, 100ull, 1000ull, 10000ull, 100000ull, 1000000ull,
    10000000ull, 100000000ull, 1000000000ull, 10000000000ull, 100000000000ull,
    1000000000000ull, 10000000000000ull, 100000000000000ull, 1000000000000000ull,
    10000000000000000ull, 100000000000000000ull, 1000000000000000000ull,
    10000000000000000000ull
};

// Bit length of a *nonzero* 32-bit integer
static inline unsigned bit_length(uint32_t n) {
#if defined(_MSC_VER)
    unsigned long i;
    _BitScanReverse(&i, (unsigned long) n);
    return (unsigned) i + 1;
#else
    return 32u - (unsigned) __builtin_clz(n);
#endif
}

// Bit length of a *nonzero* 64-bit integer
static inline unsigned bit_length(uint64_t n) {
#if defined(_MSC_VER)
    unsigned long i;
    _BitScanReverse64(&i, n);
    return (unsigned) i + 1;
#else
    return 64u - (unsigned) __builtin_clzll(n);
#endif
}

// Number of decimal digits of a 32-bit unsigned value
static inline unsigned digits10(uint32_t n) {
    unsigned bits = bit_length(n | 1);
    unsigned t = (bits * 1233) >> 12;
    return t + 1 - (n < pow10_u32[t]);
}

// Number of decimal digits of a 64-bit unsigned value
static inline unsigned digits10(uint64_t n) {
    unsigned bits = bit_length(n | 1);
    unsigned t = (bits * 1233) >> 12;
    return t + 1 - (n < pow10_u64[t]);
}

StringBuffer::StringBuffer(size_t capacity)
    : m_start(nullptr), m_cur(nullptr), m_end(nullptr) {
    if (capacity == 0)
        return;

    m_start = (char *) malloc_check(capacity);
    m_end = m_start + capacity;
    clear();
}

StringBuffer::StringBuffer(StringBuffer &&b)
    : m_start(b.m_start), m_cur(b.m_cur), m_end(b.m_end) {
    b.m_start = b.m_cur = b.m_end = nullptr;
}

StringBuffer::~StringBuffer() { free(m_start); }

void StringBuffer::delete_trailing_commas() {
    while (m_cur > m_start) {
        char ch = *(m_cur - 1);
        if (ch == ',' || ch == ' ')
            m_cur--;
        else
            break;
    }
}

StringBuffer &StringBuffer::operator=(StringBuffer &&b) {
    swap(b);
    return *this;
}

void StringBuffer::clear() {
    m_cur = m_start;
    if (m_start != m_end)
        m_start[0] = '\0';
}

void StringBuffer::rewind_to(size_t pos) {
    m_cur = m_start + pos;
    if (m_start != m_end)
        *m_cur = '\0';
}

void StringBuffer::rewind(size_t rel) {
    m_cur -= rel;
    if (m_start != m_end)
        *m_cur = '\0';
}

void StringBuffer::move_suffix(size_t suffix_start, size_t suffix_target) {
    size_t buffer_size = size(),
           suffix_size = buffer_size - suffix_start;

    // Ensure that there is extra space for moving things around
    put('\0', suffix_size);

    // Move the portion following the insertion point
    memmove(m_start + suffix_target + suffix_size,
            m_start + suffix_target, buffer_size - suffix_target);

    // Finally copy the code to the insertion point
    memcpy(m_start + suffix_target,
           m_start + buffer_size, suffix_size);

    rewind_to(buffer_size);
}

void StringBuffer::swap(StringBuffer &b) {
    std::swap(m_start, b.m_start);
    std::swap(m_cur, b.m_cur);
    std::swap(m_end, b.m_end);
}

void StringBuffer::expand(size_t nbytes) {
    // Ensure that that there is space for a trailing zero character
    nbytes += 1;

    size_t len = size(), old_capacity = capacity();

    // Increase capacity by powers of 2 until 'nbytes' fits
    size_t new_capacity = old_capacity;
    if (new_capacity == 0)
        new_capacity = 1;

    while (len + nbytes > new_capacity)
        new_capacity *= 2;

    if (new_capacity != old_capacity) {
        m_start = (char *) realloc_check(m_start, new_capacity);
        m_end = m_start + new_capacity;
        m_cur = m_start + len;
    }
}

static void reverse_str(char *start, char *end) {
    end--;
    while (start < end) {
        char tmp = *start;
        *start = *end;
        *end = tmp;
        ++start; --end;
    }
}

void StringBuffer::put_u32(uint32_t value) {
    if (unlikely(!m_cur || m_cur + MAXSIZE_U32 >= m_end))
        expand(MAXSIZE_U32);
    put_u32_unchecked(value);
}

void StringBuffer::put_u64(uint64_t value) {
    if (unlikely(!m_cur || m_cur + MAXSIZE_U64 >= m_end))
        expand(MAXSIZE_U64);
    put_u64_unchecked(value);
}


void StringBuffer::put_x32(uint32_t value) {
    if (unlikely(!m_cur || m_cur + MAXSIZE_X32 >= m_end))
        expand(MAXSIZE_X32);
    put_x32_unchecked(value);
}

void StringBuffer::put_x64(uint64_t value) {
    if (unlikely(!m_cur || m_cur + MAXSIZE_X64 >= m_end))
        expand(MAXSIZE_X64);
    put_x64_unchecked(value);
}

void StringBuffer::put_u32_unchecked(uint32_t value) {
    unsigned n = digits10(value);
    char *end = m_cur + n, *p = end;
    while (value >= 100) {
        unsigned idx = (value % 100) * 2;
        value /= 100;
        p -= 2;
        memcpy(p, &digit_pairs[idx], 2);
    }
    if (value < 10)
        *--p = (char) ('0' + value);
    else
        memcpy(p - 2, &digit_pairs[value * 2], 2);
    m_cur = end;
}

void StringBuffer::put_u64_unchecked(uint64_t value) {
    unsigned n = digits10(value);
    char *end = m_cur + n, *p = end;
    while (value >= 100) {
        unsigned idx = (unsigned) (value % 100) * 2;
        value /= 100;
        p -= 2;
        memcpy(p, &digit_pairs[idx], 2);
    }
    if (value < 10)
        *--p = (char) ('0' + value);
    else
        memcpy(p - 2, &digit_pairs[(unsigned) value * 2], 2);
    m_cur = end;
}

void StringBuffer::put_x32_unchecked(uint32_t value) {
    char *start = m_cur;
    do {
        *m_cur++ = num[value % 16];
        value /= 16;
    } while (value);

    reverse_str(start, m_cur);
}

void StringBuffer::put_x64_unchecked(uint64_t value) {
    char *start = m_cur;
    do {
        *m_cur++ = num[value % 16];
        value /= 16;
    } while (value);

    reverse_str(start, m_cur);
}

void StringBuffer::put_q64_unchecked(uint64_t value) {
    for (uint32_t i = 0; i < 16; ++i) {
        *(m_cur + 15 - i) = num[value % 16];
        value /= 16;
    }
    m_cur += 16;
}

void StringBuffer::put_unchecked(const char *str) {
    while (true) {
        char c = *str++;
        if (!c)
            break;
        *m_cur++ = c;
    }
}

size_t StringBuffer::fmt(const char *fmt, ...) {
    do {
        size_t remain = m_end - m_cur;

        va_list args;
        va_start(args, fmt);
        int rv = vsnprintf(m_cur, remain, fmt, args);
        if (rv < 0) {
            fprintf(stderr,
                    "StringBuffer::fmt(): vsnprintf failed with error code %i!",
                    rv);
            abort();
        }
        va_end(args);

        if (likely(m_cur && m_cur + rv < m_end)) {
            m_cur += rv;
            return (size_t) rv;
        } else {
            expand((size_t) rv);
        }
    } while (true);
}

size_t StringBuffer::vfmt(const char *fmt, va_list args_) {
    va_list args;
    do {
        size_t remain = m_end - m_cur;

        va_copy(args, args_);
        int rv = vsnprintf(m_cur, remain, fmt, args);
        va_end(args);

        if (rv < 0) {
            fprintf(stderr,
                    "StringBuffer::fmt(): vsnprintf failed with error code %i!",
                    rv);
            abort();
        }

        if (likely(m_cur && m_cur + rv < m_end)) {
            m_cur += rv;
            return (size_t) rv;
        } else {
            expand((size_t) rv);
        }
    } while (true);
}

void StringBuffer::fmt_llvm(size_t nargs, size_t fmt_len, const char *fmt, ...) {
    va_list args2;
    va_start(args2, fmt);

    // Bound the maximum output size: every directive other than the unbounded
    // '$s' expands to at most 8 bytes per format character + 48 bytes per
    // argument; '$s' grows the buffer on demand below.
    size_t bound = fmt_len * 8 + nargs * 48;
    if (unlikely(!m_cur || m_cur + bound >= m_end))
        expand(bound);

    // Convert the format string
    const char *p = fmt;
    char *cur = m_cur;
    char c;
    while ((c = *p++) != '\0') {
        // Fast path, plain copy
        if (likely(c != '$')) {
            *cur++ = c;
            continue;
        }

        m_cur = cur;
        {
            switch (*p++) {
                case 'u': put_u32_unchecked(va_arg(args2, uint32_t)); break;
                case 's': {
                        const char *s = va_arg(args2, const char *);
                        put(s, strlen(s));
                        // '$s' is the only unbounded directive: 'put()' grew the
                        // buffer to fit the string, so restore the worst-case
                        // headroom assumed by the unchecked writes that follow.
                        if (unlikely(m_cur + bound >= m_end))
                            expand(bound);
                    }
                    break;

                case 'w':
                    put_u32_unchecked(jitc_llvm_vector_width);
                    break;

                case 't': {
                        const Variable *v = va_arg(args2, const Variable *);
                        put_unchecked(type_name_llvm[v->type]);
                    }
                    break;

                case 'b': {
                        const Variable *v = va_arg(args2, const Variable *);
                        put_unchecked(type_name_llvm_bin[v->type]);
                    }
                    break;

                case 'd': {
                        const Variable *v = va_arg(args2, const Variable *);
                        put_unchecked(type_name_llvm_big[v->type]);
                    }
                    break;

                case 'H': {
                        const Variable *v = va_arg(args2, const Variable *);
                        put_unchecked(v->type == (uint32_t) VarType::Bool
                                          ? "i8"
                                          : type_name_llvm_abbrev[v->type]);
                    }
                    break;

                case 'h': {
                        const Variable *v = va_arg(args2, const Variable *);
                        put_unchecked(type_name_llvm_abbrev[v->type]);
                    }
                    break;

                case 'm': {
                        const Variable *v = va_arg(args2, const Variable *);
                        uint32_t type = v->type == (uint32_t) VarType::Bool
                                            ? (uint32_t) VarType::UInt8
                                            : v->type;
                        put_unchecked(type_name_llvm[type]);
                    }
                    break;

                case 'T': {
                        const Variable *v = va_arg(args2, const Variable *);
                        *m_cur ++= '<';
                        put_u32_unchecked(jitc_llvm_vector_width);
                        *m_cur ++= ' '; *m_cur ++= 'x'; *m_cur ++= ' ';
                        put_unchecked(type_name_llvm[v->type]);
                        *m_cur ++= '>';
                    }
                    break;

                case 'B': {
                        const Variable *v = va_arg(args2, const Variable *);
                        *m_cur ++= '<';
                        put_u32_unchecked(jitc_llvm_vector_width);
                        *m_cur ++= ' '; *m_cur ++= 'x'; *m_cur ++= ' ';
                        put_unchecked(type_name_llvm_bin[v->type]);
                        *m_cur ++= '>';
                    }
                    break;

                case 'D': {
                        const Variable *v = va_arg(args2, const Variable *);
                        *m_cur ++= '<';
                        put_u32_unchecked(jitc_llvm_vector_width);
                        *m_cur ++= ' '; *m_cur ++= 'x'; *m_cur ++= ' ';
                        put_unchecked(type_name_llvm_big[v->type]);
                        *m_cur ++= '>';
                    }
                    break;

                case 'M': {
                        const Variable *v = va_arg(args2, const Variable *);
                        uint32_t type = v->type == (uint32_t) VarType::Bool
                                            ? (uint32_t) VarType::UInt8
                                            : v->type;
                        *m_cur ++= '<';
                        put_u32_unchecked(jitc_llvm_vector_width);
                        *m_cur ++= ' '; *m_cur ++= 'x'; *m_cur ++= ' ';
                        put_unchecked(type_name_llvm[type]);
                        *m_cur ++= '>';
                    }
                    break;

                case 'v': {
                        const Variable *v = va_arg(args2, const Variable *);
                        put_unchecked(type_prefix[v->type]);
                        put_u32_unchecked(v->reg_index);
                    }
                    break;

                case 'V': {
                        const Variable *v = va_arg(args2, const Variable *);
                        *m_cur ++= '<';
                        put_u32_unchecked(jitc_llvm_vector_width);
                        *m_cur ++= ' '; *m_cur ++= 'x'; *m_cur ++= ' ';
                        put_unchecked(type_name_llvm[v->type]);
                        *m_cur ++= '>';
                        *m_cur ++= ' ';
                        put_unchecked(type_prefix[v->type]);
                        put_u32_unchecked(v->reg_index);
                    }
                    break;

                case 'l': {
                        const Variable *v = va_arg(args2, const Variable *);
                        VarType vt = (VarType) v->type;
                        uint64_t literal = v->literal;

                        if (vt == VarType::Float32) {
                            float f;
                            memcpy(&f, &literal, sizeof(float));
                            double d = f;
                            memcpy(&literal, &d, sizeof(uint64_t));
                            vt = VarType::Float64;
                        } else if (vt == VarType::Float16) {
                            drjit::half h;
                            memcpy((void*)&h, &literal, sizeof(drjit::half));
                            double d = float(h);
                            memcpy(&literal, &d, sizeof(uint64_t));
                            vt = VarType::Float64;
                        }

                        if (vt == VarType::Float64) {
                            *m_cur ++= '0';
                            *m_cur ++= 'x';
                            put_x64_unchecked(literal);
                        } else {
                            put_u64_unchecked(literal);
                        }
                    };
                    break;

                case 'a': {
                        const Variable *v = va_arg(args2, const Variable *);
                        put_u32_unchecked(v->unaligned ? 1 : type_size[v->type]);
                    }
                    break;

                case 'A': {
                        const Variable *v = va_arg(args2, const Variable *);
                        put_u32_unchecked(v->unaligned ? 1 : (type_size[v->type] * jitc_llvm_vector_width));
                    }
                    break;

                case 'o': {
                        const Variable *v = va_arg(args2, const Variable *);
                        put_u32_unchecked(v->param_offset / (uint32_t) sizeof(void *));
                    }
                    break;

                case 'z':
                    put_unchecked("zeroinitializer");
                    break;

                case '<':
                    if (callable_depth > 0) {
                        *m_cur ++= '<';
                        put_u32_unchecked(jitc_llvm_vector_width);
                        *m_cur ++= ' '; *m_cur ++= 'x'; *m_cur ++= ' ';
                    }
                    break;

                case '>':
                    if (callable_depth > 0)
                        *m_cur ++= '>';
                    break;

                default:
                    fprintf(stderr,
                            "StringBuffer::fmt_llvm(): encountered unsupported "
                            "character \"$%c\" in format string!\n", p[-1]);
                    abort();
            }
        }

        // Update 'cur' in case one of the put_*() functions overwrote it
        cur = m_cur;
    }
    va_end(args2);

    m_cur = cur;
    *m_cur = '\0';
}

#if defined(DRJIT_ENABLE_CUDA)
void StringBuffer::fmt_cuda(size_t nargs, size_t fmt_len, const char *fmt, ...) {
    va_list args2;
    va_start(args2, fmt);

    // Bound the maximum output size; see fmt_llvm().
    size_t bound = fmt_len * 8 + nargs * 48;
    if (unlikely(!m_cur || m_cur + bound >= m_end))
        expand(bound);

    // Convert the format string
    const char *p = fmt;
    char *cur = m_cur;
    char c;
    while ((c = *p++) != '\0') {
        // Fast path, plain copy
        if (likely(c != '$')) {
            *cur++ = c;
            continue;
        }

        m_cur = cur;
        {
            switch (*p++) {
                case 'u': put_u32_unchecked(va_arg(args2, uint32_t)); break;
                case 'Q': put_q64_unchecked(va_arg(args2, uint64_t)); break;

                case 's': {
                        const char *s = va_arg(args2, const char *);
                        put(s, strlen(s));
                        // '$s' is the only unbounded directive: 'put()' grew the
                        // buffer to fit the string, so restore the worst-case
                        // headroom assumed by the unchecked writes that follow.
                        if (unlikely(m_cur + bound >= m_end))
                            expand(bound);
                    }
                    break;

                case 't': {
                        const Variable *v = va_arg(args2, const Variable *);
                        put_unchecked(type_name_ptx[v->type]);
                    }
                    break;

                case 'B': {
                        const Variable *v = va_arg(args2, const Variable *);
                        put_unchecked(type_name_ptx_bin2[v->type]);
                    }
                    break;

                case 'b': {
                        const Variable *v = va_arg(args2, const Variable *);
                        put_unchecked(type_name_ptx_bin[v->type]);
                    }
                    break;

                case 'V': {
                        const Variable *v = va_arg(args2, const Variable *);
                        if (v->type == (uint32_t) VarType::Bool) {
                            put_unchecked("%w0");
                        } else {
                            put_unchecked(type_prefix[v->type]);
                            put_u32_unchecked(v->reg_index);
                        }
                    }
                    break;

                case 'v': {
                        const Variable *v = va_arg(args2, const Variable *);
                        put_unchecked(type_prefix[v->type]);
                        put_u32_unchecked(v->reg_index);
                    }
                    break;

                case 'l': {
                        const Variable *v = va_arg(args2, const Variable *);
                        *m_cur ++= '0';
                        *m_cur ++= 'x';
                        put_x64_unchecked(v->literal);
                    };
                    break;

                case 'a': {
                        const Variable *v = va_arg(args2, const Variable *);
                        put_u32_unchecked(type_size[v->type]);
                    }
                    break;

                case 'o': {
                        const Variable *v = va_arg(args2, const Variable *);
                        put_u32_unchecked(v->param_offset);
                    }
                    break;

                default:
                    fprintf(stderr,
                            "StringBuffer::fmt_cuda(): encountered unsupported "
                            "character \"$%c\" in format string!\n", p[-1]);
                    abort();
            }
        }

        // Update 'cur' in case one of the put_*() functions overwrote it
        cur = m_cur;
    }
    va_end(args2);

    m_cur = cur;
    *m_cur = '\0';
}
#endif

#if defined(DRJIT_ENABLE_METAL)
void StringBuffer::fmt_metal(size_t nargs, size_t fmt_len, const char *fmt, ...) {
    va_list args2;
    va_start(args2, fmt);

    // Bound the maximum output size; see fmt_llvm().
    size_t bound = fmt_len * 8 + nargs * 48;
    if (unlikely(!m_cur || m_cur + bound >= m_end))
        expand(bound);

    // Emit the formatted string
    const char *p = fmt;
    char *cur = m_cur;
    char c;
    while ((c = *p++) != '\0') {
        // Fast path, plain copy
        if (likely(c != '$')) {
            *cur++ = c;
            continue;
        }

        m_cur = cur;
        {
            switch (*p++) {
                case 'u': put_u32_unchecked(va_arg(args2, uint32_t)); break;

                case 's': {
                        const char *s = va_arg(args2, const char *);
                        put(s, strlen(s));
                        // '$s' is the only unbounded directive: 'put()' grew the
                        // buffer to fit the string, so restore the worst-case
                        // headroom assumed by the unchecked writes that follow.
                        if (unlikely(m_cur + bound >= m_end))
                            expand(bound);
                    }
                    break;

                case 't': {
                        const Variable *v = va_arg(args2, const Variable *);
                        put_unchecked(type_name_metal[v->type]);
                    }
                    break;

                case 'b': {
                        const Variable *v = va_arg(args2, const Variable *);
                        put_unchecked(type_name_metal_bin[v->type]);
                    }
                    break;

                case 'v': {
                        const Variable *v = va_arg(args2, const Variable *);
                        *m_cur++ = 'r';
                        put_u32_unchecked(v->reg_index);
                    }
                    break;

                case 'l': {
                        const Variable *v = va_arg(args2, const Variable *);
                        *m_cur++ = '0';
                        *m_cur++ = 'x';
                        put_x64_unchecked(v->literal);
                    }
                    break;

                case 'o': {
                        const Variable *v = va_arg(args2, const Variable *);
                        put_u32_unchecked(v->param_offset / (uint32_t) sizeof(void *) - 1);
                    }
                    break;

                default:
                    fprintf(stderr,
                            "StringBuffer::fmt_metal(): encountered unsupported "
                            "character \"$%c\" in format string!\n", p[-1]);
                    abort();
            }
        }

        // Update 'cur' in case one of the put_*() functions overwrote it
        cur = m_cur;
    }
    va_end(args2);

    m_cur = cur;
    *m_cur = '\0';
}
#endif
