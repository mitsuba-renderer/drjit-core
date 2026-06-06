#include "strbuf.h"
#include "internal.h"
#include "var.h"
#include "eval.h"
#include <cstdarg>

/// Global string buffer used to generate PTX/LLVM IR
StringBuffer buffer { 1024 };

static const char num[] = "0123456789abcdef";

// Two-digit lookup table for jeaii unsigned->decimal conversion
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

// Version with doubled leading  digits for values < 10
static const char digit_pairs_lead[201] =
    "00112233445566778899"
    "10111213141516171819"
    "20212223242526272829"
    "30313233343536373839"
    "40414243444546474849"
    "50515253545556575859"
    "60616263646566676869"
    "70717273747576777879"
    "80818283848586878889"
    "90919293949596979899";

// Fixed-point fractional masks used by the division-free integer formatter.
static const uint64_t mask24 = (1ull << 24) - 1,
                      mask32 = (1ull << 32) - 1,
                      mask57 = (1ull << 57) - 1;

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

// Division-free unsigned to decimal conversion based on James Anhalt ("jeaiii",
// https://github.com/jeaiii/itoa). A magnitude cascade produces a a scaled
// fixed-point number whose representation can be used to produce the output two
// digits at a time. The 2-byte stores may spill one byte past the end, which
// usage in StringBuffer permits.
static inline char *w_u32(char *b, uint32_t n) {
    if (n < 100) {
        memcpy(b, &digit_pairs_lead[n * 2], 2);
        return n < 10 ? b + 1 : b + 2;
    }
    if (n < 10000) {
        uint32_t f0 = (uint32_t) (10 * (1 << 24) / 1e3 + 1) * n;
        memcpy(b, &digit_pairs_lead[(f0 >> 24) * 2], 2);
        b -= n < 1000;
        uint32_t f2 = (f0 & mask24) * 100; memcpy(b + 2, &digit_pairs[(f2 >> 24) * 2], 2);
        return b + 4;
    }
    if (n < 1000000) {
        uint64_t f0 = (uint64_t) (10 * (1ull << 32) / 1e5 + 1) * n;
        memcpy(b, &digit_pairs_lead[(f0 >> 32) * 2], 2);
        b -= n < 100000;
        uint64_t f2 = (f0 & mask32) * 100; memcpy(b + 2, &digit_pairs[(f2 >> 32) * 2], 2);
        uint64_t f4 = (f2 & mask32) * 100; memcpy(b + 4, &digit_pairs[(f4 >> 32) * 2], 2);
        return b + 6;
    }
    if (n < 100000000) {
        uint64_t f0 = (uint64_t) (10 * (1ull << 48) / 1e7 + 1) * n >> 16;
        memcpy(b, &digit_pairs_lead[(f0 >> 32) * 2], 2);
        b -= n < 10000000;
        uint64_t f2 = (f0 & mask32) * 100; memcpy(b + 2, &digit_pairs[(f2 >> 32) * 2], 2);
        uint64_t f4 = (f2 & mask32) * 100; memcpy(b + 4, &digit_pairs[(f4 >> 32) * 2], 2);
        uint64_t f6 = (f4 & mask32) * 100; memcpy(b + 6, &digit_pairs[(f6 >> 32) * 2], 2);
        return b + 8;
    }
    // 9-10 digits: a single fixed-point cascade scaled by 2^57.
    uint64_t f0 = (uint64_t) (10 * (1ull << 57) / 1e9 + 1) * n;
    memcpy(b, &digit_pairs_lead[(f0 >> 57) * 2], 2);
    b -= n < 1000000000u;
    uint64_t f2 = (f0 & mask57) * 100; memcpy(b + 2, &digit_pairs[(f2 >> 57) * 2], 2);
    uint64_t f4 = (f2 & mask57) * 100; memcpy(b + 4, &digit_pairs[(f4 >> 57) * 2], 2);
    uint64_t f6 = (f4 & mask57) * 100; memcpy(b + 6, &digit_pairs[(f6 >> 57) * 2], 2);
    uint64_t f8 = (f6 & mask57) * 100; memcpy(b + 8, &digit_pairs[(f8 >> 57) * 2], 2);
    return b + 10;
}

// Write 'z' in [0, 1e8) as eight zero-padded digits, division-free (jeaiii).
static inline void eight_digits(char *b, uint32_t z) {
    uint64_t f0 = ((uint64_t) ((1ull << 48) / 1e6 + 1) * z >> 16) + 1;
    memcpy(b, &digit_pairs[(f0 >> 32) * 2], 2);
    uint64_t f2 = (f0 & mask32) * 100; memcpy(b + 2, &digit_pairs[(f2 >> 32) * 2], 2);
    uint64_t f4 = (f2 & mask32) * 100; memcpy(b + 4, &digit_pairs[(f4 >> 32) * 2], 2);
    uint64_t f6 = (f4 & mask32) * 100; memcpy(b + 6, &digit_pairs[(f6 >> 32) * 2], 2);
}

// 64-bit unsigned decimal to string conversion based on the jeaiii implementation
static inline char *w_u64(char *b, uint64_t value) {
    if (value <= 0xFFFFFFFFull)
        return w_u32(b, (uint32_t) value);
    uint32_t z = (uint32_t) (value % 100000000u);
    b = w_u64(b, value / 100000000u);
    eight_digits(b, z);
    return b + 8;
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

void StringBuffer::put_u32_unchecked(uint32_t value) { m_cur = w_u32(m_cur, value); }

void StringBuffer::put_u64_unchecked(uint64_t value) { m_cur = w_u64(m_cur, value); }

// Cursor-passing cores of the hex/string writers, returning the advanced cursor.
static inline char *w_x32(char *d, uint32_t value) {
    char *start = d;
    do { *d++ = num[value % 16]; value /= 16; } while (value);
    reverse_str(start, d);
    return d;
}

static inline char *w_x64(char *d, uint64_t value) {
    char *start = d;
    do { *d++ = num[value % 16]; value /= 16; } while (value);
    reverse_str(start, d);
    return d;
}

static inline char *w_q64(char *d, uint64_t value) {
    for (uint32_t i = 0; i < 16; ++i) {
        *(d + 15 - i) = num[value % 16];
        value /= 16;
    }
    return d + 16;
}

static inline char *w_str(char *d, const char *str) {
    char c;
    while ((c = *str++) != '\0')
        *d++ = c;
    return d;
}

void StringBuffer::put_x32_unchecked(uint32_t value) { m_cur = w_x32(m_cur, value); }

void StringBuffer::put_x64_unchecked(uint64_t value) { m_cur = w_x64(m_cur, value); }

void StringBuffer::put_q64_unchecked(uint64_t value) { m_cur = w_q64(m_cur, value); }

void StringBuffer::put_unchecked(const char *str) { m_cur = w_str(m_cur, str); }

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

        // '$' directive: writes go through 'cur' via the w_*() helpers; only
        // '$s' (put()/expand()) touches the m_cur member.
        switch (*p++) {
            case 'u': cur = w_u32(cur, va_arg(args2, uint32_t)); break;
            case 's': {
                    const char *s = va_arg(args2, const char *);
                    m_cur = cur;
                    put(s, strlen(s));
                    // '$s' is unbounded: put() may have grown the buffer, so
                    // re-reserve the headroom the unchecked writes assume.
                    if (unlikely(m_cur + bound >= m_end))
                        expand(bound);
                    cur = m_cur;
                }
                break;

            case 'w':
                cur = w_u32(cur, jitc_llvm_vector_width);
                break;

            case 't': {
                    const Variable *v = va_arg(args2, const Variable *);
                    cur = w_str(cur, type_name_llvm[v->type]);
                }
                break;

            case 'b': {
                    const Variable *v = va_arg(args2, const Variable *);
                    cur = w_str(cur, type_name_llvm_bin[v->type]);
                }
                break;

            case 'd': {
                    const Variable *v = va_arg(args2, const Variable *);
                    cur = w_str(cur, type_name_llvm_big[v->type]);
                }
                break;

            case 'H': {
                    const Variable *v = va_arg(args2, const Variable *);
                    cur = w_str(cur, v->type == (uint32_t) VarType::Bool
                                         ? "i8"
                                         : type_name_llvm_abbrev[v->type]);
                }
                break;

            case 'h': {
                    const Variable *v = va_arg(args2, const Variable *);
                    cur = w_str(cur, type_name_llvm_abbrev[v->type]);
                }
                break;

            case 'm': {
                    const Variable *v = va_arg(args2, const Variable *);
                    uint32_t type = v->type == (uint32_t) VarType::Bool
                                        ? (uint32_t) VarType::UInt8
                                        : v->type;
                    cur = w_str(cur, type_name_llvm[type]);
                }
                break;

            case 'T': {
                    const Variable *v = va_arg(args2, const Variable *);
                    *cur++ = '<';
                    cur = w_u32(cur, jitc_llvm_vector_width);
                    *cur++ = ' '; *cur++ = 'x'; *cur++ = ' ';
                    cur = w_str(cur, type_name_llvm[v->type]);
                    *cur++ = '>';
                }
                break;

            case 'B': {
                    const Variable *v = va_arg(args2, const Variable *);
                    *cur++ = '<';
                    cur = w_u32(cur, jitc_llvm_vector_width);
                    *cur++ = ' '; *cur++ = 'x'; *cur++ = ' ';
                    cur = w_str(cur, type_name_llvm_bin[v->type]);
                    *cur++ = '>';
                }
                break;

            case 'D': {
                    const Variable *v = va_arg(args2, const Variable *);
                    *cur++ = '<';
                    cur = w_u32(cur, jitc_llvm_vector_width);
                    *cur++ = ' '; *cur++ = 'x'; *cur++ = ' ';
                    cur = w_str(cur, type_name_llvm_big[v->type]);
                    *cur++ = '>';
                }
                break;

            case 'M': {
                    const Variable *v = va_arg(args2, const Variable *);
                    uint32_t type = v->type == (uint32_t) VarType::Bool
                                        ? (uint32_t) VarType::UInt8
                                        : v->type;
                    *cur++ = '<';
                    cur = w_u32(cur, jitc_llvm_vector_width);
                    *cur++ = ' '; *cur++ = 'x'; *cur++ = ' ';
                    cur = w_str(cur, type_name_llvm[type]);
                    *cur++ = '>';
                }
                break;

            case 'v': {
                    const Variable *v = va_arg(args2, const Variable *);
                    cur = w_str(cur, type_prefix[v->type]);
                    cur = w_u32(cur, v->reg_index);
                }
                break;

            case 'V': {
                    const Variable *v = va_arg(args2, const Variable *);
                    *cur++ = '<';
                    cur = w_u32(cur, jitc_llvm_vector_width);
                    *cur++ = ' '; *cur++ = 'x'; *cur++ = ' ';
                    cur = w_str(cur, type_name_llvm[v->type]);
                    *cur++ = '>';
                    *cur++ = ' ';
                    cur = w_str(cur, type_prefix[v->type]);
                    cur = w_u32(cur, v->reg_index);
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
                        *cur++ = '0';
                        *cur++ = 'x';
                        cur = w_x64(cur, literal);
                    } else {
                        cur = w_u64(cur, literal);
                    }
                };
                break;

            case 'a': {
                    const Variable *v = va_arg(args2, const Variable *);
                    cur = w_u32(cur, v->unaligned ? 1 : type_size[v->type]);
                }
                break;

            case 'A': {
                    const Variable *v = va_arg(args2, const Variable *);
                    cur = w_u32(cur, v->unaligned ? 1 : (type_size[v->type] * jitc_llvm_vector_width));
                }
                break;

            case 'o': {
                    const Variable *v = va_arg(args2, const Variable *);
                    cur = w_u32(cur, v->param_offset / (uint32_t) sizeof(void *));
                }
                break;

            case 'z':
                cur = w_str(cur, "zeroinitializer");
                break;

            case '<':
                if (callable_depth > 0) {
                    *cur++ = '<';
                    cur = w_u32(cur, jitc_llvm_vector_width);
                    *cur++ = ' '; *cur++ = 'x'; *cur++ = ' ';
                }
                break;

            case '>':
                if (callable_depth > 0)
                    *cur++ = '>';
                break;

            default:
                fprintf(stderr,
                        "StringBuffer::fmt_llvm(): encountered unsupported "
                        "character \"$%c\" in format string!\n", p[-1]);
                abort();
        }
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

        // '$' directive: writes go through 'cur' via the w_*() helpers; only
        // '$s' (put()/expand()) touches the m_cur member.
        switch (*p++) {
            case 'u': cur = w_u32(cur, va_arg(args2, uint32_t)); break;
            case 'Q': cur = w_q64(cur, va_arg(args2, uint64_t)); break;

            case 's': {
                    const char *s = va_arg(args2, const char *);
                    m_cur = cur;
                    put(s, strlen(s));
                    // '$s' is unbounded: put() may have grown the buffer, so
                    // re-reserve the headroom the unchecked writes below assume.
                    if (unlikely(m_cur + bound >= m_end))
                        expand(bound);
                    cur = m_cur;
                }
                break;

            case 't': {
                    const Variable *v = va_arg(args2, const Variable *);
                    cur = w_str(cur, type_name_ptx[v->type]);
                }
                break;

            case 'B': {
                    const Variable *v = va_arg(args2, const Variable *);
                    cur = w_str(cur, type_name_ptx_bin2[v->type]);
                }
                break;

            case 'b': {
                    const Variable *v = va_arg(args2, const Variable *);
                    cur = w_str(cur, type_name_ptx_bin[v->type]);
                }
                break;

            case 'V': {
                    const Variable *v = va_arg(args2, const Variable *);
                    if (v->type == (uint32_t) VarType::Bool) {
                        cur = w_str(cur, "%w0");
                    } else {
                        cur = w_str(cur, type_prefix[v->type]);
                        cur = w_u32(cur, v->reg_index);
                    }
                }
                break;

            case 'v': {
                    const Variable *v = va_arg(args2, const Variable *);
                    cur = w_str(cur, type_prefix[v->type]);
                    cur = w_u32(cur, v->reg_index);
                }
                break;

            case 'l': {
                    const Variable *v = va_arg(args2, const Variable *);
                    *cur++ = '0';
                    *cur++ = 'x';
                    cur = w_x64(cur, v->literal);
                };
                break;

            case 'a': {
                    const Variable *v = va_arg(args2, const Variable *);
                    cur = w_u32(cur, type_size[v->type]);
                }
                break;

            case 'o': {
                    const Variable *v = va_arg(args2, const Variable *);
                    cur = w_u32(cur, v->param_offset);
                }
                break;

            default:
                fprintf(stderr,
                        "StringBuffer::fmt_cuda(): encountered unsupported "
                        "character \"$%c\" in format string!\n", p[-1]);
                abort();
        }
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

        switch (*p++) {
            case 'u': cur = w_u32(cur, va_arg(args2, uint32_t)); break;

            case 's': {
                    const char *s = va_arg(args2, const char *);
                    m_cur = cur;
                    put(s, strlen(s));
                    // '$s' is unbounded: put() may have grown the buffer, so
                    // re-reserve the headroom the unchecked writes below assume.
                    if (unlikely(m_cur + bound >= m_end))
                        expand(bound);
                    cur = m_cur;
                }
                break;

            case 't': {
                    const Variable *v = va_arg(args2, const Variable *);
                    cur = w_str(cur, type_name_metal[v->type]);
                }
                break;

            case 'b': {
                    const Variable *v = va_arg(args2, const Variable *);
                    cur = w_str(cur, type_name_metal_bin[v->type]);
                }
                break;

            case 'v': {
                    const Variable *v = va_arg(args2, const Variable *);
                    *cur++ = 'r';
                    cur = w_u32(cur, v->reg_index);
                }
                break;

            case 'l': {
                    const Variable *v = va_arg(args2, const Variable *);
                    *cur++ = '0';
                    *cur++ = 'x';
                    cur = w_x64(cur, v->literal);
                }
                break;

            case 'o': {
                    const Variable *v = va_arg(args2, const Variable *);
                    cur = w_u32(cur, v->param_offset / (uint32_t) sizeof(void *) - 1);
                }
                break;

            default:
                fprintf(stderr,
                        "StringBuffer::fmt_metal(): encountered unsupported "
                        "character \"$%c\" in format string!\n", p[-1]);
                abort();
        }
    }
    va_end(args2);

    m_cur = cur;
    *m_cur = '\0';
}
#endif
