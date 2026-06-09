#include "strbuf.h"
#include "internal.h"
#include "var.h"
#include "eval.h"
#include <cstdarg>

/// Global string buffer used to generate PTX/LLVM IR or MSL
StringBuffer buffer { 1024 };

// ============================================================================
//  Free cursor-passing helpers used by the formatters. The w_*() functions
//  write to a caller-held cursor and returns its new position.
// ============================================================================

// ============================================================================
//  Decimal integer formatting
// ============================================================================

// Fixed-point fractional masks used by the division-free integer formatter.
static const uint64_t mask24 = (1ull << 24) - 1,
                      mask32 = (1ull << 32) - 1,
                      mask57 = (1ull << 57) - 1;

// Two-digit lookup table for jeaiii unsigned->decimal conversion
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

// Version with doubled leading digits for values < 10
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

// Division-free unsigned to decimal conversion based on James Anhalt ("jeaiii",
// https://github.com/jeaiii/itoa). A magnitude cascade produces a scaled
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

// Write 'z' in [0, 1e8) as eight zero-padded digits
static inline void eight_digits(char *b, uint32_t z) {
    uint64_t f0 = ((uint64_t) ((1ull << 48) / 1e6 + 1) * z >> 16) + 1;
    memcpy(b, &digit_pairs[(f0 >> 32) * 2], 2);
    uint64_t f2 = (f0 & mask32) * 100; memcpy(b + 2, &digit_pairs[(f2 >> 32) * 2], 2);
    uint64_t f4 = (f2 & mask32) * 100; memcpy(b + 4, &digit_pairs[(f4 >> 32) * 2], 2);
    uint64_t f6 = (f4 & mask32) * 100; memcpy(b + 6, &digit_pairs[(f6 >> 32) * 2], 2);
}

// 64-bit unsigned decimal to string conversion
static inline char *w_u64(char *b, uint64_t value) {
    if (value <= 0xFFFFFFFFull)
        return w_u32(b, (uint32_t) value);
    uint32_t z = (uint32_t) (value % 100000000u);
    b = w_u64(b, value / 100000000u);
    eight_digits(b, z);
    return b + 8;
}

// ============================================================================
//  Hexadecimal integer formatting
// ============================================================================

static const char num[] = "0123456789abcdef";

// Byte -> two lowercase hex chars, for two-digit-at-a-time hex conversion.
static const char hex_pairs[513] =
    "000102030405060708090a0b0c0d0e0f"
    "101112131415161718191a1b1c1d1e1f"
    "202122232425262728292a2b2c2d2e2f"
    "303132333435363738393a3b3c3d3e3f"
    "404142434445464748494a4b4c4d4e4f"
    "505152535455565758595a5b5c5d5e5f"
    "606162636465666768696a6b6c6d6e6f"
    "707172737475767778797a7b7c7d7e7f"
    "808182838485868788898a8b8c8d8e8f"
    "909192939495969798999a9b9c9d9e9f"
    "a0a1a2a3a4a5a6a7a8a9aaabacadaeaf"
    "b0b1b2b3b4b5b6b7b8b9babbbcbdbebf"
    "c0c1c2c3c4c5c6c7c8c9cacbcccdcecf"
    "d0d1d2d3d4d5d6d7d8d9dadbdcdddedf"
    "e0e1e2e3e4e5e6e7e8e9eaebecedeeef"
    "f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff";

// Number of hex digits needed to represent a value.
static inline unsigned hex_digits(uint64_t v) {
#if defined(_MSC_VER)
    unsigned long i;
    if (!_BitScanReverse64(&i, v)) return 1;
    return ((unsigned) i >> 2) + 1;
#else
    return v ? ((64u - (unsigned) __builtin_clzll(v) + 3) >> 2) : 1;
#endif
}

// 64-bit unsigned to hexadecimal string conversion
static inline char *w_x64(char *d, uint64_t value) {
    char *end = d + hex_digits(value), *p = end;
    while (value >= 0x100) {
        unsigned b = (unsigned) (value & 0xff);
        p -= 2;
        memcpy(p, &hex_pairs[b * 2], 2);
        value >>= 8;
    }
    if (value < 0x10)
        *--p = num[(unsigned) value];
    else
        memcpy(p - 2, &hex_pairs[(unsigned) value * 2], 2);
    return end;
}

// 32-bit unsigned to hexadecimal string conversion
static inline char *w_x32(char *d, uint32_t value) {
    return w_x64(d, value); // fall back to 64 case
}

// 64-bit unsigned to hexadecimal string conversion with zero padding (16 digits)
static inline char *w_q64(char *d, uint64_t value) {
    for (int i = 7; i >= 0; --i) {
        unsigned b = (unsigned) (value & 0xff);
        memcpy(d + i * 2, &hex_pairs[b * 2], 2);
        value >>= 8;
    }
    return d + 16;
}

// ============================================================================
//  String copying
// ============================================================================

// Append a short string of unknown length
static inline char *w_str(char *d, const char *str) {
    char c;
    while ((c = *str++) != '\0')
        *d++ = c;
    return d;
}

// Copy a string of known length through the cursor.
static inline char *w_strn(char *d, const char *str, size_t n) {
    memcpy(d, str, n);
    return d + n;
}

// Copy plain characters from 'p' up to the next '$' (or 'end'),
// advancing 8 bytes at a time using a SWAR test when possible.
static inline char *w_run(char *d, const char *&p, const char *end) {
    const uint64_t dollar = 0x2424242424242424ull; // '$' broadcast
    while (end - p >= 8) {
        uint64_t w; memcpy(&w, p, 8);
        memcpy(d, &w, 8);
        uint64_t x = w ^ dollar;
        uint64_t m = (x - 0x0101010101010101ull) & ~x & 0x8080808080808080ull;
        if (m) {
            unsigned n = (unsigned) (__builtin_ctzll(m) >> 3);
            p += n;
            return d + n;
        }
        p += 8; d += 8;
    }
    while (p < end && *p != '$')
        *d++ = *p++;
    return d;
}

// ============================================================================
//  Output-size bound for the single-pass formatters
// ============================================================================

// Worst-case output length of a format spanning 'fmt_len' characters and
// consuming 'nargs' arguments.
//
// Each format character expands to at most 8 bytes of fixed directive text; the
// binding case is the widest zero-argument directive, '$z' = "zeroinitializer"
// (15 bytes from 2 characters, and 2*8 >= 15). Each argument expands to at most
// 48 bytes (the widest, '$V', needs ~35). The bound is deliberately loose:
// over-reserving is free, undercounting would overflow the unchecked writes.
//
// The size of an arbitrary string insertion ('$s') cannot be upper-bounded,
// so this calculation must be revisited when one is encountered.
static inline size_t fmt_bound(size_t fmt_len, size_t nargs) {
    return fmt_len * 8 + nargs * 48;
}

// ============================================================================
//  StringBuffer methods
// ============================================================================

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

char *StringBuffer::expand(char *cur, size_t nbytes) {
    m_cur = cur;

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

    return m_cur;
}

void StringBuffer::put_u32(uint32_t value) {
    if (unlikely(!m_cur || m_cur + MAXSIZE_U32 >= m_end))
        m_cur = expand(m_cur, MAXSIZE_U32);
    put_u32_unchecked(value);
}

void StringBuffer::put_u64(uint64_t value) {
    if (unlikely(!m_cur || m_cur + MAXSIZE_U64 >= m_end))
        m_cur = expand(m_cur, MAXSIZE_U64);
    put_u64_unchecked(value);
}

void StringBuffer::put_x32(uint32_t value) {
    if (unlikely(!m_cur || m_cur + MAXSIZE_X32 >= m_end))
        m_cur = expand(m_cur, MAXSIZE_X32);
    put_x32_unchecked(value);
}

void StringBuffer::put_x64(uint64_t value) {
    if (unlikely(!m_cur || m_cur + MAXSIZE_X64 >= m_end))
        m_cur = expand(m_cur, MAXSIZE_X64);
    put_x64_unchecked(value);
}

void StringBuffer::put_u32_unchecked(uint32_t value) { m_cur = w_u32(m_cur, value); }

void StringBuffer::put_u64_unchecked(uint64_t value) { m_cur = w_u64(m_cur, value); }

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
            m_cur = expand(m_cur, (size_t) rv);
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
            m_cur = expand(m_cur, (size_t) rv);
        }
    } while (true);
}

void StringBuffer::fmt_llvm(size_t nargs, size_t fmt_len, const char *fmt, ...) {
    va_list args2;
    va_start(args2, fmt);

    // Bound the maximum output size needed by the format string and arguments
    size_t bound = fmt_bound(fmt_len, nargs);
    if (unlikely(!m_cur || m_cur + bound >= m_end))
        m_cur = expand(m_cur, bound);

    const char *p = fmt, *fmt_end = fmt + fmt_len;
    char *cur = m_cur;
    size_t arg = 0;
    while (p < fmt_end) {
        cur = w_run(cur, p, fmt_end);
        if (p == fmt_end)
            break;
        ++p; // consume '$'

        switch (*p++) {
            case 'u': cur = w_u32(cur, va_arg(args2, uint32_t)); break;

            case 's': {
                    const char *s = va_arg(args2, const char *);
                    // Inserting an arbitrary string might break the previous upper bound.
                    // Re-bound the maximum output size of the remaining fmt string and
                    // argument count and potentially allocate additional memory.
                    size_t rest = fmt_bound((size_t) (fmt_end - p), nargs - arg - 1);
                    size_t len = strlen(s);
                    if (unlikely(cur + len + rest >= m_end))
                        cur = expand(cur, len + rest);
                    cur = w_strn(cur, s, len);
                }
                break;

            case 'w':
                cur = w_u32(cur, jitc_llvm_vector_width);
                --arg;
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
                }
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
                --arg;
                break;

            case '<':
                if (callable_depth > 0) {
                    *cur++ = '<';
                    cur = w_u32(cur, jitc_llvm_vector_width);
                    *cur++ = ' '; *cur++ = 'x'; *cur++ = ' ';
                }
                --arg;
                break;

            case '>':
                if (callable_depth > 0)
                    *cur++ = '>';
                --arg;
                break;

            default:
                fprintf(stderr,
                        "StringBuffer::fmt_llvm(): encountered unsupported "
                        "character \"$%c\" in format string!\n", p[-1]);
                abort();
        }

        ++arg;
    }
    va_end(args2);

    if (unlikely(arg != nargs)) {
        fprintf(stderr,
                "StringBuffer::fmt_llvm(): given %zu args, format string "
                "accessed %zu (%s)\n", nargs, arg, fmt);
        abort();
    }

    *cur = '\0';
    m_cur = cur;
}

#if defined(DRJIT_ENABLE_CUDA)
void StringBuffer::fmt_cuda(size_t nargs, size_t fmt_len, const char *fmt, ...) {
    va_list args2;
    va_start(args2, fmt);

    // Bound the maximum output size needed by the format string and arguments
    size_t bound = fmt_bound(fmt_len, nargs);
    if (unlikely(!m_cur || m_cur + bound >= m_end))
        m_cur = expand(m_cur, bound);

    const char *p = fmt, *fmt_end = fmt + fmt_len;
    char *cur = m_cur;
    size_t arg = 0;
    while (p < fmt_end) {
        cur = w_run(cur, p, fmt_end);
        if (p == fmt_end)
            break;
        ++p; // consume '$'

        switch (*p++) {
            case 'u': cur = w_u32(cur, va_arg(args2, uint32_t)); break;

            case 'Q': cur = w_q64(cur, va_arg(args2, uint64_t)); break;

            case 's': {
                    const char *s = va_arg(args2, const char *);
                    // Inserting an arbitrary string might break the previous upper bound.
                    // Re-bound the maximum output size of the remaining fmt string and
                    // argument count and potentially allocate additional memory.
                    size_t rest = fmt_bound((size_t) (fmt_end - p), nargs - arg - 1);
                    size_t len = strlen(s);
                    if (unlikely(cur + len + rest >= m_end))
                        cur = expand(cur, len + rest);
                    cur = w_strn(cur, s, len);
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
                }
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

        ++arg;
    }
    va_end(args2);

    if (unlikely(arg != nargs)) {
        fprintf(stderr,
                "StringBuffer::fmt_cuda(): given %zu args, format string "
                "accessed %zu (%s)\n", nargs, arg, fmt);
        abort();
    }

    *cur = '\0';
    m_cur = cur;
}
#endif

#if defined(DRJIT_ENABLE_METAL)
void StringBuffer::fmt_metal(size_t nargs, size_t fmt_len, const char *fmt, ...) {
    va_list args2;
    va_start(args2, fmt);

    // Bound the maximum output size needed by the format string and arguments
    size_t bound = fmt_bound(fmt_len, nargs);
    if (unlikely(!m_cur || m_cur + bound >= m_end))
        m_cur = expand(m_cur, bound);

    const char *p = fmt, *fmt_end = fmt + fmt_len;
    char *cur = m_cur;
    size_t arg = 0;
    while (p < fmt_end) {
        cur = w_run(cur, p, fmt_end);
        if (p == fmt_end)
            break;
        ++p; // consume '$'

        switch (*p++) {
            case 'u': cur = w_u32(cur, va_arg(args2, uint32_t)); break;

            case 's': {
                    const char *s = va_arg(args2, const char *);
                    // Inserting an arbitrary string might break the previous upper bound.
                    // Re-bound the maximum output size of the remaining fmt string and
                    // argument count and potentially allocate additional memory.
                    size_t rest = fmt_bound((size_t) (fmt_end - p), nargs - arg - 1);
                    size_t len = strlen(s);
                    if (unlikely(cur + len + rest >= m_end))
                        cur = expand(cur, len + rest);
                    cur = w_strn(cur, s, len);
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

        ++arg;
    }
    va_end(args2);

    if (unlikely(arg != nargs)) {
        fprintf(stderr,
                "StringBuffer::fmt_metal(): given %zu args, format string "
                "accessed %zu (%s)\n", nargs, arg, fmt);
        abort();
    }

    *cur = '\0';
    m_cur = cur;
}
#endif
