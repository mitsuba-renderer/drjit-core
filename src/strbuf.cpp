#include "strbuf.h"
#include "internal.h"
#include "var.h"
#include "eval.h"
#include <cstdarg>

/// Global string buffer used to generate PTX/LLVM IR
StringBuffer buffer { 1024 };

static const char num[] = "0123456789abcdef";

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
    char *start = m_cur;
    do {
        *m_cur++ = num[value % 10];
        value /= 10;
    } while (value);

    reverse_str(start, m_cur);
}

void StringBuffer::put_u64_unchecked(uint64_t value) {
    char *start = m_cur;
    do {
        *m_cur++ = num[value % 10];
        value /= 10;
    } while (value);

    reverse_str(start, m_cur);
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

void StringBuffer::fmt_cuda(size_t nargs, const char *fmt, ...) {
    size_t len = 0;

    va_list args, args2;
    va_start(args, fmt);
    va_copy(args2, args);

    // Phase 1: walk through the string and determine its maximal length
    const char *p = fmt;
    size_t arg = 0;
    char c;
    while ((c = *p++) != '\0') {
        if (c != '$') {
            len++;
        } else {
            c = *p++;
            switch (c) {
                case 'u': len += MAXSIZE_U32; (void) va_arg(args, uint32_t); arg++; break;
                case 'U': len += MAXSIZE_U64; (void) va_arg(args, uint64_t); arg++; break;
                case 'x': len += MAXSIZE_X32; (void) va_arg(args, uint32_t); arg++; break;
                case 'Q':
                case 'X': len += MAXSIZE_X64; (void) va_arg(args, uint64_t); arg++; break;

                case 'c': len++; (void) va_arg(args, int); arg++; break;

                case 's':
                    len += strlen(va_arg(args, const char *));
                    arg++;
                    break;

                case 'b':
                case 't':
                    (void) va_arg(args, const Variable *); arg++;
                    len += MAXSIZE_TYPE;
                    break;

                case 'v':
                    (void) va_arg(args, const Variable *); arg++;
                    len += MAXSIZE_U32 + MAXSIZE_TYPE_PREFIX;
                    break;

                case 'a':
                    (void) va_arg(args, const Variable *); arg++;
                    len += MAXSIZE_U32;
                    break;

                case 'l':
                    (void) va_arg(args, const Variable *); arg++;
                    len += MAXSIZE_X64 + 2;
                    break;

                case 'o':
                    (void) va_arg(args, const Variable *); arg++;
                    len += MAXSIZE_U32;
                    break;

                default:
                    fprintf(stderr,
                            "StringBuffer::fmt_cuda(): encountered unsupported "
                            "character \"$%c\" in format string!\n", c);
                    abort();
            }
        }
    }

    va_end(args);

    if (nargs != arg) {
        fprintf(stderr,
                "StringBuffer::fmt_cuda(): given %zu args, format string "
                "accessed %zu (%s)\n", nargs, arg, fmt);
        abort();
    }

    // Enlarge the buffer if necessary
    if (unlikely(!m_cur || m_cur + len >= m_end))
        expand(len);

    // Phase 2: convert the string
    p = fmt;
    while ((c = *p++) != '\0') {
        if (c == '$') {
            switch (*p++) {
                case 'u': put_u32_unchecked(va_arg(args2, uint32_t)); break;
                case 'U': put_u64_unchecked(va_arg(args2, uint64_t)); break;
                case 'x': put_x32_unchecked(va_arg(args2, uint32_t)); break;
                case 'X': put_x64_unchecked(va_arg(args2, uint64_t)); break;
                case 'Q': put_q64_unchecked(va_arg(args2, uint64_t)); break;

                case 'c': *m_cur++ = (char) va_arg(args2, int); break;

                case 's': {
                        const char *s = va_arg(args2, const char *);
                        put(s, strlen(s));
                    }
                    break;

                case 't': {
                        const Variable *v = va_arg(args2, const Variable *);
                        put_unchecked(type_name_ptx[v->type]);
                    }
                    break;

                case 'b': {
                        const Variable *v = va_arg(args2, const Variable *);
                        put_unchecked(type_name_ptx_bin[v->type]);
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
                    break;
            }
        } else {
            *m_cur ++= c;
        }
    }
    va_end(args2);

    *m_cur = '\0';
}

void StringBuffer::fmt_llvm(size_t nargs, const char *fmt, ...) {
    size_t len = 0;

    va_list args, args2;
    va_start(args, fmt);
    va_copy(args2, args);

    // Phase 1: walk through the string and determine its maximal length
    const char *p = fmt;
    size_t arg = 0;
    char c;
    while ((c = *p++) != '\0') {
        if (c != '$') {
            len++;
        } else {
            c = *p++;
            switch (c) {
                case '{':
                case '}': len += 1; break;

                case 'u': len += MAXSIZE_U32; (void) va_arg(args, uint32_t); arg++; break;
                case 'U': len += MAXSIZE_U64; (void) va_arg(args, uint64_t); arg++; break;
                case 'x': len += MAXSIZE_X32; (void) va_arg(args, uint32_t); arg++; break;
                case 'Q':
                case 'X': len += MAXSIZE_X64; (void) va_arg(args, uint64_t); arg++; break;

                case 's':
                    len += strlen(va_arg(args, const char *));
                    arg++;
                    break;

                case 'w': len += MAXSIZE_U32; break;

                case 't':
                case 'd':
                case 'b':
                case 'm':
                    (void) va_arg(args, const Variable *); arg++;
                    len += MAXSIZE_TYPE;
                    break;

                case 'h':
                    (void) va_arg(args, const Variable *); arg++;
                    len += MAXSIZE_TYPE_ABBREV;
                    break;

                case 'T':
                case 'D':
                case 'B':
                case 'M':
                    (void) va_arg(args, const Variable *); arg++;
                    len += MAXSIZE_U32 + MAXSIZE_TYPE + 5;
                    break;

                case 'v':
                    (void) va_arg(args, const Variable *); arg++;
                    len += MAXSIZE_U32 + MAXSIZE_TYPE_PREFIX;
                    break;

                case 'V':
                    (void) va_arg(args, const Variable *); arg++;
                    len += 2*MAXSIZE_U32 + MAXSIZE_TYPE_PREFIX + 6;
                    break;

                case 'l':
                    (void) va_arg(args, const Variable *); arg++;
                    len += MAXSIZE_X64 + 2;
                    break;

                case 'a':
                case 'A':
                    (void) va_arg(args, const Variable *); arg++;
                    len += MAXSIZE_U32;
                    break;

                case 'o':
                    (void) va_arg(args, const Variable *); arg++;
                    len += MAXSIZE_U32;
                    break;

                case 'z':
                    len += 15;
                    break;

                case '<':
                    len += MAXSIZE_U32 + 4;
                    break;

                case '>':
                    len += 1;
                    break;

                default:
                    fprintf(stderr,
                            "StringBuffer::fmt_llvm(): encountered unsupported "
                            "character \"$%c\" in format string!\n", c);
                    abort();
            }
        }
    }

    va_end(args);

    if (nargs != arg) {
        fprintf(stderr,
                "StringBuffer::fmt_llvm(): given %zu args, format string "
                "accessed %zu (%s)\n", nargs, arg, fmt);
        abort();
    }

    // Enlarge the buffer if necessary
    if (unlikely(!m_cur || m_cur + len >= m_end))
        expand(len);

    // Phase 2: convert the string
    size_t offset = 0;
    p = fmt;
    while ((c = *p++) != '\0') {
        if (c == '$') {
            switch (*p++) {
                case '{': *m_cur++= '{'; break;
                case '}': *m_cur++= '}'; break;

                case 'u': put_u32_unchecked(va_arg(args2, uint32_t)); break;
                case 'U': put_u64_unchecked(va_arg(args2, uint64_t)); break;
                case 'x': put_x32_unchecked(va_arg(args2, uint32_t)); break;
                case 'X': put_x64_unchecked(va_arg(args2, uint64_t)); break;
                case 'Q': put_q64_unchecked(va_arg(args2, uint64_t)); break;

                case 's': {
                        const char *s = va_arg(args2, const char *);
                        put(s, strlen(s));
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
                    break;
            }
        } else if (c == '{') {
            if (jitc_llvm_opaque_pointers)
                offset = size();
        } else if (c == '|') {
            if (offset) {
                rewind_to(offset);
                offset = 0;
            } else {
                offset = size();
            }
        } else if (c == '}') {
            if (offset) {
                rewind_to(offset);
                if (jitc_llvm_opaque_pointers)
                    put_unchecked("ptr");
                offset = 0;
            }
        } else {
            *m_cur ++= c;
        }
    }
    va_end(args2);

    *m_cur = '\0';
}
