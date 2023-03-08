#pragma once
#include "common.h"
#include <cstring>

// Maximum stringified length of a 32 bit integer (decimal)
#define MAXSIZE_U32            10

// Maximum stringified length of a 64 bit integer (decimal)
#define MAXSIZE_U64            20

// Maximum stringified length of a 32 bit integer (hex)
#define MAXSIZE_X32            8

// Maximum stringified length of a 64 bit integer (hex)
#define MAXSIZE_X64            16

// Maximum stringified length of a Dr.Jit type name
#define MAXSIZE_TYPE           6

// Maximum stringified length of an abbreviated Dr.Jit type name
#define MAXSIZE_TYPE_ABBREV    3

// Maximum stringified length of a Dr.Jit type prefix
#define MAXSIZE_TYPE_PREFIX    3

/**
 * \brief String buffer class for building large strings sequentially
 *
 * This class stores a zero-terminated string and provides various convenient
 * methods to append content to its end. In doing so, it significantly
 * outperforms standard string construction utilities such as `std::snprintf`,
 * `std::ostringstream`, and `std::to_string`.
 *
 * The StringBuffer class serves a similar purpose as the popular `fmt`
 * library. Its implementation is smaller and specialized to Dr.Jit, where it
 * is used to assemble large numbers of IR statements referencing registers to
 * build a complete program.
 *
 * A few more details: When the internal storage of a StringBuffer is
 * exhausted, the implementation resizes the buffer to a larger power of two.
 * Reusing a StringBuffer (following a call to ``.clear()``) usually leads to a
 * steady state where the buffer becomes big enough and dynamic memory
 * allocation is no longer needed.
 */
struct StringBuffer {
public:
    /* ================================================================== */

    /// Create an empty StringBuffer with the desired initial capacity
    StringBuffer(size_t capacity = 0);

    /// Release the buffer
    ~StringBuffer();

    /// Move constructor
    StringBuffer(StringBuffer &&);

    /// Move assignment
    StringBuffer &operator=(StringBuffer &&);

    // Deleted copy constructor and copy assignment operator
    StringBuffer(const StringBuffer &) = delete;
    StringBuffer &operator=(const StringBuffer &) = delete;

    /* ================================================================== */

    /**
     * \brief Clear the StringBuffer, but don't release any memory
     *
     * The operation runs in constant time.
     */
    void clear();

    /**
     * \brief Remove the last \c n characters from the StringBuffer
     *
     * Values of \c n that are too large and would cause the StringBuffer to
     * rewind beyond its start are safely handled by rewinding to the
     * beginning.
     *
     * The operation runs in constant time.
     */
    void rewind_to(size_t pos);

    /// Move region `suffix_start .. size()-1` to position `suffix_target`
    void move_suffix(size_t suffix_start, size_t suffix_target);

    /// Swap the internal representation of two StringBuffer instances. O(1).
    void swap(StringBuffer &b);

    /* ================================================================== */

    /// Return the size of the current string excluding the trailing zero byte
    size_t size() const { return m_cur - m_start; }

    /// Return the current storage capacity of the StringBuffer
    size_t capacity() const { return m_end - m_start; }

    /// Provide access to the C-style string
    const char *get() { return m_start; }

    /* ================================================================== */

    /// Append a string, whose size is known at compile time
    template <size_t N> void put(const char (&str)[N]) {
        put(str, N - 1);
    }

    /// Append a string with the specified length
    void put(const char *str, size_t size) {
        if (unlikely(!m_cur || m_cur + size >= m_end))
            expand(size);

        std::memcpy(m_cur, str, size);
        m_cur += size;
        *m_cur = '\0';
    }

    /// Append a single character to the buffer
    void put(char c) {
        if (unlikely(!m_cur || m_cur + 1 >= m_end))
            expand(1);
        *m_cur++ = c;
        *m_cur = '\0';
    }

    /// Append multiple copies of a single character to the buffer
    void put(char c, size_t count) {
        if (unlikely(!m_cur || m_cur + count >= m_end))
            expand(count);

        for (size_t i = 0; i < count; ++i)
            *m_cur++ = c;
        *m_cur = '\0';
    }

    // Append a 32 bit decimal number
    void put_u32(uint32_t value);

    // Append a 64 bit decimal number
    void put_u64(uint64_t value);

    // Append a 32 bit hex number
    void put_x32(uint32_t value);

    // Append a 64 bit hex number
    void put_x64(uint64_t value);

    // Append a zero-filled 64 bit hex number
    void put_q64(uint64_t value);

    /* ================================================================== */

    /**
     * \brief CUDA-specific formatting routine. Its syntax is described at the
     * top of eval_cuda.cpp
     */
    void fmt_cuda(size_t nargs, const char *fmt, ...);

    /**
     * \brief LLVM-specific formatting routine. Its syntax is described at the
     * top of eval_llvm.cpp
     */
    void fmt_llvm(size_t nargs, const char *fmt, ...);

    /**
     * \brief Append a formatted (printf-style) string to the buffer
     *
     * Warning: this goes through `vsnprintf`, which is known to be highly
     * inefficient.
     */
#if defined(__GNUC__)
    __attribute__((__format__ (__printf__, 2, 3)))
#endif
    size_t fmt(const char *fmt, ...);

    /// Like \ref fmt, but specify arguments through a va_list.
    size_t vfmt(const char *fmt, va_list args_);

    /* ===================================================================
        Danger zone: The following functions don't check that the string
        buffer is big enough, and they do not append a trailing zero when
        writing output. They are intended to be used within other custom
        formatting routines.
       =================================================================== */

    // Append a 32 bit decimal number. The caller must check that there is space
    void put_u32_unchecked(uint32_t value);

    // Append a 64 bit decimal number. The caller must check that there is space
    void put_u64_unchecked(uint64_t value);

    // Append a 32 bit hex number. The caller must check that there is space
    void put_x32_unchecked(uint32_t value);

    // Append a 64 bit hex number. The caller must check that there is space
    void put_x64_unchecked(uint64_t value);

    // Append a zero-filled 64 bit hex number. The caller must check that there is space
    void put_q64_unchecked(uint64_t value);

    /// Append a string with the specified length
    void put_unchecked(const char *str);

private:
    /**
     * \brief Potentially expand the size of the StringBuffer so that there is
     * space for at least \c nbytes new bytes, plus a trailing zero.
     */
    void expand(size_t nbytes);

private:
    char *m_start, *m_cur, *m_end;
};

extern StringBuffer buffer;

/// Helper function used to check that fmt_cuda/fmt_llvm process all arguments
template <typename... Ts> constexpr size_t count_args(const Ts &...) {
    return sizeof...(Ts);
}

