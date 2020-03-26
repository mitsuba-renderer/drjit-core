#pragma once

#include <cstdlib>

/// Immediately terminate the application due to a fatal internal error
#if defined(__GNUC__)
    __attribute__((noreturn, __format__ (__printf__, 1, 2)))
#endif
extern void jit_fail(const char* fmt, ...);

template <typename T, size_t Align = alignof(T)>
struct aligned_allocator {
public:
    template <typename T2> struct rebind {
        using other = aligned_allocator<T2, Align>;
    };

    using value_type      = T;
    using reference       = T &;
    using const_reference = const T &;
    using pointer         = T *;
    using const_pointer   = const T *;
    using size_type       = size_t;
    using difference_type = ptrdiff_t;

    aligned_allocator() = default;
    aligned_allocator(const aligned_allocator &) = default;

    template <typename T2, size_t Align2>
    aligned_allocator(const aligned_allocator<T2, Align2> &) { }

    value_type *allocate(size_t count) {
        void *ptr;
        if (posix_memalign(&ptr, Align, sizeof(T) * count) != 0)
            jit_fail("aligned_allocator::allocate(): out of memory!");
        return (value_type *) ptr;
    }

    void deallocate(value_type *ptr, size_t) {
        free(ptr);
    }

    template <typename T2, size_t Align2>
    bool operator==(const aligned_allocator<T2, Align2> &) const {
        return Align == Align2;
    }

    template <typename T2, size_t Align2>
    bool operator!=(const aligned_allocator<T2, Align2> &) const {
        return Align != Align2;
    }
};
