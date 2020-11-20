#include <atomic>
#include <cstring>

template <typename Target, typename Source>
Target memcpy_cast(const Source &source) {
    static_assert(sizeof(Source) == sizeof(Target),
                  "memcpy_cast: sizes did not match!");
    Target target;
    memcpy(&target, &source, sizeof(Target));
    return target;
}

typedef float    f32_1  __attribute__((__vector_size__(16),  __aligned__(16)));
typedef uint32_t u32_1  __attribute__((__vector_size__(16),  __aligned__(16)));
typedef double   f64_1  __attribute__((__vector_size__(32),  __aligned__(32)));
typedef uint64_t u64_1  __attribute__((__vector_size__(32),  __aligned__(32)));

#define VARIANT(type, type1, repr, suffix)                                     \
    __attribute__((always_inline)) void scatter_add_##suffix(                  \
        void *ptr, type1 value, u32_1 index, bool mask) {                      \
        if (mask) {                                                            \
            repr *target = (repr *) ptr + index[0];                            \
            repr expected = __atomic_load_n(target, __ATOMIC_ACQUIRE),         \
                 desired;                                                      \
            do {                                                               \
                desired =                                                      \
                    memcpy_cast<repr>(memcpy_cast<type>(expected) + value[0]); \
                if (desired == expected)                                       \
                    break;                                                     \
            } while (!__atomic_compare_exchange_n(target, &expected, desired,  \
                                                  true, __ATOMIC_RELEASE,      \
                                                  __ATOMIC_RELAXED));          \
        }                                                                      \
    }

extern "C" {

VARIANT(float, f32_1, uint32_t, f32)
VARIANT(double, f64_1, uint64_t, f64)

VARIANT(uint32_t, u32_1, uint32_t, u32)
VARIANT(uint64_t, u64_1, uint64_t, u64)

};
