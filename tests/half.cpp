#define DRJIT_INCLUDE_FLOAT16_FALLBACK 1

#include <drjit-core/half.h>
#include <stdio.h>

using drjit::half;

template <typename Out, typename In>
static Out memcpy_cast(const In &src) {
    static_assert(sizeof(In) == sizeof(Out), "memcpy_cast: size mismatch!");
    Out dst;
    std::memcpy(&dst, &src, sizeof(Out));
    return dst;
}

int main(int, char **) {
#if !defined(__F16C__) && !defined(__aarch64__)
    fprintf(stderr,
            "Hardware support for half precision arithmetic is not available. "
            "There is nothing to test.\n");
    return 0;
#endif
    for (size_t i = 0; i < 0xFFFF; ++i) {
        half h = half::from_binary((uint16_t) i);
        float f1 = (float) h,
              f2 = half::float16_to_float32_fallback((uint16_t) i);
        bool nan_match = ((i & 0x7c00) == 0x7c00) && std::isnan(f1) && std::isnan(f2);

        if (memcpy_cast<uint32_t>(f1) != memcpy_cast<uint32_t>(f2) && !nan_match) {
            fprintf(stderr, "F16->F32 conversion error: 0x%x, hw=%f, ref=%f\n", (uint32_t) i, f1, f2);
            abort();
        }
    }

    for (size_t i = 0; i < 0xFFFFFFFF; ++i) {
        if ((i & 0xffffff) == 0) {
            printf(".");
            fflush(stdout);
        }

        float f = memcpy_cast<float>((uint32_t) i);
        half h1 = half(f),
             h2 = half::from_binary(half::float32_to_float16_fallback(f));

        bool nan_match = std::isnan(f) && (h1.value & 0x7c00) == 0x7c00 &&
                         (h2.value & 0x7c00) == 0x7c00;

        if (h1.value != h2.value && !nan_match) {
            printf("F32->F16 conversion error: 0x%x (%f), hw=0x%x, ref=0x%x\n", (uint32_t) i, f,
                   h1.value, h2.value);
        }
    }
	printf("\n");
    printf("Tests passed.\n");
    return 0;
}
