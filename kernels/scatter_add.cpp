#include <immintrin.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#define unlikely(x)            __builtin_expect(!!(x), 0)
#define always_inline          __attribute__((always_inline))

typedef float     f32_1  __attribute__((__vector_size__(16),  __aligned__(16)));
typedef int       i32_1  __attribute__((__vector_size__(16),  __aligned__(16)));
typedef double    f64_1  __attribute__((__vector_size__(32),  __aligned__(32)));
typedef long      i64_1  __attribute__((__vector_size__(32),  __aligned__(32)));
typedef float     f32_16 __attribute__((__vector_size__(64), __aligned__(64)));
typedef int       i32_16 __attribute__((__vector_size__(64), __aligned__(64)));
typedef double    f64_8  __attribute__((__vector_size__(64), __aligned__(64)));
typedef long      i64_8  __attribute__((__vector_size__(64), __aligned__(64)));

// #define _kortestz_mask8_u8(x, y) (x==0)

extern "C" {

always_inline
void ek_scatter_add_v1f32(void *ptr, f32_1 value, i32_1 index) {
    ((float *) ptr)[index[3]] += value[3];
}

always_inline
void ek_masked_scatter_add_v1f32(void *ptr, f32_1 value, i32_1 index, bool mask) {
    if (mask)
        ((float *) ptr)[index[3]] += value[3];
}

always_inline
void ek_scatter_add_v1i32(void *ptr, i32_1 value, i32_1 index) {
    ((int *) ptr)[index[3]] += value[3];
}

always_inline
void ek_masked_scatter_add_v1i32(void *ptr, i32_1 value, i32_1 index, bool mask) {
    if (mask)
        ((int *) ptr)[index[3]] += value[3];
}

always_inline
void ek_scatter_add_v1f64(void *ptr, f64_1 value, i64_1 index) {
    ((double *) ptr)[index[3]] += value[3];
}

always_inline
void ek_masked_scatter_add_v1f64(void *ptr, f64_1 value, i64_1 index, bool mask) {
    if (mask)
        ((double *) ptr)[index[3]] += value[3];
}

always_inline
void ek_scatter_add_v1i64(void *ptr, i64_1 value, i64_1 index) {
    ((long *) ptr)[index[3]] += value[3];
}

always_inline
void ek_masked_scatter_add_v1i64(void *ptr, i64_1 value, i64_1 index, bool mask) {
    if (mask)
        ((long *) ptr)[index[3]] += value[3];
}

always_inline
void ek_scatter_add_v16f32(void *ptr, f32_16 value, i32_16 index) {
    f32_16 value_orig = _mm512_mask_i32gather_ps(_mm512_undefined(), (__mmask16) -1,
                                              index, ptr, sizeof(float));

    i32_16 conflicts = _mm512_conflict_epi32(index);

    __mmask16 todo = _mm512_test_epi32_mask(conflicts, conflicts);

    if (unlikely(!_mm512_kortestz(todo, todo))) {
        i32_16 perm_idx = _mm512_sub_epi32(_mm512_set1_epi32(31),
                                        _mm512_lzcnt_epi32(conflicts)),
               all_ones = _mm512_set1_epi32(-1);
        do {
            f32_16 value_peer = _mm512_maskz_permutexvar_ps(todo, perm_idx, value);
            perm_idx = _mm512_mask_permutexvar_epi32(perm_idx, todo, perm_idx, perm_idx);
            value = _mm512_add_ps(value, value_peer);
            todo = _mm512_cmp_epi32_mask(all_ones, perm_idx, _MM_CMPINT_NE);
        } while (!_mm512_kortestz(todo, todo));
    }

    value = _mm512_add_ps(value, value_orig);

    _mm512_i32scatter_ps(ptr, index, value, sizeof(float));
}

always_inline
void ek_masked_scatter_add_v16f32(void *ptr, f32_16 value, i32_16 index, __mmask16 active) {
    f32_16 value_orig = _mm512_mask_i32gather_ps(_mm512_undefined(), active,
                                              index, ptr, sizeof(float));

    i32_16 conflicts = _mm512_and_si512(_mm512_conflict_epi32(index),
                                     _mm512_broadcastmw_epi32(active));

    __mmask16 todo = _mm512_test_epi32_mask(conflicts, conflicts);

    if (unlikely(!_mm512_kortestz(todo, todo))) {
        i32_16 perm_idx = _mm512_sub_epi32(_mm512_set1_epi32(31),
                                        _mm512_lzcnt_epi32(conflicts)),
               all_ones = _mm512_set1_epi32(-1);
        do {
            f32_16 value_peer = _mm512_maskz_permutexvar_ps(todo, perm_idx, value);
            perm_idx = _mm512_mask_permutexvar_epi32(perm_idx, todo, perm_idx, perm_idx);
            value = _mm512_add_ps(value, value_peer);
            todo = _mm512_mask_cmp_epi32_mask(active, all_ones, perm_idx, _MM_CMPINT_NE);
        } while (!_mm512_kortestz(todo, todo));
    }

    value = _mm512_add_ps(value, value_orig);

    _mm512_mask_i32scatter_ps(ptr, active, index, value, sizeof(float));
}

always_inline
void ek_scatter_add_v16i32(void *ptr, i32_16 value, i32_16 index) {
    i32_16 value_orig = _mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), (__mmask16) -1,
                                                    index, ptr, sizeof(uint32_t));

    i32_16 conflicts = _mm512_conflict_epi32(index);

    __mmask16 todo = _mm512_test_epi32_mask(conflicts, conflicts);

    if (unlikely(!_mm512_kortestz(todo, todo))) {
        i32_16 perm_idx = _mm512_sub_epi32(_mm512_set1_epi32(31),
                                        _mm512_lzcnt_epi32(conflicts)),
               all_ones = _mm512_set1_epi32(-1);
        do {
            i32_16 value_peer = _mm512_maskz_permutexvar_epi32(todo, perm_idx, value);
            perm_idx = _mm512_mask_permutexvar_epi32(perm_idx, todo, perm_idx, perm_idx);
            value = _mm512_add_epi32(value, value_peer);
            todo = _mm512_cmp_epi32_mask(all_ones, perm_idx, _MM_CMPINT_NE);
        } while (!_mm512_kortestz(todo, todo));
    }

    value = _mm512_add_epi32(value, value_orig);

    _mm512_i32scatter_epi32(ptr, index, value, sizeof(uint32_t));
}

always_inline
void ek_masked_scatter_add_v16i32(void *ptr, i32_16 value, i32_16 index, __mmask16 active) {
    i32_16 value_orig = _mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), active,
                                                    index, ptr, sizeof(uint32_t));

    i32_16 conflicts = _mm512_and_si512(_mm512_conflict_epi32(index),
                                     _mm512_broadcastmw_epi32(active));

    __mmask16 todo = _mm512_test_epi32_mask(conflicts, conflicts);

    if (unlikely(!_mm512_kortestz(todo, todo))) {
        i32_16 perm_idx = _mm512_sub_epi32(_mm512_set1_epi32(31),
                                        _mm512_lzcnt_epi32(conflicts)),
               all_ones = _mm512_set1_epi32(-1);
        do {
            i32_16 value_peer = _mm512_maskz_permutexvar_epi32(todo, perm_idx, value);
            perm_idx = _mm512_mask_permutexvar_epi32(perm_idx, todo, perm_idx, perm_idx);
            value = _mm512_add_epi32(value, value_peer);
            todo = _mm512_mask_cmp_epi32_mask(active, all_ones, perm_idx, _MM_CMPINT_NE);
        } while (!_mm512_kortestz(todo, todo));
    }

    value = _mm512_add_epi32(value, value_orig);

    _mm512_mask_i32scatter_epi32(ptr, active, index, value, sizeof(uint32_t));
}

always_inline
void ek_scatter_add_v8f64(void *ptr, f64_8 value, i64_8 index) {
    f64_8 value_orig = _mm512_mask_i64gather_pd(_mm512_undefined_pd(), (__mmask8) -1,
                                                index, ptr, sizeof(double));

    i64_8 conflicts = _mm512_conflict_epi64(index);

    __mmask8 todo = _mm512_test_epi64_mask(conflicts, conflicts);

    if (unlikely(!_kortestz_mask8_u8(todo, todo))) {
        i64_8 perm_idx = _mm512_sub_epi64(_mm512_set1_epi64(63),
                                          _mm512_lzcnt_epi64(conflicts)),
              all_ones = _mm512_set1_epi64(-1);
        do {
            f64_8 value_peer = _mm512_maskz_permutexvar_pd(todo, perm_idx, value);
            perm_idx = _mm512_mask_permutexvar_epi64(perm_idx, todo, perm_idx, perm_idx);
            value = _mm512_add_pd(value, value_peer);
            todo = _mm512_cmp_epi64_mask(all_ones, perm_idx, _MM_CMPINT_NE);
        } while (!_kortestz_mask8_u8(todo, todo));
    }

    value = _mm512_add_pd(value, value_orig);

    _mm512_i64scatter_pd(ptr, index, value, sizeof(double));
}

always_inline
void ek_masked_scatter_add_v8f64(void *ptr, f64_8 value, i64_8 index, __mmask8 active) {
    f64_8 value_orig = _mm512_mask_i64gather_pd(_mm512_undefined_pd(), active,
                                                index, ptr, sizeof(double));

    i64_8 conflicts = _mm512_and_si512(_mm512_conflict_epi64(index),
                                       _mm512_broadcastmb_epi64(active));

    __mmask8 todo = _mm512_test_epi64_mask(conflicts, conflicts);

    if (unlikely(!_kortestz_mask8_u8(todo, todo))) {
        i64_8 perm_idx = _mm512_sub_epi64(_mm512_set1_epi64(63),
                                          _mm512_lzcnt_epi64(conflicts)),
              all_ones = _mm512_set1_epi64(-1);
        do {
            f64_8 value_peer = _mm512_maskz_permutexvar_pd(todo, perm_idx, value);
            perm_idx = _mm512_mask_permutexvar_epi64(perm_idx, todo, perm_idx, perm_idx);
            value = _mm512_add_pd(value, value_peer);
            todo = _mm512_mask_cmp_epi64_mask(active, all_ones, perm_idx, _MM_CMPINT_NE);
        } while (!_kortestz_mask8_u8(todo, todo));
    }

    value = _mm512_add_pd(value, value_orig);

    _mm512_mask_i64scatter_pd(ptr, active, index, value, sizeof(double));
}

always_inline
void ek_scatter_add_v8i64(void *ptr, i64_8 value, i64_8 index) {
    i64_8 value_orig = _mm512_mask_i64gather_epi64(_mm512_undefined_epi32(), (__mmask8) -1,
                                                   index, ptr, sizeof(uint64_t));

    i64_8 conflicts = _mm512_conflict_epi64(index);

    __mmask8 todo = _mm512_test_epi64_mask(conflicts, conflicts);

    if (unlikely(!_kortestz_mask8_u8(todo, todo))) {
        i64_8 perm_idx = _mm512_sub_epi64(_mm512_set1_epi64(63),
                                          _mm512_lzcnt_epi64(conflicts)),
              all_ones = _mm512_set1_epi64(-1);
        do {
            i64_8 value_peer = _mm512_maskz_permutexvar_epi64(todo, perm_idx, value);
            perm_idx = _mm512_mask_permutexvar_epi64(perm_idx, todo, perm_idx, perm_idx);
            value = _mm512_add_epi64(value, value_peer);
            todo = _mm512_cmp_epi64_mask(all_ones, perm_idx, _MM_CMPINT_NE);
        } while (!_kortestz_mask8_u8(todo, todo));
    }

    value = _mm512_add_epi64(value, value_orig);

    _mm512_i64scatter_epi64(ptr, index, value, sizeof(uint64_t));
}

always_inline
void ek_masked_scatter_add_v8i64(void *ptr, i64_8 value, i64_8 index, __mmask8 active) {
    i64_8 value_orig = _mm512_mask_i64gather_epi64(_mm512_undefined_epi32(), active,
                                                   index, ptr, sizeof(uint64_t));

    i64_8 conflicts = _mm512_and_si512(_mm512_conflict_epi64(index),
                                       _mm512_broadcastmb_epi64(active));

    __mmask8 todo = _mm512_test_epi64_mask(conflicts, conflicts);

    if (unlikely(!_kortestz_mask8_u8(todo, todo))) {
        i64_8 perm_idx = _mm512_sub_epi64(_mm512_set1_epi64(63),
                                          _mm512_lzcnt_epi64(conflicts)),
              all_ones = _mm512_set1_epi64(-1);
        do {
            i64_8 value_peer = _mm512_maskz_permutexvar_epi64(todo, perm_idx, value);
            perm_idx = _mm512_mask_permutexvar_epi64(perm_idx, todo, perm_idx, perm_idx);
            value = _mm512_add_epi64(value, value_peer);
            todo = _mm512_mask_cmp_epi64_mask(active, all_ones, perm_idx, _MM_CMPINT_NE);
        } while (!_kortestz_mask8_u8(todo, todo));
    }

    value = _mm512_add_epi64(value, value_orig);

    _mm512_mask_i64scatter_epi64(ptr, active, index, value, sizeof(uint64_t));
}

}

#if 0
int main(int argc, char **argv) {
    {
        uint32_t hist[32], hist_scalar[32];
        memset(hist, 0, sizeof(uint32_t)*32);
        memset(hist_scalar, 0, sizeof(uint32_t)*32);

        for (uint32_t i = 0; i< 100000; ++i) {
            i32_16 idx, value;
            uint16_t mask = rand() & 0xFFFF;

            for (uint32_t i = 0; i < 16; ++i) {
                idx[i] = rand() % 32;
                value[i] = rand() % 5;
                if (mask & (1 << i))
                    hist_scalar[idx[i]] += value[i];
            }


            masked_scatter_add_v16i32(hist, value, idx, mask);
        }

        #pragma nounroll
        for (uint32_t i = 0; i < 16; ++i)
            if (hist[i] != hist_scalar[i])
                printf("hist[%u]=%u vs %u\n", i, hist[i], hist_scalar[i]);
        printf("Test 1 passed.\n");
    }

    {
        uint64_t hist[32], hist_scalar[32];
        memset(hist, 0, sizeof(uint64_t)*32);
        memset(hist_scalar, 0, sizeof(uint64_t)*32);

        for (uint64_t i = 0; i< 100000; ++i) {
            i64_8 idx, value;
            uint16_t mask = rand() & 0xFF;

            for (uint64_t i = 0; i < 8; ++i) {
                idx[i] = rand() % 32;
                value[i] = rand() % 5;
                if (mask & (1 << i))
                    hist_scalar[idx[i]] += value[i];
            }

            masked_scatter_add_v8i64(hist, value, idx, mask);
        }

        #pragma nounroll
        for (uint32_t i = 0; i < 16; ++i)
            if (hist[i] != hist_scalar[i])
                printf("hist[%u]=%li vs %li\n", i,
                       hist[i], hist_scalar[i]);
        printf("Test 2 passed.\n");
    }
}
#endif
