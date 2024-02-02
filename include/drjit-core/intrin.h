/*
    drjit-core/intrin.h -- Include intrinsics needed by Dr.Jit-Core/Dr.Jit

    Copyright (c) 2024 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#if defined(_MSC_VER)
#  include <intrin.h>
#else
#  if defined(__aarch64__)
#    include <arm_neon.h>
#    include <arm_fp16.h>
#  elif !defined(DRJIT_DISABLE_AVX512) || !defined(__AVX512F__)
#    include <immintrin.h>
#  else
//   Optional hack for GCC and Clang: only import headers up to AVX2+ but
//   exclude AVX512.
//
//   This is useful for projects that don't use AVX512 intrinsics but still
//   wish to benefit from the higher register count and instruction encoding.
//   The full set of AVX512 intrinsics pulls in a truly terrifying amount of
//   header code: 1.7 MiB on Clang and 1.2 MiB on GCC, whereas the alternative
//   below only needs 303 KiB on Clang and 237 KiB on GCC.
//
//   GCC/Clang really don't want us to include subsets of the intrinsics and
//   even try to detect this to error out. The code below represents the next
//   step of escalation in this game: we mess up their detection macros so that
//   these checks no longer work. Take that!
//
#    if !defined(_IMMINTRIN_H_INCLUDED)
#      define DR_IMMINTRIN_H_INCLUDED
#      define _IMMINTRIN_H_INCLUDED
#    endif
#    if !defined(__IMMINTRIN_H)
#      define DR_IMMINTRIN_H
#      define __IMMINTRIN_H
#    endif

#    include <nmmintrin.h>
#    include <avxintrin.h>
#    include <avx2intrin.h>
#    include <fmaintrin.h>
#    include <f16cintrin.h>

#    if !defined(DR_IMMINTRIN_H_INCLUDED)
#      undef DR_IMMINTRIN_H_INCLUDED
#      undef _IMMINTRIN_H_INCLUDED
#    endif
#    if !defined(DR_IMMINTRIN_H)
#      undef DR_IMMINTRIN_H
#      undef __IMMINTRIN_H
#    endif
#  endif
#endif

