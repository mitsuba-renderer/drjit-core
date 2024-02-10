/*
    drjit-core/macros.h -- Macro delarations used in various header files

    Dr.Jit is a C++ template library for efficient vectorization and
    differentiation of numerical kernels on modern processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#if defined(_MSC_VER)
#  if defined(DRJIT_BUILD)
#    define JIT_EXPORT    __declspec(dllexport)
#  else
#    define JIT_EXPORT    __declspec(dllimport)
#  endif
#  define JIT_MALLOC
#  define JIT_INLINE    __forceinline
#  define JIT_NOINLINE  __declspec(noinline)
#  define JIT_NORETURN_FORMAT
#  define JIT_NO_UBSAN
#  define JIT_BUILTIN(name) ::name
#else
#  define JIT_EXPORT    __attribute__ ((visibility("default")))
#  define JIT_MALLOC    __attribute__((malloc))
#  define JIT_INLINE    __attribute__ ((always_inline)) inline
#  define JIT_NOINLINE  __attribute__ ((noinline))
#  define JIT_NORETURN_FORMAT __attribute__((noreturn, __format__ (__printf__, 1, 2)))
#  define JIT_NO_UBSAN __attribute__ ((no_sanitize("undefined")))
#  define JIT_BUILTIN(name) ::__builtin_##name
#endif

#if defined(__cplusplus)
#  define JIT_CONSTEXPR constexpr
#  define JIT_DEF(x) = x
#  define JIT_NOEXCEPT noexcept
#  define JIT_ENUM ::
#  if !defined(NAMESPACE_BEGIN)
#    define NAMESPACE_BEGIN(name) namespace name {
#  endif
#  if !defined(NAMESPACE_END)
#    define NAMESPACE_END(name) }
#  endif
#else
#  define JIT_CONSTEXPR inline
#  define JIT_DEF(x)
#  define JIT_NOEXCEPT
#  define JIT_ENUM enum
#endif
