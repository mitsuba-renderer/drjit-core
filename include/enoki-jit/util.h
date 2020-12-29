/*
    enoki-jit/util.h -- Utility routines shared by llvm.h and cuda.h

    This file provides helper traits that are needed by the C++ array
    wrappers defined in in 'enoki/jitvar.h'.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <enoki-jit/traits.h>
#include <cstring>

NAMESPACE_BEGIN(enoki)

struct void_t { };

/// Reinterpret the binary represesentation of a data type
template<typename Target, typename Source> Target memcpy_cast(const Source &source) {
    static_assert(sizeof(Source) == sizeof(Target), "memcpy_cast: sizes did not match!");
    Target target;
    std::memcpy(&target, &source, sizeof(Target));
    return target;
}

template <typename... Args, enable_if_t<(sizeof...(Args) > 1)> = 0>
void jit_schedule(Args&&... args) {
    bool unused[] = { (jit_schedule(args), false)..., false };
    (void) unused;
}

template <typename... Args, enable_if_t<(sizeof...(Args) > 0)> = 0>
void jit_eval(Args&&... args) {
    jit_schedule(args...);
    if (sizeof...(Args) > 0)
        ::jit_eval();
}

using ssize_t = typename std::make_signed<size_t>::type;

NAMESPACE_END(enoki)
