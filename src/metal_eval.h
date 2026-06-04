/*
    src/metal_eval.h -- Metal Shading Language code generation macros.

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"

#define fmt(fmt, ...)                                                          \
    buffer.fmt_metal(count_args(__VA_ARGS__), fmt_strlen(fmt), fmt,            \
                     ##__VA_ARGS__)
#define put(...) buffer.put(__VA_ARGS__)
#define fmt_intrinsic(fmt, ...)                                                \
    do {                                                                       \
        size_t tmpoff = buffer.size();                                         \
        buffer.fmt_metal(count_args(__VA_ARGS__), fmt_strlen(fmt), fmt,        \
                         ##__VA_ARGS__);                                       \
        jitc_register_global(buffer.get() + tmpoff);                           \
        buffer.rewind_to(tmpoff);                                              \
    } while (0)
