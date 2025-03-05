/*
    src/llvm_coop_vec.h -- LLVM fallback code generation for Cooperative Vectors

    Copyright (c) 2025 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

extern void jitc_llvm_render_coop_vec(const Variable *);
extern void jitc_llvm_render_coop_vec_get(const Variable *v, const Variable *a0);
