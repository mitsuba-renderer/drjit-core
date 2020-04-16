/*
    src/eval.h -- Main computation graph evaluation routine

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <enoki-jit/jit.h>

/// Evaluate all computation that is queued on the current stream
extern void jit_eval();
