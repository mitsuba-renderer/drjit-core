#pragma once

#include <enoki/jit.h>

/// Evaluate all computation that is queued on the current stream
extern void jit_eval();
