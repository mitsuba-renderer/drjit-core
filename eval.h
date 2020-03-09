#pragma once

#include "api.h"

/// Evaluate all computation that is queued on the current stream
extern void jit_eval();

/// Call jit_eval() only if the variable 'index' requires evaluation
extern void jit_eval_var(uint32_t index);
