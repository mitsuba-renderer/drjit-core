#pragma once

#include "internal.h"

struct CUDAGreenContext;

CUDAGreenContext *jitc_cuda_green_context_make(uint32_t sm_count_requested,
                                               uint32_t *sm_count_actual,
                                               void **other_context);

void jitc_cuda_green_context_release(CUDAGreenContext *ctx);

void *jitc_cuda_green_context_enter(CUDAGreenContext *ctx);

void jitc_cuda_green_context_leave(void *token);
