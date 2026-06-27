#pragma once

static const char *jitc_amd_reduce_source = R"drjit_amd_reduce(
// amd-reduce-resource-version: reduced-stage-active-mask
#include "block_reduce.cuh"
#include "block_prefix_reduce.cuh"
#include "reduce_2.cuh"
)drjit_amd_reduce";
