/*
    src/metal_dd_preamble.h -- Double-double (DD) Float64 emulation for Metal

    Copyright (c) 2026 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.

    --------------------------------------------------------------------------

    Apple silicon GPUs lack hardware FP64 and the Metal Shading Language
    rejects the `double` keyword. To support `VarType::Float64` on the Metal
    backend, we represent each Float64 scalar as a pair of float32 values
    (hi, lo) such that the true value equals `hi + lo` and `|lo| <= 0.5 ulp(hi)`.
    This "double-double" (DD) representation gives ~31 decimal digits of
    mantissa precision -- more than IEEE 754 double's ~16 digits -- but does
    NOT extend the exponent range (still ~3.4e38).

    The arithmetic uses the well-known Dekker / Knuth algorithms:
      * TwoSum (Knuth 1969)        -- exact sum of two floats as (hi, lo)
      * FastTwoSum (Dekker 1971)   -- requires |a| >= |b|
      * TwoProductFMA              -- exact product via FMA, free on FP32
    See Hida, Li, Bailey "Library for Double-Double and Quad-Double Arithmetic"
    (2007) for the DD identities used below.

    On-demand emission
    ------------------
    Each helper exposes a `register_dd_<name>()` function that emits its MSL
    body into the kernel's global preamble via `fmt_intrinsic` (deduped by
    content hash). The codegen sites in metal_eval.cpp call these registrars
    before emitting a `dd_<name>(...)` use, so only the helpers a kernel
    actually references land in its source. Each registrar transitively calls
    its dependencies' registrars; `jitc_register_global` collapses repeats.

    Each emitted snippet is self-contained: it begins with `typedef float2
    dd_t;` (legal redeclaration with the same target type per C++11) plus
    forward declarations for every helper its body calls. This keeps every
    snippet valid regardless of the order the global table iterates them.

    Conventions
    -----------
    * `dd_t` is a `typedef float2`. Component `.x` is `hi`, `.y` is `lo`.
    * Helpers are `inline` and take/return `dd_t` by value.
*/

#pragma once

#if defined(DRJIT_ENABLE_METAL)

#include "metal_eval.h"  // fmt_intrinsic
#include "strbuf.h"      // buffer, count_args, StringBuffer
#include "eval.h"        // jitc_register_global

namespace drjit {

// ---------------------------------------------------------------------------
//  Helper: every snippet starts with this header (typedef + per-snippet
//  forward declarations). The typedef is C++11-safe to redeclare since the
//  RHS is identical across all snippets, so it doesn't matter whether the
//  global table iterates a body before or after register_dd_t() lands.
// ---------------------------------------------------------------------------
#define DD_HEADER "typedef float2 dd_t;\n"

// Standalone typedef registration — used by the Float64 input/output paths
// in metal_eval.cpp that emit `dd_t` directly via the `$t` formatter
// without going through any dd_* helper.
static inline void register_dd_t() {
    fmt_intrinsic(DD_HEADER);
}

// ===== Primitives (no dependencies) ========================================

static inline void register_dd_two_sum() {
    fmt_intrinsic(
        DD_HEADER
        "inline dd_t dd_two_sum(float a, float b) {\n"
        "    float s  = a + b;\n"
        "    float bb = s - a;\n"
        "    float e  = (a - (s - bb)) + (b - bb);\n"
        "    return dd_t(s, e);\n"
        "}\n");
}

static inline void register_dd_fast_two_sum() {
    fmt_intrinsic(
        DD_HEADER
        "inline dd_t dd_fast_two_sum(float a, float b) {\n"
        "    float s = a + b;\n"
        "    float e = b - (s - a);\n"
        "    return dd_t(s, e);\n"
        "}\n");
}

static inline void register_dd_two_prod() {
    fmt_intrinsic(
        DD_HEADER
        "inline dd_t dd_two_prod(float a, float b) {\n"
        "    float p = a * b;\n"
        "    float e = fma(a, b, -p);\n"
        "    return dd_t(p, e);\n"
        "}\n");
}

// ===== Conversions (single-line, no DD-helper dependencies) ================

static inline void register_dd_from_f32() {
    fmt_intrinsic(
        DD_HEADER
        "inline dd_t dd_from_f32(float a) { return dd_t(a, 0.0f); }\n");
}

static inline void register_dd_from_half() {
    fmt_intrinsic(
        DD_HEADER
        "inline dd_t dd_from_half(half a) { return dd_t((float) a, 0.0f); }\n");
}

static inline void register_dd_from_i32() {
    fmt_intrinsic(
        DD_HEADER
        "inline dd_t dd_from_i32(int a) { return dd_t((float) a, 0.0f); }\n");
}

static inline void register_dd_from_u32() {
    fmt_intrinsic(
        DD_HEADER
        "inline dd_t dd_from_u32(uint a) {\n"
        "    float hi = (float) a;\n"
        "    float lo = (float) ((long) a - (long) hi);\n"
        "    return dd_t(hi, lo);\n"
        "}\n");
}

static inline void register_dd_from_i64() {
    fmt_intrinsic(
        DD_HEADER
        "inline dd_t dd_from_i64(long a) {\n"
        "    float hi = (float) a;\n"
        "    float lo = (float) (a - (long) hi);\n"
        "    return dd_t(hi, lo);\n"
        "}\n");
}

static inline void register_dd_from_u64() {
    fmt_intrinsic(
        DD_HEADER
        "inline dd_t dd_from_u64(ulong a) {\n"
        "    float hi = (float) a;\n"
        "    long  diff = (long) (a - (ulong) hi);\n"
        "    float lo = (float) diff;\n"
        "    return dd_t(hi, lo);\n"
        "}\n");
}

static inline void register_dd_from_bool() {
    fmt_intrinsic(
        DD_HEADER
        "inline dd_t dd_from_bool(bool b) { return dd_t(b ? 1.0f : 0.0f, 0.0f); }\n");
}

static inline void register_dd_from_bits() {
    fmt_intrinsic(
        DD_HEADER
        "inline dd_t dd_from_bits(uint hi_bits, uint lo_bits) {\n"
        "    return dd_t(as_type<float>(hi_bits), as_type<float>(lo_bits));\n"
        "}\n");
}

static inline void register_dd_to_f32() {
    fmt_intrinsic(
        DD_HEADER
        "inline float dd_to_f32(dd_t a) { return a.x + a.y; }\n");
}

static inline void register_dd_to_half() {
    fmt_intrinsic(
        DD_HEADER
        "inline half dd_to_half(dd_t a) { return (half) (a.x + a.y); }\n");
}

static inline void register_dd_to_i32() {
    fmt_intrinsic(
        DD_HEADER
        "inline int dd_to_i32(dd_t a) { return (int) (a.x + a.y); }\n");
}

static inline void register_dd_to_u32() {
    fmt_intrinsic(
        DD_HEADER
        "inline uint dd_to_u32(dd_t a) { return (uint) (a.x + a.y); }\n");
}

static inline void register_dd_to_i64() {
    fmt_intrinsic(
        DD_HEADER
        "inline long dd_to_i64(dd_t a) { return (long) (a.x + a.y); }\n");
}

static inline void register_dd_to_u64() {
    fmt_intrinsic(
        DD_HEADER
        "inline ulong dd_to_u64(dd_t a) { return (ulong) (a.x + a.y); }\n");
}

static inline void register_dd_to_bool() {
    fmt_intrinsic(
        DD_HEADER
        "inline bool dd_to_bool(dd_t a) { return (a.x + a.y) != 0.0f; }\n");
}

// ===== Sign / abs ==========================================================

static inline void register_dd_neg() {
    fmt_intrinsic(
        DD_HEADER
        "inline dd_t dd_neg(dd_t a) { return dd_t(-a.x, -a.y); }\n");
}

static inline void register_dd_abs() {
    fmt_intrinsic(
        DD_HEADER
        "inline dd_t dd_abs(dd_t a) {\n"
        "    return a.x < 0.0f ? dd_t(-a.x, -a.y) : a;\n"
        "}\n");
}

// ===== Comparisons (operate on hi+lo via lex order on (hi, lo)) ============

static inline void register_dd_eq() {
    fmt_intrinsic(
        DD_HEADER
        "inline bool dd_eq(dd_t a, dd_t b) { return a.x == b.x && a.y == b.y; }\n");
}

static inline void register_dd_ne() {
    fmt_intrinsic(
        DD_HEADER
        "inline bool dd_ne(dd_t a, dd_t b) { return a.x != b.x || a.y != b.y; }\n");
}

static inline void register_dd_lt() {
    fmt_intrinsic(
        DD_HEADER
        "inline bool dd_lt(dd_t a, dd_t b) {\n"
        "    return a.x < b.x || (a.x == b.x && a.y < b.y);\n"
        "}\n");
}

static inline void register_dd_le() {
    fmt_intrinsic(
        DD_HEADER
        "inline bool dd_le(dd_t a, dd_t b) {\n"
        "    return a.x < b.x || (a.x == b.x && a.y <= b.y);\n"
        "}\n");
}

static inline void register_dd_gt() {
    fmt_intrinsic(
        DD_HEADER
        "inline bool dd_gt(dd_t a, dd_t b) {\n"
        "    return a.x > b.x || (a.x == b.x && a.y > b.y);\n"
        "}\n");
}

static inline void register_dd_ge() {
    fmt_intrinsic(
        DD_HEADER
        "inline bool dd_ge(dd_t a, dd_t b) {\n"
        "    return a.x > b.x || (a.x == b.x && a.y >= b.y);\n"
        "}\n");
}

// ===== Min / max (use lt/gt) ==============================================

static inline void register_dd_min() {
    register_dd_lt();
    fmt_intrinsic(
        DD_HEADER
        "inline bool dd_lt(dd_t a, dd_t b);\n"
        "inline dd_t dd_min(dd_t a, dd_t b) { return dd_lt(a, b) ? a : b; }\n");
}

static inline void register_dd_max() {
    register_dd_gt();
    fmt_intrinsic(
        DD_HEADER
        "inline bool dd_gt(dd_t a, dd_t b);\n"
        "inline dd_t dd_max(dd_t a, dd_t b) { return dd_gt(a, b) ? a : b; }\n");
}

// ===== Arithmetic =========================================================

static inline void register_dd_add() {
    register_dd_two_sum();
    register_dd_fast_two_sum();
    fmt_intrinsic(
        DD_HEADER
        "inline dd_t dd_two_sum(float a, float b);\n"
        "inline dd_t dd_fast_two_sum(float a, float b);\n"
        "inline dd_t dd_add(dd_t a, dd_t b) {\n"
        "    dd_t s = dd_two_sum(a.x, b.x);\n"
        "    dd_t t = dd_two_sum(a.y, b.y);\n"
        "    s.y += t.x;\n"
        "    s   = dd_fast_two_sum(s.x, s.y);\n"
        "    s.y += t.y;\n"
        "    s   = dd_fast_two_sum(s.x, s.y);\n"
        "    return s;\n"
        "}\n");
}

static inline void register_dd_sub() {
    register_dd_add();
    register_dd_neg();
    fmt_intrinsic(
        DD_HEADER
        "inline dd_t dd_add(dd_t a, dd_t b);\n"
        "inline dd_t dd_neg(dd_t a);\n"
        "inline dd_t dd_sub(dd_t a, dd_t b) { return dd_add(a, dd_neg(b)); }\n");
}

static inline void register_dd_mul() {
    register_dd_two_prod();
    register_dd_fast_two_sum();
    fmt_intrinsic(
        DD_HEADER
        "inline dd_t dd_two_prod(float a, float b);\n"
        "inline dd_t dd_fast_two_sum(float a, float b);\n"
        "inline dd_t dd_mul(dd_t a, dd_t b) {\n"
        "    dd_t p = dd_two_prod(a.x, b.x);\n"
        "    p.y += a.x * b.y + a.y * b.x;\n"
        "    return dd_fast_two_sum(p.x, p.y);\n"
        "}\n");
}

static inline void register_dd_fma() {
    register_dd_add();
    register_dd_mul();
    fmt_intrinsic(
        DD_HEADER
        "inline dd_t dd_add(dd_t a, dd_t b);\n"
        "inline dd_t dd_mul(dd_t a, dd_t b);\n"
        "inline dd_t dd_fma(dd_t a, dd_t b, dd_t c) {\n"
        "    return dd_add(dd_mul(a, b), c);\n"
        "}\n");
}

// ===== Reciprocal / division / sqrt =======================================

static inline void register_dd_rcp() {
    register_dd_from_f32();
    register_dd_mul();
    register_dd_sub();
    fmt_intrinsic(
        DD_HEADER
        "inline dd_t dd_from_f32(float a);\n"
        "inline dd_t dd_mul(dd_t a, dd_t b);\n"
        "inline dd_t dd_sub(dd_t a, dd_t b);\n"
        "inline dd_t dd_rcp(dd_t a) {\n"
        "    float r0 = 1.0f / a.x;\n"
        "    dd_t  r  = dd_from_f32(r0);\n"
        "    for (int i = 0; i < 2; ++i) {\n"
        "        dd_t  ar = dd_mul(a, r);\n"
        "        dd_t  two_minus_ar = dd_sub(dd_from_f32(2.0f), ar);\n"
        "        r = dd_mul(r, two_minus_ar);\n"
        "    }\n"
        "    return r;\n"
        "}\n");
}

static inline void register_dd_div() {
    register_dd_mul();
    register_dd_rcp();
    fmt_intrinsic(
        DD_HEADER
        "inline dd_t dd_mul(dd_t a, dd_t b);\n"
        "inline dd_t dd_rcp(dd_t a);\n"
        "inline dd_t dd_div(dd_t a, dd_t b) { return dd_mul(a, dd_rcp(b)); }\n");
}

static inline void register_dd_sqrt() {
    register_dd_from_f32();
    register_dd_mul();
    register_dd_sub();
    fmt_intrinsic(
        DD_HEADER
        "inline dd_t dd_from_f32(float a);\n"
        "inline dd_t dd_mul(dd_t a, dd_t b);\n"
        "inline dd_t dd_sub(dd_t a, dd_t b);\n"
        "inline dd_t dd_sqrt(dd_t a) {\n"
        "    if (a.x <= 0.0f)\n"
        "        return dd_t(sqrt(a.x), 0.0f);\n"
        "    float s0 = rsqrt(a.x);\n"
        "    dd_t  s  = dd_from_f32(s0);\n"
        "    for (int i = 0; i < 2; ++i) {\n"
        "        dd_t  ss   = dd_mul(s, s);\n"
        "        dd_t  ass  = dd_mul(a, ss);\n"
        "        dd_t  three_minus = dd_sub(dd_from_f32(3.0f), ass);\n"
        "        s = dd_mul(s, dd_mul(dd_from_f32(0.5f), three_minus));\n"
        "    }\n"
        "    return dd_mul(a, s);\n"
        "}\n");
}

static inline void register_dd_rsqrt() {
    register_dd_rcp();
    register_dd_sqrt();
    fmt_intrinsic(
        DD_HEADER
        "inline dd_t dd_rcp(dd_t a);\n"
        "inline dd_t dd_sqrt(dd_t a);\n"
        "inline dd_t dd_rsqrt(dd_t a) { return dd_rcp(dd_sqrt(a)); }\n");
}

// ===== Rounding (depend on fast_two_sum) ==================================

static inline void register_dd_floor() {
    register_dd_fast_two_sum();
    fmt_intrinsic(
        DD_HEADER
        "inline dd_t dd_fast_two_sum(float a, float b);\n"
        "inline dd_t dd_floor(dd_t a) {\n"
        "    float hi_r = floor(a.x);\n"
        "    float lo_r = (a.x == hi_r) ? floor(a.y) : 0.0f;\n"
        "    return dd_fast_two_sum(hi_r, lo_r);\n"
        "}\n");
}

static inline void register_dd_ceil() {
    register_dd_fast_two_sum();
    fmt_intrinsic(
        DD_HEADER
        "inline dd_t dd_fast_two_sum(float a, float b);\n"
        "inline dd_t dd_ceil(dd_t a) {\n"
        "    float hi_r = ceil(a.x);\n"
        "    float lo_r = (a.x == hi_r) ? ceil(a.y) : 0.0f;\n"
        "    return dd_fast_two_sum(hi_r, lo_r);\n"
        "}\n");
}

static inline void register_dd_trunc() {
    register_dd_fast_two_sum();
    fmt_intrinsic(
        DD_HEADER
        "inline dd_t dd_fast_two_sum(float a, float b);\n"
        "inline dd_t dd_trunc(dd_t a) {\n"
        "    float hi_r = trunc(a.x);\n"
        "    float lo_r = (a.x == hi_r) ? trunc(a.y) : 0.0f;\n"
        "    return dd_fast_two_sum(hi_r, lo_r);\n"
        "}\n");
}

static inline void register_dd_round() {
    register_dd_fast_two_sum();
    fmt_intrinsic(
        DD_HEADER
        "inline dd_t dd_fast_two_sum(float a, float b);\n"
        "inline dd_t dd_round(dd_t a) {\n"
        "    float hi_r = rint(a.x);\n"
        "    float lo_r = (a.x == hi_r) ? rint(a.y) : 0.0f;\n"
        "    return dd_fast_two_sum(hi_r, lo_r);\n"
        "}\n");
}

// IEEE-754 fmod: a - trunc(a/b) * b. Truncation rounds toward zero, so the
// result has the sign of `a`. Returns 0 when a==0; behaviour at b==0 follows
// the underlying float division (NaN/inf), matching the LLVM/CUDA path.
static inline void register_dd_mod() {
    register_dd_div();
    register_dd_trunc();
    register_dd_mul();
    register_dd_sub();
    fmt_intrinsic(
        DD_HEADER
        "inline dd_t dd_div(dd_t a, dd_t b);\n"
        "inline dd_t dd_trunc(dd_t a);\n"
        "inline dd_t dd_mul(dd_t a, dd_t b);\n"
        "inline dd_t dd_sub(dd_t a, dd_t b);\n"
        "inline dd_t dd_mod(dd_t a, dd_t b) {\n"
        "    return dd_sub(a, dd_mul(dd_trunc(dd_div(a, b)), b));\n"
        "}\n");
}

// ===== Transcendentals ====================================================

static inline void register_dd_exp() {
    register_dd_sub();
    register_dd_from_f32();
    register_dd_mul();
    register_dd_add();
    fmt_intrinsic(
        DD_HEADER
        "inline dd_t dd_sub(dd_t a, dd_t b);\n"
        "inline dd_t dd_from_f32(float a);\n"
        "inline dd_t dd_mul(dd_t a, dd_t b);\n"
        "inline dd_t dd_add(dd_t a, dd_t b);\n"
        "inline dd_t dd_exp(dd_t a) {\n"
        "    float e0 = exp(a.x);\n"
        "    dd_t  e  = dd_from_f32(e0);\n"
        "    dd_t d = dd_sub(a, dd_from_f32(a.x));\n"
        "    dd_t d2  = dd_mul(d, d);\n"
        "    dd_t d3  = dd_mul(d2, d);\n"
        "    dd_t d4  = dd_mul(d3, d);\n"
        "    dd_t one = dd_from_f32(1.0f);\n"
        "    dd_t c2  = dd_from_f32(0.5f);\n"
        "    dd_t c6  = dd_from_f32(1.0f / 6.0f);\n"
        "    dd_t c24 = dd_from_f32(1.0f / 24.0f);\n"
        "    dd_t corr = dd_add(one, dd_add(d, dd_add(dd_mul(c2, d2),\n"
        "                                  dd_add(dd_mul(c6, d3), dd_mul(c24, d4)))));\n"
        "    return dd_mul(e, corr);\n"
        "}\n");
}

static inline void register_dd_log() {
    register_dd_from_f32();
    register_dd_exp();
    register_dd_div();
    register_dd_add();
    register_dd_sub();
    fmt_intrinsic(
        DD_HEADER
        "inline dd_t dd_from_f32(float a);\n"
        "inline dd_t dd_exp(dd_t a);\n"
        "inline dd_t dd_div(dd_t a, dd_t b);\n"
        "inline dd_t dd_add(dd_t a, dd_t b);\n"
        "inline dd_t dd_sub(dd_t a, dd_t b);\n"
        "inline dd_t dd_log(dd_t a) {\n"
        "    if (a.x <= 0.0f)\n"
        "        return dd_t(log(a.x), 0.0f);\n"
        "    float l0 = log(a.x);\n"
        "    dd_t  l  = dd_from_f32(l0);\n"
        "    dd_t e_l   = dd_exp(l);\n"
        "    dd_t ratio = dd_div(a, e_l);\n"
        "    l = dd_add(l, dd_sub(ratio, dd_from_f32(1.0f)));\n"
        "    return l;\n"
        "}\n");
}

static inline void register_dd_log2_const() {
    register_dd_from_bits();
    fmt_intrinsic(
        DD_HEADER
        "inline dd_t dd_from_bits(uint hi_bits, uint lo_bits);\n"
        "inline dd_t dd_log2_const() {\n"
        "    return dd_from_bits(0x3f317218u, 0xb22e4e4cu);\n"
        "}\n");
}

static inline void register_dd_exp2() {
    register_dd_exp();
    register_dd_mul();
    register_dd_log2_const();
    fmt_intrinsic(
        DD_HEADER
        "inline dd_t dd_exp(dd_t a);\n"
        "inline dd_t dd_mul(dd_t a, dd_t b);\n"
        "inline dd_t dd_log2_const();\n"
        "inline dd_t dd_exp2(dd_t a) { return dd_exp(dd_mul(a, dd_log2_const())); }\n");
}

static inline void register_dd_log2() {
    register_dd_log();
    register_dd_div();
    register_dd_log2_const();
    fmt_intrinsic(
        DD_HEADER
        "inline dd_t dd_log(dd_t a);\n"
        "inline dd_t dd_div(dd_t a, dd_t b);\n"
        "inline dd_t dd_log2_const();\n"
        "inline dd_t dd_log2(dd_t a) { return dd_div(dd_log(a), dd_log2_const()); }\n");
}

static inline void register_dd_pow() {
    register_dd_exp();
    register_dd_mul();
    register_dd_log();
    fmt_intrinsic(
        DD_HEADER
        "inline dd_t dd_exp(dd_t a);\n"
        "inline dd_t dd_mul(dd_t a, dd_t b);\n"
        "inline dd_t dd_log(dd_t a);\n"
        "inline dd_t dd_pow(dd_t a, dd_t b) { return dd_exp(dd_mul(b, dd_log(a))); }\n");
}

static inline void register_dd_sin() {
    register_dd_sub();
    register_dd_from_f32();
    register_dd_mul();
    register_dd_add();
    fmt_intrinsic(
        DD_HEADER
        "inline dd_t dd_sub(dd_t a, dd_t b);\n"
        "inline dd_t dd_from_f32(float a);\n"
        "inline dd_t dd_mul(dd_t a, dd_t b);\n"
        "inline dd_t dd_add(dd_t a, dd_t b);\n"
        "inline dd_t dd_sin(dd_t a) {\n"
        "    float s0 = sin(a.x);\n"
        "    float c0 = cos(a.x);\n"
        "    dd_t d = dd_sub(a, dd_from_f32(a.x));\n"
        "    dd_t d2 = dd_mul(d, d);\n"
        "    dd_t cos_d = dd_sub(dd_from_f32(1.0f),\n"
        "                        dd_mul(dd_from_f32(0.5f), d2));\n"
        "    dd_t sin_d = dd_sub(d, dd_mul(dd_from_f32(1.0f / 6.0f), dd_mul(d2, d)));\n"
        "    return dd_add(dd_mul(dd_from_f32(s0), cos_d),\n"
        "                  dd_mul(dd_from_f32(c0), sin_d));\n"
        "}\n");
}

static inline void register_dd_cos() {
    register_dd_sub();
    register_dd_from_f32();
    register_dd_mul();
    fmt_intrinsic(
        DD_HEADER
        "inline dd_t dd_sub(dd_t a, dd_t b);\n"
        "inline dd_t dd_from_f32(float a);\n"
        "inline dd_t dd_mul(dd_t a, dd_t b);\n"
        "inline dd_t dd_cos(dd_t a) {\n"
        "    float s0 = sin(a.x);\n"
        "    float c0 = cos(a.x);\n"
        "    dd_t d = dd_sub(a, dd_from_f32(a.x));\n"
        "    dd_t d2 = dd_mul(d, d);\n"
        "    dd_t cos_d = dd_sub(dd_from_f32(1.0f),\n"
        "                        dd_mul(dd_from_f32(0.5f), d2));\n"
        "    dd_t sin_d = dd_sub(d, dd_mul(dd_from_f32(1.0f / 6.0f), dd_mul(d2, d)));\n"
        "    return dd_sub(dd_mul(dd_from_f32(c0), cos_d),\n"
        "                  dd_mul(dd_from_f32(s0), sin_d));\n"
        "}\n");
}

static inline void register_dd_tan() {
    register_dd_div();
    register_dd_sin();
    register_dd_cos();
    fmt_intrinsic(
        DD_HEADER
        "inline dd_t dd_div(dd_t a, dd_t b);\n"
        "inline dd_t dd_sin(dd_t a);\n"
        "inline dd_t dd_cos(dd_t a);\n"
        "inline dd_t dd_tan(dd_t a) { return dd_div(dd_sin(a), dd_cos(a)); }\n");
}

static inline void register_dd_tanh() {
    register_dd_exp();
    register_dd_mul();
    register_dd_sub();
    register_dd_add();
    register_dd_div();
    register_dd_from_f32();
    fmt_intrinsic(
        DD_HEADER
        "inline dd_t dd_exp(dd_t a);\n"
        "inline dd_t dd_mul(dd_t a, dd_t b);\n"
        "inline dd_t dd_sub(dd_t a, dd_t b);\n"
        "inline dd_t dd_add(dd_t a, dd_t b);\n"
        "inline dd_t dd_div(dd_t a, dd_t b);\n"
        "inline dd_t dd_from_f32(float a);\n"
        "inline dd_t dd_tanh(dd_t a) {\n"
        "    dd_t e = dd_exp(dd_mul(a, dd_from_f32(2.0f)));\n"
        "    return dd_div(dd_sub(e, dd_from_f32(1.0f)),\n"
        "                  dd_add(e, dd_from_f32(1.0f)));\n"
        "}\n");
}

#undef DD_HEADER

} // namespace drjit

#endif // defined(DRJIT_ENABLE_METAL)
