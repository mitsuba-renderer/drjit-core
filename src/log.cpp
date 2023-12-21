/*
    src/log.cpp -- Logging, log levels, assertions, string-related code.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include <cstdio>
#include <stdexcept>
#include <ctime>
#include "internal.h"
#include "log.h"
#include "strbuf.h"

#if defined(_WIN32)
#  include <windows.h>
#endif

static StringBuffer log_buffer;
static char jitc_string_buf[64];

static const char *fatal_error_msg =
    "\nDr.Jit encountered an unrecoverable error and will now shut\n"
    "down. Please re-run your program in debug mode to check for\n"
    "out-of-bounds reads, writes, and other sources of undefined\n"
    "behavior. You can do so by calling\n"
    "\n"
    "   dr.set_flag(drjit.JitFlag.Debug, True)\n"
    "\n"
    "at the beginning of the program. If these additional checks\n"
    "fail to pinpoint the problem, then you have likely found a\n"
    "bug. We are happy to help investigate and fix the problem if\n"
    "you can you create a self-contained reproducer and submit it\n"
    "at https://github.com/mitsuba-renderer/drjit.\n"
    "\n"
    "The error message of this specific failure is as follows:\n>>> ";

void jitc_log(LogLevel log_level, const char* fmt, ...) {
    if (unlikely(log_level <= state.log_level_stderr)) {
        va_list args;
        va_start(args, fmt);
        vfprintf(stderr, fmt, args);
        fputc('\n', stderr);
        va_end(args);
    }

    if (unlikely(log_level <= state.log_level_callback && state.log_callback)) {
        va_list args;
        va_start(args, fmt);
        log_buffer.clear();
        log_buffer.vfmt(fmt, args);
        va_end(args);
        state.log_callback(log_level, log_buffer.get());
    }
}

void jitc_vlog(LogLevel log_level, const char* fmt, va_list args_) {
    if (unlikely(log_level <= state.log_level_stderr)) {
        va_list args;
        va_copy(args, args_);
        vfprintf(stderr, fmt, args);
        fputc('\n', stderr);
        va_end(args);
    }

    if (unlikely(log_level <= state.log_level_callback && state.log_callback)) {
        va_list args;
        va_copy(args, args_);
        log_buffer.clear();
        log_buffer.vfmt(fmt, args);
        va_end(args);
        state.log_callback(log_level, log_buffer.get());
    }
}

void jitc_raise(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_buffer.clear();
    log_buffer.vfmt(fmt, args);
    va_end(args);

    throw std::runtime_error(log_buffer.get());
}

void jitc_vraise(const char* fmt, va_list args) {
    log_buffer.clear();
    log_buffer.vfmt(fmt, args);

    throw std::runtime_error(log_buffer.get());
}

void jitc_fail(const char* fmt, ...) noexcept {
    if (state.log_callback) {
        va_list args;
        va_start(args, fmt);
        log_buffer.clear();
        log_buffer.put(fatal_error_msg, strlen(fatal_error_msg));
        log_buffer.vfmt(fmt, args);
        va_end(args);
        state.log_callback(Error, log_buffer.get());
    } else {
        va_list args;
        va_start(args, fmt);
        fputs(fatal_error_msg, stderr);
        vfprintf(stderr, fmt, args);
        fputc('\n', stderr);
        va_end(args);
    }

    lock_release(state.lock);
    abort();
}

void jitc_vfail(const char* fmt, va_list args_) noexcept {
    if (state.log_callback) {
        va_list args;
        va_copy(args, args_);
        log_buffer.clear();
        log_buffer.put(fatal_error_msg, strlen(fatal_error_msg));
        log_buffer.vfmt(fmt, args);
        va_end(args);
        state.log_callback(Error, log_buffer.get());
    } else {
        va_list args;
        va_copy(args, args_);
        fputs(fatal_error_msg, stderr);
        vfprintf(stderr, fmt, args);
        fputc('\n', stderr);
        va_end(args);
    }

    lock_release(state.lock);
    abort();
}

/// Generate a string representing a floating point followed by a unit
static void print_float_with_unit(char *buf, size_t bufsize, double value,
                                  bool accurate, const char *unit) {
    int digits_after_comma = accurate ? 5 : 3;

    digits_after_comma =
        std::max(digits_after_comma - int(std::log10(value)), 0);

    int pos = snprintf(buf, bufsize, "%.*f", digits_after_comma, value);

    // Remove trailing zeros
    char c;
    pos--;
    while (c = jitc_string_buf[pos], pos > 0 && (c == '0' || c == '.'))
        pos--;
    pos++;

    // Append unit if there is space
    if (pos + 1 < (int) bufsize)
        buf[pos++] = ' ';

    uint32_t i = 0;
    while (unit[i] != '\0' && pos + 1 < (int) bufsize)
        buf[pos++] = unit[i++];

    buf[pos] = '\0';
}

const char *jitc_mem_string(size_t size) {
    const char *orders[] = {
        "B", "KiB", "MiB", "GiB",
        "TiB", "PiB", "EiB"
    };

    double value = (double) size;

    int i = 0;
    for (i = 0; i < 6 && value > 1024.0; ++i)
        value /= 1024.0;

    print_float_with_unit(jitc_string_buf, sizeof(jitc_string_buf),
                          value, false, orders[i]);

    return jitc_string_buf;
}

const char *jitc_time_string(float value_) {
    double value = (double) value_;

    struct Order { double factor; const char* suffix; };
    const Order orders[] = { { 0, "us" },   { 1000, "ms" },
                             { 1000, "s" }, { 60, "m" },
                             { 60, "h" },   { 24, "d" },
                             { 7, "w" },    { 52.1429, "y" } };

    int i = 0;
    for (i = 0; i < 7 && value > orders[i+1].factor; ++i)
        value /= orders[i+1].factor;

    print_float_with_unit(jitc_string_buf, sizeof(jitc_string_buf),
                          value, true, orders[i].suffix);

    return jitc_string_buf;
}

#if !defined(_WIN32)
static timespec timer_value { 0, 0 };

float timer() {
    timespec timer_value_2;
    clock_gettime(CLOCK_MONOTONIC, &timer_value_2);
    float result = (timer_value_2.tv_sec - timer_value.tv_sec) * 1e6f +
                   (timer_value_2.tv_nsec - timer_value.tv_nsec) * 1e-3f;
    timer_value = timer_value_2;
    return result;
}
#else
static LARGE_INTEGER timer_value {};
float timer_frequency_scale;

float timer() {
    LARGE_INTEGER timer_value_2;
    QueryPerformanceCounter(&timer_value_2);
    float result = timer_frequency_scale *
                   (timer_value_2.QuadPart - timer_value.QuadPart);
    timer_value = timer_value_2;
    return result;
}
#endif
