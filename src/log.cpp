/*
    src/log.cpp -- Logging, log levels, assertions, string-related code.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include <cstdio>
#include <stdexcept>
#include <ctime>
#include "internal.h"
#include "log.h"

#if defined(_WIN32)
#  include <windows.h>
#endif

static Buffer log_buffer{128};
static char jit_string_buf[64];

void jit_log(LogLevel log_level, const char* fmt, ...) {
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

void jit_vlog(LogLevel log_level, const char* fmt, va_list args_) {
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

void jit_raise(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_buffer.clear();
    log_buffer.vfmt(fmt, args);
    va_end(args);

    throw std::runtime_error(log_buffer.get());
}

void jit_vraise(const char* fmt, va_list args) {
    log_buffer.clear();
    log_buffer.vfmt(fmt, args);

    throw std::runtime_error(log_buffer.get());
}

void jit_fail(const char* fmt, ...) {
    fprintf(stderr, "\n\nCritical failure in Enoki JIT compiler: ");

    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);

    fputc('\n', stderr);
    exit(EXIT_FAILURE);
}

void jit_vfail(const char* fmt, va_list args) {
    fprintf(stderr, "Critical failure in Enoki JIT compiler: ");
    vfprintf(stderr, fmt, args);
    fputc('\n', stderr);
    exit(EXIT_FAILURE);
}

const char *jit_mem_string(size_t size) {
    const char *orders[] = {
        "B", "KiB", "MiB", "GiB",
        "TiB", "PiB", "EiB"
    };
    float value = (float) size;

    int i = 0;
    for (i = 0; i < 6 && value > 1024.f; ++i)
        value /= 1024.f;

    snprintf(jit_string_buf, 64,
             i > 0 ? "%.3g %s" : "%.0f %s", value,
             orders[i]);

    return jit_string_buf;
}

const char *jit_time_string(float value) {
    struct Order { float factor; const char* suffix; };
    const Order orders[] = { { 0, "us" },   { 1000, "ms" },
                             { 1000, "s" }, { 60, "m" },
                             { 60, "h" },   { 24, "d" },
                             { 7, "w" },    { (float) 52.1429, "y" } };

    int i = 0;
    for (i = 0; i < 7 && value > orders[i+1].factor; ++i)
        value /= orders[i+1].factor;

    snprintf(jit_string_buf, 64, "%.5g %s", value, orders[i].suffix);

    return jit_string_buf;
}

Buffer::Buffer(size_t size) : m_start(nullptr), m_cur(nullptr), m_end(nullptr) {
    m_start = (char *) malloc_check(size);
    m_end = m_start + size;
    clear();
}

size_t Buffer::fmt(const char *format, ...) {
    size_t written;
    do {
        size_t size = m_end - m_cur;
        va_list args;
        va_start(args, format);
        written = (size_t) vsnprintf(m_cur, size, format, args);
        va_end(args);

        if (likely(written < size)) {
            m_cur += written;
            break;
        }

        expand();
    } while (true);

    return written;
}

size_t Buffer::vfmt(const char *format, va_list args_) {
    size_t written;
    va_list args;
    do {
        size_t size = m_end - m_cur;
        va_copy(args, args_);
        written = (size_t) vsnprintf(m_cur, size, format, args);
        va_end(args);

        if (likely(written < size)) {
            m_cur += written;
            break;
        }

        expand();
    } while (true);
    return written;
}

void Buffer::expand(size_t minval) {
    size_t old_alloc_size = m_end - m_start,
           new_alloc_size = 2 * old_alloc_size + minval,
           used_size      = m_cur - m_start,
           copy_size      = std::min(used_size + 1, old_alloc_size);

    char *tmp = (char *) malloc_check(new_alloc_size);
    memcpy(tmp, m_start, copy_size);
    free(m_start);

    m_start = tmp;
    m_end = m_start + new_alloc_size;
    m_cur = m_start + used_size;
}

#if !defined(_WIN32)
static timespec timer_value { 0, 0 };

float timer() {
    timespec timer_value_2;
    clock_gettime(CLOCK_REALTIME, &timer_value_2);
    float result = (timer_value_2.tv_sec - timer_value.tv_sec) * 1e6f +
                   (timer_value_2.tv_nsec - timer_value.tv_nsec) * 1e-3f;
    timer_value = timer_value_2;
    return result;
}
#else
static LARGE_INTEGER timer_value{};
static LARGE_INTEGER timer_frequency{};

float timer() {
    LARGE_INTEGER value;
    QueryPerformanceCounter(&value);

    if (timer_frequency.QuadPart == 0)
        QueryPerformanceFrequency(&timer_frequency);

    float result = (float)(value.QuadPart - timer_value.QuadPart) / timer_frequency.QuadPart * 1e6f;
    timer_value = value;

    return result;
}
#endif
