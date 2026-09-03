/*
    src/log.h -- Logging, log levels, assertions, string-related code.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit-core/jit.h>
#include <stdarg.h>
#include "common.h"

static constexpr LogLevel Disable = LogLevel::Disable;
static constexpr LogLevel Error   = LogLevel::Error;
static constexpr LogLevel Warn    = LogLevel::Warn;
static constexpr LogLevel Info    = LogLevel::Info;
static constexpr LogLevel InfoSym = LogLevel::InfoSym;
static constexpr LogLevel Debug   = LogLevel::Debug;
static constexpr LogLevel Trace   = LogLevel::Trace;

#if defined(NDEBUG)
#  define jitc_trace(...) do { } while (0)
#  define jitc_assert(...) do { } while (0)
#else
#  define jitc_trace(...) jitc_log(Trace, __VA_ARGS__)
#define jitc_assert(cond, fmt, ...)                                            \
    if (unlikely(!(cond)))                                                     \
        jitc_fail("drjit: assertion failure (\"%s\") in line %i: " fmt, #cond, \
                  __LINE__, ##__VA_ARGS__);
#endif

/// Print a log message with the specified log level and message
#if defined(__GNUC__)
    __attribute__((__format__ (__printf__, 2, 3)))
#endif
extern void jitc_log(LogLevel level, const char* fmt, ...);

/// Deliver log messages queued by pool workers (see jitc_log_postpone())
extern void jitc_log_flush();

/// Print a log message with the specified log level and message
extern void jitc_vlog(LogLevel level, const char* fmt, va_list args);

/// Print a log message that has already been formatted
extern void jitc_log_msg(LogLevel level, const char *msg);

/// Raise a std::runtime_error with the given message
#if defined(__GNUC__)
    __attribute__((noreturn, __format__ (__printf__, 1, 2)))
#else
    [[noreturn]]
#endif
extern void jitc_raise(const char* fmt, ...);

/// Raise a std::runtime_error with the given message
[[noreturn]] extern void jitc_vraise(const char* fmt, va_list args);

/// Immediately terminate the application due to a fatal internal error
#if defined(__GNUC__)
    __attribute__((noreturn, __format__ (__printf__, 1, 2)))
#else
   [[noreturn]]
#endif
extern void jitc_fail(const char* fmt, ...) noexcept;

/// Immediately terminate the application due to a fatal internal error
[[noreturn]] extern void jitc_vfail(const char* fmt, va_list args) noexcept;

/// Fixed-size string returned by value by the formatting helpers below.
struct SmallString {
    char buf[32];
    const char *c_str() const { return buf; }
};

/// Convert a number of bytes into a human-readable string
extern SmallString jitc_mem_string(size_t size);

/// Convert a time in microseconds into a human-readable string
extern SmallString jitc_time_string(float us);

/// Return the number of microseconds since the previous timer() call
extern float timer();
