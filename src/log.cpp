/*
    src/log.cpp -- Logging, log levels, assertions, string-related code.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include <cstdio>
#include <stdexcept>
#include <ctime>
#include <mutex>
#include <string>
#include <vector>
#include "internal.h"
#include "log.h"
#include <nanothread/nanothread.h>

#if defined(_WIN32)
#  include <windows.h>
#endif

/// Holds a postponed log message
struct LogMessage {
    LogLevel level;
    std::string msg;
};

// Log messages waiting to be handed to ``state.log_callback``
static std::mutex log_mutex;
static std::vector<LogMessage> log_postponed;

/// Client hook registered via \ref jit_set_log_defer_callback(), if any
static LogDeferCallback log_defer_callback = nullptr;

static const char *fatal_error_msg =
    "\nDr.Jit encountered an unrecoverable error and will now shut\n"
    "down. Please re-run your program in debug mode to check for\n"
    "out-of-bounds reads, writes, and other sources of undefined\n"
    "behavior. You can do so by calling\n"
    "\n"
    "   dr.set_flag(dr.JitFlag.Debug, True)\n"
    "\n"
    "at the beginning of the program. If these additional checks\n"
    "fail to pinpoint the problem, then you have likely found a\n"
    "bug. We are happy to help investigate and fix the problem if\n"
    "you can you create a self-contained reproducer and submit it\n"
    "at https://github.com/mitsuba-renderer/drjit.\n"
    "\n"
    "The error message of this specific failure is as follows:\n>>> ";

/// Render a printf-style message into a string
static std::string jitc_vformat(const char *fmt, va_list args) {
    va_list args_2;
    va_copy(args_2, args);
    int size = vsnprintf(nullptr, 0, fmt, args_2);
    va_end(args_2);

    if (size < 0) {
        fprintf(stderr, "jitc_vformat(): vsnprintf failed!\n");
        abort();
    }

    std::string result((size_t) size, '\0');
    vsnprintf(result.data(), (size_t) size + 1, fmt, args);
    return result;
}

/// Hand queued messages to the log callback. No Dr.Jit lock may be held.
static void jitc_log_deliver() noexcept {
    std::vector<LogMessage> todo;

    {
        std::lock_guard<std::mutex> guard(log_mutex);
        todo.swap(log_postponed);
    }

    LogCallback callback = state.log_callback;
    for (LogMessage &message : todo) {
        if (unlikely(!callback))
            break;

        // The callback runs user code, which may fail in arbitrary ways
        try {
            callback(message.level, message.msg.c_str());
        } catch (...) {
            fputs(message.msg.c_str(), stderr);
            fputc('\n', stderr);
        }
    }
}

void jit_log_flush() {
    // A client may still be inside a critical section of its own
    if (log_defer_callback && log_defer_callback())
        return;

    jitc_log_deliver();
}

void jit_set_log_defer_callback(LogDeferCallback callback) {
    log_defer_callback = callback;
}

void lock_release_pending(Lock &lock) noexcept {
    lock.recursion_count = 1; // Clears LOCK_PENDING
    lock_release(lock);       // Takes the fast path and fully unlocks
    jit_log_flush();
}

/// Deliver queued messages now, or once this thread leaves its critical section
static void jitc_log_flush_or_defer() {
    if (likely(state.lock.owner.load(std::memory_order_relaxed) == thread_id()))
        lock_set_pending(state.lock);
    else
        jit_log_flush();
}

void jitc_log_flush() {
    {
        std::lock_guard<std::mutex> guard(log_mutex);
        if (log_postponed.empty())
            return;
    }

    jitc_log_flush_or_defer();
}

/// Queue `msg`, and arrange for it to be delivered at a safe moment
static void jitc_log_postpone(LogLevel log_level, std::string &&msg) {
    {
        std::lock_guard<std::mutex> guard(log_mutex);
        log_postponed.push_back({ log_level, std::move(msg) });
    }

    // Pool workers never invoke the callback themselves. The thread that waits
    // for them may hold a resource that the callback needs (e.g. the GIL), and
    // it drains the queue via jitc_log_flush() after the parallel region ends.
    if (pool_thread_id() != 0)
        return;

    jitc_log_flush_or_defer();
}

void jitc_vlog(LogLevel log_level, const char* fmt, va_list args) {
    if (likely(!jitc_log_active(log_level)))
        return;

    if (log_level <= state.log_level_stderr) {
        va_list args_2;
        va_copy(args_2, args);
        vfprintf(stderr, fmt, args_2);
        fputc('\n', stderr);
        va_end(args_2);
    }

    if (log_level > state.log_level_callback || !state.log_callback)
        return;

    jitc_log_postpone(log_level, jitc_vformat(fmt, args));
}

void jitc_log_msg(LogLevel log_level, const char *msg) {
    if (unlikely(!state.log_callback)) {
        fputs(msg, stderr);
        fputc('\n', stderr);
        return;
    }

    jitc_log_postpone(log_level, msg);
}

void jitc_log(LogLevel log_level, const char* fmt, ...) {
    if (likely(!jitc_log_active(log_level)))
        return;

    va_list args;
    va_start(args, fmt);
    jitc_vlog(log_level, fmt, args);
    va_end(args);
}

void jitc_vraise(const char* fmt, va_list args) {
    throw std::runtime_error(jitc_vformat(fmt, args));
}

void jitc_raise(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    jitc_vraise(fmt, args);
    // va_end(args); (dead code)
}

void jitc_vfail(const char* fmt, va_list args) noexcept {
    std::string msg = fatal_error_msg + jitc_vformat(fmt, args);

    // Release the lock first. Otherwise, the callback below could wait for a
    // resource held by another thread that is itself blocked on the lock.
    if (state.lock.owner.load(std::memory_order_relaxed) == thread_id()) {
        state.lock.recursion_count = 1;
        lock_release(state.lock);
    }

    // Deliver the messages leading up to the failure, then report it. Pool
    // workers write to stderr instead for the reason given in jitc_log_postpone()
    if (state.log_callback && pool_thread_id() == 0) {
        jitc_log_deliver();
        state.log_callback(Error, msg.c_str());
    } else {
        fputs(msg.c_str(), stderr);
        fputc('\n', stderr);
    }

    abort();
}

void jitc_fail(const char* fmt, ...) noexcept {
    va_list args;
    va_start(args, fmt);
    jitc_vfail(fmt, args);
    // va_end(args); (dead code)
}

/// Generate a string representing a floating point followed by a unit
static void print_float_with_unit(char *buf, size_t bufsize, double value,
                                  bool accurate, const char *unit) {
    int digits_after_comma = accurate ? 5 : 3;

    if (value == 0)
        digits_after_comma = 0;
    else
        digits_after_comma =
            std::max(digits_after_comma - int(std::log10(value)), 0);

    int pos = snprintf(buf, bufsize, "%.*f", digits_after_comma, value);

    // Remove trailing zeros
    char c;
    pos--;
    while (c = buf[pos], pos > 0 && (c == '0' || c == '.'))
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

SmallString jitc_mem_string(size_t size) {
    const char *orders[] = {
        "B", "KiB", "MiB", "GiB",
        "TiB", "PiB", "EiB"
    };

    double value = (double) size;

    int i = 0;
    for (i = 0; i < 6 && value > 1024.0; ++i)
        value /= 1024.0;

    SmallString result;
    print_float_with_unit(result.buf, sizeof(result.buf),
                          value, false, orders[i]);

    return result;
}

SmallString jitc_time_string(float value_) {
    double value = (double) value_;

    struct Order { double factor; const char* suffix; };
    const Order orders[] = { { 0, "us" },   { 1000, "ms" },
                             { 1000, "s" }, { 60, "m" },
                             { 60, "h" },   { 24, "d" },
                             { 7, "w" },    { 52.1429, "y" } };

    int i = 0;
    for (i = 0; i < 7 && value > orders[i+1].factor; ++i)
        value /= orders[i+1].factor;

    SmallString result;
    print_float_with_unit(result.buf, sizeof(result.buf),
                          value, true, orders[i].suffix);

    return result;
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
