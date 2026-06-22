/*
    src/profile.h -- Platform-agnostic profiler integration

    Copyright (c) 2024 Wenzel Jakob <wenzel.jakob@epfl.ch>

    The idea is that other profilers such as ITT could be added here in the
    future, presuming a way can be found to load the library dynamically and
    not depend on another external repository.

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "nvtx_api.h"
#include "log.h"
#include "profile.h"
#include <stdio.h>

/// Mirrors whether a profiler is attached; see declaration in profile.h.
bool jitc_profile_active = false;

#if defined(__APPLE__)
#  include <os/signpost.h>
#  include <vector>
#  include <cstring>
#  include <cstdlib>

// os_log handle for Dr.Jit signposts. Created by jitc_profile_init() and
// released by jitc_profile_shutdown().
static os_log_t jitc_signpost_log = nullptr;

// Stack of interval ids for nestable ranges
static thread_local std::vector<os_signpost_id_t> jitc_signpost_stack;
#endif

void jitc_profile_init() {
#if defined(__APPLE__)
    os_log_t log = os_log_create("org.drjit", OS_LOG_CATEGORY_DYNAMIC_TRACING);
    if (os_signpost_enabled(log)) {
        jitc_signpost_log = log;
    } else {
        os_release(log);
        jitc_signpost_log = nullptr;
    }
    jitc_profile_active = jitc_signpost_log != nullptr;
#endif
}

void jitc_profile_shutdown() {
#if defined(__APPLE__)
    if (jitc_signpost_log) {
        os_release(jitc_signpost_log);
        jitc_signpost_log = nullptr;
    }
    jitc_signpost_stack.clear();
    jitc_profile_active = false;
#endif
}

void *jitc_profile_register_string(const char *message) {
#if defined(__APPLE__)
    // No string interning for signposts; hand back a stable copy. This is
    // leaked intentionally, matching the lifetime of NVTX-registered strings
    // (ProfilerRegion instances are typically static-duration).
    return (void *) strdup(message);
#else
    if (!nvtxDomain && !jitc_nvtx_init())
        return nullptr;

    return nvtxDomainRegisterStringA(nvtxDomain, message);
#endif
}

void jitc_profile_mark(const char *message) {
#if defined(__APPLE__)
    if (!jitc_signpost_log)
        return;
    os_signpost_event_emit(jitc_signpost_log, OS_SIGNPOST_ID_EXCLUSIVE, "mark",
                           "%{public}s", message);
#else
    if (!nvtxDomain)
        return;
    nvtxEventAttributes_t event{};
    event.version = NVTX_VERSION;
    event.messageType = NVTX_MESSAGE_TYPE_ASCII;
    event.message = message;
    event.size = (uint16_t) sizeof(nvtxEventAttributes_t);
    nvtxDomainMarkEx(nvtxDomain, &event);
#endif
}

void jitc_profile_mark_handle(const void *message) {
#if defined(__APPLE__)
    jitc_profile_mark((const char *) message);
#else
    if (!nvtxDomain)
        return;
    nvtxEventAttributes_t event{};
    event.version = NVTX_VERSION;
    event.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
    event.message = message;
    event.size = (uint16_t) sizeof(nvtxEventAttributes_t);
    nvtxDomainMarkEx(nvtxDomain, &event);
#endif
}

void jitc_profile_range_push(const char *message) {
#if defined(__APPLE__)
    if (!jitc_signpost_log)
        return;
    os_signpost_id_t id = os_signpost_id_generate(jitc_signpost_log);
    jitc_signpost_stack.push_back(id);
    os_signpost_interval_begin(jitc_signpost_log, id, "range", "%{public}s",
                               message);
#else
    if (!nvtxDomain)
        return;
    nvtxEventAttributes_t event{};
    event.version = NVTX_VERSION;
    event.messageType = NVTX_MESSAGE_TYPE_ASCII;
    event.message = message;
    event.size = (uint16_t) sizeof(nvtxEventAttributes_t);
    nvtxDomainRangePushEx(nvtxDomain, &event);
#endif
}

void jitc_profile_range_push_handle(const void *message) {
#if defined(__APPLE__)
    jitc_profile_range_push((const char *) message);
#else
    if (!nvtxDomain)
        return;
    nvtxEventAttributes_t event{};
    event.version = NVTX_VERSION;
    event.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
    event.message = message;
    event.size = (uint16_t) sizeof(nvtxEventAttributes_t);
    nvtxDomainRangePushEx(nvtxDomain, &event);
#endif
}

void jitc_profile_range_pop() {
#if defined(__APPLE__)
    if (!jitc_signpost_log)
        return;
    if (jitc_signpost_stack.empty())
        jitc_fail("jit_profile_range_pop(): underflow!");
    os_signpost_id_t id = jitc_signpost_stack.back();
    jitc_signpost_stack.pop_back();
    os_signpost_interval_end(jitc_signpost_log, id, "range");
#else
    if (!nvtxDomain)
        return;
    nvtxDomainRangePop(nvtxDomain);
#endif
}
