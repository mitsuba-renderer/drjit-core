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
#include <stdio.h>

void *jitc_profile_register_string(const char *message) {
    if (!nvtxDomain && !jitc_nvtx_init())
        return nullptr;

    return nvtxDomainRegisterStringA(nvtxDomain, message);
}

void jitc_profile_mark(const char *message) {
    if (nvtxDomain) {
        nvtxEventAttributes_t event{};
        event.version = NVTX_VERSION;
        event.messageType = NVTX_MESSAGE_TYPE_ASCII;
        event.message = message;
        event.size = (uint16_t) sizeof(nvtxEventAttributes_t);
        nvtxDomainMarkEx(nvtxDomain, &event);
    }
}

void jitc_profile_mark_handle(const void *message) {
    if (nvtxDomain) {
        nvtxEventAttributes_t event{};
        event.version = NVTX_VERSION;
        event.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
        event.message = message;
        event.size = (uint16_t) sizeof(nvtxEventAttributes_t);
        nvtxDomainMarkEx(nvtxDomain, &event);
    }
}

void jitc_profile_range_push(const char *message) {
    if (nvtxDomain) {
        nvtxEventAttributes_t event{};
        event.version = NVTX_VERSION;
        event.messageType = NVTX_MESSAGE_TYPE_ASCII;
        event.message = message;
        event.size = (uint16_t) sizeof(nvtxEventAttributes_t);
        nvtxDomainRangePushEx(nvtxDomain, &event);
    }
}

void jitc_profile_range_push_handle(const void *message) {
    if (nvtxDomain) {
        nvtxEventAttributes_t event{};
        event.version = NVTX_VERSION;
        event.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
        event.message = message;
        event.size = (uint16_t) sizeof(nvtxEventAttributes_t);
        nvtxDomainRangePushEx(nvtxDomain, &event);
    }
}

void jitc_profile_range_pop() {
    if (nvtxDomain)
        nvtxDomainRangePop(nvtxDomain);
}
