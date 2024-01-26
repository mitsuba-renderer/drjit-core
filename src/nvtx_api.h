/*
    src/nvtx_api.h -- Low-level interface to the NVTX profiling API

    Copyright (c) 2024 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <stdint.h>

#if defined(_MSC_VER)
#  define NVTX_API __stdcall
#else
#  define NVTX_API
#endif

#define NVTX_VERSION 3
#define NVTX_MESSAGE_TYPE_ASCII      1
#define NVTX_MESSAGE_TYPE_REGISTERED 3

typedef void* nvtxDomainHandle_t;
typedef void* nvtxStringHandle_t;
extern nvtxDomainHandle_t nvtxDomain;

extern void jitc_nvtx_shutdown();
extern bool jitc_nvtx_init();

struct nvtxEventAttributes_t {
    uint16_t version;
    uint16_t size;
    uint32_t category;
    int32_t colorType;
    uint32_t color;
    int32_t payloadType;
    int32_t reserved0;

    union {
        uint64_t ullValue;
        int64_t llValue;
        double dValue;
        uint32_t uiValue;
        int32_t iValue;
        float fValue;
    } payload;

    int32_t messageType;
    const void *message;
};

extern void NVTX_API (*nvtxDomainMarkEx)(nvtxDomainHandle_t,
                                         const nvtxEventAttributes_t *);
extern int NVTX_API (*nvtxDomainRangePushEx)(nvtxDomainHandle_t,
                                             const nvtxEventAttributes_t *);
extern int NVTX_API (*nvtxDomainRangePop)(nvtxDomainHandle_t);
extern nvtxStringHandle_t
    NVTX_API (*nvtxDomainRegisterStringA)(nvtxDomainHandle_t, const char *);
extern nvtxDomainHandle_t NVTX_API (*nvtxDomainCreateA)(const char *);
extern void NVTX_API (*nvtxDomainDestroy)(nvtxDomainHandle_t);
extern void NVTX_API (*nvtxInitialize)(const void *);
