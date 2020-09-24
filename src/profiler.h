#if defined(ENOKI_ENABLE_ITTNOTIFY)
#  include <ittnotify.h>
#endif

#if defined(ENOKI_ENABLE_NVTX)
#  include <nvtx3/nvToolsExt.h>
#endif

#if defined(ENOKI_ENABLE_ITTNOTIFY)
extern __itt_domain *enoki_domain;
#endif

struct ProfilerRegion {
    ProfilerRegion(const char *name) : name(name) {
#if defined(ENOKI_ENABLE_ITTNOTIFY)
        itt_handle = __itt_string_handle_create(name);
#endif
    }

    const char *name;
#if defined(ENOKI_ENABLE_ITTNOTIFY)
    __itt_string_handle *itt_handle;
#endif
};

struct ProfilerPhase {
    ProfilerPhase(const ProfilerRegion &region) {
#if defined(ENOKI_ENABLE_ITTNOTIFY)
        __itt_task_begin(enoki_domain, __itt_null, __itt_null, region.itt_handle);
#endif
#if defined(ENOKI_ENABLE_NVTX)
        nvtxRangePush(region.name);
#endif
        (void) region;
    }
    ~ProfilerPhase() {
#if defined(ENOKI_ENABLE_ITTNOTIFY)
        __itt_task_end(enoki_domain);
#endif
#if defined(ENOKI_ENABLE_NVTX)
        nvtxRangePop();
#endif
    }
};
