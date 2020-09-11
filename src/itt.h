#if defined(ENOKI_ITTNOTIFY)
#  include <ittnotify.h>

extern __itt_domain *enoki_domain;

struct ProfilerRegion {
    ProfilerRegion(const char *name)
        : handle(__itt_string_handle_create(name)) { }
    __itt_string_handle *handle;
};

struct ProfilerPhase {
    ProfilerPhase(const ProfilerRegion &region) {
        __itt_task_begin(enoki_domain, __itt_null, __itt_null, region.handle);
    }
    ~ProfilerPhase() {
        __itt_task_end(enoki_domain);
    }
};
#else

struct ProfilerRegion {
    ProfilerRegion(const char *) { }
};

struct ProfilerPhase {
    ProfilerPhase(const ProfilerRegion &) { }
};

#endif
