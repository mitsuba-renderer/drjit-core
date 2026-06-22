/*
    src/profile.h -- Platform-agnostic profiler integration

    Copyright (c) 2024 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

/// Initialize profiler integration (samples whether a tool is listening).
extern void jitc_profile_init();

/// Release profiler-integration resources (paired with jitc_profile_init).
extern void jitc_profile_shutdown();

extern void *jitc_profile_register_string(const char *message);
extern void jitc_profile_mark(const char *message);
extern void jitc_profile_mark_handle(const void *handle);
extern void jitc_profile_range_push(const char *message);
extern void jitc_profile_range_push_handle(const void *handle);
extern void jitc_profile_range_pop();

/// Flag to indicate that a profiler (NVTX, Apple os_signpost) is listening.
extern bool jitc_profile_active;

struct ProfilerRegion {
    explicit ProfilerRegion(const char *ptr) : handle(jitc_profile_register_string(ptr)) { }
    const void *handle;
};

struct ProfilerPhase {
    /// Latch the decision at construction so the push/pop stays balanced even
    /// if a profiler attaches/detaches mid-scope.
    bool active;

    ProfilerPhase(const ProfilerRegion &region) : active(jitc_profile_active) {
        if (active)
            jitc_profile_range_push_handle(region.handle);
    }

    ~ProfilerPhase() {
        if (active)
            jitc_profile_range_pop();
    }
};
