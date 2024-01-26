/*
    src/profile.h -- Platform-agnostic profiler integration

    Copyright (c) 2024 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

extern void *jitc_profile_register_string(const char *message);
extern void jitc_profile_mark(const char *message);
extern void jitc_profile_mark_handle(const void *handle);
extern void jitc_profile_range_push(const char *message);
extern void jitc_profile_range_push_handle(const void *handle);
extern void jitc_profile_range_pop();

struct ProfilerRegion {
    explicit ProfilerRegion(const char *ptr) : handle(jitc_profile_register_string(ptr)) { }
    const void *handle;
};

struct ProfilerPhase {
    ProfilerPhase(const ProfilerRegion &region) {
        jitc_profile_range_push_handle(region.handle);
    }

    ~ProfilerPhase() {
        jitc_profile_range_pop();
    }
};
