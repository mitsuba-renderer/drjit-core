/*
    tests/isect.cpp -- Recorded intersection functions on the LLVM backend

    These tests stand in for Embree with a minimal traversal that invokes the
    generated callbacks through the same argument structures, so that the
    whole path (recording, deferred assembly, launch-time binding, and the
    callback ABI) runs without a ray tracing library.
*/

#include "test.h"
#include <cmath>
#include <vector>

namespace dr = drjit;

#define TEST_LLVM_FP32(name)                                                   \
    template <JitBackend Backend, typename Float, typename Int32,              \
              typename UInt32, typename Mask, template <class> class Array>    \
    void test##name();                                                         \
    TEST_REGISTER_LLVM(name, _llvm, FloatL)                                    \
    template <JitBackend Backend, typename Float, typename Int32,              \
              typename UInt32, typename Mask, template <class> class Array>    \
    void test##name()

/// Layout of RTCIntersectFunctionNArguments / RTCOccludedFunctionNArguments
struct IsectArgs {
    int *valid;
    void *user;
    unsigned int prim;
    void *context;
    void *ray;
    unsigned int N;
    unsigned int geom;
};

/// A stand-in for an Embree scene: geometries with their binding records
struct FakeGeometry {
    JitIsectBinding *binding;
    uint32_t prim_count;
};

struct FakeScene {
    std::vector<FakeGeometry> geoms;
};

static void fake_trace(const int *valid, void *scene_, void *context,
                       void *ray, int mode) {
    FakeScene *scene = (FakeScene *) scene_;
    unsigned int N = jit_llvm_vector_width();

    for (size_t g = 0; g < scene->geoms.size(); ++g) {
        const FakeGeometry &geom = scene->geoms[g];
        if (!geom.binding->code)
            jit_fail("fake_trace(): the binding has no compiled code!");

        for (uint32_t p = 0; p < geom.prim_count; ++p) {
            IsectArgs args { (int *) valid, geom.binding, p, context, ray, N,
                             (unsigned int) g };
            ((void (*)(const IsectArgs *, int)) geom.binding->code)(&args, mode);
        }
    }
}

static void fake_intersect(const int *valid, void *scene, void *context, void *ray) {
    fake_trace(valid, scene, context, ray, 0);
}

static void fake_occluded(const int *valid, void *scene, void *context, void *ray) {
    fake_trace(valid, scene, context, ray, 1);
}

/// Open a recording session and hold its placeholder inputs
struct IsectInputs {
    FloatL ox, oy, oz, dx, dy, dz, tmax, time;
    UInt32L prim;
    uint32_t in[9];

    IsectInputs() {
        jit_isect_begin(JitBackend::LLVM, VarType::Float32, in);
        ox = FloatL::steal(in[0]); oy = FloatL::steal(in[1]);
        oz = FloatL::steal(in[2]); dx = FloatL::steal(in[3]);
        dy = FloatL::steal(in[4]); dz = FloatL::steal(in[5]);
        tmax = FloatL::steal(in[6]); time = FloatL::steal(in[7]);
        prim = UInt32L::steal(in[8]);
    }
};

/// Sphere intersection body. Reports the hit point relative to the center
/// (x, y) as the two attributes.
static void sphere_body(const IsectInputs &in, const FloatL &cx,
                        const FloatL &cy, const FloatL &cz, const FloatL &r,
                        uint32_t out[4]) {
    FloatL lx = in.ox - cx, ly = in.oy - cy, lz = in.oz - cz;
    FloatL A = in.dx * in.dx + in.dy * in.dy + in.dz * in.dz,
           B = FloatL(2.f) * (lx * in.dx + ly * in.dy + lz * in.dz),
           C = lx * lx + ly * ly + lz * lz - r * r,
           disc = B * B - FloatL(4.f) * A * C,
           sq = sqrt(disc),
           inv = FloatL(1.f) / (FloatL(2.f) * A),
           t0 = (FloatL(0.f) - B - sq) * inv,
           t1 = (FloatL(0.f) - B + sq) * inv;

    MaskL ok0 = !(t0 < FloatL(0.f)) && t0 <= in.tmax,
          ok1 = !(t1 < FloatL(0.f)) && t1 <= in.tmax,
          hit = !(disc < FloatL(0.f)) && (ok0 || ok1);

    FloatL t = select(ok0, t0, t1),
           px = fmadd(in.dx, t, in.ox) - cx,
           py = fmadd(in.dy, t, in.oy) - cy;

    out[0] = jit_var_inc_ref(hit.index());
    out[1] = jit_var_inc_ref(t.index());
    out[2] = jit_var_cast(px.index(), VarType::UInt32, 1);
    out[3] = jit_var_cast(py.index(), VarType::UInt32, 1);
}

/// Record a sphere with opaque center and radius
static uint32_t record_sphere(const FloatL &cx, const FloatL &cy,
                              const FloatL &cz, const FloatL &r) {
    uint32_t out[4], func;
    {
        IsectInputs in;
        sphere_body(in, cx, cy, cz, r, out);
    }
    func = jit_isect_end("sphere", out);

    for (uint32_t i = 0; i < 4; ++i)
        jit_var_dec_ref(out[i]);

    return func;
}

struct TraceResult {
    MaskL valid;
    FloatL t, u, v;
    UInt32L prim, geom, inst;
    MaskL hit_inst;
    UInt32L flags; // occlusion queries
};

static TraceResult trace(FakeScene &scene, const FloatL &ox, const FloatL &oy,
                         const FloatL &oz, const FloatL &dx, const FloatL &dy,
                         const FloatL &dz, const FloatL &tmax,
                         bool shadow = false, uint32_t ray_flags = 0,
                         uint32_t scene_handle = 0, uint32_t func_handle = 0) {
    JitBackend backend = JitBackend::LLVM;
    // The scene and callback pointers carry a resource handle as their
    // dependency (like Mitsuba's accel handle), so that a frozen function
    // records them as inputs instead of raw literals.
    uint32_t func_v = jit_var_pointer(
        backend, (void *) (shadow ? fake_occluded : fake_intersect),
        func_handle, 0);
    uint32_t scene_v = jit_var_pointer(backend, &scene, scene_handle, 0);

    MaskL coherent(false), active(true);
    FloatL tmin(0.f), time(0.f);
    UInt32L mask(0xFFFFFFFFu), id(0u), flags(ray_flags);

    uint32_t in[14] = { coherent.index(), active.index(), ox.index(),
                        oy.index(), oz.index(), tmin.index(), dx.index(),
                        dy.index(), dz.index(), time.index(), tmax.index(),
                        mask.index(), id.index(), flags.index() };
    uint32_t out[8] = { };

    jit_llvm_ray_trace(func_v, scene_v, shadow ? 1 : 0, in, out);
    jit_var_dec_ref(func_v);
    jit_var_dec_ref(scene_v);

    TraceResult r;
    if (shadow) {
        r.valid = MaskL::steal(out[0]);
        r.flags = UInt32L::steal(out[1]);
    } else {
        r.valid = MaskL::steal(out[0]);
        r.t = FloatL::steal(out[1]);
        r.u = FloatL::steal(out[2]);
        r.v = FloatL::steal(out[3]);
        r.prim = UInt32L::steal(out[4]);
        r.geom = UInt32L::steal(out[5]);
        r.inst = UInt32L::steal(out[6]);
        r.hit_inst = MaskL::steal(out[7]);
    }
    return r;
}

/// A grid of rays in the plane z = -5 pointing along +z
struct RayGrid {
    static constexpr size_t n = 100;
    FloatL ox, oy, oz, dx, dy, dz, tmax;
    std::vector<float> x, y;

    RayGrid(float extent = 2.f, float z = -5.f, float tmax_ = 100.f) {
        for (size_t i = 0; i < 10; ++i) {
            for (size_t j = 0; j < 10; ++j) {
                x.push_back(-extent + 2.f * extent * ((float) i + 0.5f) / 10.f);
                y.push_back(-extent + 2.f * extent * ((float) j + 0.5f) / 10.f);
            }
        }
        ox = FloatL::copy(x.data(), n);
        oy = FloatL::copy(y.data(), n);
        oz = FloatL(z);
        dx = FloatL(0.f);
        dy = FloatL(0.f);
        dz = FloatL(1.f);
        tmax = FloatL(tmax_);
    }
};

/// Distance along +z from (x, y, z0) to a sphere, or infinity
static float ref_sphere(float x, float y, float z0, float cx, float cy,
                        float cz, float r) {
    float ddx = x - cx, ddy = y - cy, d2 = ddx * ddx + ddy * ddy;
    if (d2 > r * r)
        return INFINITY;
    float dz = std::sqrt(r * r - d2);
    float t = (cz - dz) - z0;
    if (t < 0)
        t = (cz + dz) - z0;
    return t < 0 ? INFINITY : t;
}

static bool approx(float a, float b, float eps = 1e-4f) {
    if (std::isinf(a) || std::isinf(b))
        return a == b;
    return std::abs(a - b) <= eps * std::max(1.f, std::abs(b));
}

TEST_LLVM_FP32(01_sphere) {
    FloatL cx = dr::opaque<FloatL>(0.f), cy = dr::opaque<FloatL>(0.f),
           cz = dr::opaque<FloatL>(0.f), r = dr::opaque<FloatL>(1.f);
    uint32_t func = record_sphere(cx, cy, cz, r);

    FakeScene scene;
    JitIsectBinding *binding = jit_isect_bind(func, (uintptr_t) &scene, 0, 0, nullptr);
    scene.geoms.push_back({ binding, 1 });
    jit_var_dec_ref(func);

    RayGrid rays;
    for (int round = 0; round < 2; ++round) {
        TraceResult res = trace(scene, rays.ox, rays.oy, rays.oz, rays.dx,
                                rays.dy, rays.dz, rays.tmax);

        for (size_t i = 0; i < RayGrid::n; ++i) {
            float t_ref = ref_sphere(rays.x[i], rays.y[i], -5.f, 0.f, 0.f, 0.f, 1.f);
            bool hit = !std::isinf(t_ref);
            jit_assert(res.valid.read(i) == hit);
            jit_assert(approx(res.t.read(i), t_ref));
            if (hit) {
                jit_assert(approx(res.u.read(i), rays.x[i]));
                jit_assert(approx(res.v.read(i), rays.y[i]));
                jit_assert(res.prim.read(i) == 0);
                jit_assert(res.geom.read(i) == 0);
                jit_assert(res.inst.read(i) == (uint32_t) -1);
            }
            jit_assert(!res.hit_inst.read(i));
        }

        // The second round runs after a kernel cache flush, which drops the
        // compiled unit; the launch must re-resolve the binding
        jit_flush_kernel_cache();
    }

    jit_isect_unbind(binding);
}

TEST_LLVM_FP32(02_two_spheres_share_unit) {
    // Two spheres whose bodies are identical except for the captured data
    FloatL c0x = dr::opaque<FloatL>(0.f), c0y = dr::opaque<FloatL>(0.f),
           c0z = dr::opaque<FloatL>(0.f), r0 = dr::opaque<FloatL>(1.f),
           c1x = dr::opaque<FloatL>(0.5f), c1y = dr::opaque<FloatL>(0.f),
           c1z = dr::opaque<FloatL>(-2.f), r1 = dr::opaque<FloatL>(0.5f);

    uint32_t func0 = record_sphere(c0x, c0y, c0z, r0),
             func1 = record_sphere(c1x, c1y, c1z, r1);

    FakeScene scene;
    JitIsectBinding *b0 = jit_isect_bind(func0, (uintptr_t) &scene, 0, 0, nullptr),
                    *b1 = jit_isect_bind(func1, (uintptr_t) &scene, 0, 0, nullptr);
    scene.geoms.push_back({ b0, 1 });
    scene.geoms.push_back({ b1, 1 });
    jit_var_dec_ref(func0);
    jit_var_dec_ref(func1);

    RayGrid rays;
    TraceResult res = trace(scene, rays.ox, rays.oy, rays.oz, rays.dx,
                            rays.dy, rays.dz, rays.tmax);

    // Both bindings resolve to the same compiled unit
    jit_assert(b0->code == b1->code);

    for (size_t i = 0; i < RayGrid::n; ++i) {
        float t0 = ref_sphere(rays.x[i], rays.y[i], -5.f, 0.f, 0.f, 0.f, 1.f),
              t1 = ref_sphere(rays.x[i], rays.y[i], -5.f, 0.5f, 0.f, -2.f, 0.5f),
              t_ref = std::min(t0, t1);
        bool hit = !std::isinf(t_ref);
        jit_assert(res.valid.read(i) == hit);
        jit_assert(approx(res.t.read(i), t_ref));
        if (hit) {
            uint32_t geom = t1 < t0 ? 1 : 0;
            jit_assert(res.geom.read(i) == geom);
            float cx = geom ? 0.5f : 0.f;
            jit_assert(approx(res.u.read(i), rays.x[i] - cx));
        }
    }

    jit_isect_unbind(b0);
    jit_isect_unbind(b1);
}

TEST_LLVM_FP32(03_occlusion_and_null) {
    FloatL cx = dr::opaque<FloatL>(0.f), cy = dr::opaque<FloatL>(0.f),
           cz = dr::opaque<FloatL>(0.f), r = dr::opaque<FloatL>(1.f);
    uint32_t func = record_sphere(cx, cy, cz, r);

    FakeScene scene;
    JitIsectBinding *binding = jit_isect_bind(func, (uintptr_t) &scene, 0, 0, nullptr);
    scene.geoms.push_back({ binding, 1 });
    jit_var_dec_ref(func);

    RayGrid rays;

    // Plain occlusion query
    TraceResult res = trace(scene, rays.ox, rays.oy, rays.oz, rays.dx,
                            rays.dy, rays.dz, rays.tmax, true);
    for (size_t i = 0; i < RayGrid::n; ++i) {
        bool hit = !std::isinf(ref_sphere(rays.x[i], rays.y[i], -5.f, 0.f, 0.f, 0.f, 1.f));
        jit_assert(res.valid.read(i) == hit);
        jit_assert(res.flags.read(i) == 0);
    }

    // Rays that skip null geometry are not blocked by a null shape, but they
    // record the encounter. The binding flag is read at trace time, so the
    // same unit serves both cases.
    binding->flags = JitIsectFlagNull;
    res = trace(scene, rays.ox, rays.oy, rays.oz, rays.dx, rays.dy, rays.dz,
                rays.tmax, true, JitIsectRaySkipNull);
    for (size_t i = 0; i < RayGrid::n; ++i) {
        bool hit = !std::isinf(ref_sphere(rays.x[i], rays.y[i], -5.f, 0.f, 0.f, 0.f, 1.f));
        jit_assert(!res.valid.read(i));
        uint32_t expected = (uint32_t) JitIsectRaySkipNull |
                            (hit ? (uint32_t) JitIsectRayHasNull : 0u);
        jit_assert(res.flags.read(i) == expected);
    }

    // Without the skip bit, a null shape occludes like any other
    res = trace(scene, rays.ox, rays.oy, rays.oz, rays.dx, rays.dy, rays.dz,
                rays.tmax, true, 0);
    for (size_t i = 0; i < RayGrid::n; ++i) {
        bool hit = !std::isinf(ref_sphere(rays.x[i], rays.y[i], -5.f, 0.f, 0.f, 0.f, 1.f));
        jit_assert(res.valid.read(i) == hit);
    }

    jit_isect_unbind(binding);
}

/// Four spheres in one geometry, radii gathered from a table by primitive index
static uint32_t record_sphere_table(const FloatL &radii) {
    uint32_t out[4], func;
    {
        IsectInputs in;
        FloatL r = dr::gather<FloatL>(radii, in.prim),
               cx = FloatL::steal(jit_var_cast(in.prim.index(), VarType::Float32, 0)) * FloatL(3.f),
               zero(0.f);
        sphere_body(in, cx, zero, zero, r, out);
    }
    func = jit_isect_end("sphere_table", out);

    for (uint32_t i = 0; i < 4; ++i)
        jit_var_dec_ref(out[i]);

    return func;
}

TEST_LLVM_FP32(04_primitive_table_and_replace) {
    float radii_1[4] = { 1.f, 0.5f, 0.25f, 0.75f },
          radii_2[4] = { 0.3f, 0.6f, 0.9f, 1.2f };
    FloatL table_1 = FloatL::copy(radii_1, 4);
    uint32_t func = record_sphere_table(table_1);

    FakeScene scene;
    JitIsectBinding *binding = jit_isect_bind(func, (uintptr_t) &scene, 0, 0, nullptr);
    scene.geoms.push_back({ binding, 4 });
    jit_var_dec_ref(func);

    // One ray through the center of each sphere, and one that misses
    float x[5] = { 0.f, 3.f, 6.f, 9.f, 12.f };
    FloatL ox = FloatL::copy(x, 5), oy(0.f), oz(-5.f), dx(0.f), dy(0.f),
           dz(1.f), tmax(100.f);

    TraceResult res = trace(scene, ox, oy, oz, dx, dy, dz, tmax);
    for (size_t i = 0; i < 5; ++i) {
        bool hit = i < 4;
        jit_assert(res.valid.read(i) == hit);
        if (hit) {
            jit_assert(approx(res.t.read(i), 5.f - radii_1[i]));
            jit_assert(res.prim.read(i) == i);
        }
    }

    // Replace the table through a new function and binding
    FloatL table_2 = FloatL::copy(radii_2, 4);
    uint32_t func_2 = record_sphere_table(table_2);
    jit_isect_unbind(binding);
    binding = jit_isect_bind(func_2, (uintptr_t) &scene, 0, 0, nullptr);
    scene.geoms[0].binding = binding;
    jit_var_dec_ref(func_2);

    res = trace(scene, ox, oy, oz, dx, dy, dz, tmax);
    for (size_t i = 0; i < 4; ++i) {
        jit_assert(res.valid.read(i));
        jit_assert(approx(res.t.read(i), 5.f - radii_2[i]));
    }

    jit_isect_unbind(binding);
}

TEST_LLVM_FP32(05_freeze_data_change) {
    // A frozen trace replays after the sphere's captured center changes,
    // as long as the intersection body is unchanged. This is the parameter
    // update case: the shape is re-recorded and re-bound outside the frozen
    // function, and replay picks up the new data block.
    JitBackend backend = JitBackend::LLVM;
    RayGrid rays;

    FloatL cz = dr::opaque<FloatL>(0.f);
    uint32_t func = record_sphere(dr::opaque<FloatL>(0.f),
                                  dr::opaque<FloatL>(0.f), cz,
                                  dr::opaque<FloatL>(1.f));
    FakeScene scene;
    JitIsectBinding *binding = jit_isect_bind(func, (uintptr_t) &scene, 0, 0, nullptr);
    scene.geoms.push_back({ binding, 1 });
    jit_var_dec_ref(func);

    // Record a trace into a frozen function. The inputs are the ray arrays
    // and the scene / callback resource handles; the output is evaluated
    // inside the recording, which makes the trace kernel part of it.
    uint32_t scene_h = jit_var_mem_map(backend, VarType::UInt64, &scene, 1, 0),
             func_h = jit_var_mem_map(backend, VarType::UInt64,
                                      (void *) fake_intersect, 1, 0);
    uint32_t inputs[9] = { rays.ox.index(), rays.oy.index(), rays.oz.index(),
                           rays.dx.index(), rays.dy.index(), rays.dz.index(),
                           rays.tmax.index(), scene_h, func_h };
    jit_freeze_start(backend, inputs, 9);
    TraceResult res = trace(scene, rays.ox, rays.oy, rays.oz, rays.dx, rays.dy,
                            rays.dz, rays.tmax, false, 0, scene_h, func_h);
    uint32_t out = res.t.index();
    jit_var_eval(out);
    Recording *rec = jit_freeze_stop(backend, &out, 1);

    for (size_t i = 0; i < RayGrid::n; ++i)
        jit_assert(approx(res.t.read(i),
                          ref_sphere(rays.x[i], rays.y[i], -5.f, 0.f, 0.f, 0.f, 1.f)));

    // Move the sphere to z = 1, re-recording the identical body and rebinding
    uint32_t func2 = record_sphere(dr::opaque<FloatL>(0.f),
                                   dr::opaque<FloatL>(0.f),
                                   dr::opaque<FloatL>(1.f),
                                   dr::opaque<FloatL>(1.f));
    jit_isect_unbind(binding);
    binding = jit_isect_bind(func2, (uintptr_t) &scene, 0, 0, nullptr);
    scene.geoms[0].binding = binding;
    jit_var_dec_ref(func2);

    jit_assert(jit_freeze_dry_run(rec, inputs));
    uint32_t replayed = 0;
    jit_freeze_replay(rec, inputs, &replayed);
    FloatL t = FloatL::steal(replayed);
    jit_var_eval(t.index());
    for (size_t i = 0; i < RayGrid::n; ++i)
        jit_assert(approx(t.read(i),
                          ref_sphere(rays.x[i], rays.y[i], -5.f, 0.f, 0.f, 1.f, 1.f)));

    jit_freeze_destroy(rec);
    jit_isect_unbind(binding);
    jit_var_dec_ref(scene_h);
    jit_var_dec_ref(func_h);
}

TEST_LLVM_FP32(06_freeze_incompatible) {
    // Replacing the body (not just the data) invalidates a frozen kernel:
    // the dry run must fail so that the caller re-traces.
    JitBackend backend = JitBackend::LLVM;
    RayGrid rays;

    uint32_t func = record_sphere(dr::opaque<FloatL>(0.f),
                                  dr::opaque<FloatL>(0.f),
                                  dr::opaque<FloatL>(0.f),
                                  dr::opaque<FloatL>(1.f));
    FakeScene scene;
    JitIsectBinding *binding = jit_isect_bind(func, (uintptr_t) &scene, 0, 0, nullptr);
    scene.geoms.push_back({ binding, 1 });
    jit_var_dec_ref(func);

    uint32_t scene_h = jit_var_mem_map(backend, VarType::UInt64, &scene, 1, 0),
             func_h = jit_var_mem_map(backend, VarType::UInt64,
                                      (void *) fake_intersect, 1, 0);
    uint32_t inputs[9] = { rays.ox.index(), rays.oy.index(), rays.oz.index(),
                           rays.dx.index(), rays.dy.index(), rays.dz.index(),
                           rays.tmax.index(), scene_h, func_h };
    jit_freeze_start(backend, inputs, 9);
    TraceResult res = trace(scene, rays.ox, rays.oy, rays.oz, rays.dx, rays.dy,
                            rays.dz, rays.tmax, false, 0, scene_h, func_h);
    uint32_t out = res.t.index();
    jit_var_eval(out);
    Recording *rec = jit_freeze_stop(backend, &out, 1);

    // A sphere table has a different body, so the kernel's unit no longer
    // matches the scene's binding
    float radii_data[1] = { 1.f };
    FloatL radii = FloatL::copy(radii_data, 1);
    uint32_t func2 = record_sphere_table(radii);
    jit_isect_unbind(binding);
    binding = jit_isect_bind(func2, (uintptr_t) &scene, 0, 0, nullptr);
    scene.geoms[0].binding = binding;
    jit_var_dec_ref(func2);

    jit_assert(!jit_freeze_dry_run(rec, inputs));

    jit_freeze_destroy(rec);
    jit_isect_unbind(binding);
    jit_var_dec_ref(scene_h);
    jit_var_dec_ref(func_h);
}

TEST_LLVM_FP32(07_record_errors) {
    JitBackend backend = JitBackend::LLVM;

    // The body must not depend on the array index
    {
        uint32_t out[4];
        bool caught = false;
        {
            IsectInputs in;
            UInt32L counter = UInt32L::steal(jit_var_counter(backend, 1));
            FloatL t = in.tmax + FloatL::steal(jit_var_cast(counter.index(), VarType::Float32, 0));
            MaskL hit(true);
            UInt32L a(0u);
            out[0] = jit_var_inc_ref(hit.index());
            out[1] = jit_var_inc_ref(t.index());
            out[2] = jit_var_inc_ref(a.index());
            out[3] = jit_var_inc_ref(a.index());
        }
        try {
            jit_isect_end("bad", out);
        } catch (const std::exception &e) {
            caught = strstr(e.what(), "array index") != nullptr;
        }
        jit_assert(caught);
        for (uint32_t i = 0; i < 4; ++i)
            jit_var_dec_ref(out[i]);
    }

    // A side effect (scatter) in the body is rejected
    {
        uint32_t out[4];
        bool caught = false;
        FloatL buf = dr::zeros<FloatL>(4);
        jit_var_eval(buf.index());
        {
            IsectInputs in;
            dr::scatter(buf, in.tmax, UInt32L(0u), MaskL(true));
            MaskL hit(true);
            UInt32L a(0u);
            out[0] = jit_var_inc_ref(hit.index());
            out[1] = jit_var_inc_ref(in.tmax.index());
            out[2] = jit_var_inc_ref(a.index());
            out[3] = jit_var_inc_ref(a.index());
        }
        try {
            jit_isect_end("bad", out);
        } catch (const std::exception &e) {
            caught = strstr(e.what(), "side effect") != nullptr;
        }
        jit_assert(caught);
        for (uint32_t i = 0; i < 4; ++i)
            jit_var_dec_ref(out[i]);
    }
}
