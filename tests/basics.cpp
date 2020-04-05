#include "test.h"
#if 0

TEST_BOTH(01_creation_destruction) {
    // Checks simple reference counting of a variable
    Float value(1234);
    (void) value;
}

TEST_BOTH(02_fill_and_print) {
    /// Checks array initialization from a given pointer, jitc_fill(), and stringification
    jitc_log(Info, "  int8_t: %s", full<Array<  int8_t>>(-111, 5).str());
    jitc_log(Info, " uint8_t: %s", full<Array< uint8_t>>( 222, 5).str());
    jitc_log(Info, " int16_t: %s", full<Array< int16_t>>(-1111, 5).str());
    jitc_log(Info, "uint16_t: %s", full<Array<uint16_t>>( 2222, 5).str());
    jitc_log(Info, " int32_t: %s", full<Array< int32_t>>(-1111111111, 5).str());
    jitc_log(Info, "uint32_t: %s", full<Array<uint32_t>>( 2222222222, 5).str());
    jitc_log(Info, " int64_t: %s", full<Array< int64_t>>(-1111111111111111111, 5).str());
    jitc_log(Info, "uint64_t: %s", full<Array<uint64_t>>( 2222222222222222222, 5).str());
    jitc_log(Info, "   float: %s", full<Array<   float>>(1.f / 3.f, 5).str());
    jitc_log(Info, "  double: %s", full<Array<  double>>(1.0 / 3.0, 5).str());
}

TEST_BOTH(03_eval_scalar) {
    /// Checks that we can evaluate a simple kernel, and that kernels are reused
    Float value(1234);
    set_label(value, "my_value");
    jitc_log(Info, "value=%s", value.str());

    value = Float(1234);
    set_label(value, "my_value");
    jitc_log(Info, "value=%s", value.str());
}

TEST_BOTH(04_eval_scalar_csa) {
    /// Checks common subexpression elimination
    Float value_1(1234),
          value_2(1235),
          value_3(1234),
          value_4 = value_1 + value_2,
          value_5 = value_1 + value_3,
          value_6 = value_1 + value_2;
    jitc_eval(value_1, value_2, value_3, value_4, value_5, value_6);
    jitc_log(Info, "value_1=%s", value_1.str());
    jitc_log(Info, "value_2=%s", value_2.str());
    jitc_log(Info, "value_3=%s", value_3.str());
    jitc_log(Info, "value_4=%s", value_4.str());
    jitc_log(Info, "value_5=%s", value_5.str());
    jitc_log(Info, "value_6=%s", value_6.str());
}

TEST_BOTH(05_argument_out) {
    /// Test kernels with very many outputs that exceed the max. size of the BOTH parameter table
    scoped_set_log_level ssll(LogLevel::Info);
    /* With reduced log level */ {
        Int32 value[1024];
        for (int i = 1; i < 1024; i *= 3) {
            Int32 out = 0;
            for (int j = 0; j < i; ++j) {
                value[j] = j;
                out += value[j].schedule();
            }
            jitc_log(Info, "value=%s vs %u", out.str(), i * (i - 1) / 2);
        }
    }
}

TEST_BOTH(06_argument_inout) {
    /// Test kernels with very many inputs that exceed the max. size of the BOTH parameter table
    scoped_set_log_level ssll(LogLevel::Info);
    /* With reduced log level */ {
        Int32 value[1024];
        for (int i = 1; i < 1024; i *= 3) {
            Int32 out = 0;
            for (int j = 0; j < i; ++j) {
                if (!value[j].valid())
                    value[j] = j;
                out += value[j].schedule();
            }
            jitc_log(Info, "value=%s vs %u", out.str(), i * (i - 1) / 2);
        }
    }
}

TEST_BOTH(07_arange) {
    /// Tests arange, and dispatch to parallel streams
    UInt32 x = arange<UInt32>(1024);
    UInt32 y = arange<UInt32>(3, 512, 7);
    jitc_schedule(x, y);
    jitc_log(Info, "value=%s", x.str());
    jitc_log(Info, "value=%s", y.str());

    using Int64 = Array<int64_t>;
    Int64 x2 = arange<Int64>(1024);
    Int64 y2 = arange<Int64>(-3, 506, 7);
    jitc_schedule(x2, y2);
    jitc_log(Info, "value=%s", x2.str());
    jitc_log(Info, "value=%s", y2.str());
}

TEST_BOTH(08_conv) {
    /* UInt32 */ {
        auto src = arange<Array<uint32_t>>(1024);
        Array<uint32_t> x_u32(src);
        Array<int32_t> x_i32(src);
        Array<uint64_t> x_u64(src);
        Array<int64_t> x_i64(src);
        Array<float> x_f32(src);
        Array<double> x_f64(src);

        jitc_schedule(x_u32, x_i32, x_u64, x_i64, x_f32, x_f64);
        jitc_log(Info, "value=%s", x_u32.str());
        jitc_log(Info, "value=%s", x_i32.str());
        jitc_log(Info, "value=%s", x_u64.str());
        jitc_log(Info, "value=%s", x_i64.str());
        jitc_log(Info, "value=%s", x_f32.str());
        jitc_log(Info, "value=%s", x_f64.str());
    }

    /* Int32 */ {
        auto src = arange<Array<int32_t>>(1024) - 512;
        Array<int32_t> x_i32(src);
        Array<int64_t> x_i64(src);
        Array<float> x_f32(src);
        Array<double> x_f64(src);

        jitc_schedule(x_i32, x_i64, x_f32, x_f64);
        jitc_log(Info, "value=%s", x_i32.str());
        jitc_log(Info, "value=%s", x_i64.str());
        jitc_log(Info, "value=%s", x_f32.str());
        jitc_log(Info, "value=%s", x_f64.str());
    }

    /* UInt64 */ {
        auto src = arange<Array<uint64_t>>(1024);
        Array<uint32_t> x_u32(src);
        Array<int32_t> x_i32(src);
        Array<uint64_t> x_u64(src);
        Array<int64_t> x_i64(src);
        Array<float> x_f32(src);
        Array<double> x_f64(src);

        jitc_schedule(x_u32, x_i32, x_u64, x_i64, x_f32, x_f64);
        jitc_log(Info, "value=%s", x_u32.str());
        jitc_log(Info, "value=%s", x_i32.str());
        jitc_log(Info, "value=%s", x_u64.str());
        jitc_log(Info, "value=%s", x_i64.str());
        jitc_log(Info, "value=%s", x_f32.str());
        jitc_log(Info, "value=%s", x_f64.str());
    }

    /* Int64 */ {
        auto src = arange<Array<int64_t>>(1024) - 512;
        Array<int32_t> x_i32(src);
        Array<int64_t> x_i64(src);
        Array<float> x_f32(src);
        Array<double> x_f64(src);

        jitc_schedule(x_i32, x_i64, x_f32, x_f64);
        jitc_log(Info, "value=%s", x_i32.str());
        jitc_log(Info, "value=%s", x_i64.str());
        jitc_log(Info, "value=%s", x_f32.str());
        jitc_log(Info, "value=%s", x_f64.str());
    }

    /* Float */ {
        auto src = arange<Array<float>>(1024) - 512;
        Array<int32_t> x_i32(src);
        Array<int64_t> x_i64(src);
        Array<float> x_f32(src);
        Array<double> x_f64(src);

        jitc_schedule(x_i32, x_i64, x_f32, x_f64);
        jitc_log(Info, "value=%s", x_i32.str());
        jitc_log(Info, "value=%s", x_i64.str());
        jitc_log(Info, "value=%s", x_f32.str());
        jitc_log(Info, "value=%s", x_f64.str());
    }

    /* Double */ {
        auto src = arange<Array<double>>(1024) - 512;
        Array<int32_t> x_i32(src);
        Array<int64_t> x_i64(src);
        Array<float> x_f32(src);
        Array<double> x_f64(src);

        jitc_schedule(x_i32, x_i64, x_f32, x_f64);
        jitc_log(Info, "value=%s", x_i32.str());
        jitc_log(Info, "value=%s", x_i64.str());
        jitc_log(Info, "value=%s", x_f32.str());
        jitc_log(Info, "value=%s", x_f64.str());
    }
}

TEST_BOTH(09_fma) {
    /* Float case */ {
        Float a(1, 2, 3, 4);
        Float b(3, 8, 1, 5);
        Float c(9, 1, 3, 0);

        Float d = fmadd(a, b, c);
        Float e = fmadd(d, b, -c);
        jitc_schedule(d, e);
        jitc_log(Info, "value=%s", d.str());
        jitc_log(Info, "value=%s", e.str());
    }

    using Double = Array<double>;
    /* Double case */ {
        Double a(1, 2, 3, 4);
        Double b(3, 8, 1, 5);
        Double c(9, 1, 3, 0);

        Double d = fmadd(a, b, c);
        Double e = fmadd(d, b, -c);
        jitc_schedule(d, e);
        jitc_log(Info, "value=%s", d.str());
        jitc_log(Info, "value=%s", e.str());
    }
}

TEST_BOTH(10_sqrt) {
    Float x = sqrt(arange<Float>(10));
    jitc_log(Info, "value=%s", x.str());
}

TEST_BOTH(11_mask) {
    Float x = arange<Float>(10);
    auto mask = x > 5;
    x = select(mask, -x, x);
    jitc_schedule(mask, x);
    jitc_log(Info, "value=%s", x.str());
    jitc_log(Info, "mask=%s", mask.str());
    Float y = select(mask, Float(1.f), Float(2.f));
    jitc_log(Info, "value_2=%s", y.str());

    Array<bool> mask_scalar_t = true;
    Array<bool> mask_scalar_f = false;
    jitc_eval(mask_scalar_t, mask_scalar_f);
    Float a = select(mask_scalar_t, x, Float(0));
    Float b = select(mask_scalar_f, x, Float(0));
    jitc_schedule(a, b);
    jitc_log(Info, "value_3=%s", a.str());
    jitc_log(Info, "value_4=%s", b.str());
}

TEST_BOTH(12_binop) {
    UInt32 a(1, 1234);
    Float  b(1.f, 1234.f);
    Array<bool> c(true, false);

    a = ~a;
    b = ~b;
    c = ~c;

    jitc_schedule(a, b, c);
    jitc_log(Info, "XOR: value_1=%s", a.str());
    jitc_log(Info, "XOR: value_2=%s", b.str());
    jitc_log(Info, "XOR: value_3=%s", c.str());

    a = ~a;
    b = ~b;
    c = ~c;

    jitc_schedule(a, b, c);
    jitc_log(Info, "XOR2: value_1=%s", a.str());
    jitc_log(Info, "XOR2: value_2=%s", b.str());
    jitc_log(Info, "XOR2: value_3=%s", c.str());

    int32_t x = 0x7fffffff;
    float y;
    memcpy(&y, &x, 4);

    UInt32 a2(0, 0xFFFFFFFF);
    Float  b2(0.f, y);
    Array<bool> c2(false, true);

    auto a3 = a | a2;
    auto b3 = b | b2;
    auto c3 = c | c2;

    jitc_schedule(a3, b3, c3);
    jitc_log(Info, "OR: value_1=%s", a3.str());
    jitc_log(Info, "OR: value_2=%s", b3.str());
    jitc_log(Info, "OR: value_3=%s", c3.str());

    auto a4 = a & a2;
    auto b4 = b & b2;
    auto c4 = c & c2;

    jitc_schedule(a4, b4, c4);
    jitc_log(Info, "AND: value_1=%s", a4.str());
    jitc_log(Info, "AND: value_2=%s", b4.str());
    jitc_log(Info, "AND: value_3=%s", c4.str());

    auto a5 = a ^ a2;
    auto b5 = b ^ b2;
    auto c5 = c ^ c2;

    jitc_schedule(a5, b5, c5);
    jitc_log(Info, "XOR: value_1=%s", a5.str());
    jitc_log(Info, "XOR: value_2=%s", b5.str());
    jitc_log(Info, "XOR: value_3=%s", c5.str());
}

TEST_BOTH(14_scatter_gather) {
    Int32 l = (-arange<Int32>(1024)).schedule();
    UInt32 index = UInt32(34, 62, 75, 2);
    Int32 value = gather(l, index);
    jitc_log(Info, "%s", value.str());
    jitc_log(Info, "%s", l.str());
    scatter(l, value * 3, index);
    value = gather(l, index);
    jitc_log(Info, "%s", value.str());
    jitc_log(Info, "%s", l.str());
}

TEST_BOTH(15_round) {
    /* Single precision */ {
        Float x(.4f, .5f, .6f, -.4f, -.5f, -.6f);

        Float x_f  = floor(x),
              x_c  = ceil(x),
              x_t  = trunc(x),
              x_r  = round(x),
              x_mi = min(x_f, x_c),
              x_ma = max(x_f, x_c);

        jitc_schedule(x_f, x_c, x_t, x_r, x_mi, x_ma);
        jitc_log(Info, "floor: %s", x_f.str());
        jitc_log(Info, "ceil:  %s", x_c.str());
        jitc_log(Info, "trunc: %s", x_t.str());
        jitc_log(Info, "round: %s", x_r.str());
        jitc_log(Info, "min:   %s", x_mi.str());
        jitc_log(Info, "max:   %s", x_ma.str());
    }

    /* Double precision */ {
        using Double = Array<double>;

        Double x(.4f, .5f, .6f, -.4f, -.5f, -.6f);

        Double x_f  = floor(x),
               x_c  = ceil(x),
               x_t  = trunc(x),
               x_r  = round(x),
               x_mi = min(x_f, x_c),
               x_ma = max(x_f, x_c);

        jitc_schedule(x_f, x_c, x_t, x_r, x_mi, x_ma);
        jitc_log(Info, "floor: %s", x_f.str());
        jitc_log(Info, "ceil:  %s", x_c.str());
        jitc_log(Info, "trunc: %s", x_t.str());
        jitc_log(Info, "round: %s", x_r.str());
        jitc_log(Info, "min:   %s", x_mi.str());
        jitc_log(Info, "max:   %s", x_ma.str());
    }
}

TEST_LLVM(16_pointer_registry) {
    const char *key_1 = "MyKey1";
    const char *key_2 = "MyKey2";

    jitc_assert(jitc_registry_get_max(key_1) == 0u);
    jitc_assert(jitc_registry_get_max(key_2) == 0u);

    uint32_t id_1 = jitc_registry_put(key_1, (void *) 0xA);
    jitc_assert(jitc_registry_get_max(key_1) == 1u);
    jitc_assert(jitc_registry_get_max(key_2) == 0u);

    uint32_t id_2 = jitc_registry_put(key_1, (void *) 0xB);
    jitc_assert(jitc_registry_get_max(key_1) == 2u);
    jitc_assert(jitc_registry_get_max(key_2) == 0u);

    uint32_t id_a = jitc_registry_put(key_2, (void *) 0xC);
    jitc_assert(jitc_registry_get_max(key_1) == 2u);
    jitc_assert(jitc_registry_get_max(key_2) == 1u);

    uint32_t id_3 = jitc_registry_put(key_1, (void *) 0xD);
    jitc_assert(jitc_registry_get_max(key_1) == 3u);
    jitc_assert(jitc_registry_get_max(key_2) == 1u);

    jitc_assert(id_1 == 1u);
    jitc_assert(id_2 == 2u);
    jitc_assert(id_a == 1u);
    jitc_assert(id_3 == 3u);

    jitc_assert(jitc_registry_get_domain((void *) 0xA) == key_1);
    jitc_assert(jitc_registry_get_domain((void *) 0xB) == key_1);
    jitc_assert(jitc_registry_get_domain((void *) 0xC) == key_2);
    jitc_assert(jitc_registry_get_domain((void *) 0xD) == key_1);

    jitc_assert(jitc_registry_get_id((void *) 0xA) == 1u);
    jitc_assert(jitc_registry_get_id((void *) 0xB) == 2u);
    jitc_assert(jitc_registry_get_id((void *) 0xC) == 1u);
    jitc_assert(jitc_registry_get_id((void *) 0xD) == 3u);

    jitc_assert(jitc_registry_get_ptr(key_1, 1) == (void *) 0xA);
    jitc_assert(jitc_registry_get_ptr(key_1, 2) == (void *) 0xB);
    jitc_assert(jitc_registry_get_ptr(key_2, 1) == (void *) 0xC);
    jitc_assert(jitc_registry_get_ptr(key_1, 3) == (void *) 0xD);

    jitc_assert(jitc_registry_get_max(key_1) == 3u);
    jitc_assert(jitc_registry_get_max(key_2) == 1u);

    jitc_registry_remove((void *) 0xA);
    jitc_registry_remove((void *) 0xB);
    jitc_registry_remove((void *) 0xC);

    id_1 = jitc_registry_put(key_1, (void *) 0xE);
    id_2 = jitc_registry_put(key_1, (void *) 0xF);
    id_a = jitc_registry_put(key_1, (void *) 0x1);

    jitc_assert(id_1 == 2u);
    jitc_assert(id_2 == 1u);
    jitc_assert(id_a == 4u);

    jitc_assert(jitc_registry_get_domain((void *) 0xE) == key_1);
    jitc_assert(jitc_registry_get_domain((void *) 0xF) == key_1);
    jitc_assert(jitc_registry_get_domain((void *) 0x1) == key_1);

    jitc_assert(jitc_registry_get_id((void *) 0xE) == 2u);
    jitc_assert(jitc_registry_get_id((void *) 0xF) == 1u);
    jitc_assert(jitc_registry_get_id((void *) 0x1) == 4u);

    jitc_assert(jitc_registry_get_ptr(key_1, 1) == (void *) 0xF);
    jitc_assert(jitc_registry_get_ptr(key_1, 2) == (void *) 0xE);
    jitc_assert(jitc_registry_get_ptr(key_1, 3) == (void *) 0xD);
    jitc_assert(jitc_registry_get_ptr(key_1, 4) == (void *) 0x1);

    jitc_registry_remove((void *) 0xD);
    jitc_registry_remove((void *) 0xE);
    jitc_registry_remove((void *) 0xF);

    jitc_assert(jitc_registry_get_max(key_1) == 4u);
    jitc_assert(jitc_registry_get_max(key_2) == 1u);

    jitc_registry_trim();

    jitc_registry_remove((void *) 0x1);

    jitc_assert(jitc_registry_get_max(key_1) == 4u);
    jitc_assert(jitc_registry_get_max(key_2) == 0u);

    jitc_registry_trim();

    jitc_assert(jitc_registry_get_max(key_1) == 0u);
    jitc_assert(jitc_registry_get_max(key_2) == 0u);
}

TEST_BOTH(17_scatter_gather_mask) {
    using Mask    = Array<bool>;
    Int32 l       = arange<Int32>(1022);
    set_label(l, "l");
    Mask result   = neq(l & Int32(1), 0);

    set_label(result, "result");
    Int32 l2      = arange<Int32>(510);
    set_label(l2, "l2");
    Mask even = gather(result, l2*2);
    set_label(even, "even");
    Mask odd  = gather(result, l2*2 + 1);
    set_label(odd, "odd");
    jitc_log(Info, "Mask  : %s", result.str());
    jitc_log(Info, "Even  : %s", even.str());
    jitc_log(Info, "Odd   : %s", odd.str());

    scatter(result, odd, l2 * 2);
    scatter(result, even, l2 * 2 + 1);
    jitc_log(Info, "Mask: %s", result.str());
}

TEST_BOTH(18_mask_propagation) {
    using Mask    = Array<bool>;

    Mask t(true), f(false), g(true, false);

    jitc_assert( t.is_literal_one());
    jitc_assert(!t.is_literal_zero());
    jitc_assert(!f.is_literal_one());
    jitc_assert( f.is_literal_zero());
    jitc_assert(!g.is_literal_one());
    jitc_assert(!g.is_literal_zero());

    jitc_assert((t && f).is_literal_zero());
    jitc_assert((f && t).is_literal_zero());
    jitc_assert((g && f).is_literal_zero());
    jitc_assert((f && g).is_literal_zero());

    jitc_assert((t || f).is_literal_one());
    jitc_assert((f || t).is_literal_one());
    jitc_assert((t || g).is_literal_one());
    jitc_assert((g || t).is_literal_one());

    jitc_assert((t ^ f).is_literal_one());
    jitc_assert((f ^ t).is_literal_one());
    jitc_assert((f ^ g).index() == g.index());
    jitc_assert((g ^ f).index() == g.index());

    UInt32 a(1, 2), b(3, 4);
    jitc_assert(select(f, a, b).index() == b.index());
    jitc_assert(select(t, a, b).index() == a.index());
}

TEST_BOTH(19_register_ptr) {
    uint32_t idx_1 = jitc_var_copy_ptr((const void *) 0x01, 0),
             idx_2 = jitc_var_copy_ptr((const void *) 0x02, 0),
             idx_3 = jitc_var_copy_ptr((const void *) 0x01, 0);

    jitc_assert(idx_1 == 1);
    jitc_assert(idx_2 == 2);
    jitc_assert(idx_3 == 1);

    jitc_var_dec_ref_ext(idx_1);
    jitc_var_dec_ref_ext(idx_2);
    jitc_var_dec_ref_ext(idx_3);

    idx_1 = jitc_var_copy_ptr((const void *) 0x01, 0);
    jitc_assert(idx_1 == 3);
    jitc_var_dec_ref_ext(idx_1);
}

TEST_BOTH(20_reinterpret_cast) {
    UInt32 result = reinterpret_array<UInt32>(Float(1.f, 2.f, 3.f));
    jitc_log(Info, "As integer: %s", result.str());

    Float result2 = reinterpret_array<Float>(result);
    jitc_log(Info, "As float: %s", result2.str());
}

TEST_BOTH(21_shifts) {
    UInt32 x(1234, (uint32_t) -1234);
    Int32  y(1234,            -1234);

    UInt32 xs1 = x >> 1, xs2 = x << 1;
    Int32  ys1 = y >> 1, ys2 = y << 1;

    jitc_schedule(xs1, xs2, ys1, ys2);
    jitc_log(Info, "xs1 : %s", xs1.str());
    jitc_log(Info, "xs2 : %s", xs2.str());
    jitc_log(Info, "ys1 : %s", ys1.str());
    jitc_log(Info, "ys2 : %s", ys2.str());
}

TEST_BOTH(22_and_or_mask) {
    using Mask = Array<bool>;

    UInt32 x(0, 1);
    Int32  y(0, 1);
    Float  z(0.f, 1.f);

    Mask m(true, false);

    UInt32 x_o = x | m, x_a = x & m;
    Int32  y_o = y | m, y_a = y & m;
    Float  z_o = z | m, z_a = z & m;
    z_o = abs(z_o);

    jitc_schedule(x_o, x_a, y_o, y_a, z_o, z_a);
    jitc_log(Info, "x_o : %s", x_o.str());
    jitc_log(Info, "x_a : %s", x_a.str());
    jitc_log(Info, "y_o : %s", y_o.str());
    jitc_log(Info, "y_a : %s", y_a.str());
    jitc_log(Info, "z_o : %s", z_o.str());
    jitc_log(Info, "z_a : %s", z_a.str());

    m = Mask(true);
    x_o = x | m; x_a = x & m;
    y_o = y | m; y_a = y & m;
    z_o = z | m; z_a = z & m;
    z_o = abs(z_o);

    jitc_schedule(x_o, x_a, y_o, y_a, z_o, z_a);
    jitc_log(Info, "x_o : %s", x_o.str());
    jitc_log(Info, "x_a : %s", x_a.str());
    jitc_log(Info, "y_o : %s", y_o.str());
    jitc_log(Info, "y_a : %s", y_a.str());
    jitc_log(Info, "z_o : %s", z_o.str());
    jitc_log(Info, "z_a : %s", z_a.str());

    m = Mask(false);
    x_o = x | m; x_a = x & m;
    y_o = y | m; y_a = y & m;
    z_o = z | m; z_a = z & m;

    jitc_schedule(x_o, x_a, y_o, y_a, z_o, z_a);
    jitc_log(Info, "x_o : %s", x_o.str());
    jitc_log(Info, "x_a : %s", x_a.str());
    jitc_log(Info, "y_o : %s", y_o.str());
    jitc_log(Info, "y_a : %s", y_a.str());
    jitc_log(Info, "z_o : %s", z_o.str());
    jitc_log(Info, "z_a : %s", z_a.str());
}

template <typename T, typename T2, typename S = typename T::Value>
T poly2(const T &x, const T2 &c0, const T2 &c1, const T2 &c2) {
    T x2 = x * x;
    return fmadd(x2, S(c2), fmadd(x, S(c1), S(c0)));
}

template <typename Value, typename Mask, typename Int>
void sincos_approx(const Value &x, Value &s_out, Value &c_out) {
    /* Joint sine & cosine function approximation based on CEPHES.
       Excellent accuracy in the domain |x| < 8192

       Redistributed under a BSD license with permission of the author, see
       https://github.com/deepmind/torch-cephes/blob/master/LICENSE.txt

     - sin (in [-8192, 8192]):
       * avg abs. err = 6.61896e-09
       * avg rel. err = 1.37888e-08
          -> in ULPs  = 0.166492
       * max abs. err = 5.96046e-08
         (at x=-8191.31)
       * max rel. err = 1.76826e-06
         -> in ULPs   = 19
         (at x=-6374.29)

     - cos (in [-8192, 8192]):
       * avg abs. err = 6.59965e-09
       * avg rel. err = 1.37432e-08
          -> in ULPs  = 0.166141
       * max abs. err = 5.96046e-08
         (at x=-8191.05)
       * max rel. err = 3.13993e-06
         -> in ULPs   = 47
         (at x=-6199.93)
    */

    using Scalar = float;

    Value xa = abs(x);

    /* Scale by 4/Pi and get the integer part */
    Int j(xa * Scalar(1.2732395447351626862));

    /* Map zeros to origin; if (j & 1) j += 1 */
    j = (j + uint32_t(1)) & uint32_t(~1u);

    /* Cast back to a floating point value */
    Value y(j);

    // Determine sign of result
    uint32_t Shift = sizeof(Scalar) * 8 - 3;
    Value sign_sin = reinterpret_array<Value>(j << Shift) ^ x;
    Value sign_cos = reinterpret_array<Value>((~(j - uint32_t(2))) << Shift);

    // Extended precision modular arithmetic
    y = xa - y * Scalar(0.78515625)
           - y * Scalar(2.4187564849853515625e-4)
           - y * Scalar(3.77489497744594108e-8);

    Value z = y * y, s, c;
    z |= eq(xa, std::numeric_limits<Scalar>::infinity());

    s = poly2(z, -1.6666654611e-1,
                  8.3321608736e-3,
                 -1.9515295891e-4) * z;

    c = poly2(z,  4.166664568298827e-2,
                 -1.388731625493765e-3,
                  2.443315711809948e-5) * z;

    s = fmadd(s, y, y);
    c = fmadd(c, z, fmadd(z, Scalar(-0.5), Scalar(1)));

    Mask polymask(eq(j & uint32_t(2), Int(0)));

    s_out = mulsign(select(polymask, s, c), sign_sin);
    c_out = mulsign(select(polymask, c, s), sign_cos);
}

TEST_LLVM(23_sincos) {
    using Mask = Array<bool>;
    Float x = linspace<Float>(0, 1, 10);
    Float xs, xc;
    sincos_approx<Float, Mask, Int32>(x, xs, xc);

    jitc_log(Info, "xs : %s", xs.str());
    jitc_log(Info, "xc : %s", xc.str());
}

TEST_BOTH(24_bitop) {
    UInt32 v(0, 1, 1234, 0xFFFFFFFF);

    UInt32 v_pop = popcnt(v),
           v_lz  = lzcnt(v),
           v_tz  = tzcnt(v);

    jitc_schedule(v_pop, v_lz, v_tz);
    jitc_log(Info, "orig : %s", v.str());
    jitc_log(Info, "pop  : %s", v_pop.str());
    jitc_log(Info, "lz   : %s", v_tz.str());
    jitc_log(Info, "tz   : %s", v_lz.str());
}

TEST_LLVM(25_wide_intrinsics) {
    jitc_llvm_set_target("skylake", "+avx2", 32);
    Float a = arange<Float>(64);
    Float b = arange<Float>(64) + 0.1f;
    Float c = max(a, b);
    Float d = fmadd(a, b, c);
    auto mask = eq(b, c);
    jitc_schedule(c, d, mask);
    jitc_log(Info, "value=%s", c.str());
    jitc_log(Info, "value=%s", d.str());
    jitc_assert(all(mask));
    jitc_llvm_set_target("skylake", "", 8);
}

TEST_LLVM(26_avx512_intrinsics) {
    jitc_llvm_set_target("skylake-avx512", "+avx512f,+avx512dq,+avx512vl", 32);
    Float a = arange<Float>(64);
    Float b = arange<Float>(64) + 0.1f;
    Float c = max(a, b);
    Float d = fmadd(a, b, c);
    auto mask = eq(b, c);
    jitc_schedule(c, d, mask);
    jitc_log(Info, "value=%s", c.str());
    jitc_log(Info, "value=%s", d.str());
    jitc_assert(all(mask));
    jitc_llvm_set_target("skylake", "", 8);
}

TEST_BOTH(27_avx512_intrinsics_round2int) {
    using Int64 = Array<int64_t>;
    using UInt64 = Array<uint64_t>;
    using Double = Array<double>;

    jitc_llvm_set_target("skylake-avx512", "+avx512f,+avx512dq,+avx512vl", 32);
    Float f(-1.1, -0.6f, -0.5f, -0.4f, 0.4f, 0.5f, 0.6f, 1.1, (float) 0xffffffff);
    Double d(f);
    jitc_eval();

    {
        auto f_i32 = ceil2int<Int32> (f);
        auto f_u32 = ceil2int<UInt32>(f);
        auto f_i64 = ceil2int<Int64> (f);
        auto f_u64 = ceil2int<UInt64>(f);
        auto d_i32 = ceil2int<Int32> (d);
        auto d_u32 = ceil2int<UInt32>(d);
        auto d_i64 = ceil2int<Int64> (d);
        auto d_u64 = ceil2int<UInt64>(d);
        jitc_schedule(f_i32, f_u32, f_i64, f_u64, d_i32, d_u32, d_i64, d_u64);
        jitc_log(Info, "input=%s", f.str());
        jitc_log(Info, "ceil_f_i32=%s", f_i32.str());
        jitc_log(Info, "ceil_f_i64=%s", f_i64.str());
        jitc_log(Info, "ceil_d_i32=%s", d_i32.str());
        jitc_log(Info, "ceil_d_i64=%s", d_i64.str());
        jitc_log(Info, "ceil_f_u32=%s", f_u32.str());
        jitc_log(Info, "ceil_f_u64=%s", f_u64.str());
        jitc_log(Info, "ceil_d_u32=%s", d_u32.str());
        jitc_log(Info, "ceil_d_u64=%s", d_u64.str());
    }

    {
        auto f_i32 = floor2int<Int32> (f);
        auto f_u32 = floor2int<UInt32>(f);
        auto f_i64 = floor2int<Int64> (f);
        auto f_u64 = floor2int<UInt64>(f);
        auto d_i32 = floor2int<Int32> (d);
        auto d_u32 = floor2int<UInt32>(d);
        auto d_i64 = floor2int<Int64> (d);
        auto d_u64 = floor2int<UInt64>(d);
        jitc_schedule(f_i32, f_u32, f_i64, f_u64, d_i32, d_u32, d_i64, d_u64);
        jitc_log(Info, "input=%s", f.str());
        jitc_log(Info, "floor_f_i32=%s", f_i32.str());
        jitc_log(Info, "floor_f_i64=%s", f_i64.str());
        jitc_log(Info, "floor_d_i32=%s", d_i32.str());
        jitc_log(Info, "floor_d_i64=%s", d_i64.str());
        jitc_log(Info, "floor_f_u32=%s", f_u32.str());
        jitc_log(Info, "floor_f_u64=%s", f_u64.str());
        jitc_log(Info, "floor_d_u32=%s", d_u32.str());
        jitc_log(Info, "floor_d_u64=%s", d_u64.str());
    }

    {
        auto f_i32 = trunc2int<Int32> (f);
        auto f_u32 = trunc2int<UInt32>(f);
        auto f_i64 = trunc2int<Int64> (f);
        auto f_u64 = trunc2int<UInt64>(f);
        auto d_i32 = trunc2int<Int32> (d);
        auto d_u32 = trunc2int<UInt32>(d);
        auto d_i64 = trunc2int<Int64> (d);
        auto d_u64 = trunc2int<UInt64>(d);
        jitc_schedule(f_i32, f_u32, f_i64, f_u64, d_i32, d_u32, d_i64, d_u64);
        jitc_log(Info, "input=%s", f.str());
        jitc_log(Info, "trunc_f_i32=%s", f_i32.str());
        jitc_log(Info, "trunc_f_i64=%s", f_i64.str());
        jitc_log(Info, "trunc_d_i32=%s", d_i32.str());
        jitc_log(Info, "trunc_d_i64=%s", d_i64.str());
        jitc_log(Info, "trunc_f_u32=%s", f_u32.str());
        jitc_log(Info, "trunc_f_u64=%s", f_u64.str());
        jitc_log(Info, "trunc_d_u32=%s", d_u32.str());
        jitc_log(Info, "trunc_d_u64=%s", d_u64.str());
    }

    {
        auto f_i32 = round2int<Int32> (f);
        auto f_u32 = round2int<UInt32>(f);
        auto f_i64 = round2int<Int64> (f);
        auto f_u64 = round2int<UInt64>(f);
        auto d_i32 = round2int<Int32> (d);
        auto d_u32 = round2int<UInt32>(d);
        auto d_i64 = round2int<Int64> (d);
        auto d_u64 = round2int<UInt64>(d);
        jitc_schedule(f_i32, f_u32, f_i64, f_u64, d_i32, d_u32, d_i64, d_u64);
        jitc_log(Info, "input=%s", f.str());
        jitc_log(Info, "round_f_i32=%s", f_i32.str());
        jitc_log(Info, "round_f_i64=%s", f_i64.str());
        jitc_log(Info, "round_d_i32=%s", d_i32.str());
        jitc_log(Info, "round_d_i64=%s", d_i64.str());
        jitc_log(Info, "round_f_u32=%s", f_u32.str());
        jitc_log(Info, "round_f_u64=%s", f_u64.str());
        jitc_log(Info, "round_d_u32=%s", d_u32.str());
        jitc_log(Info, "round_d_u64=%s", d_u64.str());
    }

    jitc_llvm_set_target("skylake", "", 8);
}

TEST_BOTH(28_scatter_add) {
    jitc_llvm_set_target("skylake-avx512", "+avx512f,+avx512dq,+avx512vl,+avx512cd", 16);
    using Double = Array<double>;
    {
        Float target = zero<Float>(16);
        UInt32 index(0, 1, 2, 0, 4, 5, 6, 7, 8, 9, 10, 2, 3, 0, 0);

        scatter_add(target, Float(1), index);
        jitc_log(Info, "target=%s", target.str());
    }

    {
        Float target = zero<Float>(16);
        UInt32 index(0, 1, 2, 0, 4, 5, 6, 7, 8, 9, 10, 10, 2, 3, 0, 0);

        scatter_add(target, Float(1), index);
        jitc_log(Info, "target=%s", target.str());
    }

    {
        Double target = zero<Double>(16);
        UInt32 index(0, 1, 2, 0, 4, 5, 6, 7, 8, 9, 10, 2, 3, 0, 0);

        scatter_add(target, Double(1), index);
        jitc_log(Info, "target=%s", target.str());
    }

    {
        Double target = zero<Double>(16);
        UInt32 index(0, 1, 2, 0, 4, 5, 6, 7, 8, 9, 10, 10, 2, 3, 0, 0);

        scatter_add(target, Double(1), index);
        jitc_log(Info, "target=%s", target.str());
    }

    {
        Float target = zero<Float>(16);
        auto mask = arange<Float>(15) < 8.f;
        UInt32 index(0, 1, 2, 0, 4, 5, 6, 7, 8, 9, 10, 2, 3, 0, 0);

        scatter_add(target, Float(1), index, mask);
        jitc_log(Info, "target=%s", target.str());
    }

    {
        Float target = zero<Float>(16);
        auto mask = arange<Float>(16) < 8.f;
        UInt32 index(0, 1, 2, 0, 4, 5, 6, 7, 8, 9, 10, 10, 2, 3, 0, 0);

        scatter_add(target, Float(1), index, mask);
        jitc_log(Info, "target=%s", target.str());
    }

    {
        Double target = zero<Double>(16);
        auto mask = arange<Double>(15) < 8.f;
        UInt32 index(0, 1, 2, 0, 4, 5, 6, 7, 8, 9, 10, 2, 3, 0, 0);

        scatter_add(target, Double(1), index, mask);
        jitc_log(Info, "target=%s", target.str());
    }

    {
        Double target = zero<Double>(16);
        UInt32 index(0, 1, 2, 0, 4, 5, 6, 7, 8, 9, 10, 10, 2, 3, 0, 0);
        auto mask = arange<Double>(16) < 8.f;

        scatter_add(target, Double(1), index, mask);
        jitc_log(Info, "target=%s", target.str());
    }
    jitc_llvm_set_target("skylake", "", 8);
}

TEST_BOTH(29_arithmetic_propagation) {
    {
        UInt32 z(0), o(1);

        jitc_assert( z.is_literal_zero());
        jitc_assert(!z.is_literal_one());
        jitc_assert(!o.is_literal_zero());
        jitc_assert( o.is_literal_one());

        UInt32 a = o + z;
        UInt32 b = o * z;
        jitc_assert(a.index() == o.index());
        jitc_assert(b.index() == z.index());
    }
    {
        using Int64 = Array<int64_t>;
        Int64 z(0), o(1);

        jitc_assert( z.is_literal_zero());
        jitc_assert(!z.is_literal_one());
        jitc_assert(!o.is_literal_zero());
        jitc_assert( o.is_literal_one());

        Int64 a = o + z;
        Int64 b = o * z;
        jitc_assert(a.index() == o.index());
        jitc_assert(b.index() == z.index());
    }
    {
        Float z(0), o(1);

        jitc_assert( z.is_literal_zero());
        jitc_assert(!z.is_literal_one());
        jitc_assert(!o.is_literal_zero());
        jitc_assert( o.is_literal_one());

        Float a = o + z;
        Float b = o * z;
        jitc_assert(a.index() == o.index());
        jitc_assert(b.index() == z.index());
    }
    {
        using Double = Array<double>;
        Double z(0), o(1);

        jitc_assert( z.is_literal_zero());
        jitc_assert(!z.is_literal_one());
        jitc_assert(!o.is_literal_zero());
        jitc_assert( o.is_literal_one());

        Double a = o + z;
        Double b = o * z;
        jitc_assert(a.index() == o.index());
        jitc_assert(b.index() == z.index());
    }
}
#endif

TEST_BOTH(30_scatter_ordering) {
    UInt32 x = zero<UInt32>(16),
           y = x + 1;
    UInt32 indices = UInt32(2, 4, 6);

    scatter(x, UInt32(1), indices);

    UInt32 z = x + 1;
    jitc_log(Info, "x:%s", x.str());
    jitc_log(Info, "x:%s", y.str());
    jitc_log(Info, "x:%s", z.str());
}
