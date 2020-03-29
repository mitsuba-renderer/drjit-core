#include "test.h"

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
    jitc_eval();
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
                out += value[j];
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
                out += value[j];
            }
            jitc_log(Info, "value=%s vs %u", out.str(), i * (i - 1) / 2);
        }
    }
}

TEST_BOTH(07_arange) {
    UInt32 x = arange<UInt32>(1024);
    UInt32 y = arange<UInt32>(3, 512, 7);
    jitc_log(Info, "value=%s", x.str());
    jitc_log(Info, "value=%s", y.str());
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
        Float e = fmsub(d, b, c);
        jitc_log(Info, "value=%s", d.str());
        jitc_log(Info, "value=%s", e.str());
    }

    using Double = Array<double>;
    /* Double case */ {
        Double a(1, 2, 3, 4);
        Double b(3, 8, 1, 5);
        Double c(9, 1, 3, 0);

        Double d = fmadd(a, b, c);
        Double e = fmsub(d, b, c);
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
    jitc_log(Info, "value=%s", x.str());
    jitc_log(Info, "mask=%s", mask.str());
    Float y = select(mask, Float(1.f), Float(2.f));
    jitc_log(Info, "value_2=%s", y.str());

    Array<bool> mask_scalar_t = true;
    Array<bool> mask_scalar_f = false;
    jitc_eval();
    Float a = select(mask_scalar_t, x, Float(0));
    Float b = select(mask_scalar_f, x, Float(0));
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

    jitc_log(Info, "XOR: value_1=%s", a.str());
    jitc_log(Info, "XOR: value_2=%s", b.str());
    jitc_log(Info, "XOR: value_3=%s", c.str());

    a = ~a;
    b = ~b;
    c = ~c;

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

    jitc_log(Info, "OR: value_1=%s", a3.str());
    jitc_log(Info, "OR: value_2=%s", b3.str());
    jitc_log(Info, "OR: value_3=%s", c3.str());

    auto a4 = a & a2;
    auto b4 = b & b2;
    auto c4 = c & c2;

    jitc_log(Info, "AND: value_1=%s", a4.str());
    jitc_log(Info, "AND: value_2=%s", b4.str());
    jitc_log(Info, "AND: value_3=%s", c4.str());

    auto a5 = a ^ a2;
    auto b5 = b ^ b2;
    auto c5 = c ^ c2;

    jitc_log(Info, "XOR: value_1=%s", a5.str());
    jitc_log(Info, "XOR: value_2=%s", b5.str());
    jitc_log(Info, "XOR: value_3=%s", c5.str());
}

TEST_BOTH(14_scatter_gather) {
    Int32 l     = -arange<Int32>(1024);
    Int32 index = Int32(34, 62, 75, 2);
    Int32 value = gather(l, index);
    jitc_log(Info, "%s", value.str());
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
    jitc_registry_remove((void *) 0x1);

    jitc_assert(jitc_registry_get_max(key_1) == 4u);
    jitc_assert(jitc_registry_get_max(key_2) == 1u);

    jitc_registry_trim();

    jitc_assert(jitc_registry_get_max(key_1) == 0u);
    jitc_assert(jitc_registry_get_max(key_2) == 0u);
}

/// Mask gathers/scatters!
