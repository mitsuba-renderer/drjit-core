#include "test.h"

TEST_BOTH(01_creation_destruction) {
    // Checks simple reference counting of a variable
    Float value(1234);
    (void) value;
}

TEST_BOTH(02_fill_and_print) {
    /// Checks array initialization from a given pointer, jitc_fill(), and stringification
    jitc_log(Info, "  int8_t: %s", Array<  int8_t>::full(-111, 5).str());
    jitc_log(Info, " uint8_t: %s", Array< uint8_t>::full( 222, 5).str());
    jitc_log(Info, " int16_t: %s", Array< int16_t>::full(-1111, 5).str());
    jitc_log(Info, "uint16_t: %s", Array<uint16_t>::full( 2222, 5).str());
    jitc_log(Info, " int32_t: %s", Array< int32_t>::full(-1111111111, 5).str());
    jitc_log(Info, "uint32_t: %s", Array<uint32_t>::full( 2222222222, 5).str());
    jitc_log(Info, " int64_t: %s", Array< int64_t>::full(-1111111111111111111, 5).str());
    jitc_log(Info, "uint64_t: %s", Array<uint64_t>::full( 2222222222222222222, 5).str());
    jitc_log(Info, "   float: %s", Array<   float>::full(1.f / 3.f, 5).str());
    jitc_log(Info, "  double: %s", Array<  double>::full(1.0 / 3.0, 5).str());
}

TEST_BOTH(03_eval_scalar) {
    /// Checks that we can evaluate a simple kernel
    Float value(1234);
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
    UInt32 x = UInt32::arange(1024);
    UInt32 y = UInt32::arange(3, 512, 7);
    jitc_log(Info, "value=%s", x.str());
    jitc_log(Info, "value=%s", y.str());
}

TEST_BOTH(08_conv) {
    /* UInt32 */ {
        auto src = Array<uint32_t>::arange(1024);
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
        auto src = Array<int32_t>::arange(1024) - 512;
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
        auto src = Array<uint64_t>::arange(1024);
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
        auto src = Array<int64_t>::arange(1024) - 512;
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
        auto src = Array<float>::arange(1024) - 512;
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
        auto src = Array<double>::arange(1024) - 512;
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
    Float x = sqrt(Float::arange(10));
    jitc_log(Info, "value=%s", x.str());
}

TEST_BOTH(11_mask) {
    Float x = Float::arange(10);
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

    int32_t x = -1;
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
