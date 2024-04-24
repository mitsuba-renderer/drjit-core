#include "test.h"

template <typename Value> struct Arr {
    using UInt32 = typename Value::template ReplaceValue<uint32_t>;
    using Bool = typename Value::template ReplaceValue<bool>;
    using Scalar = typename Value::Value;

    Arr() { m_index = 0; }
    Arr(Arr &&a) : m_index(a.m_index) { a.m_index = 0; }
    Arr(const Arr &a) : m_index(a.m_index) { jit_var_inc_ref(m_index); }

    Arr &operator=(const Arr &) = delete;
    Arr &operator=(Arr &&) = delete;

    Arr(size_t length) {
        m_index = jit_array_create(Value::Backend, Value::Type, 1, length);
    }

    Arr(size_t length, Value value) {
        uint32_t tmp = jit_array_create(Value::Backend, Value::Type, 1, length);
        m_index = jit_array_init(tmp, value.index());
        jit_var_dec_ref(tmp);
    }

    ~Arr() {
        jit_var_dec_ref(m_index);
    }

    size_t length() const {
        return jit_array_length(m_index);
    }

    void eval() {
        jit_var_eval(m_index);
    }

    void schedule() {
        jit_var_schedule(m_index);
    }

    void write(const UInt32 &offset, const Value &value, const Bool &mask = true) {
        uint32_t new_index =
            jit_array_write(m_index, offset.index(), value.index(), mask.index());
        jit_var_dec_ref(m_index);
        m_index = new_index;
    }

    Value read(const UInt32 &offset, const Bool &mask = true) {
        return Value::steal(jit_array_read(m_index, offset.index(), mask.index()));
    }

    uint32_t m_index;
};

TEST_BOTH_FLOAT_AGNOSTIC(01_literal_index) {
    Arr<Float> x(10);

    for (int i = 0; i < 10; ++i)
        x.write(i, i);

    Float result = 0;
    for (int i = 0; i < 10; ++i)
        result += x.read(i);

    jit_assert(result.read(0) == 45);
    jit_assert(x.length() == 10);
}

TEST_BOTH_FLOAT_AGNOSTIC(02_opaque_index) {
    Arr<Float> x(10);
    UInt32 opaque_0 = opaque<UInt32>(0);

    for (int i = 0; i < 10; ++i)
        x.write(UInt32(i) + opaque_0, i);

    Float result = 0;
    for (int i = 0; i < 10; ++i) {
        Float a = x.read(UInt32(i) + opaque_0);
        Float b = x.read(UInt32(i) + opaque_0);
        jit_assert(a.index() == b.index());
        result += a;
    }

    jit_assert(result.read(0) == 45);
}


TEST_BOTH_FLOAT_AGNOSTIC(03_literal_index_masked) {
    UInt32 opaque_1 = opaque<UInt32>(1);

    {
        Arr<Float> x(10);
        for (int i = 0; i < 10; ++i)
            x.write(i, i);

        Float result = 0;
        for (int i = 0; i < 10; ++i)
            result += x.read(i, Mask(UInt32(i) & opaque_1));

        jit_assert(result.read(0) == 25);
    }

    {
        Arr<Float> x(10);
        for (int i = 0; i < 10; ++i)
            x.write(i, 0);
        for (int i = 0; i < 10; ++i)
            x.write(i, i, Mask(UInt32(i) & opaque_1));

        Float result = 0;
        for (int i = 0; i < 10; ++i)
            result += x.read(i);

        jit_assert(result.read(0) == 25);
    }
}

TEST_BOTH_FLOAT_AGNOSTIC(04_opaque_index_masked) {
    UInt32 opaque_0 = opaque<UInt32>(0);
    UInt32 opaque_1 = opaque<UInt32>(1);

    {
        Arr<Float> x(10);
        for (int i = 0; i < 10; ++i)
            x.write(i, i);

        Float result = 0;
        for (int i = 0; i < 10; ++i)
            result += x.read(UInt32(i) + opaque_0, Mask(UInt32(i) & opaque_1));

        jit_assert(result.read(0) == 25);
    }

    {
        Arr<Float> x(10);
        for (int i = 0; i < 10; ++i)
            x.write(i, 0);
        for (int i = 0; i < 10; ++i)
            x.write(i, UInt32(i) + opaque_0, Mask(UInt32(i) & opaque_1));

        Float result = 0;
        for (int i = 0; i < 10; ++i)
            result += x.read(UInt32(i) + opaque_0);

        jit_assert(result.read(0) == 25);
    }
}

TEST_BOTH_FLOAT_AGNOSTIC(05_conflict) {
    UInt32 opaque_0 = opaque<UInt32>(0);
    UInt32 opaque_1 = opaque<UInt32>(1);

    {
        Arr<Float> x(10);
        for (int i = 0; i < 10; ++i)
            x.write(i, i);

        Arr<Float> y(x);
        for (int i = 0; i < 10; ++i)
            y.write(i, UInt32(i)+10);

        Float result_1 = 0;
        for (int i = 0; i < 10; ++i)
            result_1 += x.read(UInt32(i));

        Float result_2 = 0;
        for (int i = 0; i < 10; ++i)
            result_2 += y.read(UInt32(i));

        result_1.schedule();
        result_2.schedule();

        jit_assert(result_1.read(0) == 45 && result_2.read(0)==145);
    }
}

TEST_BOTH_FLOAT_AGNOSTIC(06_eval_array) {
    UInt32 opaque_0 = opaque<UInt32>(0);
    UInt32 opaque_1 = opaque<UInt32>(1);

    {
        Arr<Float> x(10);
        Float offset = arange<Float>(3);
        for (int i = 0; i < 10; ++i)
            x.write(i, Float(i) + offset);

        x.eval();

        Float result = 0;
        for (int i = 0; i < 10; ++i)
            result += x.read(UInt32(i));

        jit_assert(result.read(0) == 45);
        jit_assert(result.read(1) == 55);
        jit_assert(result.read(2) == 65);
    }
}

TEST_BOTH_FLOAT_AGNOSTIC(07_literal_init) {
    {
        Arr<Float> x(10, 0);
        Float r = 0;
        for (int i = 0; i < 10; ++i)
            r += x.read(i);
        jit_assert(r.read(0) == 0);
    }

    {
        Arr<Float> x(10, 10);
        Float r = 0;
        for (int i = 0; i < 10; ++i)
            r += x.read(i);
        jit_assert(r.read(0) == 100);
    }
}

TEST_BOTH_FLOAT_AGNOSTIC(08_complex_indexing) {
    {
        Arr<Float> x(10);
        UInt32 index = arange<UInt32>(3);

        for (int i = 0; i < 10; ++i)
            x.write((index + UInt32(i)) % 10, Float(i));

        Float r = 0;
        for (int i = 0; i < 10; ++i)
            r += x.read((index + UInt32(i)) % 10);

        jit_assert(
            r.read(0) == 45 &&
            r.read(1) == 45 &&
            r.read(2) == 45);
    }

    {
        Arr<Float> x(10, 0);
        UInt32 index = arange<UInt32>(3);

        for (int i = 0; i < 10; ++i)
            x.write((index + UInt32(i)) % 10, Float(i), Mask(UInt32(i) & 1));

        Float r = 0;
        for (int i = 0; i < 10; ++i)
            r += x.read((index + UInt32(i)) % 10);

        jit_assert(
            r.read(0) == 25 &&
            r.read(1) == 25 &&
            r.read(2) == 25);
    }

    {
        Arr<Float> x(10, 10);
        UInt32 index = arange<UInt32>(3);

        for (int i = 0; i < 10; ++i)
            x.write((index + UInt32(i)) % 10, Float(i));

        Float r = 0;
        for (int i = 0; i < 10; ++i)
            r += x.read((index + UInt32(i)) % 10, Mask(UInt32(i) & 1));

        jit_assert(
            r.read(0) == 25 &&
            r.read(1) == 25 &&
            r.read(2) == 25);
    }
}

TEST_BOTH_FLOAT_AGNOSTIC(09_write_eval_conflict) {
    Arr<Float> x(2, 1);
    Arr<Float> y(x);
    x.write(0, 2);
    x.write(1, 3);
    x.schedule();
    y.schedule();
    jit_eval();

    Float xx = x.read(0) + x.read(1);
    Float yy = y.read(0) + y.read(1);
    xx.schedule();
    yy.schedule();

    jit_assert(
        xx.read(0) == 5 &&
        yy.read(0) == 2
    );
}

TEST_BOTH_FLOAT_AGNOSTIC(10_mask_simple) {
    Arr<Mask> x(10, false);
    Arr<Mask> y(10, true);

    UInt32 ctr_1 = 0, ctr_2 = 0;
    for (int i = 0; i < 10; ++i)
        x.write(i, Mask(i & 1));

    for (int i = 0; i < 10; ++i) {
        ctr_1 += UInt32(x.read(i));
        ctr_2 += UInt32(y.read(i));
    }

    ctr_1.schedule();
    ctr_2.schedule();

    jit_assert(ctr_1.read(0) == 5 &&
               ctr_2.read(0) == 10);
}

TEST_BOTH_FLOAT_AGNOSTIC(11_mask_eval) {
    Arr<Mask> x(10, false);
    Arr<Mask> y(10, true);

    UInt32 ctr_1 = 0, ctr_2 = 0;
    for (int i = 0; i < 10; ++i)
        x.write(i, Mask(i & 1));

    ctr_1.schedule();
    ctr_2.schedule();
    jit_eval();

    for (int i = 0; i < 10; ++i) {
        ctr_1 += UInt32(x.read(i));
        ctr_2 += UInt32(y.read(i));
    }

    jit_assert(ctr_1.read(0) == 5 &&
               ctr_2.read(0) == 10);
}

TEST_BOTH_FLOAT_AGNOSTIC(12_mask_complex_indexing) {
    {
        Arr<Mask> x(10, 0);
        UInt32 index = arange<UInt32>(3);

        for (int i = 0; i < 10; ++i)
            x.write((index + UInt32(i)) % 10, Mask(i &1));

        UInt32 r = 0;
        for (int i = 0; i < 10; ++i)
            r += x.read((index + UInt32(i)) % 10);

        jit_assert(
            r.read(0) == 5 &&
            r.read(1) == 5 &&
            r.read(2) == 5);
    }

    {
        Arr<Mask> x(10, false);
        UInt32 index = arange<UInt32>(3);

        for (int i = 0; i < 10; ++i)
            x.write((index + UInt32(i)) % 10, Float(i), Mask(UInt32(i) & 1));

        UInt32 r = 0;
        for (int i = 0; i < 10; ++i)
            r += x.read((index + UInt32(i)) % 10);

        jit_assert(
            r.read(0) == 5 &&
            r.read(1) == 5 &&
            r.read(2) == 5);
    }

    {
        Arr<Mask> x(10, true);
        UInt32 index = arange<UInt32>(3);

        UInt32 r = 0;
        for (int i = 0; i < 10; ++i)
            r += x.read((index + UInt32(i)) % 10, Mask(UInt32(i) & 1));

        jit_assert(
            r.read(0) == 5 &&
            r.read(1) == 5 &&
            r.read(2) == 5);
    }
}

TEST_BOTH_FLOAT_AGNOSTIC(13_mask_conflict) {
    Arr<Mask> x(2, 1);
    Arr<Mask> y(x);
    x.write(0, false);

    UInt32 ctr_1 = UInt32(x.read(0)) + UInt32(x.read(1));
    UInt32 ctr_2 = UInt32(y.read(0)) + UInt32(y.read(1));

    ctr_1.schedule();
    ctr_2.schedule();

    jit_assert(ctr_1.read(0) == 1 && ctr_2.read(0) == 2);
}
