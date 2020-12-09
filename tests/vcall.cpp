#include "test.h"
#include <iostream>
#include <memory>

using Float = CUDAArray<float>;
using UInt32 = CUDAArray<uint32_t>;

Float global;

struct Base {
    virtual std::pair<Float, Float> func(Float x, Float y) = 0;
};

struct Class1 : Base {
    std::pair<Float, Float> func(Float x, Float y) override {
        scatter(global, y, UInt32(5));
        return { x + y, fmadd(x, y, y) };
    }
};


struct Class2 : Base {
    std::pair<Float, Float> func(Float x, Float y) override {
        return { x*y, x-y };
    }
};


void read_indices(uint32_t *out, uint32_t &index, const Float &value) {
    if (out)
        out[index] = value.index();
    index += 1;
}

void write_indices(const uint32_t *out, uint32_t &index, Float &value) {
    value = Float::from_index(out[index++]);
}

void read_indices(uint32_t *out, uint32_t &index, const std::pair<Float, Float> &value) {
    read_indices(out, index, value.first);
    read_indices(out, index, value.second);
}

void write_indices(const uint32_t *out, uint32_t &index, std::pair<Float, Float> &value) {
    write_indices(out, index, value.first);
    write_indices(out, index, value.second);
}


template <typename Func, typename... Args>
std::pair<uint32_t, uint64_t> record(int cuda, Func func, const Args&... args) {
    uint32_t se_before = jitc_side_effect_counter(cuda);
    auto result = func(args...);
    uint32_t se_total = jitc_side_effect_counter(cuda) - se_before;

    uint32_t in_count = 0, out_count = 0;
    (read_indices(nullptr, in_count, args), ...);
    read_indices(nullptr, out_count, result);

    std::unique_ptr<uint32_t[]> in(new uint32_t[in_count]),
                                out(new uint32_t[out_count]);

    in_count = 0, out_count = 0;
    (read_indices(in.get(), in_count, args), ...);
    read_indices(out.get(), out_count, result);

    uint64_t func_hash = 0;
    uint32_t id = jitc_eval_ir_var(cuda, in.get(), in_count, out.get(),
                                   out_count, se_total, &func_hash);
    return { id, func_hash };
}

template <typename... Args> auto vcall(const char *domain, UInt32 inst, const Args &... args) {
    using Result = std::pair<Float, Float>;
    int cuda = 1;

    uint32_t n_inst   = jitc_registry_get_max(domain) + 1,
             buf_size = 22 + n_inst * 23 + 4;
    if (buf_size < 128)
        buf_size = 128;

    std::unique_ptr<char[]> buf(new char[buf_size]);
    char *buf_ptr = buf.get();

    memcpy(buf_ptr, ".const .u64 $r0[] = { ", 22);
    buf_ptr += 22;

    uint32_t index = jitc_var_new_0(1, VarType::Global, "", 1, 1);
    for (uint32_t i = 0; i < n_inst; ++i) {
        Base *base = (Base *) jitc_registry_get_ptr(domain, i);
        uint32_t id, prev = index;
        uint64_t hash;

        if (base)
            std::tie(id, hash) = record(cuda,
                [&](const Args &...args) { return base->func(args...); }, empty<Float>(1), empty<Float>(1));
        else
            std::tie(id, hash) = record(cuda,
                [&](const Args &...) { return std::pair<Float, Float>(0, 0); }, empty<Float>(1), empty<Float>(1));

        index = jitc_var_new_2(cuda, VarType::Global, "", 1, index, id);
        jitc_var_dec_ref_ext(id);
        jitc_var_dec_ref_ext(prev);

        buf_ptr +=
            snprintf(buf_ptr, 23 + 1, "func_%016llx%s",
                     (unsigned long long) hash, i + 1 < n_inst ? ", " : " ");
    }

    memcpy(buf_ptr, "};\n", 4);
    uint32_t call_table = jitc_var_new_1(cuda, VarType::Global, buf.get(), 0, index);
    jitc_var_dec_ref_ext(index);

    uint32_t offset = jitc_var_new_2(cuda, VarType::UInt64,
            "mov.$t0 $r0, $r2$n"
            "mad.wide.u32 $r0, $r1, 8, $r0$n"
            "ld.const.$t0 $r0, [$r0]", 1, inst.index(), call_table);

    const uint32_t var_type_size[(int) VarType::Count] {
        0, 0, 1, 1, 1, 2, 2, 4, 4, 8, 8, 2, 4, 8, 8
    };

    // Collect input arguments
    uint32_t in_count = 0, out_count = 0;
    (read_indices(nullptr, in_count, args), ...);
    Result result(0, 0);
    read_indices(nullptr, out_count, result);
    std::unique_ptr<uint32_t[]> in(new uint32_t[in_count]),
                                out(new uint32_t[out_count]);
    in_count = 0; out_count = 0;
    (read_indices(in.get(), in_count, args), ...);
    read_indices(out.get(), out_count, Result(0, 0));

    index = jitc_var_new_0(cuda, VarType::Invalid, "", 1, 1);
    uint32_t offset_in = 0, align_in = 1;
    for (uint32_t i = 0; i < in_count; ++i) {
        uint32_t prev = index;
        index = jitc_var_new_2(cuda, VarType::Invalid, "", 1, in[i], index);
        jitc_var_dec_ref_ext(prev);
        uint32_t size = var_type_size[(uint32_t) jitc_var_type(in[i])];
        offset_in = (offset_in + size - 1) / size * size;
        offset_in += size;
        align_in = std::max(align_in, size);
    }

    uint32_t offset_out = 0, align_out = 1;
    for (uint32_t i = 0; i < out_count; ++i) {
        uint32_t size = var_type_size[(uint32_t) jitc_var_type(out[i])];
        offset_out = (offset_out + size - 1) / size * size;
        offset_out += size;
        align_out = std::max(align_out, size);
    }

    int printed = snprintf(buf.get(), buf_size,
            "\n    {\n"
	        "        .param .align %u .b8 param_in[%u];\n"
	        "        .param .align %u .b8 param_out[%u]",
	        align_in, offset_in, align_out, offset_out);

    uint32_t prev = index;
    index = jitc_var_new_1(cuda, VarType::Invalid, buf.get(), 0, index);
    jitc_var_dec_ref_ext(prev);

    offset_in = 0;
    for (uint32_t i = 0; i < in_count; ++i) {
        uint32_t size = var_type_size[(uint32_t) jitc_var_type(in[i])];
        offset_in = (offset_in + size - 1) / size * size;
        snprintf(buf.get(), buf_size, "    st.param.$t1 [param_in+%u], $r1", offset_in);
        uint32_t prev = index;
        index = jitc_var_new_2(cuda, VarType::Invalid, buf.get(), 0, in[i], index);
        jitc_var_dec_ref_ext(prev);
        offset_in += size;
    }

    prev = index;
    index = jitc_var_new_3(cuda, VarType::Invalid,
            "    call (param_out), $r1, (param_in), $r2",
            1, offset, call_table, index);
    jitc_var_dec_ref_ext(prev);
    jitc_var_dec_ref_ext(offset);
    jitc_var_dec_ref_ext(call_table);

    offset_out = 0;
    for (uint32_t i = 0; i < out_count; ++i) {
        VarType type = jitc_var_type(out[i]);
        uint32_t size = var_type_size[(uint32_t) type];
        offset_out = (offset_out + size - 1) / size * size;
        uint32_t prev = index;
        snprintf(buf.get(), buf_size, "    ld.param.$t0 $r0, [param_out+%u]", offset_out);
        index = jitc_var_new_1(cuda, type, buf.get(), 0, index);
        out[i] = index;
        jitc_var_dec_ref_ext(prev);
        offset_out += size;
    }

    prev = index;
    index = jitc_var_new_1(cuda, VarType::Invalid, "}\n",
            1, index);
    jitc_var_dec_ref_ext(prev);

    for (uint32_t i = 0; i < out_count; ++i) {
        out[i] = jitc_var_new_2(cuda, jitc_var_type(out[i]),
                                "mov.$t0 $r0, $r1", 0,
                                out[i], index);
    }

    jitc_var_dec_ref_ext(index);
    out_count = 0;
    write_indices(out.get(), out_count, result);

    return result;
}

TEST_CUDA(01_symbolic_vcall) {
    global = arange<Float>(10);
    jitc_eval(global);
    jitc_set_eval_enabled(1, 0);

    Class1 c1;
    Class2 c2;

    uint32_t i1 = jitc_registry_put("Base", &c1);
    uint32_t i2 = jitc_registry_put("Base", &c2);

    Float x = 1, y = 2;
    UInt32 inst = arange<UInt32>(3);
    std::pair<Float, Float> result = vcall("Base", inst, x, y);
    jitc_set_eval_enabled(1, 1);
    jitc_eval(result.first, result.second);
    jitc_assert(result.first == Float(0, 3, 2));
    jitc_assert(result.second == Float(0, 4, -1));
    jitc_assert(global == Float(0, 1, 2, 3, 4, 2, 6, 7, 8, 9));

    jitc_registry_remove(&c1);
    jitc_registry_remove(&c2);
    global=Float();
}
