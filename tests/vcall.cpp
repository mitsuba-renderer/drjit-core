#include "test.h"
#include <iostream>
#include <memory>
#include <vector>

template <typename Float>
void read_indices(uint32_t *out, uint32_t &index, const Float &value) {
    if (out)
        out[index] = value.index();
    index += 1;
}

template <typename Float>
void write_indices(const uint32_t *out, uint32_t &index, Float &value) {
    value = Float::steal(out[index++]);
}

template <typename Float>
void read_indices(uint32_t *out, uint32_t &index,
                  const std::pair<Float, Float> &value) {
    read_indices(out, index, value.first);
    read_indices(out, index, value.second);
}

template <typename Float>
void write_indices(const uint32_t *out, uint32_t &index,
                   std::pair<Float, Float> &value) {
    write_indices(out, index, value.first);
    write_indices(out, index, value.second);
}

template <typename Func, typename... Args>
bool record(int cuda, uint32_t &id, uint64_t &hash, std::vector<uint32_t> &extra,
            Func func, const Args &... args) {

    uint32_t se_before = jitc_side_effect_counter(cuda);
    auto result        = func(args...);
    uint32_t se_total  = jitc_side_effect_counter(cuda) - se_before;

    uint32_t in_count = 0, out_count = 0;
    (read_indices(nullptr, in_count, args), ...);
    read_indices(nullptr, out_count, result);

    std::unique_ptr<uint32_t[]> in(new uint32_t[in_count]),
                               out(new uint32_t[out_count]);

    in_count = 0, out_count = 0;
    (read_indices(in.get(), in_count, args), ...);
    read_indices(out.get(), out_count, result);

    uint32_t *extra_p = nullptr;
    uint32_t extra_count_p = 0;
    id = jitc_capture_var(cuda, in.get(), in_count, out.get(), out_count,
                          se_total, &hash, &extra_p, &extra_count_p);

    for (int i = 0; i < extra_count_p; ++i)
        extra.push_back(extra_p[i]);

    return se_total != 0;
}

template <typename Base, typename Float, typename UInt32, typename... Args>
auto vcall(const char *domain, UInt32 self, const Args &... args) {
    using Result = std::pair<Float, Float>;
    int cuda     = Float::IsCUDA;

    uint32_t n_inst = jitc_registry_get_max(domain) + 1;

    std::unique_ptr<uint32_t[]> call_id(new uint32_t[n_inst]);
    std::unique_ptr<uint64_t[]> call_hash(new uint64_t[n_inst]);
    std::unique_ptr<uint32_t[]> extra_offset(new uint32_t[n_inst]);
    std::vector<uint32_t> extra;
    bool side_effects = false;
    Result result(0, 0);

    for (uint32_t i = 0; i < n_inst; ++i) {
        Base *base = (Base *) jitc_registry_get_ptr(domain, i);

        extra_offset[i] = (uint32_t) (extra.size() * sizeof(void *));

        if (base)
            side_effects |= record(
                cuda, call_id[i], call_hash[i], extra,
                [&](const Args &... args) { return base->func(args...); },
                placeholder<Float>(), placeholder<Float>());
        else
            side_effects |= record(
                cuda, call_id[i], call_hash[i], extra,
                [&](const Args &...) { return result; },
                placeholder<Float>(), placeholder<Float>());
    }

    // Collect input arguments
    uint32_t in_count = 0;
    (read_indices(nullptr, in_count, args), ...);
    std::unique_ptr<uint32_t[]> in(new uint32_t[in_count]);
    in_count = 0;
    (read_indices(in.get(), in_count, args), ...);

    // Collect output arguments
    uint32_t out_count = 0;
    read_indices(nullptr, out_count, result);
    std::unique_ptr<uint32_t[]> out(new uint32_t[out_count]);
    out_count = 0;
    read_indices(out.get(), out_count, result);

    jitc_var_vcall(cuda, self.index(), n_inst, call_id.get(), call_hash.get(),
                   in_count, in.get(), out_count, out.get(), extra.size(),
                   extra.data(), extra_offset.get(), side_effects);

    out_count = 0;
    write_indices(out.get(), out_count, result);

    return result;
}

TEST_BOTH(01_symbolic_vcall) {
    struct Base {
        virtual std::pair<Float, Float> func(Float x, Float y) = 0;
    };

    struct Class1 : Base {
        Class1(Float &global) : global(global) { }
        std::pair<Float, Float> func(Float x, Float y) override {
            scatter(global, y, UInt32(5));
            return { x + y, fmadd(x, y, y) };
        }
        Float global;
    };

    struct Class2 : Base {
        std::pair<Float, Float> func(Float x, Float y) override {
            return { x * y, x - y };
        }
    };

    Float global = arange<Float>(10);
    jitc_eval(global);
    jitc_enable_flag(JitFlag::RecordVCalls);

    Class1 c1(global);
    Class2 c2;

    uint32_t i1 = jitc_registry_put("Base", &c1);
    uint32_t i2 = jitc_registry_put("Base", &c2);

    Float x = 1, y = 2;
    UInt32 inst = arange<UInt32>(3);
    std::pair<Float, Float> result = vcall<Base, Float, UInt32>("Base", inst, x, y);
    jitc_disable_flag(JitFlag::RecordVCalls);
    jitc_eval(result.first, result.second);
    jitc_assert(result.first == Float(0, 3, 2));
    jitc_assert(result.second == Float(0, 4, -1));
    jitc_assert(global == Float(0, 1, 2, 3, 4, 2, 6, 7, 8, 9));

    jitc_registry_remove(&c1);
    jitc_registry_remove(&c2);
    global = Float();
}
