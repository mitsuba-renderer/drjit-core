#include "test.h"
#include <memory>
#include <vector>

#if defined(ENOKI_JIT_ENABLE_OPTIX)
#  include "optix_stubs.h"
#endif

template <JitBackend Backend, typename T>
void read_indices(uint32_t *out, uint32_t &index, const JitArray<Backend, T> &value) {
    if (out)
        out[index] = value.index();
    index += 1;
}

template <JitBackend Backend, typename T>
void write_indices(const uint32_t *out, uint32_t &index, JitArray<Backend, T> &value) {
    value = JitArray<Backend, T>::steal(out[index++]);
}

template <typename T1, typename T2>
void read_indices(uint32_t *out, uint32_t &index, const std::pair<T1, T2> &value) {
    read_indices(out, index, value.first);
    read_indices(out, index, value.second);
}

template <typename T1, typename T2>
void write_indices(const uint32_t *out, uint32_t &index, std::pair<T1, T2> &value) {
    write_indices(out, index, value.first);
    write_indices(out, index, value.second);
}

template <typename Func, typename... Args>
bool record(int cuda, uint32_t &id, uint64_t &hash, std::vector<uint32_t> &extra,
            Func func, const Args &... args) {

    uint32_t se_before = jit_side_effect_counter(cuda);
    auto result        = func(args...);
    uint32_t se_total  = jit_side_effect_counter(cuda) - se_before;

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
    id = jit_capture_var(cuda, "Base", "func", in.get(), in_count, out.get(),
                          out_count, nullptr, nullptr, se_total, &hash,
                          &extra_p, &extra_count_p);

    for (int i = 0; i < extra_count_p; ++i)
        extra.push_back(extra_p[i]);

    return se_total != 0;
}

template <typename Base, typename Float, typename UInt32, typename... Args>
auto vcall(const char *domain, UInt32 self, const Args &... args) {
    using Result = std::pair<Float, Float>;
    int cuda     = Float::IsCUDA;

    uint32_t n_inst = jit_registry_get_max(domain) + 1;

    std::unique_ptr<uint32_t[]> call_id(new uint32_t[n_inst]);
    std::unique_ptr<uint64_t[]> call_hash(new uint64_t[n_inst]);
    std::unique_ptr<uint32_t[]> extra_offset(new uint32_t[n_inst]);
    std::vector<uint32_t> extra;
    bool side_effects = false;
    Result result(0, 0);

    for (uint32_t i = 0; i < n_inst; ++i) {
        Base *base = (Base *) jit_registry_get_ptr(domain, i);

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

    jit_var_vcall(cuda, "Base", "func", self.index(), n_inst, call_id.get(),
                   call_hash.get(), in_count, in.get(), out_count, out.get(),
                   nullptr, nullptr, extra.size(), extra.data(),
                   extra_offset.get(), side_effects);

    out_count = 0;
    write_indices(out.get(), out_count, result);

    return result;
}

TEST_CUDA(01_symbolic_vcall) {
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
    jit_eval(global);
    jit_enable_flag(JitFlag::RecordVCalls);

    Class1 c1(global);
    Class2 c2;

    uint32_t i1 = jit_registry_put("Base", &c1);
    uint32_t i2 = jit_registry_put("Base", &c2);

    Float x = 1, y = 2;
    UInt32 inst = arange<UInt32>(3);
    std::pair<Float, Float> result = vcall<Base, Float, UInt32>("Base", inst, x, y);
    jit_eval(result.first, result.second);
    jit_assert(result.first == Float(0, 3, 2));
    jit_assert(result.second == Float(0, 4, -1));
    jit_assert(global == Float(0, 1, 2, 3, 4, 2, 6, 7, 8, 9));

    jit_registry_remove(&c1);
    jit_registry_remove(&c2);
    jit_registry_trim();
    global = Float();
}

#if defined(ENOKI_JIT_ENABLE_OPTIX)
TEST_CUDA(01_symbolic_vcall_optix) {
    init_optix_api();
    OptixDeviceContext context = jit_optix_context();
    OptixModuleCompileOptions module_compile_options { };
    OptixPipelineCompileOptions pipeline_compile_options { };
    pipeline_compile_options.exceptionFlags =
        OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
        OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    pipeline_compile_options.numPayloadValues = 1;

    /// A simple hit/miss shader combo
    const char *miss_and_closesthit_ptx = R"(
    .version 6.0
    .target sm_50
    .address_size 64

    .entry __miss__ms() {
        .reg .b32 %r<1>;
        mov.b32 %r0, 0;
        call _optix_set_payload_0, (%r0);
        ret;
    }

    .entry __closesthit__ch() {
        .reg .b32 %r<1>;
        mov.b32 %r0, 1;
        call _optix_set_payload_0, (%r0);
        ret;
    })";

    OptixModule mod;
    char log[1024]; size_t log_size = sizeof(log);
    jit_optix_check(optixModuleCreateFromPTX(
        context, &module_compile_options, &pipeline_compile_options,
        miss_and_closesthit_ptx, strlen(miss_and_closesthit_ptx), log,
        &log_size, &mod));

    // =====================================================
    // Create program groups (raygen provided by Enoki..)
    // =====================================================

    OptixProgramGroupOptions pgo {};
    OptixProgramGroupDesc pgd[2] { };
    OptixProgramGroup pg[2];

    pgd[0].kind                         = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgd[0].miss.module                  = mod;
    pgd[0].miss.entryFunctionName       = "__miss__ms";
    pgd[1].kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgd[1].hitgroup.moduleCH            = mod;
    pgd[1].hitgroup.entryFunctionNameCH = "__closesthit__ch";

    jit_optix_check(optixProgramGroupCreate(context, pgd, 2 /* two at once */,
                                             &pgo, nullptr, nullptr, pg));

    // =====================================================
    // Shader binding table setup
    // =====================================================
    OptixShaderBindingTable sbt {};

    sbt.missRecordBase =
        jit_malloc(AllocType::HostPinned, OPTIX_SBT_RECORD_HEADER_SIZE);
    sbt.missRecordStrideInBytes     = OPTIX_SBT_RECORD_HEADER_SIZE;
    sbt.missRecordCount             = 1;

    sbt.hitgroupRecordBase =
        jit_malloc(AllocType::HostPinned, OPTIX_SBT_RECORD_HEADER_SIZE);
    sbt.hitgroupRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
    sbt.hitgroupRecordCount         = 1;

    jit_optix_check(optixSbtRecordPackHeader(pg[0], sbt.missRecordBase));
    jit_optix_check(optixSbtRecordPackHeader(pg[1], sbt.hitgroupRecordBase));

    sbt.missRecordBase =
        jit_malloc_migrate(sbt.missRecordBase, AllocType::Device, 1);
    sbt.hitgroupRecordBase =
        jit_malloc_migrate(sbt.hitgroupRecordBase, AllocType::Device, 1);

    jit_optix_configure(&pipeline_compile_options, &sbt, pg, 2);

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
    jit_eval(global);
    jit_enable_flag(JitFlag::RecordVCalls);

    Class1 c1(global);
    Class2 c2;

    uint32_t i1 = jit_registry_put("Base", &c1);
    uint32_t i2 = jit_registry_put("Base", &c2);

    Float x = 1, y = 2;
    UInt32 inst = arange<UInt32>(3);
    std::pair<Float, Float> result = vcall<Base, Float, UInt32>("Base", inst, x, y);
    jit_optix_mark(x.index());
    jit_eval(result.first, result.second);
    jit_assert(result.first == Float(0, 3, 2));
    jit_assert(result.second == Float(0, 4, -1));
    jit_assert(global == Float(0, 1, 2, 3, 4, 2, 6, 7, 8, 9));

    jit_registry_remove(&c1);
    jit_registry_remove(&c2);
    jit_registry_trim();
    global = Float();
}
#endif
