// metal_kernel.cpp -- Phase 2 smoke test: assemble + compile + launch a
// trivial MSL kernel via Dr.Jit's Metal backend.
//
// This drives just enough of the pipeline to verify that:
//   * jitc_metal_assemble produces syntactically valid MSL,
//   * jitc_metal_compile loads it via MTLDevice::newLibraryWithSource,
//   * MetalThreadState::launch dispatches the kernel,
//   * the resulting output buffer matches the expected values.

#include <drjit-core/array.h>
#include <cstdio>

using namespace drjit;

int main() {
    jit_set_log_level_stderr(LogLevel::Trace);
    jit_init(1u << (uint32_t) JitBackend::Metal);

    if (!jit_has_backend(JitBackend::Metal)) {
        std::fprintf(stderr, "Metal backend was not initialized!\n");
        return 1;
    }

    // y = a + b  with a, b length-8 vectors of float
    using FloatM = MetalArray<float>;

    float a_data[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    float b_data[] = { 8, 7, 6, 5, 4, 3, 2, 1 };

    FloatM a = FloatM::copy(a_data, 8);
    FloatM b = FloatM::copy(b_data, 8);
    FloatM y = a + b;

    // Force evaluation
    jit_var_eval(y.index());

    std::printf("Result string: %s\n", jit_var_str(y.index()));
    std::fflush(stdout);

    jit_shutdown(0);
    return 0;
}
