// Simple smoke test for the Metal backend (Phase 1).
//
// Builds with:
//   c++ -std=c++17 -I include -L build -ldrjit-core -framework Foundation \
//       -framework Metal tests/metal_smoke.cpp -o metal_smoke

#include <drjit-core/jit.h>
#include <cstdio>

int main() {
    jit_set_log_level_stderr(LogLevel::Info);

    // Initialize only the Metal backend.
    jit_init(1u << (uint32_t) JitBackend::Metal);

    if (!jit_has_backend(JitBackend::Metal)) {
        std::fprintf(stderr, "Metal backend was not initialized!\n");
        return 1;
    }

    std::printf("Metal device count: %d\n", jit_metal_device_count());
    std::fflush(stdout);
    std::printf("Active Metal device: %d\n", jit_metal_device());
    std::fflush(stdout);
    std::printf("Metal device handle: %p\n", jit_metal_device_handle());
    std::fflush(stdout);
    std::printf("Metal command queue: %p\n", jit_metal_queue());
    std::fflush(stdout);
    std::printf("Hardware ray tracing supported: %d\n",
                jit_metal_supports_ray_tracing());
    std::fflush(stdout);

    // Allocate a small device buffer to exercise the malloc path.
    void *buf = jit_malloc(JitBackend::Metal, 1024);
    std::printf("Allocated 1KB device buffer at %p\n", buf);
    std::fflush(stdout);

    // Round-trip: allocate a host-pinned (== shared mode) buffer and write
    // some data via the unified memory address.
    void *host_buf = jit_malloc(JitBackend::Metal, 64, /*shared=*/1);
    std::printf("Allocated 64B host-pinned buffer at %p\n", host_buf);
    std::fflush(stdout);
    if (host_buf) {
        // Write through the CPU-visible pointer (shared storage).
        *((uint32_t *) host_buf) = 0xDEADBEEFu;
        std::printf("Wrote 0xDEADBEEF to host-pinned buffer; read back: 0x%X\n",
                    *((uint32_t *) host_buf));
        std::fflush(stdout);
    }

    // Free both.
    jit_free(buf);
    jit_free(host_buf);

    jit_shutdown(0);
    std::printf("Done.\n");
    std::fflush(stdout);
    return 0;
}
