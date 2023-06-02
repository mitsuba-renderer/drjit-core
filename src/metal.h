/// Initialize the metal backend
extern bool jitc_metal_init();

/// Shut down the metal backend
extern void jitc_metal_shutdown();

/// Initialize a per-thread state (requires that the backend is initialized)
extern ThreadState *jitc_metal_thread_state_new();

/// Release a memory allocation made by the Metal backend
extern void jitc_metal_free(int device_id, bool shared, void *ptr);
