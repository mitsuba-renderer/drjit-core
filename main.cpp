#include "api.h"
#include <stdio.h>
#include <iostream>

int main(int argc, char **argv) {
    try {
        jitc_init();
        jitc_set_context(0, 0);

        void *ptr = jitc_malloc(AllocType::Device, 1024);
        void *ptr2 = jitc_malloc(AllocType::Host, 1025);
        void *ptr3 = jitc_malloc(AllocType::HostPinned, 1025);
        void *ptr4 = jitc_malloc(AllocType::ManagedReadMostly, 1025);
        jitc_free(ptr);
        printf("Found %u devices.\n", jitc_device_count());

        for (int i = 0; i < 3; ++i) {
            ptr = jitc_malloc(AllocType::Device, 1024);
            jitc_free(ptr);
        }

        jitc_set_context(1, 0);
        for (int i = 0; i < 3; ++i) {
            ptr = jitc_malloc(AllocType::Device, 1024);
            jitc_free(ptr);
            // jitc_flush_free();
            jitc_device_sync();
        }
        jitc_free(ptr2);
        jitc_free(ptr3);
        jitc_free(ptr4);


        jitc_shutdown();
    } catch (const std::exception &e) {
        std::cout << "Exception: "<< e.what() << std::endl;
    } catch (...) {
        std::cout << "Uncaught exception!" << std::endl;
    }
}
