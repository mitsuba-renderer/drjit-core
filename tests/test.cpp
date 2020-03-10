#include <enoki/jitvar.h>
#include <stdexcept>
#include <vector>
#include "unistd.h"

struct Test {
    const char *name;
    void (*func)();
    bool cuda;
};

static std::vector<Test> *tests = nullptr;

/**
 * Strip away information (pointers, etc.) to that it becomes possible to
 * compare the debug output of a test to a reference file
 */
char* test_sanitize_log() {
    char *buf = jitc_log_buffer(),
         *src = buf,
         *dst = src;

    // Remove all lines starting with the following text
    const char *excise[5] = {
        "jit_init(): detecting",
        " - Found CUDA",
        " - Enabling peer",
        "info    :",
        "jit_run(): cache "
    };

    while (*src != '\0') {
        char c = *src;

        if (src == buf || src[-1] == '\n') {
            bool found = false;
            for (const char *str : excise) {
                if (strncmp(str, src, strlen(str)) == 0) {
                    while (*src != '\0' && src[0] != '\n')
                        src++;
                    if (*src == '\n')
                        src++;
                    found = true;
                    break;
                }
            }
            if (found)
                continue;
        }

        // Excise pointers, since they will differ from run to run
        if (c == '<' && src[1] == '0' && src[2] == 'x') {
            src += 3; c = *src;
            while ((c >= '0' && c <= '9') ||
                   (c >= 'a' && c <= 'f') ||
                   (c >= 'A' && c <= 'F') ||
                    c == '>')
                c = *++src;
            *dst++ = '<';
            *dst++ = '@';
            *dst++ = '>';
            continue;
        }

        *dst++ = *src++;
    }
    *dst = '\0';
    return buf;
}

bool test_check_log(const char *test_name, const char *log, bool write_ref) {
    char test_fname[128];

    snprintf(test_fname, 128, "output/%s/%s.%s", TEST_NAME,
             test_name, write_ref ? "ref" : "out");

    FILE *f = fopen(test_fname, "w");
    if (!f) {
        fprintf(stderr, "\ntest_check_log(): Could not create file \"%s\"!\n", test_fname);
        exit(EXIT_FAILURE);
    }

    size_t log_len = strlen(log);
    if (fwrite(log, log_len, 1, f) != 1) {
        fprintf(stderr, "\ntest_check_log(): Error writing to \"%s\"!\n", test_fname);
        exit(EXIT_FAILURE);
    }
    fclose(f);

    if (write_ref)
        return true;

    snprintf(test_fname, 128, "output/%s/%s.ref", TEST_NAME, test_name);
    f = fopen(test_fname, "r");
    if (!f) {
        fprintf(stderr, "\ntest_check_log(): Could not open file \"%s\"!\n", test_fname);
        exit(EXIT_FAILURE);
    }

    bool result = true;

    fseek(f, 0, SEEK_END);
    result = (size_t) ftell(f) == log_len;
    fseek(f, 0, SEEK_SET);

    if (result) {
        char *tmp = (char *) malloc(log_len);
        if (fread(tmp, log_len, 1, f) != 1) {
            fprintf(stderr, "\ntest_check_log(): Could not read file \"%s\"!\n", test_fname);
            exit(EXIT_FAILURE);
        }

        result = memcmp(tmp, log, log_len) == 0;
        free(tmp);
    }

    fclose(f);
    return result;
}

int test_register(const char *name, void (*func)(), bool cuda) {
    if (!tests)
        tests = new std::vector<Test>();
    tests->push_back(Test{ name, func, cuda });
    return 0;
}

int main(int argc, char **argv) {
    char binary_path[PATH_MAX];
    if (readlink("/proc/self/exe", binary_path, PATH_MAX) == -1) {
        fprintf(stderr, "Unable to determine binary path!");
        exit(EXIT_FAILURE);
    }

    char *last_slash = strrchr(binary_path, '/');
    if (!last_slash) {
        fprintf(stderr, "Invalid binary path!");
        exit(EXIT_FAILURE);
    }
    *last_slash='\0';

    if (chdir(binary_path) == -1) {
        fprintf(stderr, "Could not change working directory!");
        exit(EXIT_FAILURE);
    }

    bool write_ref = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-w") == 0) {
            write_ref = true;
        } else {
            fprintf(stderr, "Invalid command line argument: \"%s\"\n", argv[i]);
            exit(EXIT_FAILURE);
        }
    }

    try {
        jitc_log_level_set(4);
        jitc_log_buffer_enable(1);
        fprintf(stdout, "\n");

        if (!tests) {
            fprintf(stderr, "No tests registered!\n");
            exit(EXIT_FAILURE);
        }

        std::sort(tests->begin(), tests->end(), [](const Test &a, const Test &b) {
            return strcmp(a.name, b.name) < 0;
        });

        int tests_passed = 0,
            tests_failed = 0;

        for (auto &test : *tests) {
            fprintf(stdout, " - %s .. ", test.name);
            fflush(stdout);
            jitc_init();
            jitc_device_set(test.cuda ? 0 : -1, 0);
            test.func();
            jitc_shutdown();

            char *log = test_sanitize_log();
            if (test_check_log(test.name, log, write_ref)) {
                tests_passed++;
                fprintf(stdout, "passed.\n");
            } else {
                tests_failed++;
                fprintf(stdout, "FAILED!\n");
            }

            free(log);
        }

        fprintf(stdout, "\nPassed %i/%i tests.\n", tests_passed,
                tests_passed + tests_failed);

        return tests_failed == 0 ? 0 : EXIT_FAILURE;
    } catch (const std::exception &e) {
        fprintf(stderr, "Exception: %s!\n", e.what());
    }
}
