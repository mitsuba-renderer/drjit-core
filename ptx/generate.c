#include <stdio.h>
#include <stdlib.h>
#include <lz4hc.h>
#include <xxhash.h>

char *read_file(const char *fname, size_t *size_out) {
    FILE *f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "Could not open '%s'!\n", fname);
        exit(EXIT_FAILURE);
    }

    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    char *buf = malloc(size);
    fseek(f, 0, SEEK_SET);

    if (fread(buf, size, 1, f) != 1) {
        fprintf(stderr, "Could not read '%s'!\n", fname);
        exit(EXIT_FAILURE);
    }

    fclose(f);
    *size_out = size;
    return buf;
}

void dump_hex(FILE *f, const char *name, const char *data, size_t size) {
    fprintf(f, "uint8_t %s[] = {\n", name);
    for (size_t i = 0; i < size; ++i)
        fprintf(f, "%s0x%02x%s%s",
                i % 8 == 0 ? "    " : "",
                (unsigned) (uint8_t) data[i],
                i + 1 < size? "," : "",
                (i % 8 == 7 || i + 1 == size) ? "\n" : " ");

    fprintf(f, "};\n\n");
}

int main(int argc, char **argv) {
    (void) argc; (void) argv;

    size_t kernels_ptx_size, kernels_dict_size;
    char *kernels_ptx = read_file("kernels.ptx", &kernels_ptx_size);
    char *kernels_dict = read_file("kernels.dict", &kernels_dict_size);

    int compressed_size = LZ4_compressBound(kernels_ptx_size);
    char *compressed = malloc(compressed_size);

    LZ4_streamHC_t stream;
    LZ4_resetStreamHC_fast(&stream, LZ4HC_CLEVEL_MAX);
    LZ4_loadDictHC(&stream, kernels_dict, (int) kernels_dict_size);
    compressed_size = LZ4_compress_HC_continue(&stream, kernels_ptx,
            compressed, kernels_ptx_size, compressed_size);

    FILE *f = fopen("kernels.c", "w");
    if (!f) {
        fprintf(stderr, "Could not open 'kernels.c'!");
        exit(EXIT_FAILURE);
    }

    unsigned long long hash = XXH64(kernels_ptx, kernels_ptx_size, 0);

    fprintf(f, "#include \"kernels.h\"\n\n");
    fprintf(f, "uint32_t kernels_dict_size             = %i;\n", (int) kernels_dict_size);
    fprintf(f, "uint32_t kernels_ptx_size_uncompressed = %i;\n", (int) kernels_ptx_size);
    fprintf(f, "uint32_t kernels_ptx_size_compressed   = %i;\n", compressed_size);
    fprintf(f, "size_t   kernels_ptx_hash              = %lluull;\n\n", hash);
    dump_hex(f, "kernels_dict", kernels_dict, kernels_dict_size);
    dump_hex(f, "kernels_ptx", compressed, compressed_size);
    fclose(f);

    f = fopen("kernels.h", "w");
    if (!f) {
        fprintf(stderr, "Could not open 'kernels.h'!");
        exit(EXIT_FAILURE);
    }

    fprintf(f, "#pragma once\n\n");
    fprintf(f, "#include <stdint.h>\n\n");
    fprintf(f, "#include <stddef.h>\n\n");
    fprintf(f, "#if defined(__cplusplus)\n");
    fprintf(f, "extern \"C\" {\n");
    fprintf(f, "#endif\n\n");
    fprintf(f, "extern uint32_t kernels_dict_size;\n");
    fprintf(f, "extern uint32_t kernels_ptx_size_uncompressed;\n");
    fprintf(f, "extern uint32_t kernels_ptx_size_compressed;\n");
    fprintf(f, "extern size_t kernels_ptx_hash;\n");
    fprintf(f, "extern uint8_t kernels_dict[];\n");
    fprintf(f, "extern uint8_t kernels_ptx[];\n\n");
    fprintf(f, "#if defined(__cplusplus)\n");
    fprintf(f, "}\n");
    fprintf(f, "#endif");
    fclose(f);

    free(compressed);
    free(kernels_ptx);
    free(kernels_dict);

    return EXIT_SUCCESS;
}
