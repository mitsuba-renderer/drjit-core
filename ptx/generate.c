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
    fprintf(f, "char %s[] = {\n", name);
    for (size_t i = 0; i < size; ++i)
        fprintf(f, "%s0x%02x%s%s",
                i % 8 == 0 ? "    " : "",
                (unsigned) (uint8_t) data[i],
                i + 1 < size? "," : "",
                (i % 8 == 7 || i + 1 == size) ? "\n" : " ");

    fprintf(f, "};\n\n");
}

void append(FILE *f, const char *filename, const char *prefix, char *dict, int dict_size) {
    size_t size;
    char *buf = read_file(filename, &size);

    int compressed_size = LZ4_compressBound(size);
    char *compressed = malloc(compressed_size);

    unsigned long long hash = XXH64(buf, size, 0);

    LZ4_streamHC_t stream;
    LZ4_resetStreamHC_fast(&stream, LZ4HC_CLEVEL_MAX);
    if (dict)
        LZ4_loadDictHC(&stream, dict, dict_size);
    compressed_size = LZ4_compress_HC_continue(&stream, buf,
            compressed, size, compressed_size);

    free(buf);
    free(compressed);

    fprintf(f, "int %s_size_uncompressed = %zu;\n", prefix, size);
    fprintf(f, "int %s_size_compressed   = %i;\n", prefix, compressed_size);
    fprintf(f, "size_t  %s_hash          = %lluull;\n\n", prefix, hash);
    dump_hex(f, prefix, compressed, compressed_size);
}

int main(int argc, char **argv) {
    (void) argc; (void) argv;

    FILE *f = fopen("kernels.c", "w");
    if (!f) {
        fprintf(stderr, "Could not open 'kernels.c'!");
        exit(EXIT_FAILURE);
    }


    fprintf(f, "#include \"kernels.h\"\n\n");

    size_t kernels_dict_size;
    char *kernels_dict = read_file("kernels.dict", &kernels_dict_size);

    append(f, "kernels.dict", "kernels_dict", NULL, 0);
    append(f, "kernels_50.ptx", "kernels_50", kernels_dict, kernels_dict_size);
    append(f, "kernels_70.ptx", "kernels_70", kernels_dict, kernels_dict_size);

    f = fopen("kernels.h", "w");
    if (!f) {
        fprintf(stderr, "Could not open 'kernels.h'!");
        exit(EXIT_FAILURE);
    }

    fprintf(f, "#pragma once\n\n");
    fprintf(f, "#include <stddef.h>\n\n");
    fprintf(f, "#if defined(__cplusplus)\n");
    fprintf(f, "extern \"C\" {\n");
    fprintf(f, "#endif\n\n");
    fprintf(f, "extern int    kernels_dict_size_uncompressed;\n");
    fprintf(f, "extern int    kernels_dict_size_compressed;\n");
    fprintf(f, "extern size_t kernels_dict_hash;\n");
    fprintf(f, "extern char   kernels_dict[];\n\n");
    fprintf(f, "extern int    kernels_50_size_uncompressed;\n");
    fprintf(f, "extern int    kernels_50_size_compressed;\n");
    fprintf(f, "extern size_t kernels_50_hash;\n");
    fprintf(f, "extern char   kernels_50[];\n\n");
    fprintf(f, "extern int    kernels_70_size_uncompressed;\n");
    fprintf(f, "extern int    kernels_70_size_compressed;\n");
    fprintf(f, "extern size_t kernels_70_hash;\n");
    fprintf(f, "extern char   kernels_70[];\n\n");
    fprintf(f, "#if defined(__cplusplus)\n");
    fprintf(f, "}\n");
    fprintf(f, "#endif");
    fclose(f);

    free(kernels_dict);

    return EXIT_SUCCESS;
}
