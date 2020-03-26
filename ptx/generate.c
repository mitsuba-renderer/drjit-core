#include <stdio.h>
#include <stdlib.h>
#include <lz4hc.h>

int main(int argc, char **argv) {
    FILE *f = fopen("kernels.ptx", "r");
    if (!f) {
        fprintf(stderr, "Could not open 'kernels.ptx'!\n");
        exit(EXIT_FAILURE);
    }

    fseek(f, 0, SEEK_END);
    int size = (int) ftell(f);
    char *buf = malloc(size);
    fseek(f, 0, SEEK_SET);

    if (fread(buf, size, 1, f) != 1) {
        fprintf(stderr, "Could not read 'kernels.ptx'!\n");
        exit(EXIT_FAILURE);
    }

    fclose(f);

    int temp_size = LZ4_compressBound(size);
    char *temp = malloc(temp_size);

    LZ4_stream_t stream;
    LZ4_resetStream_fast(&stream);

    int compressed_size = LZ4_compress_HC(
        buf, temp, size, temp_size, LZ4HC_CLEVEL_MAX);

    f = fopen("kernels.h", "w");
    if (!f) {
        fprintf(stderr, "Could not open 'kernels.h'!");
        exit(EXIT_FAILURE);
    }

    fprintf(f, "#pragma once\n\n");
    fprintf(f, "#include <stdint.h>\n\n");
    fprintf(f, "static uint32_t kernels_ptx_size_uncompressed = %i;\n", size);
    fprintf(f, "static uint32_t kernels_ptx_size_compressed   = %i;\n\n", compressed_size);
    fprintf(f, "static uint8_t kernels_ptx[] = {\n");
    for (int i = 0; i < compressed_size; ++i)
        fprintf(f, "%s0x%02x%s%s",
                i % 8 == 0 ? "    " : "",
                (unsigned) (uint8_t) temp[i],
                i + 1 < compressed_size ? "," : "",
                (i % 8 == 7 || i + 1 == compressed_size) ? "\n" : " ");

    fprintf(f, "};\n");

    free(temp);
    free(buf);
    fclose(f);

    return EXIT_SUCCESS;
}
