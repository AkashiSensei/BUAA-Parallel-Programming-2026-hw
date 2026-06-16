#define _DEFAULT_SOURCE
#include "mat_data.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static void usage(const char *prog) {
    fprintf(stderr,
            "用法: %s [-o DIR] [-s SEED] N1 [N2 ...]\n"
            "  为每个 N 生成 N{N}.bin，包含行主序 double 矩阵 A 与 B（不含 C）。\n"
            "  默认输出目录: 当前目录\n"
            "  默认随机种子: %u\n",
            prog, MAT_DATA_DEFAULT_SEED);
}

static int ensure_dir(const char *dir) {
    struct stat st;
    if (stat(dir, &st) == 0) {
        return S_ISDIR(st.st_mode) ? 0 : 1;
    }
    if (mkdir(dir, 0755) == 0) {
        return 0;
    }
    fprintf(stderr, "无法创建目录 %s: %s\n", dir, strerror(errno));
    return 1;
}

static int write_matrix_pair(const char *path, int32_t n, uint32_t seed) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "无法创建 %s: %s\n", path, strerror(errno));
        return 1;
    }

    MatDataHeader header = {
        .magic = MAT_DATA_MAGIC,
        .version = MAT_DATA_VERSION,
        .n = n,
        .seed = seed,
    };

    if (fwrite(&header, sizeof(header), 1, fp) != 1) {
        fprintf(stderr, "写入文件头失败: %s\n", path);
        fclose(fp);
        return 1;
    }

    const size_t count = (size_t)n * (size_t)n;
    double *matrix = malloc(count * sizeof(double));
    if (!matrix) {
        fprintf(stderr, "内存分配失败 (N=%d)\n", n);
        fclose(fp);
        return 1;
    }

    srand48((long)seed);
    for (size_t i = 0; i < count; ++i) {
        matrix[i] = drand48();
    }
    if (fwrite(matrix, sizeof(double), count, fp) != count) {
        fprintf(stderr, "写入矩阵 A 失败: %s\n", path);
        free(matrix);
        fclose(fp);
        return 1;
    }

    seed += 1u;
    srand48((long)seed);
    for (size_t i = 0; i < count; ++i) {
        matrix[i] = drand48();
    }
    if (fwrite(matrix, sizeof(double), count, fp) != count) {
        fprintf(stderr, "写入矩阵 B 失败: %s\n", path);
        free(matrix);
        fclose(fp);
        return 1;
    }

    free(matrix);
    if (fclose(fp) != 0) {
        fprintf(stderr, "关闭文件失败: %s\n", path);
        return 1;
    }

    printf("已生成 %s (N=%d, seed=%u, size=%.2f MiB)\n",
           path, n, seed - 1u, mat_data_bytes(n) / (1024.0 * 1024.0));
    return 0;
}

int main(int argc, char **argv) {
    const char *out_dir = ".";
    uint32_t base_seed = MAT_DATA_DEFAULT_SEED;

    int argi = 1;
    while (argi < argc && argv[argi][0] == '-') {
        if (strcmp(argv[argi], "-o") == 0 && argi + 1 < argc) {
            out_dir = argv[++argi];
        } else if (strcmp(argv[argi], "-s") == 0 && argi + 1 < argc) {
            base_seed = (uint32_t)strtoul(argv[++argi], NULL, 10);
        } else if (strcmp(argv[argi], "-h") == 0 || strcmp(argv[argi], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 1;
        }
        ++argi;
    }

    if (argi >= argc) {
        usage(argv[0]);
        return 1;
    }

    if (strcmp(out_dir, ".") != 0 && ensure_dir(out_dir) != 0) {
        return 1;
    }

    int status = 0;
    for (; argi < argc; ++argi) {
        char *end = NULL;
        long n_long = strtol(argv[argi], &end, 10);
        if (end == argv[argi] || *end != '\0' || n_long <= 0 || n_long > INT32_MAX) {
            fprintf(stderr, "非法矩阵规模: %s\n", argv[argi]);
            status = 1;
            continue;
        }

        int32_t n = (int32_t)n_long;
        char path[512];
        snprintf(path, sizeof(path), "%s/N%d.bin", out_dir, n);

        uint32_t seed = base_seed + (uint32_t)n;
        if (write_matrix_pair(path, n, seed) != 0) {
            status = 1;
        }
    }

    return status;
}
