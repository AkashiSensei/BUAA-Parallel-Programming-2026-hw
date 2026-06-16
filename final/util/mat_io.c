#include "mat_io.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int mat_input_path(const char *dir, int32_t n, char *buf, size_t buflen) {
    if (!dir || !buf || buflen == 0 || n <= 0) {
        return 1;
    }
    if (snprintf(buf, buflen, "%s/N%d.bin", dir, n) >= (int)buflen) {
        return 1;
    }
    return 0;
}

int mat_cref_path(const char *dir, int32_t n, char *buf, size_t buflen) {
    if (!dir || !buf || buflen == 0 || n <= 0) {
        return 1;
    }
    if (snprintf(buf, buflen, "%s/N%d_cref.bin", dir, n) >= (int)buflen) {
        return 1;
    }
    return 0;
}

int read_mat_pair(const char *path, MatDataHeader *header, double **A, double **B) {
    if (!path || !header || !A || !B) {
        return 1;
    }

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "无法打开 %s: %s\n", path, strerror(errno));
        return 1;
    }

    if (fread(header, sizeof(*header), 1, fp) != 1) {
        fprintf(stderr, "读取文件头失败: %s\n", path);
        fclose(fp);
        return 1;
    }

    if (header->magic != MAT_DATA_MAGIC) {
        fprintf(stderr, "文件 magic 不匹配: %s\n", path);
        fclose(fp);
        return 1;
    }
    if (header->version != MAT_DATA_VERSION) {
        fprintf(stderr, "不支持的输入文件版本 %u: %s\n", header->version, path);
        fclose(fp);
        return 1;
    }
    if (header->n <= 0) {
        fprintf(stderr, "非法矩阵规模 N=%d: %s\n", header->n, path);
        fclose(fp);
        return 1;
    }

    const size_t count = (size_t)header->n * (size_t)header->n;
    double *a = malloc(count * sizeof(double));
    double *b = malloc(count * sizeof(double));
    if (!a || !b) {
        fprintf(stderr, "内存分配失败 (N=%d)\n", header->n);
        free(a);
        free(b);
        fclose(fp);
        return 1;
    }

    if (fread(a, sizeof(double), count, fp) != count) {
        fprintf(stderr, "读取矩阵 A 失败: %s\n", path);
        free(a);
        free(b);
        fclose(fp);
        return 1;
    }
    if (fread(b, sizeof(double), count, fp) != count) {
        fprintf(stderr, "读取矩阵 B 失败: %s\n", path);
        free(a);
        free(b);
        fclose(fp);
        return 1;
    }

    if (fclose(fp) != 0) {
        fprintf(stderr, "关闭文件失败: %s\n", path);
        free(a);
        free(b);
        return 1;
    }

    *A = a;
    *B = b;
    return 0;
}

int read_mat_cref(const char *path, MatCRefHeader *header, double **C) {
    if (!path || !header || !C) {
        return 1;
    }

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "无法打开 %s: %s\n", path, strerror(errno));
        return 1;
    }

    if (fread(header, sizeof(*header), 1, fp) != 1) {
        fprintf(stderr, "读取参考结果文件头失败: %s\n", path);
        fclose(fp);
        return 1;
    }

    if (header->magic != MAT_CREF_MAGIC) {
        fprintf(stderr, "参考结果文件 magic 不匹配: %s\n", path);
        fclose(fp);
        return 1;
    }
    if (header->version != MAT_CREF_VERSION) {
        fprintf(stderr, "不支持的参考结果文件版本 %u: %s\n", header->version, path);
        fclose(fp);
        return 1;
    }
    if (header->n <= 0) {
        fprintf(stderr, "非法矩阵规模 N=%d: %s\n", header->n, path);
        fclose(fp);
        return 1;
    }

    const size_t count = (size_t)header->n * (size_t)header->n;
    double *c = malloc(count * sizeof(double));
    if (!c) {
        fprintf(stderr, "内存分配失败 (N=%d)\n", header->n);
        fclose(fp);
        return 1;
    }

    if (fread(c, sizeof(double), count, fp) != count) {
        fprintf(stderr, "读取矩阵 C 失败: %s\n", path);
        free(c);
        fclose(fp);
        return 1;
    }

    if (fclose(fp) != 0) {
        fprintf(stderr, "关闭文件失败: %s\n", path);
        free(c);
        return 1;
    }

    *C = c;
    return 0;
}

int write_mat_cref(const char *path, int32_t n, const double *C) {
    if (!path || !C || n <= 0) {
        return 1;
    }

    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "无法创建 %s: %s\n", path, strerror(errno));
        return 1;
    }

    MatCRefHeader header = {
        .magic = MAT_CREF_MAGIC,
        .version = MAT_CREF_VERSION,
        .n = n,
    };

    if (fwrite(&header, sizeof(header), 1, fp) != 1) {
        fprintf(stderr, "写入参考结果文件头失败: %s\n", path);
        fclose(fp);
        return 1;
    }

    const size_t count = (size_t)n * (size_t)n;
    if (fwrite(C, sizeof(double), count, fp) != count) {
        fprintf(stderr, "写入矩阵 C 失败: %s\n", path);
        fclose(fp);
        return 1;
    }

    if (fclose(fp) != 0) {
        fprintf(stderr, "关闭文件失败: %s\n", path);
        return 1;
    }

    return 0;
}
