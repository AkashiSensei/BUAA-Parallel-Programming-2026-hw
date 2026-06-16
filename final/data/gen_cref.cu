#include "mat_io.h"
#include "cuda_check.h"
#include "cublas_check.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

static void usage(const char *prog) {
    fprintf(stderr,
            "用法: %s [-i DIR] [-o DIR] N1 [N2 ...]\n"
            "  读取 DIR/N{N}.bin 中的 A、B，用 cuBLAS 在 GPU 上计算 C=AB 并保存为 DIR/N{N}_cref.bin\n"
            "  默认输入/输出目录: 当前目录\n",
            prog);
}

static int compute_and_save_cref(const char *in_dir, const char *out_dir, int32_t n) {
    char in_path[512];
    char out_path[512];
    snprintf(in_path, sizeof(in_path), "%s/N%d.bin", in_dir, n);
    snprintf(out_path, sizeof(out_path), "%s/N%d_cref.bin", out_dir, n);

    MatDataHeader header;
    double *h_A = NULL;
    double *h_B = NULL;
    if (read_mat_pair(in_path, &header, &h_A, &h_B) != 0) {
        return 1;
    }
    if (header.n != n) {
        fprintf(stderr, "文件 %s 中的 N=%d 与参数 N=%d 不一致\n", in_path, header.n, n);
        free(h_A);
        free(h_B);
        return 1;
    }

    const size_t count = (size_t)n * (size_t)n;
    double *h_C = static_cast<double *>(malloc(count * sizeof(double)));
    if (!h_C) {
        fprintf(stderr, "内存分配失败 (N=%d)\n", n);
        free(h_A);
        free(h_B);
        return 1;
    }

    double *d_A = NULL;
    double *d_B = NULL;
    double *d_C = NULL;
    CUDA_CHECK(cudaMalloc(&d_A, count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, count * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, count * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, count * sizeof(double), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const double alpha = 1.0;
    const double beta = 0.0;
    /* 行主序 C = A*B 等价于在 cuBLAS 列主序下计算 C^T = B^T * A^T */
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, n, n, n, &alpha, d_A, n, d_B,
                             n, &beta, d_C, n));

    CUBLAS_CHECK(cublasDestroy(handle));

    double *h_C_col = static_cast<double *>(malloc(count * sizeof(double)));
    if (!h_C_col) {
        fprintf(stderr, "内存分配失败 (N=%d)\n", n);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(h_A);
        free(h_B);
        free(h_C);
        return 1;
    }
    CUDA_CHECK(cudaMemcpy(h_C_col, d_C, count * sizeof(double), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            h_C[i * n + j] = h_C_col[i + j * n];
        }
    }
    free(h_C_col);

    if (write_mat_cref(out_path, n, h_C) != 0) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(h_A);
        free(h_B);
        free(h_C);
        return 1;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    printf("已生成 %s (N=%d, size=%.2f MiB)\n", out_path, n,
           mat_cref_bytes(n) / (1024.0 * 1024.0));
    return 0;
}

int main(int argc, char **argv) {
    const char *in_dir = ".";
    const char *out_dir = ".";

    int argi = 1;
    while (argi < argc && argv[argi][0] == '-') {
        if (strcmp(argv[argi], "-i") == 0 && argi + 1 < argc) {
            in_dir = argv[++argi];
        } else if (strcmp(argv[argi], "-o") == 0 && argi + 1 < argc) {
            out_dir = argv[++argi];
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

    int status = 0;
    for (; argi < argc; ++argi) {
        char *end = NULL;
        long n_long = strtol(argv[argi], &end, 10);
        if (end == argv[argi] || *end != '\0' || n_long <= 0 || n_long > INT32_MAX) {
            fprintf(stderr, "非法矩阵规模: %s\n", argv[argi]);
            status = 1;
            continue;
        }

        if (compute_and_save_cref(in_dir, out_dir, static_cast<int32_t>(n_long)) != 0) {
            status = 1;
        }
    }

    return status;
}
