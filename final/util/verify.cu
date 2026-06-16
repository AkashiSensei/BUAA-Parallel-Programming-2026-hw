#include "gemm_launch.h"
#include "mat_io.h"
#include "mat_compare.h"
#include "cuda_check.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

static void usage(const char *prog) {
    fprintf(stderr,
            "用法: %s -n N [选项]\n"
            "  运行矩阵乘法并与 cuBLAS 参考结果比较\n"
            "选项:\n"
            "  -n N           矩阵规模（必填）\n"
            "  --data-dir D   数据目录（默认 ../data）\n"
            "  --block B      线程块边长（默认 16）\n"
            "  --tol T        最大相对误差阈值（默认 1e-10）\n",
            prog);
}

int main(int argc, char **argv) {
    int n = -1;
    const char *data_dir = "../data";
    int block = 16;
    double tol = 1e-10;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--data-dir") == 0 && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (strcmp(argv[i], "--block") == 0 && i + 1 < argc) {
            block = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--tol") == 0 && i + 1 < argc) {
            tol = atof(argv[++i]);
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (n <= 0 || block <= 0) {
        usage(argv[0]);
        return 1;
    }

    char in_path[512];
    char cref_path[512];
    if (mat_input_path(data_dir, n, in_path, sizeof(in_path)) != 0 ||
        mat_cref_path(data_dir, n, cref_path, sizeof(cref_path)) != 0) {
        usage(argv[0]);
        return 1;
    }

    MatDataHeader in_header;
    MatCRefHeader cref_header;
    double *h_A = NULL;
    double *h_B = NULL;
    double *h_C_ref = NULL;
    if (read_mat_pair(in_path, &in_header, &h_A, &h_B) != 0) {
        return 1;
    }
    if (read_mat_cref(cref_path, &cref_header, &h_C_ref) != 0) {
        free(h_A);
        free(h_B);
        return 1;
    }
    if (in_header.n != n || cref_header.n != n) {
        fprintf(stderr, "输入与参考结果的矩阵规模不一致\n");
        free(h_A);
        free(h_B);
        free(h_C_ref);
        return 1;
    }

    const size_t count = (size_t)n * (size_t)n;
    const size_t bytes = count * sizeof(double);
    double *h_C = static_cast<double *>(malloc(bytes));
    if (!h_C) {
        fprintf(stderr, "内存分配失败\n");
        free(h_A);
        free(h_B);
        free(h_C_ref);
        return 1;
    }

    double *d_A = NULL;
    double *d_B = NULL;
    double *d_C = NULL;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    const dim3 block_dim((unsigned)block, (unsigned)block);
    gemm_launch(n, d_A, d_B, d_C, block_dim, 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    const double max_rel = mat_max_rel_err(n, h_C, h_C_ref, 1e-12);
    const int ok = max_rel <= tol;
    printf("N=%d max_rel_err=%.3e tol=%.1e %s\n", n, max_rel, tol, ok ? "PASS" : "FAIL");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_ref);
    free(h_C);
    return ok ? 0 : 1;
}
