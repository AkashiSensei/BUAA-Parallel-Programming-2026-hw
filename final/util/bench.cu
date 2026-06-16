#include "gemm_launch.h"
#include "mat_io.h"
#include "cuda_check.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void usage(const char *prog) {
    fprintf(stderr,
            "用法: %s -n N [选项]\n"
            "  在 GPU 上运行矩阵乘法并输出耗时（不做正确性校验）\n"
            "选项:\n"
            "  -n N                 矩阵规模（必填）\n"
            "  --data-dir D         数据目录（默认 ../data）\n"
            "  --warmup W           预热次数（默认 3）\n"
            "  --iters I            计时重复次数（默认 10）\n"
            "  --block B            线程块边长（默认 16）\n"
            "  --impl NAME          实现名称（写入 CSV，默认 gemm）\n"
            "  --csv-detail FILE    追加写入每次计时的 CSV 行\n"
            "  --csv-summary FILE   追加写入汇总统计 CSV 行\n",
            prog);
}

static void append_csv_detail(const char *path, const char *impl, int n, int block, int warmup,
                              int iters, const float *iter_ms) {
    if (!path) {
        return;
    }
    FILE *fp = fopen(path, "a");
    if (!fp) {
        fprintf(stderr, "无法写入 CSV: %s\n", path);
        return;
    }
    for (int i = 0; i < iters; ++i) {
        fprintf(fp, "%s,%d,%d,%d,%d,%.6f\n", impl, n, block, warmup, i, iter_ms[i]);
    }
    fclose(fp);
}

static void append_csv_summary(const char *path, const char *impl, int n, int block, int warmup,
                               int iters, double mean_ms, double std_ms, double var_ms,
                               double min_ms, double max_ms, double mean_gflops) {
    if (!path) {
        return;
    }
    FILE *fp = fopen(path, "a");
    if (!fp) {
        fprintf(stderr, "无法写入 CSV: %s\n", path);
        return;
    }
    fprintf(fp, "%s,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.2f\n", impl, n, block, warmup, iters,
            mean_ms, std_ms, var_ms, min_ms, max_ms, mean_gflops);
    fclose(fp);
}

int main(int argc, char **argv) {
    int n = -1;
    const char *data_dir = "../data";
    const char *impl = "gemm";
    const char *csv_detail = NULL;
    const char *csv_summary = NULL;
    int warmup = 3;
    int iters = 10;
    int block = 16;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--data-dir") == 0 && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (strcmp(argv[i], "--impl") == 0 && i + 1 < argc) {
            impl = argv[++i];
        } else if (strcmp(argv[i], "--csv-detail") == 0 && i + 1 < argc) {
            csv_detail = argv[++i];
        } else if (strcmp(argv[i], "--csv-summary") == 0 && i + 1 < argc) {
            csv_summary = argv[++i];
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--block") == 0 && i + 1 < argc) {
            block = atoi(argv[++i]);
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (n <= 0 || block <= 0 || warmup < 0 || iters <= 0) {
        usage(argv[0]);
        return 1;
    }

    float *iter_ms = static_cast<float *>(malloc((size_t)iters * sizeof(float)));
    if (!iter_ms) {
        fprintf(stderr, "内存分配失败\n");
        return 1;
    }

    char in_path[512];
    if (mat_input_path(data_dir, n, in_path, sizeof(in_path)) != 0) {
        free(iter_ms);
        usage(argv[0]);
        return 1;
    }

    MatDataHeader header;
    double *h_A = NULL;
    double *h_B = NULL;
    if (read_mat_pair(in_path, &header, &h_A, &h_B) != 0) {
        free(iter_ms);
        return 1;
    }
    if (header.n != n) {
        fprintf(stderr, "文件 %s 中 N=%d 与参数不符\n", in_path, header.n);
        free(h_A);
        free(h_B);
        free(iter_ms);
        return 1;
    }

    const size_t bytes = (size_t)n * (size_t)n * sizeof(double);
    double *d_A = NULL;
    double *d_B = NULL;
    double *d_C = NULL;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    const dim3 block_dim((unsigned)block, (unsigned)block);
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; ++i) {
        gemm_launch(n, d_A, d_B, d_C, block_dim, 0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        gemm_launch(n, d_A, d_B, d_C, block_dim, 0);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        iter_ms[i] = ms;
    }

    double total_ms = 0.0;
    double min_ms = iter_ms[0];
    double max_ms = iter_ms[0];
    for (int i = 0; i < iters; ++i) {
        total_ms += iter_ms[i];
        if (iter_ms[i] < min_ms) {
            min_ms = iter_ms[i];
        }
        if (iter_ms[i] > max_ms) {
            max_ms = iter_ms[i];
        }
    }

    const double mean_ms = total_ms / iters;
    double var_sum = 0.0;
    for (int i = 0; i < iters; ++i) {
        const double diff = iter_ms[i] - mean_ms;
        var_sum += diff * diff;
    }
    const double var_ms = (iters > 1) ? (var_sum / (iters - 1)) : 0.0;
    const double std_ms = std::sqrt(var_ms);
    const double mean_gflops = (2.0 * (double)n * (double)n * (double)n) / (mean_ms * 1e6);

    append_csv_detail(csv_detail, impl, n, block, warmup, iters, iter_ms);
    append_csv_summary(csv_summary, impl, n, block, warmup, iters, mean_ms, std_ms, var_ms,
                       min_ms, max_ms, mean_gflops);

    printf("N=%d block=%dx%d warmup=%d iters=%d mean_ms=%.4f std_ms=%.4f var_ms=%.6f "
           "min_ms=%.4f max_ms=%.4f gflops=%.2f\n",
           n, block, block, warmup, iters, mean_ms, std_ms, var_ms, min_ms, max_ms, mean_gflops);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(iter_ms);
    return 0;
}
