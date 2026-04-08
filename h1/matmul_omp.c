/*
 * 实验 1：并行方阵乘法（OpenMP）
 * 用法: ./matmul_omp <N> <threads>
 *   threads=0  : 纯串行（无 OpenMP 并行区）
 *   threads>=1 : OpenMP parallel for，线程数为给定值（1 亦为 OpenMP，单线程）
 * 标准输出一行（便于脚本解析）: N threads time_sec
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

static void matmul_serial(const double *A, const double *B, double *C, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      double sum = 0.0;
      for (int k = 0; k < n; k++) {
        sum += A[(size_t)i * (size_t)n + (size_t)k] *
               B[(size_t)k * (size_t)n + (size_t)j];
      }
      C[(size_t)i * (size_t)n + (size_t)j] = sum;
    }
  }
}

static void matmul_omp(const double *A, const double *B, double *C, int n,
                       int num_threads) {
  omp_set_num_threads(num_threads);
#pragma omp parallel for default(none) shared(A, B, C, n) schedule(static)
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      double sum = 0.0;
      for (int k = 0; k < n; k++) {
        sum += A[(size_t)i * (size_t)n + (size_t)k] *
               B[(size_t)k * (size_t)n + (size_t)j];
      }
      C[(size_t)i * (size_t)n + (size_t)j] = sum;
    }
  }
}

static void init_matrices(double *A, double *B, int n) {
  const size_t nn = (size_t)n * (size_t)n;
  for (size_t i = 0; i < nn; i++) {
    A[i] = (double)((i * 17 + 3) % 1000) * 0.001;
    B[i] = (double)((i * 31 + 7) % 1000) * 0.001;
  }
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    fprintf(stderr, "用法: %s <N> <threads>\n", argv[0]);
    fprintf(stderr, "  threads=0  纯串行；threads>=1  OpenMP（含 1 线程）\n");
    fprintf(stderr, "  输出一行: N threads time_sec\n");
    return 1;
  }

  char *end = NULL;
  long n_l = strtol(argv[1], &end, 10);
  if (end == argv[1] || *end != '\0' || n_l < 1 || n_l > 200000) {
    fprintf(stderr, "错误: N 应为 1..200000 范围内的整数\n");
    return 1;
  }
  int n = (int)n_l;

  end = NULL;
  long t_l = strtol(argv[2], &end, 10);
  if (end == argv[2] || *end != '\0' || t_l < 0 || t_l > 65536) {
    fprintf(stderr, "错误: threads 应为 0..65536 的整数\n");
    return 1;
  }
  int threads = (int)t_l;

  const size_t nn = (size_t)n * (size_t)n;
  const size_t bytes = nn * sizeof(double);

  double *A = (double *)malloc(bytes);
  double *B = (double *)malloc(bytes);
  double *C = (double *)malloc(bytes);
  if (!A || !B || !C) {
    fprintf(stderr, "错误: 内存分配失败（约需 %.2f GiB）\n",
            (3.0 * (double)bytes) / (1024.0 * 1024.0 * 1024.0));
    free(A);
    free(B);
    free(C);
    return 1;
  }

  init_matrices(A, B, n);

  double t0 = omp_get_wtime();
  if (threads == 0) {
    matmul_serial(A, B, C, n);
  } else {
    matmul_omp(A, B, C, n, threads);
  }
  double t1 = omp_get_wtime();
  const double elapsed = t1 - t0;

  printf("%d %d %.9f\n", n, threads, elapsed);

  free(A);
  free(B);
  free(C);
  return 0;
}
