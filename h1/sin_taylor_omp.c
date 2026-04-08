/*
 * 实验 2：固定 x，对 sin(x) 的泰勒展开按项并行求和（OpenMP）
 * 用法: ./sin_taylor_omp <x> <n_terms> <threads>
 *   threads=0 : 纯串行；threads>=1 : OpenMP，按行区间划分，每段用闭式首项 + 递推
 * 输出一行: x n_terms threads time_sec
 */

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

/* 第 i 项：(-1)^i * x^(2i+1) / (2i+1)!，用于各线程区间起点，避免从 j=0 快进的 O(n*线程数) 冗余 */
static double taylor_term_at(double x, long i) {
  const double sign_alt = (i & 1L) ? -1.0 : 1.0;
  const double ax = fabs(x);
  if (ax == 0.0) {
    return (i == 0) ? x : 0.0;
  }
  const double di = (double)(2 * i + 1);
  const double lfac = lgamma((double)(2 * i + 2));
  double lm = di * log(ax) - lfac;
  if (lm < -745.0) {
    return 0.0;
  }
  const double mag = exp(lm);
  return sign_alt * copysign(mag, x);
}

static double sin_taylor_serial(double x, long n) {
  if (n <= 0) {
    return 0.0;
  }
  double sum = 0.0;
  double term = x;
  for (long i = 0; i < n; i++) {
    sum += term;
    term *= -x * x / (double)((2 * i + 2) * (2 * i + 3));
  }
  return sum;
}

static double sin_taylor_omp(double x, long n, int num_threads) {
  if (n <= 0) {
    return 0.0;
  }
  double sum = 0.0;
  omp_set_num_threads(num_threads);
#pragma omp parallel default(none) shared(x, n) reduction(+ : sum)
  {
    const int tid = omp_get_thread_num();
    const int nt = omp_get_num_threads();
    const long chunk = (n + (long)nt - 1) / (long)nt;
    const long istart = (long)tid * chunk;
    const long iend = istart + chunk < n ? istart + chunk : n;

    double partial = 0.0;
    if (istart < iend) {
      double term = taylor_term_at(x, istart);
      for (long i = istart; i < iend; i++) {
        partial += term;
        term *= -x * x / (double)((2 * i + 2) * (2 * i + 3));
      }
    }
    sum += partial;
  }
  return sum;
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    fprintf(stderr, "用法: %s <x> <n_terms> <threads>\n", argv[0]);
    fprintf(stderr, "  x         自变量（双精度浮点数）\n");
    fprintf(stderr, "  n_terms   级数项数（非负整数）\n");
    fprintf(stderr, "  threads   0=串行；>=1 为 OpenMP 线程数\n");
    return 1;
  }

  char *end = NULL;
  double x = strtod(argv[1], &end);
  if (end == argv[1]) {
    fprintf(stderr, "错误: x 无法解析为浮点数\n");
    return 1;
  }

  end = NULL;
  long n = strtol(argv[2], &end, 10);
  if (end == argv[2] || *end != '\0' || n < 0) {
    fprintf(stderr, "错误: n_terms 应为合理的非负整数\n");
    return 1;
  }

  end = NULL;
  long t_l = strtol(argv[3], &end, 10);
  if (end == argv[3] || *end != '\0' || t_l < 0 || t_l > 65536) {
    fprintf(stderr, "错误: threads 应为 0..65536\n");
    return 1;
  }
  int threads = (int)t_l;

  double t0 = omp_get_wtime();
  volatile double result;
  if (threads == 0) {
    result = sin_taylor_serial(x, n);
  } else {
    result = sin_taylor_omp(x, n, threads);
  }
  double t1 = omp_get_wtime();
  (void)result;

  printf("%.17g %ld %d %.9f\n", x, n, threads, t1 - t0);
  return 0;
}
