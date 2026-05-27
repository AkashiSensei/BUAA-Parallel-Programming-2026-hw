/*
 * MPI 块矩阵乘法：二维子块划分，C_{i,j} = sum_k A_{i,k} * B_{k,j}
 * 用法: mpirun -np <p> ./matmul_mpi <N>
 *
 * 默认输出一行: N p Pr Pc time_sec
 *
 * 可选环境变量（用于单独的分析实验）:
 *   MATMUL_PROFILE=1  额外输出 PROFILE / RANK 阶段汇总
 *   MATMUL_EVENTS=1   再输出 EVENT 时间线（隐含开启记录）
 */

#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TAG_A 100
#define TAG_B 200
#define TAG_GATHER 300
#define PROF_MAX_EVENTS 8192

typedef struct {
  int seq;
  double t0;
  double t1;
  char phase[16];
  char kind[16];
  int k;
  int peer;
  int tag;
} ProfEvent;

static struct {
  int summary;
  int events;
  double origin;
  int next_seq;
  int count;
  ProfEvent ev[PROF_MAX_EVENTS];
} g_prof = {0, 0, 0.0, 0, 0, {{0}}};

static int env_enabled(const char *name) {
  const char *v = getenv(name);
  return v != NULL && v[0] != '\0' && v[0] != '0';
}

static void prof_init(double origin) {
  g_prof.events = env_enabled("MATMUL_EVENTS");
  g_prof.summary = g_prof.events || env_enabled("MATMUL_PROFILE");
  g_prof.origin = origin;
  g_prof.next_seq = 0;
  g_prof.count = 0;
}

static void prof_record(const char *phase, const char *kind, int k, int peer, int tag,
                        double t_start, double t_end) {
  if (!g_prof.summary || g_prof.count >= PROF_MAX_EVENTS) {
    return;
  }
  ProfEvent *e = &g_prof.ev[g_prof.count++];
  e->seq = g_prof.next_seq++;
  e->t0 = t_start - g_prof.origin;
  e->t1 = t_end - g_prof.origin;
  snprintf(e->phase, sizeof(e->phase), "%s", phase);
  snprintf(e->kind, sizeof(e->kind), "%s", kind);
  e->k = k;
  e->peer = peer;
  e->tag = tag;
}

static void prof_emit_all(int rank) {
  if (!g_prof.events) {
    return;
  }
  for (int i = 0; i < g_prof.count; i++) {
    const ProfEvent *e = &g_prof.ev[i];
    printf("EVENT %d %d %.9f %.9f %s %s %d %d %d\n", rank, e->seq, e->t0, e->t1,
           e->phase, e->kind, e->k, e->peer, e->tag);
  }
}

static void block_bounds(int n, int blocks, int idx, int *start, int *end) {
  int base = n / blocks;
  int rem = n % blocks;
  if (idx < rem) {
    *start = idx * (base + 1);
    *end = *start + base + 1;
  } else {
    *start = rem * (base + 1) + (idx - rem) * base;
    *end = *start + base;
  }
}

static void factor_grid(int p, int *pr, int *pc) {
  int r = (int)ceil(sqrt((double)p));
  while (r > 0) {
    if (p % r == 0) {
      int a = r;
      int b = p / r;
      if (a >= b) {
        *pr = a;
        *pc = b;
      } else {
        *pr = b;
        *pc = a;
      }
      return;
    }
    r--;
  }
  *pr = 1;
  *pc = p;
}

static void init_matrices(double *A, double *B, int n) {
  const size_t nn = (size_t)n * (size_t)n;
  for (size_t i = 0; i < nn; i++) {
    A[i] = (double)((i * 17 + 3) % 1000) * 0.001;
    B[i] = (double)((i * 31 + 7) % 1000) * 0.001;
  }
}

static void extract_block(const double *global, int n, int rs, int re, int cs, int ce,
                          double *buf) {
  const int rows = re - rs;
  const int cols = ce - cs;
  for (int ii = 0; ii < rows; ii++) {
    for (int jj = 0; jj < cols; jj++) {
      buf[(size_t)ii * (size_t)cols + (size_t)jj] =
          global[(size_t)(rs + ii) * (size_t)n + (size_t)(cs + jj)];
    }
  }
}

static void place_block(double *global, int n, int rs, int re, int cs, int ce,
                        const double *buf) {
  const int rows = re - rs;
  const int cols = ce - cs;
  for (int ii = 0; ii < rows; ii++) {
    for (int jj = 0; jj < cols; jj++) {
      global[(size_t)(rs + ii) * (size_t)n + (size_t)(cs + jj)] =
          buf[(size_t)ii * (size_t)cols + (size_t)jj];
    }
  }
}

static void gather_C_blocks(int rank, int p, int n, int Pr, int Pc, const double *C_local,
                            int c_rs, int c_re, int c_cs, int c_ce, double *C_full) {
  const int local_cnt = (c_re - c_rs) * (c_ce - c_cs);
  if (rank == 0) {
    {
      const double t0 = MPI_Wtime();
      place_block(C_full, n, c_rs, c_re, c_cs, c_ce, C_local);
      const double t1 = MPI_Wtime();
      prof_record("gather", "compute", -1, -1, TAG_GATHER, t0, t1);
    }
    for (int src = 1; src < p; src++) {
      const int i = src / Pc;
      const int j = src % Pc;
      int rs = 0, re = 0, cs = 0, ce = 0;
      block_bounds(n, Pr, i, &rs, &re);
      block_bounds(n, Pc, j, &cs, &ce);
      const int cnt = (re - rs) * (ce - cs);
      double *buf = NULL;
      {
        const double t0 = MPI_Wtime();
        buf = (double *)malloc((size_t)cnt * sizeof(double));
        const double t1 = MPI_Wtime();
        prof_record("gather", "alloc", -1, -1, -1, t0, t1);
      }
      if (!buf) {
        fprintf(stderr, "malloc failed in gather_C_blocks\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      {
        const double t0 = MPI_Wtime();
        MPI_Recv(buf, cnt, MPI_DOUBLE, src, TAG_GATHER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        const double t1 = MPI_Wtime();
        prof_record("gather", "recv", -1, src, TAG_GATHER, t0, t1);
      }
      {
        const double t0 = MPI_Wtime();
        place_block(C_full, n, rs, re, cs, ce, buf);
        const double t1 = MPI_Wtime();
        prof_record("gather", "compute", -1, src, TAG_GATHER, t0, t1);
      }
      free(buf);
    }
  } else {
    const double t0 = MPI_Wtime();
    MPI_Send(C_local, local_cnt, MPI_DOUBLE, 0, TAG_GATHER, MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();
    prof_record("gather", "send", -1, 0, TAG_GATHER, t0, t1);
  }
}

static void gemm_accum(const double *A, const double *B, double *C, int rows_a, int cols_a,
                       int cols_b) {
  for (int ii = 0; ii < rows_a; ii++) {
    for (int jj = 0; jj < cols_b; jj++) {
      double sum = C[(size_t)ii * (size_t)cols_b + (size_t)jj];
      for (int kk = 0; kk < cols_a; kk++) {
        sum += A[(size_t)ii * (size_t)cols_a + (size_t)kk] *
               B[(size_t)kk * (size_t)cols_b + (size_t)jj];
      }
      C[(size_t)ii * (size_t)cols_b + (size_t)jj] = sum;
    }
  }
}

static void send_block_profiled(const char *phase, int k_idx, int dest, int tag,
                                const double *global, int n, int rs, int re, int cs,
                                int ce, MPI_Comm comm) {
  const int rows = re - rs;
  const int cols = ce - cs;
  const size_t cnt = (size_t)rows * (size_t)cols;
  double *buf = NULL;
  {
    const double t0 = MPI_Wtime();
    buf = (double *)malloc(cnt * sizeof(double));
    const double t1 = MPI_Wtime();
    prof_record(phase, "alloc", k_idx, dest, tag, t0, t1);
  }
  if (!buf) {
    fprintf(stderr, "malloc failed in send_block\n");
    MPI_Abort(comm, 1);
  }
  {
    const double t0 = MPI_Wtime();
    extract_block(global, n, rs, re, cs, ce, buf);
    const double t1 = MPI_Wtime();
    prof_record(phase, "pack", k_idx, dest, tag, t0, t1);
  }
  {
    const double t0 = MPI_Wtime();
    MPI_Send(buf, (int)cnt, MPI_DOUBLE, dest, tag, comm);
    const double t1 = MPI_Wtime();
    prof_record(phase, "send", k_idx, dest, tag, t0, t1);
  }
  free(buf);
}

static void recv_block_profiled(const char *phase, int k_idx, int src, int tag, double *buf,
                                int cnt, MPI_Comm comm) {
  const double t0 = MPI_Wtime();
  MPI_Recv(buf, cnt, MPI_DOUBLE, src, tag, comm, MPI_STATUS_IGNORE);
  const double t1 = MPI_Wtime();
  prof_record(phase, "recv", k_idx, src, tag, t0, t1);
}

static void sum_phase_times(double *scatter_a, double *scatter_b, double *k_comm,
                            double *k_gemm, double *gather, double *other) {
  *scatter_a = *scatter_b = *k_comm = *k_gemm = *gather = *other = 0.0;
  for (int i = 0; i < g_prof.count; i++) {
    const ProfEvent *e = &g_prof.ev[i];
    const double dt = e->t1 - e->t0;
    if (strcmp(e->phase, "scatter_a") == 0) {
      *scatter_a += dt;
    } else if (strcmp(e->phase, "scatter_b") == 0) {
      *scatter_b += dt;
    } else if (strcmp(e->phase, "k_loop") == 0) {
      if (strcmp(e->kind, "compute") == 0) {
        *k_gemm += dt;
      } else if (strcmp(e->kind, "send") == 0 || strcmp(e->kind, "recv") == 0) {
        *k_comm += dt;
      } else if (strcmp(e->kind, "alloc") == 0) {
        *other += dt;
      }
    } else if (strcmp(e->phase, "gather") == 0) {
      if (strcmp(e->kind, "alloc") == 0) {
        *other += dt;
      } else {
        *gather += dt;
      }
    } else {
      *other += dt;
    }
  }
  /* k_loop 中 pack 未出现；scatter 的 pack/send/recv 全计入 scatter_* */
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank = 0, p = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  if (argc != 2) {
    if (rank == 0) {
      fprintf(stderr, "用法: mpirun -np <p> %s <N>\n", argv[0]);
    }
    MPI_Finalize();
    return 1;
  }

  char *end = NULL;
  long n_l = strtol(argv[1], &end, 10);
  if (end == argv[1] || *end != '\0' || n_l < 1 || n_l > 200000) {
    if (rank == 0) {
      fprintf(stderr, "错误: N 应为 1..200000 的整数\n");
    }
    MPI_Finalize();
    return 1;
  }
  const int n = (int)n_l;

  int Pr = 1, Pc = 1;
  factor_grid(p, &Pr, &Pc);

  int proc_row = rank / Pc;
  int proc_col = rank % Pc;

  int c_rs = 0, c_re = 0, c_cs = 0, c_ce = 0;
  block_bounds(n, Pr, proc_row, &c_rs, &c_re);
  block_bounds(n, Pc, proc_col, &c_cs, &c_ce);
  const int c_rows = c_re - c_rs;
  const int c_cols = c_ce - c_cs;

  double *C_local = (double *)calloc((size_t)c_rows * (size_t)c_cols, sizeof(double));
  double *A_owned = NULL;
  double *B_owned = NULL;
  int a_owned_cnt = 0, b_owned_cnt = 0;

  {
    int a_rs = 0, a_re = 0, a_cs = 0, a_ce = 0;
    block_bounds(n, Pr, proc_row, &a_rs, &a_re);
    block_bounds(n, Pc, proc_col, &a_cs, &a_ce);
    a_owned_cnt = (a_re - a_rs) * (a_ce - a_cs);
    A_owned = (double *)malloc((size_t)a_owned_cnt * sizeof(double));
  }
  if (proc_row < Pc) {
    int b_rs = 0, b_re = 0, b_cs = 0, b_ce = 0;
    block_bounds(n, Pc, proc_row, &b_rs, &b_re);
    block_bounds(n, Pc, proc_col, &b_cs, &b_ce);
    b_owned_cnt = (b_re - b_rs) * (b_ce - b_cs);
    B_owned = (double *)malloc((size_t)b_owned_cnt * sizeof(double));
  }

  double *A_full = NULL;
  double *B_full = NULL;
  if (rank == 0) {
    const size_t nn = (size_t)n * (size_t)n;
    A_full = (double *)malloc(nn * sizeof(double));
    B_full = (double *)malloc(nn * sizeof(double));
    if (!A_full || !B_full) {
      fprintf(stderr, "错误: 根进程矩阵分配失败\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    init_matrices(A_full, B_full, n);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double global_origin = 0.0;
  if (rank == 0) {
    global_origin = MPI_Wtime();
  }
  MPI_Bcast(&global_origin, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  const double t0 = global_origin;
  prof_init(global_origin);

  for (int i = 0; i < Pr; i++) {
    for (int k = 0; k < Pc; k++) {
      int dest = i * Pc + k;
      int rs = 0, re = 0, cs = 0, ce = 0;
      block_bounds(n, Pr, i, &rs, &re);
      block_bounds(n, Pc, k, &cs, &ce);
      const int cnt = (re - rs) * (ce - cs);
      if (rank == 0) {
        if (dest == 0) {
          const double ts = MPI_Wtime();
          extract_block(A_full, n, rs, re, cs, ce, A_owned);
          const double te = MPI_Wtime();
          prof_record("scatter_a", "compute", k, 0, TAG_A, ts, te);
        } else {
          send_block_profiled("scatter_a", k, dest, TAG_A, A_full, n, rs, re, cs, ce,
                              MPI_COMM_WORLD);
        }
      } else if (rank == dest) {
        recv_block_profiled("scatter_a", k, 0, TAG_A, A_owned, cnt, MPI_COMM_WORLD);
      }
    }
  }

  for (int k = 0; k < Pc; k++) {
    for (int j = 0; j < Pc; j++) {
      int dest = k * Pc + j;
      int rs = 0, re = 0, cs = 0, ce = 0;
      block_bounds(n, Pc, k, &rs, &re);
      block_bounds(n, Pc, j, &cs, &ce);
      const int cnt = (re - rs) * (ce - cs);
      if (rank == 0) {
        if (dest == 0) {
          const double ts = MPI_Wtime();
          extract_block(B_full, n, rs, re, cs, ce, B_owned);
          const double te = MPI_Wtime();
          prof_record("scatter_b", "compute", k, 0, TAG_B, ts, te);
        } else {
          send_block_profiled("scatter_b", k, dest, TAG_B, B_full, n, rs, re, cs, ce,
                              MPI_COMM_WORLD);
        }
      } else if (rank == dest) {
        recv_block_profiled("scatter_b", k, 0, TAG_B, B_owned, cnt, MPI_COMM_WORLD);
      }
    }
  }

  if (rank == 0) {
    free(A_full);
    free(B_full);
  }

  double *A_buf = NULL;
  double *B_buf = NULL;

  for (int k = 0; k < Pc; k++) {
    int a_rs = 0, a_re = 0, a_cs = 0, a_ce = 0;
    block_bounds(n, Pr, proc_row, &a_rs, &a_re);
    block_bounds(n, Pc, k, &a_cs, &a_ce);
    const int a_rows = a_re - a_rs;
    const int a_cols = a_ce - a_cs;
    const int a_cnt = a_rows * a_cols;

    const double *A_sub = NULL;
    const int tag_a = TAG_A + k;
    if (proc_col == k) {
      A_sub = A_owned;
      for (int j = 0; j < Pc; j++) {
        if (j == proc_col) {
          continue;
        }
        const int dest = proc_row * Pc + j;
        const double ts = MPI_Wtime();
        MPI_Send(A_owned, a_cnt, MPI_DOUBLE, dest, tag_a, MPI_COMM_WORLD);
        const double te = MPI_Wtime();
        prof_record("k_loop", "send", k, dest, tag_a, ts, te);
      }
    } else {
      free(A_buf);
      {
        const double ts = MPI_Wtime();
        A_buf = (double *)malloc((size_t)a_cnt * sizeof(double));
        const double te = MPI_Wtime();
        prof_record("k_loop", "alloc", k, -1, tag_a, ts, te);
      }
      recv_block_profiled("k_loop", k, proc_row * Pc + k, tag_a, A_buf, a_cnt,
                          MPI_COMM_WORLD);
      A_sub = A_buf;
    }

    int b_rs = 0, b_re = 0, b_cs = 0, b_ce = 0;
    block_bounds(n, Pc, k, &b_rs, &b_re);
    block_bounds(n, Pc, proc_col, &b_cs, &b_ce);
    const int b_rows = b_re - b_rs;
    const int b_cols = b_ce - b_cs;
    const int b_cnt = b_rows * b_cols;

    const double *B_sub = NULL;
    const int tag_b = TAG_B + k;
    if (proc_row == k && B_owned != NULL) {
      B_sub = B_owned;
      for (int i = 0; i < Pr; i++) {
        if (i == proc_row) {
          continue;
        }
        const int dest = i * Pc + proc_col;
        const double ts = MPI_Wtime();
        MPI_Send(B_owned, b_cnt, MPI_DOUBLE, dest, tag_b, MPI_COMM_WORLD);
        const double te = MPI_Wtime();
        prof_record("k_loop", "send", k, dest, tag_b, ts, te);
      }
    } else {
      free(B_buf);
      {
        const double ts = MPI_Wtime();
        B_buf = (double *)malloc((size_t)b_cnt * sizeof(double));
        const double te = MPI_Wtime();
        prof_record("k_loop", "alloc", k, -1, tag_b, ts, te);
      }
      recv_block_profiled("k_loop", k, k * Pc + proc_col, tag_b, B_buf, b_cnt,
                          MPI_COMM_WORLD);
      B_sub = B_buf;
    }

    if (a_cols != b_rows) {
      fprintf(stderr, "rank %d: 子块内维不匹配 a_cols=%d b_rows=%d\n", rank, a_cols,
              b_rows);
      MPI_Abort(MPI_COMM_WORLD, 2);
    }

    {
      const double ts = MPI_Wtime();
      gemm_accum(A_sub, B_sub, C_local, a_rows, a_cols, b_cols);
      const double te = MPI_Wtime();
      prof_record("k_loop", "compute", k, -1, -1, ts, te);
    }
  }

  free(A_buf);
  free(B_buf);
  free(A_owned);
  free(B_owned);

  double *C_full = NULL;
  if (rank == 0) {
    const size_t nn = (size_t)n * (size_t)n;
    C_full = (double *)calloc(nn, sizeof(double));
    if (!C_full) {
      fprintf(stderr, "错误: 汇聚矩阵 C 分配失败\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }
  gather_C_blocks(rank, p, n, Pr, Pc, C_local, c_rs, c_re, c_cs, c_ce, C_full);

  const double t1 = MPI_Wtime();
  const double elapsed = t1 - t0;

  free(C_local);
  if (rank == 0) {
    free(C_full);
  }

  if (rank == 0) {
    printf("%d %d %d %d %.9f\n", n, p, Pr, Pc, elapsed);
  }

  if (g_prof.summary) {
    double t_scatter_a = 0, t_scatter_b = 0, t_k_comm = 0, t_k_gemm = 0, t_gather = 0,
           t_other = 0;
    sum_phase_times(&t_scatter_a, &t_scatter_b, &t_k_comm, &t_k_gemm, &t_gather, &t_other);
    t_other += elapsed - (t_scatter_a + t_scatter_b + t_k_comm + t_k_gemm + t_gather);
    if (t_other < 0.0) {
      t_other = 0.0;
    }

    int local_nev = g_prof.count;
    int total_nev = 0;
    MPI_Reduce(&local_nev, &total_nev, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    const int nprof = 6;
    double local_prof[6];
    local_prof[0] = t_scatter_a;
    local_prof[1] = t_scatter_b;
    local_prof[2] = t_k_comm;
    local_prof[3] = t_k_gemm;
    local_prof[4] = t_gather;
    local_prof[5] = t_other;
    double max_prof[nprof];
    MPI_Reduce(local_prof, max_prof, nprof, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double *all_prof = NULL;
    if (rank == 0) {
      all_prof = (double *)malloc((size_t)p * (size_t)nprof * sizeof(double));
      if (!all_prof) {
        fprintf(stderr, "malloc failed for profile gather\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
    }
    MPI_Gather(local_prof, nprof, MPI_DOUBLE, all_prof, nprof, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
      printf("PROFILE %.9f %.9f %.9f %.9f %.9f %.9f\n", max_prof[0], max_prof[1],
             max_prof[2], max_prof[3], max_prof[4], max_prof[5]);
      for (int r = 0; r < p; r++) {
        const double *rp = all_prof + (size_t)r * (size_t)nprof;
        printf("RANK %d %.9f %.9f %.9f %.9f %.9f\n", r, rp[0], rp[1], rp[2], rp[3],
               rp[4]);
      }
      free(all_prof);
    }

    if (g_prof.events) {
      MPI_Barrier(MPI_COMM_WORLD);
      if (rank == 0) {
        printf("EVENTS_BEGIN %d\n", total_nev);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      prof_emit_all(rank);
      MPI_Barrier(MPI_COMM_WORLD);
      if (rank == 0) {
        printf("EVENTS_END\n");
      }
    }
  }

  MPI_Finalize();
  return 0;
}
