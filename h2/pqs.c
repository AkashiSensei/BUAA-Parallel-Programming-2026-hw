#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
  size_t lo; // inclusive
  size_t hi; // exclusive
} Task;

typedef struct {
  double *arr;
  size_t n;

  pthread_mutex_t mu;
  pthread_cond_t cv;

  Task *stack;
  size_t stack_sz;
  size_t stack_cap;

  size_t tasks_total; // queued + in-progress
  int stop;

  size_t seq_threshold;
} Context;

static inline void swap_double(double *a, double *b) {
  double t = *a;
  *a = *b;
  *b = t;
}

static inline double median3(double a, double b, double c) {
  if (a < b) {
    if (b < c) return b;
    return (a < c) ? c : a;
  } else {
    if (a < c) return a;
    return (b < c) ? c : b;
  }
}

// Hoare-style partition on [lo, hi), returns split index mid.
static size_t partition_range(double *arr, size_t lo, size_t hi) {
  size_t mid = lo + (hi - lo) / 2;
  double pivot = median3(arr[lo], arr[mid], arr[hi - 1]);

  size_t i = lo;
  size_t j = hi - 1;
  while (1) {
    while (arr[i] < pivot) i++;
    while (arr[j] > pivot) j--;
    if (i >= j) return i;
    swap_double(&arr[i], &arr[j]);
    i++;
    if (j == 0) return i;
    j--;
  }
}

static int cmp_double_asc(const void *pa, const void *pb) {
  double a = *(const double *)pa;
  double b = *(const double *)pb;
  return (a > b) - (a < b);
}

static void push_task_locked(Context *ctx, Task t) {
  if (ctx->stack_sz == ctx->stack_cap) {
    size_t new_cap = ctx->stack_cap ? (ctx->stack_cap * 2) : 1024;
    Task *p = (Task *)realloc(ctx->stack, new_cap * sizeof(Task));
    if (!p) {
      fprintf(stderr, "realloc task stack failed\n");
      ctx->stop = 1;
      pthread_cond_broadcast(&ctx->cv);
      return;
    }
    ctx->stack = p;
    ctx->stack_cap = new_cap;
  }
  ctx->stack[ctx->stack_sz++] = t;
  ctx->tasks_total++;
  pthread_cond_signal(&ctx->cv);
}

static bool pop_task_locked(Context *ctx, Task *out) {
  if (ctx->stack_sz == 0) return false;
  *out = ctx->stack[--ctx->stack_sz];
  return true;
}

static void sort_sequential(double *arr, size_t lo, size_t hi) {
  size_t len = hi - lo;
  if (len <= 1) return;
  qsort(arr + lo, len, sizeof(double), cmp_double_asc);
}

static void process_task(Context *ctx, Task t) {
  // Tail-recursive style: always continue on smaller partition, push larger.
  while ((t.hi - t.lo) > ctx->seq_threshold) {
    size_t m = partition_range(ctx->arr, t.lo, t.hi);
    Task left = (Task){.lo = t.lo, .hi = m};
    Task right = (Task){.lo = m, .hi = t.hi};

    // Choose a deterministic split even with many equal keys.
    if (left.hi - left.lo == 0 || right.hi - right.lo == 0) break;

    Task small = left, large = right;
    if ((left.hi - left.lo) > (right.hi - right.lo)) {
      small = right;
      large = left;
    }

    pthread_mutex_lock(&ctx->mu);
    if (!ctx->stop) push_task_locked(ctx, large);
    pthread_mutex_unlock(&ctx->mu);

    t = small;
  }
  sort_sequential(ctx->arr, t.lo, t.hi);
}

static void *worker_main(void *arg) {
  Context *ctx = (Context *)arg;
  while (1) {
    Task t;
    pthread_mutex_lock(&ctx->mu);
    while (!ctx->stop && ctx->stack_sz == 0 && ctx->tasks_total > 0) {
      pthread_cond_wait(&ctx->cv, &ctx->mu);
    }

    if (ctx->stop) {
      pthread_mutex_unlock(&ctx->mu);
      return NULL;
    }

    if (ctx->tasks_total == 0) {
      pthread_mutex_unlock(&ctx->mu);
      return NULL;
    }

    bool ok = pop_task_locked(ctx, &t);
    pthread_mutex_unlock(&ctx->mu);
    if (!ok) continue;

    process_task(ctx, t);

    pthread_mutex_lock(&ctx->mu);
    if (ctx->tasks_total > 0) ctx->tasks_total--;
    if (ctx->tasks_total == 0) pthread_cond_broadcast(&ctx->cv);
    pthread_mutex_unlock(&ctx->mu);
  }
}

static uint64_t xorshift64star(uint64_t *state) {
  uint64_t x = *state;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  *state = x;
  return x * 2685821657736338717ULL;
}

static double rand_double01(uint64_t *state) {
  // 53-bit precision in [0,1)
  uint64_t r = xorshift64star(state);
  return (double)(r >> 11) * (1.0 / 9007199254740992.0);
}

static double now_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void usage(const char *prog) {
  fprintf(stderr,
          "Usage: %s -n <N> -t <T> [--seed <S>] [--check] [--threshold <K>]\n"
          "  -n, --n           array size (recommended >= 20000000)\n"
          "  -t, --threads     number of worker threads (>=1)\n"
          "  --seed            RNG seed (uint64)\n"
          "  --check           verify sorted result\n"
          "  --threshold       sequential sort threshold (default 16384)\n",
          prog);
}

static bool is_sorted(const double *a, size_t n) {
  for (size_t i = 1; i < n; i++) {
    if (a[i - 1] > a[i]) return false;
  }
  return true;
}

static int parse_u64(const char *s, uint64_t *out) {
  errno = 0;
  char *end = NULL;
  unsigned long long v = strtoull(s, &end, 10);
  if (errno != 0 || end == s || *end != '\0') return -1;
  *out = (uint64_t)v;
  return 0;
}

int main(int argc, char **argv) {
  size_t n = 0;
  size_t threads = 0;
  uint64_t seed = 1;
  bool check = false;
  size_t threshold = 16384;

  for (int i = 1; i < argc; i++) {
    const char *arg = argv[i];
    if (!strcmp(arg, "-n") || !strcmp(arg, "--n")) {
      if (i + 1 >= argc) return usage(argv[0]), 2;
      uint64_t v;
      if (parse_u64(argv[++i], &v) != 0) return usage(argv[0]), 2;
      n = (size_t)v;
    } else if (!strcmp(arg, "-t") || !strcmp(arg, "--threads")) {
      if (i + 1 >= argc) return usage(argv[0]), 2;
      uint64_t v;
      if (parse_u64(argv[++i], &v) != 0) return usage(argv[0]), 2;
      threads = (size_t)v;
    } else if (!strcmp(arg, "--seed")) {
      if (i + 1 >= argc) return usage(argv[0]), 2;
      uint64_t v;
      if (parse_u64(argv[++i], &v) != 0) return usage(argv[0]), 2;
      seed = v ? v : 1;
    } else if (!strcmp(arg, "--check")) {
      check = true;
    } else if (!strcmp(arg, "--threshold")) {
      if (i + 1 >= argc) return usage(argv[0]), 2;
      uint64_t v;
      if (parse_u64(argv[++i], &v) != 0) return usage(argv[0]), 2;
      threshold = (size_t)v;
      if (threshold < 2) threshold = 2;
    } else if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
      usage(argv[0]);
      return 0;
    } else {
      usage(argv[0]);
      return 2;
    }
  }

  if (n == 0 || threads == 0) {
    usage(argv[0]);
    return 2;
  }

  double *arr = (double *)malloc(n * sizeof(double));
  if (!arr) {
    fprintf(stderr, "malloc failed for n=%zu (%.2f MiB)\n", n,
            (double)(n * sizeof(double)) / (1024.0 * 1024.0));
    return 1;
  }

  uint64_t st = seed;
  for (size_t i = 0; i < n; i++) arr[i] = rand_double01(&st);

  Context ctx;
  memset(&ctx, 0, sizeof(ctx));
  ctx.arr = arr;
  ctx.n = n;
  ctx.seq_threshold = threshold;
  pthread_mutex_init(&ctx.mu, NULL);
  pthread_cond_init(&ctx.cv, NULL);

  pthread_t *tids = (pthread_t *)calloc(threads, sizeof(pthread_t));
  if (!tids) {
    fprintf(stderr, "calloc tids failed\n");
    free(arr);
    return 1;
  }

  pthread_mutex_lock(&ctx.mu);
  push_task_locked(&ctx, (Task){.lo = 0, .hi = n});
  pthread_mutex_unlock(&ctx.mu);

  double t0 = now_sec();
  for (size_t i = 0; i < threads; i++) {
    pthread_create(&tids[i], NULL, worker_main, &ctx);
  }
  for (size_t i = 0; i < threads; i++) {
    pthread_join(tids[i], NULL);
  }
  double t1 = now_sec();

  if (ctx.stop) {
    fprintf(stderr, "aborted due to internal error\n");
    free(tids);
    free(ctx.stack);
    free(arr);
    return 1;
  }

  if (check) {
    if (!is_sorted(arr, n)) {
      fprintf(stderr, "check failed: array is NOT sorted\n");
      free(tids);
      free(ctx.stack);
      free(arr);
      return 1;
    }
  }

  printf("n=%zu threads=%zu threshold=%zu time_sec=%.6f seed=%" PRIu64 "\n", n,
         threads, threshold, (t1 - t0), seed);

  free(tids);
  free(ctx.stack);
  free(arr);
  pthread_cond_destroy(&ctx.cv);
  pthread_mutex_destroy(&ctx.mu);
  return 0;
}

